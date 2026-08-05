"""Module behaviour: compressor, cache, budgeter, router, fidelity gate."""

from __future__ import annotations

import re

import pytest

from parsimony.core.config import full_stack
from parsimony.core.proposals import ContextPatch, NoOp, ShortCircuit, TransformKind
from parsimony.core.types import ResponseClass, RouteTier
from parsimony.infra.nlp import RegexInvariantExtractor, RegexPiiDetector, split_sentences
from parsimony.modules.m1_compressor import normalise_lossless
from parsimony.modules.m2_cache import SemanticCache, canonicalise, chain_hash
from parsimony.modules.m5_budgeter import TrigramNoveltyStopper, classify
from parsimony.modules.m6_router import ArithmeticError_, format_number, safe_arithmetic, solve
from parsimony.modules.m8_fidelity import FidelityGate


# ---------------------------------------------------------------- M1 --------
class TestLosslessNormalisation:
    def test_removes_politeness_boilerplate(self):
        out = normalise_lossless("Could you please explain recursion? Thanks in advance!")
        assert "please" not in out.lower()
        assert "thanks" not in out.lower()
        assert "recursion" in out

    def test_drops_sentences_that_become_contentless(self):
        """'Thanks in advance!' must vanish entirely, not leave a stray '!'."""
        out = normalise_lossless("Explain recursion. Thanks in advance!")
        assert out == "Explain recursion."

    def test_recapitalises_after_removing_the_opener(self):
        out = normalise_lossless("Could you please explain recursion?")
        assert out[0].isupper()

    def test_leaves_fenced_code_untouched(self):
        code = "Explain this:\n```python\nx = 1   # please keep    spacing\n```"
        out = normalise_lossless(code)
        assert "x = 1   # please keep    spacing" in out

    def test_preserves_numbers_and_entities(self):
        text = "Please tell me: the budget is 50,000 dollars for Project Apollo. Thanks!"
        out = normalise_lossless(text)
        assert "50,000" in out and "Project Apollo" in out

    def test_is_idempotent(self):
        once = normalise_lossless("Hello! Could you please explain recursion? Thanks!")
        assert normalise_lossless(once) == once

    @pytest.mark.parametrize(
        "text",
        [
            "If a train travels at 80 km/h for 3.5 hours, how far does it go?",
            "What is 0.1 + 0.2?",
            "How do I merge two dictionaries in Python 3.9?",
            "Version 2.1 fixed the memory leak and 2.3 improved startup.",
            "Convert 62.137 miles to kilometres",
        ],
    )
    def test_never_splits_inside_a_decimal(self, text):
        """Regression: the sentence splitter cut on the period inside "3.5",
        rebuilding it as "3. 5" and destroying the number. The gate caught it on
        13 of 32 tier-1 proposals — but a LOSSLESS tier must not need rescuing.
        """
        out = normalise_lossless(text)
        for number in re.findall(r"\d+\.\d+", text):
            assert number in out, f"{number!r} was corrupted: {out!r}"

    @pytest.mark.parametrize(
        "text,keep",
        [
            ("Tell me about 3.14 and e.g. other constants.", "e.g."),
            ("Please ask Dr. Smith. Then tell me the result.", "Dr. Smith"),
            ("Compare cats vs. dogs please.", "vs."),
        ],
    )
    def test_never_splits_inside_an_abbreviation(self, text, keep):
        assert keep in normalise_lossless(text)

    def test_does_not_invent_capitals_mid_sentence(self):
        """Capitalising unconditionally turned "for 3.5 hours" into "3.5 Hours",
        changing text the user wrote."""
        out = normalise_lossless("If a train travels for 3.5 hours, how far?")
        assert "hours" in out and "Hours" not in out

    def test_still_recapitalises_when_it_strips_the_opener(self):
        assert normalise_lossless("Could you please explain recursion?").startswith("Explain")

    def test_leaves_a_clean_sentence_completely_alone(self):
        """No proposal at all is better than a no-op edit: it keeps the trace
        honest about how often tier 1 actually has work to do."""
        text = "What is the capital of France?"
        assert normalise_lossless(text) == text

    def test_never_increases_length(self):
        for text in ["Explain recursion.", "Hi, please help.", "```code```", "A. B. C."]:
            assert len(normalise_lossless(text)) <= len(text)


# ---------------------------------------------------------------- M2 --------
class TestCache:
    def test_canonicalisation_ignores_case_space_and_final_punctuation(self):
        assert canonicalise("  What Is  Recursion?  ") == canonicalise("what is recursion")

    def test_key_depends_on_the_model(self):
        """Report 4.6 re-runs the winner on three models. Without model_id in the
        key a Llama answer would be served during the Qwen run."""
        a = SemanticCache.make_key("q", "root", "llama-3.2-1b")
        b = SemanticCache.make_key("q", "root", "qwen2.5-1.5b")
        assert a != b

    def test_key_depends_on_the_context_chain(self):
        a = SemanticCache.make_key("and the second one?", "root", "m")
        b = SemanticCache.make_key("and the second one?", "abc123", "m")
        assert a != b

    def test_chain_depth_zero_collapses_to_root(self):
        from parsimony.core.types import Turn

        history = (Turn("1", "user", "hello"),)
        assert chain_hash(history, 0) == "root"
        assert chain_hash(history, 2) != "root"

    def test_store_then_lookup_returns_the_entry(self):
        cache = SemanticCache()
        key = SemanticCache.make_key("q", "root", "m")
        cache.store(key, "q", "an answer")
        assert cache.lookup(key).response == "an answer"

    def test_volatile_entries_expire(self):
        cache = SemanticCache(ttl_seconds=10)
        key = SemanticCache.make_key("what is the current price?", "root", "m")
        cache.store(key, "what is the current price?", "100", now=0.0)
        assert cache.lookup(key, now=5.0) is not None
        assert cache.lookup(key, now=100.0) is None

    def test_non_volatile_entries_do_not_expire(self):
        cache = SemanticCache(ttl_seconds=10)
        key = SemanticCache.make_key("what is 2+2?", "root", "m")
        cache.store(key, "what is 2+2?", "4", now=0.0)
        assert cache.lookup(key, now=1e9) is not None


# ---------------------------------------------------------------- M5 --------
class TestClassifier:
    @pytest.mark.parametrize(
        "query,expected",
        [
            ("Write a function that returns primes", ResponseClass.CODE),
            ("Summarise this article", ResponseClass.SUMMARISATION),
            ("Why does water boil at altitude?", ResponseClass.REASONING),
            ("Convert 100 km to miles", ResponseClass.ARITHMETIC),
            ("What is the capital of France", ResponseClass.FACTUAL),
        ],
    )
    def test_assigns_the_expected_class(self, query, expected):
        assert classify(query, has_history=False) is expected

    def test_detects_a_follow_up_only_when_history_exists(self):
        assert classify("And at 3000 metres?", has_history=True) is ResponseClass.FOLLOW_UP
        assert classify("And at 3000 metres?", has_history=False) is not ResponseClass.FOLLOW_UP

    def test_detects_a_follow_up_with_no_pronoun(self):
        """Real follow-ups often carry no anaphor at all, just a conjunction."""
        assert classify("And in the worst case?", has_history=True) is ResponseClass.FOLLOW_UP

    def test_detects_an_anaphoric_follow_up(self):
        assert classify("Give me an example of that.", has_history=True) is ResponseClass.FOLLOW_UP

    def test_reasoning_outranks_follow_up(self):
        """Budget follows how long the answer must be, not discourse position."""
        assert classify("Why does it change?", has_history=True) is ResponseClass.REASONING


class TestEarlyStopper:
    def test_stops_when_a_sentence_is_restated(self):
        stopper = TrigramNoveltyStopper()
        sentence = "the answer depends on the specific context you are working in."
        stopped = False
        for _ in range(2):
            for word in sentence.split():
                if stopper.observe(" " + word):
                    stopped = True
                    break
            if stopped:
                break
        assert stopped
        assert stopper.reason == "restated a sentence"

    def test_does_not_stop_on_novel_prose(self):
        stopper = TrigramNoveltyStopper()
        text = ("photosynthesis converts light into chemical energy chlorophyll absorbs "
                "photons water splits carbon dioxide becomes glucose oxygen leaves as waste")
        assert not any(stopper.observe(" " + w) for w in text.split())

    def test_ignores_very_short_repeated_fragments(self):
        """'Yes. Yes.' is under the four-word floor and must not trigger a stop."""
        stopper = TrigramNoveltyStopper()
        assert not any(stopper.observe(w) for w in ["Yes.", " Yes.", " Yes."])


# ---------------------------------------------------------------- M6 --------
class TestSafeArithmetic:
    @pytest.mark.parametrize(
        "expr,expected",
        [("2+2", "4"), ("847*23", "19481"), ("10/4", "2.5"), ("2**10", "1024"), ("7%3", "1")],
    )
    def test_evaluates_correctly(self, expr, expected):
        assert format_number(safe_arithmetic(expr)) == expected

    def test_is_exact_where_floats_are_not(self):
        """Fraction arithmetic: 0.1+0.2 is exactly 0.3, not 0.30000000000000004."""
        assert format_number(safe_arithmetic("0.1+0.2")) == "0.3"

    def test_refuses_a_huge_exponent(self):
        """Guard against burning CPU/memory on a pathological expression."""
        with pytest.raises(ArithmeticError_):
            safe_arithmetic("9**9**9")

    def test_refuses_division_by_zero(self):
        with pytest.raises(ArithmeticError_):
            safe_arithmetic("1/0")

    def test_refuses_names_and_calls(self):
        for expr in ["__import__('os')", "open('x')", "abc"]:
            with pytest.raises((ArithmeticError_, SyntaxError, ValueError)):
                safe_arithmetic(expr)


class TestDeterministicSolver:
    @pytest.mark.parametrize(
        "query,expected,handler",
        [
            ("What is 847 * 23?", "19481", "arithmetic"),
            ("Convert 100 km to miles", None, "unit_conversion"),
            ("What is 15% of 200?", "30", "percent"),
            ("How many days between 2026-01-01 and 2026-01-31?", "30", "date_diff"),
            ("30 days after 2026-01-01", "2026-01-31", "date_add"),
        ],
    )
    def test_handles_supported_shapes(self, query, expected, handler):
        result = solve(query)
        assert result is not None, query
        answer, used = result
        assert used == handler
        if expected is not None:
            assert answer == expected

    def test_converts_units_correctly(self):
        answer, _ = solve("Convert 100 km to miles")
        assert answer.startswith("62.137")

    def test_converts_temperature_correctly(self):
        answer, _ = solve("convert 100 C to F")
        assert answer.startswith("212")

    @pytest.mark.parametrize(
        "query",
        [
            "If a train travels at 80 km/h for 3.5 hours, how far does it go?",
            "What is the capital of France?",
            "Explain recursion",
        ],
    )
    def test_declines_anything_it_cannot_answer_exactly(self, query):
        """Tier 0 must be precise, not merely accurate: a wrong deterministic
        answer is delivered with total confidence and zero model involvement."""
        assert solve(query) is None


# ---------------------------------------------------------------- M8 --------
class TestFidelityGate:
    def test_passes_a_rewrite_that_preserves_everything(self, pipeline):
        ctx = pipeline.build_context("Please explain the 42 rule. Thanks!")
        from dataclasses import replace

        after = replace(ctx, query="Explain the 42 rule.")
        assert pipeline.gate.check(ctx, after, TransformKind.REWRITE, "M1").passed

    def test_rejects_a_rewrite_that_drops_a_number(self, pipeline):
        from dataclasses import replace

        ctx = pipeline.build_context("The budget is 50000 dollars.")
        after = replace(ctx, query="The budget is large.")
        verdict = pipeline.gate.check(ctx, after, TransformKind.REWRITE, "M1")
        assert not verdict.passed
        assert verdict.events[0].invariant_class == "number"

    def test_rejects_a_rewrite_that_drops_a_negation(self, pipeline):
        from dataclasses import replace

        ctx = pipeline.build_context("It is not safe to mix them.")
        after = replace(ctx, query="It is safe to mix them.")
        assert not pipeline.gate.check(ctx, after, TransformKind.REWRITE, "M1").passed

    def test_allows_select_to_drop_whole_turns(self, pipeline):
        """M3 deleting a turn is the entire point of M3 — a uniform gate would
        veto it permanently (ADR-003)."""
        from dataclasses import replace

        from parsimony.core.types import Turn

        history = (
            Turn("a", "user", "The value is 42."),
            Turn("b", "user", "Something else entirely."),
        )
        ctx = pipeline.build_context("carry on", history)
        after = replace(ctx, history=(history[1],))
        assert pipeline.gate.check(ctx, after, TransformKind.SELECT, "M3").passed

    def test_rejects_select_that_mutates_a_retained_turn(self, pipeline):
        from dataclasses import replace

        from parsimony.core.types import Turn

        history = (Turn("a", "user", "The value is 42."),)
        ctx = pipeline.build_context("carry on", history)
        after = replace(ctx, history=(Turn("a", "user", "The value is 43."),))
        assert not pipeline.gate.check(ctx, after, TransformKind.SELECT, "M3").passed

    def test_decide_patches_are_never_text_checked(self, pipeline):
        from dataclasses import replace

        ctx = pipeline.build_context("anything at all")
        after = replace(ctx, output_budget=128)
        assert pipeline.gate.check(ctx, after, TransformKind.DECIDE, "M5").passed

    def test_memoises_extraction(self, pipeline):
        """Re-extracting per check would cost ~70ms of the 120ms budget."""
        gate = FidelityGate()
        for _ in range(10):
            gate.invariants_of("the value is 42")
        assert gate.extractions == 1


# --------------------------------------------------------------- infra ------
class TestInvariantExtraction:
    def test_extracts_numbers_with_units(self):
        inv = RegexInvariantExtractor().extract("It weighs 3.5 kg and costs 50%")
        assert any("3.5" in n for n in inv.numbers)
        assert any("50" in n for n in inv.numbers)

    def test_extracts_negations(self):
        inv = RegexInvariantExtractor().extract("This is not safe and cannot be done")
        assert "not" in inv.negations and "cannot" in inv.negations

    def test_extracts_quoted_spans(self):
        inv = RegexInvariantExtractor().extract('Use the `strip()` method on "input"')
        assert "strip()" in inv.quoted and "input" in inv.quoted

    def test_skips_sentence_initial_capitals(self):
        """'What' opening a question is not an entity."""
        inv = RegexInvariantExtractor().extract("What is the capital of Australia?")
        assert "Australia" in inv.entities
        assert "What" not in inv.entities


class TestSentenceSplitting:
    def test_splits_on_terminators(self):
        assert len(split_sentences("One. Two! Three?")) == 3

    def test_keeps_fenced_code_intact(self):
        out = split_sentences("Look:\n```\na. b. c.\n```")
        assert any(s.startswith("```") for s in out)


class TestPii:
    def test_redacts_an_email(self):
        assert "@" not in RegexPiiDetector().redact("write to a.b@c.com now")

    def test_leaves_ordinary_text_untouched(self):
        text = "the capital of France is Paris"
        assert RegexPiiDetector().redact(text) == text
