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


class TestEarlyStopNeverBreaksCode:
    """The real model wrote a correct merge function and the novelty rule cut it
    off at `merged_list.append(list` -- correct code is repetitive by design."""

    MERGE = [
        "```", "python", "\n", "def", " merge", "(a", ",", " b", "):", "\n",
        "    merged", " =", " []", "\n", "    i", " =", " j", " =", " 0", "\n",
        *(["    while", " i", " <", " len", "(a", ")", " and", " j", " <", " len", "(b", "):", "\n",
           "        if", " a", "[i", "]", " <", " b", "[j", "]:", "\n",
           "            merged", ".append", "(a", "[i", "])", "\n",
           "            i", " +=", " 1", "\n"] * 6),
        "    return", " merged", "\n", "```",
    ]

    def test_code_questions_get_no_early_stop(self):
        from parsimony.modules.m5_budgeter import OutputBudgeter

        assert OutputBudgeter.stopper(full_stack(), ResponseClass.CODE) is None

    def test_prose_questions_still_do(self):
        from parsimony.modules.m5_budgeter import OutputBudgeter

        for rc in (ResponseClass.FACTUAL, ResponseClass.REASONING, ResponseClass.SUMMARISATION):
            assert isinstance(OutputBudgeter.stopper(full_stack(), rc), TrigramNoveltyStopper)

    def test_repetitive_code_inside_a_fence_is_never_stopped(self):
        stopper = TrigramNoveltyStopper()
        assert not any(stopper.observe(piece) for piece in self.MERGE)
        assert not stopper.in_code

    def test_the_same_code_without_a_fence_would_have_been_stopped(self):
        """Guards the guard: without it, this input really does trigger the rule."""
        stopper = TrigramNoveltyStopper()
        assert any(stopper.observe(piece) for piece in self.MERGE if piece != "```")

    def test_a_fence_split_across_two_pieces_is_still_seen(self):
        stopper = TrigramNoveltyStopper()
        stopper.observe("Here it is: ``")
        stopper.observe("`python")
        assert stopper.in_code
        stopper.observe("\n``")
        stopper.observe("`")
        assert not stopper.in_code

    def test_prose_after_the_code_block_is_judged_again(self):
        stopper = TrigramNoveltyStopper()
        for piece in self.MERGE:
            stopper.observe(piece)
        sentence = " This merges both lists in linear time overall."
        stopped = any(stopper.observe(" " + w) for w in (sentence * 2).split())
        assert stopped and stopper.reason == "restated a sentence"


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


class TestTheDateIsAnsweredByTheProcessNotTheModel:
    """qwen2.5-1.5b has no clock, and does not say so.

    Asked "What is the date today?" it answers "Today's date is [insert current
    date here]." -- not a refusal a reader can spot at a glance, but a sentence
    shaped like an answer with a hole in it. The host knows the date exactly, so
    tier 0 answers it and the model is never asked (ADR-016 puts this tier
    before the cache, which matters twice here: a cached date is wrong the next
    morning).
    """

    @pytest.mark.parametrize("question", [
        "What is the date today?",
        "What's today's date?",
        "whats todays date",
        "What is the date?",
        "What day is it?",
        "What day is it today?",
        "what is the current date",
        "what day of the week is it",
        "Tell me the date",
    ])
    def test_it_answers_exactly(self, question):
        from datetime import date

        from parsimony.modules.m6_router import solve

        result = solve(question)
        assert result is not None, f"{question!r} still goes to the model"
        answer, handler = result
        assert handler == "today"
        assert date.today().isoformat() in answer
        assert date.today().strftime("%A") in answer

    @pytest.mark.parametrize("question", [
        "What is the deadline date?",
        "When was the Porto office founded?",
        "What date was it?",
        "What is the date of the Porto acquisition?",
        "What day does the festival start?",
        "Will the date change?",
        "What is the date of birth on the form?",
    ])
    def test_it_leaves_alone_anything_that_is_not_about_today(self, question):
        """A question mentioning a date is not a question asking for today's."""
        from parsimony.modules.m6_router import solve

        result = solve(question)
        assert result is None or result[1] != "today", f"{question!r} was hijacked"

    def test_date_arithmetic_still_goes_to_its_own_handler(self):
        from parsimony.modules.m6_router import solve

        assert solve("How many days between 2026-01-01 and 2026-03-01?") == ("59", "date_diff")
        assert solve("30 days after 2026-01-01")[1] == "date_add"

    def test_the_model_is_never_called_for_it(self, tok):
        from parsimony.core.config import full_stack
        from parsimony.infra.providers import MockProvider
        from parsimony.pipeline.orchestrator import Pipeline

        pipe = Pipeline(full_stack(), provider=MockProvider(), tokenizer=tok)
        outcome = pipe.run("What is the date today?")
        assert outcome.row.route_tier == "DETERMINISTIC"
        assert outcome.row.tokens_in_final == 0
        assert outcome.generated is False


class TestThePromptMustFitTheWindow:
    """A prompt the runtime had to cut is a wrong answer waiting to happen.

    Ollama serves this model with a 2,048-token window unless told otherwise,
    and truncates in silence: a 4,662-token prompt came back reporting
    prompt_eval_count=2050. Nothing errors, nothing warns, and the run believes
    it measured a long context (ADR-045). Worse, a short prompt_eval_count was
    already being REPORTED as key-value cache reuse -- the same symptom, the
    opposite meaning, and the wrong one flatters the system.
    """

    def test_the_window_is_set_explicitly_and_is_not_the_default(self):
        from parsimony.infra.providers import DEFAULT_NUM_CTX, OllamaProvider

        assert DEFAULT_NUM_CTX >= 8192
        assert OllamaProvider().num_ctx == DEFAULT_NUM_CTX
        assert OllamaProvider(num_ctx=4096).num_ctx == 4096

    def test_overflow_is_decided_by_what_was_sent_not_by_the_reply(self):
        """The reply cannot tell you: asked for 4,662 tokens in a 2,048 window
        the server reported reading 1,026, so 'the count sits on num_ctx' is
        not the signature. Only the sender knows what it asked for."""
        from parsimony.infra.providers import window_overflow

        assert window_overflow(9000, {"num_ctx": 8192}) is True
        assert window_overflow(8192, {"num_ctx": 8192}) is True
        assert window_overflow(500, {"num_ctx": 8192}) is False
        assert window_overflow(None, {"num_ctx": 8192}) is False
        assert window_overflow(9000, {}) is False, "no window reported, no claim made"

    def test_truncation_is_not_reported_as_cache_reuse(self):
        from parsimony.surfaces.cli.live import LiveTurn

        view = LiveTurn("q", [])
        view.prompt_tokens = 9000
        view.stats = {"prompt_eval_count": 1026, "num_ctx": 8192}
        assert view.window_truncated is True

        reuse = LiveTurn("q", [])
        reuse.prompt_tokens = 400
        reuse.stats = {"prompt_eval_count": 120, "num_ctx": 8192}
        assert reuse.window_truncated is False, "a short read inside the window is reuse"

    def test_the_measured_panel_says_the_measurement_is_invalid(self):
        from parsimony.core.ledger import LedgerRow
        from parsimony.surfaces.cli.live import LiveTurn, measured_lines

        view = LiveTurn("q", [])
        view.prompt_tokens = 9000
        view.stats = {"prompt_eval_count": 1026, "num_ctx": 8192,
                      "prompt_eval_duration": 2_000_000_000}

        class _Outcome:
            generated = True
            row = LedgerRow(request_id="r", conversation_id="c", turn_index=0,
                            config_hash="h", run_id="run",
                            tokens_in_original=9000, tokens_in_final=9000)

        text = " ".join(measured_lines(_Outcome(), view))
        assert "did not fit" in text
        assert "not" in text and "valid" in text
        assert "earlier work was reused" not in text
