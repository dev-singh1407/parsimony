"""Invisible characters defeated the verifier. These tests keep them out.

The cache verifier reads negation particles and operative modifiers out of raw
text. A single zero-width character inside "not" splits it into fragments that
match no lexicon entry, so the negation check silently agreed and
`verify_match` passed a question against its own opposite.

Nothing caught this because the adversarial corpus is written in plain ASCII —
the false-hit rate of 0.0% is measured on text that contains no such character.
The only thing preventing a wrong answer was that the embedder happened to
score the mangled text below tau_hi, which is luck rather than a defence.
"""

from __future__ import annotations

import pytest

from parsimony.core.config import full_stack
from parsimony.infra.nlp import (
    RegexInvariantExtractor,
    jaccard,
    morphological_negations,
    operative_modifiers,
    sanitise,
)
from parsimony.modules.m2_cache import verify_match

SAFE = "Is it safe to mix bleach and vinegar?"

ATTACKS = {
    "plain": "Is it not safe to mix bleach and vinegar?",
    "zero_width_space": "Is it n​ot safe to mix bleach and vinegar?",
    "zero_width_joiner": "Is it n‍ot safe to mix bleach and vinegar?",
    "zero_width_non_joiner": "Is it n‌ot safe to mix bleach and vinegar?",
    "soft_hyphen": "Is it n­ot safe to mix bleach and vinegar?",
    "cyrillic_o": "Is it nоt safe to mix bleach and vinegar?",
    "full_width": "Is it ｎｏｔ safe to mix bleach and vinegar?",
    "rtl_override": "Is it ‮not safe to mix bleach and vinegar?",
    "word_joiner": "Is it n⁠ot safe to mix bleach and vinegar?",
}


@pytest.fixture(scope="module")
def extractor():
    return RegexInvariantExtractor()


class TestTheVerifierIsNoLongerBypassable:
    @pytest.mark.parametrize("name", sorted(ATTACKS))
    def test_negation_is_still_detected(self, name, extractor):
        assert "not" in extractor.extract(ATTACKS[name]).negations, name

    @pytest.mark.parametrize("name", sorted(ATTACKS))
    def test_the_opposite_question_is_refused(self, name, extractor):
        result = verify_match(
            extractor.extract(SAFE),
            extractor.extract(ATTACKS[name]),
            SAFE,
            ATTACKS[name],
            full_stack().cache.jaccard_min,
        )
        assert not result.passed, f"{name} would be served the opposite answer"
        assert not result.negation_agree


class TestSanitiseItself:
    @pytest.mark.parametrize(
        "raw",
        ["n​ot", "n‍ot", "n‌ot", "n­ot", "n⁠ot", "‮not"],
    )
    def test_invisible_characters_are_removed(self, raw):
        assert sanitise(raw) == "not"

    def test_confusable_letters_are_folded(self):
        assert sanitise("nоt") == "not"
        assert sanitise("рass") == "pass"

    def test_compatibility_forms_are_folded(self):
        assert sanitise("ｎｏｔ") == "not"

    def test_ordinary_text_is_untouched(self):
        for text in ("Is it safe?", "café", "naïve", "3.5 hours", ""):
            assert sanitise(text) == text

    def test_legitimate_non_latin_text_survives(self):
        """Folding must not destroy a language. Only characters that
        impersonate Latin letters are mapped; the rest pass through."""
        for text in ("こんにちは 世界", "مرحبا", "Ελληνικά κείμενο"):
            assert sanitise(text).strip()

    def test_visible_whitespace_is_preserved(self):
        """Cf is invisible; Zs is not. Stripping real spaces would merge words."""
        assert sanitise("a b") == "a b"
        assert sanitise("a b") == "a b"  # NFKC folds nbsp to a plain space

    def test_is_idempotent(self):
        for raw in ATTACKS.values():
            assert sanitise(sanitise(raw)) == sanitise(raw)


class TestTheLexicalPathsAreSanitisedToo:
    """The verifier does not rely on invariants alone — a bypass in any one of
    these is a bypass overall."""

    def test_operative_modifiers_see_through_it(self):
        assert operative_modifiers("the m​inimum value") == operative_modifiers(
            "the minimum value"
        )

    def test_morphological_negation_sees_through_it(self):
        assert morphological_negations("that is imp​ossible", "is it possible") == \
               morphological_negations("that is impossible", "is it possible")

    def test_jaccard_is_not_inflated_by_invisible_characters(self):
        """Two texts differing only by an invisible character are the same text,
        so lexical overlap must be total rather than penalised."""
        assert jaccard("mix bleach and vinegar", "mix bl​each and vinegar") == 1.0


class TestTheModelStillSeesWhatTheUserWrote:
    def test_sanitising_is_analysis_only(self):
        """A user who legitimately writes Cyrillic must get their own words
        back; folding is for matching, not for rewriting the prompt."""
        from parsimony.pipeline.orchestrator import Pipeline

        query = "Как дела?"
        outcome = Pipeline(full_stack()).run(query)
        assert query in outcome.ctx.text_payload()


class TestNonLatinQueriesSurvive:
    """M1 tier 1 judged a sentence "contentless" with [A-Za-z0-9], so every
    query written without Latin letters was deleted whole and the model
    received an EMPTY prompt. For a project written at an Indian university,
    a Hindi or Tamil question vanished silently."""

    @pytest.mark.parametrize(
        "query",
        [
            "Как дела?",
            "नमस्ते, यह क्या है?",
            "இது என்ன?",
            "你好世界",
            "مرحبا كيف حالك؟",
            "こんにちは、これは何ですか",
        ],
    )
    def test_the_query_reaches_the_model(self, query):
        from parsimony.pipeline.orchestrator import Pipeline

        payload = Pipeline(full_stack()).run(query).ctx.text_payload()
        assert payload.strip(), f"{query!r} was deleted entirely"
        assert query in payload

    def test_english_compression_still_works(self):
        """The fix must not disable tier 1 — only stop it eating alphabets it
        cannot read."""
        from parsimony.pipeline.orchestrator import Pipeline

        verbose = "Hello, could you please explain photosynthesis? Thanks!"
        payload = Pipeline(full_stack()).run(verbose).ctx.text_payload()
        assert "Hello" not in payload and "Thanks" not in payload
        assert "photosynthesis" in payload


class TestTheGateRefusesAnnihilation:
    """The deeper bug. Every other gate check asks "was a value I could extract
    lost?", which makes the guarantee conditional on the extractor's coverage.
    Non-Latin text yields no invariants, so deleting the whole question lost
    nothing the gate could name — and it passed."""

    @pytest.mark.parametrize("query", ["Как дела?", "What is this?", "你好世界", "নমস্কার"])
    def test_deleting_everything_is_refused(self, query):
        from dataclasses import replace

        from parsimony.core.proposals import TransformKind
        from parsimony.modules.m8_fidelity import FidelityGate
        from parsimony.pipeline.orchestrator import Pipeline

        pipeline = Pipeline(full_stack())
        before = pipeline.build_context(query)
        verdict = FidelityGate().check(
            before, replace(before, query=""), TransformKind.REWRITE, "M1"
        )
        assert not verdict.passed
        assert verdict.detail == "rewrite removed all content"

    def test_an_unchanged_rewrite_still_passes(self):
        from parsimony.core.proposals import TransformKind
        from parsimony.modules.m8_fidelity import FidelityGate
        from parsimony.pipeline.orchestrator import Pipeline

        pipeline = Pipeline(full_stack())
        ctx = pipeline.build_context("What is this?")
        assert FidelityGate().check(ctx, ctx, TransformKind.REWRITE, "M1").passed

    def test_reducing_to_punctuation_is_refused(self):
        """"Thanks in advance!" -> "!" is the same failure in miniature."""
        from dataclasses import replace

        from parsimony.core.proposals import TransformKind
        from parsimony.modules.m8_fidelity import FidelityGate
        from parsimony.pipeline.orchestrator import Pipeline

        pipeline = Pipeline(full_stack())
        before = pipeline.build_context("Thanks in advance!")
        verdict = FidelityGate().check(
            before, replace(before, query="!"), TransformKind.REWRITE, "M1"
        )
        assert not verdict.passed
