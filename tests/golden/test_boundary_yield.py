"""Golden tests for the position-0 boundary effect (ADR-030).

ADR-026 claimed whitespace-aligned deletion is monotone. That was true of its
495-deletion sample and false in general: on the expanded corpus three deletions
raise the token count, all of them removals of the first word.

These tests pin both the mechanism and the guard it forced, so the claim cannot
silently revert to the over-broad version.
"""

from __future__ import annotations

import pytest

from parsimony.core.config import full_stack
from parsimony.eval.corpus import load_corpus
from parsimony.infra.tokenization import HeuristicTokenizer, get_tokenizer
from parsimony.modules.m1_compressor import normalise_lossless


@pytest.fixture(scope="module")
def tok():
    t = get_tokenizer()
    if isinstance(t, HeuristicTokenizer):
        pytest.skip("real tokenizer unavailable; boundary claim is vocabulary-specific")
    return t


class TestBoundaryMechanism:
    """Two DISTINCT position-0 effects, often confused for one another."""

    @pytest.mark.parametrize("word", ["happened", "happens", "revoked", "quantify"])
    def test_effect_one_leading_space_form_is_cheaper(self, tok, word):
        """`" happened"` is 1 token; `"happened"` is 3. Losing the leading space
        is what makes first-word DELETION non-monotone."""
        assert tok.count(" " + word) < tok.count(word)

    def test_leading_space_is_not_always_cheaper(self, tok):
        """Scoping the claim: common words have a standalone token too, so the
        effect is real but not universal. Asserting the universal version is how
        ADR-030's first draft got the tier-1 mechanism wrong."""
        assert tok.count(" explain") == tok.count("explain") == 1

    def test_effect_two_capitalisation_costs_tokens(self, tok):
        """`"explain"` is 1 token; `"Explain"` is 2. Re-capitalising the opener
        is a SECOND mechanism, distinct from losing the leading space."""
        assert tok.count("Explain") > tok.count("explain")

    @pytest.mark.parametrize(
        "word,dominant",
        [
            ("explain", "capitalisation"),  # ' explain'=1, 'explain'=1, 'Explain'=2
            ("revoke", "leading space"),    # ' revoke'=1,  'revoke'=2,  'Revoke'=2
            ("quantify", "leading space"),  # ' quantify'=1,'quantify'=2,'Quantify'=2
        ],
    )
    def test_which_effect_dominates_is_word_dependent(self, tok, word, dominant):
        """The point that justifies the whole guard.

        Neither effect is universal and they do not apply to the same words. So
        whether an edit pays cannot be predicted from its shape — it has to be
        tokenised. That is precisely why negative-yield detection is a
        re-tokenisation and not a rule.
        """
        spaced, bare, capital = (tok.count(" " + word), tok.count(word),
                                 tok.count(word.capitalize()))
        if dominant == "capitalisation":
            assert spaced == bare < capital
        else:
            assert spaced < bare == capital

    def test_the_tier1_case_decomposes_as_expected(self, tok):
        """The exact arithmetic behind "5 of 9 removals saved nothing"."""
        assert tok.count("Please explain recursion.") == 4
        assert tok.count("Explain recursion.") == 4   # capital hands the saving back
        assert tok.count("explain recursion.") == 3   # lowercase would have paid

    @pytest.mark.parametrize(
        "full,trimmed",
        [
            ("What happened in the 1970s?", "happened in the 1970s?"),
            ("What happens without one?", "happens without one?"),
        ],
    )
    def test_removing_the_first_word_can_cost_tokens(self, tok, full, trimmed):
        """Shorter string, MORE tokens — the counterexample ADR-026 missed."""
        assert tok.count(trimmed) > tok.count(full)

    @pytest.mark.parametrize(
        "full,trimmed",
        [
            ("the value of x is 5", "the value x is 5"),
            ("a very good idea", "a good idea"),
            ("this is a long sentence here", "this is a sentence here"),
        ],
    )
    def test_mid_string_deletion_stays_monotone(self, tok, full, trimmed):
        """ADR-026's claim, correct once scoped to mid-string."""
        assert tok.count(trimmed) < tok.count(full)

    def test_the_corpus_still_contains_counterexamples(self, tok):
        """If this stops finding any, the corpus changed and ADR-030's
        supporting evidence needs re-deriving rather than assuming."""
        raising = 0
        for conv in load_corpus().conversations:
            for text in conv.user_turns:
                base = tok.count(text)
                words = text.split(" ")
                for i in range(len(words)):
                    if tok.count(" ".join(words[:i] + words[i + 1:])) > base:
                        raising += 1
        assert raising > 0, "no non-monotone deletions found; re-derive ADR-030"


class TestTier1YieldGuard:
    @pytest.mark.parametrize(
        "text",
        [
            "Please explain recursion.",
            "Please revoke the token.",
            "Please quantify the risk.",
            "Please happen upon it.",
        ],
    )
    def test_zero_yield_normalisations_are_rejected(self, tok, text, pipeline):
        """Tier 1 must not perturb text for no saving. Each of these removes a
        leading politeness word and hands the saving straight back at position 0.
        """
        from parsimony.core.proposals import NoOp

        # The rewrite itself does not pay...
        assert tok.count(normalise_lossless(text)) >= tok.count(text)
        # ...so the stage must decline it.
        ctx = pipeline.build_context(text)
        result = pipeline.registry.get("m1_tier1").propose(ctx, full_stack())
        assert isinstance(result, NoOp)

    @pytest.mark.parametrize(
        "text",
        [
            "Hello, what is recursion?",
            "Kindly summarise this.",
            "Thanks, explain hashing.",
        ],
    )
    def test_genuinely_paying_normalisations_still_apply(self, tok, text, pipeline):
        """The guard must not cost real savings."""
        from parsimony.core.proposals import ContextPatch

        assert tok.count(normalise_lossless(text)) < tok.count(text)
        ctx = pipeline.build_context(text)
        assert isinstance(pipeline.registry.get("m1_tier1").propose(ctx, full_stack()),
                          ContextPatch)

    def test_tier1_never_increases_the_payload(self, pipeline):
        """The invariant the guard exists to hold, over the whole corpus."""
        from parsimony.core.proposals import ContextPatch

        for conv in load_corpus().conversations:
            for text in conv.user_turns:
                ctx = pipeline.build_context(text)
                result = pipeline.registry.get("m1_tier1").propose(ctx, full_stack())
                if isinstance(result, ContextPatch):
                    assert result.evidence["tokens_after"] < result.evidence["tokens_before"]
