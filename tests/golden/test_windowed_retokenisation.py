"""Golden test guarding M1 tier 3's windowed re-tokenisation.

Tier 3 decides whether an edit pays by re-tokenising a +/-W character window
around it rather than the whole text, because the full computation is O(edits x
length) and blows the 120ms overhead budget on a long prompt.

That optimisation is only safe if byte-pair merges are local enough that the
window contains every merge the edit could disturb. "Almost always" is not good
enough for a claim the project rests on, so this re-tokenises the FULL text for
every candidate edit across the entire corpus and asserts the windowed DECISION
matches. If it ever fails, widen CompressionConfig.retokenise_window.

This is the test the module docstring refers to when it says tier 3 must not
ship without it.
"""

from __future__ import annotations

import pytest

from parsimony.core.config import CompressionConfig
from parsimony.eval.corpus import load_corpus
from parsimony.infra.tokenization import HeuristicTokenizer, get_tokenizer
from parsimony.modules.m1_tier3 import (
    DEFAULT_LEXICON,
    find_candidates,
    full_delta,
    windowed_delta,
)

SYNTHETIC = [
    "You need to do this in order to succeed at the task.",
    "It is important to note that the budget is 50,000 dollars.",
    "Due to the fact that a large number of users complained, we acted.",
    "Prior to the meeting, please review the majority of the documents.",
    "In the event that it fails, we are able to retry on a regular basis.",
    "With regard to your question: in order to proceed, act prior to Friday.",
    "```python\nx = 1  # in order to keep this\n```\nIn order to explain it.",
    "A large number of records exist. The majority of them are stale.",
]


def _corpus_texts() -> list[str]:
    texts = list(SYNTHETIC)
    texts += [q for c in load_corpus().conversations for q in c.user_turns]
    return texts


@pytest.fixture(scope="module")
def real_tokenizer():
    tok = get_tokenizer()
    if isinstance(tok, HeuristicTokenizer):
        pytest.skip("real tokenizer unavailable (offline); windowing claim untested")
    return tok


class TestWindowEquivalence:
    def test_windowed_decision_matches_full_retokenisation(self, real_tokenizer):
        window = CompressionConfig().retokenise_window
        checked = 0
        disagreements = []
        for text in _corpus_texts():
            for edit in find_candidates(text, DEFAULT_LEXICON):
                checked += 1
                windowed = windowed_delta(text, edit, real_tokenizer, window)
                full = full_delta(text, edit, real_tokenizer)
                if (windowed < 0) != (full < 0):
                    disagreements.append((text, edit.matched, windowed, full))
        assert checked > 0, "probe found no candidate edits — lexicon or corpus broken"
        assert not disagreements, (
            f"windowed re-tokenisation disagreed with full re-tokenisation on "
            f"{len(disagreements)}/{checked} edits: {disagreements[:3]}"
        )

    def test_windowed_delta_equals_full_delta_exactly(self, real_tokenizer):
        """Stronger than the decision test: the magnitudes should agree too.

        If this fails while the decision test passes, the window is marginal —
        correct today, fragile tomorrow.
        """
        window = CompressionConfig().retokenise_window
        mismatches = []
        for text in _corpus_texts():
            for edit in find_candidates(text, DEFAULT_LEXICON):
                w = windowed_delta(text, edit, real_tokenizer, window)
                f = full_delta(text, edit, real_tokenizer)
                if w != f:
                    mismatches.append((edit.matched, w, f))
        assert not mismatches, f"magnitude mismatch on {len(mismatches)} edits: {mismatches[:5]}"

    def test_a_narrow_window_is_detectably_worse(self, real_tokenizer):
        """Sanity check on the test itself: with a 0-character window the
        computation degenerates, so the suite would catch a regression that
        shrank the window to nothing."""
        results = {
            w: sum(
                windowed_delta(t, e, real_tokenizer, w)
                for t in SYNTHETIC
                for e in find_candidates(t, DEFAULT_LEXICON)
            )
            for w in (0, CompressionConfig().retokenise_window)
        }
        assert results[0] != results[CompressionConfig().retokenise_window]


class TestNegativeYieldIsMeasured:
    def test_whitespace_aligned_phrase_edits_are_monotone(self, real_tokenizer):
        """Measured finding (ADR-026), pinned as a regression test.

        Every lexicon substitution in our corpus reduces or preserves the token
        count — none increases it. If a future lexicon entry breaks this, the
        negative-yield guard is what stops it reaching a prompt, and this test
        is what tells us the regime changed.
        """
        increases = [
            (t, e.matched)
            for t in _corpus_texts()
            for e in find_candidates(t, DEFAULT_LEXICON)
            if full_delta(t, e, real_tokenizer) > 0
        ]
        assert not increases, f"phrase-level edits that RAISED token count: {increases[:5]}"

    def test_subtoken_edits_can_raise_the_token_count(self, real_tokenizer):
        """The other half of ADR-026: the phenomenon is real, just not in the
        regime phrase compression operates in."""
        from parsimony.eval.tokenizer_probe import SUBTOKEN_CASES

        raised = [
            (before, after)
            for _label, before, after in SUBTOKEN_CASES
            if real_tokenizer.count(after) > real_tokenizer.count(before)
        ]
        assert raised, "expected at least one sub-token edit to increase the token count"

    def test_shortening_often_saves_nothing(self, real_tokenizer):
        """The check's real value: rejecting ZERO-yield edits.

        An edit that shortens the text without saving a token is pure risk — it
        perturbs meaning for no gain — and these are far more common than true
        negative-yield edits.
        """
        from parsimony.eval.tokenizer_probe import SUBTOKEN_CASES

        wasted = [
            (b, a)
            for _l, b, a in SUBTOKEN_CASES
            if len(a) < len(b) and real_tokenizer.count(a) >= real_tokenizer.count(b)
        ]
        assert len(wasted) >= 3
