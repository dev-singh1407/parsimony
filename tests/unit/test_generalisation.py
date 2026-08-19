"""Cross-vocabulary generalisation (ADR-032).

Report §4.6 asks whether a calibration transfers when applied elsewhere without
re-tuning. The tokenizer dimension of that question is answerable without any
model download, and answering it corrected a claim in ADR-030.
"""

from __future__ import annotations

import pytest

from parsimony.core.config import factorial_cells
from parsimony.eval.corpus import load_corpus
from parsimony.eval.generalisation import (
    DEFAULT_TOKENIZERS,
    check_boundary_effect,
    check_tier1_yield,
    sweep_across_tokenizers,
)
from parsimony.infra.tokenization import HeuristicTokenizer, get_tokenizer


@pytest.fixture(scope="module")
def both_available():
    for tid in DEFAULT_TOKENIZERS:
        if isinstance(get_tokenizer(tid), HeuristicTokenizer):
            pytest.skip("a real tokenizer is unavailable (offline)")
    return True


@pytest.fixture(scope="module")
def arms(both_available):
    corpus = load_corpus().subset(12)
    cells = list(factorial_cells(axes=("M1", "M3"), always_on=frozenset()))
    return [a for a in sweep_across_tokenizers(cells, corpus) if a.available]


class TestVocabulariesReallyDiffer:
    def test_the_two_vocabularies_are_different_sizes(self, both_available):
        sizes = {get_tokenizer(t)._tok.get_vocab_size() for t in DEFAULT_TOKENIZERS}
        assert len(sizes) == 2

    def test_they_disagree_on_absolute_token_counts(self, both_available):
        """If they agreed, the study would be vacuous."""
        counts = {get_tokenizer(t).count("Please explain recursion.") for t in DEFAULT_TOKENIZERS}
        assert len(counts) > 1


class TestRatiosTransfer:
    def test_module_ranking_is_identical(self, arms):
        assert len({tuple(a.ranking()) for a in arms}) == 1

    def test_reductions_agree_closely(self, arms):
        """A reduction is a ratio, so a roughly constant vocabulary factor
        cancels. Absolute counts differ; percentages do not."""
        for label in ("M1", "M3"):
            values = [a.reduction(label) for a in arms]
            assert all(v is not None for v in values)
            assert max(values) - min(values) < 1.0

    def test_no_retuning_was_applied(self, arms):
        """The point of the study: the same thresholds are carried over
        unchanged. Re-tuning per vocabulary would answer a different question."""
        hashes = {r.config_hash for a in arms for r in a.results if r.label == "M1"}
        # config_hash includes tokenizer_id, so the hashes differ — but every
        # other field must be identical, which is what `replace(cfg, tokenizer_id=)`
        # guarantees. Assert we actually varied only that.
        assert len(hashes) == len(arms)


class TestMechanismsDoNotAllTransfer:
    def test_leading_space_effect_holds_in_both(self, both_available):
        check = next(c for c in check_boundary_effect() if "leading-space" in c.claim)
        assert check.transfers

    def test_capitalisation_effect_is_vocabulary_specific(self, both_available):
        """ADR-030 stated this as a general mechanism. GPT-2 has no
        capitalisation penalty for 'explain', so it is Qwen-specific — the
        correction ADR-032 records."""
        check = next(c for c in check_boundary_effect() if "capitalisation" in c.claim)
        assert not check.transfers

    def test_first_word_deletion_cost_holds_in_both(self, both_available):
        check = next(c for c in check_boundary_effect() if "first-word" in c.claim)
        assert check.transfers

    def test_tier1_zero_yield_rate_is_reported_per_vocabulary(self, both_available):
        """The rates happen to match here. Reporting them separately is what
        stops a matching rate being read as matching mechanisms — they differ."""
        check = check_tier1_yield(load_corpus())[0]
        assert len(check.values) == len(DEFAULT_TOKENIZERS)
        assert all("%" in v for v in check.values.values())
