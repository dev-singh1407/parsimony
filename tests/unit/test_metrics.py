"""Quality measurement: grading rules, proxies, and judge bias detection."""

from __future__ import annotations

import pytest

from parsimony.eval.corpus import GoldItem, load_gold
from parsimony.eval.metrics import (
    LengthBiasedMockJudge,
    QualityVector,
    embedding_similarity,
    grade,
    judge_pairwise,
    token_overlap,
)
from parsimony.infra.embedding import HashingEmbedder


class TestGoldGrading:
    def test_numeric_exact(self):
        item = GoldItem("g", "q", "210", "numeric", 0.0, ("210 minutes",))
        assert grade("The answer is 210 minutes.", item)
        assert not grade("The answer is 200 minutes.", item)

    def test_numeric_tolerance(self):
        item = GoldItem("g", "q", "62.137", "numeric", 0.01, ())
        assert grade("about 62.14 miles", item)
        assert not grade("about 62.5 miles", item)

    def test_numeric_finds_the_value_among_others(self):
        item = GoldItem("g", "q", "429", "numeric", 0.0, ())
        assert grade("With a limit of 100 per minute you get HTTP 429.", item)

    def test_contains_is_case_insensitive(self):
        item = GoldItem("g", "q", "Canberra", "contains", 0.0, ())
        assert grade("the capital is canberra, not Sydney", item)

    def test_contains_accepts_a_declared_variant(self):
        item = GoldItem("g", "q", "George Orwell", "contains", 0.0, ("Orwell",))
        assert grade("It was written by Orwell.", item)

    def test_exact_requires_the_whole_answer(self):
        item = GoldItem("g", "q", "W", "exact", 0.0, ())
        assert grade("W", item)
        assert not grade("The symbol is W.", item)

    def test_empty_response_never_passes(self):
        assert not grade("", GoldItem("g", "q", "42", "numeric", 0.0, ()))

    def test_unknown_rule_is_an_error_not_a_silent_pass(self):
        with pytest.raises(ValueError):
            grade("anything", GoldItem("g", "q", "x", "vibes", 0.0, ()))

    def test_every_gold_item_uses_a_supported_rule(self):
        for item in load_gold():
            grade("placeholder", item)  # must not raise


class TestTokenOverlap:
    def test_identical_text_scores_one(self):
        assert token_overlap("the cat sat", "the cat sat") == pytest.approx(1.0)

    def test_disjoint_text_scores_zero(self):
        assert token_overlap("alpha beta", "gamma delta") == 0.0

    def test_penalises_concision_by_construction(self):
        """The documented structural bias against M5: a correct shorter answer
        scores lower than a verbose one. Stated rather than hidden."""
        reference = "Water boils at 100 degrees Celsius at sea level under standard pressure"
        terse = "100 C"
        verbose = "Water boils at 100 degrees Celsius at sea level under normal pressure"
        assert token_overlap(terse, reference) < token_overlap(verbose, reference)

    def test_handles_empty_input(self):
        assert token_overlap("", "something") == 0.0


class TestEmbeddingSimilarity:
    def test_identical_text_scores_one(self):
        e = HashingEmbedder()
        assert embedding_similarity("hello world", "hello world", e) == pytest.approx(1.0, abs=1e-5)

    def test_unrelated_text_scores_low(self):
        e = HashingEmbedder()
        assert embedding_similarity("quantum physics", "banana bread recipe", e) < 0.3


class TestJudge:
    def test_position_swap_detects_a_length_biased_judge(self):
        """The mock judge prefers the longer answer. Because that preference
        follows the TEXT rather than the POSITION, it agrees with itself on the
        swap — so a consistent bias is not flagged as noise. This is exactly the
        distinction the swap check is meant to draw."""
        verdict = judge_pairwise(
            "q", "short", "a considerably longer answer than the other one",
            LengthBiasedMockJudge(),
        )
        assert verdict.swap_agreed
        assert not verdict.prefers_candidate
        assert verdict.score == 0.0

    def test_equal_length_answers_produce_a_tie(self):
        """When the judge has nothing to go on it flips with position, the swap
        disagrees, and the result is scored as a tie rather than a win."""
        verdict = judge_pairwise("q", "aaaa", "bbbb", LengthBiasedMockJudge())
        assert not verdict.swap_agreed
        assert verdict.score == 0.5

    def test_a_disagreeing_judge_never_scores_a_win(self):
        verdict = judge_pairwise("q", "same", "same", LengthBiasedMockJudge())
        assert verdict.score == 0.5


class TestQualityVector:
    def test_has_no_combined_score(self):
        """Averaging a proxy with a ground truth manufactures confidence.
        If someone adds an `overall` property, this test should stop them."""
        assert not hasattr(QualityVector(), "overall")
        assert not hasattr(QualityVector(), "mean")

    def test_serialises_all_four_measures_separately(self):
        keys = QualityVector().as_dict().keys()
        assert {"q_embedding_sim", "q_token_overlap", "q_judge", "q_exact_match"} <= set(keys)
