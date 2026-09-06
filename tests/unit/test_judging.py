"""A real judge, and the checks that decide whether to believe it.

The ablation's quality columns come from `LengthBiasedMockJudge`, a stand-in
that prefers the longer answer. `ModelJudge` replaces it with a real model —
and the point of this module is that a real judge is not automatically better,
only measurable.
"""

from __future__ import annotations

import pytest

from parsimony.core.types import TokenEvent
from parsimony.eval.judging import JudgeCalibration, calibrate_judge
from parsimony.eval.metrics import (
    JUDGE_PROMPT,
    LengthBiasedMockJudge,
    ModelJudge,
    judge_pairwise,
)


class Scripted:
    """A provider that returns fixed strings, one per call."""

    model_name = "scripted"

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.calls = 0

    def generate(self, prompt, params):
        self.calls += 1
        text = self.replies[(self.calls - 1) % len(self.replies)]
        return iter([TokenEvent(text=text, index=0, emitted_at_ns=0)])


class Exploding:
    model_name = "exploding"

    def generate(self, prompt, params):
        raise RuntimeError("provider is down")


class TestVerdictParsing:
    """Asked for "exactly A or B", a small instruct model answers in prose."""

    @pytest.mark.parametrize(
        "reply,expected",
        [
            ("A", "A"),
            ("B", "B"),
            (" A ", "A"),
            ("Answer: A", "A"),
            ("**B**", "B"),
            ("The better answer is B because it is clearer.", "B"),
            ("A is shorter, so B is better.", "B"),
            ("b", "B"),
        ],
    )
    def test_reads_the_verdict(self, reply, expected):
        assert ModelJudge(Scripted(reply)).compare("x") == expected

    def test_prefers_the_last_mention(self):
        """A model that reasons before deciding ends on its verdict."""
        assert ModelJudge(Scripted("A has more detail but B is correct")).compare("x") == "B"

    def test_an_article_is_not_a_verdict(self):
        """Upper-casing the reply first read the article in "A better answer is
        B" as a verdict of A. Matching is case-sensitive for that reason."""
        assert ModelJudge(Scripted("A better answer is B")).compare("x") == "B"

    @pytest.mark.parametrize("reply", ["I cannot decide.", "Both are equally good", ""])
    def test_unreadable_is_reported_not_guessed(self, reply):
        judge = ModelJudge(Scripted(reply))
        assert judge.compare("x") == ""
        assert judge.unreadable == 1

    def test_a_provider_failure_is_unreadable_not_a_crash(self):
        judge = ModelJudge(Exploding())
        assert judge.compare("x") == ""
        assert judge.unreadable == 1


class TestUnreadableScoresAsTie:
    """Before ModelJudge, every judge returned A or B by construction, so this
    case could not arise. Parsing prose as a bare False on both sides reads as
    "the swap agreed and the candidate lost" — a systematic bias against the
    candidate, injected by the judge's verbosity, inside the machinery built to
    detect judge bias."""

    def test_prose_on_both_sides_is_a_tie_not_a_loss(self):
        judge = ModelJudge(Scripted("I cannot decide."))
        verdict = judge_pairwise("q", "candidate", "reference", judge)
        assert not verdict.readable
        assert verdict.score == 0.5

    def test_a_readable_win_still_scores_one(self):
        judge = ModelJudge(Scripted("A", "B"))
        verdict = judge_pairwise("q", "candidate", "reference", judge)
        assert verdict.readable and verdict.swap_agreed
        assert verdict.score == 1.0

    def test_a_readable_loss_still_scores_zero(self):
        judge = ModelJudge(Scripted("B", "A"))
        assert judge_pairwise("q", "candidate", "reference", judge).score == 0.0

    def test_position_disagreement_is_a_tie(self):
        judge = ModelJudge(Scripted("A", "A"))
        verdict = judge_pairwise("q", "candidate", "reference", judge)
        assert verdict.readable and not verdict.swap_agreed
        assert verdict.score == 0.5


class TestCalibration:
    """Measuring the judge before reporting what it said."""

    def test_a_judge_that_always_picks_a_slot_is_unusable(self):
        cal = calibrate_judge(ModelJudge(Scripted("A")), ["q"] * 10, ["same"] * 10)
        assert cal.position_bias == pytest.approx(50.0)
        assert not cal.usable

    def test_a_balanced_judge_is_usable(self):
        cal = calibrate_judge(ModelJudge(Scripted("A", "B")), ["q"] * 10, ["same"] * 10)
        assert cal.position_bias == pytest.approx(0.0)
        assert cal.usable

    def test_an_unreadable_judge_is_unusable(self):
        cal = calibrate_judge(ModelJudge(Scripted("hmm")), ["q"] * 10, ["same"] * 10)
        assert cal.unreadable_rate == 100.0
        assert not cal.usable

    def test_bias_is_counted_over_raw_verdicts(self):
        """An earlier version counted A-preferences only among swap-AGREEING
        pairs. With identical answers the two orderings are the same prompt, so
        a deterministic judge never "agrees" — the set was empty and the metric
        reported a constant 50.0 whatever the judge did."""
        always_a = calibrate_judge(ModelJudge(Scripted("A")), ["q"] * 8, ["s"] * 8)
        balanced = calibrate_judge(ModelJudge(Scripted("A", "B")), ["q"] * 8, ["s"] * 8)
        assert always_a.position_bias != balanced.position_bias

    def test_empty_input_does_not_divide_by_zero(self):
        cal = JudgeCalibration(n=0, unreadable=0, chose_a=0)
        assert cal.position_bias == 0.0 and cal.unreadable_rate == 0.0


class TestTheMockJudgeStillWorks:
    """It is still the default in the sweep, because it is deterministic and
    free — and because a judge that is deliberately biased in a KNOWN direction
    is a better test double than one that is unpredictably biased."""

    def test_it_prefers_the_longer_answer(self):
        judge = LengthBiasedMockJudge()
        assert judge.compare(JUDGE_PROMPT.format(question="q", a="long answer", b="hi")) == "A"
        assert judge.compare(JUDGE_PROMPT.format(question="q", a="hi", b="long answer")) == "B"

    def test_the_swap_detects_that_bias_as_agreement(self):
        """Length bias is consistent under the swap, so it shows up as a
        confident verdict rather than as noise — which is precisely why a
        disagreement RATE alone cannot certify a judge."""
        verdict = judge_pairwise("q", "a much longer candidate answer", "short",
                                 LengthBiasedMockJudge())
        assert verdict.swap_agreed and verdict.score == 1.0
