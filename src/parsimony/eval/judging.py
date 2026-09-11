"""LLM-as-judge, with the judge itself measured first.

The quality columns in the ablation come from `LengthBiasedMockJudge`, a
deliberate stand-in that prefers the longer answer — built so the
position-swap machinery could be shown to detect bias before a real model
existed. This module replaces it with a real one, and does the thing most
LLM-as-judge setups skip: it measures whether the judge is worth listening to
before reporting what it said.

Three constraints, from `judge_pairwise`:

  * the judge is a DIFFERENT model from the one under test — a model comparing
    its own output against another's prefers its own, which measures
    familiarity, not quality;
  * pairwise, never absolute, because small models cannot produce calibrated
    1-10 scores;
  * every comparison runs A/B and B/A, and disagreement counts as a tie.

Why this is a separate study rather than a column in the sweep: a full
factorial with a real judge is 263 requests x 17 cells x 2 orderings ~ 8,900
model calls, hours on CPU. A stratified subset against the baseline answers
the actual question — does compression change answer quality — at a cost that
fits in a demo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from parsimony.core.config import ParsimonyConfig
from parsimony.eval.corpus import Corpus
from parsimony.eval.metrics import JUDGE_PROMPT, judge_pairwise
from parsimony.pipeline.orchestrator import Pipeline


@dataclass(frozen=True, slots=True)
class JudgeCalibration:
    """What the judge does when there is nothing to choose between.

    Handed two IDENTICAL answers, an unbiased pairwise judge is at chance: it
    should name slot A about half the time. A judge that consistently picks the
    same slot has a position bias no amount of downstream averaging removes,
    and every score it produces afterwards is suspect.

    Running this BEFORE reporting any score is the step most LLM-as-judge
    setups omit. It is cheap, it needs no ground truth, and it is the
    difference between "the compressed answers won 47% of the time" and "the
    judge is not measuring anything".
    """

    n: int
    unreadable: int
    chose_a: int

    @property
    def position_bias(self) -> float:
        """Percentage points away from the unbiased 50%. 0 is perfect.

        Counted over RAW verdicts, not over `swap_agreed` ones. With identical
        answers the two orderings are the same prompt, so a deterministic judge
        answers identically and `swap_agreed` is false by construction — an
        earlier version counted A-preferences only among agreeing pairs, which
        is an empty set, and so reported a constant 50.0 regardless of what the
        judge actually did.
        """
        readable = self.n - self.unreadable
        if not readable:
            return 0.0
        return abs(100.0 * self.chose_a / readable - 50.0)

    @property
    def unreadable_rate(self) -> float:
        return 100.0 * self.unreadable / self.n if self.n else 0.0

    @property
    def usable(self) -> bool:
        """A judge that cannot be read, or that always picks a side when there
        is nothing to choose between, is noise dressed as a measurement."""
        return self.unreadable_rate < 20.0 and self.position_bias < 30.0


@dataclass(slots=True)
class JudgeArm:
    label: str
    scores: list[float] = field(default_factory=list)
    swap_disagreements: int = 0
    unreadable: int = 0

    @property
    def n(self) -> int:
        return len(self.scores)

    @property
    def win_rate(self) -> float:
        """Mean pairwise score against the baseline. 50% is parity."""
        return 100.0 * sum(self.scores) / self.n if self.n else 0.0

    @property
    def disagreement_rate(self) -> float:
        return 100.0 * self.swap_disagreements / self.n if self.n else 0.0

    @property
    def unreadable_rate(self) -> float:
        return 100.0 * self.unreadable / self.n if self.n else 0.0


def calibrate_judge(judge, questions: list[str], answers: list[str]) -> JudgeCalibration:
    """Ask the judge to choose between an answer and itself.

    Goes through `judge.compare` directly rather than `judge_pairwise`: with
    identical texts the two orderings produce the same prompt, so the swap adds
    nothing and the interesting quantity is simply which slot the judge names
    when the two are indistinguishable. At chance that is 50/50.
    """
    unreadable = chose_a = 0
    for q, ans in zip(questions, answers):
        verdict = judge.compare(JUDGE_PROMPT.format(question=q, a=ans, b=ans)).strip().upper()
        if verdict.startswith("A"):
            chose_a += 1
        elif not verdict.startswith("B"):
            unreadable += 1
    return JudgeCalibration(n=len(answers), unreadable=unreadable, chose_a=chose_a)


def admit_judge(judge, corpus: Corpus, sample: int = 20):
    """Calibrate a judge and refuse it if it fails.

    The sweep used to take whatever judge it was handed and print its verdicts,
    with a sentence underneath advising the reader to discount them when the
    disagreement rate was high. That is the wrong place for the check: a
    measurement shown to be invalid should not be reported at all, and a note
    asking the reader to do the discounting is how an unusable number ends up
    quoted in someone's slides.

    Calibration needs no ground truth -- it hands the judge two identical
    answers and asks which is better. Anything other than a coin flip is
    position bias.

    Returns ``(judge_or_None, calibration)``. A ``None`` judge means the caller
    must omit the judge column rather than fill it.
    """
    texts = [
        t
        for conv in corpus.conversations
        for t in conv.user_turns
    ][:sample]
    if not texts:
        return None, JudgeCalibration(n=0, unreadable=0, chose_a=0)
    calibration = calibrate_judge(judge, texts, texts)
    return (judge if calibration.usable else None), calibration


@dataclass(frozen=True, slots=True)
class JudgeStudy:
    judge_model: str
    subject_model: str
    calibration: JudgeCalibration
    arms: list[JudgeArm]

    @property
    def independent(self) -> bool:
        """The judge must not be the model under test."""
        return self.judge_model != self.subject_model


def run_judge_study(
    corpus: Corpus,
    baseline_cfg: ParsimonyConfig,
    arms: list[tuple[str, ParsimonyConfig]],
    *,
    judge,
    provider,
    n_sample: int = 30,
    progress=None,
) -> JudgeStudy:
    """Compare each arm's answers against the baseline's, judged by a third model.

    The baseline is the reference on purpose: the question is not "is this
    answer good" — an unanswerable question for a 1.5B model on a laptop — but
    "did compressing the prompt change the answer for the worse", which is
    exactly what a pairwise comparison against the uncompressed run measures.
    """
    note = progress or (lambda _: None)
    sample = corpus.subset(n_sample)
    questions = [c.user_turns[0] for c in sample.conversations if c.user_turns]

    note(f"baseline answers ({len(questions)})")
    base_pipe = Pipeline(baseline_cfg, provider=provider)
    base_answers = [base_pipe.run(q).response for q in questions]

    note("calibrating the judge on identical answers")
    calibration = calibrate_judge(judge, questions, base_answers)

    out: list[JudgeArm] = []
    for label, cfg in arms:
        note(f"arm: {label}")
        pipe = Pipeline(cfg, provider=provider)
        arm = JudgeArm(label)
        for q, ref in zip(questions, base_answers):
            verdict = judge_pairwise(q, pipe.run(q).response, ref, judge)
            arm.scores.append(verdict.score)
            if not verdict.readable:
                arm.unreadable += 1
            elif not verdict.swap_agreed:
                arm.swap_disagreements += 1
        out.append(arm)

    return JudgeStudy(
        judge_model=getattr(judge, "model_id", type(judge).__name__),
        subject_model=getattr(provider, "model_name", "unknown"),
        calibration=calibration,
        arms=out,
    )
