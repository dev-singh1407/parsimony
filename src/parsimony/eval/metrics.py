"""Four quality measures, reported as a vector and NEVER averaged.

Averaging a proxy with a ground truth manufactures confidence that neither
justifies. Three of these are proxies against the baseline's own answer; only
`exact_match` compares against something a human wrote.

    embedding_similarity  proxy   cosine against the baseline response
    token_overlap         proxy   ROUGE-L F1 against the baseline response
    judge                 proxy   pairwise, position-swapped, local model
    exact_match           GROUND TRUTH, on the 40-item gold subset

A KNOWN BIAS, STATED UP FRONT
-----------------------------
`token_overlap` is structurally biased against M5. The output budgeter's job is
to produce a shorter answer, and ROUGE-L penalises exactly that. M5's row on
this metric must be read with the bias in mind rather than treated as a quality
regression. Dropping the metric instead would be less honest: it is a real
signal for the other modules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from parsimony.eval.corpus import GoldItem
from parsimony.infra.embedding import text_similarity

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_NUM_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


@dataclass(frozen=True, slots=True)
class QualityVector:
    """Four measures. There is deliberately no `.overall` property."""

    embedding_similarity: float | None = None
    token_overlap: float | None = None
    judge: float | None = None
    judge_swap_agreed: bool | None = None
    exact_match: bool | None = None

    def as_dict(self) -> dict[str, float | bool | None]:
        return {
            "q_embedding_sim": self.embedding_similarity,
            "q_token_overlap": self.token_overlap,
            "q_judge": self.judge,
            "q_judge_swap_agreed": self.judge_swap_agreed,
            "q_exact_match": self.exact_match,
        }


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1]


def token_overlap(candidate: str, reference: str) -> float:
    """ROUGE-L F1. Biased against legitimate concision — see the module docstring."""
    c, r = _tokens(candidate), _tokens(reference)
    if not c or not r:
        return 0.0
    lcs = _lcs_length(c, r)
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(c), lcs / len(r)
    return 2 * precision * recall / (precision + recall)


def embedding_similarity(candidate: str, reference: str, embedder) -> float:
    """Proxy quality measure. The implementation lives at L1 so that L2 modules
    can use it without importing the evaluation layer."""
    return text_similarity(candidate, reference, embedder)


# --------------------------------------------------------------------------
# Gold grading. Rules are declared per item IN ADVANCE (ADR-015): deciding what
# counts as correct after seeing the output is how a gold subset stops being
# ground truth.
# --------------------------------------------------------------------------


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s.%-]", "", text.lower())).strip()


def _numbers_in(text: str) -> list[float]:
    out = []
    for m in _NUM_RE.finditer(text):
        try:
            out.append(float(m.group(0).replace(",", "")))
        except ValueError:
            continue
    return out


def grade(response: str, item: GoldItem) -> bool:
    """Did the response answer the gold question correctly, under its own rule?"""
    if not response.strip():
        return False
    resp_norm = _normalise(response)

    if item.match == "numeric":
        targets = _numbers_in(item.gold_answer)
        if not targets:
            return False
        target = targets[0]
        tol = item.tolerance
        return any(
            abs(v - target) <= tol if tol > 0 else v == target for v in _numbers_in(response)
        )

    if item.match == "exact":
        candidates = [item.gold_answer, *item.acceptable_variants]
        return any(resp_norm == _normalise(c) for c in candidates)

    if item.match in ("contains", "set"):
        candidates = [item.gold_answer, *item.acceptable_variants]
        return any(_normalise(c) in resp_norm for c in candidates if c.strip())

    raise ValueError(f"unknown match rule: {item.match!r}")


# --------------------------------------------------------------------------
# Model-as-judge
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class JudgeVerdict:
    prefers_candidate: bool
    swap_agreed: bool

    @property
    def score(self) -> float:
        """1.0 candidate wins, 0.5 tie (or judge disagreed with itself), 0.0 loses."""
        if not self.swap_agreed:
            return 0.5
        return 1.0 if self.prefers_candidate else 0.0


JUDGE_PROMPT = """Which answer is better for the question? Reply with exactly A or B.

Question: {question}

A: {a}

B: {b}

Better answer (A or B):"""


def judge_pairwise(question: str, candidate: str, reference: str, judge) -> JudgeVerdict:
    """Pairwise with a position swap.

    Three constraints, all of which the report's risk register implies and none
    of which are optional:

      * the judge must not be a model under test (self-preference is not quality);
      * pairwise, not absolute — small models cannot produce calibrated 1-10 scores;
      * every comparison is run A/B and B/A, because LLM judges have a documented
        position bias. Disagreement counts as a tie, and the swap-disagreement
        RATE is reported: a high rate means the judge is noise, and knowing that
        is worth more than the score.
    """
    first = judge.compare(JUDGE_PROMPT.format(question=question, a=candidate, b=reference))
    second = judge.compare(JUDGE_PROMPT.format(question=question, a=reference, b=candidate))
    # first says "A" -> candidate wins; second says "B" -> candidate wins
    prefers_first = first.strip().upper().startswith("A")
    prefers_second = second.strip().upper().startswith("B")
    return JudgeVerdict(
        prefers_candidate=prefers_first,
        swap_agreed=(prefers_first == prefers_second),
    )


class LengthBiasedMockJudge:
    """Stand-in judge until a real model is wired in (Sprint: Ollama).

    Deliberately NOT neutral: it prefers the longer answer. That makes it a
    useful test double, because a length-preferring judge is the classic failure
    mode of LLM-as-judge, and it lets the harness demonstrate that the
    swap-disagreement machinery actually detects bias.
    """

    def compare(self, prompt: str) -> str:
        parts = prompt.split("\n\nA: ", 1)
        if len(parts) < 2:
            return "A"
        rest = parts[1].split("\n\nB: ", 1)
        if len(rest) < 2:
            return "A"
        a_text = rest[0]
        b_text = rest[1].split("\n\nBetter answer", 1)[0]
        return "A" if len(a_text) >= len(b_text) else "B"


def score_response(
    question: str,
    candidate: str,
    reference: str,
    embedder=None,
    judge=None,
    gold: GoldItem | None = None,
) -> QualityVector:
    verdict = judge_pairwise(question, candidate, reference, judge) if judge else None
    return QualityVector(
        embedding_similarity=(
            embedding_similarity(candidate, reference, embedder) if embedder else None
        ),
        token_overlap=token_overlap(candidate, reference),
        judge=verdict.score if verdict else None,
        judge_swap_agreed=verdict.swap_agreed if verdict else None,
        exact_match=grade(candidate, gold) if gold else None,
    )
