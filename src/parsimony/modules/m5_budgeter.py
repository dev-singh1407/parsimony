"""M5 — Adaptive Output Budgeter.

The only module acting on the side of the exchange that dominates CPU wall clock
(report figure 2: ~82% of a second goes to decode, not prefill). Budgets are per
response class, never global, precisely because a uniform length hint is
documented to cut output sharply on verbose models while regressing quality on
answers that were already terse — a global cap would produce a quality drop that
looks like a compression failure and get attributed to the wrong module.
"""

from __future__ import annotations

import re

from parsimony.core.config import ParsimonyConfig
from parsimony.core.proposals import ContextPatch, NoOp, Proposal, TransformKind
from parsimony.core.types import RequestContext, ResponseClass

_CODE_RE = re.compile(r"```|\bfunction\b|\bcode\b|\bpython\b|\bjavascript\b|\bsql\b|\bregex\b", re.I)
_WRITE_CODE_RE = re.compile(r"\bwrite (?:a |an )?(?:function|script|program|class|query)\b", re.I)
_SUMM_RE = re.compile(r"\bsummaris|\bsummariz|\bcondense\b|\btl;?dr\b|\bin (?:one|two|three) (?:line|sentence)", re.I)
_REASON_RE = re.compile(r"\bwhy\b|\bexplain why\b|\bcompare\b|\bdifference between\b|\bprove\b|\bderive\b", re.I)
_ARITH_RE = re.compile(r"\d\s*[-+*/^%]\s*\d|\bhow many\b|\bconvert\b|\bcalculate\b|\bpercent\b", re.I)
# A follow-up is signalled either by an anaphoric reference OR by a bare
# conjunction opener with no pronoun at all ("And at 3000 metres?"). Requiring a
# pronoun misses the second form entirely, which is the more common one in real
# chat.
_ANAPHORA_RE = re.compile(
    r"\b(?:it|its|that|this|those|these|they|them|the second|the first|"
    r"the former|the latter|the same)\b",
    re.I,
)
_FOLLOWUP_LEAD_RE = re.compile(
    r"^\s*(?:and|but|so|also|then|what about|how about|ok(?:ay)?)\b", re.I
)


def classify(query: str, has_history: bool) -> ResponseClass:
    """Rule-based classifier.

    ADR-012 recommends a logistic head over the shared query embedding, which is
    effectively free once the embedding exists for the cache lookup. Until
    embeddings land in Sprint 2 these rules stand in, and they remain afterwards
    as a high-precision override: routing a non-arithmetic query to the
    deterministic tier produces a confidently wrong answer, so that path must be
    precise rather than merely accurate.
    """
    if _WRITE_CODE_RE.search(query) or "```" in query:
        return ResponseClass.CODE
    if _SUMM_RE.search(query):
        return ResponseClass.SUMMARISATION
    # REASONING outranks FOLLOW_UP deliberately. The budget must reflect how long
    # the answer needs to be, not where the question sits in the discourse:
    # "Why does it change?" is a follow-up but still needs a reasoning-length
    # answer, whereas "And at 3000 metres?" does not.
    if _REASON_RE.search(query):
        return ResponseClass.REASONING
    if (
        has_history
        and len(query.split()) <= 12
        and (_FOLLOWUP_LEAD_RE.match(query) or _ANAPHORA_RE.search(query))
    ):
        return ResponseClass.FOLLOW_UP
    if _ARITH_RE.search(query):
        return ResponseClass.ARITHMETIC
    if _CODE_RE.search(query):
        return ResponseClass.CODE
    return ResponseClass.FACTUAL


class TrigramNoveltyStopper:
    """Streaming early-stop rule.

    Small models restate. When the fraction of *unseen* trigrams over a sliding
    window falls below a threshold, the model has stopped adding information.
    O(1) per token with a rolling set — a per-token embedding check would cost
    more than the generation it saves.
    """

    def __init__(self, window: int = 48, threshold: float = 0.25, min_tokens: int = 12) -> None:
        self.window = window
        self.threshold = threshold
        self.min_tokens = min_tokens
        self._seen: set[tuple[str, ...]] = set()
        self._recent: list[bool] = []
        self._tokens: list[str] = []
        self._buffer: list[str] = []
        self._seen_sentences: set[str] = set()
        self.stopped_at: int | None = None
        self.reason: str | None = None

    def observe(self, piece: str) -> bool:
        """Feed one token. Returns True when generation should stop."""
        self._tokens.append(piece.strip().lower())
        self._buffer.append(piece)

        # Rule A: a completed sentence that was already produced verbatim.
        # This is the literal behaviour the report describes ("the model begins
        # restating material it has already produced") and it fires long before
        # a windowed statistic can, because one restated sentence is already
        # unambiguous.
        if any(p in piece for p in ".!?"):
            sentence = _normalise_sentence("".join(self._buffer))
            self._buffer.clear()
            if len(sentence.split()) >= 4:
                if sentence in self._seen_sentences:
                    self.stopped_at = len(self._tokens)
                    self.reason = "restated a sentence"
                    return True
                self._seen_sentences.add(sentence)

        # Rule B: trigram novelty collapse — catches drift into near-repetition
        # that is not an exact restatement.
        if len(self._tokens) < 3:
            return False
        tri = tuple(self._tokens[-3:])
        novel = tri not in self._seen
        self._seen.add(tri)
        self._recent.append(novel)
        if len(self._recent) > self.window:
            self._recent.pop(0)

        if len(self._tokens) < self.min_tokens or len(self._recent) < min(self.window, 16):
            return False
        if (sum(self._recent) / len(self._recent)) < self.threshold:
            self.stopped_at = len(self._tokens)
            self.reason = "trigram novelty collapse"
            return True
        return False


def _normalise_sentence(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


class OutputBudgeter:
    module_id = "M5"
    name = "m5_budgeter"
    reads = frozenset({"query", "history"})
    writes = frozenset({"response_class", "output_budget"})

    def applies_to(self, ctx: RequestContext, cfg: ParsimonyConfig) -> bool:
        return cfg.enables("M5")

    def propose(self, ctx: RequestContext, cfg: ParsimonyConfig) -> Proposal:
        cls = classify(ctx.query, bool(ctx.history))
        budget = cfg.budget.per_class.get(cls.value)
        if budget is None:
            return NoOp("not_applicable", f"no budget configured for {cls.value}")
        return ContextPatch(
            kind=TransformKind.DECIDE,
            fields={"response_class": cls, "output_budget": budget},
            rationale=f"class={cls.value}, num_predict={budget}",
            evidence={"response_class": cls.value, "budget": budget},
        )

    @staticmethod
    def stopper(cfg: ParsimonyConfig) -> TrigramNoveltyStopper | None:
        if not (cfg.enables("M5") and cfg.budget.early_stop):
            return None
        return TrigramNoveltyStopper(
            window=cfg.budget.novelty_window,
            threshold=cfg.budget.novelty_threshold,
        )
