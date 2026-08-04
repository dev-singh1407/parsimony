"""M8 — Fidelity Gate. Always on, never ablated.

Invoked by the orchestrator on every proposal, never by modules: a module
physically cannot bypass it, forget it, or implement it inconsistently.

The key subtlety is what a REWRITE is checked *against*. Not the original
request: if M3 has already legitimately dropped a turn, the numbers in that turn
are gone by design, and checking a later rewrite against the original would
revert it for a loss it did not cause. A rewrite must preserve what the text had
*at the moment it ran*. Extraction is memoised on payload text, so the committed
state of stage N is reused as the input state of stage N+1 — roughly one
extraction per committed change rather than one per check.
"""

from __future__ import annotations

from dataclasses import dataclass

from parsimony.core.ledger import GateEvent
from parsimony.core.proposals import TransformKind
from parsimony.core.types import Invariants, RequestContext
from parsimony.infra.nlp import RegexInvariantExtractor


@dataclass(frozen=True, slots=True)
class Verdict:
    passed: bool
    events: tuple[GateEvent, ...] = ()
    detail: str = ""

    @staticmethod
    def ok() -> "Verdict":
        return Verdict(True)


class FidelityGate:
    def __init__(self, extractor=None, memo_limit: int = 4096) -> None:
        self._extractor = extractor or RegexInvariantExtractor()
        self._memo: dict[str, Invariants] = {}
        self._memo_limit = memo_limit
        self.checks = 0
        self.extractions = 0

    def invariants_of(self, text: str) -> Invariants:
        hit = self._memo.get(text)
        if hit is not None:
            return hit
        self.extractions += 1
        inv = self._extractor.extract(text)
        if len(self._memo) < self._memo_limit:
            self._memo[text] = inv
        return inv

    def check(
        self,
        before: RequestContext,
        after: RequestContext,
        kind: TransformKind,
        module_id: str,
    ) -> Verdict:
        self.checks += 1

        if kind in (TransformKind.AUGMENT, TransformKind.DECIDE):
            return Verdict.ok()

        if kind is TransformKind.SELECT:
            return self._check_select(before, after, module_id)

        return self._check_rewrite(before, after, module_id)

    def _check_rewrite(
        self, before: RequestContext, after: RequestContext, module_id: str
    ) -> Verdict:
        source = before.text_payload()
        target = after.text_payload()
        lost = self.invariants_of(source).missing_from(target)
        if not lost:
            return Verdict.ok()
        events = tuple(
            GateEvent(module_id=module_id, invariant_class=cls.value, lost_values=tuple(sorted(vals)))
            for cls, vals in lost.items()
        )
        summary = ", ".join(f"{c.value}:{len(v)}" for c, v in lost.items())
        return Verdict(False, events, f"rewrite dropped {summary}")

    def _check_select(
        self, before: RequestContext, after: RequestContext, module_id: str
    ) -> Verdict:
        """Removal of whole units is legitimate; mutating a retained unit is not."""
        originals = {t.turn_id: t.content for t in before.history}
        for turn in after.history:
            if turn.turn_id not in originals:
                return Verdict(
                    False,
                    (GateEvent(module_id, "structure", (turn.turn_id,)),),
                    "select introduced a turn that did not exist",
                )
            if originals[turn.turn_id] != turn.content:
                return Verdict(
                    False,
                    (GateEvent(module_id, "structure", (turn.turn_id,)),),
                    "select mutated a retained turn",
                )
        if after.query != before.query:
            return Verdict(
                False,
                (GateEvent(module_id, "structure", ("query",)),),
                "select must not alter the query",
            )
        return Verdict.ok()
