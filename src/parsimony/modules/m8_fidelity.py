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

import re

from dataclasses import dataclass

from parsimony.core.ledger import GateEvent
from parsimony.core.proposals import TransformKind
from parsimony.core.types import Invariants, RequestContext
from parsimony.infra.nlp import RegexInvariantExtractor, split_sentences

#: Any Unicode letter or digit. Deliberately not [A-Za-z0-9]: the whole point
#: of the annihilation check is that it holds for alphabets the extractors
#: cannot read.
_WORDLIKE_RE = re.compile(r"[^\W_]")
_SPACE_RE = re.compile(r"\s+")


def _squash(text: str) -> str:
    return _SPACE_RE.sub(" ", text).strip()


def is_sentence_extract(source: str, target: str) -> bool:
    """Can `target` be produced from `source` by deleting whole sentences?

    Checked on the text itself rather than on any structure the proposing
    module reports, so a module cannot describe an edit as an extraction and
    then return something else. Whitespace between sentences is free -- a
    module may re-join kept sentences with a space or a newline -- but every
    retained sentence must be character-identical and in its original order.
    """
    rest = _squash(target)
    if not rest:
        return True
    for sentence in split_sentences(source):
        s = _squash(sentence)
        if rest == s:
            return True
        if rest.startswith(s + " "):
            rest = rest[len(s) + 1:]
    return False


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

        if kind is TransformKind.EXTRACT:
            return self._check_extract(before, after, module_id)

        return self._check_rewrite(before, after, module_id)

    def _check_rewrite(
        self, before: RequestContext, after: RequestContext, module_id: str
    ) -> Verdict:
        source = before.text_payload()
        target = after.text_payload()

        # Annihilation check, BEFORE the invariant comparison and independent of
        # it. Every other check here asks "was a value I could extract lost?",
        # which silently makes the gate's guarantee conditional on the
        # extractor's coverage. The extractors are regex-based and Latin-only,
        # so a query in Cyrillic, Devanagari, Tamil or Chinese yields no
        # invariants at all — and a rewrite that deleted the entire question
        # therefore lost nothing the gate could name, and passed. M1 tier 1 did
        # exactly that (its "contentless sentence" test was [A-Za-z0-9]) and the
        # model received an empty prompt.
        #
        # A transform that removes ALL word characters is never legitimate,
        # whatever alphabet they were written in, so this does not depend on
        # understanding the text.
        if _WORDLIKE_RE.search(source) and not _WORDLIKE_RE.search(target):
            return Verdict(
                False,
                (GateEvent(module_id=module_id, invariant_class="content",
                           lost_values=(source[:60],)),),
                "rewrite removed all content",
            )

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
        if after.documents != before.documents:
            return Verdict(
                False,
                (GateEvent(module_id, "structure", ("documents",)),),
                "select must not alter supplied documents",
            )
        return Verdict.ok()

    def _check_extract(
        self, before: RequestContext, after: RequestContext, module_id: str
    ) -> Verdict:
        """Sentence-level removal inside documents and turns (ADR-040).

        Three guarantees, each independent of what the proposing module claims:

          1. Structure. The question is untouched; no document or turn appears
             that did not exist; turns keep their roles and order; every kept
             unit is its source with whole sentences deleted, nothing reworded.
          2. Anchors. Anything the QUESTION names -- a number, a name, a quoted
             string -- that the context contained must still be in the context.
             Deleting sentences legitimately loses numbers (that is the point),
             so a REWRITE check would refuse every extraction; but deleting the
             only sentences about the thing being asked about is never a saving.
          3. Content. Context that had words cannot be reduced to none.
        """
        def fail(cls: str, values: tuple[str, ...], detail: str) -> Verdict:
            return Verdict(False, (GateEvent(module_id, cls, values),), detail)

        if after.query != before.query:
            return fail("structure", ("query",), "extract must not alter the query")

        source_docs = {d.doc_id: d for d in before.documents}
        for doc in after.documents:
            src = source_docs.get(doc.doc_id)
            if src is None:
                return fail("structure", (doc.doc_id,), "extract introduced a document")
            if doc.title != src.title or not is_sentence_extract(src.content, doc.content):
                return fail("structure", (doc.doc_id,), "extract altered a sentence")

        if [t.turn_id for t in after.history] != [t.turn_id for t in before.history]:
            return fail("structure", ("history",), "extract must not add, drop or reorder turns")
        for old, new in zip(before.history, after.history):
            if new.role != old.role or not is_sentence_extract(old.content, new.content):
                return fail("structure", (new.turn_id,), "extract altered a sentence")

        def context_of(ctx: RequestContext) -> str:
            return "\n".join([d.content for d in ctx.documents]
                             + [t.content for t in ctx.history])

        source, target = context_of(before), context_of(after)
        if _WORDLIKE_RE.search(source) and not _WORDLIKE_RE.search(target):
            return fail("content", (source[:60],), "extract removed all context")

        asked = self.invariants_of(before.query)
        anchors = Invariants(numbers=asked.numbers, entities=asked.entities,
                             quoted=asked.quoted)
        had = anchors.missing_from(source)          # named in the question, never in context
        lost = anchors.missing_from(target)
        dropped = {cls: vals - had.get(cls, frozenset()) for cls, vals in lost.items()}
        dropped = {cls: vals for cls, vals in dropped.items() if vals}
        if dropped:
            events = tuple(GateEvent(module_id, cls.value, tuple(sorted(vals)))
                           for cls, vals in dropped.items())
            summary = ", ".join(f"{c.value}:{len(v)}" for c, v in dropped.items())
            return Verdict(False, events, f"extract dropped every mention of {summary} "
                                          f"named in the question")
        return Verdict.ok()
