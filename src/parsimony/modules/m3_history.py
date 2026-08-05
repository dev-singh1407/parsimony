"""M3 — Relevance-Aware History Manager.

Registered as TWO stages, deliberately:

    m3_history   -- WHICH turns survive   (selection)
    m3_arrange   -- WHERE they are placed (arrangement)

The report requires the effect of placement to be measured separately from the
effect of selection. That is only possible if they are separate stages with
separate ablation flags and separate ledger rows. Collapsing them into one
trim() makes the claim unmeasurable, and it is the kind of thing that is very
cheap to get right now and very expensive to retrofit in week 10.

Both emit TransformKind.SELECT, so the fidelity gate checks that retained turns
are byte-identical rather than demanding every invariant survive — dropping a
turn's numbers is the entire point of dropping the turn (ADR-003). Rolling
summarisation is the exception: it rewrites the turns it condenses, so it
declares REWRITE and gets the full invariant check.
"""

from __future__ import annotations

from dataclasses import replace as _replace
from typing import Protocol

import numpy as np

from parsimony.core.config import ParsimonyConfig
from parsimony.core.proposals import ContextPatch, NoOp, Proposal, TransformKind
from parsimony.core.types import RequestContext, Turn
from parsimony.infra.embedding import mmr_select


class HistoryStrategy(Protocol):
    name: str

    def select(
        self,
        turns: tuple[Turn, ...],
        query_vec: np.ndarray | None,
        turn_vecs: np.ndarray | None,
        cfg: ParsimonyConfig,
        token_count,
    ) -> list[int]:
        """Return indices of turns to retain, in any order."""
        ...


def _fit_budget(
    turns: tuple[Turn, ...], ranked: list[int], cfg: ParsimonyConfig, token_count
) -> list[int]:
    """Take from `ranked` until the turn or token budget is exhausted."""
    kept: list[int] = []
    used = 0
    for i in ranked:
        if len(kept) >= cfg.history.max_turns:
            break
        cost = turns[i].token_count or token_count(turns[i].content)
        if kept and used + cost > cfg.history.token_budget:
            continue
        kept.append(i)
        used += cost
    return kept


class RecencyStrategy:
    """The industry default, and the control arm.

    An honest report has to show whether any of the clever strategies actually
    beats "keep the last few turns" on short conversations. There is a real
    chance one does not.
    """

    name = "recency"

    def select(self, turns, query_vec, turn_vecs, cfg, token_count) -> list[int]:
        return sorted(_fit_budget(turns, list(reversed(range(len(turns)))), cfg, token_count))


class RelevanceStrategy:
    name = "relevance"

    def select(self, turns, query_vec, turn_vecs, cfg, token_count) -> list[int]:
        if query_vec is None or turn_vecs is None or len(turns) == 0:
            return RecencyStrategy().select(turns, query_vec, turn_vecs, cfg, token_count)
        scores = turn_vecs @ query_vec
        ranked = list(np.argsort(-scores))
        return sorted(_fit_budget(turns, [int(i) for i in ranked], cfg, token_count))


class MmrStrategy:
    """Relevance minus redundancy. Shares one MMR implementation with M1 tier 2."""

    name = "mmr"

    def select(self, turns, query_vec, turn_vecs, cfg, token_count) -> list[int]:
        if query_vec is None or turn_vecs is None or len(turns) == 0:
            return RecencyStrategy().select(turns, query_vec, turn_vecs, cfg, token_count)
        ranked = mmr_select(turn_vecs, query_vec, k=len(turns), lambda_=cfg.history.mmr_lambda)
        return sorted(_fit_budget(turns, ranked, cfg, token_count))


class SummaryStrategy:
    """Rolling summarisation, EXTRACTIVE.

    Recent turns are kept verbatim; older ones are condensed to their most
    query-relevant sentence. Extractive rather than abstractive on purpose: an
    abstractive summary needs a model call, and a module that spends model
    tokens to save model tokens has to justify itself against a baseline that
    spends none. The abstractive variant is a Sprint 6 comparison arm, and per
    ADR-010 it runs off the critical path.

    This is the one strategy that rewrites rather than selects, so it declares
    REWRITE and the gate demands every number, entity and negation survive the
    condensation.
    """

    name = "summary"
    keep_verbatim = 2

    def select(self, turns, query_vec, turn_vecs, cfg, token_count) -> list[int]:
        return list(range(len(turns)))  # nothing is dropped; content is condensed

    def condense(
        self, turns: tuple[Turn, ...], query_vec, cfg, derived
    ) -> tuple[tuple[Turn, ...], int]:
        if len(turns) <= self.keep_verbatim:
            return turns, 0

        head, tail = turns[: -self.keep_verbatim], turns[-self.keep_verbatim :]
        out: list[Turn] = []
        condensed = 0
        for turn in head:
            sentences = derived.sentences(turn.content)
            if len(sentences) < 2:
                out.append(turn)
                continue
            if query_vec is not None and getattr(derived, "has_embedder", False):
                vecs = derived.embed(list(sentences))
                best = int(np.argmax(vecs @ query_vec))
            else:
                best = 0
            picked = sentences[best]
            if derived.token_count(picked) >= derived.token_count(turn.content):
                out.append(turn)
                continue
            out.append(_replace(turn, content=picked, token_count=derived.token_count(picked)))
            condensed += 1
        return tuple(out) + tail, condensed


STRATEGIES: dict[str, HistoryStrategy] = {
    s.name: s for s in (RecencyStrategy(), RelevanceStrategy(), MmrStrategy(), SummaryStrategy())
}


class HistorySelector:
    module_id = "M3"
    name = "m3_history"
    reads = frozenset({"query", "history"})
    writes = frozenset({"history"})

    def applies_to(self, ctx: RequestContext, cfg: ParsimonyConfig) -> bool:
        return cfg.enables("M3") and len(ctx.history) > 0

    def propose(self, ctx: RequestContext, cfg: ParsimonyConfig) -> Proposal:
        d = ctx.derived
        if d is None:
            return NoOp("not_applicable", "no derived cache")

        strategy = STRATEGIES.get(cfg.history.strategy)
        if strategy is None:
            return NoOp("not_applicable", f"unknown strategy {cfg.history.strategy!r}")

        query_vec = turn_vecs = None
        if getattr(d, "has_embedder", False):
            # One batched call for every turn (ADR-006): twelve separate
            # embeddings cost ~5x a single batch of twelve.
            both = d.embed([ctx.query] + [t.content for t in ctx.history])
            query_vec, turn_vecs = both[0], both[1:]

        if isinstance(strategy, SummaryStrategy):
            new_history, condensed = strategy.condense(ctx.history, query_vec, cfg, d)
            if not condensed:
                return NoOp("no_yield", "nothing worth condensing")
            return ContextPatch(
                kind=TransformKind.REWRITE,  # condensation rewrites; it does not select
                fields={"history": new_history},
                rationale=f"rolling summary condensed {condensed} older turn(s)",
                evidence={"strategy": "summary", "condensed": condensed},
            )

        keep = strategy.select(ctx.history, query_vec, turn_vecs, cfg, d.token_count)
        if len(keep) == len(ctx.history):
            return NoOp("no_yield", f"all {len(ctx.history)} turns fit the budget")

        new_history = tuple(ctx.history[i] for i in keep)
        return ContextPatch(
            kind=TransformKind.SELECT,
            fields={"history": new_history},
            rationale=f"{strategy.name}: kept {len(keep)} of {len(ctx.history)} turns",
            evidence={
                "strategy": strategy.name,
                "kept": len(keep),
                "dropped": len(ctx.history) - len(keep),
            },
        )


class HistoryArranger:
    """Placement, measured separately from selection.

    Small models attend most strongly to the start and the end of their context
    (lost-in-the-middle). Position-aware placement moves the most query-relevant
    retained turn adjacent to the query. Whether that helps at 1-3B scale is an
    open question this stage exists to answer, which is why it is ablatable on
    its own rather than bundled into selection.
    """

    module_id = "M3"
    name = "m3_arrange"
    reads = frozenset({"query", "history"})
    writes = frozenset({"history"})

    def applies_to(self, ctx: RequestContext, cfg: ParsimonyConfig) -> bool:
        return (
            cfg.enables("M3")
            and cfg.history.arrangement == "position_aware"
            and len(ctx.history) > 2
        )

    def propose(self, ctx: RequestContext, cfg: ParsimonyConfig) -> Proposal:
        d = ctx.derived
        if d is None or not getattr(d, "has_embedder", False):
            return NoOp("not_applicable", "position-aware placement needs embeddings")
        # propose() must be safe on any input, not only on inputs applies_to
        # already approved. The orchestrator would catch a raise and degrade to
        # NoOp, but it would pollute the trace with an error that is really just
        # a missing guard.
        if len(ctx.history) < 2:
            return NoOp("not_applicable", "nothing to reorder")

        both = d.embed([ctx.query] + [t.content for t in ctx.history])
        scores = both[1:] @ both[0]
        best = int(np.argmax(scores))
        if best == len(ctx.history) - 1:
            return NoOp("no_yield", "most relevant turn is already adjacent to the query")

        moved = ctx.history[best]
        rest = tuple(t for i, t in enumerate(ctx.history) if i != best)
        return ContextPatch(
            kind=TransformKind.SELECT,  # reorder: retained turns stay byte-identical
            fields={"history": rest + (moved,)},
            rationale=f"moved turn {best} (relevance {float(scores[best]):.3f}) next to the query",
            evidence={
                "arrangement": "position_aware",
                "moved_from": best,
                "moved_to": len(ctx.history) - 1,
                "relevance": round(float(scores[best]), 4),
            },
        )


def stages() -> list:
    return [HistorySelector(), HistoryArranger()]
