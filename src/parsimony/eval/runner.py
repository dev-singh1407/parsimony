"""Sweep runner.

Token reduction is a BETWEEN-cell comparison, not a within-request one: a cell's
total is measured against the baseline cell's total over the same corpus. That
is the only definition that stays honest once modules start changing output
length as well as input length.

Each cell gets a fresh cache. Sharing one across cells would let cell order
determine hit rate, which would silently invalidate every M2 result.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from parsimony.core.config import ParsimonyConfig
from parsimony.core.types import RouteTier, Turn
from parsimony.eval.corpus import Conversation, Corpus
from parsimony.infra.ids import ulid
from parsimony.modules.m2_cache import SemanticCache
from parsimony.pipeline.orchestrator import Outcome, Pipeline
from parsimony.pipeline.registry import default_registry


@dataclass(slots=True)
class CellResult:
    label: str
    config_hash: str
    n_requests: int = 0
    tokens_in_baseline: int = 0
    tokens_in_final: int = 0
    tokens_out: int = 0
    cache_hits: int = 0
    deterministic_hits: int = 0
    gate_fires: int = 0
    early_stops: int = 0
    middleware_ms: list[float] = field(default_factory=list)

    # filled by summarise(), relative to the baseline cell
    total_reduction_pct: float = 0.0
    input_reduction_pct: float = 0.0
    output_reduction_pct: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.tokens_in_final + self.tokens_out

    @property
    def middleware_mean_ms(self) -> float:
        return statistics.fmean(self.middleware_ms) if self.middleware_ms else 0.0

    @property
    def middleware_p95_ms(self) -> float:
        if not self.middleware_ms:
            return 0.0
        ordered = sorted(self.middleware_ms)
        return ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]

    def observe(self, outcome: Outcome) -> None:
        row = outcome.row
        self.n_requests += 1
        self.tokens_in_baseline += row.tokens_in_original
        self.tokens_in_final += row.tokens_in_final
        self.tokens_out += row.tokens_out
        if row.cache_hit:
            self.cache_hits += 1
        if row.route_tier == RouteTier.DETERMINISTIC.name:
            self.deterministic_hits += 1
        if row.gate_fired:
            self.gate_fires += 1
        if row.early_stopped:
            self.early_stops += 1
        self.middleware_ms.append(row.middleware_ns / 1e6)


def run_conversation(pipeline: Pipeline, conv: Conversation) -> list[Outcome]:
    """Replay one conversation, feeding each generated answer back as history."""
    history: list[Turn] = []
    outcomes: list[Outcome] = []
    for i, user_text in enumerate(conv.user_turns):
        outcome = pipeline.run(
            user_text,
            tuple(history),
            conversation_id=conv.conversation_id,
            turn_index=i,
        )
        outcomes.append(outcome)
        history.append(Turn(turn_id=f"{conv.conversation_id}:u{i}", role="user", content=user_text))
        history.append(
            Turn(turn_id=f"{conv.conversation_id}:a{i}", role="assistant", content=outcome.response)
        )
    return outcomes


def run_cell(
    cfg: ParsimonyConfig,
    corpus: Corpus,
    *,
    provider=None,
    tokenizer=None,
    sink=None,
    run_id: str | None = None,
    pass_kind: str = "quality",
) -> CellResult:
    cache = SemanticCache(cfg.cache.ttl_seconds)
    pipeline = Pipeline(
        cfg,
        provider=provider,
        tokenizer=tokenizer,
        cache=cache,
        registry=default_registry(cache),
        sink=sink,
        run_id=run_id or ulid(),
        pass_kind=pass_kind,
        corpus_hash=corpus.corpus_hash,
    )
    result = CellResult(label=cfg.label or "unlabelled", config_hash=cfg.config_hash)
    for conv in corpus.conversations:
        for outcome in run_conversation(pipeline, conv):
            result.observe(outcome)
    return result


def sweep(
    cells: Iterable[ParsimonyConfig],
    corpus: Corpus,
    *,
    provider=None,
    tokenizer=None,
    sink=None,
    run_id: str | None = None,
    progress=None,
) -> list[CellResult]:
    run_id = run_id or ulid()
    results: list[CellResult] = []
    for cfg in cells:
        if progress is not None:
            progress(cfg.label or cfg.config_hash)
        results.append(
            run_cell(cfg, corpus, provider=provider, tokenizer=tokenizer,
                     sink=sink, run_id=run_id)
        )
    return summarise(results)


def summarise(results: Sequence[CellResult]) -> list[CellResult]:
    """Attach reductions relative to the baseline cell."""
    baseline = next((r for r in results if r.label == "baseline"), None)
    if baseline is None:
        return list(results)
    base_total = baseline.total_tokens or 1
    base_in = baseline.tokens_in_final or 1
    base_out = baseline.tokens_out or 1
    for r in results:
        r.total_reduction_pct = 100.0 * (1 - r.total_tokens / base_total)
        r.input_reduction_pct = 100.0 * (1 - r.tokens_in_final / base_in)
        r.output_reduction_pct = 100.0 * (1 - r.tokens_out / base_out)
    return list(results)


def additivity_shortfall(
    results: Sequence[CellResult], axes: Sequence[str] | None = None
) -> dict[str, float]:
    """The primary result of the whole project (Contribution 1, report figure 1).

    If savings were additive, the full stack would deliver the sum of each
    module's solo reduction. The gap between that prediction and what stacking
    actually delivers is the number no published study reports.

    Restricted to the factorial axes. A cell carrying a module that has no solo
    measurement (M6, which is studied on top of the winner rather than as an
    axis) must not appear on either side of the comparison — including it once
    produced a *negative* shortfall, which is arithmetic, not a finding.
    """
    solo = {
        r.label: r.total_reduction_pct
        for r in results
        if r.label != "baseline" and "+" not in r.label
    }
    if axes is None:
        axes = sorted(solo)
    axis_set = set(axes)
    solo = {k: v for k, v in solo.items() if k in axis_set}

    candidates = [
        r for r in results if r.label != "baseline" and set(r.label.split("+")) <= axis_set
    ]
    if not candidates or len(solo) < 2:
        return {"n_solo_modules": len(solo)}

    full = max(candidates, key=lambda r: len(r.label.split("+")))
    predicted = sum(solo.values())
    return {
        "predicted_additive_pct": predicted,
        "measured_stacked_pct": full.total_reduction_pct,
        "shortfall_pct": predicted - full.total_reduction_pct,
        "stacked_label": full.label,
        "n_solo_modules": len(solo),
    }
