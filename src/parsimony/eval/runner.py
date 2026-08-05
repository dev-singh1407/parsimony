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
from parsimony.eval.corpus import Conversation, Corpus, GoldItem
from parsimony.eval.metrics import grade, score_response
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
    prefix_tokens: list[int] = field(default_factory=list)

    # Four quality measures, kept SEPARATE. There is deliberately no combined
    # score: averaging a proxy with a ground truth manufactures confidence.
    q_embedding: list[float] = field(default_factory=list)
    q_overlap: list[float] = field(default_factory=list)
    q_judge: list[float] = field(default_factory=list)
    q_judge_disagreements: int = 0
    gold_correct: int = 0
    gold_total: int = 0

    responses: dict[tuple[str, int], str] = field(default_factory=dict)

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

    @property
    def mean_prefix_tokens(self) -> float:
        return statistics.fmean(self.prefix_tokens) if self.prefix_tokens else 0.0

    @property
    def quality_embedding(self) -> float:
        return 100.0 * statistics.fmean(self.q_embedding) if self.q_embedding else 100.0

    @property
    def quality_overlap(self) -> float:
        return 100.0 * statistics.fmean(self.q_overlap) if self.q_overlap else 100.0

    @property
    def quality_judge(self) -> float:
        return 100.0 * statistics.fmean(self.q_judge) if self.q_judge else 100.0

    @property
    def judge_disagreement_rate(self) -> float:
        """High means the judge is noise and its score should be discounted."""
        n = len(self.q_judge)
        return 100.0 * self.q_judge_disagreements / n if n else 0.0

    @property
    def gold_accuracy(self) -> float:
        return 100.0 * self.gold_correct / self.gold_total if self.gold_total else 0.0

    def observe(self, outcome: Outcome) -> None:
        row = outcome.row
        self.responses[(row.conversation_id, row.turn_index)] = outcome.response
        if row.prefix_tokens_survived is not None:
            self.prefix_tokens.append(row.prefix_tokens_survived)
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
    embedder=None,
    sink=None,
    run_id: str | None = None,
    pass_kind: str = "quality",
    reference: "CellResult | None" = None,
    judge=None,
    gold: tuple[GoldItem, ...] = (),
) -> CellResult:
    cache = SemanticCache(cfg.cache.ttl_seconds)
    pipeline = Pipeline(
        cfg,
        provider=provider,
        tokenizer=tokenizer,
        embedder=embedder,
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

    if reference is not None:
        _score_against(result, reference, corpus, pipeline.embedder, judge)
    if gold:
        _score_gold(result, pipeline, gold)
    return result


def _score_against(result, reference, corpus, embedder, judge) -> None:
    """Compare this cell's answers against the baseline cell's, per request."""
    questions = {
        (c.conversation_id, i): q
        for c in corpus.conversations
        for i, q in enumerate(c.user_turns)
    }
    for key, response in result.responses.items():
        ref = reference.responses.get(key)
        if ref is None:
            continue
        vec = score_response(
            questions.get(key, ""), response, ref, embedder=embedder, judge=judge
        )
        if vec.embedding_similarity is not None:
            result.q_embedding.append(vec.embedding_similarity)
        if vec.token_overlap is not None:
            result.q_overlap.append(vec.token_overlap)
        if vec.judge is not None:
            result.q_judge.append(vec.judge)
            if vec.judge_swap_agreed is False:
                result.q_judge_disagreements += 1


def _score_gold(result, pipeline, gold: tuple[GoldItem, ...]) -> None:
    """The only non-proxy measure. Fresh conversation ids so the cache cannot
    serve a gold answer from the main corpus run."""
    for item in gold:
        outcome = pipeline.run(item.question, conversation_id=f"gold:{item.gold_id}")
        result.gold_total += 1
        result.gold_correct += int(grade(outcome.response, item))


def sweep(
    cells: Iterable[ParsimonyConfig],
    corpus: Corpus,
    *,
    provider=None,
    tokenizer=None,
    embedder=None,
    sink=None,
    run_id: str | None = None,
    progress=None,
    judge=None,
    gold: tuple[GoldItem, ...] = (),
) -> list[CellResult]:
    """Run every cell, baseline first.

    Baseline must run first because the three proxy quality measures are
    computed against its answers. Ordering the work by dependency here means no
    caller has to know about it.
    """
    run_id = run_id or ulid()
    cells = sorted(cells, key=lambda c: c.label != "baseline")
    results: list[CellResult] = []
    reference: CellResult | None = None

    for cfg in cells:
        if progress is not None:
            progress(cfg.label or cfg.config_hash)
        result = run_cell(
            cfg, corpus,
            provider=provider, tokenizer=tokenizer, embedder=embedder,
            sink=sink, run_id=run_id,
            reference=reference, judge=judge, gold=gold,
        )
        if reference is None and cfg.label == "baseline":
            reference = result
        results.append(result)
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
