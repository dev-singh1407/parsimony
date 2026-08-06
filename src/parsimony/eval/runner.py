"""Sweep runner.

Token reduction is a BETWEEN-cell comparison, not a within-request one: a cell's
total is measured against the baseline cell's total over the same corpus. That
is the only definition that stays honest once modules start changing output
length as well as input length.

Each cell gets a fresh cache. Sharing one across cells would let cell order
determine hit rate, which would silently invalidate every M2 result.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, field, replace
from pathlib import Path
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

    joules: list[float] = field(default_factory=list)
    usd: float = 0.0
    # Per query class, so the deliverable can be a calibration table -- which
    # modules to switch on for which kind of question -- rather than one
    # aggregate percentage (Contribution 6).
    per_class_tokens: dict[str, int] = field(default_factory=dict)
    per_class_requests: dict[str, int] = field(default_factory=dict)

    responses: dict[tuple[str, int], str] = field(default_factory=dict)
    # Per-conversation totals: the resampling unit for the bootstrap. Requests
    # within a conversation are not independent (later turns carry earlier
    # answers as history), so resampling requests would understate the interval.
    per_conversation: dict[str, int] = field(default_factory=dict)

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

    @property
    def total_joules(self) -> float:
        return sum(self.joules)

    @property
    def tokens_per_joule(self) -> float:
        return self.total_tokens / self.total_joules if self.total_joules else 0.0

    def observe(self, outcome: Outcome, cls: str = "unknown") -> None:
        row = outcome.row
        self.per_class_tokens[cls] = (
            self.per_class_tokens.get(cls, 0) + row.tokens_in_final + row.tokens_out
        )
        self.per_class_requests[cls] = self.per_class_requests.get(cls, 0) + 1
        if row.joules_estimated is not None:
            self.joules.append(row.joules_estimated)
        if row.usd_equivalent is not None:
            self.usd += row.usd_equivalent
        self.responses[(row.conversation_id, row.turn_index)] = outcome.response
        self.per_conversation[row.conversation_id] = (
            self.per_conversation.get(row.conversation_id, 0)
            + row.tokens_in_final
            + row.tokens_out
        )
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
    memo=None,
    sink=None,
    run_id: str | None = None,
    pass_kind: str = "quality",
    reference: "CellResult | None" = None,
    judge=None,
    gold: tuple[GoldItem, ...] = (),
    warm_start=None,
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
        memo=memo,
        run_id=run_id or ulid(),
        pass_kind=pass_kind,
        corpus_hash=corpus.corpus_hash,
    )
    if warm_start is not None:
        from parsimony.pipeline.warm_start import warm_start as seed_cache

        seed_cache(pipeline, warm_start)

    result = CellResult(label=cfg.label or "unlabelled", config_hash=cfg.config_hash)
    for conv in corpus.conversations:
        for outcome in run_conversation(pipeline, conv):
            result.observe(outcome, conv.cls)

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
    memo=None,
    sink=None,
    run_id: str | None = None,
    progress=None,
    judge=None,
    gold: tuple[GoldItem, ...] = (),
    warm_start=None,
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
    pass_kind = "timing" if memo is None else "quality"

    for cfg in cells:
        if progress is not None:
            progress(cfg.label or cfg.config_hash)
        result = run_cell(
            cfg, corpus,
            provider=provider, tokenizer=tokenizer, embedder=embedder, memo=memo,
            sink=sink, run_id=run_id, pass_kind=pass_kind,
            reference=reference, judge=judge, gold=gold, warm_start=warm_start,
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


@dataclass(slots=True)
class SweepReport:
    """Both passes of a sweep, kept separate so neither contaminates the other."""

    quality: list[CellResult] = field(default_factory=list)
    timing: list[CellResult] = field(default_factory=list)
    quality_corpus_size: int = 0
    timing_corpus_size: int = 0
    timing_repeats: int = 1
    memo_hits: int = 0
    memo_total: int = 0
    resumed: int = 0

    @property
    def memo_hit_rate(self) -> float:
        return 100.0 * self.memo_hits / self.memo_total if self.memo_total else 0.0


def two_pass_sweep(
    cells: Sequence[ParsimonyConfig],
    corpus: Corpus,
    *,
    timing_subset: int = 50,
    timing_repeats: int = 2,
    resume_log: "Path | None" = None,
    sink=None,
    run_id: str | None = None,
    progress=None,
    judge=None,
    gold: tuple[GoldItem, ...] = (),
    provider=None,
    tokenizer=None,
    embedder=None,
) -> SweepReport:
    """The sweep as docs/05-evaluation-harness.md specifies it.

    QUALITY PASS  memo ON, full corpus, 1 repeat.
    TIMING PASS   memo OFF, stratified subset, N repeats.

    Two passes because generation memoisation is bit-exact at temperature 0 but
    takes microseconds, so a memoised run says nothing about latency. Token
    counts, quality scores and every behavioural metric are deterministic
    functions of the input, so one repeat is mathematically sufficient for them
    -- running five would be five identical numbers reported as a confidence
    interval, which is worse than useless. Latency is the only genuinely
    stochastic quantity and gets the dedicated unmemoised pass.

    `resume_log` wires the completion markers: a 16-hour unattended run WILL be
    interrupted, and without this an interruption at hour 14 costs 14 hours.
    """
    from parsimony.core.types import Mode
    from parsimony.infra.memo import CompletionLog, GenerationMemo

    run_id = run_id or ulid()
    # A sweep IS an experiment, so set the mode here rather than requiring every
    # caller to remember. Forgetting it silently disables the memo (the pipeline
    # only consults it in EXPERIMENT mode) and downgrades ledger-write failures
    # from fatal to ignored — both of which are exactly the kind of quiet
    # miscalibration this project keeps finding.
    cells = [replace(c, mode=Mode.EXPERIMENT) for c in cells]
    report = SweepReport(
        quality_corpus_size=len(corpus), timing_repeats=timing_repeats
    )
    log = CompletionLog(resume_log) if resume_log is not None else None

    def _note(message: str) -> None:
        if progress is not None:
            progress(message)

    try:
        memo = GenerationMemo()
        report.quality = sweep(
            cells, corpus,
            provider=provider, tokenizer=tokenizer, embedder=embedder, memo=memo,
            sink=sink, run_id=run_id, judge=judge, gold=gold,
            progress=lambda label: _note(f"quality: {label}"),
        )
        report.memo_hits, report.memo_total = memo.hits, memo.hits + memo.misses

        subset = corpus.subset(timing_subset)
        report.timing_corpus_size = len(subset)
        for repeat in range(timing_repeats):
            for cfg in cells:
                marker = (cfg.config_hash, repeat, "timing")
                if log is not None and log.is_done(*marker):
                    report.resumed += 1
                    continue
                _note(f"timing r{repeat + 1}: {cfg.label}")
                report.timing.append(
                    run_cell(
                        replace(cfg, seed=repeat), subset,
                        provider=provider, tokenizer=tokenizer, embedder=embedder,
                        memo=None,  # never memoised: this pass measures the clock
                        sink=sink, run_id=run_id, pass_kind="timing",
                    )
                )
                if log is not None:
                    log.mark(*marker)
    finally:
        if log is not None:
            log.close()
    return summarise_timing(report)


def summarise_timing(report: SweepReport) -> SweepReport:
    """Merge repeats of the same cell so latency has a real sample behind it."""
    merged: dict[str, CellResult] = {}
    for cell in report.timing:
        existing = merged.get(cell.label)
        if existing is None:
            merged[cell.label] = cell
            continue
        existing.middleware_ms.extend(cell.middleware_ms)
        existing.prefix_tokens.extend(cell.prefix_tokens)
        existing.joules.extend(cell.joules)
    report.timing = summarise(list(merged.values()))
    return report


def calibration_table(results: Sequence[CellResult]) -> dict[str, list[tuple[str, float]]]:
    """Per query class, every configuration ranked by token reduction.

    This is Contribution 6 in its practitioner-facing form: not one headline
    percentage, but "for THIS kind of question, switch these modules on". A
    module that helps summarisation and does nothing for arithmetic should be
    visible as exactly that, and an aggregate number hides it.
    """
    baseline = next((r for r in results if r.label == "baseline"), None)
    if baseline is None:
        return {}

    table: dict[str, list[tuple[str, float]]] = {}
    for cls, base_total in sorted(baseline.per_class_tokens.items()):
        if not base_total:
            continue
        ranked = [
            (r.label, 100.0 * (1 - r.per_class_tokens.get(cls, 0) / base_total))
            for r in results
        ]
        table[cls] = sorted(ranked, key=lambda x: -x[1])
    return table


def best_per_class(results: Sequence[CellResult]) -> dict[str, tuple[str, float]]:
    """The single recommended configuration for each query class."""
    return {cls: ranked[0] for cls, ranked in calibration_table(results).items() if ranked}


def shortfall_interval(
    results: Sequence[CellResult], axes: Sequence[str], resamples: int = 4000, seed: int = 0
):
    """Bootstrap CI for the additivity shortfall — the project's primary number.

    Resamples CONVERSATIONS, not requests. Turns within a conversation share
    history, so they are not independent observations; resampling requests would
    understate the interval and overstate our confidence.
    """
    from parsimony.eval.stats import Interval, bootstrap_ci

    by_label = {r.label: r for r in results}
    baseline = by_label.get("baseline")
    axis_set = set(axes)
    solo_labels = [a for a in axes if a in by_label]
    stacked_label = "+".join(a for a in axes if a in axis_set)
    stacked = by_label.get(stacked_label)
    if baseline is None or stacked is None or len(solo_labels) < 2:
        return None

    conversations = sorted(baseline.per_conversation)
    if not conversations:
        return None

    def shortfall_for(sample: Sequence[str]) -> float:
        base_total = sum(baseline.per_conversation.get(c, 0) for c in sample) or 1

        def reduction(cell: CellResult) -> float:
            total = sum(cell.per_conversation.get(c, 0) for c in sample)
            return 100.0 * (1 - total / base_total)

        predicted = sum(reduction(by_label[label]) for label in solo_labels)
        return predicted - reduction(stacked)

    rng = random.Random(seed)
    n = len(conversations)
    draws = [
        shortfall_for([conversations[rng.randrange(n)] for _ in range(n)])
        for _ in range(resamples)
    ]
    draws.sort()
    return Interval(
        point=shortfall_for(conversations),
        low=draws[max(0, int(0.025 * resamples) - 1)],
        high=draws[min(resamples - 1, int(0.975 * resamples))],
    )
