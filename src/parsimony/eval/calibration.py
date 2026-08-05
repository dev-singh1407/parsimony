"""Threshold calibration against the adversarial subset.

This produces Contributions 2 and 6 directly:

  * the false-cache-hit rate as a function of the similarity threshold, with
    compression on and off (Gap 3);
  * a per-configuration safe operating point, replacing the single universal
    number the literature offers (Gap 5).

METHOD
------
For each adversarial pair, store A in a fresh cache and then query with B.

    answers_differ and B hits   -> FALSE HIT   (the failure mode)
    answers_differ and B misses -> correct rejection
    same answer  and B hits     -> TRUE HIT    (the saving we want)
    same answer  and B misses   -> missed opportunity

The control pairs (answers_differ = false) matter as much as the adversarial
ones. Without them a trivially safe policy — reject everything — scores a
perfect 0% false-hit rate, so the sweep would recommend disabling the cache.
Reporting both rates is what makes the operating point meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from parsimony.core.config import ParsimonyConfig
from parsimony.eval.corpus import AdversarialPair, load_adversarial
from parsimony.infra.embedding import get_embedder
from parsimony.modules.m1_compressor import normalise_lossless
from parsimony.modules.m2_cache import SemanticCache, verify_match


class ApproximateIndexError(RuntimeError):
    """Raised rather than returning a contaminated false-hit rate (ADR-004)."""


@dataclass(frozen=True, slots=True)
class CalibrationPoint:
    tau_hi: float
    tau_lo: float
    verifier_on: bool
    compression_on: bool
    false_hits: int
    adversarial_total: int
    true_hits: int
    control_total: int
    index_is_exact: bool = True

    @property
    def false_hit_rate(self) -> float:
        return 100.0 * self.false_hits / self.adversarial_total if self.adversarial_total else 0.0

    @property
    def true_hit_rate(self) -> float:
        return 100.0 * self.true_hits / self.control_total if self.control_total else 0.0

    @property
    def is_safe(self) -> bool:
        """Report 3.3 sets the target below 2%."""
        if not self.index_is_exact:
            raise ApproximateIndexError(
                "Refusing to judge safety from an approximate index. Measured on the "
                "adversarial subset, LSH reports a 46.7% false-hit rate where exact search "
                "reports 84.4% — the approximate index simply fails to retrieve the "
                "dangerous neighbour, so the danger goes uncounted and the cache looks "
                "roughly twice as safe as it is (ADR-004)."
            )
        return self.false_hit_rate < 2.0


def _lookup_hits(
    query: str,
    stored_query: str,
    embedder,
    cfg: ParsimonyConfig,
    verifier_on: bool,
    index_factory=None,
) -> bool:
    """Would this query be served from a cache holding only `stored_query`?"""
    index = index_factory(embedder.dim) if index_factory is not None else None
    cache = SemanticCache(cfg.cache.ttl_seconds, embedder, index=index)
    vec_stored, vec_query = embedder.embed([stored_query, query])
    cache.store("k", stored_query, "stored answer", chain="root", model_id="m", vec=vec_stored)

    if SemanticCache.make_key(query, "root", "m") == SemanticCache.make_key(
        stored_query, "root", "m"
    ):
        return True  # exact tier

    found = cache.search(vec_query, "root", "m", cfg.cache.top_k)
    if not found:
        return False
    entry, score = found[0]

    if score >= cfg.cache.tau_hi:
        return True
    if score < cfg.cache.tau_lo:
        return False
    if not verifier_on:
        return True  # single-threshold behaviour: anything above tau_lo is a hit
    return verify_match(
        cache.invariants_of(query),
        entry.invariants,
        query,
        entry.query,
        cfg.cache.jaccard_min,
    ).passed


def evaluate_point(
    pairs: tuple[AdversarialPair, ...],
    cfg: ParsimonyConfig,
    embedder,
    verifier_on: bool = True,
    compression_on: bool = False,
    index_factory=None,
) -> CalibrationPoint:
    false_hits = adversarial_total = true_hits = control_total = 0

    for pair in pairs:
        a, b = pair.a, pair.b
        if compression_on:
            # The Gap 3 manipulation: the cache sees normalised text on both
            # the write and the lookup path.
            a, b = normalise_lossless(a), normalise_lossless(b)

        hit = _lookup_hits(b, a, embedder, cfg, verifier_on, index_factory)
        if pair.answers_differ:
            adversarial_total += 1
            false_hits += int(hit)
        else:
            control_total += 1
            true_hits += int(hit)

    exact = True
    if index_factory is not None:
        probe = index_factory(embedder.dim)
        exact = bool(getattr(probe, "is_exact", lambda: True)())

    return CalibrationPoint(
        tau_hi=cfg.cache.tau_hi,
        tau_lo=cfg.cache.tau_lo,
        verifier_on=verifier_on,
        compression_on=compression_on,
        false_hits=false_hits,
        adversarial_total=adversarial_total,
        true_hits=true_hits,
        control_total=control_total,
        index_is_exact=exact,
    )


DEFAULT_SWEEP = (0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.99)


def sweep_thresholds(
    base: ParsimonyConfig,
    thresholds: tuple[float, ...] = DEFAULT_SWEEP,
    pairs: tuple[AdversarialPair, ...] | None = None,
    embedder=None,
    verifier_on: bool = True,
    compression_on: bool = False,
) -> list[CalibrationPoint]:
    pairs = pairs if pairs is not None else load_adversarial()
    embedder = embedder or get_embedder(base.embedder_id)
    out = []
    for tau in thresholds:
        cfg = replace(base, cache=replace(base.cache, tau_hi=tau, tau_lo=min(tau, base.cache.tau_lo)))
        out.append(evaluate_point(pairs, cfg, embedder, verifier_on, compression_on))
    return out


@dataclass(frozen=True, slots=True)
class DedupPoint:
    """One operating point for M1 tier 2's near-duplicate threshold."""

    threshold: float
    proposed: int
    applied: int
    reverted: int
    tokens_saved: int

    @property
    def revert_rate(self) -> float:
        return 100.0 * self.reverted / self.proposed if self.proposed else 0.0

    @property
    def saving_per_edit(self) -> float:
        return self.tokens_saved / self.applied if self.applied else 0.0


DEDUP_SWEEP = (0.60, 0.65, 0.70, 0.72, 0.75, 0.78, 0.80, 0.85, 0.90)


def sweep_dedup_threshold(
    base: ParsimonyConfig,
    corpus,
    thresholds: tuple[float, ...] = DEDUP_SWEEP,
) -> list[DedupPoint]:
    """Measure tier 2's behaviour across thresholds instead of guessing one.

    The shipped default (0.80) was set by eye from a sentence pair that was not
    actually in the corpus, and tier 2 consequently never fired once in 239
    opportunities. Guessing a threshold is precisely what this project
    criticises the caching literature for, so it gets the same treatment as the
    cache: sweep it, look at where the fidelity gate starts objecting, and pick
    the operating point from data.

    The gate's revert rate is the safety signal. A threshold low enough to merge
    sentences that differ in a number or an entity shows up as reverts, not as
    silent damage.
    """
    from parsimony.eval.runner import run_conversation
    from parsimony.pipeline.orchestrator import Pipeline

    points: list[DedupPoint] = []
    for threshold in thresholds:
        cfg = replace(
            base,
            enabled_modules=frozenset({"M1"}),
            compression=replace(base.compression, dedup_threshold=threshold),
            label=f"dedup@{threshold}",
        )
        pipeline = Pipeline(cfg)
        proposed = applied = reverted = saved = 0
        for conv in corpus.conversations:
            for outcome in run_conversation(pipeline, conv):
                for trace in outcome.traces:
                    if trace.name != "m1_tier2":
                        continue
                    if trace.outcome.value == "applied":
                        proposed += 1
                        applied += 1
                        saved += trace.tokens_before - trace.tokens_after
                    elif trace.outcome.value == "reverted":
                        proposed += 1
                        reverted += 1
        points.append(DedupPoint(threshold, proposed, applied, reverted, saved))
    return points


def by_operative(
    pairs: tuple[AdversarialPair, ...], cfg: ParsimonyConfig, embedder, verifier_on: bool = True
) -> dict[str, tuple[int, int]]:
    """False hits per operative class — which edit type defeats the cache."""
    out: dict[str, list[int]] = {}
    for pair in pairs:
        if not pair.answers_differ:
            continue
        hit = _lookup_hits(pair.b, pair.a, embedder, cfg, verifier_on)
        bucket = out.setdefault(pair.operative, [0, 0])
        bucket[0] += int(hit)
        bucket[1] += 1
    return {k: (v[0], v[1]) for k, v in sorted(out.items())}
