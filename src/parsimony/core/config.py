"""Configuration is experiment identity (ADR-008).

Every threshold in the system lives here and nowhere else. A float literal
inside modules/ is a bug: it makes M7's output un-loadable and the per-model
calibration table un-assemblable.

config_hash is written into every ledger row. An ablation cell *is* a config
hash, which gives exact reproduction, drift detection, and a join key for the
ANOVA that cannot silently mismatch.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Iterator, Literal

from parsimony.core.errors import ConfigError
from parsimony.core.types import Mode

# Stage names, in the recommended execution order (docs/00-architecture.md 5).
# Deviations from the report's figure 3.4 are ADR-016 (deterministic tier first).
DEFAULT_STAGE_ORDER: tuple[str, ...] = (
    "m6a_deterministic",
    "m2_cache",
    "m3_history",
    "m3_arrange",
    "m1_tier1",
    "m1_tier2",
    "m1_tier3",
    "m4_assembler",
    "m5_budgeter",
    "m6b_router",
)

# Declared but not yet implemented. The registry skips these and records
# "not_implemented" in the trace, so the roadmap is visible in every run rather
# than silently absent. Empty: every stage in DEFAULT_STAGE_ORDER is now built.
PLANNED_STAGES: frozenset[str] = frozenset()

# The four factorial axes (report 4.6).
FACTORIAL_MODULES: tuple[str, ...] = ("M1", "M2", "M3", "M5")


@dataclass(frozen=True, slots=True)
class CompressionConfig:
    tier1_enabled: bool = True  # lossless normalisation
    tier2_enabled: bool = True  # extractive redundancy removal
    # Enabled: the windowed re-tokenisation it relies on is guarded by
    # tests/golden/test_windowed_retokenisation.py, which checks the windowed
    # delta against full re-tokenisation for every candidate edit in the corpus.
    tier3_enabled: bool = True
    # Two thresholds, because a similarity threshold is meaningless without the
    # scorer it was calibrated against. Cosine runs far hotter than Jaccard for
    # the same sentence pair. Keeping both explicit is the calibration-table
    # argument (Contribution 6) applied to our own stack rather than asserted
    # about someone else's.
    # CALIBRATED, not guessed (`parsimony calibrate-dedup`). The previous 0.80
    # was set by eye from a sentence pair that was not in the corpus, and tier 2
    # fired zero times in 239 opportunities. 0.70 is the loosest threshold whose
    # gate-revert rate is still 0%.
    #
    # It only recovers near-verbatim repeats. Under a LEXICAL encoder, genuine
    # paraphrases sit far lower -- "Rent is 1200 per month" against "Monthly
    # rent comes to 1200" scores 0.324 -- so there is no threshold that catches
    # them without also merging unrelated sentences. See ADR-028: tier 2 is
    # encoder-limited, not technique-limited.
    dedup_threshold: float = 0.70  # cosine under hashing-v1
    dedup_threshold_lexical: float = 0.62  # Jaccard fallback
    min_sentence_tokens: int = 4
    retokenise_window: int = 32
    max_ratio: float = 3.0


@dataclass(frozen=True, slots=True)
class CacheConfig:
    exact_tier: bool = True
    semantic_tier: bool = True
    # CALIBRATED FOR hashing-v1, NOT INHERITED FROM THE LITERATURE.
    #
    # Measured on our own pairs, this encoder puts the adversarial negation pair
    # ("is X safe" / "is X NOT safe") at cosine 0.924 — higher than every
    # genuine paraphrase in the set. Any tau_hi at or below that auto-accepts a
    # cache hit that returns the opposite answer, without the verifier ever
    # running. The published "safe" thresholds (0.85-0.92) do exactly that here.
    #
    # So tau_hi sits above the adversarial band and the verify zone is wide.
    # The consequence is deliberate: almost nothing auto-accepts on similarity
    # alone, and the cheap invariant verifier does the discriminating work. That
    # is the finding, not a workaround — see ADR-024.
    tau_hi: float = 0.97
    tau_lo: float = 0.75
    jaccard_min: float = 0.55
    chain_depth: int = 2
    top_k: int = 5
    ttl_seconds: int = 86_400
    # LRU capacity. The report targets an 8 GB consumer laptop (4.7), and the
    # cache is the one component designed to accumulate: without a cap it grows
    # linearly forever. 10k entries is roughly 15 MB of vectors at 384 float32
    # plus the stored text, which is affordable and bounded.
    max_entries: int = 10_000


@dataclass(frozen=True, slots=True)
class HistoryConfig:
    # The recommended configuration. "recency" and "chronological" are the
    # control arms: an honest report has to show whether the clever strategies
    # actually beat "keep the last few turns in order" on short conversations.
    strategy: Literal["recency", "relevance", "mmr", "summary"] = "mmr"
    arrangement: Literal["chronological", "position_aware"] = "position_aware"
    max_turns: int = 6
    token_budget: int = 1024
    # Relevance vs redundancy in MMR. Lives here rather than on
    # CompressionConfig because M3 is what selects with it.
    mmr_lambda: float = 0.7
    summarise_async: bool = True


@dataclass(frozen=True, slots=True)
class BudgetConfig:
    early_stop: bool = True
    novelty_window: int = 48
    novelty_threshold: float = 0.25
    per_class: dict[str, int] = field(
        default_factory=lambda: {
            "arithmetic": 48,
            "factual": 128,
            "follow_up": 160,
            "summarisation": 256,
            "code": 512,
            "reasoning": 640,
        }
    )


@dataclass(frozen=True, slots=True)
class RouterConfig:
    deterministic_tier: bool = True
    escalation_tier: bool = False
    escalation_complexity: float = 0.75
    # Weights for the transparent complexity heuristic. Here rather than in the
    # module because they are tunable quantities, and M7 must be able to emit a
    # tuned set without a code change (ADR-008).
    complexity_weights: dict[str, float] = field(
        default_factory=lambda: {
            "reason_markers": 0.30,
            "is_reasoning_class": 0.20,
            "is_code_class": 0.15,
            "n_words": 0.15,
            "n_questions": 0.10,
            "n_history_turns": 0.10,
        }
    )
    complexity_caps: dict[str, float] = field(
        default_factory=lambda: {
            "reason_markers": 2.0,
            "n_words": 60.0,
            "n_questions": 3.0,
            "n_history_turns": 6.0,
        }
    )


@dataclass(frozen=True, slots=True)
class LearnerConfig:
    """M7. Offline, but its thresholds still belong in one place."""

    redundancy_similarity_floor: float = 0.98
    min_occurrences: int = 2
    min_recurrence: int = 2


@dataclass(frozen=True, slots=True)
class EnergyConfig:
    """Report 4.5 asks for joules per query, tokens per joule and a
    dollar-equivalent column.

    Energy is DERIVED from wall clock and an assumed package power, not
    measured with a power meter. The assumption is a config field rather than a
    buried constant precisely so that it is visible in every ledger row and can
    be corrected without touching code. A reader can divide it back out.

    Prices are per million tokens at commercial rates for a small hosted model.
    The project spends nothing; the column exists to make the magnitude legible.
    """

    package_power_watts: float = 15.0  # i5-class laptop package under sustained load
    usd_per_million_input: float = 0.15
    usd_per_million_output: float = 0.60


@dataclass(frozen=True, slots=True)
class ModelConfig:
    name: str = "mock-1b"
    quantisation: str = "none"
    digest: str = "mock"
    judge_name: str | None = None


@dataclass(frozen=True, slots=True)
class ParsimonyConfig:
    mode: Mode = Mode.SERVE
    enabled_modules: frozenset[str] = frozenset({"M1", "M2", "M3", "M4", "M5", "M6"})
    stage_order: tuple[str, ...] = DEFAULT_STAGE_ORDER
    cache_lookup_on: Literal["RAW", "COMPRESSED", "BOTH"] = "RAW"
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    history: HistoryConfig = field(default_factory=HistoryConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    learner: LearnerConfig = field(default_factory=LearnerConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    tokenizer_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    # Lexical embedder by default — no PyTorch. "all-MiniLM-L6-v2" swaps in
    # behind the same protocol once the models extra is installed; every
    # similarity threshold must then be recalibrated (see infra/embedding.py).
    #
    # content-v1 over hashing-v1 (ADR-035): stopword removal and stemming, both
    # still purely lexical. At the SAME tau_hi = 0.97 it holds the false-hit
    # rate at 0.0% and raises true hits 22.2% -> 28.9%. It is also safe at 0.92
    # (31.1% true), but 0.90 is not (4.4% false), so the default stays at 0.97
    # rather than sitting one adversarial pair away from the cliff.
    embedder_id: str = "content-v1"
    system_prompt: str = "You are a concise, accurate assistant."
    # Standing facts mined by M7. Part of M4's invariant zone, so it is
    # byte-stable across turns and lengthens the reusable KV prefix.
    context_digest: str = ""
    # Content hash of the loaded PolicyBundle, recorded in every ledger row.
    # "Warm-started" vs "cold" is this field being set, not a separate code path.
    bundle_hash: str = ""
    seed: int = 0
    label: str = ""

    def __post_init__(self) -> None:
        if self.cache_lookup_on not in ("RAW", "COMPRESSED", "BOTH"):
            raise ConfigError(f"invalid cache_lookup_on: {self.cache_lookup_on!r}")
        if not self.stage_order:
            raise ConfigError("stage_order must not be empty")
        dupes = [s for s in set(self.stage_order) if self.stage_order.count(s) > 1]
        if dupes:
            raise ConfigError(f"duplicate stages in stage_order: {sorted(dupes)}")

    def enables(self, module_id: str) -> bool:
        return module_id in self.enabled_modules

    def canonical(self) -> dict[str, Any]:
        """Deterministic, JSON-safe view. Excludes cosmetic fields."""
        raw = asdict(self)
        raw.pop("label", None)
        return _canonicalise(raw)

    @property
    def config_hash(self) -> str:
        blob = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"))
        return hashlib.blake2b(blob.encode("utf-8"), digest_size=8).hexdigest()

    def with_modules(self, modules: frozenset[str], label: str = "") -> ParsimonyConfig:
        return replace(self, enabled_modules=modules, label=label)


def _canonicalise(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _canonicalise(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (frozenset, set)):
        return sorted(_canonicalise(v) for v in obj)
    if isinstance(obj, (list, tuple)):
        return [_canonicalise(v) for v in obj]
    if isinstance(obj, Mode):
        return obj.value
    return obj


# --------------------------------------------------------------------------
# Presets
# --------------------------------------------------------------------------


def baseline() -> ParsimonyConfig:
    """Everything off. Every later claim is a difference against this."""
    return ParsimonyConfig(enabled_modules=frozenset(), label="baseline")


def full_stack() -> ParsimonyConfig:
    return ParsimonyConfig(
        enabled_modules=frozenset({"M1", "M2", "M3", "M4", "M5", "M6"}), label="full"
    )


def with_cache_lookup(cfg: ParsimonyConfig, mode: str) -> ParsimonyConfig:
    """Move the cache lookup before or after compression.

    This is ADR-002 doing its job. Gap 3 asks whether compressing a query before
    it reaches the semantic cache raises or lowers the false-hit rate; that
    question is unanswerable if the order is fixed in code. Here it is one
    config value, and the two arms are otherwise byte-identical pipelines.
    """
    order = [s for s in cfg.stage_order if s not in ("m2_cache", "m2_cache_probe")]

    def _after_deterministic() -> int:
        return order.index("m6a_deterministic") + 1 if "m6a_deterministic" in order else 0

    def _after_compression() -> int:
        return max(
            (i for i, s in enumerate(order) if s.startswith("m1_")), default=len(order) - 1
        ) + 1

    if mode == "COMPRESSED":
        order.insert(_after_compression(), "m2_cache")
    elif mode == "BOTH":
        # Probe on the raw query, authoritative lookup on the compressed one.
        # One request then yields a PAIRED observation of the same cache under
        # both orderings, which is a much stronger design for Gap 3 than
        # comparing two independent runs. Insert the later position first so
        # the earlier insert does not shift it.
        order.insert(_after_compression(), "m2_cache")
        order.insert(_after_deterministic(), "m2_cache_probe")
    else:  # RAW
        order.insert(_after_deterministic(), "m2_cache")
    return replace(cfg, stage_order=tuple(order), cache_lookup_on=mode)


def factorial_cells(
    base: ParsimonyConfig | None = None,
    axes: tuple[str, ...] = FACTORIAL_MODULES,
    always_on: frozenset[str] = frozenset({"M6"}),
) -> Iterator[ParsimonyConfig]:
    """Enumerate the 2^len(axes) ablation cells.

    M6 is on in every cell by default: it is studied on top of the winning
    configuration (report 4.6), not as a factorial axis.
    """
    base = base or ParsimonyConfig()
    for bits in itertools.product([False, True], repeat=len(axes)):
        on = frozenset(m for m, b in zip(axes, bits) if b) | always_on
        label = "+".join(m for m, b in zip(axes, bits) if b) or "baseline"
        yield base.with_modules(on, label=label)
