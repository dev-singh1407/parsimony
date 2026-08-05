"""Ledger schema v1. Additive-only, forward-only (ADR-014).

New columns are nullable; columns are never renamed or dropped. Sprint 0's
baseline runs must still be readable in October, because every claim in the
report is a difference against that baseline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping

SCHEMA_VERSION = 1


class StageOutcome(Enum):
    APPLIED = "applied"
    REVERTED = "reverted"  # fidelity gate rejected the patch
    SHORT_CIRCUIT = "short_circuit"
    NOOP = "noop"
    SKIPPED = "skipped"
    ERROR = "error"
    NOT_IMPLEMENTED = "not_implemented"  # declared in stage_order, lands in a later sprint


@dataclass(frozen=True, slots=True)
class GateEvent:
    module_id: str
    invariant_class: str
    lost_values: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StageTrace:
    module_id: str
    name: str
    outcome: StageOutcome
    tokens_before: int
    tokens_after: int
    duration_ns: int
    rationale: str = ""
    evidence: Mapping[str, Any] = field(default_factory=dict)
    gate_events: tuple[GateEvent, ...] = ()

    @property
    def delta(self) -> int:
        return self.tokens_after - self.tokens_before

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["outcome"] = self.outcome.value
        return d


@dataclass(frozen=True, slots=True)
class LedgerRow:
    # identity
    request_id: str
    conversation_id: str
    turn_index: int
    config_hash: str
    run_id: str
    schema_version: int = SCHEMA_VERSION
    corpus_hash: str | None = None
    seed: int = 0
    pass_kind: str = "quality"  # "quality" | "timing" — every figure states which
    config_label: str = ""
    created_at: float = 0.0

    # model identity
    model_name: str = ""
    model_quantisation: str = ""
    model_digest: str = ""
    tokenizer_id: str = ""
    embedder_id: str = ""

    # token accounting
    tokens_in_original: int = 0
    tokens_in_final: int = 0
    tokens_per_module: Mapping[str, int] = field(default_factory=dict)
    tokens_out: int = 0
    tokens_out_budget: int | None = None

    # routing / cache
    route_tier: str = ""
    cache_consulted: bool = False
    # Persisted even on a MISS: this is what turns the whole similarity-threshold
    # sweep into an offline groupby instead of N CPU-bound runs.
    cache_top_k: tuple[tuple[str, float], ...] = ()
    cache_zone: str | None = None
    cache_verifier: Mapping[str, float] | None = None
    cache_hit: bool = False

    # fidelity
    gate_fired: bool = False
    gate_events: tuple[GateEvent, ...] = ()

    # prefix / KV
    prefix_tokens_survived: int | None = None
    prefix_ratio: float | None = None

    # timing (ns)
    ttft_ns: int | None = None
    tpot_ns: int | None = None
    total_ns: int = 0
    middleware_ns: int = 0
    per_stage_ns: Mapping[str, int] = field(default_factory=dict)
    generation_memoised: bool = False  # True disqualifies this row from latency analysis
    early_stopped: bool = False  # M5's streaming stop rule fired

    # energy and cost (report 4.5)
    joules_estimated: float | None = None
    usd_equivalent: float | None = None

    # content (hashes; text lives in the blob store)
    prompt_sha256: str = ""
    response_sha256: str = ""

    # quality (populated offline)
    q_embedding_sim: float | None = None
    q_token_overlap: float | None = None
    q_judge: float | None = None
    q_judge_swap_agreed: bool | None = None
    q_exact_match: bool | None = None

    # trace (not a column in the analysis DB; kept in JSONL for auditing)
    traces: tuple[StageTrace, ...] = ()

    @property
    def tokens_total(self) -> int:
        return self.tokens_in_final + self.tokens_out

    @property
    def tokens_total_original(self) -> int:
        return self.tokens_in_original + self.tokens_out

    def to_dict(self, include_traces: bool = True) -> dict[str, Any]:
        d: dict[str, Any] = {}
        for k, v in asdict(self).items():
            if k == "traces":
                continue
            d[k] = _jsonable(v)
        if include_traces:
            d["traces"] = [t.to_dict() for t in self.traces]
        return d


def _jsonable(v: Any) -> Any:
    if isinstance(v, Enum):
        return v.value
    if isinstance(v, tuple):
        return [_jsonable(x) for x in v]
    if isinstance(v, list):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    return v
