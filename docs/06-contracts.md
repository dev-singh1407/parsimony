# Parsimony — L0 Contracts and Ledger Schema

This is the review surface. Everything else is written against these definitions, so **get this reviewed
before building on it** — changing `RequestContext` on day 2 is free, in week 6 it is a re-run of every
experiment.

L0 imports nothing outside the standard library (`00-architecture.md` §2). `np.ndarray` appears only in
protocol signatures under `TYPE_CHECKING`.

---

## 1. Domain types

```python
# parsimony/core/types.py
from __future__ import annotations
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from typing import Any, Literal, Mapping, Protocol, Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import spacy


Role = Literal["system", "user", "assistant"]


class ResponseClass(Enum):
    FACTUAL       = "factual"
    ARITHMETIC    = "arithmetic"
    REASONING     = "reasoning"
    CODE          = "code"
    SUMMARISATION = "summarisation"
    FOLLOW_UP     = "follow_up"


class RouteTier(Enum):
    DETERMINISTIC  = 0   # zero model tokens
    CACHE_EXACT    = 1
    CACHE_SEMANTIC = 2
    MODEL_SMALL    = 3   # 1B
    MODEL_LARGE    = 4   # 3B


class InvariantClass(Enum):
    NUMBER   = auto()
    ENTITY   = auto()
    NEGATION = auto()
    QUOTED   = auto()


@dataclass(frozen=True, slots=True)
class Turn:
    turn_id: str
    role: Role
    content: str
    created_at: float
    token_count: int          # under the ACTIVE target tokenizer, never a word count


@dataclass(frozen=True, slots=True)
class Invariants:
    """Fidelity fingerprint. Extracted once in Stage 2; every gate check is a set difference."""
    numbers:   frozenset[str]
    entities:  frozenset[str]
    negations: frozenset[str]
    quoted:    frozenset[str]

    def missing_from(self, text: str) -> dict[InvariantClass, frozenset[str]]:
        """Invariant values present in self but absent from `text`. Empty dict == fidelity preserved."""
        ...

    def union(self, other: Invariants) -> Invariants: ...


@dataclass(frozen=True, slots=True)
class AssembledPrompt:
    invariant_zone: str        # byte-stable across turns; M4 alone may write it
    volatile_zone: str
    full_text: str
    prefix_token_count: int    # tokens in the invariant zone
    total_token_count: int
```

---

## 2. Request context

```python
class DerivedCache:
    """Per-request memo (ADR-006). Lazy: a request short-circuited at router tier 0
    never pays for sentence splitting or embeddings. Modules MUST go through this,
    never the embedder directly — enforced by tests/test_architecture.py."""

    def token_count(self, text: str) -> int: ...
    def sentences(self, text: str) -> tuple[str, ...]: ...

    # Text-keyed, batched, memoised. NOT `query_embedding()` / `turn_embeddings()`:
    # modules rewrite the query mid-pipeline, so a binding captured at ingestion
    # would be stale by the time M1 has run. Keying on the text itself stays
    # correct under rewriting and still collapses the four passes the naive
    # design would make.
    def embed(self, texts: list[str]) -> np.ndarray: ...   # (n, 384) L2-normalised
    def embed_one(self, text: str) -> np.ndarray: ...      # convenience: embed([t])[0]

    def stats(self) -> dict[str, int]: ...                 # what was computed; lands in the ledger


class Mode(Enum):
    SERVE      = auto()   # ledger loss is survivable; the request is what matters
    EXPERIMENT = auto()   # ledger loss is fatal; the ledger IS the result


@dataclass(frozen=True, slots=True)
class RequestContext:
    # identity
    request_id: str
    conversation_id: str
    turn_index: int
    config_hash: str
    corpus_hash: str | None            # set in EXPERIMENT mode (ADR-015)

    # immutable reference — the fidelity gate's ground truth, written once in Stage 1
    original_query: str
    original_history: tuple[Turn, ...]
    invariants: Invariants

    # working state — only the orchestrator writes, only via committed patches
    query: str
    history: tuple[Turn, ...]
    system_prompt: str
    context_digest: str

    # decisions accumulated by stages
    response_class: ResponseClass | None = None
    complexity: float | None = None
    output_budget: int | None = None
    route_tier: RouteTier | None = None
    assembled: AssembledPrompt | None = None

    derived: DerivedCache = field(default_factory=DerivedCache, compare=False)

    def text_payload(self) -> str:
        """Concatenation the fidelity gate checks REWRITE proposals against."""
        ...
```

---

## 3. Proposals

```python
class TransformKind(Enum):
    REWRITE = auto()   # same information, new surface form  -> full invariant check
    SELECT  = auto()   # deliberate removal of whole units   -> retained-span byte check
    AUGMENT = auto()   # adds content, removes nothing       -> no check
    DECIDE  = auto()   # sets a decision field, no text edit -> no check


@dataclass(frozen=True, slots=True)
class ContextPatch:
    kind: TransformKind
    fields: Mapping[str, Any]        # RequestContext field -> new value
    rationale: str
    evidence: Mapping[str, Any]      # module-specific numbers, flattened into the ledger


@dataclass(frozen=True, slots=True)
class ShortCircuit:
    response: str
    served_by: RouteTier
    evidence: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class NoOp:
    reason: Literal["disabled", "not_applicable", "no_yield", "error"]
    detail: str = ""


Proposal = ContextPatch | ShortCircuit | NoOp
```

**Invariant enforced by contract test:** a `ContextPatch` may only name fields in its stage's declared
`writes` set. A stage that writes an undeclared field fails at test time, not at 3 a.m. in hour 9 of a sweep.

---

## 4. Protocols

```python
# parsimony/core/protocols.py

class Stage(Protocol):
    module_id: str                    # "M1"
    name: str                         # "compressor"
    reads:  frozenset[str]
    writes: frozenset[str]
    def applies_to(self, ctx: RequestContext, cfg: ParsimonyConfig) -> bool: ...
    def propose(self, ctx: RequestContext, cfg: ParsimonyConfig) -> Proposal: ...


@dataclass(frozen=True, slots=True)
class GenParams:
    num_predict: int
    temperature: float = 0.0
    stop: tuple[str, ...] = ()
    seed: int = 0


@dataclass(frozen=True, slots=True)
class TokenEvent:
    text: str
    index: int
    emitted_at_ns: int                # perf_counter_ns at receipt -> TTFT and TPOT


class LLMProvider(Protocol):
    def generate(self, prompt: str, params: GenParams) -> Iterator[TokenEvent]: ...
    def tokenizer_id(self) -> str: ...
    def model_digest(self) -> str: ...       # pinned; a silent `ollama pull` becomes visible
    def model_name(self) -> str: ...
    def quantisation(self) -> str: ...


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def count(self, text: str) -> int: ...
    def offsets(self, text: str) -> list[tuple[int, int]]: ...   # windowed re-tokenisation, prefix survival
    def id(self) -> str: ...


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...   # BATCHED by contract — no single-text method
    def dim(self) -> int: ...
    def id(self) -> str: ...


class VectorIndex(Protocol):
    def add(self, vec: np.ndarray, entry_id: str) -> None: ...
    def search(self, vec: np.ndarray, k: int) -> list[tuple[str, float]]: ...
    def remove(self, entry_id: str) -> None: ...
    def size(self) -> int: ...
    def is_exact(self) -> bool: ...        # False disqualifies it from Gap 3 measurement (ADR-004)


class LedgerSink(Protocol):
    def write(self, row: LedgerRow) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...


class PiiDetector(Protocol):
    def spans(self, text: str) -> list[tuple[int, int, str]]: ...   # start, end, label
```

`Embedder` deliberately exposes **only** a batched method. A `embed_one()` convenience would be called in a
loop by someone in week 7 and would quietly cost 5× (ADR-006). Removing the temptation is cheaper than
policing it.

`VectorIndex.is_exact()` exists so the analysis layer can *refuse* to compute a false-hit rate from an
approximate index rather than silently producing a contaminated number.

---

## 5. Configuration

```python
# parsimony/core/config.py

@dataclass(frozen=True, slots=True)
class CompressionConfig:
    tier1_enabled: bool = True
    tier2_enabled: bool = True
    tier3_enabled: bool = True
    mmr_lambda: float = 0.7
    dedup_threshold: float = 0.92
    retokenise_window: int = 32        # chars; golden test asserts equivalence with full retokenisation
    max_ratio: float = 3.0

@dataclass(frozen=True, slots=True)
class CacheConfig:
    exact_tier: bool = True
    semantic_tier: bool = True
    tau_hi: float = 0.92
    tau_lo: float = 0.78
    jaccard_min: float = 0.55
    chain_depth: int = 2
    top_k: int = 5
    ttl_seconds: int = 86_400

@dataclass(frozen=True, slots=True)
class HistoryConfig:
    strategy: Literal["recency", "relevance", "mmr", "summary"] = "mmr"
    arrangement: Literal["chronological", "position_aware"] = "position_aware"
    max_turns: int = 6
    token_budget: int = 1024
    summarise_async: bool = True       # ADR-010

@dataclass(frozen=True, slots=True)
class BudgetConfig:
    per_class: Mapping[str, int] = ...
    early_stop: bool = True
    novelty_window: int = 48
    novelty_threshold: float = 0.25

@dataclass(frozen=True, slots=True)
class RouterConfig:
    deterministic_tier: bool = True
    escalation_tier: bool = False
    escalation_complexity: float = 0.75

@dataclass(frozen=True, slots=True)
class ModelConfig:
    name: str
    quantisation: Literal["Q4_K_M", "Q8_0"]
    digest: str
    judge_name: str | None = None      # must differ from `name` (05-evaluation-harness §3.3)

@dataclass(frozen=True, slots=True)
class ParsimonyConfig:
    mode: Mode
    enabled_modules: frozenset[str]
    stage_order: tuple[str, ...]
    cache_lookup_on: Literal["RAW", "COMPRESSED", "BOTH"] = "RAW"
    compression: CompressionConfig = CompressionConfig()
    cache: CacheConfig = CacheConfig()
    history: HistoryConfig = HistoryConfig()
    budget: BudgetConfig = BudgetConfig()
    router: RouterConfig = RouterConfig()
    model: ModelConfig = ...
    embedder_id: str = "all-MiniLM-L6-v2"
    seed: int = 0

    @property
    def config_hash(self) -> str:
        """BLAKE2b of canonical JSON, 16 hex chars. An ablation cell IS this value (ADR-008)."""
        ...
```

**Every threshold in the system appears above and nowhere else.** A float literal inside `modules/` is a lint
error. This is what makes M7's integration trivial (it emits a config) and the calibration table assemblable
(it is a table of these fields per model).

---

## 6. Ledger schema v1

Additive-only, forward-only (ADR-014). Nullable new columns; never rename, never drop. One row per request.

```python
@dataclass(frozen=True, slots=True)
class LedgerRow:
    # ---- identity ----
    request_id: str
    conversation_id: str
    turn_index: int
    config_hash: str
    corpus_hash: str | None
    schema_version: int                 # = 1
    run_id: str
    seed: int
    pass_kind: Literal["quality", "timing"]     # which sweep pass; every figure labels this
    created_at: float

    # ---- model identity ----
    model_name: str
    model_quantisation: str
    model_digest: str
    tokenizer_id: str
    embedder_id: str

    # ---- token accounting ----
    tokens_in_original: int
    tokens_in_final: int
    tokens_per_module: Mapping[str, int]        # module_id -> tokens after that module
    tokens_out: int
    tokens_out_budget: int | None

    # ---- routing / cache ----
    route_tier: str
    cache_consulted: bool
    cache_top_k: tuple[tuple[str, float], ...]  # (entry_id, score) — persisted even on a MISS,
                                                # which makes the threshold sweep an offline re-analysis
    cache_zone: Literal["accept", "verify", "reject", "miss"] | None
    cache_verifier: Mapping[str, float] | None  # jaccard, entity_agree, number_agree, negation_agree
    cache_hit: bool

    # ---- fidelity ----
    gate_fired: bool
    gate_events: tuple[GateEvent, ...]          # (module_id, invariant_class, lost_values)

    # ---- prefix / KV ----
    prefix_tokens_survived: int | None
    prefix_ratio: float | None

    # ---- timing (ns) ----
    ttft_ns: int | None
    tpot_ns: int | None
    total_ns: int
    middleware_ns: int                          # total minus provider time -> the <120 ms metric
    per_stage_ns: Mapping[str, int]
    generation_memoised: bool                   # True disqualifies this row from latency analysis

    # ---- energy ----
    joules_estimated: float | None

    # ---- content (hashes; text lives in the blob store) ----
    prompt_sha256: str
    response_sha256: str

    # ---- quality (populated offline) ----
    q_embedding_sim: float | None
    q_token_overlap: float | None
    q_judge: float | None
    q_judge_swap_agreed: bool | None
    q_exact_match: bool | None
```

Four fields carry more weight than their size suggests:

- **`pass_kind` + `generation_memoised`** — the harness's honesty mechanism. Any latency analysis filters
  `generation_memoised == False`; any figure states its `pass_kind`. Without these two fields, memoisation
  (`05-evaluation-harness.md` §1) would be a contamination risk instead of a pure optimisation.
- **`cache_top_k` persisted on a miss** — converts the entire similarity-threshold sweep from N CPU-bound
  runs into one `pandas` groupby. On a project that is short on CPU hours, this is the highest-leverage
  field in the schema.
- **`gate_events` with `invariant_class`** — turns the fidelity gate from a safety mechanism into a
  *finding*: "the gate fires on 11 % of tier-3 rewrites, 70 % of them numerals" is a direct contribution to
  the calibration table.
- **`per_stage_ns`** — substantiates the <120 ms overhead metric per module rather than in aggregate, which
  is what makes it actionable for a practitioner adopting the calibration table.

---

## 7. Contract test suites

`tests/contract/` holds one suite per protocol, run against **every** implementation. This is what makes
"components are replaceable" a checked claim (`00-architecture.md` §6).

| Suite | Applies to | Sample assertions |
|---|---|---|
| `test_stage_contract` | all 8 modules | patches only name declared `writes`; `propose` never mutates `ctx`; disabled ⇒ `NoOp`; declared `TransformKind` matches an observed-behaviour probe |
| `test_provider_contract` | Mock, Ollama, llama.cpp | streaming yields monotonic indices; `model_digest` stable across calls; `num_predict` respected; empty prompt handled |
| `test_index_contract` | Exact, FAISS, HNSW | top-k ordering; `is_exact()` truthful; add/remove/search consistency; recall@1 == 1.0 required when `is_exact()` |
| `test_tokenizer_contract` | HF wrappers | `count == len(encode)`; offsets cover the string exactly; round-trip stability |
| `test_sink_contract` | Jsonl, Sqlite | write→read round-trip; crash-safety to last complete record; schema version present |

Plus `tests/golden/` for the two claims that a unit test cannot cover: windowed re-tokenisation equals full
re-tokenisation across the whole corpus, and `reproduce.py` regenerates byte-identical figures from a frozen
sample ledger.
