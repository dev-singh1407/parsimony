# Parsimony — System Architecture

**Status:** Design baseline, v1.0
**Owner:** Dev Singh, Arrsh Tripathi, Alok Singh — VIT, BCSE497J
**Scope:** This document defines the structural architecture. Per-stage engineering detail is in
[`01-pipeline-stages.md`](01-pipeline-stages.md), module internals in [`02-module-specs.md`](02-module-specs.md),
justified choices in [`03-decision-log.md`](03-decision-log.md), schedule in [`04-roadmap.md`](04-roadmap.md).

---

## 1. What the architecture has to survive

Parsimony is not primarily an application. It is **a measurement instrument that happens to be usable as
middleware**. That inverts several ordinary priorities, and the architecture must be read in that light.

Four properties are load-bearing. Everything else is negotiable.

| Property | Why it is non-negotiable | What it forces |
|---|---|---|
| **Every module independently switchable** | The headline result is a 2⁴ factorial ablation. A module that cannot be cleanly disabled destroys a cell. | Modules never call each other. All coordination via the orchestrator. |
| **Stage order is data, not code** | Gap 3 asks what compression does to the cache. Answerable only if the cache can see the raw query *and* the compressed query, in separate runs. | Pipeline order is a config value validated against a dependency DAG. |
| **Every decision auditable to a row** | Success metrics include per-module token deltas, prefix survival, TTFT/TPOT split, gate firings. Retrofitted instrumentation is always wrong. | The trace is produced by the orchestrator, not by modules. Modules cannot forget to log. |
| **Middleware overhead < 120 ms** | It is a stated success metric, and a stack that costs more than it saves is a null result. | Shared computation is memoised per request. No module gets its own embedding pass or its own spaCy pass. |

**The consequence:** modules are *proposers*, not *actors*. A module inspects the request and returns a
description of the change it would like made. The orchestrator decides whether to apply it, times it, gates
it, and records it. This is the central architectural decision and the rest follows from it.

---

## 2. Layering

Dependencies point strictly downward. A layer may not import from a layer above it. This is **enforced by
`tests/test_architecture.py`**, which parses every source file and fails the suite on an upward import, a
third-party import inside L0, a module importing another module, or a threshold hard-coded outside
`ParsimonyConfig`. It is a test rather than an import-linter contract so that it runs in the existing suite
with no extra dependency and can explain *why* each rule exists when it fails.

```
┌──────────────────────────────────────────────────────────────────────┐
│ L5  SURFACES             CLI · FastAPI + OpenAI shim · dashboard ·   │
│                          MV3 extension                               │
├──────────────────────────────────────────────────────────────────────┤
│ L4  EXPERIMENTATION      corpus · sweep runner · metrics · analysis  │
├──────────────────────────────────────────────────────────────────────┤
│ L3  ORCHESTRATION        pipeline runner · stage registry · trace ·  │
│                          policy engine                               │
├──────────────────────────────────────────────────────────────────────┤
│ L2  MODULES              M1 M2 M3 M4 M5 M6 M7 M8                     │
├──────────────────────────────────────────────────────────────────────┤
│ L1  INFRASTRUCTURE       tokenizer · embedder · vector index ·       │
│                          ledger sink · blob store · providers        │
├──────────────────────────────────────────────────────────────────────┤
│ L0  CORE                 types · protocols · config · errors         │
│                          (stdlib only — zero third-party imports)    │
└──────────────────────────────────────────────────────────────────────┘
```

> **Correction (enforcement pass).** The original sketch put experimentation
> *above* surfaces. Dependency direction says otherwise: the CLI drives
> benchmarks, calibration and mining, so it imports `eval`, while nothing in
> `eval` imports a surface. Conceptual importance does not set layer order —
> who-imports-whom does. This is now checked by `tests/test_architecture.py`
> rather than asserted here.

**Why L0 has no third-party dependencies.** L0 defines `RequestContext`, `Proposal`, `Stage`, `LLMProvider`
and friends. If L0 imports pydantic, numpy or FastAPI, then a unit test of the compressor drags in a web
framework, and swapping FAISS for hnswlib becomes a core-type change. Keeping L0 at stdlib-only
(`dataclasses`, `typing`, `enum`) is what actually makes the "replaceable components" requirement true rather
than aspirational.

**Why modules sit below orchestration.** A module that imports the orchestrator can call the pipeline
recursively, and the ablation becomes meaningless because disabling a module no longer removes its effect.
The dependency arrow is the enforcement mechanism.

---

## 3. The core data model

### 3.1 `RequestContext` — frozen, versioned, replayable

```python
# L0: parsimony/core/types.py

@dataclass(frozen=True, slots=True)
class Turn:
    turn_id: str                    # ULID
    role: Literal["system", "user", "assistant"]
    content: str
    created_at: float               # epoch seconds
    token_count: int                # under the ACTIVE target tokenizer

@dataclass(frozen=True, slots=True)
class Invariants:
    """Fidelity fingerprint. Extracted ONCE at preprocess, reused by every gate check."""
    numbers:    frozenset[str]      # normalised numerics incl. units: "3.5kg", "1990"
    entities:   frozenset[str]      # spaCy PERSON/ORG/GPE/DATE/MONEY/PRODUCT
    negations:  frozenset[str]      # not, never, without, no, cannot, n't ...
    quoted:     frozenset[str]      # spans inside " " ' ' ` ` ``` ```
    def missing_from(self, text: str) -> "Invariants": ...

@dataclass(frozen=True, slots=True)
class RequestContext:
    # ---- identity ----
    request_id: str                 # ULID, primary key in the ledger
    conversation_id: str
    config_hash: str                # identifies the experiment cell

    # ---- immutable reference: the fidelity gate's ground truth ----
    original_query: str
    original_history: tuple[Turn, ...]
    invariants: Invariants

    # ---- working state: only the orchestrator mutates, via committed patches ----
    query: str
    history: tuple[Turn, ...]
    system_prompt: str
    context_digest: str             # from M7 policy bundle; part of the invariant zone

    # ---- decisions accumulated by stages ----
    response_class: ResponseClass | None = None
    complexity: float | None = None
    output_budget: int | None = None
    route_tier: RouteTier | None = None
    assembled: AssembledPrompt | None = None

    # ---- shared derived values, computed at most once ----
    derived: DerivedCache = field(default_factory=DerivedCache)
```

`frozen=True` is deliberate and is what makes the gate cheap: to evaluate a proposal the orchestrator builds
a *candidate* context with `dataclasses.replace(...)`, checks it, and either adopts it or throws it away.
There is no rollback logic anywhere in the codebase — reverting is simply not assigning.

`slots=True` matters at scale: the full sweep constructs ~7 contexts per request × 150 conversations ×
16 cells × 5 seeds × 3 models ≈ 2.5 M objects. Slots roughly halves that footprint.

### 3.2 `Proposal` — what a module returns

```python
class TransformKind(Enum):
    REWRITE = auto()   # same information, different surface form   → full invariant check
    SELECT  = auto()   # deliberate removal of whole units          → retained-span check only
    AUGMENT = auto()   # adds content, removes nothing              → no check
    DECIDE  = auto()   # sets a decision field, no text change      → no check

@dataclass(frozen=True, slots=True)
class ContextPatch:
    kind: TransformKind
    fields: Mapping[str, Any]       # field name -> new value on RequestContext
    rationale: str                  # human-readable, lands in the trace
    evidence: Mapping[str, Any]     # module-specific numbers for the ledger

@dataclass(frozen=True, slots=True)
class ShortCircuit:
    response: str
    served_by: RouteTier            # CACHE_EXACT | CACHE_SEMANTIC | DETERMINISTIC
    evidence: Mapping[str, Any]

@dataclass(frozen=True, slots=True)
class NoOp:
    reason: str                     # "disabled" | "not_applicable" | "no_yield"

Proposal = ContextPatch | ShortCircuit | NoOp
```

**`TransformKind` is the fix for the fidelity-gate problem.** M1 rewriting a sentence is `REWRITE` and every
number, entity and negation in the original must survive. M3 dropping turn 4 is `SELECT` and dropping turn
4's entities is the entire point — the gate instead verifies that the *retained* turns are byte-identical to
their originals. Without this distinction the gate either vetoes M3 permanently or waves M1 through. One
enum field, and the gate becomes a single implementation that is correct for all eight modules.

### 3.3 `Stage` — the only contract a module implements

```python
class Stage(Protocol):
    module_id: str      # "M1"
    name: str           # "compressor"
    reads:  frozenset[str]   # RequestContext fields consumed  → DAG validation
    writes: frozenset[str]   # RequestContext fields produced  → DAG validation

    def applies_to(self, ctx: RequestContext, cfg: ParsimonyConfig) -> bool: ...
    def propose(self, ctx: RequestContext, cfg: ParsimonyConfig) -> Proposal: ...
```

`reads` / `writes` are what let stage order be a config value safely: at startup the orchestrator
topologically validates the configured order and refuses to boot on a violation (e.g. M4 assembly scheduled
before M3 history selection). A misconfigured sweep cell fails in the first second, not after six hours of
CPU time.

---

## 4. The orchestrator

The entire coordination logic, in full. It is intentionally small — that is the point.

```python
def run(ctx: RequestContext, cfg: ParsimonyConfig) -> Outcome:
    traces: list[StageTrace] = []

    for stage in registry.ordered(cfg.stage_order, cfg.enabled_modules):
        t0 = perf_counter_ns()

        if not stage.applies_to(ctx, cfg):
            traces.append(StageTrace.skipped(stage, "not_applicable", t0))
            continue

        proposal = stage.propose(ctx, cfg)

        match proposal:
            case ShortCircuit() as sc:
                traces.append(StageTrace.short_circuit(stage, sc, ctx, t0))
                return Outcome(response=sc.response, traces=traces, generated=False)

            case ContextPatch() as patch:
                candidate = replace(ctx, **patch.fields)
                verdict = fidelity_gate.check(ctx, candidate, patch.kind)
                if verdict.passed:
                    traces.append(StageTrace.applied(stage, patch, ctx, candidate, t0))
                    ctx = candidate                      # commit
                else:
                    traces.append(StageTrace.reverted(stage, patch, verdict, t0))
                                                          # revert == do nothing

            case NoOp() as nop:
                traces.append(StageTrace.noop(stage, nop, t0))

    return generate(ctx, traces, cfg)
```

Five things fall out of this loop for free, none of which any module has to implement:

1. **Ablation** — `cfg.enabled_modules` filters the registry. A disabled module is not constructed, not
   timed, and cannot leak.
2. **Instrumentation** — the trace is produced here, so a module physically cannot forget to log. Token
   deltas are computed by the orchestrator from `ctx` before and `candidate` after.
3. **Fidelity gating** — one call site, one implementation, applied identically to all modules.
4. **Timing** — measured at the boundary, so per-module overhead is real and comparable. This is what
   substantiates the "< 120 ms middleware overhead" metric per module rather than in aggregate.
5. **Short-circuit accounting** — the ledger knows a cache hit served the request without a single model
   token, which is exactly the "tier that answers without a model" claim in Contribution 4.

---

## 5. Recommended stage order (and a deviation from the report)

The report's Figure in §3.4 shows Cache → History → Compressor → Assembler → Budgeter/Router. Two
refinements are recommended, both flagged for the guide's confirmation.

```
  ingest ─▶ preprocess ─▶ [M6a] deterministic tier ──hit──┐
                              │miss                        │
                              ▼                            │
                          [M2] cache lookup ──hit──────────┤
                              │miss                        │
                              ▼                            │
                          [M3] history select              │
                              ▼                            │
                          [M1] compress                    │
                              ▼                            │
                          [M4] assemble (prefix-stable)    │
                              ▼                            │
                      [M5] budget + [M6b] model tier       │
                              ▼                            │
                          generate ◀─────────────────────  │
                              ▼                            │
                          finalise ledger ◀────────────────┘
```

**Refinement A — deterministic tier before the cache.** The report makes "cached answer" router tier 1 and
"deterministic handler" tier 2. But a cache lookup costs an embedding forward pass (~5 ms on CPU) plus an
index search; the deterministic handler is a regex match (~0.1 ms). Checking the cheaper predicate first is
strictly better, and the ordering is observable in the ledger either way. `2+2` should never reach an
embedding model.

**Refinement B — cache lookup position is a config value, not a fixed slot.**

```python
cache_lookup_on: Literal["RAW", "COMPRESSED", "BOTH"] = "RAW"
```

Gap 3 asks whether compressing a query before it reaches the cache raises or lowers the false-hit rate. That
question is unanswerable if the pipeline can only run one order. `BOTH` runs the lookup twice against the
same index and records both outcomes — one request yielding a paired observation, which is a far stronger
statistical design than comparing two independent runs. **This is the single most important extensibility
requirement in the system and it is why stage order is data.**

---

## 6. Repository layout

```
parsimony/
├── pyproject.toml
├── src/parsimony/
│   ├── core/                    # L0 — stdlib only
│   │   ├── types.py             # Turn, RequestContext, Proposal, Invariants
│   │   ├── protocols.py         # Stage, LLMProvider, VectorIndex, Tokenizer, ...
│   │   ├── config.py            # ParsimonyConfig + config_hash
│   │   └── errors.py
│   ├── infra/                   # L1
│   │   ├── tokenization/        # HFTokenizer, token counting, offset mapping
│   │   ├── embedding/           # EmbeddingService (memoised, batched)
│   │   ├── storage/             # JsonlSink, SqliteSink, BlobStore, indexes
│   │   ├── providers/           # MockProvider, OllamaProvider, LlamaCppProvider
│   │   └── nlp/                 # spaCy wrapper, PII detector, sentence splitter
│   ├── modules/                 # L2 — one package per module
│   │   ├── m1_tier1/2/3/  m2_cache/  m3_history/  m4_assembler/
│   │   └── m5_budgeter/    m6_router/ m7_learner/  m8_fidelity/
│   ├── pipeline/                # L3
│   │   ├── orchestrator.py  registry.py  trace.py  policy.py
│   ├── surfaces/                # L4
│   │   ├── cli/  api/  dashboard/  extension/
│   └── eval/                    # L5
│       ├── corpus/  runners/  metrics/  analysis/
├── corpus/                      # frozen, content-hashed, version-controlled
│   ├── conversations.jsonl      # 150 conversations, 6 classes
│   ├── adversarial_pairs.jsonl  # 50 near-duplicate pairs
│   ├── gold.jsonl               # 40 human-written gold answers
│   └── MANIFEST.sha256
├── tests/
│   ├── unit/  contract/  integration/  golden/
└── docs/
```

`tests/contract/` is worth calling out: it holds one test suite that is run against *every* implementation of
a protocol. Any `VectorIndex` must pass the same 20 tests; so must any `LLMProvider`. That is what makes
"swap FAISS for hnswlib without touching the architecture" a checked claim rather than a hope.

---

## 7. Cross-cutting concerns

### 7.1 One embedding per request

The naive design embeds the query in M2 (cache lookup), again in M3 (history relevance), again in M1
(sentence dedup), and again for classification. Four forward passes of MiniLM ≈ 20 ms — a sixth of the entire
overhead budget, spent computing the same 384 floats four times.

`DerivedCache` is a per-request memo hanging off `RequestContext`:

```python
class DerivedCache:
    def embed(self, texts: list[str]) -> np.ndarray: ...  # batched, memoised by text
    def embed_one(self, text: str) -> np.ndarray: ...     # convenience wrapper
    def doc(self) -> spacy.tokens.Doc: ...            # one spaCy pass, reused by M8 and M1
```

Batching turn embeddings matters more than it looks: 12 separate MiniLM calls cost ~60 ms; one batched call
of 12 costs ~12 ms. **Batch boundaries are an architectural concern, not an optimisation to add later** —
retrofitting them means rewriting every module's call sites.

### 7.2 Configuration is experiment identity

```python
@dataclass(frozen=True)
class ParsimonyConfig:
    enabled_modules: frozenset[str]
    stage_order: tuple[str, ...]
    cache_lookup_on: Literal["RAW", "COMPRESSED", "BOTH"]
    compression: CompressionConfig
    cache: CacheConfig
    history: HistoryConfig
    budget: BudgetConfig
    router: RouterConfig
    model: ModelConfig            # name, quantisation, digest
    seed: int

    @cached_property
    def config_hash(self) -> str:  # BLAKE2b of canonical JSON, 16 hex chars
        ...
```

`config_hash` is written into every ledger row. An ablation cell *is* a config hash. This gives three things
at no cost: exact reproduction of any run from its hash; automatic detection of accidental mid-sweep config
drift; and a join key for the ANOVA that cannot silently mismatch. Thresholds live here and nowhere else, so
M7's output is a new `ParsimonyConfig` — the policy learner needs no special integration path.

### 7.3 Error policy

Optimisation modules are, by construction, optional. A module that raises must never fail the request.

| Failure | Handling |
|---|---|
| Module raises | Caught at the orchestrator boundary → treated as `NoOp("error")`, exception recorded in trace, pipeline continues |
| Gate rejects | Patch discarded, `gate_fired` flag set, pipeline continues |
| Provider unreachable | Request fails — this is the one hard error, surfaced to the caller |
| Ledger write fails | Logged to stderr, request continues; ledger loss is preferable to request loss in production, **but the sweep runner treats it as fatal** since the ledger *is* the result |

That last row is a deliberate split: the same code behaves differently under `cfg.mode = SERVE` vs
`cfg.mode = EXPERIMENT`, because the two have opposite priorities.

---

## 8. Online vs offline

M7 is not a pipeline stage and must not be modelled as one. It is a separate offline program with its own
entry point:

```
exported chat logs ──▶ [M7 policy learner] ──▶ PolicyBundle (versioned artefact)
                             │                        │
                     counterfactual replay            │ loaded at startup
                     over the ledger                  ▼
                                              ParsimonyConfig + warm cache
                                                      │
                                                      ▼
                                              online pipeline
```

`PolicyBundle` is a directory containing: a pre-populated cache index, a redundancy lexicon, a persistent
context digest, a template library, and a tuned `ParsimonyConfig`. It is content-hashed and its hash is
recorded in the ledger. Consequence: **"warm-started" vs "cold" is a bundle-presence flag**, so Figure 6's
two curves come from the same code path with one input changed — not from two code paths that could differ
for uninteresting reasons.

---

## 9. Deferred by design

Per the current directive, these are specified but not built yet:

- **Ollama / local model hosting.** `MockProvider` is a deterministic fake — it returns canned responses of
  configurable length with simulated TTFT/TPOT drawn from a fixed distribution. That is *sufficient* to
  build and unit-test the entire pipeline including the budgeter's early-stop rule, and it makes CI fast
  and hermetic. `OllamaProvider` is a ~60-line class implementing the same protocol, added in Sprint 2.
- **Browser extension.** Depends on a frozen core; scheduled last.
- **Neural compressor (LLMLingua-2).** Specified as an *ablation arm* to measure, not as the default — see
  ADR-009.

---

## 10. Open questions for the guide

1. **Refinement A** (deterministic tier before cache) reorders the report's Figure §3.4. Confirm acceptable.
2. **Seeds at temperature 0.** The report specifies 5 seeds at temperature 0. At temperature 0 decoding is
   greedy, so the seed does not change the output — the five repeats measure *latency* variance, not answer
   variance. That is a legitimate and useful thing to measure, but the report's phrasing implies output
   variance. Recommend stating this explicitly, and adding one temperature-0.7 arm if answer variance is
   wanted.
3. **Embedding model is a confound for Gap 3.** The false-hit-rate result depends on the geometry of the
   embedding space. A conclusion drawn only with MiniLM-L6-v2 is a conclusion about MiniLM. Recommend
   running the Gap 3 sweep on two embedding models (MiniLM + bge-small-en-v1.5) so the finding is about
   compression rather than about one encoder. Cost: one extra sweep of a small subset.
