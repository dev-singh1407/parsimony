# Parsimony — Architecture Decision Log

Each ADR: **context · options · decision · justification (scalability / maintainability / extensibility) ·
consequences**. ADRs are append-only; a superseded ADR is marked, never edited.

---

### ADR-001 — Modules propose, the orchestrator commits

**Options.** (a) Modules mutate a shared mutable context. (b) Modules return new contexts. (c) Modules return
*proposals*; orchestrator applies them.

**Decision — (c).** `RequestContext` is frozen; a stage returns `ContextPatch | ShortCircuit | NoOp`; the
orchestrator builds a candidate with `dataclasses.replace`, gates it, and commits or discards.

**Justification.** Five requirements collapse into one loop: ablation (skip the stage), instrumentation
(trace at the boundary), fidelity gating (one call site), timing (measured externally), revert (don't
assign). Under (a), each module must implement its own rollback and its own logging, and each will do it
slightly differently — the classic way an ablation study ends up comparing implementation quirks rather than
techniques. Under (b), nothing prevents a module from returning a context that quietly changed a field it
never declared.

**Consequences.** Modules are pure functions of `(ctx, cfg)` and are trivially unit-testable with no mocks.
The orchestrator is ~40 lines and is the only place coordination logic exists.

---

### ADR-002 — Stage order is configuration, validated against a read/write DAG

**Context.** Gap 3 asks whether compressing a query before it reaches the cache raises or lowers the
false-hit rate. That is unanswerable with a hardcoded order.

**Decision.** `cfg.stage_order: tuple[str,...]` plus `cfg.cache_lookup_on ∈ {RAW, COMPRESSED, BOTH}`. Each
stage declares `reads`/`writes`; the registry topologically validates the configured order at boot and
refuses to start on a violation.

**Justification.** The project's central research questions are *about* module interactions. An architecture
that fixes the interaction pattern in code cannot answer them. Boot-time validation means a bad sweep cell
fails in the first second rather than after six unattended CPU-hours — which, on a CPU-bound project with a
hard deadline, is the difference between one sweep and two.

**Consequences.** `BOTH` yields *paired* observations from one request — statistically much stronger than
comparing two independent runs, and roughly half the CPU cost.

---

### ADR-003 — `TransformKind` scopes the fidelity gate

**Context.** M1 must never drop a number. M3 drops entire turns as its whole purpose. A uniform "did the text
change" gate either blocks M3 permanently or waves M1 through.

**Decision.** Every patch declares `REWRITE | SELECT | AUGMENT | DECIDE`. `REWRITE` gets the full invariant
check; `SELECT` verifies retained units are byte-identical; the rest are unchecked.

**Justification.** One enum field turns an impossible requirement into one 12-line function that is correct
for all eight modules. The alternative — per-module gate configuration — puts safety policy in eight places
and guarantees drift.

**Consequences.** New modules must classify their transformation honestly. A contract test asserts every
`Stage` implementation returns a patch whose declared kind matches an observed-behaviour probe.

---

### ADR-004 — Exact brute-force vector search is the default; ANN is optional

**Context.** The report specifies FAISS. Cache size is order 10³–10⁵.

**Decision.** `ExactIndex` (numpy) by default. `FaissIndex` / `HnswIndex` behind the same protocol, for a
scaling demonstration only.

**Justification — this is the most consequential storage decision.** At N = 5 000 an exact search is a
(5000, 384) matmul, well under 0.2 ms; at N = 100 000, ~3 ms. FAISS's IVF/HNSW indexes are *approximate*:
recall < 1.0. The headline Gap 3 deliverable is the **false-cache-hit rate**. With an approximate index that
metric becomes a mixture of "the similarity policy was wrong" and "the index missed the true nearest
neighbour" — two causes that cannot be separated after the fact.

**Measured, not assumed (enforcement pass).** The paragraph above was an argument. Asserting a claim about a
component we never built is exactly what this project criticises elsewhere, so `LshIndex` was implemented
(random-projection LSH, numpy only) and the sweep re-run on the adversarial subset with the verifier off, so
the index is the only thing separating the pairs:

| index | false-hit rate | true-hit rate |
|---|---|---|
| `ExactIndex` | **84.4%** | 52.4% |
| `LshIndex` (approximate) | **46.7%** | 28.6% |

**The approximate index makes the cache look roughly twice as safe as it is.** It is not safer — it is worse
at retrieval in *both* directions (true hits fall too). It simply fails to fetch the dangerous neighbour, so
the danger is never counted. Had this project used FAISS, the headline safety number would have been
understated by **37.7 percentage points**, and the error would have flattered the result.

**Consequences.**

- One fewer heavy native dependency on the critical path, and simpler installation on Windows.
- `CalibrationPoint.is_safe` now **raises** `ApproximateIndexError` rather than returning a number computed
  from an approximate index. A guard that returns a plausible-looking figure is worse than no guard.
- `LshIndex` runs in the same `VectorIndexContract` suite as `ExactIndex`, so the swappability claim is
  checked, and `is_exact()` is contract-tested for truthfulness because the analysis layer trusts it.
- Generalises beyond this project: any paper reporting a false-hit rate over an ANN index is reporting a
  number that partly measures its index. That is worth stating in the related-work discussion.

---

### ADR-029 — The exact-hash tier has no verifier, so its canonicalisation must be lossless

**Context.** Probing degenerate inputs found a live collision class. Canonicalisation lowercases, collapses
whitespace and strips trailing punctuation — lossless for real queries, so that "what is X?" and "what is X"
share a key. But applied to inputs that are *only* punctuation or whitespace it destroys everything:

| query | canonical form | key |
|---|---|---|
| `"   \n\t  "` | `""` | `207fbd06116dbfdd` |
| `"?!..."` | `""` | `207fbd06116dbfdd` |
| `"!!!"` | `""` | `207fbd06116dbfdd` |
| `"."` | `""` | `207fbd06116dbfdd` |
| `"???"` | `""` | `207fbd06116dbfdd` |
| `""` | `""` | `207fbd06116dbfdd` |

Six distinct inputs, one key. In a live pipeline they served one another's cached answers.

**The structural point, which matters more than the bug.** The three-zone verifier — the mechanism ADR-027
shows is what actually makes the cache safe — **only guards the semantic tier**. The exact-hash tier
short-circuits before it, on the assumption that hash equality implies semantic equality. That assumption
holds exactly as long as canonicalisation is lossless, and nothing was enforcing it.

**Decision.** A query is cacheable only if its canonical form contains at least one alphanumeric character.
`store()` and `lookup()` both refuse otherwise, and the refusal is counted in `CacheStats.uncacheable`.

**Justification.** A canonical form carrying no information cannot key anything: whatever is stored under it
becomes the answer to every future degenerate query. Refusing is the only correct behaviour — there is no
threshold or verifier that rescues an empty key.

**Relation to the literature.** The report's Paper 7 (*Key Collision Attack on LLM Semantic Caching*, 2026)
searches for adversarial suffixes that force false-positive hits. This collision class needs no adversarial
search at all — it is reachable by typing "?" — and it lives in the tier that paper's threat model does not
examine, because the exact tier looks unambiguously safe. Worth a sentence in the related-work discussion:
**hash-tier canonicalisation is an unexamined attack surface, and its safety is a property of the
normalisation function rather than of the cache policy.**

**Consequences.**

- Real repeats are unaffected; only information-free queries are refused, and `uncacheable` makes the rate
  visible rather than silent.
- Any future addition to `canonicalise()` — stemming, stop-word removal, aggressive Unicode folding — widens
  the collision class and must be weighed against this, because the exact tier will not catch it.

---

### ADR-030 — Negative yield is a *position-0 boundary* effect, and tier 1 is where it bites

**Context.** ADR-026 concluded from 495 single-word deletions that whitespace-aligned edits are monotone.
Re-run on the expanded corpus (2,199 deletions), **three raise the token count** — and all three have the
same shape: removal of the *first* word.

```
"What happened in the 1970s?"   11 tokens
     "happened in the 1970s?"   12 tokens   (+1)
```

**There are TWO distinct position-0 effects.** The first draft of this ADR conflated them and attributed the
tier-1 case to the wrong one; both are confirmed directly against the tokenizer.

**Effect 1 — loss of the leading-space form.** This is what makes first-word *deletion* non-monotone.

| string | tokens |
|---|---|
| `" happened"` | **1** — `[' happened']` |
| `"happened"` | **3** — `['h', 'app', 'ened']` |

It is real but **not universal**: common words carry a standalone token too, so `" explain"` and `"explain"`
are both 1. Mid-string deletions keep the space and remain monotone, exactly as ADR-026 found.

**Effect 2 — capitalisation at position 0.** This is what makes M1 **tier 1** fail to pay.

| string | tokens |
|---|---|
| `"explain"` | **1** — `['explain']` |
| `"Explain"` | **2** — `['Ex', 'plain']` |

> **Scoped by ADR-032.** This second mechanism is **vocabulary-specific**. Re-measured on GPT-2,
> `"explain"` and `"Explain"` both cost 2 tokens — the capitalisation penalty does not exist there. Effect 1
> (leading space) transfers to both. So the *guard* is general; this particular *mechanism* is not.

Decomposed:

```
"Please explain recursion."   4   ['Please', ' explain', ' recursion', '.']
"Explain recursion."          4   ['Ex',     'plain',    ' recursion', '.']   <- capital costs +1
"explain recursion."          3   ['explain',            ' recursion', '.']   <- would have paid
```

Tier 1 strips leading politeness and then **re-capitalises the new opener** — and the capital hands the
saving straight back. Measured on representative prompts, **5 of 9 leading-word removals saved zero tokens**.

**The re-capitalisation is kept deliberately.** Dropping it would recover the token but leave the user's text
starting in lowercase, which is a visible corruption of their input — and tier 1's entire claim is to be
lossless. The correct resolution is not to stop capitalising but to *decline the edit when it does not pay*,
which is what the guard now does.

**Decision.** Apply negative-yield detection to **tier 1**, not only tier 3. Tier 1 now measures the payload
before and after and returns `NoOp` when the edit does not pay.

**Justification.** ADR-026 established that the guard's real value is rejecting **zero-yield** edits, which
perturb text for no saving and can therefore only lose meaning. It then left that guard in tier 3, the tier
least exposed to it. Tier 1 is the tier that vacates position 0 on nearly every polite prompt.

**Measured effect.** Over the full corpus, tier 1 previously applied 18 edits saving 48 tokens net. With the
guard: 12 edits, 49 tokens, **6 rejected as zero-yield** — a third of tier 1's edits were perturbing the
user's text for zero or negative benefit, and removing them slightly *improved* the total.

**Why this strengthens the report.** The report presents negative-yield detection as a refinement of tier 3
rewriting. It is more general and more mechanistic than that:

- the effect is a **position-0 boundary** phenomenon, not a neighbour-merge one;
- it is triggered by the *simplest* transformation in the stack, not the cleverest;
- it therefore applies to **any** method that strips sentence openers — including the stop-word and
  discourse-marker deletion that prompt-compression baselines routinely perform.

**Methodological note.** ADR-026's claim was true of its sample and false in general. It was caught only
because the corpus grew and the probe was re-run. Every empirical claim in this project should be re-checked
against the final corpus before the report is written.

---

### ADR-031 — Bounded memory: the cache needs LRU eviction

**Context.** Probed under sustained load, nothing in the pipeline ever evicted:

| after 400 distinct queries | entries |
|---|---|
| cache entries | 400 |
| conversations tracked for prefix survival | 400 |
| blob store entries | 415 |

All three grow linearly with traffic and are never released. Report §4.7 targets an ordinary consumer CPU
with **8 GB of RAM**, and §4.4 describes the cache as "the one component with a memory" — so this was an
unbounded leak in precisely the component designed to accumulate.

**Decision.** `CacheConfig.max_entries = 10_000` with LRU eviction; conversation tracking bounded to the 256
most recent.

**Justification.** 10k entries is roughly 15 MB of vectors at 384 float32 plus stored text — affordable and,
more importantly, *bounded*. The conversation map only ever compares against the immediately preceding turn
of the same conversation, so retaining older ones buys nothing at all.

**Two details that are easy to get wrong, both caught by tests.**

1. **The vector index must be evicted with the entry.** An orphaned vector keeps scoring in `search()` and
   returns an `entry_id` that no longer resolves — a silent miss that still costs the similarity
   computation, and a slow leak of a different kind.
2. **`touch()` is public because semantic hits are served from `search()`**, which the stage drives, not from
   `lookup()`. Refreshing recency only on exact hits would leave LRU blind to half its traffic, so a
   heavily-used paraphrase entry could be evicted while a stale exact-matched one survived.

**Consequences.**

- The default cap is asserted to exceed 4× the corpus request count, so eviction never fires during a normal
  sweep. A cap that evicted mid-run would silently depress the hit rate and corrupt every M2 result — the
  measurement must not be perturbed by the memory bound.
- `CacheStats.evicted` makes the rate visible, so if a future workload does hit the cap it shows up in the
  ledger rather than as an unexplained drop in cache performance.
- The blob store remains unbounded by design: in `EXPERIMENT` mode it is the ledger's content-addressed
  substrate and losing entries would lose results. It is disk-backed in that mode, and a `SERVE`-mode
  deployment should use `BlobStore` rather than `MemoryBlobStore`.

---

### ADR-032 — Cross-vocabulary generalisation: ratios transfer, mechanisms do not

**Context.** Report §4.6 requires re-running the winning configuration on other models "without re-tuning,
which measures directly whether a calibration transfers" — Gap 5 and Contribution 6. Three LLMs need Ollama,
which is out of scope. But the **tokenizer** is the part of the configuration that determines nearly
everything measured here: every token count, M1 tier 3's negative-yield decisions, tier 1's position-0
behaviour, and M4's prefix survival. That dimension is fully answerable today with two real vocabularies —
Qwen2.5 (151,665) and GPT-2 (50,257).

Same cells, same corpus, **no re-tuning**.

**Result 1 — reduction ratios transfer almost exactly.**

| cell | Qwen2.5 | GPT-2 |
|---|---|---|
| M1 | 0.16% | 0.16% |
| M2 | 1.72% | 1.72% |
| M3 | 12.84% | 12.84% |
| M5 | 14.27% | 14.24% |
| M1+M2+M3+M5 | 26.46% | 26.37% |

Module ranking is identical. This is expected once stated: a reduction is a *ratio*, and if a vocabulary
counts all text at roughly a constant factor the ratio cancels. Absolute counts differ substantially —
"Please explain recursion." is 4 tokens under Qwen and 5 under GPT-2 — while the percentages do not.

**That is a genuinely useful result for the report:** it means the headline reduction figures are more
portable than the report assumes, and the thing that does *not* port is the calibration underneath them.

**Result 2 — one of ADR-030's two mechanisms does not transfer.**

| claim | Qwen2.5 | GPT-2 | transfers |
|---|---|---|---|
| `" happened"` cheaper than `"happened"` | 1 vs 3 | 1 vs 3 | **yes** |
| `"explain"` cheaper than `"Explain"` | 1 vs 2 | **2 vs 2** | **no** |
| first-word deletion can raise the count | 11 vs 12 | 7 vs 8 | **yes** |

GPT-2 has no capitalisation penalty for that word. ADR-030 stated the capitalisation mechanism as though it
were general; it is Qwen-specific. Corrected there.

**Result 3 — the tier-1 zero-yield rate is identical anyway: 7/20 (35%) under both.** Different mechanisms,
same net rate on this corpus. Worth reporting precisely because it would be easy to present the matching
rate as evidence the mechanisms match, and they do not.

**Decision.** Keep the guard (it is general), scope the mechanism claim (it is not), and report the
generalisation study as covering the *tokenizer* dimension explicitly — not the model dimension.

**What this does NOT cover, stated plainly.** Decode speed, answer quality and quantisation are properties of
the model, not the tokenizer. The tokenizer arm of §4.6 is done; the model arm still needs Ollama. Claiming
otherwise would be the same category of overreach this ADR just corrected in ADR-030.

---

### ADR-005 — Dual ledger sinks: JSONL for experiments, SQLite for serving

**Context.** The report specifies SQLite. The sweep is 16 cells × 5 seeds × 3 models × 150 conversations
across parallel worker processes.

**Decision.** `LedgerSink` protocol. `JsonlSink` (append-only, one file per worker) in `EXPERIMENT` mode;
`SqliteSink` (WAL) in `SERVE` mode. `parsimony ledger import` folds JSONL into the analysis DB.

**Justification.** SQLite allows one writer at a time; parallel workers hit `SQLITE_BUSY` and either lose rows
or serialise. Across a 6+ hour unattended run that means a corrupted or partial ledger — and **the ledger is
the entire result**. JSONL has no contention, is crash-safe to the last complete line, and costs a 30-second
import. The dashboard genuinely does need live SQL, so both sinks earn their place.

**Consequences.** Ledger loss is non-fatal in `SERVE` (request continues) and fatal in `EXPERIMENT` (run
aborts). Same code, opposite priorities, one config flag.

---

### ADR-006 — Compute derived values once, batch by default

**Decision.** `DerivedCache` on `RequestContext` memoises the query embedding, batched turn embeddings and
the spaCy `Doc`, with lazy properties.

**Justification.** The naive design embeds the query four times (M2, M3, M1, classifier) ≈ 20 ms, a sixth of
the whole 120 ms budget spent recomputing 384 floats. Twelve separate turn embeddings cost ~60 ms; one
batched call costs ~12 ms. **Batch boundaries are architectural** — retrofitting them means changing every
module's call site. Laziness additionally means a request short-circuited by M6a tier 0 never pays for spaCy
or embeddings at all.

**Consequences.** Modules must go through `ctx.derived`, never call the embedder directly. Enforced by an
import-linter rule barring `modules/*` from importing `infra/embedding` directly.

---

### ADR-007 — Provider abstraction, `MockProvider` first, Ollama deferred

**Decision.** `LLMProvider` protocol with a streaming `generate()`. `MockProvider` (deterministic canned
responses, simulated TTFT/TPOT) is built first; `OllamaProvider` lands in Sprint 2.

**Justification.** Honours the current directive to defer model hosting *while still building everything*.
The orchestrator, ledger timing fields, budgeter early-stop rule and the whole ablation harness are fully
testable against a fake. CI stays hermetic and fast — a test suite that requires a running Ollama is a test
suite that stops being run. `OllamaProvider` is then ~60 lines against an interface that already has contract
tests.

**Consequences.** All Sprint 0–1 latency numbers are *simulated* and must be labelled as pipeline-correctness
evidence, not performance evidence — including at the mid-August review. Say so on the slide.

---

### ADR-008 — The config object is the experiment's identity

**Decision.** One frozen `ParsimonyConfig` holds every threshold, every enable flag, the stage order, the
model identity and the seed. `config_hash` = BLAKE2b of its canonical JSON, written into every ledger row.

**Justification.** An ablation cell *is* a config hash. This gives exact reproduction from a hash, automatic
detection of mid-sweep config drift, and a join key for the ANOVA that cannot silently mismatch. It also
makes M7's integration trivial: the policy learner emits a config, and the online system loads it — no
special path, no seven-way integration.

**Consequences.** No module may hold a hard-coded threshold. Enforced by review and by a lint rule flagging
float literals in `modules/`.

---

### ADR-009 — Rule/lexicon tier-3 compression by default; the neural compressor is a *measured arm*

**Options.** (a) hand-authored equivalence lexicon; (b) LLMLingua-2 (280 M XLM-RoBERTa token classifier);
(c) LLMLingua (perplexity scoring with a small autoregressive LM).

**Decision.** (a) as default. (b) implemented in Sprint 5 behind the same interface, **as an arm to
measure**. (c) rejected.

**Justification.** (c) is disqualified on the project's own terms — an autoregressive scorer on CPU competes
for exactly the cycles it is meant to save, which is the limitation the report identifies in Paper 1. (b) is
a 280 M-param encoder forward pass, plausibly 50–200 ms on an i5, likely exceeding the entire 120 ms overhead
budget alone. But *assuming* that is weak; **measuring** it is a contribution. A table showing that the
state-of-the-art compressor costs more CPU time than it saves at 1B scale directly substantiates the
report's critique of the literature and is a genuinely publishable negative result.

**Consequences.** (a)'s ceiling is lower and it is English/domain-specific. Stated as a limitation, partially
answered by M7 mining the lexicon from real usage.

---

### ADR-010 — Rolling summarisation runs off the critical path

**Decision.** M3's summarisation strategy summarises *after* the response returns, for the next turn. Token
cost charged to the ledger; latency not charged to the user.

**Justification.** Summarisation is a model call — synchronously it would dominate the latency it is meant to
reduce and would make M3 look strictly worse than a fixed window. Placement must be decided now; making it
synchronous first and async later means rewriting M3's interface and re-running every M3 cell.

**Consequences.** Needs a background task runner and a "digest is one turn stale" semantic, which must be
documented in the results.

---

### ADR-011 — Prefix survival is measured in tokens, not bytes

**Decision.** Longest common **token** prefix via tokenizer offset mapping.

**Justification.** llama.cpp reuses the KV cache at token granularity. A byte-level measure over-reports
across multi-byte boundaries and under-reports when a byte edit leaves the token sequence unchanged. Since
prefix survival is Contribution 3's headline number, measuring it in the wrong unit undermines the
contribution.

**Consequences.** Requires tokenizer offset mapping — available in HF `tokenizers`, another reason to prefer
it over `tiktoken`.

---

### ADR-012 — Classification via a logistic head on the shared embedding, rules as override

**Options.** (a) hand-written rules (report's choice for M5); (b) logistic regression on the query embedding;
(c) LLM-as-classifier.

**Decision.** (b), with (a) as a high-precision override for arithmetic/unit/date patterns.

**Justification.** The embedding is already computed for the cache lookup, so a 384→6 logistic head costs
~5 µs — effectively free, and it is trained on the corpus labels we are authoring anyway. Rules alone are
brittle across six classes. (c) costs a model call to save model calls, which is self-defeating. The rule
override is retained because arithmetic detection must be *precise* — routing a non-arithmetic query to
tier 0 produces a confidently wrong answer.

**Consequences.** Needs labelled corpus data — but the corpus is already labelled by class, so the training
set is free.

---

### ADR-013 — PII detected at preprocess, redacted at write boundaries

**Decision.** Spans detected once in Stage 2; redaction applied by M8 immediately before any cache write or
ledger blob write.

**Justification.** The model *should* see the user's real text — redacting before generation would degrade
answers for no privacy gain on a fully local system. Persistent stores should *never* see it, because the
cache is the one component with memory and the ledger is the one artefact that ships with the report.
Detecting early and redacting late satisfies both.

**Consequences.** Cache hits are matched on redacted keys, which slightly changes matching semantics. Must be
documented and its effect on hit rate reported.

---

### ADR-014 — Additive-only, forward-only ledger schema

**Decision.** New columns nullable; never rename, never drop; `schema_version` travels with every run.

**Justification.** The ledger schema will change as M4–M7 land. Sprint 0's baseline runs must still be
readable in October, because every claim in the report is a difference against that baseline. A destructive
migration in week 9 silently invalidates weeks 1–8.

---

### ADR-015 — Corpus frozen and content-hashed before the first sweep

**Decision.** `corpus/MANIFEST.sha256` is committed and its hash recorded in every ledger row. Gold answers
are written before any system output is inspected.

**Justification.** The report's own risk register anticipates the criticism that a self-authored corpus is
tuned to the system. A committed hash with a git timestamp predating the sweep is the only real answer to
that, and it costs nothing if done now — and is impossible to retrofit later.

---

### ADR-016 — Deterministic router tier runs *before* the cache

**Context.** The report's §3.4 figure orders cache first; §4.4 lists cache as router tier 1 and the
deterministic handler as tier 2.

**Decision.** Deterministic tier (M6a) runs first.

**Justification.** A cache lookup costs an embedding forward pass (~5 ms) plus a search; the deterministic
handler is a regex match (~0.1 ms). Checking the cheaper predicate first is strictly better and changes no
result — the ledger records which tier served the request either way. `2+2` should never reach an embedding
model.

**Consequences.** A deviation from the report's figure. Flagged for the guide's confirmation; trivially
reversible since order is configuration (ADR-002).

---

### ADR-017 — Seeds at temperature 0 measure latency variance, not answer variance

**Context.** The report specifies five seeds at temperature 0.

**Observation.** At temperature 0 decoding is greedy, so the seed does not change the output. The five
repeats measure *timing* variance — genuinely useful on CPU, where thermal state and scheduling produce real
spread — but not answer variance.

**Decision.** Keep five seeds; **relabel** them in the report as repeats for latency confidence intervals.
Add one temperature-0.7 arm on a corpus subset if answer variance is wanted.

**Justification.** A reviewer who notices this will question the whole statistical treatment. Stating it
first converts a vulnerability into evidence of rigour. (Note: llama.cpp can exhibit minor non-determinism
from threaded reductions, so a small amount of output variance may appear — worth measuring and reporting
rather than assuming either way.)

---

### ADR-018 — The embedding model is a confound for Gap 3

**Observation.** The false-hit rate is a property of the embedding geometry. A result from MiniLM-L6-v2 alone
is a result *about MiniLM-L6-v2*.

**Decision.** Run the Gap 3 threshold sweep on two encoders — MiniLM-L6-v2 and bge-small-en-v1.5 — on the
50-pair adversarial subset only.

**Justification.** Cost is small (the adversarial subset is 100 queries, and the sweep is an offline
re-analysis of persisted scores per ADR/Stage 6). The payoff is that Contribution 2 becomes a claim about
compression rather than about one encoder. This is the difference between "we measured a number" and "we
measured an effect".

---

### ADR-019 — Generation memoisation in EXPERIMENT mode

**Context.** The report's §4.6 protocol costs an estimated ~36 days of continuous CPU (arithmetic in
`05-evaluation-harness.md` §1). The runway allows ~10 days of sweep time.

**Decision.** In `EXPERIMENT` mode, memoise model outputs on
`(prompt_sha256, model_digest, gen_params_hash)`.

**Justification.** At temperature 0 decoding is greedy, so identical prompt bytes and an identical pinned
model digest produce identical output. The memo therefore returns exactly what the model would have
returned — it is bit-exact, not an approximation, and cannot bias any result. Across 16 cells the prompt
repetition rate is high: cells differing only in M2's flag produce byte-identical prompts on every cache
miss. Architecturally free, since the content-addressed blob store already exists.

**Consequences.** The memo destroys latency measurement (50 µs vs 10 s). Handled by ADR-020, and made
auditable by the `generation_memoised` ledger field, which every latency analysis filters on.

---

### ADR-020 — The sweep splits into a quality pass and a timing pass

**Decision.** Two passes with different settings: quality (memo ON, full corpus, 1 repeat) and timing
(memo OFF, 50-conversation stratified subset, 2 repeats, `n_workers = 1`). Every ledger row carries
`pass_kind`; every figure states which pass it came from.

**Justification.** Token counts, quality scores, cache/gate/router behaviour are all *deterministic
functions of the input* at temperature 0 — one repeat is mathematically sufficient, and five would be five
identical numbers reported as a confidence interval, which is worse than useless. Latency is the only
genuinely stochastic quantity, and it gets a dedicated unmemoised pass at one worker, because running
parallel workers against a single CPU-saturating Ollama instance measures contention rather than the system.

**Consequences.** Confidence intervals for latency come from the ~200 within-pass request observations via
bootstrap, not from repeat count (ADR-017 already established that repeats at temperature 0 measure timing).
Supersedes the "five seeds" reading of report §4.6 while preserving its intent.

---

### ADR-021 — Report effect size as the headline, p-values as a footnote

**Context.** The ANOVA will have ~600 observations per cell.

**Decision.** Report partial η² for every main effect and interaction, with p as a footnote. Report the
additivity shortfall — `Σ individual_reductions − measured_stacked_reduction` — with a bootstrap CI, as a
standalone table.

**Justification.** At n = 600 everything is significant; p-values will carry no information. The project's
central claim (Contribution 1, Figure 1) is about the *magnitude* by which stacked savings fall short of
additive, so effect size is the natural unit. Reporting p alone would make the headline result directional
when it can be quantitative.

---

### ADR-022 — Model-as-judge: pairwise, position-swapped, never a model under test

**Decision.** The judge is the 3B escalation model or a separate model, never one being evaluated. Pairwise
comparison, not absolute scoring. Every comparison run A/B and B/A; disagreement counts as a tie. The
swap-disagreement rate is reported. Judge prompt is version-controlled and its hash is in the ledger.

**Justification.** Using Llama 1B to judge Llama 1B's output measures self-preference, not quality. Small
models cannot produce calibrated 1–10 scores but can pick between two answers. LLM judges have a documented
position bias, and position-swapping is the standard control. Reporting the swap-disagreement rate is the
important part: a high rate means the judge is noise, and knowing the metric is unreliable is worth more
than the metric.

**Consequences.** Doubles judge inference cost. Judging runs offline over stored responses, so it does not
touch the timing pass.

---

### ADR-023 — Bootstrap confidence intervals, not t-intervals

**Decision.** 10 000-resample percentile bootstrap. Report medians and IQR alongside means.

**Justification.** CPU latency distributions are right-skewed with a long tail — thermal throttling and
scheduler effects produce occasional large outliers. A symmetric t-interval on a skewed distribution
understates the upper tail, which is precisely the region that matters for a latency claim.

---

### ADR-024 — There is no safe `tau_hi` on similarity alone; the verifier does the work

**Context.** Measured on our own sentence pairs under the default encoder (`hashing-v1`):

| cosine | pair |
|---|---|
| **0.924** | "Is it safe to mix bleach and vinegar?" / "Is it **not** safe to mix bleach and vinegar?" |
| 0.869 | "capital of **Australia**" / "capital of **Austria**" |
| 0.850 | "capital of France" / "capital **city** of France" |
| 0.833 | "Explain recursion." / "**Please** explain recursion." |
| 0.653 | "What causes rain?" / "What causes rainfall to occur?" |
| −0.026 | unrelated pair |

**The adversarial negation pair is the most similar pair in the set** — above every genuine paraphrase. The
thresholds quoted as safe across the caching literature (0.85–0.92) would auto-accept it as a hit and return
the opposite answer, with the verifier never running.

**Decision.** `tau_hi = 0.97`, `tau_lo = 0.75`. Almost nothing auto-accepts on similarity alone; the wide
verify zone routes borderline matches to the cheap invariant check (numbers, entities, negations, lexical
overlap).

**Justification.** This is not a workaround for a weak encoder — it is the mechanism the three-zone design
exists for, and the measurement says the discriminating signal is simply not in the vector geometry.
Modifiers and negations are *low-magnitude* edits in any bag-of-features space while being *total* inversions
of meaning. A stronger encoder narrows the gap but does not close it: MiniLM also places negation pairs high,
which is precisely why the key-collision attack literature exists.

**Consequences.**

- Directly substantiates Gap 5: published safe settings do **not** transfer, and here they fail in the
  dangerous direction rather than merely the inefficient one.
- Every threshold in the system is now annotated with the scorer it was calibrated against. Swapping the
  encoder invalidates them — which is Contribution 6 (a calibration table, not a universal number)
  demonstrated on our own stack rather than asserted about someone else's.
- The verify zone carrying most traffic makes verifier cost a first-class concern, and is why it must stay
  a set comparison rather than a second neural pass.

---

### ADR-025 — Position-aware placement and KV prefix reuse are in direct conflict, and the conflict is invisible to token counts

**Context.** M3's arrangement step moves the most query-relevant retained turn adjacent to the query,
exploiting the start/end attention bias documented in *Lost in the Middle*. M4 pins an invariant zone to the
prompt head so the KV prefix survives. Measured over a 5-turn conversation:

| configuration | mean prefix tokens reused | input tokens sent |
|---|---|---|
| M4 off (volatile head) | 2.4 | — |
| M4 on, chronological | **55.8** | 626 |
| M4 on, position-aware | **10.4** | 614 |

**The two M4-on arms retain the same turns; only the order differs.** Position-aware placement costs 81% of
the prefix reuse while sending 2% *fewer* tokens — so a token-counting metric scores it as the better
configuration.

> Figures re-measured after the ADR-029 cache fix and the ADR-030 tier-1 guard; the earlier draft recorded
> 94.4/47.6. The direction and magnitude of the effect are unchanged, and it is now larger. The 626/614 gap
> is a MockProvider artefact — its response length depends on the prompt hash, so reordering perturbs
> downstream history lengths; with a real model the counts would be identical.

**Decision.** Keep both arrangements as independently ablatable options rather than picking one. Report
prefix survival alongside token count for every arrangement.

**Justification.** This is Contribution 3 reproduced inside our own stack, and it is a sharper example than
the compression case the report anticipates: compression at least *shows up* as a token reduction, so a
practitioner sees a trade. Reordering shows up as nothing at all. Every metric in the compression literature
would score these two configurations identically, and one of them is roughly twice as expensive to prefill.

**Consequences.**

- The recommended default (`position_aware`) is not obviously correct. Which arm wins depends on whether the
  attention benefit outweighs the re-prefill cost, and that is a *measurement*, not a preference — it needs
  the real-model latency pass to settle, since with a mock provider the prefill cost is synthetic.
- Strengthens the case for `arrangement` being config rather than code: the answer may differ per model and
  per conversation length, which is another row in the calibration table.
- A follow-up worth running: a "position-aware-once" arrangement that fixes the order on first computation
  and keeps it stable thereafter, capturing most of the attention benefit at a fraction of the prefix cost.

---

### ADR-026 — Negative yield is real, but not in the regime phrase compression operates in

**Context.** The report states that "deleting a word can raise the token count by breaking a merge in its
neighbours", and negative-yield detection is presented as the novel detail of M1 tier 3. That is an
assumption underpinning a headline contribution, so it was measured rather than trusted
(`parsimony tokenprobe`, Qwen2.5-1.5B tokenizer).

| edit regime | tested | saved tokens | saved nothing | **cost** tokens |
|---|---|---|---|---|
| phrase substitution | 24 | 24 | 0 | 0 |
| single-word deletion | 495 | 491 | 4 | **0** |
| sub-token edit | 8 | 0 | 5 | **3** |

**Whitespace-aligned edits are monotone *mid-string*.** Modern BPE encodes a leading space into the token
(`" word"`), so deleting a word from the middle removes exactly its tokens and leaves the merges on either
side intact.

> **Superseded in part by ADR-030.** The original sample (495 deletions) contained no counterexample and
> this ADR claimed monotonicity outright. On the expanded corpus (2,199 deletions) **three deletions raise
> the count** — all of them removals of the *first* word. See ADR-030 for the mechanism; the mid-string
> claim stands.

**Sub-token edits are not monotone.** `"running" → "runing"` is 2 tokens → **3**. `"unbelievable" →
"unbelievble"` is 4 → **5**. Shorter text, more tokens.

**Decision.** Keep the check, and reframe what it is for. It rejects `delta >= 0`, not merely `delta > 0`.

**Justification.** The check's real value turned out to be rejecting **zero-yield** edits, which are far more
common than true negative-yield ones: 5 of 8 sub-token edits shortened the text and saved nothing at all,
and "in order to" → "inorder to" and "New York" → "NewYork" both cost 0 tokens. An edit that perturbs the
text without saving a token is pure risk — it can only lose meaning — and no published method checks for it
because they all count characters or words rather than the target model's tokens.

**Consequences for the report.** The claim must be stated precisely rather than broadly, and the precise
version is more useful:

- For **phrase- and word-level** compression (what Parsimony does, and what most practical systems do)
  shortening is safe and the guard is cheap insurance costing one re-tokenisation per candidate.
- For **character- or subword-level** compression — which includes perplexity-scored token dropping of the
  LLMLingua family — the guard is essential, because that is exactly the regime where merges break.

This turns an assumption into a measured, bounded claim, and it identifies precisely which prior methods are
exposed to the failure. That is a stronger contribution than the unqualified version.

---

### ADR-027 — The verifier needs an operative-modifier check, and the threshold is not what makes the cache safe

**Context.** With the adversarial subset authored (45 adversarial pairs + 5 controls), the first calibration
sweep gave a **26.7% false-hit rate that did not improve above τ_hi = 0.95**. Since raising the threshold
stopped helping, the verifier — not the similarity cutoff — was the binding constraint. The per-class
breakdown localised it exactly:

| operative | false-hit rate, initial | after this ADR |
|---|---|---|
| number | 0% | 0% |
| entity | 11% | **0%** |
| negation | 31% | **0%** |
| **modifier** | **78%** | **0%** |
| **overall** | **26.7%** | **0.0%** |

**Three gaps, each found by inspecting the surviving false hits rather than by guessing.**

1. **Operative modifiers.** "What is the **minimum** temperature" against "the **maximum** temperature"
   changes no number, no entity and no negation particle, and leaves lexical overlap high because one word
   in six differs. Nothing in the verifier could see it. Added `operative_modifiers()`, a closed lexicon of
   terms whose substitution inverts a question — min/max, first/last, before/after, import/export,
   average/median, increase/decrease, include/exclude.

2. **Negation without a particle.** Two forms slipped through: *morphological*
   ("possible"/"impossible", "refundable"/"non refundable") and *lexical* ("thin the blood" / "**fail to**
   thin the blood"). Added prefix detection that only fires when the stem appears in the counterpart query —
   so "international" is not treated as negating "national" — plus negating verbs (fail, lack, prevent,
   exclude, prohibit, deny).

3. **Alphanumeric identifiers.** "pandas" against "Panda3D" reported *zero entities on both sides*: the
   proper-noun pattern `[A-Z][a-z]+\b` cannot match across the digit. Added `_ALNUM_ID_RE`, which also
   catches Python3, GPT4 and room identifiers like B4.

**Decision.** Verifier checks are: number ∧ entity ∧ negation ∧ **modifier** ∧ lexical floor. Operating
point τ_hi = 0.97, τ_lo = 0.75.

**Justification and the headline claim.** The safe operating point is **not** a threshold. Removing the
verifier and relying on similarity alone cannot reach the report's <2% target at *any* threshold in the
sweep, because the adversarial pairs sit at higher cosine than genuine paraphrases (ADR-024). The verifier
takes the false-hit rate to 0.0% — and it is four set comparisons, costing microseconds, against the
cross-encoder or NLI pass the literature would reach for.

**Costs, stated plainly.**

- The modifier lexicon cannot distinguish opposition from synonymy: "brief" and "short" are both listed, so
  that legitimate pair is conservatively rejected. A curated antonym-pair structure would fix it and is
  future work.
- True-hit rate at τ_hi = 0.97 is 1/5 controls. **Five controls is far too small a denominator to estimate
  a true-hit rate**, and this must not be reported as one. The real true-hit measurement comes from the
  paraphrase class of the main corpus; the adversarial subset is a *safety* instrument, not a utility one.
  Expanding the control set is the first corpus follow-up.
- Every threshold here is calibrated against `hashing-v1`. Swapping the encoder invalidates them, which is
  Contribution 6's point demonstrated on our own stack.

---

### ADR-028 — M1 tier 2 is encoder-limited, not technique-limited

**Context.** Tier 2 (extractive redundancy removal) fired **zero times in 239 opportunities**. The shipped
threshold of 0.80 had been set by eye from a sentence pair that was not actually in the corpus — precisely
the practice this project criticises the caching literature for. Sweeping it properly
(`parsimony calibrate-dedup`) and then measuring every intended duplicate in the summarisation class gave:

| intended duplicate pair | cosine |
|---|---|
| "The library will close at 6 PM on weekdays" / "The library shuts at 6 PM on weekdays" | 0.729 |
| "Remote work reduces commuting time" / "Remote work cuts commuting time significantly" | 0.635 |
| "The course runs for 12 weeks" / "The course lasts 12 weeks in total" | 0.561 |
| "The policy takes effect on 1 April" / "The policy starts on 1 April for full time staff" | 0.486 |
| "Rent is 1200 per month" / "Monthly rent comes to 1200" | **0.412** |
| "The recipe needs 250 g of flour" / "You will need 250 g flour for this" | **0.321** |
| "produces no direct carbon emissions" / "emit no carbon dioxide while operating" | **0.319** |

> Values are for the sentence pairs standalone. Measured in corpus context they run 0.01–0.09 lower,
> because the split retains prefixes like `"Summarise: "`; an earlier draft quoted the in-context figure
> (0.324) as though it were the clean pair. The conclusion is unaffected — both are far below any usable
> threshold.

**The bottom three are the same fact reworded** — exactly what tier 2 exists to delete — and the lexical
encoder places them barely above unrelated text.

**Decision.** Set `dedup_threshold = 0.70`, the loosest value with a 0% gate-revert rate. Report tier 2's
near-zero contribution as an **encoder property**, not a technique failure.

**Justification.** There is no threshold that recovers those pairs. Reaching 0.32 would merge sentences that
share nothing but function words, and the fidelity gate would not save us — these paraphrases carry the same
numbers and entities, so every invariant check passes while the meaning of the *summary* changes. Lowering
the threshold trades a silent quality loss for a token saving, which is the opposite of the trade this
project is trying to characterise.

**Consequences.**

- Tier 2's ~0 pp contribution in the ablation is now *explained* rather than merely reported. Without this
  measurement the honest-looking conclusion "extractive redundancy removal does not help at small scale"
  would have been drawn, and it would have been **wrong** — the technique was never given a working
  similarity signal.
- This is the sharpest concrete argument in the project for the encoder swap. It converts "MiniLM would
  presumably be better" into "MiniLM is required for tier 2 to function at all, and here is the
  quantified gap".
- Strengthens Contribution 6 beyond its original claim. The report proposes a calibration table of safe
  settings per model. This finding says something stronger: for some module/encoder combinations **no safe
  setting exists**, and a calibration table that only ever reports a number would hide that.

---

### ADR-033 — M7's value is a property of the traffic, not of the module

**Context.** The project is titled "a stacked, **self-improving** optimisation layer". M7 was built — it
mines a `PolicyBundle` from logs, `warm_start` loads one into a live pipeline, and the CLI exposes both — but
nothing in `reproduce.py` ever ran it. "Self-improving" rested on code rather than on a number, which is the
one claim in the title a reviewer can check in ten seconds.

The obvious measurement is worthless. Replaying a bundle over the conversations it was mined from is
guaranteed to hit, because the cache seed contains those exact questions; that measures memorisation. The
real question is whether a bundle mined from one set of conversations helps on a **disjoint** set.

**Decision.** Split by conversation, stratified by class, mine from one half, and measure the other half cold
against warm — then repeat that across a range of traffic recurrence rates.

**What the first run showed, and why it changed the design.** On the real corpus, transfer was **+0.17 pp**
from a bundle containing 2 cache seeds, 1 redundancy phrase, 0 templates and an **empty** standing-context
digest. Before reporting "M7 does not transfer", we measured the corpus:

> **1.9% of user turns repeat a question asked earlier.** Four questions, across 263 turns.

M7 mines repetition. The ablation corpus was authored for *ablation diversity* — six classes of deliberately
distinct conversations — which is exactly the right shape for measuring M1/M2/M3/M5 and exactly the wrong
shape for measuring a module that learns from recurrence. This is the same distinction as ADR-028: **M7 is
corpus-limited, not technique-limited.**

So the deliverable is not a number but a curve.

| actual recurrence | seeds | cold | warm | transfer | extra hits | extra gate fires |
|---|---|---|---|---|---|---|
| 0.0% | 0 | 7.21% | 7.21% | **+0.00 pp** | 0 | 0 |
| 7.5% | 3 | 11.18% | 12.49% | +1.31 pp | +1 | 0 |
| 22.5% | 8 | 14.99% | 20.41% | +5.43 pp | +4 | 0 |
| 33.3% | 10 | 23.38% | 31.93% | +8.55 pp | +6 | 0 |
| 45.0% | 12 | 33.28% | 47.99% | +14.70 pp | +10 | 0 |
| 57.5% | 16 | 45.69% | 63.52% | **+17.83 pp** | +12 | 0 |

**Justification.** A single number for "does learning transfer" is not answerable, because the answer depends
entirely on the traffic. Turning it into a calibration curve is the same deliverable the project promises
everywhere else, extended from *which modules for which query class* to *which modules for which traffic
shape*.

The **exact zero at 0% recurrence is what makes the rest credible**. With nothing repeated there is nothing
to mine, the bundle is empty, and the measurement returns precisely +0.00 pp and 0 extra hits. A study whose
null condition does not come out null is measuring its own plumbing.

**Extra gate fires are zero at every level.** A seeded cache can serve an answer mined from a *different*
question, so warm-starting could have bought tokens by serving wrong answers. It did not.

**Two integrity notes.**

- Traces are **synthetic in their repetition structure only**. Every question is a real corpus question and
  every answer a real pipeline answer; what is imposed is how often questions recur. Modelled as a hot set
  plus a long tail, because that is the shape assistant traffic actually takes — a support desk, a docs bot
  and a classroom tool all see a few questions repeatedly against a tail of one-offs.
- The first generator sampled the tail **with** replacement, so the birthday paradox manufactured repeats on
  its own: a trace requested at 0% recurrence measured **25%** actual recurrence. The x-axis was detached
  from the thing it claimed to vary, and the curve would have been meaningless. Tail draws are now without
  replacement, and both the target and the *measured* rate are reported.

**Consequences.**

- "Self-improving" is now a measured claim with a stated precondition: M7 pays above roughly 5-10% traffic
  recurrence and is worth nothing below it.
- The headline ablation's silence on M7 is explained rather than hidden. It is a fact about the corpus.
- Gives the report a defensible deployment recommendation: mine bundles for repetitive workloads
  (support, documentation, teaching), not for exploratory ones.

---

### ADR-034 — Prefill dominates on CPU, so input reduction buys the expensive half

**Context.** ADR-007 deferred the real provider, and every latency number until now came from
`MockProvider`'s two constants (120 ms TTFT, 65 ms/token). The project's core argument is a chain — *shorter
prompt, fewer tokens, less time and energy* — of which only the first link was ever measured. Research gap
2 is precisely the prefill/decode split, and it was unanswerable by construction.

With `qwen2.5:1.5b-instruct` (Q4_K_M) on a Ryzen 7 5800HS it is answerable.

**Decision.** Read Ollama's server-reported `prompt_eval_duration` and `eval_duration` rather than wall-clock
TTFT. TTFT conflates prefill with HTTP and scheduling overhead; the gap is specifically about the split.

**Findings.**

| input tokens | prefill | decode | ms / input token | prefill share |
|---|---|---|---|---|
| 146 | 1,360 ms | 122 ms | 9.31 | 91.7% |
| 257 | 2,119 ms | 124 ms | 8.25 | 94.5% |
| 474 | 3,902 ms | 165 ms | 8.23 | 95.9% |
| 906 | 7,691 ms | 131 ms | 8.49 | 98.3% |
| 1,338 | 11,609 ms | 136 ms | 8.68 | 98.8% |

Prefill is linear in input length at roughly **8.5 ms per input token** and accounts for **92-99%** of
total time. Decode is nearly constant. **The simulation was wrong in the direction that mattered**: it
assumed 120 ms TTFT, understating the prompt side by more than an order of magnitude, and 65 ms/token
decode against a real 37-47 ms.

Applying the measured rate to the existing ablation converts every token result into wall clock:

| cell | input tokens saved | prefill saved (corpus) | per request |
|---|---|---|---|
| M5 | 3,660 | 31.1 s | 118 ms |
| M3 | 6,469 | 55.0 s | 209 ms |
| M1+M2+M3+M5 | 9,308 | 79.1 s | 301 ms |
| **full stack +M4+M6** | **11,836** | **100.6 s** | **383 ms** |

**Second finding — prompt order has a wall-clock price.** ADR-025 measured position-aware placement in
*prefix tokens reused*, a proxy nobody outside this project reports. Ollama reuses the KV cache across
requests, so the proxy has a price:

| arrangement | tokens | first call | steady state | reuse |
|---|---|---|---|---|
| stable prefix (M4) | 1,490 | 14,569 ms | **212 ms** | **98.5%** |
| volatile head | 1,497 | 19,014 ms | **18,914 ms** | **0.5%** |

Same content, 7 tokens apart (0.5%), **~80x the steady-state cost**. Every metric in the compression
literature scores these two configurations identically. Reproduced across runs: 98.5%/98.7% against
0.5%/-4.7%.

**Three measurement traps, all of which produced a plausible flat line rather than an error.**

1. **Prefix reuse.** Consecutive probes sharing a prefix let the server skip prefill entirely. The first
   attempt reported a flat ~2,100 ms TTFT across an 85x range of prompt sizes.
2. **Context truncation.** With `num_ctx` unset, 7,000- and 14,000-token prompts were both silently cut to
   2,050 tokens — and because truncation removes the *head*, it deleted the unique marker and made two
   different prompts identical, reporting 0.03 ms/token. The window is now explicit and any prompt whose
   measured token count falls short of the request is dropped with a reason.
3. **Cache persistence across processes.** Ollama's KV cache outlives the Python process, so with fixed
   prompts only the very first execution measures prefill. A re-run reported 0.37 ms/token against the first
   run's 8.25, and turned the prefix arms into nonsense (-10,092% "reuse"). Every run now carries a fresh
   nonce. **A measurement valid only the first time it is ever executed would have failed live in front of a
   reviewer running the demo twice.**

**Consequences.**

- The project's central claim stops being an assumption. On CPU, input tokens *are* the dominant cost, and
  the simulation understated how much.
- M4 graduates from a proxy metric to a wall-clock result, and its payoff is far larger than the token
  counts suggested — because it is worth ~0 tokens and ~18 seconds.
- Every latency figure produced before this ADR should be read as pipeline-correctness evidence only, which
  is what `model_digest = "mock:v1"` was always there to signal.
