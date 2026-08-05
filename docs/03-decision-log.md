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
neighbour" — two causes that cannot be separated after the fact. Exact search makes the measurement measure
the thing it claims to measure. Extensibility is preserved by the protocol; a reviewer asking "does this
scale?" gets a FAISS run with contract tests proving equivalence at the accept/reject level.

**Consequences.** One fewer heavy native dependency on the critical path. FAISS becomes optional, which also
simplifies installation on Windows — a real consideration for this team's environment.

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

| configuration | mean prefix tokens reused |
|---|---|
| M4 off (volatile head) | 2.4 |
| M4 on, chronological | **94.4** |
| M4 on, position-aware | **47.6** |

**The two M4-on arms retain the same turns and send the same number of tokens.** Only the order differs.
Position-aware placement halves prefix reuse for exactly zero token difference.

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
