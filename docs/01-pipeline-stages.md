# Parsimony — End-to-End Pipeline: Stage Specifications

Eight canonical processing stages. Each is specified as: **Objective · Inputs · Outputs · Techniques ·
Libraries · Advantages · Disadvantages · Alternatives · Recommendation · Integration**.

Stages are the *architectural* decomposition (what kind of work happens). Modules M1–M8 are the *research*
decomposition (what gets ablated). They are not the same partition, and conflating them is a common way these
systems go wrong. The mapping:

| Stage | Modules involved |
|---|---|
| 1 Ingestion | — (infrastructure) |
| 2 Preprocessing | M8 (fingerprint extraction, PII detection) |
| 3 Storage | — (infrastructure, serves all) |
| 4 Retrieval | M2 lookup, M3 candidate gathering, M7 bundle load |
| 5 Processing | M1, M3 condensation, M4 |
| 6 Ranking | M2 verifier, M3 scoring, M6 tier scoring |
| 7 Decision-making | M2 policy, M5, M6, M8 verdicts |
| 8 Output generation | M5 early-stop, provider, ledger finalisation |

---

## Stage 1 — Ingestion

**Objective.** Convert every inbound request, from any of three surfaces, into exactly one canonical
`InboundRequest`, so that no downstream code knows or cares where a request came from.

**Inputs.** CLI invocation; `POST /v1/chat/completions` (OpenAI wire format); dashboard websocket message;
sweep-runner corpus row.

**Outputs.** `InboundRequest{request_id: ULID, conversation_id, messages: tuple[Turn,...], surface,
overrides: ParsimonyConfig|None, received_at}`.

**Techniques.** Adapter per surface → narrow internal DTO. ULID for `request_id` (128-bit, lexicographically
sortable by time — the ledger gets chronological ordering from the primary key with no extra index).
Lossless `raw` passthrough field so the OpenAI adapter can round-trip fields Parsimony does not model.

**Libraries.** `python-ulid`; `fastapi` + `pydantic` for the HTTP surface only (L4 — never leaks into L0);
`typer` for CLI.

**Advantages.** One canonical form means one set of pipeline tests. Adding a fourth surface later (VS Code
extension, Slack bot) is one adapter and zero core changes.

**Disadvantages.** A translation layer that can silently drop fields; mitigated by the `raw` passthrough and
a contract test asserting OpenAI round-trip fidelity for the fields we claim to support.

**Alternatives considered.**

| Option | Verdict |
|---|---|
| Each surface owns its own handler | Rejected — triples the pipeline test matrix, and the surfaces will drift |
| Adopt the OpenAI schema as the internal canonical type | Rejected — couples L0 to a vendor schema carrying `tools`, `logprobs`, `function_call` we never use, and which changes without our consent |
| Adapters → narrow internal DTO | **Chosen** |

**Recommendation.** Narrow DTO with adapters. The OpenAI shim is a translation at the edge, which is
precisely where the report's "change one base URL" claim needs it to be.

**Integration.** Emits `InboundRequest` → Stage 2 constructs the first `RequestContext` from it. The
`original_query` / `original_history` fields are populated here and are never written again.

---

## Stage 2 — Preprocessing

**Objective.** Compute, exactly once, every derived quantity the rest of the pipeline needs; and establish
the fidelity ground truth against which all later transformations are checked.

**Inputs.** `InboundRequest`, active `ParsimonyConfig`, loaded `PolicyBundle`.

**Outputs.** Fully-populated `RequestContext` with `invariants`, `response_class`, `complexity`, PII spans,
and a primed `DerivedCache`.

**Techniques.**

1. **Safety normalisation only** — Unicode NFKC, control-character stripping, CRLF→LF. Deliberately *not*
   whitespace collapse or markdown stripping: those reduce tokens, so they belong to M1 tier 1 and must be
   ablatable. Putting them here would contaminate the baseline and inflate every measured saving.
2. **Tokenisation** — exact counts under the *target model's own* tokenizer. Not word counts, not
   `tiktoken`. The report's tokenizer-aware claim depends on this.
3. **Invariant extraction** — numbers via regex (incl. unit suffixes, ranges, ordinals); entities via spaCy
   `en_core_web_sm` NER; negations via a closed lexicon + `dep_ == "neg"`; quoted spans via a paired-
   delimiter scan. Emitted as an `Invariants` frozenset bundle.
4. **PII detection** — spans only; redaction is applied at write boundaries (Stage 3), never here. Detecting
   early and redacting late means the model still sees the user's real text (correct behaviour) while the
   cache and ledger never do.
5. **Query classification** — response class ∈ {factual, arithmetic, reasoning, code, summarisation,
   follow-up} plus a scalar complexity estimate. Feeds M5's budget and M6's tier.

**Libraries.** `tokenizers` (HF, Rust backend — ~1 M tok/s, offset mapping available); `spaCy` +
`en_core_web_sm` (~13 MB, ~10 ms on short text); `regex`; `sentence-transformers`.

**Advantages.** One spaCy pass and one embedding pass serve four consumers. Invariants computed here make
every subsequent gate check a set-difference (microseconds) rather than a re-parse (milliseconds) — with up
to seven gate checks per request, this is the difference between a 15 ms and a 90 ms gate cost.

**Disadvantages.** Front-loads ~20 ms onto *every* request, including ones the deterministic router would
have answered in 0.1 ms.
**Mitigation:** two-phase preprocessing. Phase A is regex-only (numbers, arithmetic detection, exact-hash
key) and runs before the deterministic tier. Phase B (spaCy + embedding) runs lazily, on first access via
`DerivedCache`, so a request short-circuited by M6a or the exact-hash cache tier never pays for it. This is
why `DerivedCache` uses lazy properties rather than eager computation.

**Alternatives considered.**

| Concern | Options | Verdict |
|---|---|---|
| Classification | (a) hand-written rules; (b) logistic head on the query embedding; (c) LLM-as-classifier | **(b) + (a) as override.** The embedding is already computed for cache lookup, so a 384→6 logistic head costs ~5 µs. Rules alone are brittle across the 6 corpus classes; an LLM call costs the tokens we are trying to save — self-defeating. Rules stay as a high-precision override for arithmetic/unit patterns. |
| Entity extraction | (a) spaCy `sm`; (b) spaCy `trf`; (c) regex + gazetteer; (d) GLiNER | **(a).** `trf` is a BERT pass (~80 ms) — blows the budget. Regex-only misses PERSON/ORG, which is exactly what a compressor deletes. `sm` is the accuracy/latency knee. |
| PII detection | (a) Presidio; (b) regex + spaCy NER | **(b) default, (a) behind the `PiiDetector` protocol.** Presidio pulls a large dependency tree and adds 30–50 ms. For a local-only, single-user research system, regex (email/phone/card/IP/Aadhaar) + NER is proportionate. Swappable if a reviewer objects. |

**Recommendation.** Two-phase lazy preprocessing as above. Freeze the spaCy model version in
`pyproject.toml` — an NER model upgrade mid-project silently changes gate behaviour and invalidates
comparisons against earlier runs.

**Integration.** Produces the `RequestContext` that flows through every remaining stage. `invariants` is
read by Stage 7's gate on every proposal; `DerivedCache` is read by Stages 4, 5 and 6.

---

## Stage 3 — Storage

**Objective.** Persist the token ledger, the semantic cache, prompt/response blobs and policy bundles, with
write characteristics that survive a 16-cell × 5-seed × 3-model unattended sweep.

**Inputs.** `StageTrace` rows, generation results, cache writes, bundle artefacts.

**Outputs.** Durable ledger; queryable analysis database; vector index; content-addressed blobs.

**Techniques.**

*Ledger — dual sink behind one protocol.*

```python
class LedgerSink(Protocol):
    def write(self, row: LedgerRow) -> None: ...
    def flush(self) -> None: ...
```

- `JsonlSink` — append-only, one file per worker per run. Used in `EXPERIMENT` mode.
- `SqliteSink` — WAL mode, indexed. Used in `SERVE` mode, where the dashboard needs live queries.
- `parsimony ledger import` folds JSONL into the analysis DB after a sweep.

**Why not SQLite for the sweep** (a deviation from the report's §4.1): the sweep runs N worker processes in
parallel on one machine. SQLite permits one writer at a time; under WAL, concurrent writers get
`SQLITE_BUSY` and back off. Across a 6+ hour unattended run that is either lost rows or serialised workers —
and a corrupted or partial ledger *is* a lost experiment, since the ledger is the entire result. Append-only
JSONL has no contention (each worker owns its file), is crash-safe to the last complete line, and is
trivially recoverable. The cost is a 30-second import step. That is an excellent trade.

*Vector index.*

```python
class VectorIndex(Protocol):
    def add(self, vec: np.ndarray, entry_id: str) -> None: ...
    def search(self, vec: np.ndarray, k: int) -> list[tuple[str, float]]: ...
```

- `ExactIndex` (numpy) — **default**.
- `FaissIndex`, `HnswIndex` — behind the same protocol, for the scaling demonstration.

**Why exact search is the default, deviating from the report's §4.1 FAISS choice:** this is the most
consequential storage decision in the project. The cache holds order 10³–10⁵ entries. An exact search is a
single `(N, 384) @ (384,)` matmul: at N = 100 000 that is 38 MFLOP ≈ 3 ms on a modern CPU, and at realistic
N = 5 000 it is under 0.2 ms. FAISS with an IVF or HNSW index is *approximate* — it has recall < 1.0. The
headline Gap 3 deliverable is the **false-cache-hit rate**. If the index is approximate, the measured
false-hit rate is a mixture of "the similarity policy was wrong" and "the ANN index missed the true nearest
neighbour", and those two are not separable after the fact. **Exact search makes the measurement measure the
thing it claims to measure.** FAISS stays available and contract-tested so the scaling claim can still be
demonstrated — but it must not be in the path of the research result.

*Blobs.* Prompts and responses are large and highly repetitive across 16 cells. Store SHA-256 → text once;
ledger rows carry the 64-char hash. Deduplication across cells is substantial (the invariant prefix is
byte-identical by construction) and it makes the ledger itself small enough to keep in git-lfs.

*Schema evolution.* Additive-only, forward-only. New columns are nullable; columns are never renamed or
removed; a `schema_version` row travels with each run. The ledger schema **will** change as M4–M7 land, and
Sprint 0's baseline runs must remain readable in October.

**Libraries.** stdlib `sqlite3`, `json`, `hashlib`; `numpy`; `duckdb` (analysis only — reads the JSONL and
Parquet directly with zero import for exploratory work); `faiss-cpu` optional.

**Advantages.** No server, no daemon, no network. Whole project state is a directory that can be zipped and
attached to the report.

**Disadvantages.** No concurrent multi-machine access; irrelevant here by design.

**Alternatives considered.** Postgres (operational weight unjustifiable for single-user local research);
ChromaDB (bundles an embedding model and a storage engine and a distance policy — three decisions we need to
own separately, and it obscures exactly the geometry Gap 3 is about); Parquet-only (poor for incremental
append during a live run).

**Integration.** Stage 4 reads the index; Stage 7 writes cache entries (post-redaction); Stage 8 finalises
the ledger row; Stage 5's analysis (L5) reads the imported DB.

---

## Stage 4 — Retrieval

**Objective.** Fetch every candidate the pipeline might use — cache entries, historical turns, policy
templates — without yet deciding which to use. Retrieval is deliberately separated from ranking and from
decision-making.

**Inputs.** `RequestContext` (query text, query embedding, context chain), `VectorIndex`, `PolicyBundle`.

**Outputs.** `CacheCandidates` (top-k with scores), `HistoryCandidates` (all prior turns + embeddings),
`TemplateMatches`.

**Techniques.**

1. **Exact-hash cache tier.** Key = BLAKE2b(canonical_query ‖ context_chain_hash ‖ model_id ‖
   prompt_schema_version). Sub-microsecond dict lookup. Note the key includes `model_id` — a cached Llama
   answer must not be served during a Qwen run, or the cross-model generalisation study in §4.6 is silently
   corrupted.
2. **Context chain.** Following MeanCache: `chain_hash = H(H(turn_{n-1}) ‖ H(turn_{n-2}) ‖ ... )` over the
   last *k* parent turns, `k` configurable (default 2, ablatable). Prevents "and what about the second one?"
   in conversation A being served from conversation B.
3. **Semantic tier.** Cosine over L2-normalised MiniLM embeddings, top-k = 5 (not 1 — the three-zone verifier
   in Stage 6 needs alternatives to compare against).
4. **History candidate gathering.** All prior turns, embedded in **one batched forward pass**. Batching is
   the difference between ~60 ms and ~12 ms on a 12-turn conversation.
5. **Volatility / TTL filter.** Entries tagged volatile (queries about time, prices, "current", "latest") get
   a TTL and are filtered at retrieval, not at write. A stale entry stays in the index for analysis (we want
   to *report* how often TTL fires) but is not returned.

**Libraries.** `sentence-transformers` (all-MiniLM-L6-v2, 384-d, ~22 M params, ~5 ms/query CPU); `numpy`;
`hashlib`.

**Advantages.** Retrieving before deciding means the ledger records *what was available and rejected*, not
just what was used. That turns the threshold sweep in §4.6 into an offline re-analysis of one run rather
than one run per threshold — a large saving on a CPU-bound project.

**Disadvantages.** Retrieves candidates that are usually discarded. At k=5 the cost is negligible.

**Alternatives considered.**

| Concern | Options | Verdict |
|---|---|---|
| Embedding model | MiniLM-L6-v2 · bge-small-en-v1.5 · gte-small · e5-small-v2 | **MiniLM default** (fastest, well-understood, 384-d). **bge-small as a second arm for the Gap 3 sweep only** — see `00-architecture.md` §10.3: a false-hit conclusion drawn on one encoder is a conclusion about that encoder. |
| Similarity | cosine · dot · euclidean | **Cosine on normalised vectors** = dot product; one matmul, and thresholds are interpretable on [-1,1] which matters because thresholds are a reported deliverable. |
| Chain depth k | 1 · 2 · 3 · full | **2, ablatable.** k=1 misses two-step anaphora; full chain makes every follow-up key unique and the hit rate collapses to zero. |

**Recommendation.** As above. Record the top-5 with scores in the ledger *always*, even on a miss — that
data is what makes the threshold sweep cheap.

**Integration.** Candidates flow to Stage 6 (ranking) which scores them, then Stage 7 (decision) which
accepts or rejects.

---

## Stage 5 — Processing

**Objective.** Transform the retained content: compress it, condense it, and assemble it into a final prompt
whose invariant prefix is byte-stable across turns.

**Inputs.** `RequestContext`, selected history, target tokenizer, redundancy lexicon from `PolicyBundle`.

**Outputs.** Compressed query and history; `AssembledPrompt{invariant_zone, volatile_zone, full_text,
prefix_token_count}`.

**Techniques.** (Module internals in `02-module-specs.md`; the stage-level view:)

1. **M1 tier 1 — lossless normalisation.** Whitespace, markdown scaffolding, boilerplate. Provably
   information-preserving.
2. **M1 tier 2 — extractive redundancy removal.** Sentence-split → embed → MMR select. Drops sentences whose
   embedding is near-duplicate of a retained one.
3. **M1 tier 3 — tokenizer-aware rewriting.** Among equivalent phrasings, choose the one costing fewer
   tokens *under the target model's vocabulary*, with **negative-yield detection**: every candidate edit is
   re-tokenised and reverted if it does not actually reduce the count. BPE merges are context-dependent, so
   deleting a word can *raise* the token count by breaking a merge in its neighbours — the report's genuinely
   novel detail, and it only works if this check is in the inner loop.
4. **M3 condensation** — rolling summarisation as one of four strategies.
5. **M4 assembly** — two zones. Invariant (system prompt + context digest) is byte-stable and untouched by
   any module. Volatile (trimmed history + compressed query) carries all rewriting.

**Libraries.** `tokenizers` (offset mapping is essential for windowed re-tokenisation); `sentence-transformers`;
`spaCy` senter or `pysbd` for sentence splitting.

**Advantages.** Confining all rewriting to the suffix is what makes the KV prefix survive; it converts Gap 4
from a problem into a design property.

**Disadvantages.**
- Negative-yield detection is O(n_edits × tokenise). Naively re-tokenising the full text per candidate edit
  on a 2 000-token prompt with 40 candidates = 80 000 token-ops per request. **Mitigation:** windowed
  re-tokenisation — re-tokenise only ±32 characters around the edit, since BPE merges are local. A golden
  test asserts windowed results match full re-tokenisation across the corpus; if the assertion ever fails the
  window widens. This is a real correctness risk and it needs the test, not just the optimisation.
- Rolling summarisation costs a model call. **Mitigation:** it runs *off the critical path* — after the
  response is returned, summarise in the background for the *next* turn. Its token cost is still charged to
  the ledger (honesty), but its latency is not charged to the user. This is an architectural placement
  decision that has to be made now: retrofitting async summarisation means restructuring M3's interface.

**Alternatives considered.**

| Concern | Options | Verdict |
|---|---|---|
| Tier-3 compressor | (a) lexicon/rule substitution; (b) LLMLingua-2 (XLM-RoBERTa token classifier); (c) LLMLingua (perplexity scoring with a small LM) | **(a) as default, (b) as a measured ablation arm.** (c) is disqualified on the project's own terms: it is an autoregressive forward pass, so on CPU it competes for exactly the cycles it is meant to save — the limitation the report identifies in Paper 1. (b) is a 280 M-param encoder, plausibly 50–200 ms on an i5 — likely to exceed the entire 120 ms overhead budget on its own. Rather than assume that, **measure it**: implementing (b) behind the same protocol and reporting its overhead is a publishable negative result that directly substantiates the report's critique of the literature. |
| Sentence splitting | regex · pysbd · spaCy senter | **spaCy senter** — the `Doc` already exists in `DerivedCache`, so it is free. |
| Selection | greedy threshold · MMR · LexRank · clustering | **MMR** — one implementation shared with M3, one set of tests, one λ to explain. |

**Recommendation.** Tiers strictly ordered 1→2→3, each independently switchable, each proposing separately
so the fidelity gate sees three small proposals rather than one large one. A tier-3 rewrite that drops a
negation is then reverted *without* discarding the safe tier-1 savings.

**Integration.** `AssembledPrompt` → Stage 7 sets the budget and tier → Stage 8 generates.

---

## Stage 6 — Ranking

**Objective.** Score candidates. Ranking makes no decisions and mutates no state — it only produces ordered,
scored lists. This separation is what makes thresholds re-tunable offline.

**Inputs.** `CacheCandidates`, `HistoryCandidates`, query embedding, `Invariants`.

**Outputs.** Scored + ordered candidates with per-candidate feature vectors written to the ledger.

**Techniques.**

1. **Cache candidate scoring.** Cosine similarity, plus the verifier features computed for *every* candidate
   in the verify band: token-level Jaccard, entity-set agreement, number-set agreement, **negation
   agreement**. The last is the one the literature omits and the one the adversarial subset is built to
   exercise: 50 pairs differing by exactly one operative token, and an operative token is very often a
   negation ("is X safe" / "is X *not* safe") or a number.
2. **History ranking.** Four strategies, one protocol:
   - *Recency* — baseline, O(1), surprisingly hard to beat on short chats.
   - *Embedding relevance* — cosine(turn, query).
   - *MMR* — `argmax_i [ λ·sim(t_i, q) − (1−λ)·max_j sim(t_i, s_j) ]`, λ default 0.7. O(n²), n < 100, fine.
   - *Rolling summarisation* — condenses older turns rather than dropping them.
3. **Position-aware placement.** Scoring *what to keep* and choosing *where to put it* are separate
   operations and must be measured separately — the report is explicit about this and the interface must
   honour it: `select() -> ranked turns`, then `arrange() -> ordered turns`. Two functions, two ablation
   flags, two effects. Collapsing them into one method makes the report's claim unmeasurable.
4. **Router tier scoring.** Feature vector: response class, arithmetic-pattern match, entity count, token
   length, complexity, template match. **All features are logged whether used or not**, so a learned router
   can later be fit from the ledger with no interface change and no new data collection.

**Libraries.** `numpy`; `scikit-learn` (logistic regression for classifier/router, if fitted).

**Advantages.** Because scores and features are persisted, the entire similarity-threshold sweep in §4.6 is
an offline `pandas` groupby over one run's ledger, not 6 CPU-hours of re-running. On a CPU-bound project
this is the difference between running the sweep once and running it three times.

**Disadvantages.** Wider ledger rows. Mitigated by blob-hashing the text and keeping only numerics inline.

**Alternatives considered.** Cross-encoder re-ranking for cache candidates (rejected — a second neural
forward pass per query, same self-defeating-overhead argument as the neural compressor; the cheap lexical +
entity + negation verifier is the report's own design and is defensible precisely because it is cheap).

**Integration.** Scored candidates → Stage 7 applies thresholds and decides.

---

## Stage 7 — Decision-making

**Objective.** Convert scores into actions, under one central, versioned policy. Every threshold in the
system lives here and nowhere else.

**Inputs.** Scored candidates, `ParsimonyConfig`, `Invariants`, proposals from every module.

**Outputs.** Cache accept/reject/verify verdict; route tier; output budget; fidelity verdicts; committed
`RequestContext`.

**Techniques.**

1. **Three-zone cache policy** (replacing a single threshold):
   ```
   sim ≥ τ_hi (0.92)          → ACCEPT
   τ_lo (0.78) ≤ sim < τ_hi   → VERIFY: require entity ∧ number ∧ negation agreement,
                                 and Jaccard ≥ j_min; else REJECT
   sim < τ_lo                 → REJECT
   ```
   Both τ are per-model calibrated values and are outputs of the project (Contribution 6), not constants.
2. **Fidelity gate**, dispatched on `TransformKind`:
   - `REWRITE` → `Invariants(original).difference(candidate_text)` must be empty. Set difference on
     frozensets — microseconds, because extraction already happened in Stage 2.
   - `SELECT` → retained units must be byte-identical to their originals; removal is legitimate.
   - `AUGMENT` / `DECIDE` → no text check.
   On failure: revert that module for that turn, set `gate_fired`, record which invariant class was
   violated. That last field is a genuine finding — "the gate fires 11 % of the time and 70 % of those are
   numbers" is a reportable characterisation of small-model-safe compression.
3. **Output budget.** Per response class, never global — uniform length hints are documented to damage
   already-terse answers (report §2.2), so a global cap would produce a quality regression that looks like a
   compression failure.
4. **Router tier.** Cheapest sufficient tier: deterministic → cache → 1B → 3B.

**Libraries.** stdlib. This stage is deliberately pure Python with no dependencies — it is the part most
likely to be re-tuned and it must be trivially unit-testable and fast.

**Advantages.** Single-site policy means M7's output is a config file, thresholds are greppable and
diffable, and no module can quietly hold a hard-coded 0.85.

**Disadvantages.** Central policy risks becoming a god object. Mitigated by keeping it *stateless* — pure
functions from (scores, config) to verdict.

**Alternatives considered.** Per-module thresholds (rejected — makes M7 integrate with seven modules instead
of one config, and makes the calibration table impossible to assemble); learned end-to-end policy (premature
— log the features now, fit later if the data supports it).

**Integration.** Emits the committed context → Stage 8 generates, or a `ShortCircuit` → Stage 8 finalises
the ledger with `generated=False`.

---

## Stage 8 — Output generation

**Objective.** Invoke the provider, stop generation as early as is safe, and close the ledger row with the
prefill/decode split the project's central claim rests on.

**Inputs.** `AssembledPrompt`, `output_budget`, `route_tier`, `LLMProvider`.

**Outputs.** Response text; `LedgerRow`; four quality scores (offline); surface-specific formatting.

**Techniques.**

1. **Streaming provider protocol.**
   ```python
   class LLMProvider(Protocol):
       def generate(self, prompt: str, params: GenParams) -> Iterator[TokenEvent]: ...
       def tokenizer_id(self) -> str: ...
       def model_digest(self) -> str: ...   # pinned; recorded in the ledger
   ```
   Streaming is mandatory, not a feature: TTFT and TPOT cannot be measured from a blocking call, and the
   early-stop rule needs to observe tokens as they arrive. `model_digest` is recorded so a silent `ollama
   pull` mid-project cannot invalidate comparisons without leaving a trace.
2. **Early stopping.** Two cheap detectors on the stream:
   - *Trigram novelty ratio* over a sliding window — when the fraction of unseen trigrams in the last N
     tokens falls below a threshold, the model is restating. O(1) per token.
   - *Structural completion* per response class (closing fence for code, terminal punctuation after the
     expected item count for lists).
   Sentence-level embedding checks are run at most every ~3 sentences, never per token — a per-token
   embedding pass would cost more than the generation it saves.
3. **Budget enforcement.** `num_predict` passed to the provider. Belt and braces: the early-stop rule
   usually fires first; `num_predict` is the hard ceiling.
4. **Ledger finalisation.** TTFT = first `TokenEvent` timestamp − dispatch. TPOT = (last − first) /
   (n_tokens − 1). Recorded separately, never summed into "latency", because separating them *is*
   Contribution 4's evidence.
5. **Prefix survival.** Longest common **token** prefix against the previous request in the same
   conversation — computed via tokenizer offset mapping, not byte comparison. llama.cpp reuses the KV cache
   at token granularity; a byte-level measure would over-report on multi-byte boundaries and under-report
   when a byte change does not alter the token sequence.

**Libraries.** `httpx` (streaming, async-capable) for the Ollama HTTP API; stdlib `time.perf_counter_ns`.

**Advantages.** The provider protocol keeps the "no Ollama yet" constraint compatible with full pipeline
development: `MockProvider` returns deterministic canned responses with simulated TTFT/TPOT, so the
budgeter's early-stop rule, the ledger's timing fields and the whole orchestrator are testable and
CI-hermetic before any model is installed. `OllamaProvider` is then ~60 lines against a tested interface.

**Disadvantages.** Simulated timings are not real timings; every latency number must be re-measured once
`OllamaProvider` lands, and Sprint 0/1 results must be labelled as pipeline-correctness evidence, not
performance evidence. This must be stated plainly in the review demo.

**Alternatives considered.** Non-streaming (disqualified — destroys TTFT/TPOT and early stopping, the
project's two most novel measurements); direct `llama-cpp-python` bindings (keeps in-process control of the
KV cache and would give *direct* prefix-reuse observability rather than inference from timing — worth
prototyping in Sprint 4 as a second provider, since Gap 4 is a headline contribution and direct measurement
beats inferred).

**Integration.** Closes the request. The ledger row it writes is the input to L5 analysis and to M7's
counterfactual replay — which is why the ledger schema is part of the architecture rather than an
afterthought.

---

## Data flow, end to end

```
InboundRequest
   │  Stage 1: adapters → canonical DTO
   ▼
RequestContext ─────────── original_query / original_history / Invariants  (frozen forever)
   │  Stage 2: phase A regex → phase B lazy (spaCy, embedding)
   ▼
   ├─▶ Stage 4 retrieve ─▶ Stage 6 rank ─▶ Stage 7 decide ──┬─▶ ShortCircuit ──┐
   │                                                         │                  │
   ▼                                                         ▼                  │
Stage 5 process (M1, M3, M4) ◀── each proposal gated by Stage 7 ──┐             │
   │                                                              │             │
   ▼                                                              │             │
Stage 7 budget + tier                                             │             │
   │                                                              │             │
   ▼                                                              │             │
Stage 8 generate (streaming, early stop) ◀─────────────────────────┘            │
   │                                                                            │
   ▼                                                                            │
LedgerRow ◀─────────────────────────────────────────────────────────────────────┘
   │
   ▼  Stage 3 persist (JSONL in EXPERIMENT, SQLite in SERVE)
   │
   ▼  offline
M7 counterfactual replay ──▶ PolicyBundle ──▶ warm-starts next run
```
