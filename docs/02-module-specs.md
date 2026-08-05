# Parsimony — Module Specifications M1–M8

Each module is a `Stage` implementation. It declares `reads`/`writes`, returns a `Proposal`, and knows
nothing about any other module. Ablation flags map 1:1 onto module IDs.

**Factorial axes:** M1, M2, M3, M5 (2⁴ = 16 cells). **Always on:** M8, ledger. **Studied separately:** M4
(prefix study), M6 (on top of the winner), M7 (warm-start study).

---

## M1 — Tokenizer-Aware Compressor

**Objective.** Reduce input tokens under the *target model's own vocabulary*, never reducing information the
fidelity gate protects.

**reads** `query, history, invariants, derived.doc` · **writes** `query, history`
**Emits** three separate `ContextPatch(kind=REWRITE)` proposals, one per tier.

### Tier 1 — Lossless normalisation
Collapse runs of whitespace; strip markdown scaffolding (`###`, `**`, table pipes) where it carries no
semantics; remove boilerplate politeness ("please", "could you kindly", "thanks in advance") matched against
a lexicon; normalise list markers.

*Expected yield:* 5–12 % of input tokens on chat-style text. Zero risk. Ship first — it is the cheapest
demonstrable win and it de-risks the mid-August review.

### Tier 2 — Extractive redundancy removal
`sentences = senter(doc)` → batched embed → MMR select with λ from config. Drops sentences whose embedding is
near-duplicate of an already-retained sentence.

*Expected yield:* 8–20 % on multi-sentence prompts, ~0 % on the short single-sentence queries that dominate
the factual class. **Report per class** — an aggregate number hides that this tier does nothing for half the
corpus.

### Tier 3 — Tokenizer-aware rewriting + negative-yield detection
The novel contribution. Candidate edits come from (a) a hand-authored equivalence lexicon
(`"in order to" → "to"`, `"at this point in time" → "now"`), and (b) M7's mined personal redundancy lexicon.

```
for edit in candidates_sorted_by_expected_yield:
    window     = text[edit.start - W : edit.end + W]          # W = 32 chars
    before     = len(tok.encode(window))
    after      = len(tok.encode(apply(window, edit)))
    if after < before:          apply and continue            # real yield
    else:                       revert and record NEGATIVE_YIELD
```

**Why the window is necessary and why it is dangerous.** Full re-tokenisation per candidate is O(n_edits × n)
— 40 edits on a 2 000-token prompt is 80 000 token-ops per request, well beyond budget. BPE merges are local,
so a ±32-char window is *almost always* sufficient. "Almost always" is not good enough for a correctness
claim, so a golden test re-tokenises the full text for every edit across the whole corpus and asserts the
windowed decision matches. If it ever fails, W widens. **Do not ship tier 3 without that test** — a silent
window bug turns the project's most novel claim into a measurement artefact.

*Expected yield:* 5–10 % beyond tier 2, and a genuinely interesting negative-yield rate to report (the
fraction of "obvious" deletions that increase the token count is, as far as the report's literature review
shows, unpublished).

**Libraries.** `tokenizers` (HF, with `return_offsets_mapping=True`), `sentence-transformers`, `spaCy`.

**Advantages.** Every tier independently ablatable; gate sees three small proposals, so a bad tier-3 rewrite
does not cost you tier-1's safe savings.
**Disadvantages.** Tier 3's lexicon is hand-authored and therefore English- and domain-specific; its ceiling
is lower than a neural compressor's. Stated as a limitation, partially answered by M7 mining the lexicon
from real logs.

**Alternatives.** LLMLingua-2 (280 M XLM-R) as a fourth tier behind the same interface — implement it in
Sprint 5 **as a measured arm**, not a default. If it costs 150 ms on an i5, that number is itself a
contribution: it substantiates the report's critique that the compression literature ignores CPU scorer cost.

---

## M2 — Two-Tier Semantic Cache

**Objective.** Serve semantically equivalent questions without a model call, and defend that decision against
near-duplicates.

**reads** `query, conversation_id, derived.embed` · **writes** — (short-circuits or no-ops)
**Emits** `ShortCircuit(served_by=CACHE_*)` or `NoOp`.

**Design.**

| Tier | Mechanism | Cost | Purpose |
|---|---|---|---|
| 0 | Exact hash: `BLAKE2b(canon(q) ‖ chain_hash ‖ model_id ‖ schema_v)` | ~1 µs | Identical repeats |
| 1 | Cosine over normalised MiniLM embeddings, exact search, top-5 | ~5 ms | Paraphrases |

**Three-zone policy** (Stage 7 owns the thresholds):

```
accept  sim ≥ τ_hi
verify  τ_lo ≤ sim < τ_hi  →  entity_agree ∧ number_agree ∧ negation_agree ∧ jaccard ≥ j_min
reject  sim < τ_lo
```

`negation_agree` is the addition beyond the report's text and it is the one that will actually carry the
adversarial subset: 50 pairs differing by one operative token, and operative tokens are disproportionately
negations and numerals. Two queries at cosine 0.94 that disagree on a negation are the canonical false hit.

**Cache key includes `model_id`.** Non-obvious and important: §4.6 re-runs the winning configuration on three
models. Without `model_id` in the key, a Llama-generated answer is served during the Qwen run and the
generalisation study measures nothing.

**Volatility + TTL.** Entries tagged volatile by a rule set (temporal deixis, price/rate terms, "current",
"latest") carry a TTL and are filtered at retrieval. Keep expired entries in the index — how often TTL fires
is a reportable number.

**PII.** Every cache write passes through M8's redactor first. The cache is the one component with
persistent memory, so it is the one component where a leak persists.

**Libraries.** `numpy` (exact search), `sentence-transformers`, `hashlib`.

**Advantages.** Zero model tokens on a hit; the ledger proves it. Exact search means the measured false-hit
rate is attributable to the policy, not to ANN recall (see `01-pipeline-stages.md` Stage 3).
**Disadvantages.** Cold on day one — which is Gap 6, and is M7's job.

**Alternatives.** GPTCache off the shelf (rejected — it bundles embedding, storage and threshold policy into
one opinionated package; we need to own the threshold policy because it is a deliverable). ChromaDB (same
objection, plus it obscures the embedding geometry Gap 3 is about).

---

## M3 — Relevance-Aware History Manager

**Objective.** Choose which prior turns survive, and where they are placed.

**reads** `query, history, derived.embed` · **writes** `history`
**Emits** `ContextPatch(kind=SELECT)` for selection, then a separate `ContextPatch(kind=SELECT)` for
arrangement.

**Two operations, deliberately separate.**

```python
class HistoryStrategy(Protocol):
    def select(self, turns, query_emb, budget) -> list[Turn]: ...
class Arrangement(Protocol):
    def arrange(self, selected, query_emb) -> list[Turn]: ...
```

The report requires the effect of *placement* to be measured separately from the effect of *selection*. That
is only possible if they are separate interfaces with separate flags. Collapsing them into one `trim()` makes
the claim unmeasurable, and it is the kind of thing that is very expensive to fix in week 10.

**Four strategies** (`select`): recency · embedding relevance · MMR (λ=0.7) · rolling summarisation.
**Two arrangements**: chronological (control) · position-aware (most-relevant turn placed adjacent to the
query, exploiting the start/end attention bias documented in *Lost in the Middle*).

**Rolling summarisation runs off the critical path.** After the response is returned, a background task
summarises older turns for the *next* turn. Its token cost is charged to the ledger (honesty); its latency is
not charged to the user (correctness). This placement must be decided now — making it synchronous first and
async later means rewriting the interface.

`SELECT` gating: the gate verifies retained turns are byte-identical to their originals. Rolling
summarisation is the exception — it is `REWRITE` over the turns it condenses, and gets the full invariant
check.

**Libraries.** `numpy`, `sentence-transformers`.

**Advantages.** Head-to-head comparison of four strategies on identical inputs is a clean, publishable
sub-result and is cheap once the harness exists.
**Disadvantages.** Summarisation costs tokens and introduces a second model dependency inside the pipeline.

**Alternatives.** Fixed window (the industry default; keep it as the *control* arm — an honest report needs
to show whether any of the clever strategies beats "keep the last 4 turns", and there is a real chance one
does not).

---

## M4 — Prefix-Stable Prompt Assembler

**Objective.** Make the KV-cache prefix survive, and measure its survival.

**reads** `system_prompt, context_digest, history, query` · **writes** `assembled`
**Emits** `ContextPatch(kind=AUGMENT)`.

**Two zones.**

```
┌─ INVARIANT ZONE ── byte-stable across all turns of a conversation ─┐
│  system instruction  +  M7 persistent context digest              │
├─ VOLATILE ZONE ────────────────────────────────────────────────────┤
│  trimmed history  +  compressed query                              │
└────────────────────────────────────────────────────────────────────┘
```

No other module may write to the invariant zone. Enforced structurally: `system_prompt` and `context_digest`
are absent from every other module's `writes` set, and the registry validates this at boot.

**Prefix survival measurement.** Longest common **token** prefix vs the previous request in the same
conversation, via tokenizer offset mapping.

```python
prev_ids = tok.encode(prev_prompt).ids
curr_ids = tok.encode(curr_prompt).ids
survived = common_prefix_len(prev_ids, curr_ids)
ratio    = survived / len(curr_ids)
```

Token granularity, not bytes: llama.cpp reuses KV at token granularity, so a byte measure over-reports across
multi-byte boundaries and under-reports when a byte edit leaves the token sequence unchanged. Reported as a
first-class ledger field alongside token count — that pairing *is* Contribution 3.

**Libraries.** `tokenizers`.

**Advantages.** Turns Gap 4 from a hazard into an instrument.
**Disadvantages.** Timing-based inference of KV reuse is indirect.
**Recommendation:** prototype a second provider on `llama-cpp-python` in Sprint 4. In-process bindings expose
prompt-cache state directly, giving a *direct* measurement of Contribution 3 rather than one inferred from
latency. Worth the day it costs, for a headline contribution.

---

## M5 — Adaptive Output Budgeter

**Objective.** Cut generated tokens — the side that dominates CPU wall-clock — without flattening answers that
were already terse.

**reads** `response_class, complexity` · **writes** `output_budget`
**Emits** `ContextPatch(kind=DECIDE)`.

**Per-class budgets, never global.** A uniform length hint is documented to cut output 74–86 % on verbose
models *and* to regress quality on terse ones (report §2.2). A global cap would produce a quality drop that
looks like a compression failure and would be attributed to the wrong module in the ablation.

| Class | `num_predict` | Early-stop rule |
|---|---|---|
| arithmetic | 48 | terminal punctuation after a numeral |
| factual | 128 | trigram novelty |
| follow-up | 160 | trigram novelty |
| code | 512 | closing fence |
| reasoning | 640 | trigram novelty, wide window |
| summarisation | 256 | target sentence count reached |

**Early stop — trigram novelty ratio.** Over a sliding window of the last N generated tokens, the fraction of
trigrams not seen earlier in the response. Below threshold ⇒ the model is restating ⇒ stop. O(1) per token
with a rolling set. Sentence-level embedding checks run at most every ~3 sentences; a per-token embedding
pass would cost more than the generation it saves.

**Libraries.** stdlib (`collections.deque`, sets). Deliberately dependency-free — this runs per token.

**Advantages.** The only module acting on the 82 %-of-wall-clock side of Figure 2, so its latency effect
should be the largest in the stack. Also the easiest module to demo convincingly.
**Disadvantages.** Truncation risk on long-tail answers. Mitigation: the gold subset's exact-match score is
the tripwire; an early-stop threshold that improves latency while dropping exact-match is a bad operating
point and the Pareto frontier will show it.

**Alternatives.** Prompt-level length instruction ("answer in ≤50 words") instead of `num_predict` — worth a
comparison arm, since it costs input tokens but yields more coherent short answers than hard truncation. Cheap
to add, and a nice extra row in the results table.

---

## M6 — Escalation Router

**Objective.** Serve each query from the cheapest sufficient tier, including the tier the routing literature
omits: no model at all.

**reads** `response_class, query, invariants` · **writes** `route_tier`
**Emits** `ShortCircuit(served_by=DETERMINISTIC)` at tier 0, else `ContextPatch(kind=DECIDE)`.

| Tier | Handler | Model tokens |
|---|---|---|
| 0 | Deterministic: arithmetic, unit/currency conversion, date arithmetic, M7 template lookup | **0** |
| 1 | Cache (M2) | **0** |
| 2 | 1B model | baseline |
| 3 | 3B model, for classifier-flagged reasoning | higher |

**Split across the pipeline.** M6a (tier 0) runs *before* the cache — a regex match costs ~0.1 ms against the
cache's ~5 ms embedding pass, so checking the cheaper predicate first is strictly better. M6b (model tier)
runs after assembly, when the final token count is known and can inform the decision.

**Determinism is the point.** Tier 0 uses exact evaluation (`Decimal` arithmetic, `pint` for units, `dateutil`
for dates) — never an LLM. An arithmetic answer from tier 0 is correct by construction, which is a stronger
claim than any 1B model can make on the same input, and it makes "zero model tokens, higher accuracy" a
demonstrable headline in the demo.

**Libraries.** stdlib `decimal`/`fractions`; `pint`; `python-dateutil`.

**Advantages.** Restores routing to a setting with no paid tier to escalate to; adds the missing zero-cost
tier.
**Disadvantages.** Tier-0 coverage is narrow — expect < 15 % of the corpus. Report coverage honestly rather
than tuning the corpus toward it (the report's own risk register flags corpus-tuning as an anticipated
criticism).

**Alternatives.** RouteLLM-style learned router (log the features now, fit later — the interface already
supports it because Stage 6 persists every feature whether used or not).

---

## M7 — Conversation-Mined Policy Learner

**Objective.** Turn the user's own chat logs into a warm start. **Offline program, not a pipeline stage.**

**Input.** Exported chat logs (ChatGPT/Claude JSON export, or Parsimony's own ledger).
**Output.** `PolicyBundle/` — content-hashed directory:

```
PolicyBundle/
├── cache_seed.jsonl      # recurring Q→A pairs, pre-embedded
├── redundancy.txt        # phrases this user writes that provably never change the answer
├── digest.md             # standing facts re-explained every session → M4's invariant zone
├── templates.jsonl       # recurring query shapes → M6 tier 0
├── config.json           # tuned ParsimonyConfig
└── MANIFEST.sha256
```

**Counterfactual replay** is the mechanism, and it is what makes the redundancy lexicon *provable* rather
than heuristic: for each logged turn, re-run it with candidate phrase X removed, and compare the response
against the logged original under the four quality measures. A phrase whose removal never changes the answer
across ≥ N occurrences enters the lexicon. This is expensive (one model call per candidate per occurrence) —
hence offline, hence batched, hence run overnight.

**Advantages.** Directly answers Gap 6; converts "cache is cold on day one" from a limitation into a result.
Because it replays *real logged traffic* rather than a synthetic suite, it doubles as the tuning harness for
every threshold in the system.
**Disadvantages.** Requires the team to have exportable logs of sufficient volume. **Contingency:** if
personal logs are thin, generate a synthetic-but-realistic log corpus from the 150-conversation corpus and
state clearly that the warm-start result is demonstrated on synthetic logs. That is an honest, defensible
fallback and it should be decided *early*, not in week 11.

**Privacy.** Runs fully locally; PII-redacted before anything is written. Nothing leaves the machine — a
stated project property that must be true in code, not just in the report.

---

## M8 — Fidelity Gate

**Objective.** Reject any transformation that silently changes meaning. **Always on, never ablated** — it is
instrumentation and safety, not optimisation.

**Invoked by the orchestrator on every proposal**, not by modules. Modules cannot bypass it, forget it, or
implement it inconsistently.

```python
def check(before: RequestContext, after: RequestContext, kind: TransformKind) -> Verdict:
    match kind:
        case REWRITE:
            lost = before.invariants.missing_from(after.text_payload())
            return Verdict.fail(lost) if lost else Verdict.ok()
        case SELECT:
            return Verdict.ok() if retained_units_byte_identical(before, after) \
                   else Verdict.fail("retained unit mutated")
        case AUGMENT | DECIDE:
            return Verdict.ok()
```

**Cost.** Because `Invariants` was extracted once in Stage 2, each check is a frozenset difference —
microseconds. With up to seven proposals per request, extracting invariants per check instead would cost
~70 ms of the 120 ms budget. This is the single clearest example of why "compute derived values once" is an
architectural decision rather than an optimisation.

**Second responsibility — PII redaction at write boundaries.** Applied before any cache write and before any
ledger blob write. Detection happened in Stage 2; redaction happens here, at the boundary, so the model still
receives the user's real text while persistent stores never do.

**What to report.** Gate-fire rate per module, and the *class* of invariant violated (number / entity /
negation / quoted). "The gate fires on 11 % of tier-3 rewrites, 70 % of which are numerals" is a
characterisation of what safe compression means at 1B scale — a direct contribution to the calibration table,
and one that no amount of aggregate token-reduction reporting would surface.

**Libraries.** `spaCy`, `regex`. **Alternatives.** NLI-model-based entailment checking (rejected — a neural
forward pass on the critical path, the same self-defeating-overhead argument that disqualifies the
perplexity-based compressor; worth an offline validation experiment on a sample to check the cheap gate's
agreement with an NLI oracle, which would be a nice methodological footnote).
