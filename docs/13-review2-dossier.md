# Abstract

Language models charge by the token and, on a CPU-only laptop, spend almost all of their time reading the
prompt rather than writing the answer. A literature has grown around reducing that cost — prompt
compression, semantic caching, KV-cache reuse, model routing, output budgeting — but it has grown in
**isolated strands**. Each technique is proposed, measured against an uncompressed baseline, and published
alone.

This project surveys **40 papers** across eight strands, extracts their limitations with respect to a
CPU-only single-user deployment, and derives six research gaps. It then builds **Parsimony**: a middleware
layer of seven optimisation modules plus an always-on fidelity gate, in which every module *proposes* a
change and a single orchestrator *commits* it only after a safety check. Because every module is
independently switchable, the whole system is a 2⁴ factorial experiment rather than a fixed pipeline.

Measured over 151 conversations and 263 requests against `qwen2.5:1.5b-instruct` running locally on CPU, the
full stack removes **33.9%** of tokens with **no loss of answer accuracy** (92.5% → 97.5% on 40 gold items,
zero regressions). Three findings are new. **Savings do not compound**: an additivity shortfall of 1.63
percentage points, and its size is a property of the configuration rather than a constant. **Prefill
dominates CPU inference** at 92–99% of total time, ~8.5 ms per input token, making input reduction worth far
more than the literature's GPU-centric framing implies. And **the cache thresholds published as safe are
unsafe here**: an adversarial negation pair scores higher than every genuine paraphrase in our set, so the
verifier — not the threshold — is what makes reuse safe, taking the false-answer rate from 26.7% to 0.0%.

---

# 1. Literature Survey

Forty papers were reviewed across eight strands. The table records, for each: the **method** it proposes,
its **limitation with respect to this project's setting**, and the **metrics** it reports.

> **How to read the limitation column.** These are limitations *relative to a CPU-only, single-user,
> multi-turn deployment with several techniques composed together*. They are not criticisms of the papers on
> their own terms — most are datacentre-serving or single-technique works, and are sound in that context.
> The gap this project addresses is precisely the setting they do not target.

## 1.1 Prompt compression

| Ref | Method | Limitation w.r.t. this work | Metrics reported |
|---|---|---|---|
| [1] | Selective Context: drop low self-information lexical units, scored by a small LM | Needs a second model in the loop; measured alone, never against a cache | latency, memory, task score |
| [2] | LLMLingua: budget controller + token-level iterative compression | Up to 20× compression, but on GPU and in isolation; no interaction study | compression ratio, task score |
| [3] | LongLLMLingua: conditional perplexity for long contexts | Targets very long contexts; a laptop assistant rarely reaches them | speedup, task score |
| [4] | LLMLingua-2: data distillation for task-agnostic compression | Faithfulness targeted but not adversarially tested | compression, faithfulness |
| [5] | Evaluator heads for long-context compression | Requires access to attention internals; not available through a served API | latency, accuracy |
| [6] | SCOPE: generative rather than extractive compression | Generation cost added to the request being compressed | compression, quality |
| [7] | Gisting / in-context compression | Trained gist tokens; needs fine-tuning, out of reach on target hardware | compression, task score |
| [8] | UltraGist: lengthy-context compression | Long-context focus; training required | compression, accuracy |
| [9] | ATACompressor: adaptive task-aware compression | Task-conditioned; assumes task labels are known | compression, task score |
| [10] | SARA: selective+adaptive retrieval with compression | RAG-specific; assumes a retrieval corpus | retrieval quality |
| [11] | PCToolkit: unified compression toolkit | A harness, not a method; does not compose across technique families | comparative scores |
| [12] | Empirical study of prompt compression across three LLMs | **Names two failure modes** (altered-semantic and information-loss hallucination) but proposes no run-time guard | hallucination rates |
| [13] | Information preservation in prompt compression | Measures ~30–50 point groundedness drops; diagnosis without a mechanism to prevent it | groundedness |
| [14] | Fidelity loss in compressed financial analysis | Shows fluent, plausible, decision-changing compression | decision agreement |
| [15] | Compression paradox: provider-dependent energy effects | **Compression does not uniformly reduce energy** — measured on hosted APIs, not CPU | energy per request |

## 1.2 Semantic caching

| Ref | Method | Limitation w.r.t. this work | Metrics reported |
|---|---|---|---|
| [16] | GPTCache: embed the query, serve above a similarity threshold | Single threshold; no verification step; thresholds quoted without adversarial validation | hit rate, latency, cost |
| [17] | GPT Semantic Cache over Redis | Same single-threshold design at a different storage layer | hit rate, latency |
| [18] | MeanCache: user-centric federated semantic cache | Introduces a context chain; hit rate inside a conversation not examined | hit rate, privacy |
| [19] | ContextCache: context-aware multi-turn caching | Closest to our setting; does not report false-hit rate on adversarial pairs | hit rate |
| [20] | Generative caching for LLMs | Synthesises rather than reuses; adds generation cost | hit rate, quality |
| [21] | Key collision attack on LLM semantic caching | Treats collisions as an **adversarial search** problem; ordinary confusable pairs are not the threat model | attack success rate |
| [22] | SAFE-CACHE: cluster-centroid caching for adversarial resilience | Concludes vector similarity is insufficient — and answers with a redesign rather than a cheap verifier | robustness, hit rate |

## 1.3 KV cache and prefix reuse

| Ref | Method | Limitation w.r.t. this work | Metrics reported |
|---|---|---|---|
| [23] | PagedAttention / vLLM: OS-style paged KV memory | GPU serving; the application-side cost of an unstable prefix is not priced | throughput, memory waste |
| [24] | vLLM automatic prefix caching: hash KV blocks, share prefixes | States the stability requirement; does not measure what violating it costs an application | reuse rate |
| [25] | ChunkAttention: prefix-aware KV cache, two-phase partition | Kernel-level; assumes a served multi-tenant workload | throughput |
| [26] | Sparse prefix caching for hybrid/recurrent serving | Architecture-specific | throughput |
| [27] | Multi-segment attention for KV management | Serving-side; no application-level guidance | latency, memory |

## 1.4 Positional effects in context

| Ref | Method | Limitation w.r.t. this work | Metrics reported |
|---|---|---|---|
| [28] | **Lost in the Middle**: accuracy peaks when relevant content is at the beginning or end | Motivates reordering by relevance — which **destroys prefix reuse**, a conflict neither strand names | QA accuracy by position |
| [29] | Adaptive Focus Memory: per-message fidelity levels | Heuristic tiers; no measurement of the cost of reordering | token usage |
| [30] | Survey of multi-turn interaction capabilities | Survey; no composed measurement | mixed |
| [31] | Survey of memory mechanisms for LLMs | Survey; taxonomy rather than measurement | mixed |

## 1.5 Routing and cascades

| Ref | Method | Limitation w.r.t. this work | Metrics reported |
|---|---|---|---|
| [32] | FrugalGPT: prompt adaptation, approximation, LLM cascade | Up to 98% cost cut **against commercial APIs**; assumes a large capable tier exists | cost, accuracy |
| [33] | RouteLLM: routers trained on human preference data | Assumes the strong model is meaningfully better — untrue at 1.5B vs 3B on our set | cost, benchmark scores |
| [34] | Semantic router for vLLM | Serving-side routing; datacentre assumption | latency, cost |
| [35] | UCCI: calibrated uncertainty for cascade routing | Requires calibrated uncertainty; expensive on CPU | cost, accuracy |
| [36] | Uncertainty-based **on-device** LLM routing | Closest setting; routes between device and cloud, not within a local stack | accuracy, offload rate |
| [37] | Cluster, route, escalate: cost-aware serving | Cluster-level; no fidelity guarantee on the cheap path | cost, quality |

## 1.6 Output length control

| Ref | Method | Limitation w.r.t. this work | Metrics reported |
|---|---|---|---|
| [38] | Length-difference positional encoding with a countdown | **Requires fine-tuning** — out of reach on the target hardware | length adherence |
| [39] | BudgetThinker: budget-aware generation with control tokens | Requires training with control tokens | budget adherence, accuracy |
| [40] | Reasoning under strict output-length constraints | Characterises the cost of truncation; no per-class budgeting scheme | accuracy vs length |

## 1.7 The papers most relevant to this project

Four of the forty sit closest to what this work does, and each is the direct antecedent of one contribution.

**[28] Lost in the Middle (TACL 2024)** and **[24] vLLM automatic prefix caching (SOSP-lineage)** are the pair
this project measures *against each other*. The first says put relevant content at the prompt's edges. The
second requires the prompt head to be byte-stable across turns. **Doing the first destroys the second**, and
neither literature names the conflict. §6.2 prices it at ~80×.

**[16] GPTCache (NLP-OSS 2023)** established the design every semantic cache now follows: embed, compare,
serve above a threshold. Our adversarial set shows that threshold is the wrong place to put the safety
(§6.3), which is the same conclusion **[22] SAFE-CACHE (Scientific Reports 2026)** reaches independently — but
reached here with four set comparisons costing microseconds rather than a cluster-centroid redesign.

**[12] An Empirical Study on Prompt Compression (ICLR 2025)** is the closest antecedent of the fidelity gate.
It names the two failure modes — altered-semantic and information-loss hallucination — and measures how often
they occur. It does not propose a run-time mechanism to *prevent* them. M8 is that mechanism: every proposed
edit is checked for lost numbers, entities, negations and operative modifiers before it is committed.

---

# 2. Research Gaps

Six gaps follow from the limitation column, in the order the survey exposes them.

**Gap 1 — Compositional effects are unmeasured.** Every technique in §1.1–§1.6 is evaluated alone against an
uncompressed baseline. Whether their savings *add* when composed is unknown, because no study runs them in
one pipeline.

**Gap 2 — The prefill/decode split is uncharacterised for CPU-only single-user inference.** The energy and
serving literature measures GPU datacentre throughput. Which half of the request a laptop actually pays for
is not established.

**Gap 3 — Compression × cache interaction.** Normalising a query before the cache sees it should raise the
hit rate. Whether it also raises the *false* hit rate is unexamined; §1.1 and §1.2 do not cite each other's
failure modes.

**Gap 4 — Published cache thresholds are not adversarially validated at small scale.** The 0.85–0.92 range is
quoted as safe. Whether near-identical, opposite-meaning questions sit above or below it is not reported.

**Gap 5 — Whether a calibration transfers.** Thresholds are published as if universal. No work re-runs an
identical protocol on a second configuration without re-tuning.

**Gap 6 — Whether mined policy transfers to unseen conversations.** Self-improving systems are typically
evaluated on the data they were mined from, which measures memorisation rather than transfer.

---

# 3. Research Questions and Problem Statement

## 3.1 Research questions

**RQ1.** When several token-reduction techniques are composed in one pipeline, do their individual savings
add — and if not, by how much do they fall short?

**RQ2.** On CPU-only hardware, which half of a request dominates the cost, and what is one input token worth
in milliseconds?

**RQ3.** Can a semantic cache be made safe against near-identical, opposite-meaning questions without a
second neural forward pass?

**RQ4.** Does a calibration — thresholds, module settings — transfer to a different configuration without
re-tuning?

## 3.2 Problem statement

> Build a middleware layer that measurably reduces the token cost of interacting with a small language model
> on CPU-only consumer hardware, **without changing the meaning of any request**, and instrument it so that
> the contribution of every individual module, and every interaction between modules, is separately
> measurable and independently reproducible.

The emphasis is deliberate. A system that saves tokens by quietly damaging requests is not a contribution;
the safety property is the constraint under which every optimisation must operate.

---

# 4. Contributions

**Contribution 1 — The additivity shortfall, measured.** A full 2⁴ factorial over compressor × cache ×
history manager × output budgeter with bootstrap confidence intervals and partial η² effect sizes. Savings do
not compound: the shortfall is **1.63 pp, 95% CI [−0.02, +3.23]**. Further, its magnitude is a property of
the *configuration* — improving the encoder moved it from 2.53 pp to 1.63 pp, because a cache that hits more
often overlaps its neighbours less. (Answers RQ1, Gap 1.)

**Contribution 2 — The CPU cost structure, and the price of prompt order.** Prefill is **92–99%** of total
time, linear at **~8.5 ms per input token**. This converts every token result into wall clock and establishes
why input reduction is the thing worth doing. It also prices a conflict the literature does not name: a
volatile token at prompt position 0 costs **212 ms → 18,914 ms** in steady-state prefill for the same content.
(Answers RQ2, Gap 2.)

**Contribution 3 — A verifier that makes reuse safe without a second model.** An adversarial set of 45 pairs
one operative token apart, plus 45 controls. The negation pair sits at cosine **0.924 — above every genuine
paraphrase in the set** — so no threshold separates them. Four set comparisons costing microseconds take the
false-answer rate from **26.7% to 0.0%**, including three checks no surveyed cache performs: operative
modifiers, morphological negation, and alphanumeric identifiers. (Answers RQ3, Gaps 3 and 4.)

**Contribution 4 — Calibration transfer, tested both ways.** Re-running the protocol against a second real
vocabulary shows **reduction ratios transfer** (within 0.1 pp, identical module ranking) while **the
mechanisms behind them do not** — one of our own two explanations for an effect proved specific to one
tokenizer. Separately, mined policy transfer is shown to be a function of traffic repetition: **+0.00 pp at 0%
recurrence, +17.83 pp at 57%**, with zero fidelity violations at any level. (Answers RQ4, Gaps 5 and 6.)

---

# 5. Proposed Method

## 5.1 The governing design decision

Every module **proposes** a change; a single orchestrator **commits** it. A proposal is a description of an
edit, not the edit itself. Three properties follow that no module has to implement:

1. **Independent ablation.** Switching a module off genuinely removes its effect, so a factorial cell means
   what it claims.
2. **A single safety choke point.** Every edit passes one gate, so fidelity is not re-implemented seven
   times.
3. **Reverting is free.** A refused proposal is simply not assigned.

Stage order is **configuration, not code**, validated against a reads/writes dependency graph. Gap 3 is only
answerable because the cache lookup can be moved before or after the compressor without editing the pipeline.

## 5.2 The pipeline, tier by tier

Requests flow top to bottom. Each stage may short-circuit, propose an edit, or do nothing — and reports which.

<div style="text-align:center;margin:1.4em 0;">
<svg viewBox="0 0 620 552" width="100%" style="max-width:640px" xmlns="http://www.w3.org/2000/svg">
  <rect x="86" y="14" width="352" height="30" rx="4" fill="#eceff1" stroke="#37474f" stroke-width="1.1"/>
  <text x="262" y="33" font-size="10" font-weight="700" fill="#16181d" text-anchor="middle">USER QUESTION</text>
  <path d="M262 44 L262 56" stroke="#9aa0a6" stroke-width="1"/>
  <path d="M259 52 L262 56 L265 52" fill="#9aa0a6"/>
  
  <rect x="86" y="62" width="352" height="34" rx="4" fill="#ffffff" stroke="#1b5e20" stroke-width="1.1"/>
  <rect x="86" y="62" width="5" height="34" rx="2" fill="#1b5e20"/>
  <text x="100" y="76" font-size="9.5" font-weight="600" fill="#1b5e20">M6a</text>
  <text x="136" y="76" font-size="9.5" font-weight="600" fill="#16181d">Deterministic router</text>
  <text x="100" y="89" font-size="8.6" fill="#5a6069">Maths / dates answered exactly</text>
  <text x="430" y="83" font-size="8.4" fill="#5a6069" text-anchor="end">23 tokens → 0</text>
  <path d="M262 96 L262 104" stroke="#9aa0a6" stroke-width="1"/>
  <path d="M259 101 L262 105 L265 101" fill="#9aa0a6"/>
  <rect x="86" y="104" width="352" height="34" rx="4" fill="#ffffff" stroke="#1b5e20" stroke-width="1.1"/>
  <rect x="86" y="104" width="5" height="34" rx="2" fill="#1b5e20"/>
  <text x="100" y="118" font-size="9.5" font-weight="600" fill="#1b5e20">M2</text>
  <text x="136" y="118" font-size="9.5" font-weight="600" fill="#16181d">Semantic cache</text>
  <text x="100" y="131" font-size="8.6" fill="#5a6069">Reuse an earlier answer, verified</text>
  <text x="430" y="125" font-size="8.4" fill="#5a6069" text-anchor="end">prompt → stored answer</text>
  <path d="M262 138 L262 146" stroke="#9aa0a6" stroke-width="1"/>
  <path d="M259 143 L262 147 L265 143" fill="#9aa0a6"/>
  <rect x="86" y="146" width="352" height="34" rx="4" fill="#ffffff" stroke="#0d47a1" stroke-width="1.1"/>
  <rect x="86" y="146" width="5" height="34" rx="2" fill="#0d47a1"/>
  <text x="100" y="160" font-size="9.5" font-weight="600" fill="#0d47a1">M3a</text>
  <text x="136" y="160" font-size="9.5" font-weight="600" fill="#16181d">History selector</text>
  <text x="100" y="173" font-size="8.6" fill="#5a6069">Keep only relevant prior turns</text>
  <text x="430" y="167" font-size="8.4" fill="#5a6069" text-anchor="end">8 turns → 6</text>
  <path d="M262 180 L262 188" stroke="#9aa0a6" stroke-width="1"/>
  <path d="M259 185 L262 189 L265 185" fill="#9aa0a6"/>
  <rect x="86" y="188" width="352" height="34" rx="4" fill="#ffffff" stroke="#0d47a1" stroke-width="1.1"/>
  <rect x="86" y="188" width="5" height="34" rx="2" fill="#0d47a1"/>
  <text x="100" y="202" font-size="9.5" font-weight="600" fill="#0d47a1">M3b</text>
  <text x="136" y="202" font-size="9.5" font-weight="600" fill="#16181d">History arranger</text>
  <text x="100" y="215" font-size="8.6" fill="#5a6069">Order the kept turns</text>
  <text x="430" y="209" font-size="8.4" fill="#5a6069" text-anchor="end">order changes cost</text>
  <path d="M262 222 L262 230" stroke="#9aa0a6" stroke-width="1"/>
  <path d="M259 227 L262 231 L265 227" fill="#9aa0a6"/>
  <rect x="86" y="230" width="352" height="34" rx="4" fill="#ffffff" stroke="#4a148c" stroke-width="1.1"/>
  <rect x="86" y="230" width="5" height="34" rx="2" fill="#4a148c"/>
  <text x="100" y="244" font-size="9.5" font-weight="600" fill="#4a148c">M1a</text>
  <text x="136" y="244" font-size="9.5" font-weight="600" fill="#16181d">Tier 1 — boilerplate</text>
  <text x="100" y="257" font-size="8.6" fill="#5a6069">Strip greetings and politeness</text>
  <text x="430" y="251" font-size="8.4" fill="#5a6069" text-anchor="end">28 → 17 tokens</text>
  <path d="M262 264 L262 272" stroke="#9aa0a6" stroke-width="1"/>
  <path d="M259 269 L262 273 L265 269" fill="#9aa0a6"/>
  <rect x="86" y="272" width="352" height="34" rx="4" fill="#ffffff" stroke="#4a148c" stroke-width="1.1"/>
  <rect x="86" y="272" width="5" height="34" rx="2" fill="#4a148c"/>
  <text x="100" y="286" font-size="9.5" font-weight="600" fill="#4a148c">M1b</text>
  <text x="136" y="286" font-size="9.5" font-weight="600" fill="#16181d">Tier 2 — redundancy</text>
  <text x="100" y="299" font-size="8.6" fill="#5a6069">Delete a repeated fact</text>
  <text x="430" y="293" font-size="8.4" fill="#5a6069" text-anchor="end">36 → 26 tokens</text>
  <path d="M262 306 L262 314" stroke="#9aa0a6" stroke-width="1"/>
  <path d="M259 311 L262 315 L265 311" fill="#9aa0a6"/>
  <rect x="86" y="314" width="352" height="34" rx="4" fill="#ffffff" stroke="#4a148c" stroke-width="1.1"/>
  <rect x="86" y="314" width="5" height="34" rx="2" fill="#4a148c"/>
  <text x="100" y="328" font-size="9.5" font-weight="600" fill="#4a148c">M1c</text>
  <text x="136" y="328" font-size="9.5" font-weight="600" fill="#16181d">Tier 3 — word level</text>
  <text x="100" y="341" font-size="8.6" fill="#5a6069">Shorten wording, guarded</text>
  <text x="430" y="335" font-size="8.4" fill="#5a6069" text-anchor="end">rejects zero-yield edits</text>
  <path d="M262 348 L262 356" stroke="#9aa0a6" stroke-width="1"/>
  <path d="M259 353 L262 357 L265 353" fill="#9aa0a6"/>
  <rect x="86" y="356" width="352" height="34" rx="4" fill="#ffffff" stroke="#b71c1c" stroke-width="1.1"/>
  <rect x="86" y="356" width="5" height="34" rx="2" fill="#b71c1c"/>
  <text x="100" y="370" font-size="9.5" font-weight="600" fill="#b71c1c">M4</text>
  <text x="136" y="370" font-size="9.5" font-weight="600" fill="#16181d">Prefix-stable assembler</text>
  <text x="100" y="383" font-size="8.6" fill="#5a6069">Invariant text first</text>
  <text x="430" y="377" font-size="8.4" fill="#5a6069" text-anchor="end">212 ms vs 18,914 ms</text>
  <path d="M262 390 L262 398" stroke="#9aa0a6" stroke-width="1"/>
  <path d="M259 395 L262 399 L265 395" fill="#9aa0a6"/>
  <rect x="86" y="398" width="352" height="34" rx="4" fill="#ffffff" stroke="#e65100" stroke-width="1.1"/>
  <rect x="86" y="398" width="5" height="34" rx="2" fill="#e65100"/>
  <text x="100" y="412" font-size="9.5" font-weight="600" fill="#e65100">M5</text>
  <text x="136" y="412" font-size="9.5" font-weight="600" fill="#16181d">Output budgeter</text>
  <text x="100" y="425" font-size="8.6" fill="#5a6069">Cap answer length by class</text>
  <text x="430" y="419" font-size="8.4" fill="#5a6069" text-anchor="end">48 … 640 tokens</text>
  <path d="M262 432 L262 440" stroke="#9aa0a6" stroke-width="1"/>
  <path d="M259 437 L262 441 L265 437" fill="#9aa0a6"/>
  <rect x="86" y="440" width="352" height="34" rx="4" fill="#ffffff" stroke="#37474f" stroke-width="1.1"/>
  <rect x="86" y="440" width="5" height="34" rx="2" fill="#37474f"/>
  <text x="100" y="454" font-size="9.5" font-weight="600" fill="#37474f">M6b</text>
  <text x="136" y="454" font-size="9.5" font-weight="600" fill="#16181d">Escalation router</text>
  <text x="100" y="467" font-size="8.6" fill="#5a6069">Bigger model needed?</text>
  <text x="430" y="461" font-size="8.4" fill="#5a6069" text-anchor="end">calibrated, off by default</text>
  <path d="M262 474 L262 482" stroke="#9aa0a6" stroke-width="1"/>
  <path d="M259 479 L262 483 L265 479" fill="#9aa0a6"/>
  <rect x="86" y="486" width="352" height="30" rx="4" fill="#eceff1" stroke="#37474f" stroke-width="1.1"/>
  <text x="262" y="505" font-size="10" font-weight="700" fill="#16181d" text-anchor="middle">PROMPT SENT TO THE MODEL</text>

  <rect x="456" y="62" width="150" height="412" rx="5" fill="#fff8e1" stroke="#b8860b" stroke-width="1.3"/>
  <text x="531" y="84" font-size="9.6" font-weight="700" fill="#7a5c00" text-anchor="middle">M8 · FIDELITY GATE</text>
  <text x="531" y="102" font-size="8.3" fill="#5a4a00" text-anchor="middle">Checks EVERY proposal</text>
  <text x="531" y="116" font-size="8.3" fill="#5a4a00" text-anchor="middle">above before it is applied.</text>
  <text x="531" y="136" font-size="8.3" fill="#5a4a00" text-anchor="middle">Refuses any edit that</text>
  <text x="531" y="150" font-size="8.3" fill="#5a4a00" text-anchor="middle">would drop a number,</text>
  <text x="531" y="164" font-size="8.3" fill="#5a4a00" text-anchor="middle">name, negation or</text>
  <text x="531" y="178" font-size="8.3" fill="#5a4a00" text-anchor="middle">qualifier.</text>
  <text x="531" y="202" font-size="8.3" font-weight="600" fill="#7a5c00" text-anchor="middle">False-answer rate</text>
  <text x="531" y="216" font-size="8.3" font-weight="600" fill="#7a5c00" text-anchor="middle">26.7% → 0.0%</text>
  <path d="M446 152 L456 152" stroke="#b8860b" stroke-width="1.2"/>
  <path d="M450 148 L446 152 L450 156" fill="#b8860b"/>

  <text x="20" y="132" font-size="8.4" fill="#5a6069" transform="rotate(-90 20 132)" text-anchor="middle">MODULES PROPOSE &#183; ORCHESTRATOR COMMITS</text>
</svg>
<div style="font-size:8.6pt;color:#5a6069;margin-top:.4em;">
<strong>Figure 1.</strong> The request pipeline. Each tier may answer outright, propose an edit, or do
nothing &mdash; and records which. Every proposal passes the fidelity gate before it is committed.
</div>
</div>


| # | Tier | What it actually does | Input → Output | Why it exists |
|---|---|---|---|---|
| 1 | **M6a — Deterministic router** | Recognises arithmetic and date questions and computes the answer exactly, with an AST evaluator rather than `eval` | question → answer, or pass through | The cheapest request is the one never sent. 23 tokens → **0**. |
| 2 | **M2 — Semantic cache** | Embeds the question, retrieves near neighbours, and applies a **three-zone policy**: accept, reject, or *verify*. Borderline matches are settled by four set comparisons over numbers, entities, negations and operative modifiers | question → stored answer, or pass through | Reuse without a second model call — and without serving the opposite answer. |
| 3 | **M3a — History selector** | Scores prior turns for relevance with MMR and keeps only those that fit the budget | N turns → M turns | In a long conversation most turns are irrelevant to the current question. |
| 4 | **M3b — History arranger** | Orders the kept turns | turns → ordered turns | Where a turn sits changes both attention and prefix reuse (§6.2). |
| 5 | **M1a — Tier 1, boilerplate** | Removes greetings, politeness and markdown, losslessly and sentence-aware | text → shorter text | Politeness carries no instruction and costs real tokens. |
| 6 | **M1b — Tier 2, redundancy** | Deletes a sentence that restates a fact already present, above a similarity threshold calibrated from data | text → shorter text | The same fact stated twice is paid for twice. |
| 7 | **M1c — Tier 3, word level** | Shortens long-winded phrasing, with a **negative-yield guard** that rejects any edit not reducing the token count | text → shorter text | Shortening text does not always shorten *tokens*; the guard measures rather than assumes. |
| 8 | **M4 — Prefix-stable assembler** | Assembles the final prompt with invariant content first and volatile content last | parts → prompt | Keeps the prompt head byte-stable so the model reuses its KV cache. Worth ~0 tokens and ~18 seconds. |
| 9 | **M5 — Output budgeter** | Classifies the question and sets `num_predict` accordingly — 48 for arithmetic, 640 for reasoning — plus a streaming early stop on trigram novelty | question → token budget | Small models ramble. Right-sizing beats truncating. |
| 10 | **M6b — Escalation router** | Scores complexity and decides whether a larger model is warranted | question → route | Calibrated to the observed distribution, and deliberately **off**: measurement showed a 3B model scores item-for-item identically to 1.5B here. |
| — | **M8 — Fidelity gate** | **Runs on every proposal from every stage.** Compares invariants before and after; refuses any edit that would drop a number, entity, negation or operative modifier, or that would empty the payload | proposal → commit / refuse | The constraint under which all of the above operate. |
| — | **M7 — Policy learner** | Offline: mines recurring questions, standing facts and templates from past conversations into a `PolicyBundle` that warm-starts the cache | logs → bundle | Value is a property of the traffic (§6.5), not of the module. |

## 5.3 What the ledger records

Every stage writes one row: module, outcome (applied / reverted / short-circuit / no-op), tokens before and
after, duration, rationale, and any gate event. **A stage that did nothing still writes a row** — an invisible
stage is an unauditable one, and the whole evaluation depends on the trace being complete.

---

# 6. Results and Discussion

All figures regenerate from raw logs with a single command in about 100 seconds. **719 automated tests pass.**

## 6.1 The headline: savings do not compound

Full 2⁴ factorial, 151 conversations, 263 requests, 17 cells.

| Effect | Estimate | Partial η² |
|---|---|---|
| M5 output budgeter | +13.44 pp | 0.556 |
| M3 history manager | +11.82 pp | 0.430 |
| M2 semantic cache | +1.94 pp | 0.012 |
| M1 compressor | +0.23 pp | 0.000 |
| M3 × M5 interaction | **−0.70 pp** | 0.002 |

Full stack: **+33.9%** total token reduction. Both material interaction terms are negative and both involve
M5 — trimming history and shortening output reduce the same conversation.

> **Additivity shortfall: 1.63 pp, 95% CI [−0.02, +3.23].**

And the sharper result: the shortfall is **configuration-dependent**.

| Encoder | M2 effect | M3×M5 | Additivity shortfall |
|---|---|---|---|
| `hashing-v1` | +1.61 pp | −1.14 | 2.53 pp, [+0.93, +3.99] |
| `content-v1` (default) | +1.94 pp | −0.70 | 1.63 pp, [−0.02, +3.23] |

Improving the encoder made the cache hit more often, so it overlapped its neighbours less. The weaker encoder
was **not** reinstated to protect the interval.

## 6.2 Prefill dominates, and prompt order has a price

Measured on `qwen2.5:1.5b-instruct` (Q4_K_M), Ryzen 7 5800HS, reading the server's own prefill/decode split.

| Input tokens | Prefill | Decode | Prefill share |
|---|---|---|---|
| 146 | 1,360 ms | 122 ms | 91.7% |
| 474 | 3,902 ms | 165 ms | 95.9% |
| 1,338 | 11,609 ms | 136 ms | 98.8% |

**~8.5 ms per input token.** The full stack's 11,836 saved input tokens are **100.6 s of prefill across the
corpus, 383 ms per request**.

| Prompt arrangement | Tokens | Steady-state prefill | Prefix reuse |
|---|---|---|---|
| Stable prefix (M4) | 1,490 | **212 ms** | **98.5%** |
| Volatile head | 1,497 | **18,914 ms** | **0.5%** |

Same content, 0.5% apart in token count, **~80× apart in cost**.

## 6.3 The verifier, not the threshold

45 adversarial pairs plus 45 controls.

| Cosine | Pair |
|---|---|
| **0.924** | "Is it safe to mix bleach and vinegar?" / "Is it **not** safe…" |
| 0.869 | "capital of **Australia**" / "capital of **Austria**" |
| 0.653 | "What causes rain?" / "What causes rainfall to occur?" |

The adversarial pair scores **above every genuine paraphrase**, so the published 0.85–0.92 range would
auto-accept it and return the opposite answer.

| Operative | Initial false-hit rate | After the verifier |
|---|---|---|
| modifier | 78% | **0%** |
| negation | 31% | **0%** |
| entity | 11% | **0%** |
| **overall** | **26.7%** | **0.0%** |

## 6.4 Quality is not the price

40 gold items, real model.

| Configuration | Gold accuracy |
|---|---|
| Baseline | 37/40 — 92.5% |
| Full stack | **39/40 — 97.5%** |

Paired per item: **zero regressions.** Not one answer the baseline got right was lost to compression.

## 6.5 Transfer

**Across vocabularies:** reduction ratios transfer within 0.1 pp with identical module ranking; the
*mechanisms* do not — one of our two stated explanations for a tokenizer effect proved Qwen-specific.

**Across conversations:** M7's value as a function of traffic repetition —

| Traffic recurrence | Transfer | Extra cache hits | Extra gate fires |
|---|---|---|---|
| 0.0% | **+0.00 pp** | 0 | 0 |
| 22.5% | +5.43 pp | +4 | 0 |
| 57.5% | **+17.83 pp** | +12 | 0 |

The exact zero at 0% recurrence is what makes the rest credible.

## 6.6 Threats to validity

1. **Single hardware configuration.** The 8.5 ms/token rate is not claimed to generalise.
2. **Small gold set.** 40 items; after a grading fix only two defeat both models, and both are arithmetic
   already handled without a model — so §6.4 measures the gold set as much as the pipeline.
3. **Lexical encoder.** Operating points shift with the encoder; rankings do not.
4. **Corpus recurrence is 1.9%**, which is why §6.5 reports a curve over synthetic traffic rather than one
   number. Repetition structure is synthetic; every question and answer in it is real.

---

# 7. Conclusion

The efficiency literature has produced a rich toolbox and a thin account of what happens when the tools are
used together. This project's contribution is not a new compression algorithm but a **measurement
instrument**: eight techniques in one harness, each independently switchable, every decision written to an
auditable ledger.

What that instrument found, repeatedly, is that **published operating points are configuration-specific**. A
threshold quoted as safe is unsafe here. A placement strategy justified by a well-cited attention result is
80× more expensive here. A calibration transfers as a ratio but not as a mechanism. An escalation premise
that holds at 70B does not hold at 3B.

For this class of system, the deliverable is a **calibration procedure, not a number**.

---

# 8. References

1. Li, Y., Dong, B., Lin, C., Guerin, F. *Compressing Context to Enhance Inference Efficiency of Large Language Models.* EMNLP 2023. arXiv:2310.06201
2. Jiang, H., Wu, Q., Lin, C.-Y., Yang, Y., Qiu, L. *LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models.* EMNLP 2023. arXiv:2310.05736
3. Jiang, H. et al. *LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression.* arXiv:2310.06839
4. *LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression.* arXiv:2403.12968
5. *Efficient Prompt Compression with Evaluator Heads for Long-Context Transformer Inference.* arXiv:2501.12959
6. *SCOPE: A Generative Approach for LLM Prompt Compression.* arXiv:2508.15813
7. *Long Context In-Context Compression by Getting to the Gist of Gisting.* arXiv:2504.08934
8. *Compressing Lengthy Context With UltraGist.* arXiv:2405.16635
9. *ATACompressor: Adaptive Task-Aware Compression for Efficient Long-Context Processing in LLMs.* arXiv:2602.03226
10. *SARA: Selective and Adaptive Retrieval-augmented Generation with Context Compression.* arXiv:2507.05633
11. *PCToolkit: A Unified Plug-and-Play Prompt Compression Toolkit of Large Language Models.* arXiv:2403.17411
12. *An Empirical Study on Prompt Compression for Large Language Models.* ICLR 2025. openreview.net/pdf?id=lbFVTPv4s6
13. *Understanding and Improving Information Preservation in Prompt Compression for LLMs.* arXiv:2503.19114
14. *When Summaries Distort Decisions: Information Fidelity in LLM-Compressed Financial Analysis.* arXiv:2606.29251
15. *The Compression Paradox in LLM Inference: Provider-Dependent Energy Effects of Prompt Compression.* arXiv:2603.23528
16. Bang, F. *GPTCache: An Open-Source Semantic Cache for LLM Applications.* NLP-OSS @ EMNLP 2023. aclanthology.org/2023.nlposs-1.24/
17. *GPT Semantic Cache: Reducing LLM Costs and Latency via Semantic Embedding Caching.* arXiv:2411.05276
18. *MeanCache: User-Centric Semantic Caching for LLM Web Services.* arXiv:2403.02694
19. *ContextCache: Context-Aware Semantic Cache for Multi-Turn Queries in Large Language Models.* arXiv:2506.22791
20. *A Generative Caching System for Large Language Models.* arXiv:2503.17603
21. *From Similarity to Vulnerability: Key Collision Attack on LLM Semantic Caching.* arXiv:2601.23088
22. *Enhancing adversarial resilience in semantic caching for secure retrieval augmented generation systems.* Scientific Reports, 2026. nature.com/articles/s41598-026-36721-w
23. Kwon, W. et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP 2023
24. vLLM. *Automatic Prefix Caching.* docs.vllm.ai/en/v0.7.0/design/automatic_prefix_caching.html
25. *ChunkAttention: Efficient Self-Attention with Prefix-Aware KV Cache and Two-Phase Partition.* arXiv:2402.15220
26. *Sparse Prefix Caching for Hybrid and Recurrent LLM Serving.* arXiv:2605.05219
27. *Multi-Segment Attention: Efficient KV-Cache Management for Faster LLM Serving.* arXiv:2606.02964
28. Liu, N. F. et al. *Lost in the Middle: How Language Models Use Long Contexts.* TACL 12:157–173, 2024. arXiv:2307.03172
29. *Adaptive Focus Memory for Language Models.* arXiv:2511.12712
30. *A Survey on Multi-Turn Interaction Capabilities of Large Language Models.* arXiv:2501.09959
31. *From Human Memory to AI Memory: A Survey on Memory Mechanisms in the Era of LLMs.* arXiv:2504.15965
32. Chen, L., Zaharia, M., Zou, J. *FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance.* arXiv:2305.05176
33. Ong, I. et al. *RouteLLM: Learning to Route LLMs with Preference Data.* arXiv:2406.18665
34. *When to Reason: Semantic Router for vLLM.* arXiv:2510.08731
35. *UCCI: Calibrated Uncertainty for Cost-Optimal LLM Cascade Routing.* arXiv:2605.18796
36. *Confident or Seek Stronger: Uncertainty-Based On-device LLM Routing.* arXiv:2502.04428
37. *Cluster, Route, Escalate: Cascaded Framework for Cost-Aware LLM Serving.* arXiv:2606.27457
38. *Precise length control for large language models.* Natural Language Processing Journal, Elsevier
39. *BudgetThinker: Empowering Budget-aware LLM Reasoning with Control Tokens.* arXiv:2508.17196
40. *An Empirical Study of LLM Reasoning Ability Under Strict Output Length Constraint.* arXiv:2504.14350

---

*Every URL was verified during preparation. Before submission, read each paper you cite in the report body:
a survey is only as trustworthy as the reading behind it.*
