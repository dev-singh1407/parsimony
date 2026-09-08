<div class="cover">
  <div class="uni">VIT UNIVERSITY</div>
  <div class="uni-sub">Vellore Institute of Technology</div>
  <div class="uni-sub2">Vellore, Tamil Nadu, India</div>
  <h1>Token-Efficient LLM Interaction<br>on CPU-Only Hardware</h1>
  <div class="sub">Parsimony &mdash; A Stacked, Self-Improving Optimisation Layer<br>for Small Language Models</div>
  <div class="course">Natural Language Processing Project</div>
  <div class="members-h">Team Members</div>
  <table class="team">
    <tr><td>Arrsh Tripathi</td><td>23BCI0191</td></tr>
    <tr><td>Alok Singh</td><td>23BCI0158</td></tr>
    <tr><td>Dev Singh</td><td>23BCE0794</td></tr>
  </table>
  <div class="foot">
    <span class="lab">Programme</span> B.Tech &nbsp;&middot;&nbsp; <span class="lab">Course Code</span> BCSE497J &nbsp;&middot;&nbsp; Project I<br>
    <span class="lab">Faculty Guide</span> Dr Sathya K<br>
    <span class="lab">Submission Date</span> September 2026
  </div>
</div>

# Abstract

Language models charge by the token and, on a CPU-only laptop, spend almost all of their time reading the
prompt rather than writing the answer. A literature has grown around reducing that cost — prompt compression,
semantic caching, key-value cache reuse, model routing, output budgeting — but it has grown in **isolated
strands**. Each technique is proposed, measured against an uncompressed baseline, and published alone.

This report surveys **40 papers** across eight strands, extracts the limitation of each with respect to a
CPU-only single-user deployment, and derives **six research gaps**. It then presents **Parsimony**: a
middleware layer of seven optimisation modules plus an always-on fidelity gate, in which every module
*proposes* a change and a single orchestrator *commits* it only after a safety check. Because every module is
independently switchable, the whole system is a 2⁴ factorial experiment rather than a fixed pipeline.

Measured over 151 conversations and 263 requests against `qwen2.5:1.5b-instruct` running locally on CPU, the
full stack removes **33.9%** of tokens with **no loss of answer accuracy** (92.5% → 97.5% on 40 gold items,
zero regressions). Three findings are new. **Savings do not compound**: an additivity shortfall of 1.63
percentage points whose size is a property of the configuration rather than a constant. **Prefill dominates
CPU inference** at 92–99% of total time, ~8.5 ms per input token, making input reduction worth far more than
the GPU-centric framing of the literature implies. And **the cache thresholds published as safe are unsafe
here**: an adversarial negation pair scores higher than every genuine paraphrase in our set, so the verifier —
not the threshold — is what makes reuse safe, taking the false-answer rate from 26.7% to 0.0%.

**Keywords:** prompt compression, semantic caching, KV-cache reuse, small language models, CPU inference,
factorial ablation, fidelity verification.

<div class="pagebreak"></div>

# 1. Literature Review

Forty papers were reviewed across eight strands. For each, the table below records the **method** it proposes,
its **limitation with respect to this project's setting**, and the **metrics** it reports.

> **How to read the limitation column.** These are limitations *relative to a CPU-only, single-user,
> multi-turn deployment with several techniques composed together*. They are not criticisms of the papers on
> their own terms — most are datacentre-serving or single-technique works, and are sound in that context. The
> gap this project addresses is precisely the setting they do not target.

## 1.1 Strand A — Prompt compression

| Ref | Method proposed | Limitation w.r.t. this work | Metrics reported |
|---|---|---|---|
| [1] | Selective Context: drop low self-information lexical units, scored by a small LM | Needs a second model in the loop; measured alone, never against a cache | latency, memory, task score |
| [2] | LLMLingua: budget controller + token-level iterative compression | Up to 20× compression, but on GPU and in isolation; no interaction study | compression ratio, task score |
| [3] | LongLLMLingua: conditional perplexity for long contexts | Targets very long contexts; a laptop assistant rarely reaches them | speedup, task score |
| [4] | LLMLingua-2: data distillation for task-agnostic compression | Faithfulness targeted but not adversarially tested | compression, faithfulness |
| [5] | Evaluator heads for long-context compression | Requires access to attention internals; unavailable through a served API | latency, accuracy |
| [6] | SCOPE: generative rather than extractive compression | Generation cost added to the very request being compressed | compression, quality |
| [7] | Gisting / in-context compression | Trained gist tokens; needs fine-tuning, out of reach on target hardware | compression, task score |
| [8] | UltraGist: lengthy-context compression | Long-context focus; training required | compression, accuracy |
| [9] | ATACompressor: adaptive task-aware compression | Task-conditioned; assumes task labels are known in advance | compression, task score |
| [10] | SARA: selective and adaptive retrieval with compression | RAG-specific; assumes a retrieval corpus exists | retrieval quality |
| [11] | PCToolkit: unified compression toolkit | A harness, not a method; does not compose across technique families | comparative scores |
| [12] | Empirical study of prompt compression across three LLMs | **Names two failure modes** (altered-semantic and information-loss hallucination) but proposes no run-time guard | hallucination rates |
| [13] | Information preservation in prompt compression | Measures ~30–50 point groundedness drops; diagnosis without a prevention mechanism | groundedness |
| [14] | Fidelity loss in compressed financial analysis | Shows compression that is fluent, plausible and decision-changing | decision agreement |
| [15] | Compression paradox: provider-dependent energy effects | **Compression does not uniformly reduce energy** — measured on hosted APIs, not CPU | energy per request |

## 1.2 Strand B — Semantic caching

| Ref | Method proposed | Limitation w.r.t. this work | Metrics reported |
|---|---|---|---|
| [16] | GPTCache: embed the query, serve above a similarity threshold | Single threshold; no verification step; thresholds quoted without adversarial validation | hit rate, latency, cost |
| [17] | GPT Semantic Cache over Redis | The same single-threshold design at a different storage layer | hit rate, latency |
| [18] | MeanCache: user-centric federated semantic cache | Introduces a context chain; hit rate *inside* a conversation not examined | hit rate, privacy |
| [19] | ContextCache: context-aware multi-turn caching | Closest to our setting; does not report false-hit rate on adversarial pairs | hit rate |
| [20] | Generative caching for LLMs | Synthesises rather than reuses; adds generation cost | hit rate, quality |
| [21] | Key-collision attack on LLM semantic caching | Treats collisions as an **adversarial search** problem; ordinary confusable pairs are not the threat model | attack success rate |
| [22] | SAFE-CACHE: cluster-centroid caching for adversarial resilience | Concludes vector similarity is insufficient — and answers with a redesign rather than a cheap verifier | robustness, hit rate |

## 1.3 Strand C — KV cache and prefix reuse

| Ref | Method proposed | Limitation w.r.t. this work | Metrics reported |
|---|---|---|---|
| [23] | PagedAttention / vLLM: OS-style paged KV memory | GPU serving; the application-side cost of an unstable prefix is not priced | throughput, memory waste |
| [24] | vLLM automatic prefix caching: hash KV blocks, share prefixes | States the stability requirement; does not measure what violating it costs an application | reuse rate |
| [25] | ChunkAttention: prefix-aware KV cache, two-phase partition | Kernel-level; assumes a served multi-tenant workload | throughput |
| [26] | Sparse prefix caching for hybrid/recurrent serving | Architecture-specific | throughput |
| [27] | Multi-segment attention for KV management | Serving-side; no application-level guidance | latency, memory |

## 1.4 Strand D — Positional effects in context

| Ref | Method proposed | Limitation w.r.t. this work | Metrics reported |
|---|---|---|---|
| [28] | **Lost in the Middle**: accuracy peaks when relevant content sits at the beginning or the end | Motivates reordering by relevance — which **destroys prefix reuse**, a conflict neither strand names | QA accuracy by position |
| [29] | Adaptive Focus Memory: per-message fidelity levels | Heuristic tiers; no measurement of the cost of reordering | token usage |
| [30] | Survey of multi-turn interaction capabilities | Survey; no composed measurement | mixed |
| [31] | Survey of memory mechanisms for LLMs | Survey; taxonomy rather than measurement | mixed |

## 1.5 Strand E — Routing and cascades

| Ref | Method proposed | Limitation w.r.t. this work | Metrics reported |
|---|---|---|---|
| [32] | FrugalGPT: prompt adaptation, approximation, LLM cascade | Up to 98% cost cut **against commercial APIs**; assumes a large capable tier exists | cost, accuracy |
| [33] | RouteLLM: routers trained on human preference data | Assumes the strong model is meaningfully better — untrue at 1.5B vs 3B on our set | cost, benchmark scores |
| [34] | Semantic router for vLLM | Serving-side routing; datacentre assumption | latency, cost |
| [35] | UCCI: calibrated uncertainty for cascade routing | Requires calibrated uncertainty; expensive on CPU | cost, accuracy |
| [36] | Uncertainty-based **on-device** LLM routing | Closest setting; routes between device and cloud, not within a local stack | accuracy, offload rate |
| [37] | Cluster, route, escalate: cost-aware serving | Cluster-level; no fidelity guarantee on the cheap path | cost, quality |

## 1.6 Strand F — Output length control

| Ref | Method proposed | Limitation w.r.t. this work | Metrics reported |
|---|---|---|---|
| [38] | Length-difference positional encoding with a countdown | **Requires fine-tuning** — out of reach on the target hardware | length adherence |
| [39] | BudgetThinker: budget-aware generation with control tokens | Requires training with control tokens | budget adherence, accuracy |
| [40] | Reasoning under strict output-length constraints | Characterises the cost of truncation; no per-class budgeting scheme | accuracy vs length |

## 1.7 Summary of the limitation column

Across all forty papers, five limitations recur and together define the space this project occupies.

| # | Recurring limitation | Papers |
|---|---|---|
| L1 | Evaluated **in isolation** against an uncompressed baseline; never composed with another technique | [1]–[11], [16]–[20], [38]–[40] |
| L2 | Assumes **GPU / datacentre serving**, where the cost structure is the opposite of a CPU laptop | [2], [3], [23]–[27], [32]–[35], [37] |
| L3 | Failure modes are **diagnosed but not prevented** at run time | [12], [13], [14], [21], [22] |
| L4 | Operating points (ratios, thresholds) **quoted as universal**, calibrated only on GPT-class models | [2], [4], [16], [17], [19] |
| L5 | Requires **training or fine-tuning**, or access to model internals | [5], [7], [8], [38], [39] |

## 1.8 The papers most relevant to this project

Four of the forty sit closest to what this work does, and each is the direct antecedent of one contribution.

**[28] Lost in the Middle (TACL 2024)** and **[24] vLLM automatic prefix caching** are the pair this project
measures *against each other*. The first says put relevant content at the prompt's edges. The second requires
the prompt head to be byte-stable across turns. **Doing the first destroys the second**, and neither
literature names the conflict. §7.2 prices it at ~80×.

**[16] GPTCache (NLP-OSS @ EMNLP 2023)** established the design every semantic cache now follows: embed,
compare, serve above a threshold. Our adversarial set shows the threshold is the wrong place to put the
safety (§7.3) — the same conclusion **[22] SAFE-CACHE (Scientific Reports, 2026)** reaches independently, but
reached here with four set comparisons costing microseconds rather than a cluster-centroid redesign.

**[12] An Empirical Study on Prompt Compression (ICLR 2025)** is the closest antecedent of the fidelity gate.
It names the two failure modes — altered-semantic and information-loss hallucination — and measures how often
they occur. It does not propose a run-time mechanism to *prevent* them. Module M8 is that mechanism: every
proposed edit is checked for lost numbers, entities, negations and operative modifiers before it is
committed.

<div class="pagebreak"></div>

# 2. Research Gaps Identified

Six gaps follow from the limitation column of §1, in the order the survey exposes them.

## 2.1 Gap 1: The Techniques Have Never Been Stacked

<div class="gapbox">
<div class="gt">Gap 1 &mdash; Isolated Modules, Unknown Interactions</div>
<div class="gb">
<p><span class="lbl">Current Problem:</span> Compression, caching, history trimming and routing each have
their own literature, their own benchmarks and their own tuned hyper-parameters. No published study runs all
of them in one live pipeline, so the field has no evidence about whether their savings compound.</p>
<p><span class="lbl">Research Gap:</span> Do the savings add up, or do the modules eat each other's lunch? A
compressor that rewrites a query changes what the cache sees. A trimmer that drops turns changes what the
compressor has left to remove. Nobody has measured the sign or size of these interactions.</p>
<p><span class="lbl">Our Solution:</span> A complete two-to-the-fourth factorial ablation over the four
headline modules, five seeds per cell, reported with confidence intervals and a two-way analysis of variance
on the interaction terms &mdash; the first stacked-interaction table for this combination of techniques.</p>
</div>
</div>

**Status: answered.** The additivity shortfall is **1.63 pp, 95% CI [−0.02, +3.23]**, and both material
interaction terms are negative (§7.1).

## 2.2 Gap 2: The Output Side of the Bill Is Ignored

<div class="gapbox">
<div class="gt">Gap 2 &mdash; Everyone Optimises the Prompt, Nobody Optimises the Answer</div>
<div class="gb">
<p><span class="lbl">Current Problem:</span> Prompt compression, caching and pruning all shrink the input.
That made sense for metered APIs, where the prompt is what is billed and generation runs on a datacentre GPU.
On a consumer CPU the economics invert: prefill is a parallel matrix operation, while decode is strictly
sequential, so a small model spends a large share of its wall-clock time producing tokens, not reading
them.</p>
<p><span class="lbl">Recent Evidence:</span> Industry measurements report that an explicit length hint can cut
output tokens by seventy-four to eighty-six percent on verbose models, but that applying the same hint
uniformly regresses quality on models whose answers are already terse. Output length therefore has to be set
adaptively, per query.</p>
<p><span class="lbl">Research Gap:</span> No study reports the split between prefill and decode cost for
CPU-hosted small models, and no token-optimisation stack contains an output-side module at all.</p>
<p><span class="lbl">Our Solution:</span> We instrument time-to-first-token and time-per-output-token
separately, and add an adaptive output budgeter with a streaming early-stop rule as a first-class,
independently ablated module.</p>
</div>
</div>

**Status: answered, and the expected direction was wrong.** Read from the server's own counters rather than
wall-clock TTFT, **prefill is 92–99% of the time**, not the minority share the framing above anticipated
(§7.2). The output budgeter remains the single largest token effect (+13.44 pp) because it is the only module
that touches the output side at all.

## 2.3 Gap 3: Compression and Caching Have Never Met

<div class="gapbox">
<div class="gt">Gap 3 &mdash; The Compression&ndash;Cache Interaction Is Unmeasured</div>
<div class="gb">
<p><span class="lbl">Current Problem:</span> Semantic caches are known to produce false hits: two queries that
are lexically close but semantically different get matched, and the user receives an answer to a question they
did not ask. 2026 security work shows this can even be induced deliberately.</p>
<p><span class="lbl">Recent Discovery:</span> Nobody has asked what compression does to that failure mode. The
argument runs both ways. Compression strips modifiers, and modifiers are often exactly what distinguishes two
near-identical questions, which would make false hits worse. But compression also normalises phrasing, so two
ways of asking the same thing collapse onto the same key, which would make true hits better.</p>
<p><span class="lbl">Research Gap:</span> Does prompt compression raise or lower the false-cache-hit rate, and
does the safe similarity threshold move?</p>
<p><span class="lbl">Our Solution:</span> A purpose-built adversarial subset of near-duplicate question pairs
that differ in exactly one operative token, swept across similarity thresholds with compression on and off,
reporting the threshold shift explicitly.</p>
</div>
</div>

**Status: answered, and the question dissolves.** No threshold separates the classes at all — the adversarial
negation pair scores *above* every genuine paraphrase — so the correct answer is not a threshold shift but a
verifier (§7.3).

## 2.4 Gap 4: Fewer Tokens Can Cost More Time

<div class="gapbox">
<div class="gt">Gap 4 &mdash; Optimisation That Defeats the Key-Value Cache</div>
<div class="gb">
<p><span class="lbl">Current Problem:</span> Local runtimes such as llama.cpp reuse the key-value cache for any
prompt prefix identical to the previous request, so a stable system prompt is processed once and then skipped.
This is the single largest latency win available on CPU, and it is entirely invisible to a token counter.</p>
<p><span class="lbl">Recent Discovery:</span> Every technique in the literature rewrites the prompt. A
compressor that alters the opening sentence, or a trimmer that drops an early turn, invalidates the prefix and
forces a full re-prefill. The system then sends fewer tokens and takes longer.</p>
<p><span class="lbl">Research Gap:</span> No published work measures token savings and prefix-cache survival
together. Token reduction is treated as a proxy for cost, and on local hardware that proxy can invert.</p>
<p><span class="lbl">Our Solution:</span> A prefix-stable prompt assembler that pins invariant content to the
front of the prompt and confines all rewriting to the suffix, plus a direct measurement of the crossover point
at which further compression starts costing time.</p>
</div>
</div>

**Status: answered.** Same content, 0.5% apart in token count, **~80× apart in cost**: 212 ms against 18,914
ms of steady-state prefill (§7.2).

## 2.5 Gap 5: Thresholds Tuned on Models That Do Not Break

<div class="gapbox">
<div class="gt">Gap 5 &mdash; Small Models Are More Brittle Than the Tuning Set</div>
<div class="gb">
<p><span class="lbl">Current Problem:</span> The compression ratios and cache thresholds quoted as safe across
the literature were all calibrated against GPT-3.5 and GPT-4 class models.</p>
<p><span class="lbl">Recent Discovery:</span> 2026 work on prompt sensitivity finds that small open models flip
a substantial fraction of their predictions across semantically equivalent paraphrases, while GPT-3.5 and
GPT-4 rarely change theirs. A compressed prompt is, from the model's point of view, an aggressive
paraphrase.</p>
<p><span class="lbl">Research Gap:</span> Do published safe settings transfer downward, or does the safe
compression ratio for a one-billion-parameter model sit far below the published figure?</p>
<p><span class="lbl">Our Solution:</span> Re-run the standard protocol on local models and quantisation levels,
and publish a calibration table giving the safe operating point per model rather than a single universal
number.</p>
</div>
</div>

**Status: answered.** The published 0.85–0.92 cache range is **unsafe at this scale** — it auto-accepts a
negation pair at cosine 0.924 and serves the opposite answer (§7.3). Reduction *ratios*, by contrast, do
transfer across vocabularies within 0.1 pp (§7.5).

## 2.6 Gap 6: Every Cache in the Literature Starts Empty

<div class="gapbox">
<div class="gt">Gap 6 &mdash; The Unused Signal in the User's Own Conversations</div>
<div class="gb">
<p><span class="lbl">Current Problem:</span> Every caching result is reported on a warm cache, after a workload
has already run. On day one the cache is empty, the hit rate is zero, and the user pays full price for exactly
the questions they have asked many times before in some other chat window.</p>
<p><span class="lbl">Recent Discovery:</span> MeanCache showed that roughly a third of real queries are highly
similar to a previous one, and that keeping the cache on the user's own device is both viable and more private
&mdash; but it still learns from scratch and relies on federation across many users to personalise.</p>
<p><span class="lbl">Research Gap:</span> A single user's own conversation history is a large, free, perfectly
on-distribution corpus that no optimisation system consumes. Nothing in the literature mines it to warm-start
a cache, to learn which phrases that user always writes and never needs, or to extract the standing context
they re-explain every session.</p>
<p><span class="lbl">Our Solution:</span> An offline policy learner that replays exported chat logs,
counterfactually re-scores each turn under candidate policies, and emits a pre-populated cache, a personal
redundancy lexicon, a persistent context digest and a library of recurring query templates &mdash; all
locally, with no data leaving the machine.</p>
</div>
</div>

**Status: answered, and made conditional.** Mined policy is worth **+0.00 pp at 0% traffic recurrence and
+17.83 pp at 57%** on held-out conversations (§7.5). "Self-improving" is a property of the traffic, not of the
module.

<div class="pagebreak"></div>

# 3. Research Questions and Problem Statement

## 3.1 Research questions

**RQ1.** When several token-reduction techniques are composed in one pipeline, do their individual savings
add — and if not, by how much do they fall short? *(Gap 1)*

**RQ2.** On CPU-only hardware, which half of a request dominates the cost, and what is one input token worth
in milliseconds? *(Gaps 2, 4)*

**RQ3.** Can a semantic cache be made safe against near-identical, opposite-meaning questions without a second
neural forward pass? *(Gaps 3, 4)*

**RQ4.** Does a calibration — thresholds, module settings, mined policy — transfer to a different
configuration or to unseen conversations without re-tuning? *(Gaps 5, 6)*

## 3.2 Problem statement

> Build a middleware layer that measurably reduces the token cost of interacting with a small language model
> on CPU-only consumer hardware, **without changing the meaning of any request**, and instrument it so that
> the contribution of every individual module, and every interaction between modules, is separately measurable
> and independently reproducible.

The emphasis is deliberate. A system that saves tokens by quietly damaging requests is not a contribution; the
safety property is the constraint under which every optimisation must operate.

## 3.3 Objectives

| # | Objective | Verified by |
|---|---|---|
| O1 | Compose eight techniques in one pipeline with every module independently switchable | 2⁴ factorial sweep, 17 cells |
| O2 | Establish the CPU prefill/decode cost structure and convert tokens to milliseconds | server-reported prefill/decode counters |
| O3 | Drive the false-answer rate of cache reuse to zero without a second model | 45 adversarial pairs + 45 controls |
| O4 | Show that no optimisation changes an answer that was previously correct | 40 gold items, paired per item |
| O5 | Keep total middleware overhead under 120 ms per request | ledger timings on every stage |
| O6 | Make every run reproducible from raw logs by one command | `reproduce.py`, ~100 s |

<div class="pagebreak"></div>

# 4. Contributions

**Contribution 1 — The additivity shortfall, measured.** A full 2⁴ factorial over compressor × cache × history
manager × output budgeter with bootstrap confidence intervals and partial η² effect sizes. Savings do not
compound: the shortfall is **1.63 pp, 95% CI [−0.02, +3.23]**. Further, its magnitude is a property of the
*configuration* — improving the encoder moved it from 2.53 pp to 1.63 pp, because a cache that hits more often
overlaps its neighbours less. *(Answers RQ1, closes Gap 1.)*

**Contribution 2 — The CPU cost structure, and the price of prompt order.** Prefill is **92–99%** of total
time, linear at **~8.5 ms per input token**. This converts every token result into wall clock and establishes
why input reduction is the thing worth doing. It also prices a conflict the literature does not name: a
volatile token at prompt position 0 costs **212 ms → 18,914 ms** in steady-state prefill for the same content.
*(Answers RQ2, closes Gaps 2 and 4.)*

**Contribution 3 — A verifier that makes reuse safe without a second model.** An adversarial set of 45 pairs
one operative token apart, plus 45 controls. The negation pair sits at cosine **0.924 — above every genuine
paraphrase in the set** — so no threshold separates them. Four set comparisons costing microseconds take the
false-answer rate from **26.7% to 0.0%**, including three checks no surveyed cache performs: operative
modifiers, morphological negation, and alphanumeric identifiers. *(Answers RQ3, closes Gaps 3 and 4.)*

**Contribution 4 — Calibration transfer, tested both ways.** Re-running the protocol against a second real
vocabulary shows **reduction ratios transfer** (within 0.1 pp, identical module ranking) while **the mechanisms
behind them do not** — one of our own two explanations for an effect proved specific to one tokenizer.
Separately, mined-policy transfer is shown to be a function of traffic repetition: **+0.00 pp at 0% recurrence,
+17.83 pp at 57%**, with zero fidelity violations at any level. *(Answers RQ4, closes Gaps 5 and 6.)*

<div class="pagebreak"></div>

# 5. Proposed Method

## 5.1 The governing design decision

Every module **proposes** a change; a single orchestrator **commits** it. A proposal is a description of an
edit, not the edit itself. Three properties follow that no individual module has to implement:

1. **Independent ablation.** Switching a module off genuinely removes its effect, so a factorial cell means
   what it claims.
2. **A single safety choke point.** Every edit passes one gate, so fidelity is not re-implemented seven times.
3. **Reverting is free.** A refused proposal is simply not assigned.

Stage order is **configuration, not code**, validated against a reads/writes dependency graph. Gap 3 is only
answerable because the cache lookup can be moved before or after the compressor without editing the pipeline.

## 5.2 The pipeline, tier by tier

Requests flow top to bottom. Each stage may short-circuit, propose an edit, or do nothing — and reports which.

| # | Tier | What it actually does | Input → Output | Why it exists |
|---|---|---|---|---|
| 1 | **M6a — Deterministic router** | Recognises arithmetic and date questions and computes the answer exactly, with an AST evaluator rather than `eval` | question → answer, or pass through | The cheapest request is the one never sent. 23 tokens → **0**. |
| 2 | **M2 — Semantic cache** | Embeds the question, retrieves near neighbours, applies a **three-zone policy**: accept, reject, or *verify*. Borderline matches are settled by four set comparisons over numbers, entities, negations and operative modifiers | question → stored answer, or pass through | Reuse without a second model call — and without serving the opposite answer. |
| 3 | **M3a — History selector** | Scores prior turns for relevance with MMR and keeps only those that fit the budget | N turns → M turns | In a long conversation most turns are irrelevant to the current question. |
| 4 | **M3b — History arranger** | Orders the kept turns | turns → ordered turns | Where a turn sits changes both attention and prefix reuse (§7.2). |
| 5 | **M1a — Tier 1, boilerplate** | Removes greetings, politeness and markdown, losslessly and sentence-aware | text → shorter text | Politeness carries no instruction and costs real tokens. |
| 6 | **M1b — Tier 2, redundancy** | Deletes a sentence that restates a fact already present, above a similarity threshold calibrated from data | text → shorter text | The same fact stated twice is paid for twice. |
| 7 | **M1c — Tier 3, word level** | Shortens long-winded phrasing, with a **negative-yield guard** that rejects any edit not reducing the token count | text → shorter text | Shortening text does not always shorten *tokens*; the guard measures rather than assumes. |
| 8 | **M4 — Prefix-stable assembler** | Assembles the final prompt with invariant content first and volatile content last | parts → prompt | Keeps the prompt head byte-stable so the model reuses its KV cache. Worth ~0 tokens and ~18 seconds. |
| 9 | **M5 — Output budgeter** | Classifies the question and sets `num_predict` accordingly — 48 for arithmetic, 640 for reasoning — plus a streaming early stop on trigram novelty | question → token budget | Small models ramble. Right-sizing beats truncating. |
| 10 | **M6b — Escalation router** | Scores complexity and decides whether a larger model is warranted | question → route | Calibrated to the observed distribution, and deliberately **off**: a 3B model scored item-for-item identically to 1.5B here. |
| — | **M8 — Fidelity gate** | **Runs on every proposal from every stage.** Compares invariants before and after; refuses any edit that would drop a number, entity, negation or operative modifier, or that would empty the payload | proposal → commit / refuse | The constraint under which all of the above operate. |
| — | **M7 — Policy learner** | Offline: mines recurring questions, standing facts and templates from past conversations into a `PolicyBundle` that warm-starts the cache | logs → bundle | Value is a property of the traffic (§7.5), not of the module. |

## 5.3 Experimental protocol

| Element | Setting |
|---|---|
| Design | 2⁴ full factorial over M1 × M2 × M3 × M5, 17 cells |
| Corpus | 151 conversations, 263 requests; 45 adversarial pairs + 45 controls; 40 gold items |
| Model | `qwen2.5:1.5b-instruct`, Q4_K_M quantisation, via Ollama, CPU-only, offline |
| Hardware | AMD Ryzen 7 5800HS, 16 GB RAM, no GPU used |
| Statistics | Bootstrap 95% confidence intervals, partial η² effect sizes, two-way ANOVA on interactions, exact McNemar for paired accuracy |
| Reproduction | `python reproduce.py --out figures` — every table regenerates in ~100 s |
| Verification | 719 automated tests, including architecture-layering and golden-output tests |

<div class="pagebreak"></div>

# 6. System Architecture

This section enumerates what the architecture actually contains: its layers, its components, the data that
flows between them, and the record it leaves behind.

## 6.1 The request path

<div style="text-align:center;margin:1.1em 0;">
<svg viewBox="0 0 620 552" width="100%" style="max-width:600px" xmlns="http://www.w3.org/2000/svg">
  <rect x="86" y="14" width="352" height="30" rx="4" fill="#eceff1" stroke="#37474f" stroke-width="1.1"/>
  <text x="262" y="33" font-size="10" font-weight="700" fill="#16181d" text-anchor="middle">USER QUESTION</text>
  <path d="M262 44 L262 56" stroke="#9aa0a6" stroke-width="1"/>
  <path d="M259 52 L262 56 L265 52" fill="#9aa0a6"/>
  <rect x="86" y="62" width="352" height="34" rx="4" fill="#ffffff" stroke="#1b5e20" stroke-width="1.1"/>
  <rect x="86" y="62" width="5" height="34" rx="2" fill="#1b5e20"/>
  <text x="100" y="76" font-size="9.5" font-weight="600" fill="#1b5e20">M6a</text>
  <text x="136" y="76" font-size="9.5" font-weight="600" fill="#16181d">Deterministic router</text>
  <text x="100" y="89" font-size="8.6" fill="#5a6069">Maths / dates answered exactly</text>
  <text x="430" y="83" font-size="8.4" fill="#5a6069" text-anchor="end">23 tokens &#8594; 0</text>
  <path d="M262 96 L262 104" stroke="#9aa0a6" stroke-width="1"/>
  <path d="M259 101 L262 105 L265 101" fill="#9aa0a6"/>
  <rect x="86" y="104" width="352" height="34" rx="4" fill="#ffffff" stroke="#1b5e20" stroke-width="1.1"/>
  <rect x="86" y="104" width="5" height="34" rx="2" fill="#1b5e20"/>
  <text x="100" y="118" font-size="9.5" font-weight="600" fill="#1b5e20">M2</text>
  <text x="136" y="118" font-size="9.5" font-weight="600" fill="#16181d">Semantic cache</text>
  <text x="100" y="131" font-size="8.6" fill="#5a6069">Reuse an earlier answer, verified</text>
  <text x="430" y="125" font-size="8.4" fill="#5a6069" text-anchor="end">prompt &#8594; stored answer</text>
  <path d="M262 138 L262 146" stroke="#9aa0a6" stroke-width="1"/>
  <path d="M259 143 L262 147 L265 143" fill="#9aa0a6"/>
  <rect x="86" y="146" width="352" height="34" rx="4" fill="#ffffff" stroke="#0d47a1" stroke-width="1.1"/>
  <rect x="86" y="146" width="5" height="34" rx="2" fill="#0d47a1"/>
  <text x="100" y="160" font-size="9.5" font-weight="600" fill="#0d47a1">M3a</text>
  <text x="136" y="160" font-size="9.5" font-weight="600" fill="#16181d">History selector</text>
  <text x="100" y="173" font-size="8.6" fill="#5a6069">Keep only relevant prior turns</text>
  <text x="430" y="167" font-size="8.4" fill="#5a6069" text-anchor="end">8 turns &#8594; 6</text>
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
  <text x="136" y="244" font-size="9.5" font-weight="600" fill="#16181d">Tier 1 &#8212; boilerplate</text>
  <text x="100" y="257" font-size="8.6" fill="#5a6069">Strip greetings and politeness</text>
  <text x="430" y="251" font-size="8.4" fill="#5a6069" text-anchor="end">28 &#8594; 17 tokens</text>
  <path d="M262 264 L262 272" stroke="#9aa0a6" stroke-width="1"/>
  <path d="M259 269 L262 273 L265 269" fill="#9aa0a6"/>
  <rect x="86" y="272" width="352" height="34" rx="4" fill="#ffffff" stroke="#4a148c" stroke-width="1.1"/>
  <rect x="86" y="272" width="5" height="34" rx="2" fill="#4a148c"/>
  <text x="100" y="286" font-size="9.5" font-weight="600" fill="#4a148c">M1b</text>
  <text x="136" y="286" font-size="9.5" font-weight="600" fill="#16181d">Tier 2 &#8212; redundancy</text>
  <text x="100" y="299" font-size="8.6" fill="#5a6069">Delete a repeated fact</text>
  <text x="430" y="293" font-size="8.4" fill="#5a6069" text-anchor="end">36 &#8594; 26 tokens</text>
  <path d="M262 306 L262 314" stroke="#9aa0a6" stroke-width="1"/>
  <path d="M259 311 L262 315 L265 311" fill="#9aa0a6"/>
  <rect x="86" y="314" width="352" height="34" rx="4" fill="#ffffff" stroke="#4a148c" stroke-width="1.1"/>
  <rect x="86" y="314" width="5" height="34" rx="2" fill="#4a148c"/>
  <text x="100" y="328" font-size="9.5" font-weight="600" fill="#4a148c">M1c</text>
  <text x="136" y="328" font-size="9.5" font-weight="600" fill="#16181d">Tier 3 &#8212; word level</text>
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
  <text x="430" y="419" font-size="8.4" fill="#5a6069" text-anchor="end">48 &#8230; 640 tokens</text>
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
  <text x="531" y="84" font-size="9.6" font-weight="700" fill="#7a5c00" text-anchor="middle">M8 &#183; FIDELITY GATE</text>
  <text x="531" y="102" font-size="8.3" fill="#5a4a00" text-anchor="middle">Checks EVERY proposal</text>
  <text x="531" y="116" font-size="8.3" fill="#5a4a00" text-anchor="middle">above before it is applied.</text>
  <text x="531" y="136" font-size="8.3" fill="#5a4a00" text-anchor="middle">Refuses any edit that</text>
  <text x="531" y="150" font-size="8.3" fill="#5a4a00" text-anchor="middle">would drop a number,</text>
  <text x="531" y="164" font-size="8.3" fill="#5a4a00" text-anchor="middle">name, negation or</text>
  <text x="531" y="178" font-size="8.3" fill="#5a4a00" text-anchor="middle">qualifier.</text>
  <text x="531" y="202" font-size="8.3" font-weight="600" fill="#7a5c00" text-anchor="middle">False-answer rate</text>
  <text x="531" y="216" font-size="8.3" font-weight="600" fill="#7a5c00" text-anchor="middle">26.7% &#8594; 0.0%</text>
  <path d="M446 152 L456 152" stroke="#b8860b" stroke-width="1.2"/>
  <path d="M450 148 L446 152 L450 156" fill="#b8860b"/>
  <text x="20" y="132" font-size="8.4" fill="#5a6069" transform="rotate(-90 20 132)" text-anchor="middle">MODULES PROPOSE &#183; ORCHESTRATOR COMMITS</text>
</svg>
<div class="figcap"><strong>Figure 1.</strong> The request pipeline. Each tier may answer outright, propose an
edit, or do nothing &mdash; and records which. Every proposal passes the fidelity gate before it is
committed.</div>
</div>

## 6.2 What is in the architecture

The system is built in six layers. A layer may depend only on layers below it, and this is enforced by an
automated test rather than by convention.

| Layer | Name | What it contains |
|---|---|---|
| **L0** | Contracts | Pure type and protocol definitions: `Request`, `Turn`, `Proposal` (`ContextPatch` / `ShortCircuit` / `NoOp`), `LedgerRow`, `TransformKind`. No logic, no imports from above. |
| **L1** | Infrastructure | Tokeniser (Qwen2.5 vocabulary, 151,665 entries), embedders (`content-v1` default, `hashing-v1` for comparison), text normalisation and Unicode sanitisation, providers (Ollama, Mock), generation memoisation. |
| **L2** | Modules | M1–M8. Each is a pure function of its inputs that returns a **proposal**; none mutates the request. |
| **L3** | Orchestrator | Applies stages in the configured order, submits every proposal to the gate, commits or refuses, and writes one ledger row per stage. |
| **L4** | Evaluation | Factorial sweep runner, quality measures, bootstrap and effect-size statistics, threshold calibration, cross-vocabulary generalisation, latency instrumentation, learning study, judge calibration. |
| **L5** | Surfaces | The command-line interface: `run`, `sweep`, `compare`, `tour`, `ask`, `judge`, `learning`, `latency`. |

## 6.3 The eight modules

<div class="archbox">
<p><strong>M1 &mdash; Compressor.</strong> Three tiers (boilerplate, redundancy, word level), sentence-aware,
with a negative-yield guard and windowed re-tokenisation so token counts are measured, never estimated.</p>
<p><strong>M2 &mdash; Two-tier cache.</strong> Exact-match tier plus a semantic tier with a three-zone policy
(accept / verify / reject) and a four-check invariant verifier.</p>
<p><strong>M3 &mdash; History manager.</strong> Four selection strategies and a separate arrangement stage, so
<em>what</em> is kept and <em>where</em> it is placed are independently ablatable.</p>
<p><strong>M4 &mdash; Prefix-stable assembler.</strong> Builds the prompt invariant-first, and carries a
token-level prefix-survival instrument that reports the reuse percentage directly.</p>
<p><strong>M5 &mdash; Output budgeter.</strong> Per-class token budgets plus a streaming early stop on trigram
novelty.</p>
<p><strong>M6 &mdash; Router.</strong> A deterministic tier that answers arithmetic and date questions using
zero model tokens, and a calibrated escalation tier.</p>
<p><strong>M7 &mdash; Policy learner.</strong> Offline counterfactual replay of past conversations into a
<code>PolicyBundle</code> that warm-starts the cache and the redundancy lexicon.</p>
<p><strong>M8 &mdash; Fidelity gate.</strong> Always on, scoped by <code>TransformKind</code>, checking numbers,
entities, negations, operative modifiers and payload emptiness on every proposal.</p>
</div>

## 6.4 Cross-cutting components

| Component | Role |
|---|---|
| **Ledger (v1)** | One row per stage per request — module, outcome, tokens before/after, duration, rationale, gate event, provider content digest. Dual sinks (JSONL + in-memory). **A stage that did nothing still writes a row**; an invisible stage is an unauditable one. |
| **Fidelity gate** | The single safety choke point. Runs on every proposal from every stage, before commitment. |
| **Provider layer** | `OllamaProvider` (stdlib `urllib`, no SDK) and `MockProvider`. A run that asks for the real model and cannot reach it **refuses** rather than silently falling back. |
| **Generation memoisation** | Bit-exact at temperature 0, so repeated sweep cells do not re-pay for identical generations. |
| **Stage-order validator** | Checks the configured order against a reads/writes dependency graph before the pipeline runs. |
| **Corpus** | 151 conversations, 45 adversarial pairs, 45 controls, 40 gold items, CC BY 4.0. |
| **Test suite** | 719 tests, including `test_architecture.py`, which fails the build if a layer imports upward. |

## 6.5 Technology stack

| Concern | Choice | Why |
|---|---|---|
| Language | Python 3.11 | Available, typed, no build step |
| Model runtime | Ollama, `qwen2.5:1.5b-instruct` (Q4_K_M) | Free, offline, CPU-only, openly licensed |
| Tokenisation | `tokenizers` (Qwen2.5 vocabulary) | The counts must come from the model's own vocabulary |
| Numerics | `numpy` | Embeddings and statistics |
| CLI | `typer` + `rich` | The demo surface |
| Testing | `pytest` | 719 tests |
| Deliberately absent | PyTorch, any GPU dependency, any paid API | The target machine is a CPU-only laptop |

<div class="pagebreak"></div>

# 7. Results and Discussion

All figures regenerate from raw logs with a single command in about 100 seconds. **719 automated tests pass.**

## 7.1 The headline: savings do not compound

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

| Encoder | M2 effect | M3 × M5 | Additivity shortfall |
|---|---|---|---|
| `hashing-v1` | +1.61 pp | −1.14 | 2.53 pp, [+0.93, +3.99] |
| `content-v1` (default) | +1.94 pp | −0.70 | 1.63 pp, [−0.02, +3.23] |

Improving the encoder made the cache hit more often, so it overlapped its neighbours less. The weaker encoder
was **not** reinstated to protect the interval: choosing a component known to be worse because it yields a more
publishable number is the failure mode this project is written against.

## 7.2 Prefill dominates, and prompt order has a price

Measured on `qwen2.5:1.5b-instruct` (Q4_K_M), Ryzen 7 5800HS, reading the server's own prefill/decode split
rather than wall-clock time-to-first-token.

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

Same content, 0.5% apart in token count, **~80× apart in cost**. Every metric in the compression literature
scores those two configurations identically.

## 7.3 The verifier, not the threshold

45 adversarial pairs plus 45 controls.

| Cosine | Pair |
|---|---|
| **0.924** | "Is it safe to mix bleach and vinegar?" / "Is it **not** safe…" |
| 0.869 | "capital of **Australia**" / "capital of **Austria**" |
| 0.653 | "What causes rain?" / "What causes rainfall to occur?" |

The adversarial pair scores **above every genuine paraphrase**, so the published 0.85–0.92 range would
auto-accept it and return the opposite answer.

| Operative difference | Initial false-hit rate | After the verifier |
|---|---|---|
| modifier | 78% | **0%** |
| negation | 31% | **0%** |
| entity | 11% | **0%** |
| **overall** | **26.7%** | **0.0%** |

## 7.4 Quality is not the price

40 gold items, real model.

| Configuration | Gold accuracy |
|---|---|
| Baseline | 37/40 — 92.5% |
| Full stack | **39/40 — 97.5%** |

Paired per item: **zero regressions.** Not one answer the baseline got right was lost to compression. The two
gains are arithmetic routed to M6's deterministic tier; two discordant pairs is not significant (exact McNemar
p = 0.50), so the claim is *no measurable degradation* rather than improvement.

## 7.5 Transfer

**Across vocabularies.** Re-running the entire sweep against GPT-2's 50,257-entry vocabulary with thresholds
carried over unchanged: reduction ratios land within 0.1 pp and the module ranking is identical. The
*mechanisms* fare worse — one of our two stated explanations for a tokenizer effect proved specific to Qwen.

**Across conversations.** M7's value, mined from one half of the conversations and tested on a disjoint half:

| Traffic recurrence | Transfer | Extra cache hits | Extra gate fires |
|---|---|---|---|
| 0.0% | **+0.00 pp** | 0 | 0 |
| 22.5% | +5.43 pp | +4 | 0 |
| 57.5% | **+17.83 pp** | +12 | 0 |

The exact zero at 0% recurrence is what makes the rest credible.

## 7.6 Summary of metrics

| Metric | Baseline | With Parsimony |
|---|---|---|
| Total token reduction | — | **33.9%** |
| Gold accuracy (40 items) | 92.5% | **97.5%**, zero regressions |
| False-answer rate on adversarial pairs | 26.7% | **0.0%** |
| Prefill share of wall clock | 91.7 – 98.8% | unchanged (measured, not optimised away) |
| Prefix-cache reuse | 0.5% (volatile head) | **98.5%** |
| Prefill saved across the corpus | — | **100.6 s** (383 ms per request) |
| Middleware overhead per request | — | under 120 ms |
| Mined-policy transfer at 57% recurrence | — | **+17.83 pp** |

## 7.7 Threats to validity

1. **Single hardware configuration.** The 8.5 ms/token rate is not claimed to generalise.
2. **Small gold set.** 40 items; after a grading fix only two defeat both models, and both are arithmetic
   already handled without a model — so §7.4 measures the gold set as much as the pipeline.
3. **Lexical encoder.** Operating points shift with the encoder; rankings do not.
4. **Corpus recurrence is 1.9%**, which is why §7.5 reports a curve over synthetic traffic rather than a single
   number. The repetition structure is synthetic; every question and answer in it is real.

<div class="pagebreak"></div>

# 8. Conclusion

The efficiency literature has produced a rich toolbox and a thin account of what happens when the tools are
used together. This project's contribution is not a new compression algorithm but a **measurement
instrument**: eight techniques in one harness, each independently switchable, every decision written to an
auditable ledger.

What that instrument found, repeatedly, is that **published operating points are configuration-specific**. A
threshold quoted as safe is unsafe here. A placement strategy justified by a well-cited attention result is 80×
more expensive here. A calibration transfers as a ratio but not as a mechanism. An escalation premise that
holds at 70B does not hold at 3B.

For this class of system, the deliverable is a **calibration procedure, not a number**.

# 9. References

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
