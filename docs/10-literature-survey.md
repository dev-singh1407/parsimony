# Literature Survey and Positioning

**Parsimony — Token-Efficient LLM Interaction on CPU-Only Hardware**
VIT University · B.Tech BCSE497J Project I · Guide: Dr Sathya K
Arrsh Tripathi (23BCI0191) · Alok Singh (23BCI0158) · Dev Singh (23BCE0794)

---

## Abstract

Running a language model on a CPU-only consumer laptop makes every token expensive. A literature has grown
around reducing that cost — prompt compression, semantic caching, KV-cache reuse, model routing, output
budgeting — but it has grown in **isolated strands**. Each technique is proposed, measured against an
uncompressed baseline, and published alone. Almost nothing measures what happens when they are composed.

This survey covers **48 works** across eight strands. It identifies six gaps, of which the central one is
compositional: the field reports per-technique savings that are implicitly assumed to add, and we find they
do not. We then state precisely what this project does differently, and give the measurements that support
each claim.

The recurring theme of our findings is that **published operating points do not survive contact with a
different configuration.** A cache similarity threshold quoted as safe in the literature serves the opposite
answer on our adversarial set. A prompt-ordering strategy motivated by a well-cited attention result costs
80× in wall-clock time. A compression tier that appears to do nothing turns out to be limited by its encoder
rather than its technique. In every case the discrepancy was found by measuring, not by reasoning.

---

## 1. Motivation and Scope

The project targets a specific and under-served deployment: **a small instruction-tuned model (1–3 B
parameters, 4-bit quantised) running on an 8 GB consumer laptop with no GPU.** This matters because the
efficiency literature is overwhelmingly written for datacentre serving, where the scarce resource is GPU
memory bandwidth and the unit of optimisation is throughput across concurrent requests. On a single CPU with
one user, the cost structure is different — and, as §6 shows, differently *shaped* than the literature's
assumptions imply.

Scope of the survey: works are included if they bear on the cost of an individual request to a language
model, or on the validity of measuring that cost. Training-time efficiency (distillation, pruning,
architecture search) is out of scope except where it produces the small models we deploy.

---

## 2. Strand 1 — Prompt Compression

The largest and most active strand. The premise is that prompts contain redundancy that can be removed
before the model sees them.

**Foundational work.** Li et al. [1] (EMNLP 2023) introduced *Selective Context*, using a small language
model to compute the self-information of each lexical unit and dropping the least informative, reporting
reduced memory cost and latency at comparable task performance. Jiang et al. [2] (EMNLP 2023) introduced
*LLMLingua*, a coarse-to-fine method combining a budget controller with token-level iterative compression,
reporting up to 20× compression with little performance loss on GSM8K, BBH, ShareGPT and Arxiv-March23.
*LongLLMLingua* [3] extended this to long-context scenarios using conditional perplexity, reporting up to
2.6× end-to-end speedup. *LLMLingua-2* [4] reframed the problem as data distillation for task-agnostic
compression, explicitly targeting faithfulness.

**Subsequent development.** The strand has broadened considerably: evaluator-head methods for long-context
inference [5], generative rather than extractive compression [6], gisting and context distillation [7],
learned context summarisation [8], adaptive task-aware compression [9], and selective retrieval-augmented
compression [10]. A unified toolkit [11] now exists for comparing methods.

**Critical evaluation of the strand.** More recent work has begun to test the strand's own claims, and this
is where it becomes directly relevant to us. An ICLR 2025 empirical study [12] evaluated compression across
GPT-3.5-turbo, GPT-4o-mini and Claude-3-Haiku and characterised two failure modes: *Altered Semantic
Hallucination* and *Information Loss Hallucination*. Chirkova et al. [13] measured information preservation
directly and report substantial groundedness drops — approximately 30 points on HotpotQA and around 50 on
QuAC and arXiv-summarisation. A financial-analysis study [14] found compressed contexts that remain fluent
and factually plausible while changing downstream decisions. Most strikingly for our purposes, [15] reports
a *compression paradox*: prompt compression's energy effects are provider-dependent, and compression does
not uniformly reduce energy.

> **What this strand assumes and does not check.** Compression is measured against an uncompressed baseline,
> in isolation. We found no work in this strand that measures compression **in the presence of a semantic
> cache**, which is our Gap 3 — and the interaction turns out to be real and negative.

---

## 3. Strand 2 — Semantic Caching

**Foundational work.** Bang [16] (NLP-OSS 2023) introduced *GPTCache*, the reference open-source semantic
cache: queries are embedded, compared against stored entries by vector similarity, and served from cache
above a threshold. *GPT Semantic Cache* [17] applied the same idea over Redis. *MeanCache* [18] introduced a
user-centric federated variant. *ContextCache* [19] addressed the multi-turn case, where a query's meaning
depends on conversation history. A generative caching system [20] extended the idea beyond exact response
reuse.

**The threshold problem.** Every system in this strand depends on a similarity threshold, and the literature
converges on values around 0.8–0.92. Reported guidance holds that thresholds below 0.8 raise hit rates while
introducing irrelevant matches, and higher thresholds cause false misses.

**Security of the strand.** Two very recent works establish that this threshold is attackable. [21]
demonstrates *key collision attacks* on LLM semantic caching — adversaries craft embedding-collision queries
that satisfy the similarity threshold while being semantically different. [22] (Scientific Reports, 2026)
proposes SAFE-CACHE, a cluster-centroid strategy, on the explicit premise that **vector similarity alone is
insufficient for semantic verification.**

> **Where we depart.** The attack literature searches for adversarial *suffixes* — crafted inputs. We find a
> collision class that needs no search at all: six degenerate inputs (`""`, `"   "`, `"?"`, `"?!..."`,
> `"!!!"`, `"."`) canonicalise to one cache key, reachable by typing a question mark. It sits in the
> *exact-hash* tier, which the collision threat model does not examine because hash equality looks
> unambiguously safe.

---

## 4. Strand 3 — KV Cache and Prefix Reuse

**Foundational work.** Kwon et al. [23] (SOSP 2023) introduced *PagedAttention* and vLLM, applying OS
virtual-memory concepts to KV cache management and reducing fragmentation by an order of magnitude.
Critically for this project, vLLM's *automatic prefix caching* [24] hashes KV blocks so that requests sharing
a prefix reuse the same physical blocks without recomputation. *ChunkAttention* [25] introduced a
prefix-aware KV cache with two-phase partitioning. Recent work extends prefix caching to hybrid and
recurrent architectures [26] and to multi-segment management [27].

**The consequence the literature states but does not price.** Prefix reuse only helps if the prefix is
**byte-stable across turns**. This is stated as a design constraint in serving systems; we could find no work
that measures what a *volatile* prefix costs an application in wall-clock terms.

---

## 5. Strand 4 — Positional Effects in Context

Liu et al. [28] (TACL 2024) established *Lost in the Middle*: model performance is highest when relevant
information sits at the beginning or end of the input and degrades markedly in the middle. This is among the
most-cited results in the strand and directly motivates **position-aware placement** — putting the most
relevant retrieved content at the extremities of the prompt.

> **The contradiction we measure.** Position-aware placement is a per-request quality optimisation. Prefix
> caching is a cross-request cost optimisation. **They are in direct conflict**, because reordering by
> relevance changes the prompt head on every turn and destroys the reusable prefix. We could not find this
> tension named in either strand, and §7 gives its price.

---

## 6. Strand 5 — Model Routing and Cascades

**Foundational work.** Chen et al. [29] introduced *FrugalGPT*, combining prompt adaptation, LLM
approximation and cascades, reporting up to 98% cost reduction against commercial APIs. Ong et al. [30]
introduced *RouteLLM*, training routers on human preference data to choose between a stronger and weaker
model, reporting over 2× cost reduction without quality loss on MT-Bench, MMLU and GSM8K.

**Subsequent development.** The strand now includes semantic routing for vLLM [31], calibrated-uncertainty
cascade routing [32], uncertainty-based **on-device** routing [33], and clustered cascade frameworks [34].

> **The premise we test and fail to confirm.** Every work in this strand assumes escalation to a larger model
> improves answers. At our scale it does not: llama3.2:3b and qwen2.5:1.5b score **item-for-item identically**
> on our gold set. The strand's premise is a datacentre premise, where "larger" means 70B rather than 3B.

---

## 7. Strand 6 — Output Length Control

Over-generation is a recognised waste channel: RLHF-trained reward models exhibit systematic length bias,
treating verbosity as quality. Methods include length-difference positional encoding with a countdown
mechanism for precise control [35], budget-aware generation with control tokens [36], prompt-based
one-shot exact length control [37], and studies of reasoning ability under strict length constraints [38].

> **Our position.** We use the cheapest possible mechanism — a per-class `num_predict` budget plus a
> streaming trigram-novelty early stop — precisely because the alternatives require fine-tuning, which is out
> of reach on the target hardware. Our contribution here is not the mechanism but the measurement: M5 is the
> **largest single contributor** in our ablation (+13.44 pp), which is not where the compression literature's
> emphasis would predict.

---

## 8. Strand 7 — Evaluation Validity

This strand is methodological, and it is the one that most changed our practice.

**LLM-as-judge and its biases.** Shi et al. [39] systematically studied *position bias* in LLM-as-a-judge,
introducing repetition stability, position consistency and preference fairness as metrics. [40] quantified
*self-preference bias* — LLMs systematically favouring their own outputs — demonstrating it significantly in
GPT-4. Follow-up work quantifies and mitigates it [41] and evaluates scoring bias more broadly [42].

> **What we add.** The strand documents these biases; it less often prescribes a **pre-registration check**.
> We calibrate the judge against itself before reporting anything it says (§10, Contribution 8), and our
> judge failed that check — which is why we do not report its verdict as a quality result.

---

## 9. Strand 8 — Small Models, Quantisation and Energy

**Small and on-device models.** Comprehensive surveys cover small language models [43, 44] and on-device
deployment [45], with a broader survey of efficient inference serving [46]. `llama.cpp` and the GGUF format
underpin most CPU deployment; a systematic evaluation of its quantisation levels on Llama-3.1-8B-Instruct
[47] is recent. AWQ [48] and related post-training quantisation work [49] establish the 4-bit regime our
target model uses (Q4_K_M).

**Energy.** Measurement frameworks now exist at scale: [50] reports over 32,500 measurements across 21 GPU
configurations and 155 architectures; [51] finds inference now accounts for more than half of LLM lifecycle
carbon emissions; [52] benchmarks energy, water and carbon across 30 models.

> **The gap for our setting.** All three energy works measure **GPU datacentre** inference. None measures a
> CPU-only single-user laptop, where the cost structure is different — and where, as we show, prefill
> dominates to a degree that changes which optimisation is worth doing.

---

## 10. Research Gaps Identified

Synthesising the eight strands, six gaps are addressable within this project's constraints.

| # | Gap | Why the literature leaves it open |
|---|---|---|
| **1** | **Compositional effects are unmeasured.** Each technique is evaluated alone against an uncompressed baseline. Whether their savings *add* is unknown. | Publishing incentives favour isolated contributions; a composition study requires a factorial design and a shared harness, which no single technique paper needs. |
| **2** | **The prefill/decode split is not characterised for CPU-only single-user inference.** | The energy and serving literature [46, 50–52] measures GPU datacentre throughput. |
| **3** | **Compression × cache interaction.** Does normalising a query before the cache sees it raise hit rate, or raise *false* hit rate? | Strand 1 and Strand 2 do not cite each other's failure modes. |
| **4** | **Safety of published cache thresholds at small scale.** The 0.8–0.92 range is quoted without adversarial validation on near-identical, opposite-meaning pairs. | [21, 22] establish attackability but as a *search* problem; ordinary confusable pairs are not the threat model. |
| **5** | **Whether a calibration transfers.** Thresholds are reported as if universal. | Requires re-running an identical protocol on a second configuration, which is rarely done. |
| **6** | **Whether mined policy transfers to unseen conversations.** | Self-improving systems are typically evaluated on the data they were mined from. |

---

## 11. What We Do Differently

Nine points of departure, each tied to a measurement in this repository. Every figure below regenerates from
raw logs with `python reproduce.py`.

### 11.1 We measure composition, not techniques (Gap 1)

A full 2⁴ factorial over compressor × cache × history manager × output budgeter, 151 conversations, 263
requests, 17 cells, with bootstrap confidence intervals and partial η² effect sizes.

**Result:** savings do **not** compound. The additivity shortfall is **1.63 pp, 95% CI [−0.02, +3.23]** under
our default configuration.

**And the sharper finding:** the shortfall is **not a constant of the technique stack — it is a property of
the configuration.** Improving our encoder made the cache hit more often, which made it overlap its
neighbours *less*, moving the shortfall from 2.53 pp [+0.93, +3.99] to 1.63 pp [−0.02, +3.23]. We kept the
better encoder and reported both. No work in Strand 1 or 2 is positioned to observe this, because observing
it requires holding four techniques in one harness while varying a fifth component.

### 11.2 We price the conflict between two literatures (Gaps 2, 4)

*Lost in the Middle* [28] motivates position-aware placement. Prefix caching [23, 24] requires a stable
prompt head. We measure the collision:

| arrangement | tokens | steady-state prefill | prefix reuse |
|---|---|---|---|
| stable prefix | 1,490 | **212 ms** | **98.5%** |
| volatile head | 1,497 | **18,914 ms** | **0.5%** |

**Same content, 0.5% apart in token count, ~80× apart in cost.** Every metric in the compression literature
scores these two configurations identically, because token count cannot see the difference.

### 11.3 We show the verifier, not the threshold, is what makes a cache safe (Gap 4)

On 45 adversarial pairs (one operative token apart, opposite answers) plus 45 controls:

- The adversarial negation pair sits at cosine **0.924** — **higher than every genuine paraphrase in our
  set.** The literature's "safe" 0.85–0.92 would auto-accept it and serve the opposite answer.
- Similarity alone does not reach a <2% false-hit target at **any** threshold in the sweep.
- Adding four cheap set-comparison checks took the false-hit rate from **26.7% → 0.0%**.

Three of those checks are, as far as this survey found, not performed by any system in Strand 2: **operative
modifiers** (min/max, which change no number, entity or negation particle), **morphological and lexical
negation** ("possible"/"impossible", "fails to"), and **alphanumeric identifiers** ("pandas" vs "Panda3D",
where the proper-noun pattern cannot match across a digit). This aligns with [22]'s conclusion that vector
similarity is insufficient — but reaches it with four set comparisons costing microseconds rather than a
cluster-centroid redesign.

### 11.4 We establish the cost structure that justifies the whole enterprise (Gap 2)

Measured on `qwen2.5:1.5b-instruct` (Q4_K_M), Ryzen 7 5800HS, reading the server's own prefill/decode split
rather than wall-clock TTFT:

| input tokens | prefill | decode | prefill share |
|---|---|---|---|
| 146 | 1,360 ms | 122 ms | 91.7% |
| 474 | 3,902 ms | 165 ms | 95.9% |
| 1,338 | 11,609 ms | 136 ms | 98.8% |

**Prefill is linear at ~8.5 ms per input token and is 92–99% of total time.** On CPU, input tokens *are* the
cost. This converts our token results into wall clock: the full stack's 11,836 saved input tokens are
**100.6 s of prefill across the corpus, 383 ms per request.**

### 11.5 We test whether our own calibration transfers (Gap 5)

Re-ran the entire sweep against a second real vocabulary (GPT-2's 50,257 against Qwen2.5's 151,665),
thresholds carried over unchanged.

**Reduction ratios transfer** — within 0.1 pp, module ranking identical, because a ratio cancels a roughly
constant vocabulary factor. **Mechanisms do not.** We had attributed a compression artefact to two BPE
position-0 effects; only one survives. `"explain"` costs 1 token against `"Explain"`'s 2 under Qwen, but
GPT-2 charges 2 for both. Half of our own explanation was a Qwen-specific fact wearing a general claim's
clothes.

### 11.6 We make "self-improving" a measured claim, not a title (Gap 6)

Mining from one half of the conversations and measuring on a **disjoint** half:

| traffic recurrence | transfer | extra cache hits | extra gate fires |
|---|---|---|---|
| 0.0% | **+0.00 pp** | 0 | 0 |
| 22.5% | +5.43 pp | +4 | 0 |
| 57.5% | **+17.83 pp** | +12 | 0 |

The value of a learned policy is **a property of the traffic, not of the module**. Our own ablation corpus
sits at 1.9% recurrence — authored for ablation diversity, which is the right shape for measuring the other
modules and the wrong shape for this one. The exact zero at 0% recurrence is what makes the rest credible.

### 11.7 We report a negative result on escalation (Strand 5)

| model | size | gold | wall clock |
|---|---|---|---|
| qwen2.5:1.5b-instruct | 0.92 GB | 36/40 | 139 s |
| llama3.2:3b | 2.0 GB | 36/40 | 161 s |

Item for item identical — zero questions where one succeeded and the other failed — for 16% more time and
twice the memory. We also found our escalation threshold (0.75) was **unreachable**: the observed maximum
complexity was 0.406, so the tier could never fire. Both are reported; the tier is calibrated and left off.

### 11.8 We calibrate the judge before believing it (Strand 7)

Following [39–42], we use an independent judge (llama3.2:3b judging qwen2.5:1.5b) with position swapping.
Then, before reporting anything, we show it **two identical answers** and ask it to choose:

- position bias: **50.0 pp** — it names the same slot every time
- swap disagreement on the real comparison: **91.7%**
- **verdict: not usable**

Its 45.8% win rate for the full stack is therefore *not reported as a quality result*. That number is
publishable-looking — near parity, with a ready story attached — and nothing in it reveals the instrument was
broken. The calibration costs one call per item and needs no ground truth.

### 11.9 We attack our own safety component

Beyond the corpus, we fuzzed the pipeline. Two findings:

- **The verifier was bypassable with invisible characters.** A zero-width space inside "not" splits it into
  fragments matching no negation lexicon entry, so `verify_match` passed a question against its own opposite.
  Seven variants worked (ZWSP, ZWJ, ZWNJ, word joiner, soft hyphen, Cyrillic homoglyph, full-width). This is
  a *different* attack class from [21]'s embedding collisions: it defeats the **verifier**, not the
  similarity metric. Fixed by Unicode sanitisation before analysis.
- **Every non-Latin query was being deleted entirely.** A Hindi, Tamil, Russian or Chinese question matched
  no `[A-Za-z0-9]`, was treated as contentless, and the model received an empty prompt. The fidelity gate did
  not catch it, because every check it makes asks "was a value I could *extract* lost?" — which silently
  makes its guarantee conditional on the extractor's language coverage.

---

## 12. Threats to Validity

Stated plainly, because several of our findings are about other people's unstated assumptions.

1. **Single hardware configuration.** All latency results are from one Ryzen 7 5800HS. The ~8.5 ms/token
   prefill rate is not claimed to generalise.
2. **Small gold set.** 40 items. After our grading fix, only two defeat both models, and both are arithmetic
   already handled by the deterministic tier — so the gold set contains no item in the difficulty band where
   1.5B fails and 3B succeeds. §11.7 therefore measures the **gold set** as much as the models.
3. **Corpus scale.** 151 conversations, 263 requests. Adequate for a factorial with effect sizes; not
   adequate for the p-values we deliberately do not report.
4. **Lexical encoder.** Our embedder is lexical, not neural. §11.5 shows this changes operating points but
   not rankings.
5. **Corpus recurrence.** At 1.9%, our corpus is the wrong shape for M7, which is why §11.6 reports a curve
   over synthetic traffic rather than a single number. The repetition structure of those traces is synthetic;
   every question and answer in them is real.

---

## 13. Conclusion

The efficiency literature has produced a rich toolbox and a thin account of what happens when the tools are
used together. This project's contribution is not a new compression algorithm — it is a **measurement
instrument** that holds eight techniques in one harness, switches each independently, and records every
decision to an auditable ledger.

What that instrument found, repeatedly, is that **published operating points are configuration-specific**. A
threshold quoted as safe is unsafe here. A placement strategy justified by a well-cited attention result is
80× more expensive here. A calibration transfers as a ratio but not as a mechanism. An escalation premise
that holds at 70B does not hold at 3B. Each of these is a small result; together they support one claim: for
this class of system, **the deliverable is a calibration procedure, not a number.**

---

## References

All URLs verified September 2026.

1. Li, Y., Dong, B., Lin, C., Guerin, F. *Compressing Context to Enhance Inference Efficiency of Large Language Models.* EMNLP 2023. https://aclanthology.org/2023.emnlp-main.391/ · arXiv:2310.06201
2. Jiang, H., Wu, Q., Lin, C.-Y., Yang, Y., Qiu, L. *LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models.* EMNLP 2023. https://aclanthology.org/2023.emnlp-main.825/ · arXiv:2310.05736
3. Jiang, H. et al. *LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression.* arXiv:2310.06839
4. *LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression.* arXiv:2403.12968
5. *Efficient Prompt Compression with Evaluator Heads for Long-Context Transformer Inference.* arXiv:2501.12959
6. *SCOPE: A Generative Approach for LLM Prompt Compression.* arXiv:2508.15813
7. *Long Context In-Context Compression by Getting to the Gist of Gisting.* arXiv:2504.08934
8. *Compressing Lengthy Context With UltraGist.* arXiv:2405.16635
9. *ATACompressor: Adaptive Task-Aware Compression for Efficient Long-Context Processing in LLMs.* arXiv:2602.03226
10. *SARA: Selective and Adaptive Retrieval-augmented Generation with Context Compression.* arXiv:2507.05633
11. *PCToolkit: A Unified Plug-and-Play Prompt Compression Toolkit of Large Language Models.* arXiv:2403.17411
12. *An Empirical Study on Prompt Compression for Large Language Models.* ICLR 2025. https://openreview.net/pdf?id=lbFVTPv4s6
13. *Understanding and Improving Information Preservation in Prompt Compression for LLMs.* arXiv:2503.19114
14. *When Summaries Distort Decisions: Information Fidelity in LLM-Compressed Financial Analysis.* arXiv:2606.29251
15. *The Compression Paradox in LLM Inference: Provider-Dependent Energy Effects of Prompt Compression.* arXiv:2603.23528
16. Bang, F. *GPTCache: An Open-Source Semantic Cache for LLM Applications Enabling Faster Answers and Cost Savings.* NLP-OSS @ EMNLP 2023. https://aclanthology.org/2023.nlposs-1.24/
17. *GPT Semantic Cache: Reducing LLM Costs and Latency via Semantic Embedding Caching.* arXiv:2411.05276
18. *MeanCache: User-Centric Semantic Caching for LLM Web Services.* arXiv:2403.02694
19. *ContextCache: Context-Aware Semantic Cache for Multi-Turn Queries in Large Language Models.* arXiv:2506.22791
20. *A Generative Caching System for Large Language Models.* arXiv:2503.17603
21. *From Similarity to Vulnerability: Key Collision Attack on LLM Semantic Caching.* arXiv:2601.23088
22. *Enhancing adversarial resilience in semantic caching for secure retrieval augmented generation systems.* Scientific Reports, 2026. https://www.nature.com/articles/s41598-026-36721-w
23. Kwon, W. et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP 2023.
24. vLLM. *Automatic Prefix Caching (design documentation).* https://docs.vllm.ai/en/v0.7.0/design/automatic_prefix_caching.html
25. *ChunkAttention: Efficient Self-Attention with Prefix-Aware KV Cache and Two-Phase Partition.* arXiv:2402.15220
26. *Sparse Prefix Caching for Hybrid and Recurrent LLM Serving.* arXiv:2605.05219
27. *Multi-Segment Attention: Enabling Efficient KV-Cache Management for Faster Large Language Model Serving.* arXiv:2606.02964
28. Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., Liang, P. *Lost in the Middle: How Language Models Use Long Contexts.* TACL 12:157–173, 2024. https://aclanthology.org/2024.tacl-1.9/ · arXiv:2307.03172
29. Chen, L., Zaharia, M., Zou, J. *FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance.* arXiv:2305.05176
30. Ong, I. et al. *RouteLLM: Learning to Route LLMs with Preference Data.* arXiv:2406.18665
31. *When to Reason: Semantic Router for vLLM.* arXiv:2510.08731
32. *UCCI: Calibrated Uncertainty for Cost-Optimal LLM Cascade Routing.* arXiv:2605.18796
33. *Confident or Seek Stronger: Exploring Uncertainty-Based On-device LLM Routing.* arXiv:2502.04428
34. *Cluster, Route, Escalate: Cascaded Framework for Cost-Aware LLM Serving.* arXiv:2606.27457
35. *Precise length control for large language models.* Natural Language Processing Journal, Elsevier. https://www.sciencedirect.com/science/article/pii/S2949719125000196
36. *BudgetThinker: Empowering Budget-aware LLM Reasoning with Control Tokens.* arXiv:2508.17196
37. *Prompt-Based One-Shot Exact Length-Controlled Generation with LLMs.* arXiv:2508.13805
38. *An Empirical Study of LLM Reasoning Ability Under Strict Output Length Constraint.* arXiv:2504.14350
39. *Judging the Judges: A Systematic Study of Position Bias in LLM-as-a-Judge.* arXiv:2406.07791
40. *Self-Preference Bias in LLM-as-a-Judge.* arXiv:2410.21819
41. *Quantifying and Mitigating Self-Preference Bias of LLM Judges.* arXiv:2604.22891
42. *Evaluating Scoring Bias in LLM-as-a-Judge.* arXiv:2506.22316
43. *A Survey of Small Language Models.* arXiv:2410.20011
44. *A Comprehensive Survey of Small Language Models in the Era of Large Language Models.* arXiv:2411.03350
45. *On-Device Language Models: A Comprehensive Review.* arXiv:2409.00088
46. *Taming the Titans: A Survey of Efficient LLM Inference Serving.* arXiv:2504.19720
47. *Which Quantization Should I Use? A Unified Evaluation of llama.cpp Quantization on Llama-3.1-8B-Instruct.* arXiv:2601.14277
48. Lin, J. et al. *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.* arXiv:2306.00978
49. *Resource-Efficient Language Models: Quantization for Fast and Accessible Inference.* arXiv:2505.08620
50. *From Prompts to Power: Measuring the Energy Footprint of LLM Inference.* arXiv:2511.05597
51. *Quantifying the Energy Consumption and Carbon Emissions of LLM Inference via Simulations.* arXiv:2507.11417
52. *How Hungry is AI? Benchmarking Energy, Water, and Carbon Footprint of LLM Inference.* arXiv:2505.09598

---

### A note on citation practice

Every reference above was located by search and its URL checked during preparation. Where a work is cited for
a specific quantitative claim, that claim is drawn from the abstract or the source's own summary. **Before
submission, retrieve and read each paper you intend to cite in the report body** — a survey is only as
trustworthy as the reading behind it, and several entries here are recent preprints whose venue status may
have changed.
