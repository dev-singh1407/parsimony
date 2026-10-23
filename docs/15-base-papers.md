# Base Papers

**Project.** Parsimony — Token-Efficient LLM Interaction on CPU-Only Hardware
**Course.** BCSE497J, Project I · B.Tech · VIT, Vellore
**Team.** Arrsh Tripathi (23BCI0191) · Alok Singh (23BCI0158) · Dev Singh (23BCE0794)
**Guide.** Dr Sathya K
**Prepared.** 23 October 2026

The survey in [`10-literature-survey.md`](10-literature-survey.md) covers forty papers. Two of
them are *base papers* in the strict sense: the work is built on them, inherits their design,
and sets out to test a claim each of them leaves unexamined. One for each of the two
load-bearing modules. They are stated here with what is taken from each, what is changed, and
which experiment in the paper reports the outcome.

---

## Base Paper 1 — the compression side

> **LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models**
> Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, Lili Qiu (Microsoft)
> *Proceedings of EMNLP 2023.* arXiv:2310.05736
> <https://arxiv.org/abs/2310.05736> · <https://aclanthology.org/2023.emnlp-main.825/>

**What it establishes.** That a prompt can be shortened *before* a black-box model reads it,
and the three-part shape every later system inherits: a **budget controller**, a
**compression step**, and an **alignment** between the small model doing the compressing and
the large one doing the answering. It reports up to 20× compression with little loss on
GSM8K, BBH, ShareGPT and Arxiv-March23.

**What Parsimony takes.** The problem statement, and the budget controller — which survives
as module **M5**, the output budgeter.

**What Parsimony changes.**

| | LLMLingua | Parsimony M1 |
|---|---|---|
| Granularity | token level | sentence level |
| Signal | perplexity under a second LM | BM25 blended with a sentence encoder |
| Query | task-agnostic | **question-aware** |
| Cost | a second forward pass | no second model; CPU-affordable |
| Auditability | a shorter string | every sentence's score and reason is shown |

**The claim this project tests.** LLMLingua is evaluated in isolation, against an
uncompressed baseline, on datacentre GPUs. Parsimony asks what happens when compression is
*composed* with caching, history management and routing, on a CPU.

**Where it is answered.** Experiments **E1** (savings are not additive — four modules summing
to 29.0 pp deliver 27.4), **E2** (prefill is 91.7–98.8 % of CPU inference time, so the cost
model inverts relative to GPU serving) and **E6** (the *relevance floor*, not the token
budget, is the binding constraint on multi-hop recall — raising the budget saturates at 85 %
while lowering the floor reaches 90 %).

---

## Base Paper 2 — the reuse side

> **GPTCache: An Open-Source Semantic Cache for LLM Applications Enabling Faster Answers and Cost Savings**
> Fu Bang
> *Proceedings of the 3rd Workshop on Natural Language Processing Open Source Software
> (NLP-OSS 2023), co-located with EMNLP 2023*, pages 212–218.
> DOI 10.18653/v1/2023.nlposs-1.24 · <https://aclanthology.org/2023.nlposs-1.24/>

**What it establishes.** The design every semantic cache follows: **embed** the incoming
query, **compare** it against stored queries, **serve** the stored answer when similarity
exceeds a threshold. Reports 2–10× faster responses on cache hits against OpenAI's service.

**What Parsimony takes.** The architecture wholesale — two-tier lookup, embedded keys,
similarity search — as module **M2**.

**What Parsimony changes.** It rejects the sufficiency of the threshold, which is the single
assumption GPTCache's design rests on.

| | GPTCache | Parsimony M2 |
|---|---|---|
| Accept decision | similarity ≥ threshold | similarity gate **then a verifier** |
| Verifier | — | numbers, entities, negations, operative modifiers, question kind |
| Safety evidence | not measured adversarially | 45 adversarial near-duplicate pairs |
| Cost of the verifier | — | microseconds, no second neural pass |

**The claim this project tests.** Whether *any* similarity threshold can separate a genuine
paraphrase from a one-token meaning inversion.

**Where it is answered.** Experiment **E3**: the false-hit rate is **flat at 51.1 % for every
threshold from 0.80 to 0.99** — the threshold has no usable setting, because a negation
inversion is, to an encoder, a near-duplicate. Four set comparisons costing microseconds take
it to **0.0 % (0/45)**.

**And the question GPTCache leaves open.** A safe cache is worthless if it never fires.
Experiment **E7** measures the other side — how much *legitimate* reuse a verified cache
permits — and finds the verified cache reused only **4 of 40** genuinely reworded questions.
Correcting the text the encoder is shown, the overlap measure, the comparison of what kind of
answer is wanted, and two entity-extraction faults raises that to **31 of 40**, with the
false-serve count at **zero throughout**.

---

## Why these two and not others

The survey's other strong candidates, and why each is related work rather than a base paper:

| Paper | Relation | Why not a base paper |
|---|---|---|
| **[3] LongLLMLingua** (arXiv:2310.06839) | **Closest methodological sibling.** The one prior system that makes compression *question-aware*, as M1 is. | Its mechanism is conditional perplexity under a second language model — the cost a CPU-only deployment cannot pay, and which this work exists to avoid. |
| **[1] Selective Context** (Li et al., EMNLP 2023, arXiv:2310.06201) | Foundational: self-information to drop low-value units. | Query-agnostic. M1's entire design turns on being question-aware. |
| **[23] PagedAttention / vLLM** (Kwon et al., SOSP 2023) | M4's prefix-stability work assumes this KV-reuse model. | Infrastructure Parsimony runs *on*, not a technique it extends. |
| **[28] Lost in the Middle** (Liu et al., TACL 2024) | Supplies one half of the conflict E2 prices. | A finding about model behaviour, not a system this one is built on. |

> If the guide wants the base papers framed strictly around **question-aware compression**,
> substitute **LongLLMLingua [3]** for **LLMLingua [2]**. LLMLingua is proposed here because
> it is the canonical statement of the problem and the origin of the budget controller the
> system actually uses; LongLLMLingua is its extension rather than its foundation.

---

## One-line summary for a title slide

> Parsimony composes the prompt-compression idea of **LLMLingua** with the semantic-cache
> design of **GPTCache**, on CPU-only hardware, and replaces the similarity threshold that
> GPTCache relies on with an invariant verifier — because no threshold separates a paraphrase
> from a negation.
