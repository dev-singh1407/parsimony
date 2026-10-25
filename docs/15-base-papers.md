# Base Papers

**Project.** Parsimony — Token-Efficient LLM Interaction on CPU-Only Hardware
**Course.** BCSE497J, Project I · B.Tech · VIT, Vellore
**Team.** Arrsh Tripathi (23BCI0191) · Alok Singh (23BCI0158) · Dev Singh (23BCE0794)
**Guide.** Dr Sathya K
**Prepared.** 23 October 2026

Two base papers, one for each of the project's two load-bearing modules. Both are recent —
AAAI 2025 and ICLR 2026 — and both are closer to what this project actually built than the
2023 papers that founded the field. The full PDFs accompany this note.

---

## Base Paper 1 — the compression side

> **Prompt Compression with Context-Aware Sentence Encoding for Fast and Improved LLM Inference**
> Barys Liskavets, Maxim Ushakov, Konstantin Yastrebov, Shuvendu Roy, Ali Etemad, Alexander Wong, Shane Luke
> **AAAI-25** — *Proceedings of the AAAI Conference on Artificial Intelligence*, **39(23), 24595–24604**
> DOI [10.1609/aaai.v39i23.34639](https://doi.org/10.1609/aaai.v39i23.34639) · arXiv **[2409.01227](https://arxiv.org/abs/2409.01227)**
> **arXiv v1 2 Sep 2024 · last revised Dec 2024 · published 2025**
> File: `papers/base/CPC-AAAI2025-Liskavets-arXiv-2409.01227.pdf` (6 pp)

**What it establishes.** *Context-aware prompt compression* (CPC): compression at **sentence
granularity**, driven by a **sentence encoder that scores each sentence's relevance to the
question being asked**. The encoder is trained contrastively — positives are sentences
relevant to the question, negatives are irrelevant context. CPC reports inference **up to
10.93× faster than the best token-level method**.

**Why this is the base paper and not LLMLingua.** It is the same design decision Parsimony's
M1 makes, made independently and published at AAAI:

| | CPC (AAAI-25) | Parsimony M1 |
|---|---|---|
| Granularity | sentence | sentence |
| Query | **question-aware** | **question-aware** |
| Scoring | context-aware sentence encoder | BM25 blended with a sentence encoder |
| Second LLM needed | no | no |
| Target | faster inference | faster inference **on CPU** |

CPC's headline — that sentence-level question-aware compression beats token-level methods on
speed by an order of magnitude — is the empirical justification for the architecture
Parsimony chose. This project can now cite a 2025 AAAI result where it previously had only
its own reasoning.

**What Parsimony adds.** Three things CPC does not address.

1. **The binding constraint.** CPC scores sentences; it does not ask which constraint stops
   selection. Experiment **E6** finds it is the **relevance floor**, not the token budget:
   raising the budget saturates at 85 % recall while lowering the floor reaches 90 %. Six
   mechanisms built to *find* second-hop evidence found it and had it discarded below the
   floor.
2. **A floor that is read rather than fixed.** Replacing the constant with a reading of the
   score distribution at its steepest *multiplicative* fall keeps identical evidence on 81
   held-out items while sending less context (26.6 % against 27.0 %).
3. **Composition.** CPC is evaluated alone. **E1** measures it stacked with caching, history
   and routing, and finds savings are **not additive** — four modules summing to 29.0 pp
   deliver 27.4.

**Foundational ancestor, for the related-work section:** LLMLingua (Jiang et al., EMNLP 2023,
arXiv:2310.05736) states the problem and contributes the budget controller that survives as
M5. LongLLMLingua (arXiv:2310.06839) makes compression question-aware but pays for it with
conditional perplexity under a second language model — the cost a CPU-only deployment cannot
afford, and the reason CPC's encoder-based approach is the relevant antecedent.

---

## Base Paper 2 — the reuse side

> **vCache: Verified Semantic Prompt Caching**
> Luis Gaspar Schroeder, Aditya Desai, Alejandro Cuadron, Kyle Chu, Shu Liu, Mark Zhao,
> Stephan Krusche, Alfons Kemper, Matei Zaharia, Joseph E. Gonzalez
> **ICLR 2026** (accepted) · arXiv **[2502.03771](https://arxiv.org/abs/2502.03771)**
> **arXiv v1 6 Feb 2025 · v5 21 Feb 2026**
> File: `papers/base/vCache-ICLR2026-Schroeder-arXiv-2502.03771.pdf` (18 pp)

**What it establishes.** That **a single static similarity threshold gives a semantic cache no
formal correctness guarantee**, producing unexpected error rates and suboptimal hit rates.
vCache instead lets the user state an error-rate bound and learns a **separate threshold per
cached prompt** online. Reported: up to **12.5× higher hit rates** and **26× lower error
rates** than static-threshold and fine-tuned-embedding baselines.

**Why this is the strongest base paper in the set.** It is independent confirmation of this
project's central cache finding, by a group including Zaharia and Gonzalez, at ICLR 2026.

> **Parsimony E3** (measured before this paper was read): the false-hit rate is **flat at
> 51.1 % for every threshold from 0.80 to 0.99**. There is no usable setting, because to an
> encoder a one-token negation inversion *is* a near-duplicate.

Both papers reach the same diagnosis and **diverge on the remedy**, which is exactly the
relationship a base paper should have:

| | vCache (ICLR 2026) | Parsimony M2 |
|---|---|---|
| Diagnosis | one static threshold cannot be correct | **same** |
| Remedy | learn a **per-prompt threshold** online | a **non-neural invariant verifier** |
| Guarantee offered | a statistical error-rate bound | a *categorical* refusal when an invariant differs |
| Cost | online learning, per-entry state | microseconds; no second neural pass |
| Suits | high-volume serving | **single-user CPU deployment** |

Parsimony's answer is the cheaper one and it is the one a CPU can afford: rather than
estimating how wrong a threshold is likely to be, compare numbers, entities, negations,
operative modifiers and the kind of answer wanted, and refuse categorically when any of them
differ. **Four set comparisons take the false-hit rate to 0.0 % (0/45).**

**What Parsimony adds.** vCache raises hit rates under an error bound; it does not measure how
much *legitimate* reuse a verified cache permits. **E7** does: the verified cache reused only
**4 of 40** genuinely reworded questions, and the corrections raise that to **29 of 40** with
the false-serve count **zero throughout**. The published sweep shows 30 of 40 is reachable
only at the price of two wrong answers — a trade this project declines.

**Foundational ancestor, for the related-work section:** GPTCache (Bang, NLP-OSS 2023,
`2023.nlposs-1.24`) established the embed–compare–serve design that both vCache and Parsimony
inherit and both argue is insufficient as stated.

---

## Summary for a title slide

> Parsimony sits between **CPC** (AAAI-25) — sentence-level, question-aware compression — and
> **vCache** (ICLR 2026) — the finding that no static similarity threshold makes cache reuse
> safe. It composes both on CPU-only hardware, and replaces vCache's learned threshold with a
> microsecond invariant verifier that refuses categorically rather than statistically.

---

## Changes recommended to the project report

These follow from the two base papers above and from the eleven decisions recorded since the
report was last built on 11 September. Listed in the order I would make them.

**1. Replace the base-paper framing (§ Literature Review).** The report currently positions
against LLMLingua and GPTCache, both 2023. Lead with CPC and vCache instead and demote the
2023 pair to *foundational ancestors*. This is the single highest-value change: it moves the
project's claimed novelty from "we did something nobody has done" — which is harder to defend
each year — to "we reached the same diagnosis as ICLR 2026 independently, and chose a cheaper
remedy for a different deployment". The second claim is stronger **and** easier to defend in a
viva.

**2. Add the vCache agreement explicitly (§ Discussion → *Agreement and disagreement with
prior work*).** State that E3's finding was reached before vCache was read, and that vCache
confirms it on different data at a different scale. Independent convergence is evidence; a
lone result is an anecdote.

**3. Add the three new experiments (§ Experiments).** Already drafted into `latex/paper.tex`
as **E6** (the relevance floor is the binding constraint), **E7** (reuse of reworded questions
rises 4/40 → 29/40 at zero false serves) and **E8** (the fidelity gate admitted 22 of 40
meaning-changing rewrites; now 3). The report should carry the same three.

**4. Make the methodological lesson explicit (§ Conclusion).** Every component had passed
every test written for it, and those tests were small — eight examples for the gate, five
pairs for the cache. The project's own first reading of cache reuse was "4 of 5" when the
honest figure was 45 %, with two wrong answers being served that nobody had looked for. This
is the most transferable result in the work and it currently appears nowhere.

**5. Keep the negative results visible.** The three rewrites the gate still admits, the
fourteen rewordings still refused, and the sweep row showing 30/40 costs two wrong answers.
An examiner trusts a results section that names its own limits more than one that does not.

**6. Update the dates and counts.** The report is dated September 2026 and quotes 1,251 tests
and 52 ADRs; the tree is now at **1,346 tests** and **60 ADRs**. The `tools/project_counts.py`
generator already reads these from source — regenerate rather than edit by hand.
