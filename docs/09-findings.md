# Parsimony — Findings to date

**Status:** all eight modules built · **595 tests passing** · every number below regenerates with
`python reproduce.py`

This is the results summary. Design rationale lives in [`03-decision-log.md`](03-decision-log.md) (34 ADRs);
this document is what those decisions *found*.

**Which numbers came from where.** Sections 1–7 and 9 run against `MockProvider`, a deterministic stand-in:
token counts, module logic, the fidelity gate, the cache verifier and every statistic are real, and the
sweeps are reproducible because the mock is. **Section 8 is measured against a real model** —
`qwen2.5:1.5b-instruct` (Q4_K_M) on CPU via Ollama — and that is where every latency claim now comes from.

The distinction is enforced, not asserted: each ledger row carries the provider's content digest, `mock:v1`
against `ollama:<digest>`, so no simulated run can be mistaken for a real one after the fact. Attaching the
real model changed **no** token result, because the tokenizer was already Qwen2.5's. It changed the latency
picture substantially, and §8 says how.

---

## 1. The headline: savings do not compound

Full 2⁴ factorial over M1/M2/M3/M5, 151 conversations, 263 requests, 17 cells.

| effect | estimate | partial η² |
|---|---|---|
| **M5** output budgeter | +13.01 pp | 0.546 |
| **M3** history manager | +11.69 pp | 0.441 |
| **M2** semantic cache | +1.61 pp | 0.008 |
| **M1** compressor | +0.14 pp | 0.000 |
| M3×M5 interaction | **−1.14 pp** | 0.004 |

Full stack reaches **+33.5%** total token reduction. **Every interaction term that is non-zero is negative**
— M3×M5 −1.14, M2×M5 −0.10, M1×M5 −0.02. The other eight are zero to six decimal places, so the honest
statement is not "the modules always interfere" but "where they interact at all, they interfere." The three
non-zero terms all involve M5, which is the tell: M5 shortens output, and only modules that change what there
is to shorten can overlap with it.

> **Additivity shortfall: 2.53 percentage points, 95% CI [+0.93, +3.99].**

The interval excludes zero, so the shortfall is a real effect rather than noise. This is Contribution 1: no
published study runs these modules in one pipeline, so the field has no evidence about whether their savings
compound. They do not — and the largest overlap is M3×M5, because trimming history and shortening output
both reduce the same conversation.

Effect size is reported rather than p-values. With a saturated single-replicate design there are no residual
degrees of freedom to test against, and at this number of observations a p-value would report sample size
rather than importance (ADR-021).

---

## 2. The published cache thresholds are unsafe at this scale

Measured on 45 adversarial pairs (one operative token apart) plus 21 controls.

The adversarial **negation** pair sits at cosine **0.924** — *higher than every genuine paraphrase in the
set*:

| cosine | pair |
|---|---|
| **0.924** | "Is it safe to mix bleach and vinegar?" / "Is it **not** safe…" |
| 0.869 | "capital of **Australia**" / "capital of **Austria**" |
| 0.850 | "capital of France" / "capital **city** of France" |
| 0.653 | "What causes rain?" / "What causes rainfall to occur?" |

The thresholds quoted as safe across the caching literature (0.85–0.92) would **auto-accept the negation
pair and return the opposite answer**, with the verifier never running.

### The verifier, not the threshold, is what makes the cache safe

Chasing the surviving false hits one at a time took the rate from **26.7% → 0.0%**:

| operative | initial | now |
|---|---|---|
| modifier | 78% | **0%** |
| negation | 31% | **0%** |
| entity | 11% | **0%** |
| number | 0% | **0%** |
| **overall** | **26.7%** | **0.0%** |

Three gaps, each found by inspecting the surviving failures rather than by guessing (ADR-027):

- **Operative modifiers.** "minimum" vs "maximum" changes no number, no entity and no negation particle, and
  leaves lexical overlap high. Nothing in the verifier could see it.
- **Negation without a particle.** Morphological ("possible"/"impossible") and lexical ("*fail to* thin the
  blood").
- **Alphanumeric identifiers.** "pandas" vs "Panda3D" reported *zero entities on both sides* — the
  proper-noun pattern cannot match across a digit.

Raising the threshold cannot substitute: similarity alone does not reach the <2% target at *any* threshold
in the sweep, because adversarial pairs sit above genuine paraphrases. The verifier is four set comparisons
costing microseconds, where the literature reaches for a cross-encoder.

**Cost, stated plainly.** At the safe operating point (τ_hi = 0.97) the true-hit rate is 33.3% (7/21
controls). The cache is safe and conservative; two thirds of legitimate paraphrases are missed. Loosening to
0.90 buys 4.8 points of true hits for 8.9% false hits — a bad trade, which is how 0.95–0.97 was chosen from
data rather than preference.

---

## 3. Saving tokens can cost time, invisibly

Mean KV prefix tokens reused over a 5-turn conversation:

| configuration | prefix tokens reused | input tokens sent |
|---|---|---|
| M4 off (volatile head) | 2.4 | — |
| M4 on, chronological | **55.8** | 626 |
| M4 on, position-aware | **10.4** | 614 |

**The two M4-on arms retain the same turns; only the order differs.** Position-aware placement — which
exists to exploit the start/end attention bias — costs **81% of the prefix reuse** while sending 2% *fewer*
tokens (ADR-025).

Every metric in the compression literature would score the position-aware arm as the marginally better one.
It is roughly five times more expensive to prefill. This is Contribution 3, and it is a sharper case than
the compression example the report anticipates: compression at least *shows up* as a token reduction, so a
practitioner sees a trade. Reordering shows up as an improvement.

> **Precision note.** The 626/614 gap is a MockProvider artefact, not a property of reordering: the mock's
> response length is a function of the prompt hash, so changing turn order perturbs downstream history
> lengths. With a real model the same turns in a different order send an identical token count and the
> comparison becomes exact. The prefix-reuse gap is not affected — it is measured on the prompts actually
> assembled.

---

## 4. Negative yield is a position-0 effect, and it hits the simplest tier

The report states that deleting a word can raise the token count. Measured (`parsimony tokenprobe`):

| edit regime | tested | saved tokens | saved nothing | **cost** tokens |
|---|---|---|---|---|
| phrase substitution | 21 | 21 | 0 | **0** |
| single-word deletion | 2,199 | 2,183 | 13 | **3** |
| sub-token edit | 8 | 0 | 5 | **3** |

**Mid-string deletion is monotone.** BPE encodes the leading space into the token, so removing a word from
the middle takes exactly its tokens and leaves neighbouring merges intact.

**Deleting the *first* word is not.** All three increases are first-word removals, and there are **two
distinct mechanisms**, neither universal:

| effect | example | tokens |
|---|---|---|
| loses the leading-space form | `" happened"` → `"happened"` | 1 → **3** |
| capitalised at position 0 | `"explain"` → `"Explain"` | 1 → **2** |

**This hits M1 tier 1 — the simplest tier in the stack.** Tier 1 strips leading politeness and re-capitalises
the new opener, which vacates position 0 and triggers both effects:

```
"Please explain recursion."  →  "Explain recursion."
        4 tokens                      4 tokens          — the saving is handed straight back
```

Measured over the corpus, **6 of 18 tier-1 edits saved zero or negative tokens** — a third of the tier's work
was perturbing the user's text for nothing. Tier 1 now runs the same negative-yield guard as tier 3; applied
edits fell 18 → 12 while tokens saved rose 48 → 49 (ADR-030).

**Why this matters for the report.** Negative-yield detection is presented as a refinement of tier-3
rewriting. It is more general than that: the effect is a *position-0 boundary* phenomenon, it is triggered by
the least clever transformation in the stack, and **which of the two mechanisms applies is word-dependent** —
`explain` loses to capitalisation, `revoke` to the leading space, `quantify` to neither.

That is the strongest form of the argument: **you cannot predict whether an edit pays from its shape; you
have to tokenise it.** Which is exactly why the guard is a re-tokenisation and not a rule — and why any
method that strips sentence openers, including the stop-word and discourse-marker deletion that compression
baselines routinely perform, is exposed to it.

**Methodological note.** ADR-026 originally claimed whitespace-aligned deletion was monotone outright. That
was true of its 495-deletion sample and false at 2,199. It was caught only because the corpus grew and the
probe was re-run — so every empirical claim here gets re-checked against the final corpus before the report
is written.

---

## 5. Approximate search would have flattered the safety result

ADR-004 argued that an approximate index contaminates the false-hit rate. Rather than leave that as an
assertion, `LshIndex` was implemented and measured (verifier off, so the index is the only thing separating
the pairs):

| index | false-hit rate | true-hit rate |
|---|---|---|
| `ExactIndex` | **84.4%** | 52.4% |
| `LshIndex` (approximate) | **46.7%** | 28.6% |

The approximate index does not make the cache safer — it is worse at retrieval in *both* directions. It
simply fails to fetch the dangerous neighbour, so the danger goes uncounted. **Using FAISS would have
understated the headline safety number by 37.7 percentage points, in the flattering direction.**

Generalises beyond this project: any paper reporting a false-hit rate over an ANN index is reporting a
number that partly measures its index.

---

## 6. Which module to switch on depends on the question

Contribution 6, in the form a practitioner can act on. Token reduction per query class:

| cell | arithmetic | code | factual | follow-up | paraphrase | summarisation |
|---|---|---|---|---|---|---|
| M5 | +16.8 | +11.9 | +12.6 | +14.7 | +13.1 | +11.2 |
| **M3** | 0.0 | 0.0 | 0.0 | **+17.7** | 0.0 | 0.0 |
| **M2** | 0.0 | 0.0 | 0.0 | +0.2 | **+31.1** | 0.0 |
| M1 | 0.0 | +2.8 | +1.1 | 0.0 | −0.5 | −0.2 |
| full + M6 | **+69.3** | +27.7 | +22.8 | +32.4 | +46.0 | +19.2 |

_Sub-1% values are noise: the mock's response length is a function of the prompt hash, so any change to the
prompt perturbs output length slightly. Treat M1's per-class row as "roughly zero except on code"._

**M3 helps exactly one class. So does M2.** M3 only has history to trim in multi-turn conversations; M2 only
has repeats to serve in the paraphrase class. Averaged together both look mediocre; per class each is
decisive for its own workload. A single headline percentage hides this completely — which is the argument
for a calibration table rather than a number.

M6's deterministic tier gives arithmetic **+69.3%** while *raising* gold-subset accuracy from 5% to 35%,
because exact arithmetic is correct by construction where a 1B model guesses.

---

## 7. Calibrations transfer as ratios, not as mechanisms

Report §4.6 asks whether a calibration survives being applied elsewhere without re-tuning. Three LLMs need
Ollama, but the **tokenizer** determines every token count, every negative-yield decision and prefix
survival — so that dimension is answerable now, with two real vocabularies (Qwen2.5 at 151,665, GPT-2 at
50,257). Same cells, same corpus, **no re-tuning**.

| cell | Qwen2.5 | GPT-2 |
|---|---|---|
| M1 | 0.16% | 0.16% |
| M2 | 1.72% | 1.72% |
| M3 | 12.84% | 12.84% |
| M5 | 14.27% | 14.24% |
| M1+M2+M3+M5 | 26.46% | 26.37% |

**Reduction ratios transfer almost exactly**, and module ranking is identical — because a reduction is a
ratio, and a roughly constant vocabulary factor cancels. Absolute counts differ substantially ("Please
explain recursion." is 4 tokens under Qwen, 5 under GPT-2).

**But the mechanisms do not all transfer:**

| claim | Qwen2.5 | GPT-2 | transfers |
|---|---|---|---|
| `" happened"` cheaper than `"happened"` | 1 vs 3 | 1 vs 3 | yes |
| `"explain"` cheaper than `"Explain"` | 1 vs 2 | **2 vs 2** | **no** |
| first-word deletion can raise the count | 11 vs 12 | 7 vs 8 | yes |

GPT-2 has no capitalisation penalty. **ADR-030 stated that mechanism as general; it is Qwen-specific**, and
finding that is precisely what a generalisation study is for (ADR-032).

The tier-1 zero-yield rate is 7/20 (35%) under *both* — matching rates from differing mechanisms, which is
worth reporting because the matching number would otherwise be read as agreement.

**Scope, stated plainly:** this is the tokenizer dimension of §4.6. Decode speed, answer quality and
quantisation belong to the model and still need a real provider.

## 8. On a real model, the prompt side is the expensive half

Everything before this section was measured against `MockProvider`, which invents TTFT and TPOT from two
constants. `qwen2.5:1.5b-instruct` (Q4_K_M) now runs locally on CPU — the same model whose vocabulary the
token counts already used, so attaching it invalidated nothing.

Reading Ollama's own prefill/decode split rather than wall-clock TTFT (which conflates prefill with HTTP and
scheduling overhead):

| input tokens | prefill | decode | ms / input token | prefill share |
|---|---|---|---|---|
| 146 | 1,360 ms | 122 ms | 9.31 | 91.7% |
| 257 | 2,119 ms | 124 ms | 8.25 | 94.5% |
| 474 | 3,902 ms | 165 ms | 8.23 | 95.9% |
| 906 | 7,691 ms | 131 ms | 8.49 | 98.3% |
| 1,338 | 11,609 ms | 136 ms | 8.68 | 98.8% |

**Prefill is linear at ~8.5 ms per input token and is 92–99% of total time.** This is research gap 2, and it
is the empirical foundation the project previously had to assume: on CPU, input tokens *are* the cost.

The simulation was wrong in the direction that mattered — it assumed 120 ms TTFT, understating the prompt
side by more than an order of magnitude, and 65 ms/token decode against a real 37–47 ms.

Applying the measured rate converts every token result into wall clock:

| cell | input tokens saved | prefill saved | per request |
|---|---|---|---|
| M5 | 3,660 | 31.1 s | 118 ms |
| M3 | 6,469 | 55.0 s | 209 ms |
| M1+M2+M3+M5 | 9,308 | 79.1 s | 301 ms |
| **full stack** | **11,836** | **100.6 s** | **383 ms** |

**And prompt order has a price.** ADR-025 measured position-aware placement in *prefix tokens reused*, a
proxy nobody outside this project reports. Ollama reuses the KV cache across requests, so:

| arrangement | tokens | first call | steady state | reuse |
|---|---|---|---|---|
| stable prefix (M4) | 1,490 | 14,569 ms | **212 ms** | **98.5%** |
| volatile head | 1,497 | 19,014 ms | **18,914 ms** | **0.5%** |

Same content, 7 tokens apart, **~80× the steady-state cost**. M4 is worth roughly zero tokens and eighteen
seconds.

Three traps were caught on the way, each of which produced a plausible flat line rather than an error:
consecutive probes sharing a prefix; silent context truncation deleting the unique head and making two
prompts identical; and Ollama's KV cache outliving the Python process, so only the first-ever execution
measured prefill. Every run now carries a fresh nonce (ADR-034).

### Compression does not cost accuracy

The gold column read 5% for almost every cell. That was `MockProvider` being unable to answer questions at
all, not a pipeline failure — and it made the most important quality question unanswerable. With a real
model, scored on the same 40 gold items:

| provider | cell | gold accuracy |
|---|---|---|
| mock | baseline | 2/40 — 5.0% |
| mock | full stack | 14/40 — 35.0% |
| **real** | **baseline** | **36/40 — 90.0%** |
| **real** | **full stack** | **38/40 — 95.0%** |

Paired, per item:

| | count |
|---|---|
| both correct | 36 |
| both wrong | 2 |
| **baseline right, full stack wrong** | **0** |
| baseline wrong, full stack right | 2 |

**Zero regressions.** Removing a third of the tokens did not lose a single answer the baseline got right.
That is the claim worth making, and it is the one this project exists to test.

The two gains are `847 * 23` and a date difference — both routed to M6's deterministic tier, which computes
exactly and sends the model nothing. Two discordant pairs is not statistically significant (exact McNemar,
two-sided **p = 0.50**), so the *statistical* claim stops at "no measurable degradation". The mechanism,
though, is not chance: a calculator will beat a 1.5B model at three-digit multiplication every time.

## 9. "Self-improving" is a property of the traffic

M7 mines a policy bundle from past conversations. Measured honestly — mine from one half of the
conversations, measure on a **disjoint** half, cold against warm:

| recurrence | seeds | cold | warm | transfer | extra hits | extra gate fires |
|---|---|---|---|---|---|---|
| 0.0% | 0 | 7.21% | 7.21% | **+0.00 pp** | 0 | 0 |
| 7.5% | 3 | 11.18% | 12.49% | +1.31 pp | +1 | 0 |
| 22.5% | 8 | 14.99% | 20.41% | +5.43 pp | +4 | 0 |
| 33.3% | 10 | 23.38% | 31.93% | +8.55 pp | +6 | 0 |
| 45.0% | 12 | 33.28% | 47.99% | +14.70 pp | +10 | 0 |
| 57.5% | 16 | 45.69% | 63.52% | **+17.83 pp** | +12 | 0 |

The **exact zero at 0% recurrence** is what makes the rest credible: nothing repeated, nothing mined, nothing
gained. A study whose null condition does not come out null is measuring its own plumbing.

**Extra gate fires are zero at every level.** A seeded cache can serve an answer mined from a different
question, so warm-starting could have bought tokens by serving wrong answers. It did not.

**Why M7 contributes nothing to the headline ablation:** the corpus's recurrence is **1.9%** — four repeated
questions across 263 turns. It was authored for ablation diversity, the right shape for M1/M2/M3/M5 and the
wrong shape for a module that learns from repetition. That is a fact about the corpus, not the module — the
same distinction as §8's encoder finding.

## 10. Two limitations we can name precisely

**M1 tier 2 is encoder-limited, not technique-limited (ADR-028).** Intended near-duplicates span:

| cosine | pair |
|---|---|
| 0.729 | "The library will close at 6 PM on weekdays" / "The library shuts at 6 PM on weekdays" |
| **0.412** | "Rent is 1200 per month" / "Monthly rent comes to 1200" |
| **0.321** | "The recipe needs 250 g of flour" / "You will need 250 g flour for this" |

The bottom two are the same fact reworded — exactly what tier 2 exists to delete — and the lexical encoder
places them barely above unrelated text. No threshold recovers them without merging sentences that share
nothing but function words.

Without this measurement the natural conclusion would have been *"extractive redundancy removal does not
help at small scale"* — and it would have been **wrong**. The technique was never given a working similarity
signal. This is the concrete, quantified case for swapping the lexical encoder for MiniLM.

**Memory was unbounded in the component designed to accumulate (ADR-031).** Probed under sustained load, 400
distinct queries produced 400 cache entries, 400 tracked conversations and 415 blobs, with nothing ever
released. Report §4.7 targets an 8 GB consumer laptop. Now LRU-bounded at 10,000 entries, with the vector
index evicted alongside each entry — an orphaned vector would keep scoring in `search()` and return an id
that no longer resolves. The cap sits 38× above the corpus request count, so eviction never fires during a
sweep and every result is byte-identical with it in place.

**The exact-hash tier has no verifier (ADR-029).** Six degenerate inputs — `""`, `"   "`, `"?"`, `"?!..."`,
`"!!!"`, `"."` — canonicalised to the empty string and therefore to one cache key, and served each other's
answers. The three-zone verifier guards only the *semantic* tier; the exact tier short-circuits before it,
trusting hash equality as semantic equality. That holds exactly as long as canonicalisation is lossless, and
nothing enforced it.

Worth a line in related work: the key-collision literature searches for adversarial suffixes. This collision
class needs no search — it is reachable by typing `?` — and it sits in the tier that threat model does not
examine, because the exact tier looks unambiguously safe.

---

## 11. What is not yet measured

- **Real latency.** Everything runs on `MockProvider`. TTFT/TPOT, the prefill/decode split behind Gap 2, and
  the energy column become real the moment a provider is attached. The two-pass sweep (memoised quality
  pass, unmemoised timing pass) is built and waiting for it.
- **Cross-model generalisation.** The calibration table has one model in it. The harness runs three.
- **A real judge.** The model-as-judge is a length-biased stand-in — deliberately, so the swap-disagreement
  machinery can be shown to detect bias. Its 91–98% disagreement rate on near-identical answers is the
  machinery working, not a quality signal.
- **True-hit rate at scale.** 21 control pairs is a thin denominator; it should roughly double.

## Reproducing all of it

```bash
python reproduce.py --out figures
```

Roughly 40 seconds: 17 cells, both passes, 11 CSVs and a full report. Generation memoisation avoids **78.2%**
of model calls and is bit-exact — ablation, effects and Pareto output are byte-identical with it disabled.
