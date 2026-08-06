# Parsimony — Findings to date

**Status:** all eight modules built · **480 tests passing** · every number below regenerates with
`python reproduce.py`

This is the results summary. Design rationale lives in [`03-decision-log.md`](03-decision-log.md) (29 ADRs);
this document is what those decisions *found*.

**One caveat governs everything here.** Generation runs against `MockProvider`, not a real model. Token
counts, module logic, the fidelity gate, the cache verifier, the statistics and every behavioural metric are
real. **Latency and energy figures are simulated** and are marked as such wherever they appear; every ledger
row records `model_digest = "mock:v1"` so no simulated run can later be mistaken for a real one. Attaching
Ollama makes the latency claims real without changing any of the findings below.

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

Full stack reaches **+33.5%** total token reduction. **Every interaction term is negative.**

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

## 7. Two limitations we can name precisely

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

**The exact-hash tier has no verifier (ADR-029).** Six degenerate inputs — `""`, `"   "`, `"?"`, `"?!..."`,
`"!!!"`, `"."` — canonicalised to the empty string and therefore to one cache key, and served each other's
answers. The three-zone verifier guards only the *semantic* tier; the exact tier short-circuits before it,
trusting hash equality as semantic equality. That holds exactly as long as canonicalisation is lossless, and
nothing enforced it.

Worth a line in related work: the key-collision literature searches for adversarial suffixes. This collision
class needs no search — it is reachable by typing `?` — and it sits in the tier that threat model does not
examine, because the exact tier looks unambiguously safe.

---

## What is not yet measured

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
