# Parsimony — Findings to date

**Status:** all eight modules built · **853 tests passing** · every number below regenerates with
`python reproduce.py`

This is the results summary. Design rationale lives in [`03-decision-log.md`](03-decision-log.md) (39 ADRs);
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
| **M5** output budgeter | +13.43 pp | 0.556 |
| **M3** history manager | +11.82 pp | 0.430 |
| **M2** semantic cache | +1.93 pp | 0.012 |
| **M1** compressor | +0.23 pp | 0.000 |
| M3×M5 interaction | **−0.70 pp** | 0.002 |

Full stack reaches **+33.9%** total token reduction. The two material interaction terms are both negative —
**M3×M5 −0.70** and **M2×M5 −0.11** — and both involve M5, which is the tell: M5 shortens output, so only
modules that change what there is to shorten can overlap with it. Every other term sits within ±0.02 and is
indistinguishable from zero here, so the honest statement is not "the modules always interfere" but "where
they interact at all, they interfere."

> **Additivity shortfall: 1.66 percentage points, 95% CI [+0.02, +3.25].**

This is Contribution 1, and the honest version of it is more interesting than the original. No published
study runs these modules in one pipeline, so the field has no evidence about whether their savings compound.
They do not. But **how much they fail to compound is a property of the configuration, not a constant**:

| encoder | M2 effect | M3×M5 | additivity shortfall |
|---|---|---|---|
| `hashing-v1` | +1.61 pp | −1.14 | 2.53 pp, **[+0.93, +3.99]** — excludes zero |
| `content-v1` (default) | +1.93 pp | −0.70 | 1.66 pp, **[+0.02, +3.25]** — clears zero by 0.02 |

Improving the encoder (ADR-035) made the cache hit more often, so it overlapped its neighbours less and the
shortfall shrank until its interval reached zero. **The weaker encoder was not reinstated to protect the
result.** Choosing a component known to be inferior because it produces a more publishable number is the
failure mode this project is written against — the same instinct that would have had us quote the
literature's 0.85 cache threshold and never run the adversarial set.

The largest overlap remains M3×M5 under both encoders, because trimming history and shortening output both
reduce the same conversation.

Effect size is reported rather than p-values. With a saturated single-replicate design there are no residual
degrees of freedom to test against, and at this number of observations a p-value would report sample size
rather than importance (ADR-021).

---

## 2. The published cache thresholds are unsafe at this scale

Measured on 45 adversarial pairs (one operative token apart) plus **45 controls** — pairs that mean the same
thing and *should* hit.

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

**Cost, stated plainly.** At the operating point (τ_hi = 0.97) the true-hit rate is **28.9% (13/45
controls)** with the default `content-v1` encoder. The cache is safe and conservative; roughly seven in ten
legitimate paraphrases are still missed.

| τ_hi | false hits | true hits |
|---|---|---|
| 0.90 | **4.4%** | 35.6% |
| 0.92 | 0.0% | 31.1% |
| **0.97 (default)** | **0.0%** | **28.9%** |

τ_hi stays at 0.97 rather than 0.92, even though 0.92 is also safe here and buys 2.2 more points. 0.90 is
*not* safe, so 0.92 sits one threshold step from a 4.4% false-hit rate. In a component whose failure mode is
serving the opposite answer, a step of margin is worth more than 2.2 points of hit rate (ADR-035).

Two revisions to this number are worth recording, because they moved it in opposite directions.

> **Doubling the controls lowered it.** On the original 21 controls the rate read 33.3% (7/21) under
> `hashing-v1`. Twenty-four further controls — synonymous superlatives, abbreviation/expansion pairs, digit
> versus spelled ordinals, clause reordering — took it to 22.2%, because the added pairs are harder than the
> ones already there. **The false-hit rate did not move: 0.0%, unchanged.** The safety claim survived a
> doubled denominator; the *benefit* claim had been optimistic on a thin one.
>
> **Fixing the encoder raised it again.** `content-v1` (ADR-035) took it from 22.2% to **28.9%** at the same
> threshold and the same 0.0% false-hit rate — the gain coming from exactly the paraphrases §11 identifies
> as the encoder's blind spot.

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
| **real** | **baseline** | **37/40 — 92.5%** |
| **real** | **full stack** | **39/40 — 97.5%** |

**Zero regressions.** Removing a third of the tokens did not lose a single answer the baseline got right.
That is the claim worth making, and it is the one this project exists to test.

The two gains are `847 * 23` and a date difference — both routed to M6's deterministic tier, which computes
exactly and sends the model nothing. Two discordant pairs is not statistically significant (exact McNemar,
two-sided **p = 0.50**), so the *statistical* claim stops at "no measurable degradation". The mechanism,
though, is not chance: a calculator will beat a 1.5B model at three-digit multiplication every time.

> **A grading bug found by inspecting the failures rather than the totals.** Asked for the chemical symbol
> for tungsten, the model answered *"The chemical symbol for tungsten is W."* and was scored **wrong**. The
> `exact` rule required the whole response to equal the gold answer, which no conversational model can
> satisfy — and a unit test asserted precisely that, pinning the implementation instead of the intent and
> making the bug permanent. Only 1 of the 40 items uses the rule, which is why it survived. `exact` now
> matches the answer as a standalone token, and the figures above are the corrected ones (ADR-037).

### A bigger model does not help here

M6's escalation tier routes hard queries to a larger model. Two questions, both now answerable, both
answered no.

**It could not fire.** `escalation_complexity` shipped at 0.75 against an observed maximum complexity of
**0.406** across 237 routed requests — 0 escalations, and none possible. Dead code wearing a configuration
option, the same failure as the 0.80 dedup threshold in §11. It is now calibrated to 0.20, the ~90th
percentile, so the option means something when enabled.

**And it would not have helped.** On the 40 gold items:

| model | size | gold | wall clock |
|---|---|---|---|
| qwen2.5:1.5b-instruct | 0.92 GB | 36/40 | 139 s |
| llama3.2:3b | 2.0 GB | 36/40 | 161 s |

Item for item identical — 36 both right, 4 both wrong, **zero** where one model succeeded and the other
failed — for 16% more wall clock and twice the memory. So `escalation_tier` stays off, now for a measured
reason instead of an accidental one (ADR-037).

**The caveat matters more than the finding.** After the grading fix, only two gold items defeat both models,
and both are arithmetic that M6's deterministic tier already answers without a model. So the gold set holds
no question in the band where a 1.5B model fails and a 3B model succeeds — this measures the *gold set* as
much as the models. The defensible claim is narrow: escalation is not justified **on this corpus**.

### The judge failed its own calibration

With a real model available, the obvious next step is LLM-as-judge — and `judge_pairwise` names the
constraint: the judge must not be the model under test. So `llama3.2:3b` judges `qwen2.5:1.5b-instruct`,
different families, no self-preference.

Before believing anything it says, it was shown **the same answer in both slots** and asked to choose. A
question with no right answer. At chance it names slot A half the time.

| | |
|---|---|
| position bias on identical answers | **50.0 pp** — it always picks the same slot |
| unreadable verdicts | 0.0% |
| **verdict** | **not usable** |

Its 91.7% swap-disagreement rate on the real comparison says the same thing from the other side: it is
answering by position, not by content. So its 45.8% win rate for the full stack is not a quality
measurement — it is what a coin looks like when you write down which way up it landed.

**Why this is reported rather than dropped.** 45.8% is a publishable-looking number. It sits near parity, it
has a ready story ("compression costs a little quality"), and nothing in the number reveals that the
instrument was broken. The calibration is the only thing between it and a results table, it needs no ground
truth, and it costs one call per item — and it is the step most LLM-as-judge setups skip (ADR-036).

`LengthBiasedMockJudge` therefore stays the default: a judge biased in a *known* direction is a better
instrument than one biased in an unmeasured one. The quality claim rests on the measures that need no judge
— gold accuracy above, token overlap, embedding similarity.

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
same distinction as §11's encoder finding.

## 10. Fuzzing found a bypass the adversarial corpus could not

Every safety number above is measured against 45 adversarial pairs and 45 controls, all plain ASCII English.
0.0% false hits is a statement about *that corpus*. Twenty hostile inputs — empty, 8,000 words, emoji,
control characters, RTL overrides, SQL injection, surrogates — were put through the full stack. **Nothing
crashed.** Two results were wrong, and both mattered.

**The verifier could be bypassed with an invisible character.** It reads negation particles out of raw text.
One zero-width character inside "not" splits it into fragments matching no lexicon entry, so the negation
check *agrees* and the verifier passes a question against its own opposite:

| variant | negation seen? | verifier |
|---|---|---|
| `Is it not safe to mix bleach and vinegar?` | yes | rejects ✓ |
| `Is it n`·`ot safe…` (zero-width space) | **no** | **passes** ✗ |
| soft hyphen · zero-width joiner · word joiner | **no** | **passes** ✗ |
| Cyrillic `о` in place of Latin `o` | **no** | **passes** ✗ |

Seven variants. The only thing between that and serving the opposite answer was the embedder happening to
score the mangled text at 0.880, below τ_hi — **luck, not a defence**, and dependent on a property of an
encoder that ADR-035 has already changed once.

Fixed by `sanitise()`: NFKC, drop every Unicode category **Cf** character (invisible by definition, so it
cannot carry meaning a reader intended), fold Cyrillic and Greek look-alikes. Applied at all four places the
verifier reads text — a bypass in any one is a bypass overall — and deliberately **never** to the text sent
to the model, because a user who writes Cyrillic must get their own words back.

**Every non-Latin query was being deleted outright.** Found by a test written for the bypass, which failed
for an unrelated reason. M1 tier 1 classified contentless debris with `[A-Za-z0-9]`, so a query with no Latin
letter matched nothing, was dropped whole, and the model received an **empty prompt**:

```
"Как дела?"   "नमस्ते, यह क्या है?"   "இது என்ன?"   "你好世界"   →   ""
```

For a project written at an Indian university, a question in Hindi or Tamil vanished silently.

**And the gate could not see it — the deeper problem.** Every check the fidelity gate makes asks *"was a
value I could extract lost?"*. That silently makes its guarantee **conditional on the extractor's language
coverage**. The extractors are Latin-only regexes, so non-Latin text yields no invariants at all: deleting
the entire question lost nothing the gate could name, and it passed. An always-on gate whose guarantee is
void for most of the world's writing systems is not the gate the architecture claims.

The gate now refuses any transform that removes **all** word characters, checked before the invariant
comparison and independent of it — the only kind of check that can hold for languages the extractors cannot
read. Widening the regex would have fixed the instance and left the class (ADR-038).

**Every headline number is unchanged** by these fixes: +33.9% full stack, 0.0% false hits, 1.66 pp shortfall.
They close holes without moving a result, which is what a security fix should look like when the original
measurements were sound. What changed is the *scope* of the safety claim: 0.0% is now a statement about a
corpus **and a sanitiser**, rather than about a corpus that happened to contain no adversarial Unicode.

## 11. Two limitations we can name precisely

**M1 tier 2 was encoder-limited, not technique-limited (ADR-028) — and the encoder has since been partly
fixed (ADR-035).** Intended near-duplicates scored:

| pair | `hashing-v1` | `content-v1` |
|---|---|---|
| "The library will close at 6 PM on weekdays" / "…shuts at 6 PM on weekdays" | 0.729 | 0.738 |
| "Rent is 1200 per month" / "Monthly rent comes to 1200" | **0.412** | 0.561 |
| "The recipe needs 250 g of flour" / "You will need 250 g flour for this" | **0.321** | **0.847** |

The bottom two are the same fact reworded — exactly what tier 2 exists to delete — and the original lexical
encoder placed them barely above unrelated text. Without that measurement the natural conclusion would have
been *"extractive redundancy removal does not help at small scale"*, and it would have been **wrong**: the
technique was never given a working similarity signal.

Dropping stopwords and stemming what remains — still purely lexical, still no PyTorch — recovers much of the
gap, most dramatically on the worst pair. So the ADR-028 recommendation is **partly discharged, and the
remaining case for MiniLM is weaker than it looked.** What survives is that the middle pair sits at 0.561,
still below any threshold that could safely merge it, so tier 2 remains limited by its similarity signal
rather than by its logic.

A warning attaches to the fix, recorded in full in ADR-035: the first stopword list was an ordinary
information-retrieval one, which strips "not". Under it, *"Is it safe to mix bleach and vinegar?"* and *"Is
it **not** safe…"* embedded to **cosine 1.000** — bit-identical vectors for opposite questions — putting an
11.1% false-hit floor under every threshold including 0.99. Stopword lists are built for retrieval, where
negation is noise. In a cache verifier it is the entire signal.

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

## 12. What is not yet measured

- **Real latency.** Everything runs on `MockProvider`. TTFT/TPOT, the prefill/decode split behind Gap 2, and
  the energy column become real the moment a provider is attached. The two-pass sweep (memoised quality
  pass, unmemoised timing pass) is built and waiting for it.
- **Cross-model generalisation.** The calibration table has one model in it. The harness runs three.
- **A real judge.** The model-as-judge is a length-biased stand-in — deliberately, so the swap-disagreement
  machinery can be shown to detect bias. Its 91–98% disagreement rate on near-identical answers is the
  machinery working, not a quality signal.
- **True-hit rate at scale.** Now 45 controls against 45 adversarial. The false-hit rate held at 0.0%
  and the true-hit rate fell to 22.2%, so the remaining question is not the denominator but the encoder:
  the missed paraphrases are the same lexical-similarity failure ADR-028 quantifies.

## Reproducing all of it

```bash
python reproduce.py --out figures
```

Roughly 40 seconds: 17 cells, both passes, 11 CSVs and a full report. Generation memoisation avoids **78.2%**
of model calls and is bit-exact — ablation, effects and Pareto output are byte-identical with it disabled.
