# Parsimony — Findings to date

**Status:** all eight modules built · **1,089 tests passing** · every number below regenerates with
`python reproduce.py`

This is the results summary. Design rationale lives in [`03-decision-log.md`](03-decision-log.md) (45 ADRs);
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
| **M1** compressor | +10.97 pp | 0.365 |
| **M5** output budgeter | +10.80 pp | 0.354 |
| **M3** history manager | +6.78 pp | 0.140 |
| **M1×M3 interaction** | **−4.99 pp** | 0.076 |
| **M2** semantic cache | +1.96 pp | 0.012 |
| M1×M5 interaction | −1.75 pp | 0.009 |

Full stack reaches **+39.6%** total token reduction (input alone: −47.0%).

> **Additivity shortfall: 15.13 percentage points, 95% CI [+11.15, +18.27].**

**The largest interaction in the design is M1×M3, at −4.99 pp, and it is there because both modules are
paid out of the same tokens.** A positive third-order term follows it — **M1×M3×M5 +0.45 pp** — which is
what inclusion–exclusion predicts when three modules compete for one pool: the pairwise overlaps
double-count, and the triple corrects for it. M3 drops earlier turns; M1's context tier shortens the turns that survive
(§14). Whichever runs first collects the saving, and the second finds less to do — so running both buys far
less than the sum of running each. Every interaction that matters is negative, and the two largest involve
M1, which is the tell: it is the module whose reach now overlaps its neighbours'.

This is Contribution 1, and the honest version is stronger than the original. No published study runs these
modules in one pipeline, so the field has no evidence about whether their savings compound. They do not, and
**how much they fail to compound is a property of the configuration, not a constant** — the same experiment
gave 1.69 pp when M1 only touched the question and 15.13 pp once it could also touch the history:

| configuration | M1 effect | largest interaction | additivity shortfall |
|---|---|---|---|
| `hashing-v1` encoder, M1 on the question only | +0.22 pp | M3×M5, −1.14 then | 2.53 pp, **[+0.93, +3.99]** |
| `content-v1`, M1 on the question only | +0.24 pp | M3×M5, −0.75 then | 1.69 pp, **[+0.04, +3.21]** |
| `content-v1`, M1 on history too (shipped) | +10.97 pp | **M1×M3 −4.99** | **15.13 pp, [+11.15, +18.27]** |
| `all-minilm`, M1 on history too | +11.75 pp | M1×M3, −4.40 there | 14.81 pp, [+10.97, +18.08] |

A reviewer should read that table as a warning about single-number claims, including ours: "savings do not
add" is robust across all four rows, and "they fall short by N points" is not a property of the techniques at
all. **The weaker configurations were not reinstated to protect a smaller-looking number** — and note that
the shortfall grew as the system got better, which is the direction that makes the finding less flattering
and more useful.

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

### The encoder that fixes the missed paraphrases breaks the safety design

The cost above — seven in ten legitimate paraphrases missed — is an encoder property, and the roadmap's
first item was to replace `content-v1` with MiniLM. Doing it produced the result this project exists to
find (ADR-041):

| encoder | design | false answers | true hits | free-typing pairs | ms per question |
|---|---|---|---|---|---|
| content-v1 (lexical) | accept zone above τ_hi | 0/45 — 0.0% | 13/45 — 28.9% | 37/37 | 1 |
| content-v1 (lexical) | verify every hit | 0/45 — 0.0% | 12/45 — 26.7% | 37/37 | 1 |
| all-minilm (neural) | accept zone above τ_hi | **8/45 — 17.8%** | 23/45 — 51.1% | 36/37 | 51 |
| all-minilm (neural) | **verify every hit** | **0/45 — 0.0%** | **17/45 — 37.8%** | **37/37** | 51 |

The adversarial negation pair scores **0.924** under the lexical encoder and **0.996** under MiniLM. Under
the three-zone policy that is the difference between "below τ_hi, so the verifier sees it" and "above τ_hi,
so nothing does" — 17.8% false answers at τ_hi = 0.97, and still 2.2% at 0.99. **No threshold rescues it**,
because a negation is a smaller edit than a rephrasing in any embedding space, and a better space makes that
worse rather than better.

So part of the 0.0% reported since §2 was a property of the encoder's *weakness*, not of the policy.
Verification now runs on every candidate whatever it scores (`verify_always`, microseconds on memoised
invariants): 0.0% false answers at every threshold, for both encoders, and answer reuse rises **26.7% →
37.8%** with the neural encoder. The threshold stops being a safety parameter and becomes a candidate gate —
recalibrated to τ_lo = 0.70 for MiniLM, because two genuine rephrasings sit below the lexical encoder's 0.75.

**And it recovers the long-context answers the lexical score lost.** M1's context selector uses the same
encoder, so the swap applies there too. On the 45 held-out long-context questions the neural selector scores
**40/45 - exactly what sending the whole document scores** - at 25.5% of the context and 4.1 s of prefill
against 8.4 s, where the lexical selector scored 36/45. On the 30-question confirmation split both score
27/30. Across both splits it wins five of the six questions the two encoders disagree on, including the
"daily dose" against "once a day" miss that prompted the swap.

**And `localhost` was costing two seconds a call.** The first embedding measurement showed a flat ~2,040 ms
per request that scaled with nothing; `curl` to the same endpoint took 0.23 s. `localhost` resolves to `::1`
first, Ollama listens on IPv4, and the connection has to time out before the client falls back. At
`127.0.0.1` the same call takes **31 ms**. Every Ollama call in the project paid it, and none of the
runtime-counter numbers — prefill, decode, tokens, accuracy — could see it, because those counters start
after the connection. Wall-clock figures measured before the fix carry the constant on both sides of every
comparison. It took an *absolute* expectation of what MiniLM should cost to notice a two-second tax that
years of relative comparisons would have hidden.

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

**Every headline number is unchanged** by these fixes: the full stack's reduction, the 0.0% false-hit rate and the shortfall all read exactly as they did before them.
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

## 13. Compression has nothing to compress until the request carries context

The compressor was the weakest module in every table: **0.23%** of tokens on the conversation corpus. The
explanation is in the corpus, not the module — the median question there is **six words** long. Prefill is
92–99% of wall clock (§8), and the input tokens of a real request are mostly *context*: retrieved passages,
a pasted report, an earlier long answer. A request had nowhere to put any of that, so the module was
measured on the one input where its ceiling is a rounding error (ADR-040).

Requests now carry documents, and M1 has a context tier that removes whole sentences from them — the ones
this question does not need — while the fidelity gate checks, independently, that every kept sentence is
verbatim, in order, and that anything the question names is still present.

### A benchmark where compression is worth seconds

`corpus/longctx_*.jsonl`: 102 documents in 17 fictional collections, 85 questions, six documents (~1,000
tokens) per question, with deliberate distractors. Fictional, so the model cannot answer from memory:
**the closed-book arm scores 1/45**, which is what makes every other row mean something. Splits are by
collection — dev (10) for tuning, test (45) reported, test2 (30) authored later and left untouched. Every
baseline gets the same token budget Parsimony used on that question.

| method | correct | 95% CI | context kept | prompt tokens | prefill | vs full |
|---|---|---|---|---|---|---|
| full context | 40/45 — 88.9% | 76.5–95.2 | 100% | 727 | 8.39 s | — |
| **Parsimony** | **36/45 — 80.0%** | 66.2–89.1 | **21.2%** | **219** | **2.38 s** | p = 0.125 |
| BM25 top sentences | 32/45 — 71.1% | 56.6–82.3 | 21.0% | 216 | 2.34 s | p = 0.039 |
| stopword removal | 29/45 — 64.4% | 49.8–76.8 | 63.7% | 509 | 5.62 s | p = 0.003 |
| truncate to budget | 14/45 — 31.1% | 19.5–45.7 | 19.7% | 197 | 2.07 s | p < 0.001 |
| random sentences | 6/45 — 13.3% | 6.3–26.2 | 20.8% | 227 | 2.38 s | p < 0.001 |
| no context | 1/45 — 2.2% | 0.4–11.6 | 0% | 61 | 0.46 s | p < 0.001 |

**A fifth of the context, 3.5× less prefill, and no significant difference from sending everything** (exact
McNemar on 4 discordant pairs). Every obvious method at the same budget *does* lose significantly. The claim
stops there: "not significantly worse" is not "as good", and 45 items cannot detect a difference below
roughly ten points.

**The failure modes separate cleanly by question kind.** Truncation answers **0 of 9** questions whose answer
sentence opens with a pronoun — that sentence is in the back half of a document truncation never reaches —
against **9 of 9** for Parsimony, which finds it by letting the sentence inherit the name from the sentence
before it and then keeps that sentence so "It" still refers to something. Plain BM25 retrieval fails hardest
on negation (4/9 against 7/9): a question and its negation share their content words, so ranking alone cannot
separate them. This is the same finding as §2, in a different module.

**What did not pay.** Removing the anchor guarantee, or the relevance floor, scored 37/45 — nominally above
the full method. Those are one-item differences and mean nothing at this sample size, so the honest reading is
that the method as a whole beats the baselines while its individual components are not separable on 45
questions. The floor does have a measured price: without it the prompt keeps 34.6% of the context instead of
21.2% for no accuracy difference we can see.

### It replicates on questions nothing was tuned against

Two failures above suggested obvious fixes — count a document's title toward its relevance, and judge the
relevance floor before the anchor bonus. Both are *post-hoc*: proposed after seeing these results, which is
the point at which a result stops being evidence. So they were measured on **test2**: 30 questions over six
collections authored beforehand and never looked at during tuning.

| method | correct | context kept | prefill | vs full |
|---|---|---|---|---|
| full context | 29/30 — 96.7% | 100% | 5.94 s | — |
| **Parsimony, as first frozen** | **27/30 — 90.0%** | **21.0%** | **1.70 s** | p = 0.500 |
| Parsimony + the two "fixes" | 26/30 — 86.7% | 22.6% | 1.81 s | p = 0.250 |
| stopword removal | 22/30 — 73.3% | 64.9% | 4.18 s | p = 0.016 |
| BM25 top sentences | 20/30 — 66.7% | 22.3% | 1.82 s | p = 0.004 |
| random sentences | 9/30 — 30.0% | 22.0% | 1.82 s | p < 0.001 |
| truncate to budget | 4/30 — 13.3% | 20.6% | 1.62 s | p < 0.001 |
| no context | 1/30 — 3.3% | 0% | 0.41 s | p < 0.001 |

**The method replicates. The improvement does not** — it scores one item worse and sends 5% more tokens, so
it was not adopted. One item is noise in both directions; that is precisely why a change that cannot be
shown to help does not ship. It stays available as `context_v2()` for re-measurement against a neural
encoder, where a title carries meaning word overlap cannot see.

The first table would have been the published one. The improvements were designed against its failures, and
re-measuring them there would have shown a gain by construction.

**Where it still fails.** Of the four questions Parsimony got wrong and full context got right, one asked for
a "daily dose" where the document says "once a day" — no lexical relevance score connects those, and that is
the ceiling this tier shares with the lexical encoder of ADR-028.

A second limitation is visible without any benchmark: ask an attached handbook something it says nothing
about ("what is the capital of Peru?") and the selector still keeps about 40% of it. Relevance is scored
relative to the best sentence in the request, so when nothing is relevant the least irrelevant sentences
still win places. The budget bounds the damage and the answer is unaffected — the model has no evidence
either way — but the tokens are wasted, and an absolute floor on the raw relevance score is the obvious
next measurement.

### Irrelevant context is not inert — and relative scoring cannot drop it

Found by using the system, not by running the benchmark: attach a staff handbook, ask *"what is the capital
of Peru?"*, and ~30% of it was sent anyway. Every score in the tier is relative to the best sentence present,
so with nothing relevant the *least irrelevant* sentences still win places. The benchmark could not see this
— every question in it is answerable from its own documents (ADR-042).

An absolute check now runs first: how much of the question's vocabulary appears in the context at all, and
the best sentence's cosine. On the tuning split the populations do not overlap — on-topic 0.70–1.00 coverage
and 0.55–0.85 cosine, off-topic 0.00–0.25 and 0.07–0.21. When it fires, one sentence is kept, because an
empty context reads as an instruction with a missing attachment.

Twenty off-topic questions were authored (8 for tuning, 12 reported), each answerable by the model alone and
absent from its documents:

| method | correct | context kept | prompt tokens | prefill |
|---|---|---|---|---|
| no context at all | **12/12** | 0% | 57 | 0.49 s |
| full context | **11/12** | 100% | 656 | 8.68 s |
| relative relevance only | — | 31.8% | — | — |
| **Parsimony** | **12/12** | **4.1%** | **90** | **0.92 s** |

**Sending everything cost 8.7 s and one answer.** Asked how many strings a violin has, with six irrelevant
documents attached, the model answered **six**. The closed-book arm scoring 12/12 is what makes that legible:
the model knew, and the context talked it out of knowing. Compression here is not a tax on quality — it is
what protects the answer.

The check fires on **0 of 75** answerable questions, and evidence recall on the on-topic splits is unchanged.

## 14. Keeping the last few turns answers none of them

M3 chooses which earlier turns survive and M1's context tier shortens the ones that do, and both were scored
only on tokens removed — a metric that rewards removing the answer. The conversation corpus cannot score
anything else: its assistant turns come from a mock provider, so "the answer changed" and "the answer got
worse" are the same event there (ADR-043).

`corpus/followups.jsonl` is the missing case: 20 conversations that state a fact in the first turn — a
server's memory, an allergy, a policy number — spend four exchanges elsewhere, and end with a question only
that turn can answer. 14 are held out and reported:

| how history is handled | correct | fact kept | prompt tokens | prefill |
|---|---|---|---|---|
| every turn, verbatim | 13/14 — 92.9% | 14/14 | 199 | 1.91 s |
| **relevance (MMR), as shipped** | **13/14 — 92.9%** | **14/14** | **165** | **1.55 s** |
| keep the last 4 turns | **0/14 — 0.0%** | **0/14** | 121 | 1.06 s |
| no history at all (control) | 1/14 — 7.1% | 0/14 | 56 | 0.29 s |

**Keeping the last few turns is what a token budget plus recency gives you, and it answers none of them** —
indistinguishable from sending no history at all (0/14 against 1/14). Relevance selection matches sending
every turn, exactly, on **17% fewer tokens and 19% less prefill**: the saving is free here, and the fact
survived all fourteen times.

The sentence-compression arm is identical to the row above it because the tier never fired: those turns are
30–45 tokens and its gates were set for attached documents. Ten further conversations put the fact inside a
**120–160 token answer**, which is what a real assistant turn looks like. Seven held out:

| how history is handled | correct | fact kept | prompt tokens | prefill |
|---|---|---|---|---|
| every turn, verbatim | 6/7 | 7/7 | 251 | 2.60 s |
| relevance (MMR) | 6/7 | 7/7 | 227 | 2.31 s |
| **relevance + sentence compression** | **6/7** | **7/7** | **159** | **1.52 s** |
| keep the last 4 turns | 0/7 | 0/7 | 107 | 0.99 s |

**30% fewer tokens than selection alone, 34% less prefill, no answers lost.** The answer sentence survived
all seven times while the rest of its turn went.

**The bug that corpus found.** On the first run the tier fired on one of the ten. M3's position-aware
arrangement moves the most relevant turn to the end of the list it hands on, and the tier protected "the last
two turns" — so it protected exactly the turn worth compressing. Two modules each correct alone, wrong in
composition: the project's own thesis arriving as a bug. Protection now means recent in the *conversation*,
by turn id.

With the gates set for turns rather than documents, M1's contribution to the factorial changes character
entirely: **+0.24 pp → +10.97 pp**, making it the largest single effect in the design. The full stack goes
from 33.3% to **39.6%** total reduction, and the additivity shortfall from 1.69 pp to **15.13 pp [11.15,
18.27]** — because M1 and M3 now compete for the same tokens. The thesis of this project, in one number.

## Reproducing all of it

```bash
python reproduce.py --out figures
```

Roughly 40 seconds: 17 cells, both passes, 11 CSVs and a full report. Generation memoisation avoids **78.2%**
of model calls and is bit-exact — ablation, effects and Pareto output are byte-identical with it disabled.
