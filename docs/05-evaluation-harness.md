# Parsimony — Evaluation Harness (L5)

The pipeline is the apparatus; this layer is the experiment. It is specified to the same depth as the
pipeline because **the results are the deliverable** — a beautifully engineered stack that cannot complete
its sweep produces no contributions at all.

---

## 1. The compute problem, stated plainly

The report's §4.6 protocol is: full factorial over M1/M2/M3/M5 (16 cells), each run over the whole corpus,
five seeds, then generalisation across three models and two quantisation levels. On an i5-class CPU with no
GPU, that does not fit in the runway. The arithmetic, with assumptions stated:

| Quantity | Assumption | Value |
|---|---|---|
| Conversations | corpus spec | 150 |
| Requests per conversation | avg turns | ~4 |
| Requests per full pass | | **600** |
| Decode throughput, 1B Q4_K_M on i5 | conservative | 10–20 tok/s |
| Avg output tokens, unoptimised | | ~150 |
| Decode time per request | 150 / 15 | ~10 s |
| Prefill + middleware | | ~1 s |
| **Time per pass** | 600 × 11 s | **~1.8 h** |

Then:

- 16 cells × 5 repeats = 80 passes → **~147 h ≈ 6 days of continuous CPU, per model**
- × 3 models → ~18 days
- × 2 quantisation levels → **~36 days of unbroken compute**

on a shared laptop that also has to run development, thermally throttles, and is needed for coursework. The
schedule allows roughly 10 days of sweep time. **The protocol is over budget by 3–4×, and that is before a
single failed run.**

This is not a reason to weaken the experiment. It is a reason to design the harness around four
optimisations, none of which cost any scientific validity.

### Fix 1 — Generation memoisation (the big one)

At temperature 0, decoding is greedy: **identical prompt bytes + identical model digest + identical
generation parameters ⇒ identical output**. Across 16 cells an enormous fraction of prompts repeat — every
pair of cells differing only in M2's enable flag produces byte-identical prompts on every cache miss, and
cells differing only in M5 produce identical prompts with different `num_predict`.

```python
class GenerationMemo:
    """EXPERIMENT mode only. Key: (prompt_sha256, model_digest, gen_params_hash)."""
    def get(self, key) -> GeneratedText | None: ...
    def put(self, key, text: GeneratedText) -> None: ...
```

**This cannot bias any result**, because it returns exactly what the model would have returned. It is a pure
compute optimisation, and it is architecturally free — the content-addressed blob store already exists
(`01-pipeline-stages.md` Stage 3).

**Critical constraint: the memo destroys latency measurement.** A memo hit takes 50 µs, not 10 s. So the
sweep splits into two passes with different purposes:

| Pass | Memo | Corpus | Repeats | Produces |
|---|---|---|---|---|
| **Quality pass** | ON | full 150 | 1 | token counts, quality scores, cache/gate/router behaviour |
| **Timing pass** | OFF | 50-conversation stratified subset | 2 | TTFT, TPOT, wall-clock, prefix survival, energy |

Token counts, quality and every behavioural metric are deterministic functions of the input, so one repeat
is *mathematically sufficient* for them — running five is not more rigorous, it is five identical numbers.
Latency is the only genuinely stochastic quantity, and it gets a dedicated, honest, unmemoised pass. Every
figure states which pass it came from.

### Fix 2 — Repeats measure latency, so derive CIs from within-pass samples

Per ADR-017, seeds at temperature 0 do not change output. A single timing pass over 200 requests yields 200
latency observations; bootstrap confidence intervals from those. The second repeat exists to confirm the
distribution is stable across runs (thermal drift, background load), not to build the interval. **5 repeats →
2.**

### Fix 3 — Generalisation applies to the winner, not to all 16 cells

The report already says this (§4.6: "re-running *the winning configuration* on all three models and both
quantisation levels"). Making it explicit in the harness prevents the combinatorial blow-up: 3 models × 2
quant applies to ~2 configurations (winner + baseline), not 16.

### Fix 4 — Stratified subset for the factorial, full corpus for the winner

The 16-cell factorial establishes *interaction structure*; that does not need 150 conversations. Run the
factorial on a 50-conversation stratified subset, then re-run the winning cell and the baseline on the full
150. Report both, and report the subset/full agreement as a validity check.

### Revised budget

| Stage | Estimate |
|---|---|
| Quality pass, 16 cells, memoised, full corpus | ~4–6 h (dominated by first-touch misses) |
| Timing pass, 16 cells × 2, 50-conv subset | ~12–16 h |
| Focused studies (M1×M2 offline re-analysis, M1×M4 sweep) | ~6 h |
| Generalisation: 2 configs × 3 models × 2 quant, subset | ~10 h |
| **Total** | **~35–40 h**, comfortably inside two overnight windows plus slack |

**Action for Sprint 2:** replace every assumption in the table above with a measured number from
`OllamaProvider`, and re-plan then. The estimates here are order-of-magnitude; the *structure* of the fix is
what matters and it holds regardless of the exact throughput.

---

## 2. Harness architecture

```
corpus/ (frozen, hashed)
     │
     ▼
CellEnumerator ──▶ list[ParsimonyConfig]         # each cell IS a config_hash (ADR-008)
     │
     ▼
SweepRunner ──▶ WorkQueue ──▶ N worker processes
     │                              │
     │                              ├─ each owns one JsonlSink file (ADR-005)
     │                              ├─ each writes a completion marker per (cell, seed)
     │                              └─ shared read-mostly GenerationMemo (SQLite, WAL, mostly reads)
     ▼
parsimony ledger import ──▶ analysis DB (DuckDB)
     │
     ├─▶ MetricSuite      (4 quality measures, never averaged)
     ├─▶ StatsSuite       (bootstrap CIs, 2-way ANOVA, partial η²)
     ├─▶ ParetoAnalysis   (frontier + knee)
     └─▶ reproduce.py     (every figure, from raw logs, one command)
```

**Resumability is a requirement, not a nicety.** A 16-hour unattended run *will* be interrupted — sleep,
update, thermal shutdown, power. Each `(cell, seed, conversation)` triple writes a completion marker on
finish; `SweepRunner` skips completed triples on restart. Without this, one interruption at hour 14 costs
14 hours. With it, it costs one conversation.

**Worker count.** Not `os.cpu_count()`. Ollama itself is multi-threaded and saturates the CPU on a single
request; running 4 workers against one Ollama instance produces 4× the contention and roughly 1× the
throughput, while destroying the timing measurements. **Default `n_workers = 1` for the timing pass** (it is
the only honest setting) and `n_workers = cpu_count // 2` for the memoised quality pass, where most requests
never touch the model at all.

---

## 3. Quality measurement

Four measures, **never averaged into a single score** — the report is explicit and it is right: averaging a
proxy with a ground truth manufactures false confidence.

### 3.1 Embedding similarity (proxy)
Cosine between the candidate response embedding and the *baseline* response embedding, same encoder as the
cache. Cheap, insensitive to phrasing. Weakness: high similarity between a correct and a confidently wrong
answer that shares vocabulary.

### 3.2 Token overlap (proxy)
ROUGE-L F1 against the baseline response. Weakness: penalises legitimate concision — which the output
budgeter produces *by design*, so this metric is structurally biased against M5. **State that bias
explicitly** and read M5's row on this metric with it in mind; that is more honest than dropping the metric.

### 3.3 Local model-as-judge (proxy)
The most fragile measure; specify it tightly or it produces noise dressed as data.

- **Judge model must not be a model under test.** Using Llama 1B to judge Llama 1B's output measures
  self-preference. Use the 3B escalation model, or a separate judge, and pin its digest.
- **Pairwise, not absolute.** Small models cannot produce calibrated 1–10 scores; they can pick between two
  answers.
- **Position-swap every comparison.** LLM judges have a documented position bias. Run A/B and B/A; a
  disagreement counts as a tie. Report the swap-disagreement rate — a high rate means the judge is noise and
  the metric should be discounted, and knowing that is worth more than the score.
- **Fixed prompt, version-controlled**, its hash recorded in the ledger.

### 3.4 Exact match on the gold subset (ground truth)
40 questions with human-written gold answers. Normalised exact match plus a documented numeric-tolerance
rule for arithmetic. **This is the only measure in the entire evaluation that is not a proxy** — anchor
every quality claim to it, and when the proxies and the gold subset disagree, the gold subset wins.

### 3.5 Human check
100 sampled outputs, blind-labelled by the team, per the report's risk register. Assign before the sweep
runs so it does not get skipped at the end. Report inter-rater agreement across the three of you — three
raters make this nearly free and it substantially strengthens the claim.

---

## 4. Statistics

**Bootstrap CIs, not t-intervals.** Latency distributions on CPU are right-skewed with a long tail (thermal
throttling, scheduler). A symmetric t-interval on a skewed distribution understates the upper tail. 10 000
bootstrap resamples, percentile method. Report medians and IQR alongside means.

**Two-way ANOVA on interaction terms**, per the report. One caveat to build in from the start: with 600
observations per cell, *everything* is statistically significant. p-values will be uninformative.

> **Report partial η² (effect size) as the headline, with p as a footnote.** "M1×M2 interaction, partial
> η² = 0.03" says the interaction is real but small. "p < 0.001" says only that you had a lot of data. The
> project's core claim is about the *size* of the shortfall from additivity, so effect size is the natural
> unit and it is what makes Contribution 1 quantitative rather than directional.

**Additivity shortfall**, the number Figure 1 is about:

```
shortfall = (Σ individual_reductions) − measured_stacked_reduction
```

Report it with a CI. This is the primary result of the whole project. It deserves its own table, not a
sentence.

**Pareto frontier.** Each of the 16 cells is a point in (total token reduction, quality retained). Compute
the non-dominated set and the knee (max distance from the line joining the extremes). Deliverable is the
frontier and its knee, not any single configuration — as the report states.

**Libraries.** `scipy.stats`, `statsmodels` (ANOVA with proper type-II/III sums of squares — do not
hand-roll it), `numpy`, `pandas`/`duckdb`, `matplotlib`.

---

## 5. `reproduce.py`

One command regenerates every figure and table in the report from raw logs:

```bash
python reproduce.py --ledger runs/2026-10-15/ --out figures/
```

**Build it in Sprint 0 with one figure, not in Sprint 6 with twenty.** A reproduction script written at the
end is a script that was never tested; a script grown alongside the results is a script that always worked.
It also catches ledger schema drift immediately — if a Sprint 3 change breaks a Sprint 1 figure, you find out
that week rather than in October (ADR-014).

Each figure function declares the ledger fields it consumes. A schema change that removes a consumed field
fails `reproduce.py` in CI, which is exactly the tripwire ADR-014 needs.

---

## 6. Validity threats and how the harness answers them

| Threat | Answer built into the harness |
|---|---|
| Self-authored corpus tuned to the system | Corpus frozen + SHA-256 committed before the first sweep; hash in every ledger row; gold answers written before any system output is inspected (ADR-015) |
| Proxy metrics flatter the system | Four measures reported separately; gold subset is the anchor; the ROUGE bias against M5 stated up front |
| Judge model is unreliable | Pairwise + position-swap; swap-disagreement rate reported; judge never a model under test |
| Memoisation contaminates results | Memo is bit-exact by construction (temp 0, pinned digest); latency measured only on unmemoised passes; each figure labels its pass |
| Approximate index contaminates the false-hit rate | Exact search (ADR-004) |
| Config drift mid-sweep | `config_hash` in every row; a cell with two distinct hashes fails analysis loudly |
| Model changed mid-project | `model_digest` pinned and recorded; a digest change is visible in the ledger |
| Subset ≠ full corpus | Winner and baseline re-run on the full 150; agreement reported as a validity check |
