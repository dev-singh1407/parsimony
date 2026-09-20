# Parsimony

**Token-Efficient LLM Interaction on CPU-Only Hardware**
A stacked, self-improving optimisation layer for small language models.

VIT University · B.Tech BCSE497J Project I · Guide: Dr Sathya K

## Team

| Name | Reg. no | GitHub |
|---|---|---|
| Arrsh Tripathi | 23BCI0191 | [@Arrsh14](https://github.com/Arrsh14) |
| Alok Singh | 23BCI0158 | [@heyxalok](https://github.com/heyxalok) |
| Dev Singh | 23BCE0794 | [@dev-singh1407](https://github.com/dev-singh1407) |

---

## Status — all eight modules built, full pipeline runs end to end

```bash
python reproduce.py --out figures
```

**1,110 tests passing.** Every table below regenerates from a live run in ~40 s. Setup and commands:
[`docs/08-setup.md`](docs/08-setup.md).

| Module | State |
|---|---|
| **M1** compressor — tiers 1–3, negative-yield detection, windowed re-tokenisation | built, golden-tested |
| **M2** two-tier cache — exact + semantic, three-zone verifier | built |
| **M3** history manager — 4 strategies, separate arrangement stage | built |
| **M4** prefix-stable assembler + token-level prefix-survival instrument | built |
| **M5** output budgeter + streaming early stop | built |
| **M6** router — deterministic tier (0 model tokens) + escalation | built |
| **M7** policy learner — counterfactual replay, PolicyBundle, warm start | built |
| **M8** fidelity gate — `TransformKind`-scoped, always on | built |

Plus: ledger v1 with dual sinks, generation memoisation, factorial sweep runner, four quality measures,
bootstrap/effect-size/Pareto statistics, threshold calibration, a cross-vocabulary generalisation study, and
`reproduce.py`.

**Surfaces.** A terminal view that draws each layer as it runs (`parsimony chat`, `ask`), a local web
visualiser you can scrub through stage by stage (`parsimony web`), LaTeX and proof-document export
(`parsimony export`), and the studies as commands: `longctx`, `followups`, `longbench`, `compare`,
`calibrate`.

**Still deferred:** the OpenAI-compatible proxy and the browser extension.

**Runs against a real model.** `qwen2.5:1.5b-instruct` (Q4_K_M) via Ollama, CPU-only, offline — the same
model whose vocabulary the token counts already used, so attaching it invalidated nothing. `--provider
ollama` selects it; `--provider mock` keeps the deterministic stand-in for fast, reproducible sweeps. A run
that asks for the real model and cannot reach it **refuses** rather than falling back, because a run that
believes it measured a real model and actually measured a fake one is the worst failure available here.
Every ledger row carries the provider's content digest, so the two can never be confused after the fact.

### Headline results (151 conversations, 263 requests, 17 cells)

| effect | estimate | partial η² |
|---|---|---|
| M1 compressor | +10.97 pp | 0.365 |
| M5 output budgeter | +10.80 pp | 0.354 |
| M3 history manager | +6.78 pp | 0.140 |
| M1×M3 interaction | −4.99 pp | 0.076 |
| M2 semantic cache | +1.96 pp | 0.012 |

Full stack reaches **+39.6%** total token reduction, −47.0% on the input side alone. **The largest
interaction is M1×M3 at −4.99 pp**, and it is there because both are paid out of the same tokens: M3 drops
earlier turns, M1's context tier shortens the ones that survive. Whichever runs first collects the saving.
Every pairwise interaction that matters is negative — the modules eat each other's lunch rather than
compounding.

**The additivity shortfall depends on the configuration, and that is the sharper result.** Improving the
encoder (ADR-035) made the cache hit more often, which made it overlap its neighbours *less*:

| configuration | M1 effect | largest interaction | additivity shortfall |
|---|---|---|---|
| `hashing-v1`, M1 on the question only | +0.22 pp | M3×M5, −1.14 then | 2.53 pp, **[+0.93, +3.99]** |
| `content-v1`, M1 on the question only | +0.24 pp | M3×M5, −0.75 then | 1.69 pp, **[+0.04, +3.21]** |
| `content-v1`, M1 on history too (shipped) | +10.97 pp | **M1×M3 −4.99** | **15.13 pp, [+11.15, +18.27]** |
| `all-minilm`, M1 on history too | +11.75 pp | M1×M3, −4.40 there | 14.81 pp, [+10.97, +18.08] |

A positive third-order term, **M1×M3×M5 +0.45 pp**, sits under those: when three modules compete for the
same tokens the pairwise overlaps double-count, and the triple corrects for it.

So savings do not compound — but *by how much they fail to compound* is a property of the components, not a
constant of the technique stack. Under the better encoder the shortfall is smaller, and its interval clears zero by
0.02 pp — close enough to the boundary that no weight should be put on which side
of it the bound falls. The weaker encoder was not reinstated to protect the interval: picking a component known to be
worse because it yields a more publishable number is the failure mode this project is written against.

### On a real model, prefill is 92–99% of the time

Measured against `qwen2.5:1.5b-instruct` on a Ryzen 7 CPU, reading Ollama's own prefill/decode split rather
than wall-clock TTFT:

| input tokens | prefill | decode | prefill share |
|---|---|---|---|
| 146 | 1,360 ms | 122 ms | 91.7% |
| 474 | 3,902 ms | 165 ms | 95.9% |
| 1,338 | 11,609 ms | 136 ms | 98.8% |

Prefill is linear at **~8.5 ms per input token**. That converts every token result in this project into wall
clock: the full stack's 11,836 saved input tokens are **100.6 seconds of prefill across the corpus, 383 ms
per request**. The old `MockProvider` assumed 120 ms TTFT — understating the prompt side by an order of
magnitude, in the direction that mattered (ADR-034).

And it costs nothing in accuracy. On the 40 gold items the real model scores **92.5% at baseline and 97.5%
with the full stack**, and **not one answer the baseline got right was lost**. The two gains are arithmetic
routed to M6's deterministic tier; two discordant pairs is not significant (exact McNemar p = 0.50), so the
claim is *no measurable degradation* rather than improvement. The old 5% gold column was the mock being
unable to answer at all.

**Escalating to a bigger model buys nothing here (ADR-037).** `llama3.2:3b` scores 36/40 against
`qwen2.5:1.5b`'s 36/40 — item for item identical, zero questions where the larger model succeeded and the
smaller failed — for 16% more wall clock and twice the memory. M6's escalation threshold was also set at
0.75 against an observed maximum complexity of **0.406**, so the tier could not fire at all. It is now
calibrated to 0.20 and deliberately left off.

### On long context, the compressor removes four fifths of the prompt

The conversation corpus asks six-word questions, and the compressor saved 0.23% of its tokens because there
was nothing in them to remove. Real requests carry *context* — retrieved passages, an attached report, an
earlier long answer — and that is where prefill goes. Requests now carry documents, and M1's context tier
keeps only the sentences the current question needs, checked by the fidelity gate: every kept sentence
verbatim and in order, and anything the question names still present (ADR-040).

Measured on 45 held-out questions over six fictional documents each, `qwen2.5:1.5b-instruct` on this laptop.
Every baseline gets the token budget Parsimony used on that question:

| method | correct | context kept | prompt tokens | prefill | vs full context |
|---|---|---|---|---|---|
| full context | 41/45 — 91.1% | 100% | 618 | 11.05 s | — |
| **Parsimony** | **40/45 — 88.9%** | **26.1%** | **161** | **3.45 s** | **p = 1.000** |
| Parsimony, lexical encoder | 38/45 — 84.4% | 22.1% | 137 | 1.91 s | p = 0.250 |
| BM25 top sentences | 34/45 — 75.6% | 25.8% | 159 | 3.43 s | p = 0.065 |
| stopword removal | 31/45 — 68.9% | 63.7% | 393 | 7.67 s | p = 0.013 |
| truncate to budget | 16/45 — 35.6% | 24.5% | 151 | 3.10 s | p < 0.001 |
| random sentences | 9/45 — 20.0% | 25.5% | 157 | 3.59 s | p < 0.001 |
| no context at all | 0/45 — 0.0% | 0% | 61 | 0.63 s | p < 0.001 |

A quarter of the context, 3.2× less prefill, and **one item between it and sending everything** — p = 1.000,
which is as close to "no difference" as a paired test on 45 items can report. Every obvious method at the
same budget loses significantly. The closed-book row is the control that makes the rest meaningful: these
documents are fictional, so nothing here can be answered from memory, and 0/45 is what that should look like.

The lexical row is there because the default arm is not a fixed thing: `best_config` upgrades to a neural
encoder wherever one is reachable, so on a machine with an embedding model served, "Parsimony" *is* the
neural arm. Running both under one name once produced two rows that differed by a single item and by
nothing else — the same configuration, twice, with the difference coming from the per-prompt nonce
(ADR-046). The encoder each arm ran with is now recorded on every row.


The difference is sharpest where the answer sentence begins with a pronoun. Truncation answers **0 of 9**
such questions and Parsimony **9 of 9**, because a sentence inherits the name from the sentence before it,
and that sentence is then kept so "It" still refers to something.

It replicates: on 30 further questions authored afterwards and never tuned against, Parsimony scores
**26/30** where sending the whole document scores 28/30, BM25 top-k scores 19/30 and truncation 7/30. Two changes made after reading the first run's failures were measured there and **not adopted** — they
scored one item worse and sent 5% more tokens, which is what a held-out split is for.

**Irrelevant context is not inert.** Ask an attached handbook something it says nothing about and the
right amount to send is almost none. Sending it all costs 8.7 s of prefill — and an answer: asked how many
strings a violin has with six irrelevant documents attached, the model said *six*. With no context at all it
is right every time (12/12). Parsimony keeps 4.1% of the documents, takes 0.92 s, and is also right every
time (ADR-042).

### "Why not just keep the last few turns?" — because it answers none of them

Every chat framework keeps the most recent turns until a token budget is full. On 14 held-out conversations
that state a fact first and ask about it last, that default answers **0 of 14**, which is worse than sending
no history at all. Relevance selection answers **13 of 14** — exactly what sending every turn achieves — on
17% fewer tokens and 19% less prefill (ADR-043).

| how history is handled | correct | fact kept | prompt tokens | prefill |
|---|---|---|---|---|
| every turn, verbatim | 13/14 | 14/14 | 199 | 1.91 s |
| **relevance (MMR), as shipped** | **13/14** | **14/14** | **165** | **1.55 s** |
| keep the last 4 turns | **0/14** | **0/14** | 121 | 1.06 s |
| no history at all (control) | 1/14 | 0/14 | 56 | 0.29 s |

Where the fact sits inside a *long* earlier answer, compressing that answer to its relevant sentences sends
**30% fewer tokens again (227 → 159) with nothing lost** — 6/7 either way, the answer sentence surviving
every time.

### Watch it happen

```bash
parsimony ask --file examples/staff-handbook.md     # attach a file and ask about it
parsimony chat "What is the travel budget for Tallinn?" -f examples/staff-handbook.md --compare
parsimony longctx --show kestrel_q2                 # one question: every sentence kept or removed
```

Each turn is drawn as it runs: every layer with its real duration and what it changed, the prompt bar
shrinking as cuts are committed, a clock while the model reads the prompt, the answer streaming in against
the budget the answer limiter set, and then the prompt the model actually received with every removed span
struck through. `--compare` asks the same question again with every layer switched off, from cold, and puts
the two measurements side by side — on one attached handbook that is 8.9 s of reading against 3.0 s, for the
same answer.

### See it in a browser, and take the evidence away with you

```bash
parsimony web                                       # a local page: heatmap, A/B, demo counters
parsimony export --latex                            # booktabs tables + a Beamer deck from figures/*.csv
parsimony export --proof kestrel_q2                 # a PDF showing every removal, labelled
```

`parsimony web` serves one page from the standard library — no framework, no install, nothing leaves the
machine. It shades every sentence of a document by the score the encoder gave it and names the decision
behind each one on hover; it runs the same question with the layers off and on against the real model and
plots both as they generate (one after the other, because two generations on one CPU measure contention,
not compression); and it keeps a running count of tokens pruned and seconds saved, labelled *estimated*
unless this machine timed the rate itself.

`parsimony export` writes the results out without retyping them: the tables read `figures/*.csv`, so a
number in the report cannot drift from the run that produced it, and the proof document reproduces the
compressed prompt with every removed sentence struck through and tagged with the branch that removed it.

### A better encoder broke the safety design — and that is the finding

Replacing the lexical encoder with MiniLM (45 MB, served by the same local runtime, no PyTorch) was the
roadmap's first item. It fixes what the lexical encoder cannot see — and it takes the cache's false-answer
rate from **0.0% to 17.8%** (ADR-041):

| encoder | design | false answers | true hits | ms per question |
|---|---|---|---|---|
| content-v1 (lexical) | accept zone above τ_hi | 0/45 — 0.0% | 13/45 — 28.9% | 1 |
| all-minilm (neural) | accept zone above τ_hi | **8/45 — 17.8%** | 23/45 — 51.1% | 51 |
| all-minilm (neural) | **verify every hit** | **0/45 — 0.0%** | **17/45 — 37.8%** | 51 |

The adversarial negation pair scores 0.924 under the lexical encoder and **0.996** under MiniLM — above the
threshold at which the old design skipped verification entirely. No threshold fixes that: a negation is a
smaller edit than a rephrasing in any embedding space, so a better space makes it worse. The accept zone was
safe only because the encoder was weak. Verification now runs on every candidate, costs microseconds, holds
the false-answer rate at 0.0% for both encoders, and lifts answer reuse from 26.7% to 37.8%.

In M1's context selector the same encoder recovers answers a lexical score loses: **40/45 on the
long-context test set against 41/45 for full context** — one item, p = 1.000 — at 26.1% of the tokens and
3.45 s of prefill against 11.05 s. The lexical selector scores 38/45 at 22.1%, so the encoder buys two
items for 4% more context.

**`localhost` was also costing two seconds per call.** A flat ~2,040 ms per Ollama request that `curl` did
not pay: `localhost` resolves to `::1` first, Ollama listens on IPv4, and the attempt has to time out. At
`127.0.0.1` the same call takes **31 ms**. It is invisible in every prefill and decode figure here — those
counters start after the connection — but every wall-clock number measured before the fix carries it, on
both sides of every comparison.

### Five findings that changed the design

**The published cache thresholds are unsafe here (ADR-024, ADR-027).** The adversarial negation pair sits at
cosine 0.924 — *higher than every genuine paraphrase*. The literature's "safe" 0.85–0.92 would auto-accept it
and serve the opposite answer. Measurement drove the verifier from a 26.7% false-hit rate to **0.0%**, and
the fix was three checks nothing in the caching literature performs: operative modifiers (min/max),
morphological and lexical negation, and alphanumeric identifiers.

**Position-aware placement is worth ~0 tokens and ~18 seconds (ADR-025, ADR-034).** Moving one volatile token
to the head of a prompt — a "turn 3 of 7" preamble — changes the token count by 0.5% and the steady-state
prefill cost from **212 ms to 18,914 ms**. Ollama reuses the KV cache across requests, so a stable prefix
gets **98.5%** reuse and a volatile head gets **0.5%**. Every metric in the compression literature scores
those two configurations identically.

**Negative yield is real but not where the report claims (ADR-026).** Across 495 word deletions and every
lexicon substitution, none raised the token count — modern BPE encodes the leading space, so whitespace-
aligned edits are monotone. Sub-token edits *do* raise it ("running" → "runing" is 2 tokens → 3). The guard
earns its place by rejecting **zero-yield** edits, which perturb text for no saving at all.

**A calibration transfers as a ratio, not as a mechanism (ADR-032).** Run the whole sweep against a second
real vocabulary — GPT-2's 50,257 against Qwen2.5's 151,665, thresholds carried over unchanged — and the
reduction percentages land within 0.1 pp and the module ranking is identical, because a ratio cancels a
roughly constant vocabulary factor. The *explanations* fare worse: ADR-030 attributed negative yield to two
BPE position-0 effects, and only one survives. `"explain"` costs 1 token against `"Explain"`'s 2 under Qwen,
but GPT-2 charges 2 for both. Half of that ADR was a Qwen fact wearing a general claim's clothes, and this is
the study that undressed it.

**"Self-improving" is a property of the traffic, not of the module (ADR-033).** M7 mines a policy bundle from
past conversations; measured properly — mine from one half of the conversations, test on a disjoint half —
it delivers **+0.00 pp at 0% traffic recurrence and +17.83 pp at 57%**, with zero extra fidelity-gate fires
at every level. The ablation corpus sits at **1.9% recurrence** because it was authored for ablation
diversity, which is why M7 shows nothing in the headline table. That is a fact about the corpus, not the
module — the same distinction as ADR-028.

## Documents

| Doc | Contents |
|---|---|
| [`docs/00-architecture.md`](docs/00-architecture.md) | Layering, core data model, orchestrator, stage ordering, repo layout, cross-cutting concerns |
| [`docs/01-pipeline-stages.md`](docs/01-pipeline-stages.md) | The eight processing stages, each with objective / inputs / outputs / techniques / libraries / pros / cons / alternatives / recommendation / integration |
| [`docs/02-module-specs.md`](docs/02-module-specs.md) | M1–M8 internals and ablation wiring |
| [`docs/03-decision-log.md`](docs/03-decision-log.md) | 47 ADRs with justification and consequences. **The intellectual core** — several record where measurement contradicted the plan |
| [`docs/04-roadmap.md`](docs/04-roadmap.md) | Re-planned 12-week schedule, sprint plan, milestone gates, scope-cut order, risks |
| [`docs/05-evaluation-harness.md`](docs/05-evaluation-harness.md) | The compute budget problem and its fix; sweep runner; four quality measures; statistics; validity threats |
| [`docs/06-contracts.md`](docs/06-contracts.md) | Complete L0 type and protocol definitions + the ledger schema. **Review this first** |
| [`docs/07-corpus-spec.md`](docs/07-corpus-spec.md) | Authoring guide for the 150 conversations, 50 adversarial pairs and 40 gold answers. Actionable today, no code required |
| [`docs/08-setup.md`](docs/08-setup.md) | Environment, install, and how to run each command |
| [`docs/10-literature-survey.md`](docs/10-literature-survey.md) | **Literature survey.** 54 papers across eight strands, the six research gaps they leave open, and what this project does differently — with the measurement backing each claim |
| [`docs/11-demo-runbook.md`](docs/11-demo-runbook.md) | **Demo runbook.** A five-act, twelve-minute walkthrough with the exact commands, their measured run times, what to say at each, and the failure modes that actually happen |
| [`docs/12-demo-questions.md`](docs/12-demo-questions.md) | **Demo questions.** One per tier plus five that fire several at once, each with the modules it actually triggers measured rather than assumed |
| [`docs/13-review2-dossier.md`](docs/13-review2-dossier.md) | **Review-2 dossier.** The 40-paper limitations table, the six gaps it exposes, research questions, four contributions, the pipeline tier by tier with a figure, and every result. Renders to PDF |
| [`docs/14-project-report.md`](docs/14-project-report.md) | **Project report.** The full report in the school's section order — literature review, six gaps, RQs, contributions, method, architecture, results — with a plain-language layer over every finding. Renders to PDF |
| [`docs/09-findings.md`](docs/09-findings.md) | **Read this one first.** Every result in plain prose, with the numbers re-derived from live runs |

## The one-paragraph version

Parsimony is a middleware layer between an application and a locally hosted small language model. Seven
optimisation modules (compressor, semantic cache, history manager, prefix-stable assembler, output budgeter,
escalation router, conversation-mined policy learner) plus an always-on fidelity gate. The deliverable is not
a headline percentage but a **calibrated operating curve**: for a given model, quantisation and query class,
which modules should be on and at what setting.

Architecturally this means the system is **a measurement instrument that happens to be usable as
middleware** — which is why modules propose rather than act, why stage order is configuration rather than
code, and why the ledger schema is treated as part of the architecture.

## Four load-bearing properties

1. **Every module independently switchable** — the headline result is a 2⁴ factorial ablation.
2. **Stage order is data, not code** — Gap 3 (compression × cache interaction) is unanswerable otherwise.
3. **Every decision auditable to a ledger row** — retrofitted instrumentation is always wrong.
4. **Middleware overhead under 120 ms** — a stack costing more than it saves is a null result.

## Licence

Code: **MIT** — see [LICENSE](LICENSE). Corpus: **CC BY 4.0** — see [corpus/LICENSE](corpus/LICENSE).

Every dependency is open source and every model weight is openly licensed; the project has no paid component
of any kind. Authorship and citation details are in [AUTHORS.md](AUTHORS.md).

## Next actions

See [`docs/04-roadmap.md`](docs/04-roadmap.md) §6.
