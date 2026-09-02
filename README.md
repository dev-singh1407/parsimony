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

**500 tests passing.** Every table below regenerates from a live run in ~40 s. Setup and commands:
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

**Deferred by request:** Ollama and real model hosting; the dashboard, OpenAI-compatible proxy and browser
extension.

**What is real:** exact token counts under two real vocabularies (Qwen2.5 and GPT-2), every module's logic, the gate and its
reverts, the cache verifier, the ablation harness and all statistics. **What is simulated:** model responses
and every latency figure — `MockProvider` synthesises TTFT/TPOT. Every ledger row records
`model_digest = "mock:v1"`, so no simulated run can later be mistaken for a real one.

### Headline results (151 conversations, 263 requests, 17 cells)

| effect | estimate | partial η² |
|---|---|---|
| M5 output budgeter | +13.01 pp | 0.546 |
| M3 history manager | +11.69 pp | 0.441 |
| M2 semantic cache | +1.61 pp | 0.008 |
| M1 compressor | +0.14 pp | 0.000 |

Full stack reaches **+33.5%** total token reduction. **Every interaction term that is non-zero is negative**
(M3×M5 −1.14, M2×M5 −0.10, M1×M5 −0.02; the remaining 8 are zero to six decimal places) — where the modules
interact at all, they eat each other's lunch rather than compounding. The additivity shortfall is
**2.53 pp, 95% CI [+0.93, +3.99]**, which excludes zero: that is Contribution 1, measured rather than assumed.

### Four findings that changed the design

**The published cache thresholds are unsafe here (ADR-024, ADR-027).** The adversarial negation pair sits at
cosine 0.924 — *higher than every genuine paraphrase*. The literature's "safe" 0.85–0.92 would auto-accept it
and serve the opposite answer. Measurement drove the verifier from a 26.7% false-hit rate to **0.0%**, and
the fix was three checks nothing in the caching literature performs: operative modifiers (min/max),
morphological and lexical negation, and alphanumeric identifiers.

**Position-aware placement halves KV prefix reuse for zero token difference (ADR-025).** 94.4 → 47.6 prefix
tokens reused, with the same turns and the same token count. Every metric in the compression literature
scores those two configurations identically.

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

## Documents

| Doc | Contents |
|---|---|
| [`docs/00-architecture.md`](docs/00-architecture.md) | Layering, core data model, orchestrator, stage ordering, repo layout, cross-cutting concerns |
| [`docs/01-pipeline-stages.md`](docs/01-pipeline-stages.md) | The eight processing stages, each with objective / inputs / outputs / techniques / libraries / pros / cons / alternatives / recommendation / integration |
| [`docs/02-module-specs.md`](docs/02-module-specs.md) | M1–M8 internals and ablation wiring |
| [`docs/03-decision-log.md`](docs/03-decision-log.md) | 32 ADRs with justification and consequences. **The intellectual core** — several record where measurement contradicted the plan |
| [`docs/04-roadmap.md`](docs/04-roadmap.md) | Re-planned 12-week schedule, sprint plan, milestone gates, scope-cut order, risks |
| [`docs/05-evaluation-harness.md`](docs/05-evaluation-harness.md) | The compute budget problem and its fix; sweep runner; four quality measures; statistics; validity threats |
| [`docs/06-contracts.md`](docs/06-contracts.md) | Complete L0 type and protocol definitions + the ledger schema. **Review this first** |
| [`docs/07-corpus-spec.md`](docs/07-corpus-spec.md) | Authoring guide for the 150 conversations, 50 adversarial pairs and 40 gold answers. Actionable today, no code required |
| [`docs/08-setup.md`](docs/08-setup.md) | Environment, install, and how to run each command |
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
