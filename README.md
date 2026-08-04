# Parsimony

**Token-Efficient LLM Interaction on CPU-Only Hardware**
A stacked, self-improving optimisation layer for small language models.

VIT University · B.Tech BCSE497J Project I · Arrsh Tripathi (23BCI0191), Alok Singh (23BCI0158),
Dev Singh (23BCE0794) · Guide: Dr Sathya K

---

## Status — Sprint 0/1 complete, runs today

```bash
python -m parsimony.surfaces.cli.main demo
```

**156 tests, all passing, under a second.** Setup and commands: [`docs/08-setup.md`](docs/08-setup.md).

| Built | Deferred |
|---|---|
| L0 contracts, orchestrator, stage registry + DAG validation | **M3** history manager (Sprint 3) |
| **M1** compressor tiers 1–2 (tier 3 written, gated off until its golden test) | **M4** prefix-stable assembler (Sprint 4) |
| **M2** exact-hash cache with context chain and TTL | **M6b** model-tier escalation (Sprint 4) |
| **M5** output budgeter + streaming early-stop | Semantic cache tier — needs embeddings (Sprint 2) |
| **M6a** deterministic tier — exact arithmetic, units, dates | Ollama / real models (Sprint 2, ADR-007) |
| **M8** fidelity gate with `TransformKind` scoping | Quality metrics, ANOVA, Pareto (Sprint 6) |
| Ledger v1, JSONL + SQLite sinks, blob store | Dashboard, proxy, extension (Sprint 6) |
| Factorial sweep runner, corpus loader, 35-conversation corpus v0 | |

**What is real:** token counts (exact, Qwen2.5 vocabulary), every module's logic, the gate and its reverts,
the ablation harness, the ledger. **What is simulated:** model responses and all latency figures —
`MockProvider` synthesises TTFT/TPOT. Every row records `model_digest = "mock:v1"` so no simulated run can
later be mistaken for a real one.

### Measured on the corpus today

| Cell | total tokens | reduction |
|---|---|---|
| baseline | 5415 | — |
| M1 | 5378 | 0.7% |
| M2 | 5283 | 2.4% |
| M5 | 4781 | 11.7% |
| M1+M2+M5 | 4621 | **14.7%** |
| M1+M2+M5+M6 | 4407 | **18.6%** |

M5 dominates because it is the only module acting on output tokens — the ~82% side of the CPU cost the
report's Figure 2 is about. M1's solo contribution is small on this corpus because most queries are already
terse; that is an honest floor, not a bug, and it is why the corpus scales to 150 in Sprint 3.

**Gap 3, measured on day one:** moving the cache lookup from before to after the compressor — one entry in a
config list, no code change — takes cache hits from 2 to 4. That is ADR-002 earning its place.

## Documents

| Doc | Contents |
|---|---|
| [`docs/00-architecture.md`](docs/00-architecture.md) | Layering, core data model, orchestrator, stage ordering, repo layout, cross-cutting concerns |
| [`docs/01-pipeline-stages.md`](docs/01-pipeline-stages.md) | The eight processing stages, each with objective / inputs / outputs / techniques / libraries / pros / cons / alternatives / recommendation / integration |
| [`docs/02-module-specs.md`](docs/02-module-specs.md) | M1–M8 internals and ablation wiring |
| [`docs/03-decision-log.md`](docs/03-decision-log.md) | 23 ADRs with justification and consequences |
| [`docs/04-roadmap.md`](docs/04-roadmap.md) | Re-planned 12-week schedule, sprint plan, milestone gates, scope-cut order, risks |
| [`docs/05-evaluation-harness.md`](docs/05-evaluation-harness.md) | The compute budget problem and its fix; sweep runner; four quality measures; statistics; validity threats |
| [`docs/06-contracts.md`](docs/06-contracts.md) | Complete L0 type and protocol definitions + the ledger schema. **Review this first** |
| [`docs/07-corpus-spec.md`](docs/07-corpus-spec.md) | Authoring guide for the 150 conversations, 50 adversarial pairs and 40 gold answers. Actionable today, no code required |

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

## Next actions

See [`docs/04-roadmap.md`](docs/04-roadmap.md) §6.
