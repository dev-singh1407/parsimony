# Parsimony — Setup and Usage

## 1. Requirements

Python 3.10+. That is the whole list. **No GPU, no Ollama, no model downloads, no
API key, nothing running in the background.** Total install is about 50 MB.

## 2. The environment lives outside OneDrive — deliberately

A virtualenv is roughly 3,000 small files. Created inside `Desktop\final year\`,
OneDrive tries to sync every one of them: sustained CPU, disk and upload for zero
benefit, and it is the most likely way this project would slow the machine down.
The same will apply to model files in Sprint 2 (~6 GB of GGUF weights).

**Rule: source and docs in OneDrive (backed up, small). Environments, run
artefacts and model weights outside it (large, regenerable).** `.gitignore`
already excludes `runs/`, `blobs/`, `*.db`, `models/` and `.venv/` as a second
line of defence.

### Footprint today

| Location | Size | Synced by OneDrive? |
|---|---|---|
| `final year/parsimony/` source, docs, corpus, tests | ~1 MB | yes — intended |
| `final year/parsimony/runs/` ledgers | ~3 MB per full sweep | **yes — watch this** |
| `~/.venvs/parsimony` | ~135 MB | no |
| `~/.cache/huggingface` (Qwen tokenizer) | ~7 MB | no |

Nothing runs in the background. No daemon, no service, no scheduled task, no
autostart entry. Deleting `~/.venvs/parsimony` and the project folder removes
everything this project has put on the machine.

**Before Sprint 5's full sweep**, point the ledger somewhere outside OneDrive —
that run produces far more rows than the ~400 a local `bench` writes:

```bash
python -m parsimony.surfaces.cli.main bench --out "$HOME/parsimony-runs"
```

### First-time setup (Windows)

```bash
python -m venv "$HOME/.venvs/parsimony"
```

```bash
"$HOME/.venvs/parsimony/Scripts/python.exe" -m pip install -e ".[dev]"
```

The tokenizer for Qwen2.5-1.5B (~11 MB) is fetched from Hugging Face on first
use and cached in `~/.cache/huggingface` — also outside OneDrive. If there is no
network, the system falls back to `HeuristicTokenizer` and **warns loudly**,
because token counts from an approximate tokenizer are not comparable with real
ones and must never be mixed into a results table.

## 3. Running it

Everything below runs in a couple of seconds.

```bash
"$HOME/.venvs/parsimony/Scripts/python.exe" -m parsimony.surfaces.cli.main demo
```

| Command | What it does |
|---|---|
| `... main demo` | The scripted review walkthrough |
| `... main chat "What is 847 * 23?"` | One query with the full stage trace |
| `... main chat "..." --baseline` | Same query with every module off |
| `... main chat "..." --repeat` | Sends it twice, so the cache hit is visible |
| `... main bench` | 2³ factorial ablation + the M6 row, writes a JSONL ledger |
| `... main gap3` | The compression × cache interaction experiment |
| `... main corpus` | Corpus composition and its freeze hash |
| `... main ledger-import runs/<id>.jsonl` | Fold a run into the SQLite analysis DB |

Tests:

```bash
"$HOME/.venvs/parsimony/Scripts/python.exe" -m pytest -q
```

## 4. What is real and what is simulated

This distinction must be stated out loud at the review; a reviewer who discovers
it themselves will discount everything else.

| Real | Simulated / deferred |
|---|---|
| Token counts (Qwen2.5 vocabulary, exact) | Model responses — `MockProvider`, deterministic canned text |
| Compression, cache, router, budgeter logic | Latency: TTFT 120 ms / TPOT 65 ms are **synthesised constants** |
| Fidelity gate and every revert it performs | Answer quality — no quality metrics yet |
| Ablation harness, ledger, config hashing | Semantic cache tier — needs embeddings (Sprint 2) |
| Early-stop rule (fires on real repetition) | M3 history manager, M4 assembler, M6b escalation |

Every ledger row records `model_digest = "mock:v1"`, so no simulated run can be
mistaken for a real one later (ADR-007).

## 5. Layout

```
parsimony/
├── docs/            00 architecture · 01 stages · 02 modules · 03 ADRs
│                    04 roadmap · 05 evaluation · 06 contracts · 07 corpus · 08 setup
├── src/parsimony/
│   ├── core/        L0 — types, protocols, config, ledger schema (stdlib only)
│   ├── infra/       L1 — tokenizer, NLP, providers, storage, ids
│   ├── modules/     L2 — M1 M2 M5 M6 M8   (M3, M4, M6b in later sprints)
│   ├── pipeline/    L3 — orchestrator, registry, assembly
│   ├── surfaces/    L4 — CLI + trace rendering
│   └── eval/        L5 — corpus loader, sweep runner
├── corpus/          30 conversations, 6 classes  (scales to 150 in Sprint 3)
├── tests/           unit + contract  (156 tests)
└── runs/            ledgers — gitignored, regenerable
```

## 6. Where to start reading

1. [`06-contracts.md`](06-contracts.md) — the types everything is written against.
2. `src/parsimony/pipeline/orchestrator.py` — the whole coordination loop, ~40 lines.
3. [`03-decision-log.md`](03-decision-log.md) — why anything is the way it is.

## 7. Adding a module

1. Write a class with `module_id`, `name`, `reads`, `writes`, `applies_to`, `propose`.
2. Return `ContextPatch` / `ShortCircuit` / `NoOp`. Never mutate the context.
3. Declare the right `TransformKind` — this is what scopes the fidelity gate (ADR-003).
4. Register it in `pipeline/registry.py:default_registry` and remove it from
   `PLANNED_STAGES`.
5. Put every threshold in `ParsimonyConfig`. A float literal in `modules/` is a bug.
6. The contract suite picks it up automatically — run `pytest`.

No orchestrator change is needed. That is the design working.
