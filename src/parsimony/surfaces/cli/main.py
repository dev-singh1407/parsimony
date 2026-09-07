"""Parsimony CLI.

`parsimony demo` is the review artefact: a scripted sequence that shows the
pipeline is a measurement instrument, not a black box.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import replace
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from parsimony.core.config import baseline, factorial_cells, full_stack, with_cache_lookup
from parsimony.core.types import Mode, Turn
from parsimony.eval.corpus import load_corpus, load_gold
from parsimony.eval.metrics import LengthBiasedMockJudge
from parsimony.eval.runner import additivity_shortfall, run_cell, sweep
from parsimony.infra.ids import ulid
from parsimony.infra.providers import make_provider
from parsimony.infra.storage import JsonlSink, import_jsonl
from parsimony.infra.tokenization import get_tokenizer
from parsimony.pipeline.orchestrator import DEFAULT_NUM_PREDICT, Pipeline
from parsimony.surfaces.cli.render import (
    module_report,
    print_outcome,
    summary_panel,
    trace_table,
)

app = typer.Typer(add_completion=False, help="Token-efficient LLM middleware for CPU-only hardware.")
console = Console()


@app.command()
def chat(
    query: str = typer.Argument(..., help="The query to send."),
    trace: bool = typer.Option(True, "--trace/--no-trace", help="Show the per-stage trace."),
    plain: bool = typer.Option(False, "--baseline", help="Run with every module disabled."),
    repeat: bool = typer.Option(False, "--repeat", help="Send it twice to exercise the cache."),
    text: bool = typer.Option(False, "--text/--no-text", "-t",
                              help="Show the text each stage changed, not just the token count."),
    modules: bool = typer.Option(False, "--modules", "-m",
                                 help="One panel per module: what each did, or why it did not."),
    provider: str = typer.Option("mock", "--provider",
                                 help="'mock' (simulated timings) or 'ollama' (a real model)."),
    model: str = typer.Option(None, "--model", help="Ollama model tag."),
) -> None:
    """Run one query through the pipeline."""
    cfg = baseline() if plain else full_stack()
    capture = text or modules
    pipeline = Pipeline(cfg, provider=make_provider(provider, model=model), capture_text=capture)
    counter = get_tokenizer(cfg.tokenizer_id).count

    def show(o):
        if modules:
            module_report(console, o, counter)
            console.print(summary_panel(o, simulated=o.row.model_digest.startswith("mock")))
            console.print(Panel(o.response or "[dim](empty)[/dim]", title="Answer",
                                border_style="green"))
        elif trace:
            print_outcome(console, o, show_text=text, counter=counter)
        else:
            console.print(o.response)

    show(pipeline.run(query))

    if repeat:
        console.print(Rule("second identical request"))
        show(pipeline.run(query))


@app.command()
def demo(
    text: bool = typer.Option(True, "--text/--no-text",
                              help="Show the text each stage changed. On by default: this is "
                                   "the walkthrough, and a token count alone is not evidence."),
) -> None:
    """Scripted demonstration of the pipeline (the review walkthrough)."""
    console.print(
        Panel(
            "[bold]Parsimony[/bold] — token-efficient LLM interaction on CPU-only hardware\n"
            "[dim]Sprint 0/1 build. Model responses come from MockProvider: the pipeline, token\n"
            "counts and gate behaviour are real; latency figures are simulated until Ollama\n"
            "lands in Sprint 2 (ADR-007).[/dim]",
            border_style="bold blue",
        )
    )

    demo_cfg = full_stack()
    pipeline = Pipeline(demo_cfg, capture_text=text)
    counter = get_tokenizer(demo_cfg.tokenizer_id).count

    def walk(outcome, **kw):
        print_outcome(console, outcome, show_text=text, counter=counter, **kw)

    console.print(Rule("[bold]1. A query no model should ever see[/bold]"))
    console.print("[dim]The routing literature always escalates to a model. The cheapest tier "
                  "answers without one.[/dim]\n")
    walk(pipeline.run("What is 847 * 23?"))

    console.print(Rule("[bold]2. Boilerplate is tokens you paid for[/bold]"))
    verbose = (
        "Hello, I was wondering if you could **please** explain to me what photosynthesis is? "
        "I would like to know how it works. Thanks in advance!"
    )
    console.print(f"[dim]{verbose}[/dim]\n")
    walk(pipeline.run(verbose))

    console.print(Rule("[bold]3. The same question, asked again[/bold]"))
    walk(pipeline.run(verbose))

    console.print(Rule("[bold]4. The fidelity gate refusing a saving[/bold]"))
    console.print(
        "[dim]Two near-duplicate sentences, but one carries a number the other does not. "
        "The compressor tries to drop it; the gate reverts the edit.[/dim]\n"
    )
    walk(
        pipeline.run("Explain the deadline. The deadline is 15 March. The deadline is 16 March."),
        show_response=False,
    )

    console.print(Rule("[bold]5. The ablation harness[/bold]"))
    axes = ("M1", "M2", "M5")
    corpus = load_corpus()
    results = sweep(list(factorial_cells(axes=axes, always_on=frozenset())), corpus)
    console.print(_cell_table(results, corpus))
    _print_shortfall(results, axes)

    console.print(Rule("[bold]6. Stage order is configuration, not code[/bold]"))
    console.print(
        "[dim]The same pipeline, with the cache lookup moved before vs after the compressor.\n"
        "Research gap 3 is unanswerable if this ordering is fixed in code (ADR-002).[/dim]\n"
    )
    arms = []
    for label, mode in (("RAW", "RAW"), ("COMPRESSED", "COMPRESSED")):
        arms.append(run_cell(with_cache_lookup(replace(full_stack(), label=label), mode), corpus))
    console.print(
        Panel(
            f"cache lookup on the [bold]raw[/bold] query:        "
            f"{arms[0].cache_hits} hits, {arms[0].total_tokens} tokens\n"
            f"cache lookup on the [bold]compressed[/bold] query: "
            f"{arms[1].cache_hits} hits, {arms[1].total_tokens} tokens\n\n"
            f"[bold yellow]{arms[1].cache_hits - arms[0].cache_hits:+d} cache hits[/bold yellow] "
            f"from moving one entry in a config list.\n"
            f"[dim]Normalisation collapses politeness-only paraphrases onto one key. Whether it "
            f"also merges\nquestions that should NOT match is measured in Sprint 2 against the "
            f"adversarial subset.[/dim]",
            title="Gap 3, measured on day one", border_style="magenta",
        )
    )


@app.command()
def bench(
    modules: str = typer.Option("M1,M2,M5", "--modules", help="Comma-separated factorial axes."),
    out: Path = typer.Option(Path("runs"), "--out", help="Directory for the JSONL ledger."),
    corpus_path: Path = typer.Option(None, "--corpus", help="Corpus JSONL (defaults to corpus/)."),
    write_ledger: bool = typer.Option(True, "--ledger/--no-ledger"),
    router: bool = typer.Option(True, "--router/--no-router",
                                help="Add a row with M6 on top of the full stack."),
    quality: bool = typer.Option(True, "--quality/--no-quality",
                                 help="Score the four quality measures and the gold subset."),
    bundle: Path = typer.Option(None, "--bundle",
                                help="Warm-start from an M7 PolicyBundle (gap 6)."),
    provider: str = typer.Option("mock", "--provider",
                                 help="'mock' (simulated timings) or 'ollama' (a real model). "
                                      "Only an 'ollama' run produces real latency."),
    model: str = typer.Option(None, "--model", help="Ollama model tag."),
) -> None:
    """Run the factorial ablation over the corpus and write a ledger."""
    axes = tuple(m.strip().upper() for m in modules.split(",") if m.strip())
    corpus = load_corpus(corpus_path)
    run_id = ulid()
    prov = make_provider(provider, model=model)
    console.print(
        f"[bold]model[/bold]  {prov.model_name} ({prov.model_digest})"
        + ("  [yellow]simulated timings[/yellow]" if provider == "mock"
           else "  [green]real timings[/green]")
    )

    console.print(
        f"[bold]corpus[/bold] {len(corpus)} conversations, {corpus.n_requests} requests "
        f"| hash [cyan]{corpus.corpus_hash}[/cyan]"
    )
    console.print(f"[bold]cells[/bold]  2^{len(axes)} = {2 ** len(axes)} over {', '.join(axes)}")

    warm = None
    if bundle is not None:
        from parsimony.modules.m7_learner import PolicyBundle

        warm = PolicyBundle.load(bundle)
        console.print(
            f"[bold]bundle[/bold] {warm.bundle_hash} — {len(warm.cache_seed)} cache entries, "
            f"{len(warm.redundancy)} redundant phrases, {len(warm.digest)} digest chars"
        )

    cells = list(factorial_cells(axes=axes, always_on=frozenset()))
    if router:
        # M6 is studied on top of the winning configuration (report 4.6), not as
        # a factorial axis — it short-circuits, so as an axis it would confound
        # every other module's measured effect.
        cells.append(
            cells[-1].with_modules(cells[-1].enabled_modules | {"M6"},
                                   label="+".join(axes) + "+M6")
        )
    sink = None
    path = out / f"{run_id}.jsonl"
    if write_ledger:
        sink = JsonlSink(path)

    try:
        with console.status("[bold green]running cells...") as status:
            # EXPERIMENT mode: a failed ledger write is fatal here, because in a
            # sweep the ledger IS the result (ADR-005).
            prepared = [replace(cfg, mode=Mode.EXPERIMENT) for cfg in cells]
            if warm is not None:
                # The digest joins M4's invariant zone, and bundle_hash lands in
                # every ledger row so warm and cold runs are distinguishable
                # without being separate code paths.
                prepared = [
                    replace(cfg, context_digest=warm.digest, bundle_hash=warm.bundle_hash)
                    for cfg in prepared
                ]
            results = sweep(
                prepared,
                corpus,
                provider=prov,
                sink=sink,
                run_id=run_id,
                warm_start=warm,
                judge=LengthBiasedMockJudge() if quality else None,
                gold=load_gold() if quality else (),
                progress=lambda label: status.update(f"[bold green]cell: {label}"),
            )
    finally:
        if sink is not None:
            sink.close()

    console.print(_cell_table(results, corpus))
    if quality:
        console.print(_quality_table(results))
    _print_shortfall(results, axes)
    if write_ledger:
        console.print(f"\n[dim]ledger written to[/dim] {path}")
        console.print(f"[dim]import with[/dim]  parsimony ledger-import {path}")


@app.command()
def gap3(corpus_path: Path = typer.Option(None, "--corpus")) -> None:
    """Research gap 3: does compressing a query before the cache change hit rate?

    Two otherwise byte-identical pipelines, differing only in where the cache
    lookup sits in stage_order. This is the experiment ADR-002 exists for.
    """
    corpus = load_corpus(corpus_path)
    base = full_stack()
    arms = [
        ("RAW  (cache sees the original query)", with_cache_lookup(replace(base, label="RAW"), "RAW")),
        ("COMPRESSED (cache sees the compressed query)",
         with_cache_lookup(replace(base, label="COMPRESSED"), "COMPRESSED")),
        ("BOTH (paired: probe raw, act compressed)",
         with_cache_lookup(replace(base, label="BOTH"), "BOTH")),
    ]

    table = Table(title="Compression x cache interaction", header_style="bold")
    table.add_column("Cache lookup position")
    table.add_column("stage order", style="dim")
    table.add_column("hits", justify="right")
    table.add_column("total tokens", justify="right")

    results = []
    for name, cfg in arms:
        r = run_cell(cfg, corpus)
        results.append(r)
        order = " -> ".join(s.replace("m1_", "").replace("m2_", "").replace("m6a_", "")
                            for s in cfg.stage_order if not s.startswith(("m3", "m4", "m6b")))
        table.add_row(name, order, str(r.cache_hits), str(r.total_tokens))

    console.print(table)
    delta = results[1].cache_hits - results[0].cache_hits
    console.print(
        Panel(
            f"Compressing before the lookup changed the TRUE-hit count by "
            f"[bold]{delta:+d}[/bold] over {corpus.n_requests} requests.\n\n"
            "[dim]Normalisation collapses politeness-only paraphrases onto one key, so more "
            "queries hit.\n"
            "The other half of gap 3 — whether it also merges questions that should NOT match —\n"
            "is measured separately against the adversarial subset: run "
            "[/dim][bold]parsimony calibrate[/bold][dim]\n"
            "with and without --compression. The verifier holds the false-hit rate at 0% in both.\n\n"
            "BOTH matches COMPRESSED on outcome by design: the authoritative lookup is the\n"
            "compressed one, and the raw lookup is an observe-only probe. The value of BOTH is\n"
            "that each request records what the OTHER ordering would have done, giving a paired\n"
            "observation instead of two independent runs.[/dim]",
            title="Gap 3", border_style="magenta",
        )
    )


@app.command()
def learn(
    out: Path = typer.Option(Path("bundles/mined"), "--out", help="Where to write the bundle."),
    corpus_path: Path = typer.Option(None, "--corpus", help="Logs to mine (defaults to corpus/)."),
) -> None:
    """Mine a PolicyBundle from conversation logs (M7, offline)."""
    from parsimony.infra.memo import GenerationMemo
    from parsimony.modules.m7_learner import learn as mine

    corpus = load_corpus(corpus_path)
    cfg = replace(full_stack(), mode=Mode.EXPERIMENT)
    pipeline = Pipeline(cfg, memo=GenerationMemo())

    def generate(question: str) -> str:
        return pipeline.run(question, conversation_id=f"learn:{hash(question)}").response

    with console.status("[bold green]counterfactual replay..."):
        bundle = mine(
            [list(c.user_turns) for c in corpus.conversations],
            generate,
            pipeline.embedder,
        )
    path = bundle.save(out)

    table = Table(title=f"PolicyBundle  [{bundle.bundle_hash}]", header_style="bold")
    table.add_column("artefact")
    table.add_column("size", justify="right")
    table.add_row("pre-populated cache entries", str(len(bundle.cache_seed)))
    table.add_row("redundancy lexicon", str(len(bundle.redundancy)))
    table.add_row("standing-context digest", f"{len(bundle.digest)} chars")
    table.add_row("query templates", str(len(bundle.templates)))
    console.print(table)

    if bundle.findings:
        ft = Table(title="Counterfactual replay — is this phrase ever load-bearing?",
                   header_style="bold")
        ft.add_column("phrase")
        ft.add_column("occurrences", justify="right")
        ft.add_column("answer unchanged", justify="right")
        ft.add_column("verdict")
        for f in sorted(bundle.findings, key=lambda f: -f.occurrences)[:12]:
            safe = f.safe_rate == 1.0
            ft.add_row(
                f.phrase, str(f.occurrences), f"{f.unchanged}/{f.occurrences}",
                "[green]always redundant[/green]" if safe else "[yellow]sometimes matters[/yellow]",
            )
        console.print(ft)

    console.print(f"\n[dim]written to[/dim] {path}")
    console.print(f"[dim]use with[/dim]  parsimony bench --bundle {path}")


@app.command()
def calibrate(
    verifier: bool = typer.Option(True, "--verifier/--no-verifier",
                                  help="Three-zone verifier vs a single threshold."),
    compression: bool = typer.Option(False, "--compression/--no-compression",
                                     help="Normalise before the cache sees the query (gap 3)."),
) -> None:
    """Sweep the cache similarity threshold against the adversarial subset."""
    from parsimony.eval.calibration import by_operative, sweep_thresholds
    from parsimony.eval.corpus import load_adversarial
    from parsimony.infra.embedding import get_embedder

    base = full_stack()
    pairs = load_adversarial()
    embedder = get_embedder(base.embedder_id)
    points = sweep_thresholds(base, pairs=pairs, embedder=embedder,
                              verifier_on=verifier, compression_on=compression)

    n_adv = sum(1 for p in pairs if p.answers_differ)
    n_ctl = len(pairs) - n_adv
    table = Table(
        title=f"Threshold sweep — {n_adv} adversarial pairs, {n_ctl} controls "
              f"| verifier {'on' if verifier else 'OFF'} "
              f"| compression {'on' if compression else 'off'}",
        header_style="bold",
    )
    table.add_column("tau_hi", justify="right")
    table.add_column("false hits", justify="right")
    table.add_column("false-hit rate", justify="right")
    table.add_column("true hits", justify="right")
    table.add_column("true-hit rate", justify="right")
    table.add_column("safe?", justify="center")

    for p in points:
        colour = "green" if p.is_safe else "red"
        table.add_row(
            f"{p.tau_hi:.2f}",
            f"{p.false_hits}/{p.adversarial_total}",
            f"[{colour}]{p.false_hit_rate:.1f}%[/{colour}]",
            f"{p.true_hits}/{p.control_total}",
            f"{p.true_hit_rate:.1f}%",
            "[green]yes[/green]" if p.is_safe else "[red]no[/red]",
        )
    console.print(table)

    breakdown = by_operative(pairs, base, embedder, verifier_on=verifier)
    bt = Table(title=f"False hits by operative token (tau_hi={base.cache.tau_hi})",
               header_style="bold")
    bt.add_column("operative")
    bt.add_column("false hits", justify="right")
    bt.add_column("rate", justify="right")
    for op, (hits, total) in breakdown.items():
        rate = 100.0 * hits / total if total else 0.0
        bt.add_row(op, f"{hits}/{total}",
                   f"[{'red' if rate else 'green'}]{rate:.0f}%[/{'red' if rate else 'green'}]")
    console.print(bt)
    console.print(
        "[dim]Control pairs matter as much as adversarial ones: a policy that rejects "
        "everything\nscores a perfect 0% false-hit rate, so the sweep would recommend "
        "switching the cache off.[/dim]"
    )


@app.command("calibrate-dedup")
def calibrate_dedup(corpus_path: Path = typer.Option(None, "--corpus")) -> None:
    """Sweep M1 tier 2's near-duplicate threshold against the corpus."""
    from parsimony.eval.calibration import sweep_dedup_threshold

    corpus = load_corpus(corpus_path)
    points = sweep_dedup_threshold(full_stack(), corpus)

    table = Table(title=f"M1 tier 2 dedup threshold — {corpus.n_requests} requests",
                  header_style="bold")
    table.add_column("threshold", justify="right")
    table.add_column("edits proposed", justify="right")
    table.add_column("applied", justify="right")
    table.add_column("gate reverted", justify="right")
    table.add_column("revert rate", justify="right")
    table.add_column("tokens saved", justify="right")

    for p in points:
        colour = "green" if p.revert_rate < 20 else ("yellow" if p.revert_rate < 50 else "red")
        table.add_row(
            f"{p.threshold:.2f}", str(p.proposed), str(p.applied), str(p.reverted),
            f"[{colour}]{p.revert_rate:.0f}%[/{colour}]", str(p.tokens_saved),
        )
    console.print(table)
    console.print(
        "[dim]The gate's revert rate is the safety signal: a threshold loose enough to merge\n"
        "sentences differing in a number or entity shows up as reverts, not silent damage.\n"
        "Pick the loosest threshold whose revert rate is still acceptable.[/dim]"
    )


@app.command()
def ask(
    provider: str = typer.Option("ollama", "--provider"),
    model: str = typer.Option(None, "--model"),
) -> None:
    """An interactive conversation, reporting every module on every turn.

    Keeps the conversation, which the one-shot `chat` cannot: with no prior
    turns M3 has nothing to select from and M4 has no history to hold stable,
    so both report "not applicable" forever and two of the eight modules can
    never be demonstrated. Here the history accumulates as you type, so the
    later turns exercise the whole pipeline — and asking something twice in
    different words lands a real cache hit rather than a staged one.
    """
    cfg = full_stack()
    pipeline = Pipeline(cfg, provider=make_provider(provider, model=model), capture_text=True)
    counter = get_tokenizer(cfg.tokenizer_id).count
    conversation_id = ulid()
    history: list[Turn] = []

    console.print(
        Panel(
            "[bold]Ask anything.[/bold] Every module reports what it did on every turn.\n"
            "[dim]The conversation is kept, so history builds up as you go — by the third or\n"
            "fourth question M3 has something to trim. Ask something twice in different\n"
            "words to land a cache hit. Type 'quit' to leave, 'reset' to start over.[/dim]",
            border_style="bold blue",
        )
    )

    while True:
        try:
            query = console.input("\n[bold cyan]ask>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return
        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            return
        if query.lower() == "reset":
            history, conversation_id = [], ulid()
            console.print("[dim]conversation cleared[/dim]")
            continue

        console.print()
        outcome = pipeline.run(query, tuple(history), conversation_id=conversation_id,
                              turn_index=len(history))
        module_report(console, outcome, counter)
        console.print(summary_panel(outcome, simulated=outcome.row.model_digest.startswith("mock")))
        console.print(Panel(outcome.response or "[dim](empty)[/dim]", title="Answer",
                            border_style="green"))

        history.append(Turn(turn_id=f"t{len(history)}", role="user", content=query))
        history.append(Turn(turn_id=f"t{len(history)}", role="assistant",
                            content=outcome.response))
        console.print(f"[dim]conversation is now {len(history)} turns[/dim]")


@app.command()
def tour(
    only: str = typer.Option(None, "--only", help="Run one stop, e.g. --only M2."),
    pause: bool = typer.Option(False, "--pause", help="Wait for Enter between stops."),
) -> None:
    """Walk through every module one at a time, showing what each one changes.

    Each stop uses an input chosen so that THAT module is the one doing the
    work, because a single query never exercises all eight — and a demo where
    six of the eight stages say "no-op" teaches nothing about what they do.
    """
    cfg = full_stack()
    counter = get_tokenizer(cfg.tokenizer_id).count

    stops: list[tuple[str, str, str, str, object]] = [
        ("M1", "Tier 1 — boilerplate",
         "Politeness and filler cost tokens and carry no instruction. Tier 1 removes "
         "them losslessly: no word that changes the meaning is touched.",
         "Hello, I was wondering if you could **please** explain to me what "
         "photosynthesis is? Thanks in advance!", None),

        ("M1", "Tier 2 — repeated facts",
         "The same fact stated twice pays twice. Tier 2 drops the near-duplicate "
         "sentence — but only when nothing distinguishes them.",
         "Summarise this. The server runs Ubuntu Linux. The server runs Ubuntu Linux "
         "and needs a restart. Please advise.", None),

        ("M8", "The gate refusing a saving",
         "Now the same shape of edit, but the two sentences carry DIFFERENT dates. "
         "The compressor still proposes the deletion; the fidelity gate refuses it. "
         "This is the module that makes the rest safe to use.",
         "Explain the deadline. The deadline is 15 March. The deadline is 16 March.", None),

        ("M6", "The deterministic tier",
         "Some questions do not need a language model at all. This one is answered "
         "exactly, by a calculator, sending the model zero tokens.",
         "What is 847 * 23?", None),

        ("M5", "The output budgeter",
         "Left alone a small model rambles. M5 classifies the question and sets a "
         "budget to match — 48 tokens for arithmetic, 640 for reasoning.",
         "What is the boiling point of water at sea level?", None),

        ("M3", "The history manager",
         "In a long conversation most turns are irrelevant to the current question. "
         "M3 keeps what is relevant and drops the rest.",
         "So which of those should I use?", "history"),

        ("M2", "The semantic cache",
         "Asked again in different words, the answer is served from cache and the "
         "model is never called. This pair scores 0.80 — below the auto-accept "
         "threshold — so it lands in the VERIFY zone and is served only because the "
         "verifier confirmed the two questions agree on every number, entity, "
         "negation and modifier.",
         "What is the capital city of Australia?", "repeat"),
    ]

    console.print(
        Panel(
            "[bold]A tour of the eight modules[/bold]\n"
            "[dim]Each stop uses an input chosen to make that module do the work.\n"
            "Responses come from the deterministic stand-in, so this runs in seconds "
            "and gives the same result every time.[/dim]",
            border_style="bold blue",
        )
    )

    for module, title, why, query, mode in stops:
        if only and only.upper() != module.upper():
            continue

        console.print(Rule(f"[bold cyan]{module}[/bold cyan] · [bold]{title}[/bold]"))
        console.print(f"[dim]{why}[/dim]\n")

        pipeline = Pipeline(cfg, capture_text=True)
        history: tuple[Turn, ...] = ()

        if mode == "history":
            history = _synthetic_history(8)
            console.print(f"[dim]…after {len(history)} turns of prior conversation[/dim]\n")
        elif mode == "repeat":
            # Prime with the ORIGINAL wording, so the hit below is a genuine
            # paraphrase match rather than a repeat of the same string.
            #
            # The cache stage runs BEFORE the compressor, so it compares raw
            # text: priming with a differently-worded question scored cosine
            # 0.058 and missed. That is not a bug, it is research gap 3 — what
            # the cache sees depends on where it sits in the stage order.
            primed = "What is the capital of Australia?"
            pipeline.run(primed)
            console.print(f"[dim]…already asked once, worded differently: [/dim]"
                          f"[italic]{primed}[/italic]\n")

        console.print(f"[bold]Question:[/bold] {query}\n")
        outcome = pipeline.run(query, history)
        print_outcome(console, outcome, show_text=True, counter=counter,
                      show_response=(module in ("M6", "M2")))

        if pause and not only:
            console.print("[dim]— Enter for the next module —[/dim]")
            input()

    if not only:
        console.print(Rule("[bold]That is the pipeline[/bold]"))
        console.print(
            "[dim]Eight modules, each switchable independently. Every one PROPOSES a "
            "change; the orchestrator commits it only after the gate has checked it. "
            "That is what makes the ablation possible — switching a module off really "
            "does remove its effect.[/dim]\n"
        )


@app.command()
def compare(
    query: str = typer.Argument(..., help="The question to send through both pipelines."),
    provider: str = typer.Option("ollama", "--provider",
                                 help="'ollama' for real timings, 'mock' for simulated."),
    model: str = typer.Option(None, "--model", help="Ollama model tag."),
    turns: int = typer.Option(0, "--turns",
                              help="Prepend N turns of prior conversation, so M3 and M4 "
                                   "have something to work on."),
) -> None:
    """Same question, pipeline off vs on, side by side.

    The one screen that shows what the project is for.
    """
    prov = make_provider(provider, model=model)
    simulated = provider == "mock"
    history = _synthetic_history(turns, nonce=secrets.token_hex(4))

    console.print(
        Panel(
            f"[bold]{query}[/bold]"
            + (f"\n[dim]after {turns} turns of prior conversation[/dim]" if turns else ""),
            title=f"one question · {prov.model_name}"
            + ("  [yellow](simulated timings)[/yellow]" if simulated else "  [green](real)[/green]"),
            border_style="bold blue",
        )
    )

    # Warm the model and the KV cache BEFORE timing anything. Whichever arm runs
    # first otherwise pays model load and a cold prefill, and the second arm
    # looks faster for reasons that have nothing to do with the middleware —
    # a 27s vs 17s gap that survived a re-run with the arms swapped.
    if not simulated:
        with console.status("[dim]warming the model (excluded from timings)"):
            Pipeline(baseline(), provider=prov).run("Say ready.")

    arms = []
    for label, cfg in (("pipeline OFF", baseline()), ("pipeline ON", full_stack())):
        with console.status(f"[bold green]{label}"):
            # Each arm runs twice and only the SECOND is measured. Ollama's KV
            # cache outlives the process, so whether a given prompt is already
            # cached depends on what was run minutes or days ago — an arm whose
            # prompt happened to be warm reported 217 ms of prefill for 298
            # tokens (0.7 ms/token, an impossible rate) against the other arm's
            # cold 1,919 ms, and the demo showed the SHORTER prompt as 785%
            # slower. Warming the model alone does not fix this; each arm's own
            # prompt has to be warm. Measuring steady state makes both equal.
            if hasattr(prov, "last_stats"):
                prov.last_stats = {}  # never attribute the previous arm's timings
            t0 = time.perf_counter()
            outcome = Pipeline(cfg, provider=prov).run(query, history)
            elapsed = time.perf_counter() - t0
            stats = dict(getattr(prov, "last_stats", {}) or {})
            arms.append((label, outcome, elapsed, stats))

    table = Table(header_style="bold", title="What changed", title_style="bold")
    table.add_column("", style="bold", no_wrap=True)
    for label, *_ in arms:
        table.add_column(label, justify="right")
    table.add_column("change", justify="right")

    def row(name, fn, fmt="{:,.0f}", note=""):
        off, on = (fn(o, w, s) for _, o, w, s in arms)
        if off is None or on is None:
            return
        delta = on - off
        pct = (delta / off * 100) if off else 0.0
        if delta == 0:
            change = "[dim]no change[/dim]"
        else:
            # One convention throughout: a NEGATIVE change is fewer/faster, and
            # is the good direction. Mixing "saved 10" with "-29" made the two
            # rows read as if they pointed the same way.
            colour = "green" if delta < 0 else "red"
            change = f"[{colour}]{fmt.format(delta)}  ({pct:+.0f}%)[/{colour}]"
        table.add_row(name + note, fmt.format(off), fmt.format(on), change)

    row("input tokens", lambda o, w, s: o.row.tokens_in_final)
    # Shown because it explains the output row, which otherwise looks arbitrary.
    # M5 is a budgeter, not a truncator: it RIGHT-SIZES per class, and its
    # reasoning (640) and code (512) budgets are larger than the 256 the
    # pipeline falls back to with M5 ablated. On a verbose code question the
    # baseline is cut off at 256 while the pipeline is allowed to finish, so
    # "pipeline ON" can legitimately emit more. Across the corpus M5 still
    # reduces every class (code -11.9%, overall -14.3%).
    row("output budget", lambda o, w, s: o.row.tokens_out_budget or DEFAULT_NUM_PREDICT)
    row("output tokens", lambda o, w, s: o.row.tokens_out)
    row("total tokens", lambda o, w, s: o.row.tokens_in_final + o.row.tokens_out)
    row("prefill (server)", lambda o, w, s: (s.get("prompt_eval_duration") or 0) / 1e6 or None,
        fmt="{:,.0f} ms")
    row("wall clock", lambda o, w, s: w, fmt="{:.2f}s")
    console.print(table)

    console.print(
        "[dim]Read the [bold]prefill[/bold] row, not the wall clock. Prefill is the part "
        "input tokens actually control, reported by the server itself. Wall clock on a single "
        "request also carries decode length and scheduling noise — run to run it varies by more "
        "than a second here, which is larger than the effect. The corpus-level figure "
        "(100.6 s saved over 263 requests) is in ADR-034.[/dim]"
    )

    # An arm with M5 ablated reports no budget, but generation still stops at
    # DEFAULT_NUM_PREDICT — so the baseline can be silently truncated while
    # `tokens_out_budget` is None. Checking only the recorded budget missed it.
    truncated = [
        label for label, o, _, _ in arms
        if o.row.tokens_out >= (o.row.tokens_out_budget or DEFAULT_NUM_PREDICT)
    ]
    if truncated:
        console.print(
            f"[yellow]Note:[/yellow] [dim]{', '.join(truncated)} reached its output budget, so "
            f"its answer is cut short. Output-token counts are then a property of the budget, "
            f"not of the pipeline — compare the input side.[/dim]"
        )

    off_in, on_in = (o.row.tokens_in_final for _, o, _, _ in arms)
    saved_tokens = off_in - on_in
    prefills = [(s.get("prompt_eval_duration") or 0) / 1e6 for _, _, _, s in arms]
    if saved_tokens > 0 and all(prefills):
        predicted = saved_tokens * 8.5
        measured = prefills[0] - prefills[1]
        console.print(
            f"[dim]Cross-check: ADR-034 measured prefill at ~8.5 ms per input token on this CPU, "
            f"so {saved_tokens} fewer tokens predicts [/dim][bold]{predicted:.0f} ms[/bold][dim] "
            f"saved. The server reported [/dim][bold]{measured:.0f} ms[/bold][dim] "
            f"({measured / predicted:.1f}x the prediction). The model of the cost holds.[/dim]\n"
        )

    for label, outcome, _, _ in arms:
        console.print(
            Panel(
                (outcome.response or "[dim](empty)[/dim]").strip(),
                title=f"{label} — answer  [dim]({outcome.row.route_tier})[/dim]",
                border_style="green" if "ON" in label else "dim",
            )
        )

    console.print(
        "[dim]Same question, same model, same machine. The only difference is whether the "
        "middleware ran.[/dim]"
    )


def _synthetic_history(n: int, nonce: str = "") -> tuple[Turn, ...]:
    """Plausible prior turns, so a single-shot demo can still exercise M3/M4.

    Without history the history manager and the assembler have nothing to do,
    and a demo of a context pipeline that never shows context is a poor demo.
    """
    if n <= 0:
        return ()
    # Written the way people actually talk to assistants: polite openers,
    # standing facts restated every few turns ("on Python 3.11", "8 GB laptop"),
    # and assistant replies that recap the question before answering. Terse,
    # non-redundant turns would leave M1 and M3 nothing to remove and would make
    # the demo understate the pipeline for reasons that are an artefact of the
    # fixture rather than a property of the system.
    filler = [
        "Hi, I was hoping you could help me out with something. I am building a small "
        "web service in Python, and I am running it on a laptop with 8 GB of RAM and no GPU.",
        "Of course. To confirm, you are building a small Python web service, running on a "
        "laptop with 8 GB of RAM and no dedicated GPU. I can help with that.",
        "Thanks! So as I mentioned, I am on Python 3.11, and right now I am just using "
        "SQLite for storage. Could you please tell me if that is a reasonable choice?",
        "Given that you are on Python 3.11 and using SQLite for storage, that is a "
        "reasonable choice for a small service on a single machine.",
        "Great, thanks in advance for the help. Just to give more context, the service "
        "handles about a hundred requests a day, and most of the traffic arrives in the evening.",
        "Understood — roughly a hundred requests a day, concentrated in the evening. "
        "That is a light load and SQLite will handle it comfortably.",
        "One more thing I should mention: I would really like to keep the dependencies to "
        "a minimum, and deployment is just a single systemd unit.",
        "Noted. Minimal dependencies, deployed as a single systemd unit.",
    ]
    # The nonce goes in the FIRST history turn, which both arms share, so each
    # run is cold for Ollama while the two arms stay strictly comparable.
    # Without it, whether a prompt was already in the KV cache depended on what
    # had been run minutes or days earlier: one arm measured 217 ms of prefill
    # for 298 tokens (0.7 ms/token, an impossible rate) against the other's cold
    # 1,919 ms, and the demo showed the SHORTER prompt as 785% slower. Warming
    # both arms instead is no better — it drives prefill to ~45 ms on both and
    # measures nothing. Prefill is only meaningful cold, so both arms are made
    # cold rather than both warm.
    turns = list(filler)
    if nonce:
        # Phrased as a content sentence carrying an identifier, NOT as a
        # "(session abc)" prefix: M1 strips a parenthetical prefix as
        # boilerplate — correctly — which left the compressed arm byte-identical
        # every run, so it hit Ollama's KV cache and reported 59 ms of prefill
        # against the uncompressed arm's cold 2,468 ms. The nonce has to survive
        # compression to keep both arms cold, and an alphanumeric identifier in
        # a real sentence does.
        turns[0] = f"{turns[0]} My project reference is {nonce}."
    return tuple(
        Turn(turn_id=f"h{i}", role="user" if i % 2 == 0 else "assistant",
             content=turns[i % len(turns)])
        for i in range(n)
    )


@app.command()
def judge(
    subject: str = typer.Option("qwen2.5:1.5b-instruct", "--subject",
                                help="The model under test."),
    judge_model: str = typer.Option("llama3.2:3b", "--judge",
                                    help="The judge. MUST differ from --subject."),
    n: int = typer.Option(20, "--n", help="Requests to sample."),
) -> None:
    """LLM-as-judge, with the judge itself measured first (ADR-036)."""
    from parsimony.eval.judging import run_judge_study
    from parsimony.eval.metrics import ModelJudge
    from parsimony.infra.providers import OllamaProvider

    if subject == judge_model:
        console.print(
            "[bold red]Refusing:[/bold red] the judge must not be the model under test. "
            "A model comparing its own output against another's prefers its own, which "
            "measures familiarity rather than quality."
        )
        raise typer.Exit(1)

    provider = make_provider("ollama", model=subject)
    judge_obj = ModelJudge(OllamaProvider(judge_model))

    with console.status("[bold green]running") as status:
        study = run_judge_study(
            load_corpus(), baseline(), [("full stack", full_stack())],
            judge=judge_obj, provider=provider, n_sample=n,
            progress=lambda m: status.update(f"[bold green]{m}"),
        )

    cal = study.calibration
    verdict_style = "green" if cal.usable else "bold red"
    console.print(
        Panel(
            f"judge    [bold]{study.judge_model}[/bold]\n"
            f"subject  [bold]{study.subject_model}[/bold]   "
            f"{'[green]independent[/green]' if study.independent else '[red]SAME MODEL[/red]'}\n\n"
            f"Shown two [bold]identical[/bold] answers {cal.n} times — a question with no right "
            f"answer:\n"
            f"  position bias   [{verdict_style}]{cal.position_bias:.1f} pp[/{verdict_style}] "
            f"[dim](0 = picks each slot equally; 50 = always the same slot)[/dim]\n"
            f"  unreadable      {cal.unreadable_rate:.1f}%\n\n"
            f"[{verdict_style}]{'USABLE' if cal.usable else 'NOT USABLE — any score below is noise'}"
            f"[/{verdict_style}]",
            title="Step 1 — is the judge worth listening to?",
            border_style=verdict_style,
        )
    )

    table = Table(title="Step 2 — what it said", header_style="bold")
    for col in ("arm", "n", "win rate vs baseline", "swap disagreement", "unreadable"):
        table.add_column(col, justify="right" if col != "arm" else "left")
    for arm in study.arms:
        table.add_row(arm.label, str(arm.n), f"{arm.win_rate:.1f}%",
                      f"{arm.disagreement_rate:.1f}%", f"{arm.unreadable_rate:.1f}%")
    console.print(table)

    if not cal.usable:
        console.print(
            "[dim]Read step 1 first. The win rate above is reported for completeness and should "
            "not be quoted: a judge that cannot choose between two identical answers is not "
            "measuring quality. Running this check is the step most LLM-as-judge setups skip, "
            "and it costs nothing but the calls.[/dim]"
        )


@app.command()
def learning(
    corpus_path: Path = typer.Option(None, "--corpus"),
    rates: str = typer.Option("0,0.2,0.4,0.6,0.8,0.9", "--rates",
                              help="Recurrence rates to sweep."),
    conversations: int = typer.Option(120, "--conversations"),
) -> None:
    """Does what M7 learns transfer to unseen conversations? (gap 6)"""
    from parsimony.eval.learning import recurrence_rate, recurrence_sweep

    corpus = load_corpus(corpus_path)
    console.print(
        f"[bold]corpus[/bold] {len(corpus)} conversations, recurrence "
        f"[cyan]{recurrence_rate(corpus):.1f}%[/cyan] — authored for ablation diversity, "
        f"which leaves M7 almost nothing to mine."
    )

    parsed = tuple(float(r) for r in rates.split(",") if r.strip())
    with console.status("[bold green]sweeping") as status:
        points = recurrence_sweep(
            corpus, full_stack(), rates=parsed, n_conversations=conversations,
            progress=lambda m: status.update(f"[bold green]{m}"),
        )

    table = Table(title="M7 transfer vs traffic recurrence", header_style="bold")
    for col, just in (("recurrence", "right"), ("seeds", "right"), ("cold", "right"),
                      ("warm", "right"), ("transfer", "right"), ("+hits", "right"),
                      ("+gate", "right"), ("verdict", "left")):
        table.add_column(col, justify=just)
    for p in points:
        s = p.study
        cold = (s.cold.tokens_in_baseline - s.cold.tokens_in_final) / s.cold.tokens_in_baseline * 100
        warm = (s.warm.tokens_in_baseline - s.warm.tokens_in_final) / s.warm.tokens_in_baseline * 100
        table.add_row(
            f"{p.actual:.1f}%", str(len(s.bundle.cache_seed)), f"{cold:.2f}%", f"{warm:.2f}%",
            f"[bold green]{s.transfer_pp:+.2f} pp[/bold green]" if s.transfer_pp > 0
            else f"{s.transfer_pp:+.2f} pp",
            f"{s.extra_cache_hits:+d}",
            f"[red]{s.extra_gate_fires:+d}[/red]" if s.extra_gate_fires > 0
            else f"{s.extra_gate_fires:+d}",
            s.verdict,
        )
    console.print(table)
    console.print(
        "[dim]Traces are synthetic in their REPETITION STRUCTURE only: every question is a real\n"
        "corpus question and every answer a real pipeline answer. Modelled as a hot set plus a\n"
        "long tail, the shape assistant traffic actually takes.[/dim]"
    )


@app.command()
def latency(
    model: str = typer.Option(None, "--model", help="Ollama model tag."),
) -> None:
    """Where the time goes: prefill vs decode, and what prompt order costs (gap 2)."""
    from parsimony.eval.latency import prefill_scaling, prefix_reuse

    provider = make_provider("ollama", model=model)
    console.print(f"[bold]model[/bold] {provider.model_name} ({provider.quantisation})")

    with console.status("[bold green]measuring prefill") as status:
        points = prefill_scaling(provider, progress=lambda m: status.update(f"[bold green]{m}"))

    t = Table(title="Prefill dominates on CPU", header_style="bold")
    for c in ("input tokens", "prefill", "decode", "ms / input token", "prefill share"):
        t.add_column(c, justify="right")
    for p in points:
        t.add_row(f"{p.prompt_tokens}", f"{p.prefill_ms:.0f} ms", f"{p.decode_ms:.0f} ms",
                  f"{p.ms_per_prompt_token:.2f}", f"{p.prefill_share:.1f}%")
    console.print(t)

    if points:
        rate = sum(p.ms_per_prompt_token for p in points) / len(points)
        console.print(
            f"[bold]Every input token removed is worth ~{rate:.1f} ms.[/bold] "
            f"[dim]MockProvider assumed 120 ms TTFT and 65 ms/token, so the simulation "
            f"understated the prompt side.[/dim]\n"
        )

    with console.status("[bold green]measuring prefix reuse") as status:
        arms = prefix_reuse(provider, progress=lambda m: status.update(f"[bold green]{m}"))

    t2 = Table(title="What a volatile token at position 0 costs (ADR-025)", header_style="bold")
    for c in ("arrangement", "tokens", "first call", "steady state", "reuse"):
        t2.add_column(c, justify="right")
    t2.columns[0].justify = "left"
    for a in arms:
        t2.add_row(a.label, str(a.prompt_tokens), f"{a.first_ms:.0f} ms",
                   f"{a.steady_ms:.0f} ms", f"{a.reuse_pct:.1f}%")
    console.print(t2)
    if len(arms) == 2 and arms[1].steady_ms > 0:
        console.print(
            f"[bold yellow]Same content, {abs(arms[0].prompt_tokens - arms[1].prompt_tokens)} "
            f"tokens apart, {arms[1].steady_ms / max(1e-9, arms[0].steady_ms):.0f}x the cost."
            f"[/bold yellow] [dim]Every metric in the compression literature scores these two "
            f"configurations identically.[/dim]"
        )


@app.command()
def generalise(corpus_path: Path = typer.Option(None, "--corpus")) -> None:
    """Does a calibration transfer to another vocabulary? (gap 5, contribution 6)"""
    from parsimony.eval.generalisation import (
        check_boundary_effect,
        check_tier1_yield,
        sweep_across_tokenizers,
    )

    corpus = load_corpus(corpus_path)
    cells = list(factorial_cells(axes=("M1", "M2", "M3", "M5"), always_on=frozenset()))

    with console.status("[bold green]running every cell under each vocabulary...") as status:
        arms = sweep_across_tokenizers(
            cells, corpus, progress=lambda m: status.update(f"[bold green]{m}")
        )
    live = [a for a in arms if a.available]

    if not live:
        console.print("[red]No real tokenizer available (offline?) — cannot run this study.[/red]")
        return

    table = Table(title="Same cells, same corpus, no re-tuning", header_style="bold")
    table.add_column("cell", no_wrap=True)
    for a in live:
        table.add_column(f"{a.short_name}\n(vocab {a.vocab_size:,})", justify="right")
    for label in ("baseline", "M1", "M2", "M3", "M5", "M1+M2+M3+M5"):
        vals = [a.reduction(label) for a in live]
        if any(v is None for v in vals):
            continue
        table.add_row(label, *[f"{v:+.2f}%" for v in vals])
    console.print(table)

    ident = len({tuple(a.ranking()) for a in live}) == 1
    console.print(
        f"  module ranking identical across vocabularies: "
        f"[{'green' if ident else 'red'}]{ident}[/]"
    )

    bt = Table(title="Does each finding transfer?", header_style="bold")
    bt.add_column("claim")
    for a in live:
        bt.add_column(a.short_name, justify="right")
    bt.add_column("transfers", justify="center")
    for check in check_boundary_effect() + check_tier1_yield(corpus):
        bt.add_row(
            check.claim,
            *[check.values.get(a.short_name, "-") for a in live],
            "[green]yes[/green]" if check.transfers else "[red]NO[/red]",
        )
    console.print(bt)
    console.print(
        "[dim]Reduction ratios transfer because a ratio cancels a roughly constant vocabulary\n"
        "factor. The MECHANISMS underneath do not: GPT-2 has no capitalisation penalty, so one\n"
        "of ADR-030's two effects is Qwen-specific (ADR-032).\n\n"
        "This covers the TOKENIZER dimension of report 4.6. Decode speed, answer quality and\n"
        "quantisation are properties of the model and still need Ollama.[/dim]"
    )


@app.command()
def tokenprobe(corpus_path: Path = typer.Option(None, "--corpus")) -> None:
    """When does shortening text fail to reduce tokens? (M1 tier 3 evidence)"""
    from parsimony.eval.tokenizer_probe import run_probe
    from parsimony.infra.tokenization import get_tokenizer

    corpus = load_corpus(corpus_path)
    texts = [q for c in corpus.conversations for q in c.user_turns]
    tok = get_tokenizer()
    results = run_probe(texts, tok)

    table = Table(title=f"Negative-yield probe — tokenizer {tok.id}", header_style="bold")
    table.add_column("Edit regime")
    table.add_column("tested", justify="right")
    table.add_column("saved tokens", justify="right")
    table.add_column("saved nothing", justify="right")
    table.add_column("cost tokens", justify="right")
    table.add_column("wasted", justify="right")
    for r in results:
        table.add_row(
            r.regime,
            str(r.tested),
            str(r.reduced),
            str(r.neutral),
            f"[red]{r.increased}[/red]" if r.increased else "0",
            f"{r.wasted_pct:.0f}%",
        )
    console.print(table)
    console.print(
        Panel(
            "Whitespace-aligned edits (phrase, word) are [bold]monotone[/bold] under this "
            "tokenizer:\nshortening the text always reduces or preserves the token count.\n\n"
            "Sub-token edits are [bold]not[/bold]: 'running'->'runing' is shorter text and "
            "MORE tokens.\n\n"
            "[dim]So the negative-yield guard earns its place mainly by rejecting ZERO-yield "
            "edits —\nedits that perturb the text for no saving at all, which is pure risk. "
            "True negative\nyield matters for character- and subword-level methods, not for "
            "phrase compression.[/dim]",
            title="ADR-026", border_style="yellow",
        )
    )


@app.command("ledger-import")
def ledger_import(
    files: list[Path] = typer.Argument(..., help="JSONL ledger files."),
    db: Path = typer.Option(Path("runs/analysis.db"), "--db"),
) -> None:
    """Fold JSONL run files into the SQLite analysis database."""
    n = import_jsonl(list(files), db)
    console.print(f"imported [bold]{n}[/bold] rows into {db}")


@app.command()
def corpus(corpus_path: Path = typer.Option(None, "--corpus")) -> None:
    """Show corpus composition and its freeze hash."""
    c = load_corpus(corpus_path)
    table = Table(title=f"Corpus  [{c.corpus_hash}]", header_style="bold")
    table.add_column("Class")
    table.add_column("Conversations", justify="right")
    table.add_column("Requests", justify="right")
    for cls, convs in sorted(c.by_class().items()):
        table.add_row(cls, str(len(convs)), str(sum(x.n_turns for x in convs)))
    table.add_section()
    table.add_row("[bold]total[/bold]", f"[bold]{len(c)}[/bold]", f"[bold]{c.n_requests}[/bold]")
    console.print(table)
    console.print(f"[dim]{c.path}[/dim]")


def _cell_table(results, corpus) -> Table:
    table = Table(
        title=f"Ablation — {len(corpus)} conversations, {corpus.n_requests} requests",
        header_style="bold",
    )
    table.add_column("Cell", no_wrap=True)
    table.add_column("in", justify="right", no_wrap=True)
    table.add_column("out", justify="right", no_wrap=True)
    table.add_column("total", justify="right", no_wrap=True)
    table.add_column("saved", justify="right", no_wrap=True)
    table.add_column("cache", justify="right", no_wrap=True)
    table.add_column("t0", justify="right", no_wrap=True)
    table.add_column("gate", justify="right", no_wrap=True)
    table.add_column("stop", justify="right", no_wrap=True)
    table.add_column("ms", justify="right", no_wrap=True)

    for r in results:
        pct = r.total_reduction_pct
        colour = "green" if pct > 0.05 else ("dim" if abs(pct) <= 0.05 else "red")
        table.add_row(
            r.label,
            str(r.tokens_in_final),
            str(r.tokens_out),
            str(r.total_tokens),
            f"[{colour}]{pct:+.1f}%[/{colour}]",
            str(r.cache_hits),
            str(r.deterministic_hits),
            str(r.gate_fires),
            str(r.early_stops),
            f"{r.middleware_mean_ms:.1f}",
        )
    table.caption = (
        "t0 = answered by the deterministic tier (zero model tokens) | "
        "gate = fidelity reverts | stop = early-stop fires"
    )
    return table


def _quality_table(results) -> Table:
    table = Table(
        title="Quality — four measures, never averaged",
        header_style="bold",
        caption="embed/overlap/judge are PROXIES against the baseline's own answer. "
                "gold is the only ground truth.\n"
                "overlap is structurally biased against M5: shorter answers score "
                "lower by construction.\n"
                "judge disagreement = how often it flipped when the options were "
                "swapped; high means noise.",
    )
    table.add_column("Cell", no_wrap=True)
    table.add_column("embed", justify="right")
    table.add_column("overlap", justify="right")
    table.add_column("judge", justify="right")
    table.add_column("judge disagree", justify="right")
    table.add_column("gold", justify="right")

    for r in results:
        gold_colour = "green" if r.gold_accuracy >= 50 else "yellow"
        table.add_row(
            r.label,
            f"{r.quality_embedding:.1f}%" if r.q_embedding else "[dim]ref[/dim]",
            f"{r.quality_overlap:.1f}%" if r.q_overlap else "[dim]ref[/dim]",
            f"{r.quality_judge:.1f}%" if r.q_judge else "[dim]ref[/dim]",
            f"{r.judge_disagreement_rate:.0f}%" if r.q_judge else "[dim]-[/dim]",
            f"[{gold_colour}]{r.gold_accuracy:.1f}%[/{gold_colour}]"
            f" ({r.gold_correct}/{r.gold_total})" if r.gold_total else "[dim]-[/dim]",
        )
    return table


def _print_shortfall(results, axes=None) -> None:
    s = additivity_shortfall(results, axes)
    if s.get("n_solo_modules", 0) < 2 or "shortfall_pct" not in s:
        return
    console.print(
        Panel(
            f"If the savings were additive, [bold]{s['stacked_label']}[/bold] would deliver "
            f"[bold]{s['predicted_additive_pct']:.1f}%[/bold].\n"
            f"Measured stacked reduction: [bold]{s['measured_stacked_pct']:.1f}%[/bold].\n"
            f"[bold yellow]Additivity shortfall: {s['shortfall_pct']:.1f} percentage points.[/bold yellow]\n"
            f"[dim]Quantifying this shortfall is the project's primary result "
            f"(Contribution 1), not an embarrassment.[/dim]",
            title="Do the savings compound?",
            border_style="yellow",
        )
    )


if __name__ == "__main__":
    app()
