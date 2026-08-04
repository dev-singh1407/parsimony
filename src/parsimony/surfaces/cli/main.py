"""Parsimony CLI.

`parsimony demo` is the review artefact: a scripted sequence that shows the
pipeline is a measurement instrument, not a black box.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from parsimony.core.config import baseline, factorial_cells, full_stack, with_cache_lookup
from parsimony.core.types import Mode
from parsimony.eval.corpus import load_corpus
from parsimony.eval.runner import additivity_shortfall, run_cell, summarise, sweep
from parsimony.infra.ids import ulid
from parsimony.infra.storage import JsonlSink, import_jsonl
from parsimony.pipeline.orchestrator import Pipeline
from parsimony.surfaces.cli.render import print_outcome, trace_table

app = typer.Typer(add_completion=False, help="Token-efficient LLM middleware for CPU-only hardware.")
console = Console()


@app.command()
def chat(
    query: str = typer.Argument(..., help="The query to send."),
    trace: bool = typer.Option(True, "--trace/--no-trace", help="Show the per-stage trace."),
    plain: bool = typer.Option(False, "--baseline", help="Run with every module disabled."),
    repeat: bool = typer.Option(False, "--repeat", help="Send it twice to exercise the cache."),
) -> None:
    """Run one query through the pipeline."""
    cfg = baseline() if plain else full_stack()
    pipeline = Pipeline(cfg)

    outcome = pipeline.run(query)
    if trace:
        print_outcome(console, outcome)
    else:
        console.print(outcome.response)

    if repeat:
        console.print(Rule("second identical request"))
        again = pipeline.run(query)
        if trace:
            print_outcome(console, again)
        else:
            console.print(again.response)


@app.command()
def demo() -> None:
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

    pipeline = Pipeline(full_stack())

    console.print(Rule("[bold]1. A query no model should ever see[/bold]"))
    console.print("[dim]The routing literature always escalates to a model. The cheapest tier "
                  "answers without one.[/dim]\n")
    print_outcome(console, pipeline.run("What is 847 * 23?"))

    console.print(Rule("[bold]2. Boilerplate is tokens you paid for[/bold]"))
    verbose = (
        "Hello, I was wondering if you could **please** explain to me what photosynthesis is? "
        "I would like to know how it works. Thanks in advance!"
    )
    console.print(f"[dim]{verbose}[/dim]\n")
    print_outcome(console, pipeline.run(verbose))

    console.print(Rule("[bold]3. The same question, asked again[/bold]"))
    print_outcome(console, pipeline.run(verbose))

    console.print(Rule("[bold]4. The fidelity gate refusing a saving[/bold]"))
    console.print(
        "[dim]Two near-duplicate sentences, but one carries a number the other does not. "
        "The compressor tries to drop it; the gate reverts the edit.[/dim]\n"
    )
    print_outcome(
        console,
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
) -> None:
    """Run the factorial ablation over the corpus and write a ledger."""
    axes = tuple(m.strip().upper() for m in modules.split(",") if m.strip())
    corpus = load_corpus(corpus_path)
    run_id = ulid()

    console.print(
        f"[bold]corpus[/bold] {len(corpus)} conversations, {corpus.n_requests} requests "
        f"| hash [cyan]{corpus.corpus_hash}[/cyan]"
    )
    console.print(f"[bold]cells[/bold]  2^{len(axes)} = {2 ** len(axes)} over {', '.join(axes)}")

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
        results = []
        with console.status("[bold green]running cells...") as status:
            for cfg in cells:
                status.update(f"[bold green]cell: {cfg.label}")
                # EXPERIMENT mode: a failed ledger write is fatal here, because
                # in a sweep the ledger IS the result (ADR-005).
                results.append(
                    run_cell(
                        replace(cfg, mode=Mode.EXPERIMENT),
                        corpus,
                        sink=sink,
                        run_id=run_id,
                    )
                )
        results = summarise(results)
    finally:
        if sink is not None:
            sink.close()

    console.print(_cell_table(results, corpus))
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
    verdict = (
        f"Compressing before the lookup changed the hit count by [bold]{delta:+d}[/bold] "
        f"over {corpus.n_requests} requests."
    )
    console.print(
        Panel(
            f"{verdict}\n"
            "[dim]Normalisation collapses politeness-only paraphrases onto the same key, so more\n"
            "queries hit. The open question the literature leaves unanswered is whether the same\n"
            "collapsing also merges questions that should NOT match — measured against the\n"
            "adversarial pair subset in Sprint 2, once the semantic tier exists.[/dim]",
            title="Gap 3", border_style="magenta",
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
