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
from parsimony.eval.corpus import load_corpus, load_gold
from parsimony.eval.metrics import LengthBiasedMockJudge
from parsimony.eval.runner import additivity_shortfall, run_cell, sweep
from parsimony.infra.ids import ulid
from parsimony.infra.providers import make_provider
from parsimony.infra.storage import JsonlSink, import_jsonl
from parsimony.infra.tokenization import get_tokenizer
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
    text: bool = typer.Option(False, "--text/--no-text", "-t",
                              help="Show the text each stage changed, not just the token count."),
    provider: str = typer.Option("mock", "--provider",
                                 help="'mock' (simulated timings) or 'ollama' (a real model)."),
    model: str = typer.Option(None, "--model", help="Ollama model tag."),
) -> None:
    """Run one query through the pipeline."""
    cfg = baseline() if plain else full_stack()
    pipeline = Pipeline(cfg, provider=make_provider(provider, model=model), capture_text=text)
    counter = get_tokenizer(cfg.tokenizer_id).count

    def show(o):
        if trace:
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
