"""Trace rendering.

The trace table is the demo artefact: it is how a reviewer sees that the
pipeline is a measurement instrument rather than a black box. Every stage
appears, including ones that were skipped, did nothing, or are scheduled for a
later sprint — an invisible stage is an unauditable one.
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from parsimony.core.ledger import StageOutcome

_STYLE = {
    StageOutcome.APPLIED: "green",
    StageOutcome.REVERTED: "yellow",
    StageOutcome.SHORT_CIRCUIT: "bold cyan",
    StageOutcome.NOOP: "dim",
    StageOutcome.SKIPPED: "dim",
    StageOutcome.ERROR: "bold red",
    StageOutcome.NOT_IMPLEMENTED: "dim italic",
}

_LABEL = {
    StageOutcome.APPLIED: "applied",
    StageOutcome.REVERTED: "REVERTED",
    StageOutcome.SHORT_CIRCUIT: "SHORT-CIRCUIT",
    StageOutcome.NOOP: "no-op",
    StageOutcome.SKIPPED: "skipped",
    StageOutcome.ERROR: "ERROR",
    StageOutcome.NOT_IMPLEMENTED: "sprint 3+",
}


def trace_table(outcome) -> Table:
    table = Table(title="Pipeline trace", title_style="bold", header_style="bold")
    table.add_column("Stage", no_wrap=True)
    table.add_column("Mod", justify="center", no_wrap=True)
    table.add_column("Outcome", no_wrap=True)
    table.add_column("Tokens", justify="right", no_wrap=True)
    table.add_column("Δ", justify="right", no_wrap=True)
    table.add_column("µs", justify="right", no_wrap=True)
    table.add_column("Detail")

    for t in outcome.traces:
        style = _STYLE[t.outcome]
        delta = t.tokens_after - t.tokens_before
        if t.outcome is StageOutcome.SHORT_CIRCUIT:
            tokens, delta_text = f"{t.tokens_before} -> 0", Text("bypass", style="bold cyan")
        elif delta == 0:
            tokens, delta_text = str(t.tokens_before), Text("-", style="dim")
        else:
            tokens = f"{t.tokens_before} -> {t.tokens_after}"
            delta_text = Text(f"{delta:+d}", style="green" if delta < 0 else "red")

        detail = t.rationale
        for ev in t.gate_events:
            detail += f"  [{ev.invariant_class}: {', '.join(ev.lost_values)}]"

        table.add_row(
            t.name,
            t.module_id,
            Text(_LABEL[t.outcome], style=style),
            tokens,
            delta_text,
            f"{t.duration_ns / 1000:.0f}",
            Text(detail, style=style if t.outcome in (StageOutcome.REVERTED, StageOutcome.ERROR) else ""),
        )
    return table


def summary_panel(outcome, simulated: bool = True) -> Panel:
    """Per-request summary.

    Deliberately reports the INPUT reduction only. A 'total reduction' figure
    would need the baseline's output length for this same query, which a single
    request cannot know — total reduction is a between-cell comparison and lives
    in the ablation table (docs/05-evaluation-harness.md). Showing one here would
    silently compare input-before against input+output-after.
    """
    row = outcome.row
    in_before, in_after = row.tokens_in_original, row.tokens_in_final
    saved_pct = ((in_before - in_after) / in_before * 100) if in_before else 0.0

    lines = [
        f"[bold]served by[/bold]      {row.route_tier}"
        + ("  [bold cyan](no model tokens at all)[/bold cyan]" if not outcome.generated else ""),
    ]

    if outcome.generated:
        lines.append(
            f"[bold]input tokens[/bold]   {in_before} -> {in_after}"
            + (f"   [green]{saved_pct:.0f}% fewer[/green]" if saved_pct > 0 else "")
        )
        budget = f" of {row.tokens_out_budget} budgeted" if row.tokens_out_budget else ""
        stopped = "  [green](early-stopped)[/green]" if row.early_stopped else ""
        lines.append(f"[bold]output tokens[/bold]  {row.tokens_out}{budget}{stopped}")
    else:
        lines.append(
            f"[bold]prompt avoided[/bold] {in_before} input tokens never sent"
        )
        lines.append("[bold]output tokens[/bold]  0   [green](nothing generated)[/green]")

    lines.append(
        f"[bold]middleware[/bold]     {row.middleware_ns / 1e6:.1f} ms"
        + ("  [green](budget 120 ms)[/green]" if row.middleware_ns < 120e6
           else "  [red](over 120 ms budget)[/red]")
    )
    if row.ttft_ns is not None:
        lines.append(
            f"[bold]TTFT / TPOT[/bold]    {row.ttft_ns / 1e6:.0f} ms / "
            f"{(row.tpot_ns or 0) / 1e6:.0f} ms per token"
            + ("  [yellow](simulated)[/yellow]" if simulated else "")
        )
    if row.prefix_tokens_survived is not None and row.prefix_tokens_survived > 0:
        lines.append(
            f"[bold]KV prefix[/bold]      {row.prefix_tokens_survived} tokens survived "
            f"({row.prefix_ratio:.0%} of prompt)"
        )
    if row.gate_fired:
        n = len(row.gate_events)
        lines.append(f"[bold yellow]fidelity gate[/bold yellow]  FIRED - {n} transformation(s) reverted")
    lines.append(f"[dim]config {row.config_hash}  |  model {row.model_name} ({row.model_digest})[/dim]")

    return Panel("\n".join(lines), title="Summary", border_style="blue")


def print_outcome(console: Console, outcome, show_response: bool = True) -> None:
    console.print(trace_table(outcome))
    console.print(summary_panel(outcome, simulated=outcome.row.model_digest.startswith("mock")))
    if show_response:
        console.print(Panel(outcome.response or "[dim](empty)[/dim]", title="Response",
                            border_style="green"))
