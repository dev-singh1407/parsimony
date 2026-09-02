"""Trace rendering.

The trace table is the demo artefact: it is how a reviewer sees that the
pipeline is a measurement instrument rather than a black box. Every stage
appears, including ones that were skipped, did nothing, or are scheduled for a
later sprint — an invisible stage is an unauditable one.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from rich.console import Console, Group
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


_MAX_SHOWN_CHARS = 700


def _split_words(text: str) -> list[str]:
    """Split into words with their surrounding whitespace attached.

    Lossless: "".join(_split_words(s)) == s for every s. The second branch
    exists for runs with no word in them at all — a bare "\\n\\n" between turns
    is content the panel must reproduce, and a word-only pattern drops it.
    """
    return re.findall(r"\s*\S+\s*|\s+", text)


def _diff_text(before: str, after: str, *, show: str) -> Text:
    """One side of a word-level diff.

    `show="before"` marks deleted words; `show="after"` marks inserted ones.
    Unchanged words are rendered plainly so the surviving meaning is what the
    eye lands on first.
    """
    a, b = _split_words(before), _split_words(after)
    out = Text()
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, a, b).get_opcodes():
        if tag == "equal":
            out.append("".join(a[i1:i2]))
        elif show == "before" and tag in ("delete", "replace"):
            out.append("".join(a[i1:i2]), style="red strike")
        elif show == "after" and tag in ("insert", "replace"):
            out.append("".join(b[j1:j2]), style="bold green")
    return out


def _elide(text: Text) -> Text:
    """A projector has finite height; a 9-turn history does not."""
    if len(text.plain) <= _MAX_SHOWN_CHARS:
        return text
    clipped = text[:_MAX_SHOWN_CHARS]
    clipped.append(f"\n[... {len(text.plain) - _MAX_SHOWN_CHARS} more characters]", style="dim")
    return clipped


def text_delta_panels(console: Console, outcome, counter=None) -> None:
    """Show what each stage did to the text itself.

    The trace table proves a stage removed 19 tokens. This shows *which* 19,
    which is the difference between a reviewer trusting the number and checking
    it.
    """
    counter = counter or (lambda s: None)
    for d in outcome.text_deltas:
        if not d.changed and not d.short_circuited:
            continue

        if d.short_circuited:
            console.print(
                Panel(
                    _elide(Text(d.before)),
                    title=f"{d.stage} [{d.module_id}] — served without a model; "
                          f"this prompt was never sent",
                    border_style="cyan",
                )
            )
            continue

        n_before, n_after = counter(d.before), counter(d.after)
        if d.reverted:
            head = (f"{d.stage} [{d.module_id}] — [bold yellow]edit REVERTED "
                    f"by the fidelity gate[/bold yellow]")
            sub_before, sub_after = "kept (what the model sees)", "refused (what it would have lost)"
            border = "yellow"
        else:
            head = f"{d.stage} [{d.module_id}]"
            sub_before, sub_after = "before", "after"
            border = "green"

        def _label(name: str, n: int | None) -> str:
            return f"{name}  —  {n} tokens" if n is not None else name

        body = Group(
            Panel(_elide(_diff_text(d.before, d.after, show="before")),
                  title=_label(sub_before, n_before), border_style="dim", title_align="left"),
            Panel(_elide(_diff_text(d.before, d.after, show="after")),
                  title=_label(sub_after, n_after), border_style="dim", title_align="left"),
        )
        console.print(Panel(body, title=head, border_style=border, title_align="left"))


def print_outcome(console: Console, outcome, show_response: bool = True,
                  show_text: bool = False, counter=None) -> None:
    console.print(trace_table(outcome))
    if show_text:
        text_delta_panels(console, outcome, counter)
    console.print(summary_panel(outcome, simulated=outcome.row.model_digest.startswith("mock")))
    if show_response:
        console.print(Panel(outcome.response or "[dim](empty)[/dim]", title="Response",
                            border_style="green"))
