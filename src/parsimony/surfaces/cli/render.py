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


#: What a removed space is drawn as, so a change that saves a real token is
#: not rendered as two identical-looking panels.
_VISIBLE_SPACE = "·"
_VISIBLE_NEWLINE = "⏎"


def _reveal(fragment: str) -> str:
    """Draw whitespace so it can be seen when it is the thing that changed."""
    return (
        fragment.replace("\n", _VISIBLE_NEWLINE)
        .replace("\t", "→")
        .replace(" ", _VISIBLE_SPACE)
    )


def _diff_text(before: str, after: str, *, show: str) -> Text:
    """One side of a word-level diff.

    `show="before"` marks deleted words; `show="after"` marks inserted ones.
    Unchanged words are rendered plainly so the surviving meaning is what the
    eye lands on first.

    Whitespace-only changes are drawn with a visible middle dot. Collapsing a
    double space or dropping a trailing one removes a real token, and without
    this the panel reported "before 9 tokens / after 8 tokens" above two
    strings that looked character-for-character identical — a display that
    contradicts itself in front of whoever is being shown it.
    """
    a, b = _split_words(before), _split_words(after)
    out = Text()
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, a, b).get_opcodes():
        old, new = "".join(a[i1:i2]), "".join(b[j1:j2])
        if tag == "equal":
            out.append(old)
            continue

        mine, style = (
            (old, "red strike") if show == "before" else (new, "bold green")
        )
        if not mine:
            continue
        # Reveal the whitespace when whitespace is the whole of the change —
        # either the fragment is pure whitespace, or the two sides differ only
        # in it ("friend " against "friend", which is a word replacement to the
        # differ but an invisible one to the reader).
        invisible = not mine.strip() or old.strip() == new.strip()
        out.append(_reveal(mine) if invisible else mine, style=style)
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
            if d.before.strip() == d.after.strip():
                # Says out loud what the middle dots show, for the case where
                # the only difference is whitespace and a reader could
                # reasonably conclude the panel is broken.
                head += "  [dim](whitespace only — · marks a removed space)[/dim]"

        def _label(name: str, n: int | None) -> str:
            return f"{name}  —  {n} tokens" if n is not None else name

        body = Group(
            Panel(_elide(_diff_text(d.before, d.after, show="before")),
                  title=_label(sub_before, n_before), border_style="dim", title_align="left"),
            Panel(_elide(_diff_text(d.before, d.after, show="after")),
                  title=_label(sub_after, n_after), border_style="dim", title_align="left"),
        )
        console.print(Panel(body, title=head, border_style=border, title_align="left"))


#: What each module is for, in one line, for the per-module report.
_MODULE_ROLE = {
    "m6a_deterministic": ("M6", "answer without a model if the question allows it"),
    "m2_cache":          ("M2", "reuse an earlier answer to an equivalent question"),
    "m2_cache_probe":    ("M2", "record what the cache would have matched"),
    "m3_history":        ("M3", "keep only the conversation turns that matter"),
    "m3_arrange":        ("M3", "order the kept turns for the model's attention"),
    "m1_tier1":          ("M1", "strip boilerplate and politeness, losslessly"),
    "m1_tier2":          ("M1", "drop sentences that repeat a fact already stated"),
    "m1_tier3":          ("M1", "shorten wording where it costs nothing to do so"),
    "m4_assembler":      ("M4", "put invariant text first so the model can reuse its work"),
    "m5_budgeter":       ("M5", "cap the answer length to suit the question type"),
    "m6b_router":        ("M6", "decide whether a bigger model is needed"),
}


def module_report(console: Console, outcome, counter=None) -> None:
    """One panel per module, for every request — including the ones that did nothing.

    The text-delta panels alone only appear for stages that rewrote the
    payload, so a typical query produced two panels out of eleven stages and
    said nothing about the rest. A reader cannot tell "this module was not
    needed here" from "this module does not exist", and both look the same:
    absent.

    So every stage reports. A stage that rewrote text shows before and after; a
    stage that made a decision without touching text says which decision; a
    stage that did nothing says why not.
    """
    counter = counter or (lambda s: None)
    deltas = {d.stage: d for d in getattr(outcome, "text_deltas", ())}

    for trace in outcome.traces:
        module, role = _MODULE_ROLE.get(trace.name, (trace.module_id, ""))
        delta = deltas.get(trace.name)
        outcome_name = trace.outcome

        if outcome_name is StageOutcome.SHORT_CIRCUIT:
            console.print(Panel(
                f"[bold cyan]Answered here. The model was never called.[/bold cyan]\n"
                f"[dim]{trace.rationale}[/dim]",
                title=f"[bold]{module}[/bold] · {trace.name} — {role}",
                border_style="cyan", title_align="left"))
            continue

        if outcome_name is StageOutcome.REVERTED and delta is not None:
            console.print(Panel(
                Group(
                    Panel(_elide(_diff_text(delta.before, delta.after, show="before")),
                          title=f"kept — {counter(delta.before)} tokens",
                          border_style="dim", title_align="left"),
                    Panel(_elide(_diff_text(delta.before, delta.after, show="after")),
                          title=f"REFUSED — would have been {counter(delta.after)} tokens",
                          border_style="dim", title_align="left"),
                    Text(f"\n{trace.rationale}", style="yellow"),
                ),
                title=f"[bold]{module}[/bold] · {trace.name} — "
                      f"[bold yellow]edit blocked by the fidelity gate[/bold yellow]",
                border_style="yellow", title_align="left"))
            continue

        if delta is not None and delta.changed:
            saved = trace.tokens_before - trace.tokens_after
            head = (f"[bold]{module}[/bold] · {trace.name} — {role}"
                    + (f"   [bold green]−{saved} tokens[/bold green]" if saved > 0 else ""))

            if delta.history_changed:
                # The change is in the conversation behind the question, not in
                # the question. Showing the whole payload here buried a 24-token
                # question under several hundred tokens of a previous ANSWER —
                # the user could not see what happened to what they typed.
                dropped = delta.turns_before - delta.turns_after
                summary = (f"[bold]Your question was not changed.[/bold]\n"
                           f"[dim]{trace.rationale}[/dim]")
                if dropped > 0:
                    summary = (f"[bold]Trimmed the conversation: {delta.turns_before} → "
                               f"{delta.turns_after} turns ({dropped} dropped).[/bold]\n"
                               f"[dim]Your question is untouched. {trace.rationale}[/dim]")
                console.print(Panel(summary, title=head + "  [dim](conversation, not "
                                    "your question)[/dim]",
                                    border_style="green", title_align="left"))
                continue

            b, a = delta.query_before, delta.query_after
            before_n, after_n = counter(b), counter(a)
            note = ("  [dim](whitespace only — · marks a removed space)[/dim]"
                    if b.strip() == a.strip() else "")
            console.print(Panel(
                Group(
                    Panel(_elide(_diff_text(b, a, show="before")),
                          title=f"your question before — {before_n} tokens",
                          border_style="dim", title_align="left"),
                    Panel(_elide(_diff_text(b, a, show="after")),
                          title=f"after — {after_n} tokens",
                          border_style="dim", title_align="left"),
                ),
                title=head + note, border_style="green", title_align="left"))
            continue

        # Changed something that is not the text: a budget, a route, an ordering.
        if outcome_name is StageOutcome.APPLIED:
            console.print(Panel(
                f"[green]{trace.rationale}[/green]\n"
                f"[dim]Changes how the request is handled, not the text itself.[/dim]",
                title=f"[bold]{module}[/bold] · {trace.name} — {role}",
                border_style="green", title_align="left"))
            continue

        # Did nothing, and the reason is the interesting part.
        console.print(Panel(
            f"[dim]{trace.rationale or _LABEL[outcome_name]}[/dim]",
            title=f"[bold]{module}[/bold] · {trace.name} — "
                  f"[dim]{_LABEL[outcome_name]}[/dim]",
            border_style="grey37", title_align="left"))


def _changed_runs(before: str, after: str) -> tuple[list[str], list[str]]:
    """Contiguous runs this edit removed and added.

    RUNS, not a flat word list. Flattening non-adjacent fragments into one
    string invents phrases that were never in the text: deleting "Hello, could
    you please" and "Thanks!" from opposite ends of a sentence reported
    `removed "Hello, could you please explain is? Thanks!"` — which implies
    "explain" was removed when it was kept.
    """
    a, b = _split_words(before), _split_words(after)
    removed, added = [], []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, a, b).get_opcodes():
        if tag in ("delete", "replace"):
            run = "".join(a[i1:i2]).strip()
            if run:
                removed.append(run)
        if tag in ("insert", "replace"):
            run = "".join(b[j1:j2]).strip()
            if run:
                added.append(run)
    return removed, added


def _vanished(before: str, after: str) -> list[str]:
    """Runs that genuinely left the text, not ones the differ merely realigned.

    A raw removed-run list reported that "summarise" was deleted and "Summarise"
    kept — true to the differ, and nonsense to a reader, because the word is
    still there. Anything whose normalised form survives in the after text is
    dropped from the list.
    """
    surviving = set(re.findall(r"[a-z0-9']+", after.lower()))

    def word(tok: str) -> str:
        m = re.findall(r"[a-z0-9']+", tok.lower())
        return m[0] if m else ""

    out = []
    for run in _changed_runs(before, after)[0]:
        tokens = run.split()
        # Trim words that are still in the text off both ends. The differ aligns
        # "…please summarise" against "Summarise", so the run arrives carrying a
        # word that never left; reporting it as deleted is simply wrong.
        while tokens and word(tokens[0]) in surviving:
            tokens.pop(0)
        while tokens and word(tokens[-1]) in surviving:
            tokens.pop()
        if tokens:
            out.append(" ".join(tokens))
    return out


def _quote(runs: list[str], limit: int = 4) -> str:
    """Each run quoted separately, so nothing is implied to be contiguous."""
    if not runs:
        return "—"
    shown = " · ".join(f'"{r}"' for r in runs[:limit])
    return shown + (f" · … {len(runs) - limit} more" if len(runs) > limit else "")


def explain_report(console: Console, outcome, counter=None) -> None:
    """A written account of what was done to this request, and why.

    The per-module panels show WHAT changed. This says what it MEANS: which
    modules acted, what each one removed or decided, the reasoning behind it,
    and what it cost or saved. Without it a viewer sees a wall of before/after
    boxes and has to reconstruct the argument themselves — which, in a demo, is
    the presenter's job and not the audience's.

    Only the modules that acted appear. Silence is reported once, as a count,
    rather than as nine panels saying nothing happened.
    """
    counter = counter or (lambda s: 0)
    deltas = {d.stage: d for d in getattr(outcome, "text_deltas", ())}
    row = outcome.row

    acted, quiet = [], []
    for trace in outcome.traces:
        module, role = _MODULE_ROLE.get(trace.name, (trace.module_id, ""))
        if trace.outcome in (StageOutcome.APPLIED, StageOutcome.REVERTED,
                             StageOutcome.SHORT_CIRCUIT):
            acted.append((trace, module, role, deltas.get(trace.name)))
        else:
            quiet.append((trace, module))

    table = Table(
        title="What happened to this request, and why",
        title_style="bold", header_style="bold", show_lines=True, expand=True,
    )
    table.add_column("Module", style="bold cyan", no_wrap=True)
    table.add_column("What it did")
    table.add_column("Effect", justify="right", no_wrap=True)

    for trace, module, role, delta in acted:
        if trace.outcome is StageOutcome.SHORT_CIRCUIT:
            table.add_row(
                f"{module}\n[dim]{trace.name}[/dim]",
                f"[bold cyan]Answered the question here — the model was never called.[/bold cyan]\n"
                f"[dim]{trace.rationale}[/dim]",
                f"[bold green]{trace.tokens_before} tokens\nnever sent[/bold green]",
            )
            continue

        if trace.outcome is StageOutcome.REVERTED:
            lost = _changed_runs(delta.before, delta.after)[0] if delta else []
            table.add_row(
                f"{module}\n[dim]{trace.name}[/dim]",
                f"[bold yellow]Proposed an edit; the fidelity gate REFUSED it.[/bold yellow]\n"
                f"Would have removed {_quote(lost)}\n"
                f"[dim]Blocked because: {trace.rationale}[/dim]",
                "[yellow]0 tokens\n(safety first)[/yellow]",
            )
            continue

        saved = trace.tokens_before - trace.tokens_after
        if delta is not None and delta.history_changed:
            # Quoting the payload here quoted a previous ANSWER: a question
            # about Ubuntu reported removing "**Choose" and "Pivot**:", which
            # came from a quicksort reply three turns earlier. Describe the
            # conversation change structurally instead.
            dropped = delta.turns_before - delta.turns_after
            what = (f"[dim]{role.capitalize()}.[/dim]\n"
                    + (f"Trimmed the conversation: {delta.turns_before} → "
                       f"{delta.turns_after} turns.\n" if dropped > 0 else "")
                    + "[dim]Your question itself was not changed. "
                    + f"{trace.rationale}[/dim]")
        elif delta is not None and delta.query_changed:
            removed, added = _changed_runs(delta.query_before, delta.query_after)
            what = f"[dim]{role.capitalize()}.[/dim]\nRemoved {_quote(removed)}"
            if added:
                what += f"\nReplaced with {_quote(added)}"
            what += f"\n[dim]{trace.rationale}[/dim]"
        else:
            what = f"[dim]{role.capitalize()}.[/dim]\n{trace.rationale}"

        effect = (f"[bold green]−{saved} tokens[/bold green]" if saved > 0
                  else "[dim]no token change[/dim]")
        table.add_row(f"{module}\n[dim]{trace.name}[/dim]", what, effect)

    console.print(table)

    if quiet:
        names = ", ".join(f"{m} {t.name.split('_', 1)[-1]}" for t, m in quiet)
        console.print(
            f"[dim]Not needed here ({len(quiet)}): {names}. "
            f"A module that reports doing nothing is as auditable as one that acts — "
            f"the trace never hides a stage.[/dim]"
        )

    before, after = row.tokens_in_original, row.tokens_in_final
    if outcome.generated and before:
        pct = (before - after) / before * 100
        console.print(
            f"\n[bold]Net effect:[/bold] the prompt went from [bold]{before}[/bold] to "
            f"[bold green]{after}[/bold green] tokens — [bold green]{pct:.0f}% fewer[/bold green]"
            f", and the answer is unchanged in meaning because every edit passed the gate."
        )
    elif not outcome.generated:
        console.print(
            f"\n[bold]Net effect:[/bold] the model was [bold cyan]never called[/bold cyan]. "
            f"All {before} input tokens and the whole generation were avoided."
        )


#: Plain-English name and one-line job for each stage. Used by the walkthrough,
#: where "m1_tier2" and "mmr: kept 6 of 8 turns" are debug output, not an
#: explanation — a viewer should not have to already know the architecture to
#: follow what the system just did.
_PLAIN = {
    "m6a_deterministic": ("Calculator", "answers maths and dates without any AI"),
    "m2_cache":          ("Memory", "reuses an answer given before"),
    "m2_cache_probe":    ("Memory probe", "records what the memory would have matched"),
    "m3_history":        ("History trimmer", "drops old turns that no longer matter"),
    "m3_arrange":        ("History arranger", "reorders the kept turns"),
    "m1_tier1":          ("Politeness remover", "deletes greetings and filler"),
    "m1_tier2":          ("Duplicate remover", "deletes a sentence that repeats a fact"),
    "m1_tier3":          ("Wordiness trimmer", "shortens long-winded phrasing"),
    "m4_assembler":      ("Prompt arranger", "puts unchanging text first so the AI can reuse its work"),
    "m5_budgeter":       ("Answer limiter", "decides how long the answer may be"),
    "m6b_router":        ("Router", "decides whether a bigger AI is needed"),
}

_MS_PER_INPUT_TOKEN = 8.5  # measured on this CPU, ADR-034


def walkthrough(console: Console, outcome, question: str, counter=None) -> None:
    """A numbered, plain-English account of what the pipeline did.

    The panel-per-module view showed everything and explained nothing: eleven
    boxes, module ids, and rationales like "mmr: kept 6 of 8 turns" or
    "complexity 0.087 < 0.200". That is a debug trace. Someone being shown the
    system for the first time needs to know what changed, in their own words,
    and why it was safe — and needs the steps that did nothing to stay out of
    the way rather than filling the screen.
    """
    counter = counter or (lambda s: 0)
    deltas = {d.stage: d for d in getattr(outcome, "text_deltas", ())}
    row = outcome.row

    start = row.tokens_in_original
    console.print()
    console.print(Panel(
        f"[bold]{question}[/bold]\n\n[dim]{start} tokens as typed[/dim]",
        title="[bold]Your question[/bold]", border_style="bold blue", title_align="left"))

    step = 0
    idle = []
    for trace in outcome.traces:
        name, job = _PLAIN.get(trace.name, (trace.module_id, ""))
        delta = deltas.get(trace.name)
        acted = trace.outcome in (
            StageOutcome.APPLIED, StageOutcome.REVERTED, StageOutcome.SHORT_CIRCUIT)
        if not acted:
            idle.append(name)
            continue

        step += 1
        saved = trace.tokens_before - trace.tokens_after
        body = Text()

        if trace.outcome is StageOutcome.SHORT_CIRCUIT:
            head = f"[bold cyan]Step {step} — {name}[/bold cyan]  [dim]{job}[/dim]"
            console.print(Panel(
                f"[bold cyan]Answered it here. The AI was never used.[/bold cyan]\n\n"
                f"[bold]{outcome.response.strip()[:200]}[/bold]\n\n"
                f"[dim]All {trace.tokens_before} tokens of the prompt were saved — "
                f"there was no need to send anything.[/dim]",
                title=head, border_style="cyan", title_align="left"))
            continue

        if trace.outcome is StageOutcome.REVERTED:
            lost = _vanished(delta.query_before, delta.query_after) if delta else []
            console.print(Panel(
                f"[bold yellow]It wanted to delete this — and was STOPPED.[/bold yellow]\n\n"
                f"  would have deleted:  [red]{_quote(lost)}[/red]\n\n"
                f"[bold]Why it was stopped:[/bold] {_why_blocked(trace)}\n"
                f"[dim]Your question was left exactly as you wrote it. "
                f"A saving was available and refused.[/dim]",
                title=f"[bold yellow]Step {step} — Safety check[/bold yellow]  "
                      f"[dim]blocks any edit that would change the meaning[/dim]",
                border_style="yellow", title_align="left"))
            continue

        if delta is not None and delta.history_changed:
            dropped = delta.turns_before - delta.turns_after
            console.print(Panel(
                f"Dropped [bold]{dropped}[/bold] old turns from the conversation "
                f"({delta.turns_before} → {delta.turns_after}).\n"
                f"[dim]Your question itself was not touched.[/dim]",
                title=f"[bold green]Step {step} — {name}[/bold green]  [dim]{job}[/dim]"
                      + (f"   [bold green]saved {saved} tokens[/bold green]" if saved > 0 else ""),
                border_style="green", title_align="left"))
            continue

        if delta is not None and delta.query_changed:
            gone = _vanished(delta.query_before, delta.query_after)
            body.append("was:  ", style="dim")
            body.append(f'"{delta.query_before.strip()}"\n', style="dim")
            body.append("now:  ", style="dim")
            body.append(f'"{delta.query_after.strip()}"\n', style="bold")
            if gone:
                body.append("\nwords that went: ", style="dim")
                body.append(_quote(gone), style="red")
            console.print(Panel(
                body,
                title=f"[bold green]Step {step} — {name}[/bold green]  [dim]{job}[/dim]"
                      + (f"   [bold green]saved {saved} tokens[/bold green]" if saved > 0 else ""),
                border_style="green", title_align="left"))
            continue

        console.print(Panel(
            f"{_plain_decision(trace)}",
            title=f"[bold green]Step {step} — {name}[/bold green]  [dim]{job}[/dim]",
            border_style="green", title_align="left"))

    if idle:
        console.print(f"[dim]Not needed for this question: {', '.join(idle)}.[/dim]")

    end = row.tokens_in_final
    if outcome.generated and start:
        cut = start - end
        pct = cut / start * 100
        saved_ms = cut * _MS_PER_INPUT_TOKEN
        console.print(Panel(
            f"[bold]{start} tokens  →  {end} tokens[/bold]     "
            + (f"[bold green]{cut} fewer ({pct:.0f}%)[/bold green]" if cut > 0
               else "[dim]no reduction on this one[/dim]")
            + (f"\n[dim]On this laptop the AI spends about {_MS_PER_INPUT_TOKEN} ms reading each "
               f"token, so that is roughly [/dim][bold]{saved_ms/1000:.2f} seconds[/bold]"
               f"[dim] of waiting removed.[/dim]" if cut > 0 else "")
            + "\n[dim]Every deletion passed the safety check, so the meaning is unchanged.[/dim]",
            title="[bold]Result[/bold]", border_style="bold blue", title_align="left"))
    elif not outcome.generated:
        console.print(Panel(
            f"[bold cyan]The AI was never used.[/bold cyan] All {start} tokens saved.",
            title="[bold]Result[/bold]", border_style="bold cyan", title_align="left"))


def _why_blocked(trace) -> str:
    """Turn a gate rationale into something a person can read."""
    kinds = {ev.invariant_class for ev in trace.gate_events}
    lost = sorted({v for ev in trace.gate_events for v in ev.lost_values})
    if lost:
        what = {"number": "a number", "entity": "a name", "negation": "a negation",
                "modifier": "a qualifier"}
        label = " and ".join(what.get(k, k) for k in sorted(kinds)) or "information"
        return f"it would have lost {label} — {', '.join(repr(v) for v in lost)}"
    return trace.rationale


def _plain_decision(trace) -> str:
    """Plain wording for the stages that decide rather than edit."""
    ev = trace.evidence or {}
    if trace.name == "m5_budgeter":
        return (f"Classified as [bold]{ev.get('response_class', '?')}[/bold], so the answer may "
                f"run to [bold]{ev.get('budget', '?')}[/bold] tokens.\n"
                f"[dim]A short factual question does not need a long answer, and an unbounded "
                f"one rambles.[/dim]")
    if trace.name == "m6b_router":
        return (f"Judged simple enough for the small model.\n"
                f"[dim]Complexity scored {ev.get('complexity', 0):.2f}; anything below the "
                f"threshold stays on the small model.[/dim]")
    if trace.name == "m4_assembler":
        return ("Put the unchanging part of the prompt first.\n"
                "[dim]The AI can then reuse the work it did on that part last turn instead of "
                "re-reading it. Measured at up to 80x on repeat turns.[/dim]")
    return trace.rationale


def print_outcome(console: Console, outcome, show_response: bool = True,
                  show_text: bool = False, counter=None) -> None:
    console.print(trace_table(outcome))
    if show_text:
        text_delta_panels(console, outcome, counter)
    console.print(summary_panel(outcome, simulated=outcome.row.model_digest.startswith("mock")))
    if show_response:
        console.print(Panel(outcome.response or "[dim](empty)[/dim]", title="Response",
                            border_style="green"))
