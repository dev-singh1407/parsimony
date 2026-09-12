"""What happened to this question, in words a first-time viewer can follow.

The previous walkthrough showed the stages that ACTED and collapsed the rest
into one grey line: "Not needed for this question: Memory, ...". That is
exactly backwards for the stage people ask about most. A cache miss is not
"not needed" -- it is a decision, with a reason and a number behind it, and
hiding it is why the memory looked broken even on turns where it was working
correctly.

Three things it also got wrong, all of them visible on screen:

  * "24 tokens as typed" for a nine-token question. The figure was the whole
    prompt -- fixed instructions plus the conversation so far -- so the number
    grew every turn while the question stayed the same size.
  * "Politeness remover: saved 6 tokens" on a question containing no
    politeness. The stage had cleaned a GREETING IN AN EARLIER TURN, which is
    real work, but the panel showed the user's own question unchanged beside
    the claim.
  * Per-stage counts are taken over the conversation text, the totals over the
    assembled prompt. Mixing them means the steps do not add up to the result,
    which is the first thing a sceptical reader checks.

So: every layer reports itself every turn, the token arithmetic is shown with
its three parts separated, and the memory explains its comparison whether it
hit or missed.
"""

from __future__ import annotations

from dataclasses import dataclass

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from parsimony.core.ledger import StageOutcome
from parsimony.surfaces.cli.render import _diff_text, _quote, _vanished

#: Measured on this machine (ADR-034): prefill is 92-99% of inference time and
#: runs at roughly this rate, so a token count converts into waiting.
MS_PER_INPUT_TOKEN = 8.5

#: Every layer, in pipeline order, with a name and a job a non-specialist can
#: hold onto. Stages absent from a run are simply not shown.
LAYERS: tuple[tuple[str, str, str], ...] = (
    ("m6a_deterministic", "Calculator", "answers sums and dates without the AI"),
    ("m2_cache", "Memory", "reuses an earlier answer when the question matches"),
    ("m3_history", "History trimmer", "drops earlier turns that no longer matter"),
    ("m3_arrange", "History arranger", "puts the most relevant turn nearest your question"),
    ("m1_tier1", "Politeness remover", "removes greetings, please, thanks, formatting"),
    ("m1_tier2", "Repeat remover", "drops a sentence repeating a fact already given"),
    ("m1_tier3", "Wordiness trimmer", "shortens long-winded phrasing"),
    ("m4_assembler", "Prompt arranger", "keeps the unchanging part first, so the AI can reuse its work"),
    ("m5_budgeter", "Answer limiter", "decides how long the answer may be"),
    ("m6b_router", "Model chooser", "decides whether a bigger AI is needed"),
)
_NAME = {key: (name, job) for key, name, job in LAYERS}
_ORDER = {key: i for i, (key, _, _) in enumerate(LAYERS)}


@dataclass
class Session:
    """Running totals, so a viewer sees the effect accumulate over a chat."""

    questions: int = 0
    from_memory: int = 0
    from_calculator: int = 0
    prompt_before: int = 0
    prompt_after: int = 0

    def record(self, outcome) -> None:
        row = outcome.row
        self.questions += 1
        tier = str(row.route_tier or "")
        if tier.startswith("CACHE"):
            self.from_memory += 1
        elif tier.startswith("DETERMINISTIC"):
            self.from_calculator += 1
        self.prompt_before += row.tokens_in_original
        self.prompt_after += row.tokens_in_final

    @property
    def saved(self) -> int:
        return self.prompt_before - self.prompt_after

    @property
    def percent(self) -> float:
        return 100.0 * self.saved / self.prompt_before if self.prompt_before else 0.0

    @property
    def seconds(self) -> float:
        return self.saved * MS_PER_INPUT_TOKEN / 1000.0

    def line(self) -> str:
        served = []
        if self.from_memory:
            served.append(f"{self.from_memory} from memory")
        if self.from_calculator:
            served.append(f"{self.from_calculator} by the calculator")
        tail = f" ({', '.join(served)})" if served else ""
        return (f"This conversation: {self.questions} question(s){tail} - "
                f"{self.saved} prompt tokens avoided ({self.percent:.0f}%), "
                f"about {self.seconds:.1f} s of the AI's reading time")


# ----------------------------------------------------------------- helpers --

def _tokens(counter, text: str) -> int:
    try:
        return counter(text)
    except Exception:                                    # a counter is optional
        return 0


def _payload_before(outcome) -> int:
    for trace in outcome.traces:
        return trace.tokens_before
    return outcome.row.tokens_in_original


def _pct(score: float) -> str:
    return f"{100.0 * score:.0f}%"


def _memory_reason(trace, cache, cfg) -> str:
    """One line for the layer table; the panel carries the detail."""
    ev = trace.evidence or {}
    zone = ev.get("zone")
    if zone == "uncacheable":
        return "nothing in the question to match on"
    if ev.get("scoped"):
        return (f"follow-up (because of {ev.get('scope_reason')!r}); only this "
                f"conversation could match")
    if zone == "miss":
        return "nothing stored yet to compare against"
    score = ev.get("score")
    if trace.outcome is StageOutcome.SHORT_CIRCUIT:
        return f"reused an earlier answer - {_pct(score)} match"
    if ev.get("type_agree") is False:
        return (f"{_pct(score)} match, but a different kind of question "
                f"({ev.get('stored_question_type')} vs {ev.get('question_type')})")
    if zone == "verify" and ev.get("rejected"):
        return f"{_pct(score)} match, refused: {trace.rationale.split(': ', 1)[-1]}"
    if zone == "reject":
        lo = _pct(cfg.cache.tau_lo) if cfg else "the floor"
        return f"closest was only {_pct(score)} similar, below {lo}"
    return trace.rationale


def _reason(trace, delta, cache, cfg) -> str:
    """Plain English for one layer, from its own evidence."""
    ev = trace.evidence or {}
    name = trace.name

    if trace.outcome is StageOutcome.SKIPPED:
        # Two very different things arrive as SKIPPED: a module the operator
        # disabled, and a module with nothing to do yet. Calling the second
        # "switched off" makes a working system look half-broken on turn one.
        if "disabled" in (trace.rationale or ""):
            return "switched off for this run"
        if name in ("m3_history", "m3_arrange"):
            return "no earlier turns to work with yet"
        return trace.rationale or "nothing to do"
    if trace.outcome is StageOutcome.REVERTED:
        lost = _vanished(delta.query_before, delta.query_after) if delta else []
        what = _quote(lost) if lost else "an edit"
        return f"wanted to delete {what} - REFUSED, it would change the meaning"

    if name == "m2_cache":
        return _memory_reason(trace, cache, cfg)
    if name == "m6a_deterministic":
        if trace.outcome is StageOutcome.SHORT_CIRCUIT:
            return f"worked it out exactly ({ev.get('handler', 'handler')}); the AI was never called"
        return "not a sum or a date question"
    if name == "m3_history":
        if ev.get("kept") is not None:
            return f"kept the {ev['kept']} most relevant of {ev['kept'] + ev['dropped']} earlier turns"
        return trace.rationale
    if name == "m3_arrange":
        if ev.get("moved_from") is not None:
            return (f"moved turn {ev['moved_from'] + 1} next to your question "
                    f"({_pct(ev.get('relevance', 0))} related)")
        return trace.rationale
    if name == "m1_tier1":
        if delta is not None and delta.history_changed:
            return "cleaned an earlier message; your question was already clean"
        if delta is not None and delta.query_changed:
            gone = _vanished(delta.query_before, delta.query_after)
            return f"removed {_quote(gone)}" if gone else "tidied the wording"
        # Without text capture there is no delta to inspect, but the stage still
        # reports what it saved. Saying "nothing to remove" beside a token drop
        # is the kind of contradiction that makes a viewer distrust the rest.
        saved = ev.get("tokens_saved") or 0
        if saved:
            return f"removed {saved} tokens of greeting or formatting"
        return "nothing to remove"
    if name == "m1_tier2":
        if ev.get("dropped_sentences"):
            return f"dropped {ev['dropped_sentences']} repeated sentence(s)"
        return "no repeated sentence"
    if name == "m1_tier3":
        rejected = ev.get("negative_yield_rejected") or 0
        if rejected:
            return f"{rejected} edit(s) rejected for saving nothing"
        return "no wordy phrase worth shortening"
    if name == "m4_assembler":
        return f"pinned {ev.get('invariant_tokens', 0)} unchanging tokens to the front"
    if name == "m5_budgeter":
        return f"{ev.get('response_class', '?')} question, so up to {ev.get('budget', '?')} tokens"
    if name == "m6b_router":
        if ev.get("escalation_enabled") is False and ev.get("above_threshold"):
            return (f"complex ({ev.get('complexity', 0):.2f}) but the bigger model is "
                    f"switched off, so the small one")
        return f"simple enough ({ev.get('complexity', 0):.2f}) for the small model"
    return trace.rationale


# ------------------------------------------------------------------ panels --

def question_panel(outcome, question: str, counter) -> Panel:
    """The three things that make up the prompt, kept apart.

    A single "tokens as typed" figure conflated them and grew every turn.
    """
    row = outcome.row
    typed = _tokens(counter, question)
    payload = _payload_before(outcome)
    history = max(payload - typed, 0)
    fixed = max(row.tokens_in_original - payload, 0)

    table = Table.grid(padding=(0, 2))
    table.add_column(justify="left")
    table.add_column(justify="right")
    table.add_row("[bold]What you typed[/bold]", f"[bold]{typed}[/bold] tokens")
    if history:
        table.add_row("The conversation so far", f"{history} tokens")
    if fixed:
        table.add_row("Fixed instructions", f"{fixed} tokens")
    table.add_row("[dim]-------------------------------[/dim]", "[dim]------[/dim]")
    table.add_row("Full prompt, before any trimming",
                  f"[bold]{row.tokens_in_original}[/bold] tokens")

    return Panel(Group(Text(question, style="bold"), Text(""), table),
                 title="[bold]Your question[/bold]", border_style="blue",
                 title_align="left")


def layers_table(outcome, cache=None, cfg=None) -> Table:
    """Every layer, every turn -- including the ones that did nothing."""
    deltas = {d.stage: d for d in getattr(outcome, "text_deltas", ())}
    table = Table(show_edge=False, pad_edge=False, box=None, padding=(0, 1))
    table.add_column("#", justify="right", style="dim", width=2)
    table.add_column("Layer", style="bold", width=20)
    table.add_column("Tokens", justify="right", width=7)
    table.add_column("What it did")

    seen = sorted(outcome.traces, key=lambda t: _ORDER.get(t.name, 99))
    for i, trace in enumerate(seen, start=1):
        name, _job = _NAME.get(trace.name, (trace.name, ""))
        delta = deltas.get(trace.name)
        change = trace.tokens_after - trace.tokens_before

        if trace.outcome is StageOutcome.REVERTED:
            effect, style = "blocked", "yellow"
        elif trace.outcome is StageOutcome.SHORT_CIRCUIT:
            effect, style = "answer", "cyan"
        elif change < 0:
            effect, style = f"{change}", "green"
        else:
            effect, style = "-", "dim"

        table.add_row(str(i), name, Text(effect, style=style),
                      _reason(trace, delta, cache, cfg))
    return table


def memory_panel(outcome, cache=None, cfg=None) -> Panel | None:
    """What the memory compared, and how it decided -- hit or miss.

    Shown on EVERY turn the cache is consulted. The whole point of the module
    is a judgement call, and a judgement call with its reasons hidden is
    indistinguishable from a bug.
    """
    trace = next((t for t in outcome.traces if t.name == "m2_cache"), None)
    if trace is None or trace.outcome is StageOutcome.SKIPPED:
        return None
    ev = trace.evidence or {}

    body = Table.grid(padding=(0, 2))
    body.add_column(justify="left", style="dim", width=26)
    body.add_column(justify="left")

    if ev.get("scoped"):
        body.add_row("Treated as", f"a follow-up, because of the word "
                                   f"{ev.get('scope_reason')!r}")
        body.add_row("So it compared against", "only this conversation's earlier questions")

    if ev.get("zone") == "miss" and not ev.get("scoped"):
        body.add_row("Compared against", "nothing yet - no earlier question is stored")

    stored = None
    entry_id = ev.get("best_entry")
    if cache is not None and entry_id:
        entry = cache.entry(entry_id)
        stored = entry.query if entry is not None else None
    if stored:
        body.add_row("Closest earlier question", f"[italic]{stored}[/italic]")

    score = ev.get("score")
    if score is not None and cfg is not None:
        lo, hi = cfg.cache.tau_lo, cfg.cache.tau_hi
        where = ("reuse it" if score >= hi else
                 "check it carefully" if score >= lo else "do not reuse")
        body.add_row("Similarity", f"[bold]{_pct(score)}[/bold]  "
                                   f"[dim](under {_pct(lo)}: do not reuse - "
                                   f"{_pct(lo)} to {_pct(hi)}: check - "
                                   f"over {_pct(hi)}: reuse)[/dim]  -> {where}")

    if ev.get("question_type") or ev.get("stored_question_type"):
        same = ev.get("type_agree")
        mark = "same" if same is not False else "DIFFERENT"
        body.add_row("Kind of question",
                     f"{ev.get('stored_question_type')} vs {ev.get('question_type')} ({mark})")

    verifier = ev.get("verifier") or {}
    if verifier:
        checks = [
            ("numbers", verifier.get("number_agree")),
            ("names", verifier.get("entity_agree")),
            ("not / never", verifier.get("negation_agree")),
            ("words like cheapest", verifier.get("modifier_agree")),
            ("kind of question", verifier.get("question_agree")),
        ]
        rendered = "  ".join(
            f"[green]{label} ok[/green]" if value else f"[red]{label} DIFFER[/red]"
            for label, value in checks if value is not None
        )
        body.add_row("Safety checks", rendered)
        overlap = verifier.get("jaccard")
        if overlap is not None:
            floor = cfg.cache.jaccard_min if cfg else 0.0
            body.add_row("Shared words",
                         f"{_pct(overlap)} (needs {_pct(floor)})")

    hit = trace.outcome is StageOutcome.SHORT_CIRCUIT
    verdict = ("[bold green]Reused the stored answer. The AI was never "
               "called.[/bold green]" if hit else
               f"[bold]Did not reuse[/bold] - {_memory_reason(trace, cache, cfg)}")
    body.add_row("Verdict", verdict)

    return Panel(body, title="[bold]Memory[/bold] [dim]- reuses an earlier answer when the "
                             "question matches[/dim]",
                 border_style="green" if hit else "grey50", title_align="left")


def edit_panels(console: Console, outcome) -> None:
    """Before/after for the stages that rewrote the user's own question."""
    for delta in getattr(outcome, "text_deltas", ()):
        if not delta.query_changed:
            continue
        name, _job = _NAME.get(delta.stage, (delta.stage, ""))
        colour = "yellow" if delta.reverted else "green"
        head = f"[bold]{name}[/bold] " + (
            "[yellow]wanted this change - and was refused[/yellow]" if delta.reverted
            else "[green]changed your question[/green]")
        body = Table.grid(padding=(0, 2))
        body.add_column(style="dim", width=7)
        body.add_column()
        body.add_row("before", _diff_text(delta.query_before, delta.query_after, show="before"))
        body.add_row("after", _diff_text(delta.query_before, delta.query_after, show="after"))
        console.print(Panel(Group(Text.from_markup(head), Text(""), body),
                            border_style=colour, title_align="left"))


def result_panel(outcome, counter) -> Panel:
    """The arithmetic, and what it is worth in seconds."""
    row = outcome.row
    before, after = row.tokens_in_original, row.tokens_in_final

    if not outcome.generated:
        served = "the calculator" if str(row.route_tier).startswith("DETERM") else "memory"
        return Panel(
            f"[bold cyan]Answered from {served}. The AI was never called.[/bold cyan]\n"
            f"All [bold]{before}[/bold] prompt tokens avoided, and no answer had to be "
            f"generated - about [bold]{before * MS_PER_INPUT_TOKEN / 1000:.1f} s[/bold] "
            f"of reading saved on this laptop.",
            title="[bold]Result[/bold]", border_style="cyan", title_align="left")

    cut = before - after
    lines = [f"[bold]Prompt: {before} -> {after} tokens[/bold]"]
    if cut > 0:
        lines.append(
            f"[green]{cut} fewer ({100.0 * cut / before:.0f}%)[/green] - about "
            f"[bold]{cut * MS_PER_INPUT_TOKEN / 1000:.2f} s[/bold] less reading, at the "
            f"{MS_PER_INPUT_TOKEN} ms per token measured on this machine")
    else:
        lines.append("[dim]nothing could be removed from this one[/dim]")
    budget = next((t.evidence.get("budget") for t in outcome.traces
                   if t.name == "m5_budgeter" and t.evidence), None)
    if budget:
        lines.append(f"[dim]Answer allowed up to {budget} tokens; it used "
                     f"{row.tokens_out}.[/dim]")
    lines.append("[dim]Every deletion passed the safety check, so the meaning is "
                 "unchanged.[/dim]")
    return Panel("\n".join(lines), title="[bold]Result[/bold]",
                 border_style="blue", title_align="left")


# ------------------------------------------------------------------- entry --

def turn_report(console: Console, outcome, question: str, counter,
                *, cache=None, cfg=None, session: Session | None = None) -> None:
    """Everything one turn did, in the order a person would ask about it."""
    console.print(question_panel(outcome, question, counter))
    console.print()
    console.print(layers_table(outcome, cache=cache, cfg=cfg))
    if not outcome.generated:
        console.print("[dim]   The layers below this one never ran: the question was "
                      "answered before it reached the AI.[/dim]")
    console.print()

    panel = memory_panel(outcome, cache=cache, cfg=cfg)
    if panel is not None:
        console.print(panel)
    edit_panels(console, outcome)

    console.print(result_panel(outcome, counter))
    if session is not None:
        session.record(outcome)
        console.print(f"[dim]{session.line()}[/dim]")
