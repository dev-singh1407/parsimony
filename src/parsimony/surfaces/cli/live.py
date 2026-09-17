"""A request, shown while it happens.

The earlier explanation was a report: the request finished, then a page of
panels described what had happened. Accurate, and easy to disbelieve -- a
viewer saw the claim "the prompt went from 1,034 tokens to 402" and had to take
it on trust, because nothing on screen moved.

Here the screen follows the request:

  * each layer's row fills in as that layer runs, with its real duration and
    what it changed;
  * the prompt bar shrinks as each cut is committed;
  * while the AI reads the prompt, a clock runs -- that wait IS the prompt
    cost, the thing the layers exist to shorten;
  * the answer streams in word by word, with the real rate and the limit the
    answer limiter set; if the early stop fires, it fires on screen;
  * afterwards, the prompt the AI actually received is shown with every
    removed span struck through AND labelled (strike-through is a terminal
    style; a transcript loses it and removed text would read as kept), and the
    runtime's own counters say how many tokens it read, how many it reused from
    the previous turn, and how long reading took.

Nothing is slowed down or animated for effect. Layers that take a fraction of
a millisecond appear at once, and say so; the seconds a viewer waits are the
model's, measured. `compare` then runs the same question with every layer
switched off on the same model, so "faster" is a measurement on this laptop
rather than an estimate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from parsimony.core.ledger import StageOutcome
from parsimony.pipeline.orchestrator import PipelineObserver
from parsimony.surfaces.cli.explain import _NAME, MS_PER_INPUT_TOKEN, _reason

BAR_WIDTH = 36

#: The display uses block and tick characters. A classic Windows console runs
#: in cp1252, which cannot encode them, and Rich raises UnicodeEncodeError
#: while drawing -- the request succeeds and the screen dies. So the glyphs are
#: chosen from what this terminal can actually encode.
FANCY = {"full": "█", "empty": "░", "ok": "✓", "hand": "✋",
         "star": "★", "dot": "·", "caret": "▌", "bang": "!"}
PLAIN = {"full": "#", "empty": ".", "ok": "+", "hand": "x", "star": "*", "dot": "-",
         "caret": "_", "bang": "!"}


def glyphs_for(console: Console) -> dict:
    encoding = getattr(getattr(console, "file", None), "encoding", None) or "utf-8"
    try:
        "".join(FANCY.values()).encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return PLAIN
    return FANCY


def _ms(ns: int) -> str:
    ms = ns / 1e6
    if ms < 0.1:
        return "<0.1 ms"
    return f"{ms:.1f} ms" if ms < 100 else f"{ms:.0f} ms"


def _secs(seconds: float) -> str:
    return f"{seconds:.1f} s" if seconds >= 1 else f"{seconds * 1000:.0f} ms"


def token_bar(original: int, now: int, width: int = BAR_WIDTH, g: dict = FANCY) -> Text:
    """The prompt as a bar: kept tokens solid, removed tokens hollow."""
    original = max(original, 1)
    kept = max(0, min(width, round(width * now / original)))
    bar = Text()
    bar.append(g["full"] * kept, style="bright_blue")
    bar.append(g["empty"] * (width - kept), style="red")
    cut = original - now
    bar.append(f"  {now:,} tokens", style="bold")
    if cut > 0:
        bar.append(f"   {cut:,} removed ({100 * cut / original:.0f}%)", style="green")
    return bar


@dataclass
class LiveTurn(PipelineObserver):
    """Observer that renders one request live. Pass as `Pipeline.run(observer=...)`."""

    question: str
    stage_names: list[str]
    cfg: object = None
    cache: object = None
    simulated: bool = False
    title: str = "Parsimony"
    attachments: str = ""
    g: dict = field(default_factory=lambda: dict(FANCY))

    ctx: object = None                # the request as it arrived, for the audit
    audit: object = None              # m1_context.ContextAudit, once that stage has run
    kv_per_token: tuple | None = None  # (bytes, how it was derived)
    hud_width: int = 34
    traces: list = field(default_factory=list)
    tokens_original: int = 0
    tokens_now: int = 0
    prompt_tokens: int | None = None
    phase: str = "layers"            # layers -> reading -> writing -> done | answered
    t_prompt: float | None = None
    t_first: float | None = None
    t_last: float | None = None
    pieces: list = field(default_factory=list)
    stop_reason: str | None = None
    stats: dict = field(default_factory=dict)
    budget: int | None = None

    # -- observer ------------------------------------------------------------

    def begin(self, ctx, tokens_original: int) -> None:
        self.tokens_original = self.tokens_now = tokens_original
        self.ctx = ctx

    def stage(self, trace) -> None:
        self.traces.append(trace)
        if trace.name == "m1_context" and self.ctx is not None and self.cfg is not None:
            # Recomputed rather than smuggled through the trace: evidence goes
            # into every ledger row, and per-sentence detail would multiply the
            # ledger's size for data no analysis reads. The embeddings are
            # memoised on this request's derived cache, so the recomputation is
            # arithmetic over vectors it already has.
            try:
                from parsimony.modules.m1_context import audit as audit_context

                self.audit = audit_context(self.ctx, self.cfg)
            except Exception:
                self.audit = None
        if trace.outcome is StageOutcome.APPLIED:
            self.tokens_now = max(0, self.tokens_now - (trace.tokens_before - trace.tokens_after))
        if trace.outcome is StageOutcome.SHORT_CIRCUIT:
            self.phase = "answered"
        if trace.name == "m5_budgeter" and trace.evidence:
            self.budget = trace.evidence.get("budget")

    def prompt(self, text: str, tokens: int) -> None:
        self.prompt_tokens = self.tokens_now = tokens
        self.phase = "reading"
        self.t_prompt = time.perf_counter()

    def token(self, event) -> None:
        now = time.perf_counter()
        if self.t_first is None:
            self.t_first = now
            self.phase = "writing"
        self.t_last = now
        self.pieces.append(event.text)

    def stopped(self, reason: str) -> None:
        self.stop_reason = reason

    def generated(self, stats: dict) -> None:
        self.stats = stats or {}
        self.phase = "done"
        if self.t_last is None:
            self.t_last = time.perf_counter()

    # -- measurements --------------------------------------------------------

    @property
    def reading_seconds(self) -> float | None:
        if self.stats.get("prompt_eval_duration"):
            return self.stats["prompt_eval_duration"] / 1e9
        if self.t_prompt is None:
            return None
        end = self.t_first if self.t_first is not None else time.perf_counter()
        return end - self.t_prompt

    @property
    def writing_seconds(self) -> float | None:
        if self.stats.get("eval_duration"):
            return self.stats["eval_duration"] / 1e9
        if self.t_first is None:
            return None
        return (self.t_last or time.perf_counter()) - self.t_first

    @property
    def tokens_read(self) -> int | None:
        """Tokens the runtime actually processed -- fewer than the prompt when it
        reused the previous turn's work on an unchanged prefix."""
        return self.stats.get("prompt_eval_count")

    #: Below this, the runtime cannot have read the prompt: it reused work it
    #: had already done for an identical prefix. Real prefill on this class of
    #: machine is milliseconds per token, not microseconds.
    CACHED_RATE_MS = 1.0

    @property
    def reused_earlier_work(self) -> bool:
        """Did the runtime skip the prompt because it had just seen it?

        Ollama reports every prompt token in prompt_eval_count even when its
        key-value cache made them free, so the giveaway is the rate, not the
        count. Saying "reading took 64 ms" without saying why would credit the
        layers with a saving they did not make.
        """
        read, secs = self.tokens_read or self.prompt_tokens, self.reading_seconds
        if self.simulated or not read or not secs:
            return False
        return 1000 * secs / read < self.CACHED_RATE_MS

    @property
    def ms_per_token(self) -> float:
        read, secs = self.tokens_read or self.prompt_tokens, self.reading_seconds
        if read and secs and not self.simulated and not self.reused_earlier_work:
            return 1000 * secs / read
        return MS_PER_INPUT_TOKEN

    # -- rendering -----------------------------------------------------------

    def _layer_rows(self) -> Table:
        table = Table(box=None, show_edge=False, pad_edge=False, padding=(0, 1),
                      header_style="bold dim")
        table.add_column("", width=2)
        table.add_column("Layer", width=18, style="bold")
        table.add_column("Time", justify="right", width=8)
        table.add_column("Tokens", justify="right", width=7)
        table.add_column("What happened", overflow="fold")
        done = {t.name: t for t in self.traces}
        answered = self.phase == "answered" or any(
            t.outcome is StageOutcome.SHORT_CIRCUIT for t in self.traces)
        running_marked = False
        for name in self.stage_names:
            label = _NAME.get(name, (name, ""))[0]
            trace = done.get(name)
            if trace is None:
                if answered:
                    table.add_row(Text(self.g["dot"], style="dim"), Text(label, style="dim"),
                                  "", "", Text("not needed - already answered", style="dim"))
                elif not running_marked and self.phase == "layers":
                    table.add_row(Spinner("dots", style="cyan"), label, "", "",
                                  Text("working...", style="cyan"))
                    running_marked = True
                else:
                    table.add_row(Text(self.g["dot"], style="dim"), Text(label, style="dim"),
                                  "", "", "")
                continue
            change = trace.tokens_after - trace.tokens_before
            if trace.outcome is StageOutcome.SHORT_CIRCUIT:
                mark = Text(self.g["star"], style="bold cyan")
                tokens = Text("answer", style="cyan")
            elif trace.outcome is StageOutcome.REVERTED:
                mark = Text(self.g["hand"], style="yellow")
                tokens = Text("refused", style="yellow")
            elif trace.outcome is StageOutcome.APPLIED and change < 0:
                mark = Text(self.g["ok"], style="green")
                tokens = Text(f"{change:,}", style="bold green")
            elif trace.outcome is StageOutcome.ERROR:
                mark, tokens = Text(self.g["bang"], style="red"), Text("error", style="red")
            else:
                mark, tokens = Text(self.g["ok"], style="dim green"), Text("-", style="dim")
            why = _reason(trace, None, self.cache, self.cfg)
            style = "dim" if trace.outcome in (StageOutcome.SKIPPED, StageOutcome.NOOP) \
                and change == 0 else ""
            table.add_row(mark, label, Text(_ms(trace.duration_ns), style="dim"), tokens,
                          Text(why, style=style))
        return table

    def _model_line(self) -> Text | Table:
        who = "The simulated AI" if self.simulated else "The AI"
        if self.phase == "answered":
            return Text(f"  {who} was never called: 0 tokens read, 0 written.",
                        style="bold cyan")
        if self.phase == "layers":
            return Text("")
        grid = Table.grid(padding=(0, 1))
        grid.add_column(width=2)
        grid.add_column()
        read = self.reading_seconds or 0.0
        if self.phase == "reading":
            grid.add_row(Spinner("dots", style="yellow"),
                         Text.from_markup(f"[yellow]{who} is reading the prompt "
                                          f"({self.prompt_tokens:,} tokens)... "
                                          f"[bold]{read:.1f} s[/bold][/yellow]  "
                                          f"[dim]this wait is what the layers shorten[/dim]"))
            return grid
        n = len(self.pieces)
        writing = self.writing_seconds or 0.0
        rate = (n - 1) / writing if writing > 0 and n > 1 else 0.0
        limit = f" of {self.budget} allowed" if self.budget else ""
        reading = Text.from_markup(
            f"[green]{self.g['ok']}[/green] read the prompt "
            + ("[dim](simulated: no real reading time)[/dim]" if self.simulated
               else f"in [bold]{_secs(read)}[/bold]"))
        if self.tokens_read is not None and self.prompt_tokens:
            reused = self.prompt_tokens - self.tokens_read
            if reused > 0:
                reading.append(f"  ({self.tokens_read:,} new tokens; {reused:,} reused from the "
                               f"previous turn)", style="dim")
        grid.add_row("", reading)
        if self.phase == "writing":
            grid.add_row(Spinner("dots", style="green"),
                         Text.from_markup(f"writing: [bold]{n}[/bold] tokens{limit}  "
                                          f"[dim]{rate:.0f} tokens/s[/dim]"))
        else:
            line = Text.from_markup(
                f"[green]{self.g['ok']}[/green] wrote [bold]{n}[/bold] tokens{limit}"
                + ("" if self.simulated else f" in [bold]{_secs(writing)}[/bold]"))
            if self.stop_reason:
                line.append(f"   stopped early: {self.stop_reason}", style="bold yellow")
            grid.add_row("", line)
        return grid

    # -- dashboard panes ---------------------------------------------------

    def document_pane(self, height: int = 24):
        """The attached text, struck through as the selector decides against it."""
        if self.audit is None:
            source = getattr(self.ctx, "original_documents", ()) or ()
            body = Text("\n\n".join(f"{d.title or d.doc_id}\n{d.content[:400]}"
                                     for d in source[:3]) or "no documents attached",
                        style="dim")
            return Panel(body, title="[bold]What you attached[/bold]",
                         border_style="grey42", padding=(0, 1))

        lines = Text()
        shown = 0
        folded = 0
        for unit in self.audit.units:
            if unit.tag == "PROTECTED":
                continue
            if shown >= height:
                folded += 1
                continue
            shown += 1
            if unit.kept:
                lines.append(unit.text + " ", style="bright_green")
            else:
                lines.append(unit.text + " ", style="grey30 strike")
        if folded:
            lines.append(f"\n[... {folded} further sentences, decided the same way]",
                         style="dim")
        kept = sum(1 for u in self.audit.units if u.kept and u.tag != "PROTECTED")
        total = sum(1 for u in self.audit.units if u.tag != "PROTECTED")
        return Panel(lines, border_style="green",
                     title=f"[bold]Context[/bold] [green]{kept} kept[/green] / "
                           f"[grey50]{total - kept} removed[/grey50]",
                     title_align="left", padding=(0, 1))

    def hud_pane(self, width: int = 34):
        """Measured counters. Anything derived says so, and shows its arithmetic.

        Built as fitted lines rather than a table: the pane is narrow, and a
        column layout truncated every note into "of 890 t..." -- a HUD whose
        units are cut off is worse than no HUD.
        """
        inner = max(24, width - 4)
        body = Text()

        def line(label: str, value: str, note: str = "", style: str = "bold") -> None:
            pad = max(1, inner - len(label) - len(value))
            body.append(label, style="dim")
            body.append(" " * pad)
            body.append(value + "\n", style=style)
            if note:
                body.append(f"  {note[:inner - 2]}\n", style="dim")

        removed = max(0, self.tokens_original - (self.prompt_tokens or self.tokens_now))
        sent = self.prompt_tokens if self.prompt_tokens is not None else self.tokens_now
        line("prompt", f"{sent:,} tokens", f"of {self.tokens_original:,} as written")
        if self.tokens_original:
            line("removed", f"{removed:,}", f"{100 * removed / self.tokens_original:.0f}% of the prompt",
                 style="bold green")

        derived = getattr(self.ctx, "derived", None)
        embed_ns = getattr(derived, "embed_ns", 0) if derived is not None else 0
        selector = next((t for t in self.traces if t.name == "m1_context"), None)
        if selector is not None:
            sentences = (selector.evidence or {}).get("sentences")
            if embed_ns:
                line("encoder", _ms(embed_ns), f"scored {sentences} sentences")
                line("ranking", _ms(max(0, selector.duration_ns - embed_ns)), "BM25, anchors, MMR")
            else:
                line("selector", _ms(selector.duration_ns),
                     f"ranked {sentences} sentences" if sentences else "")

        read = self.reading_seconds
        if self.phase == "reading":
            line("AI reading", f"{read or 0:.1f} s", "in progress", style="bold yellow")
        elif read is not None and self.phase in ("writing", "done"):
            if self.reused_earlier_work:
                line("AI read", "reused", "the runtime had this prompt cached")
            else:
                line("AI read", _secs(read), f"{self.ms_per_token:.1f} ms per token")
        if self.pieces:
            wrote = self.writing_seconds or 0.0
            note = ("simulated" if self.simulated else
                    f"{(len(self.pieces) - 1) / wrote:.0f} tokens/s"
                    if wrote > 0.05 and len(self.pieces) > 1 else "")
            line("AI wrote", f"{len(self.pieces)} tokens", note)

        body.append("\n")
        saved = removed * self.ms_per_token / 1000
        line("not read", _secs(saved), f"at {self.ms_per_token:.1f} ms/token, measured here"
             if not self.simulated and not self.reused_earlier_work
             else f"at {self.ms_per_token:.1f} ms/token, measured", style="bold green")
        if self.kv_per_token and removed:
            per_token, how = self.kv_per_token
            line("KV cache", f"{removed * per_token / 1e6:.1f} MB", "never allocated",
                 style="bold green")
            body.append(Text(f"  {how}\n", style="dim italic"))
        return Panel(body, title="[bold]Measured[/bold]", border_style="bright_blue",
                     title_align="left", padding=(0, 1))

    def dashboard(self, height: int = 40, width: int = 110):
        self.hud_width = max(30, width // 3)
        layout = Layout()
        head = Text()
        head.append(f"{self.title}  ", style="bold bright_blue")
        head.append(self.question, style="bold")
        if self.attachments:
            head.append(f"\n{self.attachments}", style="dim")
        answer = Text("".join(self.pieces).strip() or "...", style="")
        if self.phase == "writing":
            answer.append(self.g["caret"], style="blink bold")
        layout.split_column(
            Layout(Panel(head, border_style="bright_blue", padding=(0, 1)), name="head", size=4),
            Layout(name="body", ratio=2),
            Layout(self._layer_rows(), name="layers", size=min(14, max(6, height // 3))),
            Layout(Panel(answer, title="[bold]Answer[/bold]", title_align="left",
                         border_style="green", padding=(0, 1)), name="answer", size=6),
        )
        layout["body"].split_row(Layout(self.document_pane(height=max(6, height // 3)),
                                        name="doc", ratio=2),
                                 Layout(self.hud_pane(self.hud_width), name="hud", ratio=1))
        return layout

    def __rich__(self):
        head = Text()
        head.append(f"{self.title}  ", style="bold bright_blue")
        head.append(self.question, style="bold")
        parts = [head]
        if self.attachments:
            parts.append(Text(f"with {self.attachments}", style="dim"))
        parts.append(Text(""))
        bar = Text("Prompt  ", style="bold")
        bar.append_text(token_bar(self.tokens_original, self.tokens_now, BAR_WIDTH, self.g))
        parts.append(bar)
        parts.append(Text(""))
        parts.append(self._layer_rows())
        parts.append(Text(""))
        parts.append(self._model_line())
        if self.pieces:
            answer = Text("".join(self.pieces).strip())
            if self.phase == "writing":
                answer.append(self.g["caret"], style="blink bold")
            parts.append(Panel(answer, title="[bold]Answer[/bold]", title_align="left",
                               border_style="green"))
        return Group(*parts)


def run_live(console: Console, pipeline, question: str, history=(), *, documents=(),
             conversation_id=None, turn_index: int = 0, title: str = "Parsimony",
             attachments: str = ""):
    """Run one request with the live view; returns (outcome, view).

    With documents attached the screen becomes a fixed dashboard -- text on the
    left striking through as the selector decides, counters on the right -- and
    that view is transient: when the request finishes it gives way to the
    permanent record (what the model received, what was measured), so the
    scrollback stays readable and nothing is only visible while it moves.
    """
    from parsimony.infra.providers import kv_bytes_per_token

    stage_names = [s.name for s in pipeline.registry.ordered(pipeline.cfg)]
    view = LiveTurn(question, stage_names, cfg=pipeline.cfg, cache=pipeline.cache,
                    simulated=pipeline.provider.model_digest.startswith("mock"),
                    title=title, attachments=attachments, g=glyphs_for(console))
    try:
        view.kv_per_token = kv_bytes_per_token(pipeline.provider)
    except Exception:
        view.kv_per_token = None

    height = console.size.height or 40
    as_dashboard = bool(documents) and console.is_terminal and height >= 24

    class _Screen:
        def __rich__(self):
            return view.dashboard(height, console.size.width or 110) if as_dashboard else view

    with Live(_Screen(), console=console, refresh_per_second=12, transient=as_dashboard):
        outcome = pipeline.run(question, tuple(history), conversation_id=conversation_id,
                               turn_index=turn_index, documents=documents, observer=view)
        if view.phase not in ("answered", "done"):
            view.phase = "answered" if not outcome.generated else "done"
    if as_dashboard:
        # The dashboard has gone; leave the same layers table behind it.
        console.print(view._layer_rows())
    return outcome, view


# ------------------------------------------------------- after the request --


def _sentences_view(before: str, after: str, limit: int = 3) -> Text:
    """Kept sentences plain, removed ones struck through; long removed runs folded."""
    from parsimony.infra.nlp import split_sentences

    kept = set(split_sentences(after))
    out = Text()
    removed_run: list[str] = []

    def flush() -> None:
        # Every removed run is LABELLED, not only struck through. Strike-through
        # is a terminal style: paste the same panel into a transcript, a report
        # or a screenshot tool that drops styling and removed text reads as
        # kept, which is the one misreading this panel exists to prevent.
        if not removed_run:
            return
        if len(removed_run) == 1:
            out.append("[removed] ", style="red dim")
            out.append(removed_run[0] + " ", style="red strike dim")
        elif len(removed_run) <= limit:
            out.append(f"[removed, {len(removed_run)} sentences] ", style="red dim")
            for s in removed_run:
                out.append(s + " ", style="red strike dim")
        else:
            out.append(f"[removed, {len(removed_run)} sentences] ", style="red dim")
            out.append(removed_run[0] + " ", style="red strike dim")
            out.append(f"[... {len(removed_run) - 1} more] ", style="dim")
        removed_run.clear()

    for s in split_sentences(before):
        if s in kept:
            flush()
            out.append(s + " ")
        else:
            removed_run.append(s)
    flush()
    return out


def received_panel(outcome) -> Panel:
    """What the AI was actually sent, with everything removed struck through."""
    from parsimony.surfaces.cli.render import _diff_text

    ctx = outcome.ctx
    body = Table.grid(padding=(0, 2))
    body.add_column(style="bold", width=18, overflow="fold")
    body.add_column(overflow="fold")

    if ctx.system_prompt:
        body.add_row("Instructions", Text(ctx.system_prompt + "   (always first, never changes)",
                                          style="dim"))

    kept_turns = {t.turn_id: t for t in ctx.history}
    for turn in ctx.original_history:
        who = "You, earlier" if turn.role == "user" else "AI, earlier"
        now = kept_turns.get(turn.turn_id)
        if now is None:
            snippet = turn.content if len(turn.content) < 90 else turn.content[:87] + "..."
            body.add_row(Text(who, style="dim"), Text(snippet + "   (dropped: not relevant now)",
                                                      style="red strike dim"))
        elif now.content != turn.content:
            body.add_row(who, _sentences_view(turn.content, now.content))
        else:
            snippet = turn.content if len(turn.content) < 160 else turn.content[:157] + "..."
            body.add_row(who, Text(snippet))

    kept_docs = {d.doc_id: d for d in ctx.documents}
    for doc in ctx.original_documents:
        label = doc.title or doc.doc_id
        now = kept_docs.get(doc.doc_id)
        if now is None:
            body.add_row(Text(label, style="dim"),
                         Text("whole section removed: nothing in it bears on the question",
                              style="red strike dim"))
        else:
            body.add_row(label, _sentences_view(doc.content, now.content))

    if ctx.query != ctx.original_query:
        body.add_row("Your question", _diff_text(ctx.original_query, ctx.query, show="before"))
        body.add_row("sent as", Text(ctx.query, style="bold"))
    else:
        body.add_row("Your question", Text(ctx.query + "   (unchanged)"))

    return Panel(body, title="[bold]What the AI actually received[/bold]  "
                             "[dim]anything marked [removed] was not sent[/dim]",
                 border_style="bright_blue", title_align="left")


def measured_panel(outcome, view: LiveTurn) -> Panel:
    """Real counters from this request, and what the untrimmed prompt would have cost."""
    row = outcome.row
    lines = []
    if not outcome.generated:
        est = row.tokens_in_original * view.ms_per_token / 1000
        lines.append(f"[bold cyan]Answered without the AI.[/bold cyan] Sending the question would "
                     f"have meant reading {row.tokens_in_original:,} tokens - about "
                     f"[bold]{est:.1f} s[/bold] on this laptop - and then writing an answer.")
    else:
        before, after = row.tokens_in_original, row.tokens_in_final
        rate = view.ms_per_token
        read = view.reading_seconds or 0.0
        source = ("the runtime's own timer" if view.tokens_read is not None else
                  "wall clock" if not view.simulated else "simulated")
        lines.append(f"Prompt: [bold]{before:,}[/bold] tokens as written -> [bold]{after:,}[/bold] "
                     f"sent ({before - after:,} removed).")
        if view.simulated:
            lines.append(f"Reading time is not measured here - this is the simulated model. At the "
                         f"{rate:.1f} ms per token measured on the real one, {after:,} tokens is "
                         f"about {after * rate / 1000:.1f} s against {before * rate / 1000:.1f} s.")
        elif view.reused_earlier_work:
            lines.append(f"Reading took only [bold]{_secs(read)}[/bold]: the runtime had already "
                         f"processed this exact prompt and reused that work. Rates below are the "
                         f"{rate:.1f} ms per token measured on this machine from cold "
                         f"(docs/09-findings.md).")
        else:
            lines.append(f"Reading took [bold]{_secs(read)}[/bold] ({source}), "
                         f"{rate:.1f} ms per token.")
        if view.tokens_read is not None and view.tokens_read < after:
            lines.append(f"The AI re-read only {view.tokens_read:,} of them: the first "
                         f"{after - view.tokens_read:,} were identical to last turn's prompt, so "
                         f"its earlier work was reused. The prompt arranger keeps that start "
                         f"unchanged.")
        if before > after and not view.simulated:
            lines.append(f"The full {before:,}-token prompt would have taken about "
                         f"[bold]{before * rate / 1000:.1f} s[/bold] to read at that rate "
                         f"[dim](estimate - type 'compare' to measure it)[/dim].")
        if view.stop_reason:
            lines.append(f"The answer was cut short because it {view.stop_reason}.")
    if view.simulated:
        lines.append("[yellow]Simulated AI: the layers are real, the timings are not. "
                     "Start Ollama for real ones.[/yellow]")
    return Panel("\n".join(lines), title="[bold]Measured[/bold]", border_style="grey50",
                 title_align="left")


def comparison_table(ours, ours_view: LiveTurn, plain, plain_view: LiveTurn) -> Table:
    """The same question with and without the layers, both measured."""
    t = Table(title="The same question, measured both ways on this laptop", header_style="bold",
              title_style="bold")
    t.add_column("")
    t.add_column("Without Parsimony", justify="right")
    t.add_column("With Parsimony", justify="right")
    t.add_column("Difference", justify="right", style="bold green")

    def row(label, a, b, fmt, better_lower=True):
        if a is None or b is None:
            t.add_row(label, "-" if a is None else fmt(a), "-" if b is None else fmt(b), "")
            return
        diff = a - b
        pct = f" ({100 * diff / a:.0f}%)" if a else ""
        t.add_row(label, fmt(a), fmt(b), (f"-{fmt(diff)}{pct}" if diff > 0 else
                                          Text(f"+{fmt(-diff)}", style="yellow") if diff < 0
                                          else "same"))

    row("Prompt tokens", plain.row.tokens_in_final, ours.row.tokens_in_final if ours.generated
        else 0, lambda v: f"{v:,}")
    row("Time reading the prompt", plain_view.reading_seconds,
        ours_view.reading_seconds if ours.generated else 0.0, _secs)
    row("Answer tokens", len(plain_view.pieces), len(ours_view.pieces), lambda v: f"{v:,}")
    total_plain = (plain.row.total_ns or 0) / 1e9
    total_ours = (ours.row.total_ns or 0) / 1e9
    row("Total time", total_plain, total_ours, _secs)
    return t


# ------------------------------------------------------------ conversation --


@dataclass
class LiveSession:
    """Running totals across a conversation, from measured values where they exist."""

    questions: int = 0
    from_memory: int = 0
    from_calculator: int = 0
    tokens_written: int = 0          # prompt tokens as the user's content stood
    tokens_sent: int = 0
    seconds_reading: float = 0.0     # measured
    seconds_full: float = 0.0        # the untrimmed prompts at this turn's measured rate
    simulated: bool = False

    def record(self, outcome, view: LiveTurn) -> None:
        row = outcome.row
        self.questions += 1
        self.simulated = self.simulated or view.simulated
        tier = str(row.route_tier or "")
        if tier.startswith("CACHE"):
            self.from_memory += 1
        elif tier.startswith("DETERMINISTIC"):
            self.from_calculator += 1
        self.tokens_written += row.tokens_in_original
        self.tokens_sent += row.tokens_in_final if outcome.generated else 0
        rate = view.ms_per_token / 1000
        self.seconds_full += row.tokens_in_original * rate
        if outcome.generated:
            self.seconds_reading += view.reading_seconds or 0.0

    def line(self) -> str:
        removed = self.tokens_written - self.tokens_sent
        pct = 100 * removed / self.tokens_written if self.tokens_written else 0
        served = []
        if self.from_memory:
            served.append(f"{self.from_memory} from memory")
        if self.from_calculator:
            served.append(f"{self.from_calculator} by the calculator")
        tail = f" ({', '.join(served)})" if served else ""
        clock = "simulated" if self.simulated else "measured"
        return (f"This conversation: {self.questions} question(s){tail}  |  "
                f"{self.tokens_sent:,} of {self.tokens_written:,} prompt tokens sent "
                f"({pct:.0f}% removed)  |  the AI spent {self.seconds_reading:.1f} s reading "
                f"({clock}) instead of about {self.seconds_full:.1f} s")


def show_turn(console: Console, pipeline, question: str, history=(), *, documents=(),
              conversation_id=None, turn_index: int = 0, session: LiveSession | None = None,
              attachments: str = ""):
    """One question, live, then what the AI received and what it measured."""
    from parsimony.surfaces.cli.explain import edit_panels, memory_panel

    outcome, view = run_live(console, pipeline, question, history, documents=documents,
                             conversation_id=conversation_id, turn_index=turn_index,
                             attachments=attachments)
    if not outcome.generated:
        console.print(Panel(outcome.response or "", title="[bold]Answer[/bold]",
                            title_align="left", border_style="cyan"))
    trace = next((t for t in outcome.traces if t.name == "m2_cache"), None)
    ev = (trace.evidence or {}) if trace is not None else {}
    interesting = (trace is not None and (trace.outcome is StageOutcome.SHORT_CIRCUIT
                   or (ev.get("score") is not None
                       and ev["score"] >= pipeline.cfg.cache.tau_lo)))
    if interesting:
        panel = memory_panel(outcome, cache=pipeline.cache, cfg=pipeline.cfg)
        if panel is not None:
            console.print(panel)
    # The edit the gate refused, and any edit to the question, shown as text.
    # A layer row can say an edit was refused; only this shows what it was.
    edit_panels(console, outcome)
    if outcome.generated:
        console.print(received_panel(outcome))
    console.print(measured_panel(outcome, view))
    if session is not None:
        session.record(outcome, view)
        console.print(Text(session.line(), style="dim"))
    return outcome, view


def compare_turn(console: Console, provider, cfg, question: str, history=(), *, documents=(),
                 **_ignored):
    """Ask the same question twice on the same model -- with the layers, and without.

    BOTH arms are run here, fresh, rather than reusing the answer already on
    screen. Two reasons, and both would otherwise flatter Parsimony:

      * the runtime keeps the key-value cache of the prompt it just processed,
        so re-sending a prompt it has already seen is nearly free -- the
        earlier answer's prompt would be measured as 64 ms rather than the 3 s
        it actually cost;
      * the memory would short-circuit the repeat, and the comparison would be
        about the cache rather than about the prompt.

    Each arm gets its own nonce in the always-first instruction line, so
    neither can reuse the other's work, and its own empty memory.
    """
    import secrets
    from dataclasses import replace as _replace

    from parsimony.core.config import baseline
    from parsimony.pipeline.orchestrator import Pipeline

    def cold(config, title):
        nonce = secrets.token_hex(3)
        marked = _replace(config, system_prompt=f"{config.system_prompt} [{nonce}]")
        pipe = Pipeline(marked, provider=provider, capture_text=False)
        return run_live(console, pipe, question, history, documents=documents, title=title)

    console.print(Text("\nThe same question, asked twice from cold: with the layers, "
                       "then without them.", style="bold"))
    ours, ours_view = cold(cfg, "With Parsimony")
    plain, plain_view = cold(baseline(), "Without Parsimony")
    console.print(comparison_table(ours, ours_view, plain, plain_view))
    if plain_view.simulated:
        console.print(Text("Simulated AI: start Ollama for a real measurement.", style="yellow"))
    else:
        console.print(Text("Both runs used the same model on this laptop, one after the other, "
                           "each from cold; nothing here is estimated.", style="dim"))
    return plain, plain_view
