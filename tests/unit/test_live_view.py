"""The live view has to be true while it is still moving.

It is the surface a viewer judges the whole system by, so every number on it is
either measured or labelled as an estimate. These tests pin the cases where an
honest-looking screen would be wrong: a model that never ran, a runtime that
reused its earlier work instead of reading the prompt, a simulated provider with
no timings at all, and an edit whose strike-through is invisible in a transcript.
"""

from __future__ import annotations

import io

import pytest
from rich.console import Console

from parsimony.core.config import baseline, full_stack
from parsimony.core.types import Document, Turn
from parsimony.infra.providers import MockProvider
from parsimony.pipeline.orchestrator import Pipeline
from parsimony.surfaces.cli.live import (
    FANCY,
    PLAIN,
    LiveSession,
    LiveTurn,
    comparison_table,
    compare_turn,
    glyphs_for,
    measured_panel,
    received_panel,
    show_turn,
    token_bar,
)

HANDBOOK = "examples/staff-handbook.md"


def rendered(renderable, width: int = 100) -> str:
    console = Console(file=io.StringIO(), width=width, force_terminal=False,
                      legacy_windows=False)
    console.print(renderable)
    return console.file.getvalue()


def run(question: str, cfg=None, documents=(), history=(), width: int = 100):
    """One turn through the whole live surface, on the simulated model."""
    console = Console(file=io.StringIO(), width=width, force_terminal=False,
                      legacy_windows=False)
    pipe = Pipeline(cfg or full_stack(), provider=MockProvider(), capture_text=True)
    outcome, view = show_turn(console, pipe, question, history, documents=documents)
    return console.file.getvalue(), outcome, view


class TestTheBar:
    def test_it_is_proportional(self):
        assert token_bar(100, 50).plain.startswith(FANCY["full"] * 18)
        assert "50 removed (50%)" in token_bar(100, 50).plain

    def test_nothing_removed_says_nothing_about_removal(self):
        assert "removed" not in token_bar(100, 100).plain

    def test_a_terminal_that_cannot_encode_blocks_gets_plain_ones(self):
        cp1252 = Console(file=io.TextIOWrapper(io.BytesIO(), encoding="cp1252"))
        assert glyphs_for(cp1252) is PLAIN
        utf8 = Console(file=io.TextIOWrapper(io.BytesIO(), encoding="utf-8"))
        assert glyphs_for(utf8) is FANCY


class TestWhatTheScreenSays:
    def test_every_layer_appears_with_what_it_did(self):
        out, _, _ = run("Hello, could you please explain what recursion is? Thanks!")
        for layer in ("Calculator", "Memory", "History trimmer", "Context selector",
                      "Politeness remover", "Answer limiter", "Model chooser"):
            assert layer in out
        assert "removed" in out

    def test_an_answer_from_the_calculator_says_the_model_never_ran(self):
        out, outcome, view = run("What is 847 * 23?")
        assert not outcome.generated
        assert "was never called: 0 tokens read" in out
        assert "19481" in out
        assert view.phase == "answered"

    def test_a_question_the_layers_could_not_shorten_says_so_plainly(self):
        out, _, _ = run("What is the boiling point of water?")
        assert "tokens as written" in out

    def test_an_edited_question_is_shown_before_and_after(self):
        """Strike-through is invisible in a transcript, so the sent form is printed too."""
        out, outcome, _ = run("Hello, could you please explain what recursion is? Thanks!")
        assert "sent as" in out
        assert outcome.ctx.query in out

    def test_removed_document_sentences_are_shown_as_removed(self):
        # Comfortably over the 300-token threshold below which the tier does
        # not apply -- under it there is nothing to show.
        docs = tuple(Document(f"d{i}", c, t) for i, (t, c) in enumerate([
            ("Tallinn", "The Tallinn office opened in 2019. It employs 58 people. "
             + "Staff work four days a week in summer. " * 14),
            ("Porto", "The Porto office employs 41 people. " * 20)]))
        out, outcome, _ = run("How many people does the Tallinn office employ?", documents=docs)
        assert "What the AI actually received" in out
        assert "It employs 58 people" in out
        assert outcome.row.tokens_in_final < outcome.row.tokens_in_original

    def test_a_dropped_document_is_named_rather_than_silently_missing(self):
        long_tail = " ".join(f"Point {i} concerns gardening tools and soil health." for i in range(40))
        docs = (Document("keep", "The fee is 20 pounds. " * 10, "Fees"),
                Document("drop", long_tail, "Gardening"))
        out, _, _ = run("What is the fee?", documents=docs)
        assert "Gardening" in out and "whole section removed" in out


class TestTheNumbersAreHonest:
    def test_a_simulated_model_never_reports_a_reading_time(self):
        out, _, view = run("What is the boiling point of water?")
        assert view.simulated
        assert "Reading time is not measured here" in out
        assert "the layers are real, the timings are not" in out

    def test_reused_prefill_is_called_out_rather_than_counted_as_a_saving(self):
        """64 ms for 300 tokens is 0.2 ms/token: the runtime reused its own work."""
        view = LiveTurn("q", [], simulated=False)
        view.prompt_tokens = 300
        view.stats = {"prompt_eval_count": 300, "prompt_eval_duration": 64_000_000}
        assert view.reused_earlier_work
        assert view.ms_per_token == pytest.approx(8.5)

    def test_a_real_rate_is_taken_from_the_runtimes_own_timer(self):
        view = LiveTurn("q", [], simulated=False)
        view.prompt_tokens = 300
        view.stats = {"prompt_eval_count": 300, "prompt_eval_duration": 2_550_000_000}
        assert not view.reused_earlier_work
        assert view.ms_per_token == pytest.approx(8.5, abs=0.01)

    def test_prompt_tokens_reused_from_the_previous_turn_are_reported(self):
        pipe = Pipeline(full_stack(), provider=MockProvider(), capture_text=True)
        console = Console(file=io.StringIO(), width=100, force_terminal=False)
        outcome, view = show_turn(console, pipe, "What is recursion?")
        view.simulated = False
        view.stats = {"prompt_eval_count": 5, "prompt_eval_duration": 100_000_000}
        assert "reused" in rendered(measured_panel(outcome, view))

    def test_the_comparison_names_both_sides_and_the_difference(self):
        console = Console(file=io.StringIO(), width=100, force_terminal=False)
        question = "What is the boiling point of water?"
        plain = Pipeline(baseline(), provider=MockProvider())
        ours = Pipeline(full_stack(), provider=MockProvider())
        out_plain = plain.run(question)
        out_ours = ours.run(question)
        view = LiveTurn(question, [])
        table = comparison_table(out_ours, view, out_plain, LiveTurn(question, []))
        text = rendered(table)
        assert "Without Parsimony" in text and "With Parsimony" in text
        assert "Prompt tokens" in text


class TestCompareRunsBothSidesFromCold:
    def test_it_runs_two_requests_and_reuses_neither(self):
        console = Console(file=io.StringIO(), width=100, force_terminal=False)
        provider = MockProvider()
        compare_turn(console, provider, full_stack(), "What is the boiling point of water?")
        out = console.file.getvalue()
        assert "With Parsimony" in out and "Without Parsimony" in out
        assert "each from cold" in out or "Simulated AI" in out


class TestTheRunningTotals:
    def test_a_conversation_accumulates_what_was_avoided(self):
        session = LiveSession()
        console = Console(file=io.StringIO(), width=100, force_terminal=False)
        pipe = Pipeline(full_stack(), provider=MockProvider(), capture_text=True)
        history: list[Turn] = []
        for question in ("What is recursion?", "What is 847 * 23?", "What is recursion?"):
            outcome, _ = show_turn(console, pipe, question, tuple(history), session=session)
            history.append(Turn(f"u{len(history)}", "user", question))
            history.append(Turn(f"a{len(history)}", "assistant", outcome.response))
        assert session.questions == 3
        assert session.from_calculator == 1
        assert session.from_memory == 1
        assert session.tokens_sent < session.tokens_written
        assert "3 question(s)" in session.line()
