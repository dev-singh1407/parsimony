"""Showing the text, not just the token count.

A trace row reading "-19 tokens" is auditable only by someone who trusts the
counter. These tests cover the display path that makes the same claim checkable
by eye, and — more importantly — that it stays out of the measurement path.
"""

from __future__ import annotations

from rich.console import Console

from parsimony.core.config import baseline, full_stack
from parsimony.pipeline.orchestrator import Pipeline, TextDelta
from parsimony.surfaces.cli.render import _diff_text, _split_words, text_delta_panels

VERBOSE = (
    "Hello, I was wondering if you could **please** explain to me what photosynthesis is? "
    "I would like to know how it works. Thanks in advance!"
)
GATED = "Explain the deadline. The deadline is 15 March. The deadline is 16 March."


class TestCaptureIsOffByDefault:
    """The sweep writes ~4,500 ledger rows. Carrying two payload strings per
    stage on each would multiply its size for data no analysis reads."""

    def test_no_deltas_unless_asked(self):
        assert Pipeline(full_stack()).run(VERBOSE).text_deltas == ()

    def test_ledger_row_never_carries_text(self):
        row = Pipeline(full_stack(), capture_text=True).run(VERBOSE).row
        blob = repr(row)
        assert "Thanks in advance" not in blob
        assert not hasattr(row.traces[0], "before")

    def test_capture_does_not_change_the_result(self):
        """If it did, the demo would be showing a different pipeline than the
        one that produced the numbers."""
        cold = Pipeline(full_stack()).run(VERBOSE).row
        warm = Pipeline(full_stack(), capture_text=True).run(VERBOSE).row
        assert cold.tokens_in_final == warm.tokens_in_final
        assert cold.tokens_per_module == warm.tokens_per_module


class TestWhatGetsCaptured:
    def test_an_applied_edit_records_both_sides(self):
        deltas = Pipeline(full_stack(), capture_text=True).run(VERBOSE).text_deltas
        applied = [d for d in deltas if not d.reverted and d.changed]
        assert applied
        d = applied[0]
        assert "Thanks in advance" in d.before
        assert "Thanks in advance" not in d.after

    def test_a_reverted_edit_records_what_was_refused(self):
        """The edit the gate blocked is invisible in the committed text, and it
        is the most informative thing the pipeline produces."""
        deltas = Pipeline(full_stack(), capture_text=True).run(GATED).text_deltas
        reverted = [d for d in deltas if d.reverted]
        assert reverted, "expected the gate to refuse the duplicate-sentence edit"
        d = reverted[0]
        assert "16 March" in d.before, "the kept text must retain the date"
        assert "16 March" not in d.after, "the refused candidate is what dropped it"

    def test_a_cache_hit_records_the_prompt_never_sent(self):
        p = Pipeline(full_stack(), capture_text=True)
        p.run(VERBOSE)
        deltas = p.run(VERBOSE).text_deltas
        short = [d for d in deltas if d.short_circuited]
        assert short and short[0].after == ""

    def test_baseline_captures_nothing_because_nothing_changes(self):
        deltas = Pipeline(baseline(), capture_text=True).run(VERBOSE).text_deltas
        assert not [d for d in deltas if d.changed]


class TestWordDiff:
    def test_splitting_is_lossless(self):
        for s in (VERBOSE, GATED, "", "   ", "one"):
            assert "".join(_split_words(s)) == s

    def test_deleted_words_are_marked_on_the_before_side(self):
        t = _diff_text("keep this drop that", "keep this", show="before")
        assert t.plain == "keep this drop that"
        styled = "".join(t.plain[s.start:s.end] for s in t.spans)
        assert "drop" in styled and "keep" not in styled

    def test_inserted_words_are_marked_on_the_after_side(self):
        t = _diff_text("keep this", "keep this and more", show="after")
        styled = "".join(t.plain[s.start:s.end] for s in t.spans)
        assert "more" in styled

    def test_unchanged_text_is_left_unstyled(self):
        assert _diff_text("same words", "same words", show="before").spans == []

    def test_the_gate_case_marks_the_date_that_would_have_been_lost(self):
        t = _diff_text(GATED, "Explain the deadline. The deadline is 15 March.", show="before")
        styled = "".join(t.plain[s.start:s.end] for s in t.spans)
        assert "16" in styled, "the reviewer must see which date the edit dropped"


class TestRendering:
    def test_panels_render_without_error_and_include_the_text(self):
        console = Console(width=100, record=True)
        outcome = Pipeline(full_stack(), capture_text=True).run(VERBOSE)
        text_delta_panels(console, outcome, counter=len)
        out = console.export_text()
        assert "Thanks in advance" in out

    def test_a_reverted_panel_says_so(self):
        console = Console(width=100, record=True)
        outcome = Pipeline(full_stack(), capture_text=True).run(GATED)
        text_delta_panels(console, outcome, counter=len)
        assert "REVERTED" in console.export_text()

    def test_long_text_is_elided_rather_than_flooding_the_screen(self):
        console = Console(width=100, record=True)
        long_before = "word " * 400
        outcome = type("O", (), {"text_deltas": (TextDelta("s", "M1", long_before, "word"),)})()
        text_delta_panels(console, outcome, counter=len)
        assert "more characters" in console.export_text()

    def test_unchanged_stages_are_not_shown(self):
        console = Console(width=100, record=True)
        outcome = type("O", (), {"text_deltas": (TextDelta("s", "M1", "same", "same"),)})()
        text_delta_panels(console, outcome, counter=len)
        assert console.export_text().strip() == ""
