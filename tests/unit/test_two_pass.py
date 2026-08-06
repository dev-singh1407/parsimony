"""The two-pass sweep and its resumability.

docs/05-evaluation-harness.md specified this design; the runner did not
implement it. CompletionLog existed and was tested but was never wired into
sweep() — the third instance of documented-but-not-built found in this project,
after cache_lookup_on="BOTH" and the layering enforcement.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from parsimony.core.config import factorial_cells
from parsimony.eval.corpus import load_corpus
from parsimony.eval.runner import two_pass_sweep


@pytest.fixture(scope="module")
def small_corpus():
    return load_corpus().subset(8)


@pytest.fixture(scope="module")
def cells():
    return list(factorial_cells(axes=("M1", "M2"), always_on=frozenset()))


class TestTwoPassSweep:
    def test_runs_both_passes(self, cells, small_corpus):
        report = two_pass_sweep(cells, small_corpus, timing_subset=4, timing_repeats=1)
        assert report.quality
        assert report.timing

    def test_quality_pass_uses_the_full_corpus(self, cells, small_corpus):
        report = two_pass_sweep(cells, small_corpus, timing_subset=4, timing_repeats=1)
        assert report.quality_corpus_size == len(small_corpus)

    def test_timing_pass_uses_a_smaller_subset(self, cells, small_corpus):
        report = two_pass_sweep(cells, small_corpus, timing_subset=4, timing_repeats=1)
        assert report.timing_corpus_size <= len(small_corpus)

    def test_quality_pass_is_memoised(self, cells, small_corpus):
        """The whole point: an estimated 36 days of CPU becomes tractable."""
        report = two_pass_sweep(cells, small_corpus, timing_subset=4, timing_repeats=1)
        assert report.memo_total > 0
        assert report.memo_hits > 0

    def test_repeats_are_merged_into_one_cell_per_label(self, cells, small_corpus):
        """Latency needs a real sample; two repeats of a cell are one result
        with twice the observations, not two competing rows."""
        one = two_pass_sweep(cells, small_corpus, timing_subset=4, timing_repeats=1)
        two = two_pass_sweep(cells, small_corpus, timing_subset=4, timing_repeats=2)
        assert len(one.timing) == len(two.timing) == len(cells)
        one_obs = sum(len(c.middleware_ms) for c in one.timing)
        two_obs = sum(len(c.middleware_ms) for c in two.timing)
        assert two_obs > one_obs

    def test_timing_rows_are_never_memoised(self, cells, small_corpus):
        """A memo hit takes microseconds. If one reached the timing pass, every
        latency figure in the report would be flattered by it."""
        from parsimony.infra.storage import MemorySink

        sink = MemorySink()
        two_pass_sweep(cells, small_corpus, timing_subset=4, timing_repeats=1, sink=sink)
        timing_rows = [r for r in sink.rows if r.pass_kind == "timing"]
        assert timing_rows
        assert not any(r.generation_memoised for r in timing_rows)

    def test_both_passes_are_labelled_in_the_ledger(self, cells, small_corpus):
        from parsimony.infra.storage import MemorySink

        sink = MemorySink()
        two_pass_sweep(cells, small_corpus, timing_subset=4, timing_repeats=1, sink=sink)
        assert {r.pass_kind for r in sink.rows} == {"quality", "timing"}


class TestResumability:
    def test_a_second_run_skips_completed_cells(self, cells, small_corpus):
        """An interruption at hour 14 should cost one cell, not 14 hours."""
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "completed.log"
            first = two_pass_sweep(cells, small_corpus, timing_subset=4,
                                   timing_repeats=1, resume_log=log)
            second = two_pass_sweep(cells, small_corpus, timing_subset=4,
                                    timing_repeats=1, resume_log=log)
            assert first.resumed == 0
            assert second.resumed == len(cells)
            assert second.timing == []

    def test_without_a_log_nothing_is_skipped(self, cells, small_corpus):
        """The default must always regenerate everything: a reproduction script
        that silently reuses stale work is not a reproduction script."""
        two_pass_sweep(cells, small_corpus, timing_subset=4, timing_repeats=1)
        again = two_pass_sweep(cells, small_corpus, timing_subset=4, timing_repeats=1)
        assert again.resumed == 0
        assert again.timing

    def test_repeats_are_tracked_separately(self, cells, small_corpus):
        """Repeat 0 completing must not mark repeat 1 as done."""
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "completed.log"
            two_pass_sweep(cells, small_corpus, timing_subset=4,
                           timing_repeats=1, resume_log=log)
            second = two_pass_sweep(cells, small_corpus, timing_subset=4,
                                    timing_repeats=2, resume_log=log)
            assert second.resumed == len(cells)   # repeat 0 skipped
            assert len(second.timing) == len(cells)  # repeat 1 still ran


class TestNoSilentFallback:
    def test_middleware_section_refuses_memoised_data(self):
        """Regression on a bug in the fix itself: `ctx.timing or ctx.results`
        silently fell back to the memoised quality pass when the timing pass was
        fully resumed, reporting dictionary-lookup times as latency."""
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        import reproduce

        ctx = reproduce.Context(results=[], corpus=None, out=Path("."), timing=[])
        rendered = reproduce.render_middleware(ctx)
        assert "No unmemoised timing data" in rendered
        assert "|" not in rendered.split("\n")[0]  # a notice, not a table of fake numbers
