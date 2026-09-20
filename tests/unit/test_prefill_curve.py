"""Fitting prefill against prompt length, and refusing to over-read the fit.

The project converts tokens into seconds with a single constant. These tests
pin the arithmetic that says when that is fair, and -- more importantly -- the
filtering that stops a cache hit being read as a very fast prefill, which would
drag the constant down and overstate every saving derived from it.
"""

from __future__ import annotations

import json

import pytest

from parsimony.eval.prefill_curve import Curve, fit, observations, rate_table


class TestTheFit:
    def test_it_recovers_a_known_curve(self):
        a, b = 8.0, 2e-3
        points = [(n, a * n + b * n * n) for n in (200, 500, 1000, 2000, 4000, 8000)]
        curve = fit(points)
        assert curve.linear_ms == pytest.approx(a, rel=1e-6)
        assert curve.quadratic_ms == pytest.approx(b, rel=1e-6)
        assert curve.r2 == pytest.approx(1.0, abs=1e-9)

    def test_a_purely_linear_machine_gets_no_quadratic_term(self):
        points = [(n, 9.0 * n) for n in (100, 400, 900, 1600, 2500, 4900)]
        curve = fit(points)
        assert curve.linear_ms == pytest.approx(9.0, rel=1e-6)
        assert curve.quadratic_ms == pytest.approx(0.0, abs=1e-9)
        assert curve.doubling_point() is None, "nothing doubles if nothing curves"

    def test_the_doubling_point_is_where_the_two_terms_are_equal(self):
        curve = Curve(n=9, linear_ms=10.0, quadratic_ms=1e-3, r2=1.0,
                      min_tokens=100, max_tokens=9000)
        assert curve.doubling_point() == 10_000
        assert curve.ms_per_token(10_000) == pytest.approx(2 * curve.linear_ms)

    def test_too_few_points_is_no_fit_rather_than_a_confident_one(self):
        assert fit([(100, 900.0), (200, 1800.0)]) is None

    def test_the_rate_grows_with_length(self):
        curve = fit([(n, 8.0 * n + 2e-3 * n * n) for n in (200, 800, 1600, 3200, 6400)])
        rates = [curve.ms_per_token(n) for n in (500, 2000, 8000)]
        assert rates == sorted(rates), "a quadratic term must show up as a rising rate"


class TestWhatIsExcluded:
    def test_a_reused_prefix_is_not_counted_as_prefill(self, tmp_path):
        """The runtime reports the full prompt count even when its key-value
        cache made those tokens free. Left in, those rows pull the fitted rate
        down and every second-saved figure derived from it goes up."""
        rows = [
            {"prompt_tokens": 800, "prefill_ms": 9600},    # 12.0 ms/token: real
            {"prompt_tokens": 800, "prefill_ms": 48},      # 0.06 ms/token: cache
            {"prompt_tokens": 400, "prefill_ms": 4800},
            {"prompt_tokens": 1600, "prefill_ms": 21000},
        ]
        path = tmp_path / "rows.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        got = observations([path])
        assert (800, 48.0) not in got
        assert len(got) == 3

    def test_rows_without_both_numbers_are_skipped(self, tmp_path):
        rows = [{"prompt_tokens": 500}, {"prefill_ms": 100}, {"prompt_tokens": 0,
                                                              "prefill_ms": 0}]
        path = tmp_path / "rows.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        assert observations([path]) == []

    def test_a_missing_file_is_not_an_error(self, tmp_path):
        assert observations([tmp_path / "absent.jsonl"]) == []


class TestTheBands:
    def test_it_reports_what_was_measured_not_only_what_was_fitted(self):
        points = [(n, 8.0 * n + 2e-3 * n * n) for n in
                  (100, 300, 700, 900, 1500, 1900, 5000, 7000)]
        bands = rate_table(points)
        assert bands, "bands are the part of this that cannot be argued with"
        for band in bands:
            assert band["n"] >= 1
            assert band["ms_per_token"] > 0
        rates = [b["ms_per_token"] for b in bands]
        assert rates == sorted(rates)

    def test_an_empty_band_is_omitted_rather_than_shown_as_zero(self):
        points = [(n, 9.0 * n) for n in (100, 200, 300, 400)]
        bands = rate_table(points)
        assert [b["to"] for b in bands] == [500]
