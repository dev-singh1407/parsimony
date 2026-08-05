"""Bootstrap intervals, factorial effects, and the Pareto frontier."""

from __future__ import annotations

import pytest

from parsimony.eval.stats import (
    ParetoPoint,
    bootstrap_ci,
    factorial_effects,
    frontier_above_floor,
    iqr,
    knee_point,
    median,
    pareto_frontier,
)


class TestBootstrap:
    def test_point_estimate_is_the_observed_statistic(self):
        assert bootstrap_ci([1, 2, 3, 4, 5], resamples=500).point == pytest.approx(3.0)

    def test_interval_brackets_the_point(self):
        ci = bootstrap_ci([1, 2, 3, 4, 5, 6, 7, 8], resamples=1000)
        assert ci.low <= ci.point <= ci.high

    def test_is_deterministic_for_a_fixed_seed(self):
        """Figures must reproduce byte-identically."""
        a = bootstrap_ci([3, 1, 4, 1, 5, 9, 2, 6], resamples=500, seed=7)
        b = bootstrap_ci([3, 1, 4, 1, 5, 9, 2, 6], resamples=500, seed=7)
        assert (a.low, a.point, a.high) == (b.low, b.point, b.high)

    def test_narrows_as_the_sample_grows(self):
        narrow = bootstrap_ci([5.0] * 200 + [6.0] * 200, resamples=800)
        wide = bootstrap_ci([5.0, 6.0], resamples=800)
        assert (narrow.high - narrow.low) < (wide.high - wide.low)

    def test_works_for_statistics_with_no_closed_form(self):
        """The actual reason bootstrap is used here.

        The additivity shortfall is a nonlinear function of several aggregates
        and has no analytic standard error; so do the median and p95 we report.
        A t-interval is not merely less accurate for these — it does not exist.
        """
        values = [1.0] * 90 + [50.0] * 10
        med = bootstrap_ci(values, statistic=median, resamples=2000)
        p95 = bootstrap_ci(
            values, statistic=lambda xs: sorted(xs)[int(0.95 * len(xs)) - 1], resamples=2000
        )
        assert med.low <= med.point <= med.high
        assert p95.point > med.point

    def test_a_skewed_sample_still_produces_a_valid_interval(self):
        values = [1.0] * 90 + [50.0] * 10
        ci = bootstrap_ci(values, resamples=3000)
        assert ci.low < ci.point < ci.high

    def test_empty_input_is_safe(self):
        assert bootstrap_ci([]).point == 0.0

    def test_excludes_zero_detects_a_real_effect(self):
        assert bootstrap_ci([5.0] * 50, resamples=500).excludes_zero
        assert not bootstrap_ci([-1.0, 1.0] * 50, resamples=500).excludes_zero


class TestFactorialEffects:
    def _additive(self):
        """A perfectly additive design: A adds 10, B adds 4, no interaction."""
        return {
            frozenset(): 0.0,
            frozenset({"A"}): 10.0,
            frozenset({"B"}): 4.0,
            frozenset({"A", "B"}): 14.0,
        }

    def test_recovers_main_effects(self):
        effects = {e.name: e for e in factorial_effects(self._additive(), ("A", "B"))}
        assert effects["A"].estimate == pytest.approx(10.0)
        assert effects["B"].estimate == pytest.approx(4.0)

    def test_additive_design_has_no_interaction(self):
        effects = {e.name: e for e in factorial_effects(self._additive(), ("A", "B"))}
        assert effects["AxB"].estimate == pytest.approx(0.0)
        assert effects["AxB"].partial_eta_sq == pytest.approx(0.0)

    def test_detects_a_negative_interaction(self):
        """Overlapping modules: together they deliver less than the sum."""
        responses = dict(self._additive())
        responses[frozenset({"A", "B"})] = 11.0  # 3pp short of additive
        effects = {e.name: e for e in factorial_effects(responses, ("A", "B"))}
        assert effects["AxB"].estimate < 0
        assert effects["AxB"].partial_eta_sq > 0

    def test_effect_shares_sum_to_one(self):
        effects = factorial_effects(self._additive(), ("A", "B"))
        assert sum(e.partial_eta_sq for e in effects) == pytest.approx(1.0)

    def test_results_are_ordered_by_effect_size(self):
        effects = factorial_effects(self._additive(), ("A", "B"))
        shares = [e.partial_eta_sq for e in effects]
        assert shares == sorted(shares, reverse=True)

    def test_incomplete_factorial_is_an_error(self):
        """A missing cell must fail loudly rather than silently biasing effects."""
        with pytest.raises(ValueError, match="incomplete"):
            factorial_effects({frozenset(): 0.0, frozenset({"A"}): 1.0}, ("A", "B"))


class TestPareto:
    def _points(self):
        return [
            ParetoPoint("baseline", 0.0, 100.0),
            ParetoPoint("a", 10.0, 98.0),
            ParetoPoint("b", 20.0, 95.0),
            ParetoPoint("dominated", 5.0, 90.0),
            ParetoPoint("c", 40.0, 80.0),
        ]

    def test_excludes_dominated_configurations(self):
        labels = {p.label for p in pareto_frontier(self._points())}
        assert "dominated" not in labels

    def test_keeps_non_dominated_configurations(self):
        labels = {p.label for p in pareto_frontier(self._points())}
        assert {"baseline", "a", "b", "c"} <= labels

    def test_frontier_is_sorted_by_reduction(self):
        frontier = pareto_frontier(self._points())
        assert [p.reduction for p in frontier] == sorted(p.reduction for p in frontier)

    def test_knee_is_on_the_frontier(self):
        frontier = pareto_frontier(self._points())
        assert knee_point(frontier) in frontier

    def test_quality_floor_filters_and_ranks(self):
        eligible = frontier_above_floor(self._points(), 90.0)
        assert all(p.quality >= 90.0 for p in eligible)
        assert eligible[0].label == "b"  # best reduction among those above the floor

    def test_impossible_floor_returns_nothing(self):
        assert frontier_above_floor(self._points(), 101.0) == []


class TestRobustStatistics:
    def test_median_of_even_and_odd_samples(self):
        assert median([1, 2, 3]) == 2
        assert median([1, 2, 3, 4]) == 2.5

    def test_iqr_brackets_the_median(self):
        values = list(range(100))
        q1, q3 = iqr(values)
        assert q1 < median(values) < q3

    def test_median_resists_an_outlier_that_moves_the_mean(self):
        values = [1.0] * 99 + [1000.0]
        assert median(values) == 1.0
        assert sum(values) / len(values) > 10
