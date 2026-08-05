"""Statistics for the factorial ablation.

Three deliberate choices, each with a reason that outlives this project:

BOOTSTRAP, NOT t-INTERVALS (ADR-023). Two reasons, in order of strength:

  1. The additivity shortfall -- the project's primary number -- is a NONLINEAR
     function of several cell aggregates (a sum of ratios, minus a ratio). It
     has no closed-form standard error, so there is no t-interval to compute.
  2. We report medians and p95 latency, which likewise have no closed form.

A weaker reason, stated precisely because it is easy to overclaim: CPU latency
is right-skewed, so a t-interval on a SMALL sample would have poor coverage. At
the sample sizes here the bootstrap distribution of a mean is close to
symmetric anyway -- the CLT does its job -- so this is about correctness of
method, not about a large numerical difference on the mean.

EFFECT SIZE AS THE HEADLINE, p AS A FOOTNOTE (ADR-021). At a few hundred
observations per cell, everything is significant. "M1xM2 interaction, partial
eta-squared 0.03" says the interaction is real but small. "p < 0.001" says only
that we collected a lot of data. The project's central claim is about the
MAGNITUDE of the shortfall from additivity, so effect size is the natural unit.

THE FRONTIER, NOT A WINNER. The deliverable is a calibrated operating curve --
which combination to switch on for a given quality floor -- not a single
headline percentage.

No scipy or statsmodels: a 2^k factorial with one aggregate per cell is exactly
computable by contrasts, and the bootstrap is twenty lines. Adding two large
dependencies to run t-tests we have argued against would be poor taste.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from typing import Callable, Sequence


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Interval:
    point: float
    low: float
    high: float

    def __str__(self) -> str:
        return f"{self.point:.2f} [{self.low:.2f}, {self.high:.2f}]"

    @property
    def excludes_zero(self) -> bool:
        return (self.low > 0) or (self.high < 0)


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], float] | None = None,
    resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap. Deterministic given `seed`, so figures reproduce."""
    if not values:
        return Interval(0.0, 0.0, 0.0)
    statistic = statistic or (lambda xs: sum(xs) / len(xs))
    observed = statistic(values)
    if len(values) == 1:
        return Interval(observed, observed, observed)

    rng = random.Random(seed)
    n = len(values)
    stats: list[float] = []
    for _ in range(resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        stats.append(statistic(sample))
    stats.sort()
    lo = stats[max(0, int((alpha / 2) * resamples) - 1)]
    hi = stats[min(resamples - 1, int((1 - alpha / 2) * resamples))]
    return Interval(observed, lo, hi)


def median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def iqr(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    ordered = sorted(values)
    q1 = ordered[len(ordered) // 4]
    q3 = ordered[(3 * len(ordered)) // 4]
    return (q1, q3)


# --------------------------------------------------------------------------
# 2^k factorial effects
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Effect:
    name: str
    order: int
    estimate: float
    sum_squares: float
    partial_eta_sq: float

    @property
    def is_interaction(self) -> bool:
        return self.order > 1


def factorial_effects(
    responses: dict[frozenset[str], float], axes: Sequence[str]
) -> list[Effect]:
    """Main effects and interactions for a single-replicate 2^k design.

    `responses` maps the set of enabled modules in a cell to that cell's
    response value (e.g. total token reduction).

    Computed by contrasts rather than by fitting a linear model: with one
    aggregate per cell the design is saturated and the contrast form is exact.

    Note on the missing error term: a saturated single-replicate design has no
    residual degrees of freedom, so there is no F test to run. Partial
    eta-squared here is each effect's share of the total variation explained,
    which is the descriptive quantity we actually want to report.
    """
    axes = list(axes)
    k = len(axes)
    cells = list(itertools.product([False, True], repeat=k))
    missing = [c for c in cells if frozenset(a for a, b in zip(axes, c) if b) not in responses]
    if missing:
        raise ValueError(f"factorial is incomplete: {len(missing)} cell(s) absent")

    values = {c: responses[frozenset(a for a, b in zip(axes, c) if b)] for c in cells}

    effects: list[Effect] = []
    total_ss = 0.0
    raw: list[tuple[str, int, float, float]] = []

    for order in range(1, k + 1):
        for combo in itertools.combinations(range(k), order):
            contrast = 0.0
            for cell, value in values.items():
                sign = 1.0
                for idx in combo:
                    sign *= 1.0 if cell[idx] else -1.0
                contrast += sign * value
            estimate = contrast / (2 ** (k - 1))
            ss = (contrast**2) / (2**k)
            total_ss += ss
            raw.append(("x".join(axes[i] for i in combo), order, estimate, ss))

    for name, order, estimate, ss in raw:
        effects.append(
            Effect(
                name=name,
                order=order,
                estimate=estimate,
                sum_squares=ss,
                partial_eta_sq=(ss / total_ss) if total_ss > 0 else 0.0,
            )
        )
    effects.sort(key=lambda e: -e.partial_eta_sq)
    return effects


# --------------------------------------------------------------------------
# Pareto frontier
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ParetoPoint:
    label: str
    reduction: float
    quality: float


def pareto_frontier(points: Sequence[ParetoPoint]) -> list[ParetoPoint]:
    """Non-dominated set: maximise reduction AND quality.

    A point is dominated if another is at least as good on both axes and
    strictly better on one.
    """
    frontier = []
    for p in points:
        dominated = any(
            (q.reduction >= p.reduction and q.quality >= p.quality)
            and (q.reduction > p.reduction or q.quality > p.quality)
            for q in points
        )
        if not dominated:
            frontier.append(p)
    return sorted(frontier, key=lambda p: p.reduction)


def knee_point(frontier: Sequence[ParetoPoint]) -> ParetoPoint | None:
    """The frontier point furthest from the line joining its extremes.

    The reportable deliverable is the frontier and its knee, not any single
    configuration.
    """
    if len(frontier) < 3:
        return frontier[-1] if frontier else None
    first, last = frontier[0], frontier[-1]
    dx = last.reduction - first.reduction
    dy = last.quality - first.quality
    norm = math.hypot(dx, dy)
    if norm == 0:
        return frontier[0]

    best, best_dist = frontier[0], -1.0
    for p in frontier:
        dist = abs(dy * p.reduction - dx * p.quality + last.reduction * first.quality
                   - last.quality * first.reduction) / norm
        if dist > best_dist:
            best, best_dist = p, dist
    return best


def frontier_above_floor(
    points: Sequence[ParetoPoint], quality_floor: float
) -> list[ParetoPoint]:
    """Configurations meeting a quality floor, best reduction first.

    This is the practitioner-facing form of the result: "at a 90% quality floor,
    switch these on."
    """
    eligible = [p for p in points if p.quality >= quality_floor]
    return sorted(eligible, key=lambda p: -p.reduction)
