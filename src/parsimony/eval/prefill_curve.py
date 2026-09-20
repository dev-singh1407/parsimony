"""How prefill actually scales, past the range the constant was fitted in.

The project converts token savings into seconds with one number: 8.5 ms per
input token (ADR-034). That was measured at 146, 474 and 1,338 tokens, where it
fits well -- and then used everywhere, including on prompts an order of
magnitude longer.

Attention is quadratic in sequence length, so it cannot stay linear. The
question is where the quadratic term starts to matter on this machine, and the
answer decides whether "8.5 ms per token" is a fair conversion or a convenient
one. Any run that records `prompt_tokens` beside `prefill_ms` is already an
observation of this, so the curve is fitted from runs that were done for other
reasons rather than from a benchmark written to prove a point.

    prefill_ms ~= a * n + b * n^2

Fitted by ordinary least squares on the two terms, with no intercept: a prompt
of zero tokens takes no time to read, and letting the line float free lets it
buy a better fit with a constant that has no physical meaning.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Curve:
    """`prefill_ms = linear_ms * n + quadratic_ms * n^2`, and what it implies."""

    n: int
    linear_ms: float           # ms per token, the term the constant captures
    quadratic_ms: float        # ms per token^2, the term it cannot
    r2: float
    min_tokens: int
    max_tokens: int

    def predict(self, tokens: int) -> float:
        return self.linear_ms * tokens + self.quadratic_ms * tokens * tokens

    def ms_per_token(self, tokens: int) -> float:
        """The effective rate at a given prompt length."""
        return self.predict(tokens) / tokens if tokens else 0.0

    def doubling_point(self) -> int | None:
        """Prompt length at which the effective rate is twice the linear term.

        This is where quoting a single ms-per-token figure stops being a
        simplification and starts being wrong: n = linear / quadratic.
        """
        if self.quadratic_ms <= 0:
            return None
        return int(self.linear_ms / self.quadratic_ms)


def observations(paths: list[Path]) -> list[tuple[int, float]]:
    """(prompt_tokens, prefill_ms) from any run that recorded both.

    Rows whose prefill is implausibly fast for their length are dropped: the
    runtime reuses work on an identical prefix, and those rows measure the
    key-value cache rather than prefill (ADR-034). One millisecond per token is
    far below anything this machine does cold.
    """
    out: list[tuple[int, float]] = []
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            tokens, ms = row.get("prompt_tokens"), row.get("prefill_ms")
            if not tokens or not ms or ms <= 0:
                continue
            if ms / tokens < 1.0:
                continue
            out.append((int(tokens), float(ms)))
    return out


def fit(points: list[tuple[int, float]]) -> Curve | None:
    """Least squares for a*n + b*n^2, solved directly on the 2x2 normal equations."""
    if len(points) < 4:
        return None
    s11 = s12 = s22 = t1 = t2 = 0.0
    for n, ms in points:
        x1, x2 = float(n), float(n) ** 2
        s11 += x1 * x1
        s12 += x1 * x2
        s22 += x2 * x2
        t1 += x1 * ms
        t2 += x2 * ms
    det = s11 * s22 - s12 * s12
    if det == 0:
        return None
    a = (t1 * s22 - t2 * s12) / det
    b = (t2 * s11 - t1 * s12) / det
    mean = sum(ms for _, ms in points) / len(points)
    ss_res = sum((ms - (a * n + b * n * n)) ** 2 for n, ms in points)
    ss_tot = sum((ms - mean) ** 2 for _, ms in points)
    return Curve(n=len(points), linear_ms=a, quadratic_ms=b,
                 r2=1 - ss_res / ss_tot if ss_tot else 0.0,
                 min_tokens=min(n for n, _ in points),
                 max_tokens=max(n for n, _ in points))


def rate_table(points: list[tuple[int, float]], edges=(500, 1000, 2000, 4000, 8000, 16000)):
    """Measured ms-per-token by prompt-length band, alongside the fitted curve.

    The bands are the honest version: a curve can be argued with, but the
    average rate actually observed between 4,000 and 8,000 tokens cannot.
    """
    curve = fit(points)
    bands = []
    lo = 0
    for hi in edges:
        inside = [(n, ms) for n, ms in points if lo < n <= hi]
        if inside:
            bands.append({
                "from": lo, "to": hi, "n": len(inside),
                "ms_per_token": sum(ms / n for n, ms in inside) / len(inside),
                "fitted": curve.ms_per_token((lo + hi) // 2) if curve else None,
            })
        lo = hi
    return bands
