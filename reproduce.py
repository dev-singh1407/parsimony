#!/usr/bin/env python
"""Regenerate every table and figure in the report from raw data.

    python reproduce.py --out figures/

Built early and grown alongside the results rather than written at the end: a
reproduction script authored in the final week is a script that was never
tested. Growing it in step also makes it a schema tripwire (ADR-014) -- every
section declares the ledger fields it consumes, so a change that drops a
consumed field fails here instead of silently producing an empty column in
October.

Deterministic: fixed seeds throughout, so re-running produces byte-identical
output.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).parent / "src"))

from parsimony.core.config import factorial_cells, full_stack  # noqa: E402
from parsimony.core.types import Mode  # noqa: E402
from parsimony.eval.calibration import by_operative, sweep_thresholds  # noqa: E402
from parsimony.eval.corpus import load_adversarial, load_corpus, load_gold  # noqa: E402
from parsimony.eval.metrics import LengthBiasedMockJudge  # noqa: E402
from parsimony.eval.runner import (  # noqa: E402
    CellResult,
    additivity_shortfall,
    shortfall_interval,
    sweep,
)
from parsimony.eval.stats import (  # noqa: E402
    ParetoPoint,
    bootstrap_ci,
    factorial_effects,
    frontier_above_floor,
    knee_point,
    pareto_frontier,
)
from parsimony.eval.tokenizer_probe import run_probe  # noqa: E402
from parsimony.infra.embedding import get_embedder  # noqa: E402
from parsimony.infra.memo import GenerationMemo  # noqa: E402
from parsimony.infra.tokenization import get_tokenizer  # noqa: E402

AXES = ("M1", "M2", "M3", "M5")
QUALITY_FLOOR = 90.0


@dataclass
class Section:
    """A report section, with the ledger fields it consumes declared.

    The declaration is the tripwire: `check_schema` fails loudly if a field a
    section reads has disappeared from LedgerRow.
    """

    key: str
    title: str
    consumes: tuple[str, ...]
    render: Callable[["Context"], str]


@dataclass
class Context:
    results: list[CellResult]
    corpus: object
    out: Path


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _write_csv(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(headers)
        w.writerows(rows)


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------


def render_ablation(ctx: Context) -> str:
    headers = ["cell", "input", "output", "total", "reduction %", "cache hits",
               "tier 0", "gate fires", "early stops", "middleware ms"]
    rows = [
        [r.label, str(r.tokens_in_final), str(r.tokens_out), str(r.total_tokens),
         f"{r.total_reduction_pct:+.1f}", str(r.cache_hits), str(r.deterministic_hits),
         str(r.gate_fires), str(r.early_stops), f"{r.middleware_mean_ms:.2f}"]
        for r in ctx.results
    ]
    _write_csv(ctx.out / "ablation.csv", headers, rows)
    return _table(headers, rows)


def render_quality(ctx: Context) -> str:
    headers = ["cell", "embedding sim %", "token overlap %", "judge %",
               "judge disagreement %", "gold correct", "gold %"]
    rows = [
        [r.label,
         f"{r.quality_embedding:.1f}" if r.q_embedding else "ref",
         f"{r.quality_overlap:.1f}" if r.q_overlap else "ref",
         f"{r.quality_judge:.1f}" if r.q_judge else "ref",
         f"{r.judge_disagreement_rate:.0f}" if r.q_judge else "-",
         f"{r.gold_correct}/{r.gold_total}", f"{r.gold_accuracy:.1f}"]
        for r in ctx.results
    ]
    _write_csv(ctx.out / "quality.csv", headers, rows)
    return (
        _table(headers, rows)
        + "\n\nThe four measures are never averaged. Only `gold` is ground truth; the other "
        "three compare against the baseline's own answer. `token overlap` is structurally "
        "biased against M5, whose job is to produce shorter answers. A high judge "
        "disagreement rate means the judge flipped when the options were swapped and its "
        "score should be discounted."
    )


def render_effects(ctx: Context) -> str:
    by_label = {r.label: r for r in ctx.results}
    responses: dict[frozenset[str], float] = {}
    for r in ctx.results:
        modules = frozenset() if r.label == "baseline" else frozenset(r.label.split("+"))
        if modules <= set(AXES):
            responses[modules] = r.total_reduction_pct
    try:
        effects = factorial_effects(responses, AXES)
    except ValueError as exc:
        return f"_Factorial incomplete: {exc}_"

    headers = ["effect", "order", "estimate (pp)", "partial eta squared"]
    rows = [[e.name, str(e.order), f"{e.estimate:+.2f}", f"{e.partial_eta_sq:.4f}"]
            for e in effects]
    _write_csv(ctx.out / "effects.csv", headers, rows)

    interactions = [e for e in effects if e.is_interaction]
    largest = max(interactions, key=lambda e: abs(e.estimate)) if interactions else None
    note = (
        f"\n\nLargest interaction: **{largest.name}**, estimate {largest.estimate:+.2f} pp, "
        f"partial eta squared {largest.partial_eta_sq:.4f}."
        if largest else ""
    )
    return (
        _table(headers, rows)
        + note
        + "\n\nEffect size is the headline; there is no p column. With a saturated "
        "single-replicate design there are no residual degrees of freedom to test "
        "against, and at this number of observations a p-value would report sample "
        "size rather than importance."
    )


def render_shortfall(ctx: Context) -> str:
    summary = additivity_shortfall(ctx.results, AXES)
    if "shortfall_pct" not in summary:
        return "_Not enough solo cells to compute the shortfall._"
    interval = shortfall_interval(ctx.results, AXES)
    ci = f" (95% CI [{interval.low:+.2f}, {interval.high:+.2f}])" if interval else ""
    return (
        f"- Sum of individual reductions: **{summary['predicted_additive_pct']:.2f}%**\n"
        f"- Measured stacked reduction ({summary['stacked_label']}): "
        f"**{summary['measured_stacked_pct']:.2f}%**\n"
        f"- **Additivity shortfall: {summary['shortfall_pct']:.2f} percentage points{ci}**\n\n"
        "Bootstrap resamples conversations, not requests: turns within a conversation "
        "share history and are not independent observations.\n\n"
        "Quantifying this shortfall is the primary result. No published study runs these "
        "modules in one pipeline, so the field has no evidence about whether their savings "
        "compound."
    )


def render_pareto(ctx: Context) -> str:
    points = [
        ParetoPoint(r.label, r.total_reduction_pct,
                    r.quality_embedding if r.q_embedding else 100.0)
        for r in ctx.results
    ]
    frontier = pareto_frontier(points)
    knee = knee_point(frontier)
    eligible = frontier_above_floor(points, QUALITY_FLOOR)

    headers = ["cell", "token reduction %", "quality retained %", "on frontier"]
    on = {p.label for p in frontier}
    rows = [[p.label, f"{p.reduction:+.1f}", f"{p.quality:.1f}",
             "yes" if p.label in on else ""] for p in points]
    _write_csv(ctx.out / "pareto.csv", headers, rows)

    best = eligible[0] if eligible else None
    return (
        _table(headers, rows)
        + f"\n\n- Frontier: {', '.join(p.label for p in frontier)}\n"
        + (f"- Knee point: **{knee.label}** "
           f"({knee.reduction:+.1f}% reduction, {knee.quality:.1f}% quality)\n" if knee else "")
        + (f"- Best configuration meeting a {QUALITY_FLOOR:.0f}% quality floor: "
           f"**{best.label}** at {best.reduction:+.1f}%\n" if best
           else f"- No configuration meets the {QUALITY_FLOOR:.0f}% quality floor.\n")
        + "\nThe deliverable is the frontier and its knee, not a single configuration."
    )


def render_calibration(ctx: Context) -> str:
    base = full_stack()
    pairs = load_adversarial()
    embedder = get_embedder(base.embedder_id)

    headers = ["tau_hi", "false hits", "false-hit rate %", "true hits",
               "true-hit rate %", "meets <2% target"]
    rows = []
    for arm_name, verifier_on in (("verifier on", True), ("verifier off", False)):
        for p in sweep_thresholds(base, pairs=pairs, embedder=embedder,
                                  verifier_on=verifier_on):
            rows.append([f"{p.tau_hi:.2f} ({arm_name})",
                         f"{p.false_hits}/{p.adversarial_total}",
                         f"{p.false_hit_rate:.1f}",
                         f"{p.true_hits}/{p.control_total}",
                         f"{p.true_hit_rate:.1f}",
                         "yes" if p.is_safe else "no"])
    _write_csv(ctx.out / "calibration.csv", headers, rows)

    breakdown = by_operative(pairs, base, embedder, verifier_on=True)
    bh = ["operative", "false hits", "rate %"]
    br = [[op, f"{h}/{t}", f"{100.0 * h / t if t else 0:.0f}"]
          for op, (h, t) in breakdown.items()]
    _write_csv(ctx.out / "calibration_by_operative.csv", bh, br)

    return (
        _table(headers, rows)
        + "\n\n### False hits by operative token, at the operating point\n\n"
        + _table(bh, br)
        + "\n\nThresholds are calibrated against the active encoder and do not transfer. "
        "Similarity alone cannot reach the target at any threshold, because adversarial "
        "pairs sit at higher cosine than genuine paraphrases; the verifier is what makes "
        "the cache safe."
    )


def render_tokenprobe(ctx: Context) -> str:
    texts = [q for c in ctx.corpus.conversations for q in c.user_turns]
    results = run_probe(texts, get_tokenizer())
    headers = ["edit regime", "tested", "saved tokens", "saved nothing",
               "cost tokens", "wasted %"]
    rows = [[r.regime, str(r.tested), str(r.reduced), str(r.neutral),
             str(r.increased), f"{r.wasted_pct:.0f}"] for r in results]
    _write_csv(ctx.out / "tokenprobe.csv", headers, rows)
    return (
        _table(headers, rows)
        + "\n\nWhitespace-aligned edits are monotone under this tokenizer. Sub-token edits "
        "are not. The negative-yield guard therefore earns its place mainly by rejecting "
        "ZERO-yield edits, which perturb the text for no saving at all."
    )


def render_middleware(ctx: Context) -> str:
    headers = ["cell", "mean ms", "95% CI", "p95 ms", "mean prefix tokens reused"]
    rows = []
    for r in ctx.results:
        ci = bootstrap_ci(r.middleware_ms, resamples=2000)
        rows.append([r.label, f"{r.middleware_mean_ms:.2f}",
                     f"[{ci.low:.2f}, {ci.high:.2f}]",
                     f"{r.middleware_p95_ms:.2f}",
                     f"{r.mean_prefix_tokens:.1f}"])
    _write_csv(ctx.out / "middleware.csv", headers, rows)
    return (
        _table(headers, rows)
        + "\n\nMiddleware overhead excludes generation. Percentile bootstrap, not "
        "t-intervals: latency is right-skewed and a symmetric interval understates the "
        "upper tail.\n\n"
        "**These timings come from MockProvider and are simulated.** They measure the "
        "middleware, which is real, but not model latency, which is not."
    )


SECTIONS: tuple[Section, ...] = (
    Section("ablation", "Factorial ablation",
            ("tokens_in_final", "tokens_out", "cache_hit", "route_tier", "gate_fired",
             "early_stopped", "middleware_ns"), render_ablation),
    Section("quality", "Quality — four measures, never averaged",
            ("q_embedding_sim", "q_token_overlap", "q_judge", "q_exact_match"),
            render_quality),
    Section("effects", "Main effects and interactions", ("tokens_in_final", "tokens_out"),
            render_effects),
    Section("shortfall", "Additivity shortfall", ("tokens_in_final", "tokens_out"),
            render_shortfall),
    Section("pareto", "Pareto frontier", ("tokens_in_final", "tokens_out",
                                          "q_embedding_sim"), render_pareto),
    Section("calibration", "Cache threshold calibration",
            ("cache_top_k", "cache_zone", "cache_verifier"), render_calibration),
    Section("tokenprobe", "Negative-yield probe", (), render_tokenprobe),
    Section("middleware", "Middleware overhead and prefix reuse",
            ("middleware_ns", "prefix_tokens_survived"), render_middleware),
)


def check_schema() -> list[str]:
    """ADR-014 tripwire: every declared field must still exist on LedgerRow."""
    from parsimony.core.ledger import LedgerRow

    known = set(LedgerRow.__dataclass_fields__)
    missing = []
    for section in SECTIONS:
        for field_name in section.consumes:
            if field_name not in known:
                missing.append(f"{section.key} -> {field_name}")
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("figures"))
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--no-quality", action="store_true")
    parser.add_argument("--no-memo", action="store_true",
                        help="Disable generation memoisation (the timing-pass setting).")
    args = parser.parse_args()

    missing = check_schema()
    if missing:
        print("SCHEMA DRIFT — sections consume fields that no longer exist:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus(args.corpus)
    print(f"corpus {len(corpus)} conversations, {corpus.n_requests} requests "
          f"(hash {corpus.corpus_hash})")

    cells = list(factorial_cells(axes=AXES, always_on=frozenset()))
    cells.append(
        cells[-1].with_modules(cells[-1].enabled_modules | {"M4", "M6"},
                               label="+".join(AXES) + "+M4+M6")
    )
    # EXPERIMENT mode: a failed ledger write becomes fatal (the ledger IS the
    # result), and it is the only mode in which the generation memo is
    # consulted at all — a served request must never receive a memoised answer.
    cells = [replace(c, mode=Mode.EXPERIMENT) for c in cells]
    # Quality pass: memo ON. Bit-exact at temperature 0, so this is a pure
    # compute optimisation (ADR-019). Latency from this pass is meaningless and
    # every row is flagged `generation_memoised` so analysis can exclude it.
    memo = None if args.no_memo else GenerationMemo()
    print(f"running {len(cells)} cells (memo {'off' if args.no_memo else 'on'})...")
    results = sweep(
        cells, corpus,
        memo=memo,
        judge=None if args.no_quality else LengthBiasedMockJudge(),
        gold=() if args.no_quality else load_gold(),
        progress=lambda label: print(f"  {label}"),
    )
    if memo is not None:
        print(f"  memo: {memo.hits} hits / {memo.hits + memo.misses} generations "
              f"({memo.hit_rate:.1f}% avoided)")

    ctx = Context(results=results, corpus=corpus, out=args.out)
    parts = [
        "# Parsimony — reproduced results",
        "",
        f"Corpus hash `{corpus.corpus_hash}` · {len(corpus)} conversations · "
        f"{corpus.n_requests} requests · {len(cells)} cells",
        "",
        "Generated by `reproduce.py`. Every number below comes from a live run; "
        "nothing is transcribed by hand.",
        "",
    ]
    for section in SECTIONS:
        print(f"  rendering {section.key}")
        parts += [f"## {section.title}", "", section.render(ctx), ""]

    report = args.out / "report.md"
    report.write_text("\n".join(parts), encoding="utf-8")
    print(f"\nwrote {report} and {len(list(args.out.glob('*.csv')))} CSV files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
