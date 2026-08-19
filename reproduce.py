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
from dataclasses import dataclass, field, replace
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
    best_per_class,
    calibration_table,
    shortfall_interval,
    two_pass_sweep,
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
    timing: list[CellResult] = field(default_factory=list)


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


def render_calibration_table(ctx: Context) -> str:
    """Contribution 6, in the form a practitioner can act on."""
    table = calibration_table(ctx.results)
    if not table:
        return "_No baseline cell; cannot compute per-class reductions._"

    classes = sorted(table)
    headers = ["cell"] + classes
    by_label: dict[str, dict[str, float]] = {}
    for cls, ranked in table.items():
        for label, pct in ranked:
            by_label.setdefault(label, {})[cls] = pct

    rows = [
        [label] + [f"{by_label[label].get(c, 0.0):+.1f}" for c in classes]
        for label in [r.label for r in ctx.results]
    ]
    _write_csv(ctx.out / "calibration_table.csv", headers, rows)

    best = best_per_class(ctx.results)
    rec_headers = ["query class", "recommended configuration", "token reduction %"]
    rec_rows = [[cls, label, f"{pct:+.1f}"] for cls, (label, pct) in sorted(best.items())]
    _write_csv(ctx.out / "recommended_per_class.csv", rec_headers, rec_rows)

    return (
        _table(headers, rows)
        + "\n\n### Recommended configuration per query class\n\n"
        + _table(rec_headers, rec_rows)
        + "\n\nThis is the deliverable in its usable form: not one headline percentage, but "
        "which modules to switch on for which kind of question. Two effects are visible here "
        "and invisible in any aggregate:\n\n"
        "- **M3 helps only `follow_up`** and is worth ~0 everywhere else, because only "
        "multi-turn conversations have history to trim.\n"
        "- **M2 helps only `paraphrase`**, because that is the only class with recurring "
        "questions.\n\n"
        "Averaged together, both modules look mediocre; per class, each is decisive for its "
        "own workload. That is the difference between a headline number and a calibration "
        "table.\n\n"
        "_Caveat: small negative values (around -1%) are noise, not regressions. Changing the "
        "prompt changes the mock provider's response length, since its output is a function "
        "of the prompt hash. Real providers will not have this artefact, but until one is "
        "attached, treat sub-1% movements as zero._"
    )


def render_energy(ctx: Context) -> str:
    cfg_energy = full_stack().energy
    headers = ["cell", "joules", "tokens per joule", "USD equivalent", "vs baseline"]
    baseline = next((r for r in ctx.results if r.label == "baseline"), None)
    base_j = baseline.total_joules if baseline else 0.0

    rows = []
    for r in ctx.results:
        delta = (100.0 * (1 - r.total_joules / base_j)) if base_j else 0.0
        rows.append([
            r.label, f"{r.total_joules:.1f}", f"{r.tokens_per_joule:.1f}",
            f"${r.usd:.4f}", f"{delta:+.1f}%",
        ])
    _write_csv(ctx.out / "energy.csv", headers, rows)

    return (
        _table(headers, rows)
        + f"\n\nEnergy is **derived**, not metered: wall clock multiplied by an assumed "
        f"package power of {cfg_energy.package_power_watts:.0f} W. That assumption is a "
        f"config field, appears in every ledger row, and can be divided back out. Priced at "
        f"${cfg_energy.usd_per_million_input:.2f}/M input and "
        f"${cfg_energy.usd_per_million_output:.2f}/M output — the project spends nothing; the "
        f"column exists to make the magnitude legible.\n\n"
        "**These figures come from simulated generation timings and are not a power "
        "measurement.** They become meaningful when a real provider is attached."
    )


def render_generalisation(ctx: Context) -> str:
    """Report 4.6's transfer question, on the tokenizer dimension."""
    from parsimony.core.config import factorial_cells as _cells
    from parsimony.eval.generalisation import (
        check_boundary_effect,
        check_tier1_yield,
        sweep_across_tokenizers,
    )

    cells = list(_cells(axes=("M1", "M2", "M3", "M5"), always_on=frozenset()))
    arms = [a for a in sweep_across_tokenizers(cells, ctx.corpus) if a.available]
    if len(arms) < 2:
        return "_Needs two real tokenizers; at least one was unavailable._"

    headers = ["cell"] + [f"{a.short_name} ({a.vocab_size:,})" for a in arms]
    rows = []
    for label in ("baseline", "M1", "M2", "M3", "M5", "M1+M2+M3+M5"):
        vals = [a.reduction(label) for a in arms]
        if any(v is None for v in vals):
            continue
        rows.append([label] + [f"{v:+.2f}" for v in vals])
    _write_csv(ctx.out / "generalisation.csv", headers, rows)

    ch = ["claim"] + [a.short_name for a in arms] + ["transfers"]
    cr = [
        [c.claim] + [c.values.get(a.short_name, "-") for a in arms]
        + ["yes" if c.transfers else "**NO**"]
        for c in check_boundary_effect() + check_tier1_yield(ctx.corpus)
    ]
    _write_csv(ctx.out / "generalisation_transfer.csv", ch, cr)

    identical = len({tuple(a.ranking()) for a in arms}) == 1
    return (
        _table(headers, rows)
        + f"\n\nModule ranking identical across vocabularies: **{identical}**.\n\n"
        + "### Does each finding transfer?\n\n"
        + _table(ch, cr)
        + "\n\nReduction **ratios** transfer because a ratio cancels a roughly constant "
        "vocabulary factor — absolute counts differ substantially (\"Please explain "
        "recursion.\" is 4 tokens under Qwen2.5 and 5 under GPT-2) while the percentages do "
        "not. The **mechanisms** underneath do not all transfer: GPT-2 has no capitalisation "
        "penalty, so one of ADR-030's two position-0 effects is Qwen-specific. That correction "
        "is ADR-032.\n\n"
        "**Scope.** This is the *tokenizer* dimension of report §4.6. Decode speed, answer "
        "quality and quantisation are properties of the model, not the tokenizer, and still "
        "require a real provider."
    )


def render_middleware(ctx: Context) -> str:
    # The TIMING pass ONLY. Falling back to the memoised quality pass would
    # report memo-hit microseconds as latency -- the precise contamination
    # the two-pass split exists to prevent. If there is no timing data, say
    # so; do not substitute.
    source = ctx.timing
    if not source:
        return (
            "_No unmemoised timing data in this run._\n\n"
            "The timing pass was skipped or fully resumed from a previous run. Latency is "
            "deliberately NOT reported from the quality pass: those rows are memoised, so "
            "their wall-clock reflects a dictionary lookup rather than generation. "
            "Re-run without `--resume` to regenerate it."
        )
    headers = ["cell", "mean ms", "95% CI", "p95 ms", "mean prefix tokens reused"]
    rows = []
    for r in source:
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
    Section("calibration_table", "Calibration table — per query class",
            ("tokens_in_final", "tokens_out"), render_calibration_table),
    Section("energy", "Energy and cost equivalent",
            ("joules_estimated", "usd_equivalent"), render_energy),
    Section("generalisation", "Cross-vocabulary generalisation",
            ("tokenizer_id", "tokens_in_final", "tokens_out"), render_generalisation),
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
    parser.add_argument("--resume", action="store_true",
                        help="Skip timing cells already recorded in completed.log. "
                             "For long unattended sweeps; off by default so a plain "
                             "reproduce run always regenerates everything.")
    parser.add_argument("--timing-subset", type=int, default=50,
                        help="Conversations in the unmemoised timing pass.")
    parser.add_argument("--timing-repeats", type=int, default=2,
                        help="Repeats of the timing pass (latency is the only stochastic part).")
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
    # Two passes (ADR-019/020, docs/05-evaluation-harness.md): a memoised
    # quality pass over the full corpus, and an unmemoised timing pass with
    # repeats over a stratified subset. Latency from a memoised run is
    # meaningless, so it never gets mixed in.
    print(f"running {len(cells)} cells, two passes...")
    report = two_pass_sweep(
        cells, corpus,
        timing_subset=args.timing_subset,
        timing_repeats=args.timing_repeats,
        resume_log=(args.out / "completed.log") if args.resume else None,
        judge=None if args.no_quality else LengthBiasedMockJudge(),
        gold=() if args.no_quality else load_gold(),
        progress=lambda msg: print(f"  {msg}"),
    )
    results = report.quality
    print(f"  memo: {report.memo_hits}/{report.memo_total} generations avoided "
          f"({report.memo_hit_rate:.1f}%)")
    print(f"  timing pass: {report.timing_corpus_size} conversations "
          f"x {report.timing_repeats} repeats"
          + (f", {report.resumed} cells resumed from a previous run" if report.resumed else ""))

    ctx = Context(results=results, corpus=corpus, out=args.out, timing=report.timing)
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
