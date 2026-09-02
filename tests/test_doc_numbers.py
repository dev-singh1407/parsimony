"""The prose must agree with the figures.

Documentation drift has been the most persistent defect class in this project:
a number gets fixed in the pipeline, the CSVs regenerate, and a sentence in
README.md keeps quoting the old value. A reader who cannot trust the headline
numbers has no reason to trust anything else, so this is checked mechanically
rather than by re-reading.

These tests read the committed `figures/` CSVs, which `reproduce.py` writes.
If they fail, the fix is usually to correct the prose — but check first
whether the pipeline changed and the CSVs are the thing that moved.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
README = ROOT / "README.md"
FINDINGS = ROOT / "docs" / "09-findings.md"

pytestmark = pytest.mark.skipif(
    not (FIGURES / "ablation.csv").exists(),
    reason="figures/ not generated; run `python reproduce.py --out figures`",
)


def _rows(name: str) -> list[dict[str, str]]:
    with (FIGURES / name).open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _effects() -> dict[str, float]:
    return {r["effect"]: float(r["estimate (pp)"]) for r in _rows("effects.csv")}


def _docs() -> list[tuple[str, str]]:
    return [(p.name, p.read_text(encoding="utf-8")) for p in (README, FINDINGS)]


class TestMainEffectsQuotedCorrectly:
    """Each module's effect is quoted in both documents as `+N.NN pp`."""

    @pytest.mark.parametrize("module", ["M1", "M2", "M3", "M5"])
    def test_quoted_effect_matches_csv(self, module):
        truth = _effects()[module]
        pattern = re.compile(
            rf"\*\*{module}\*\*[^|\n]*\|\s*([+-][\d.]+) pp|"
            rf"\|\s*{module} [^|\n]*\|\s*([+-][\d.]+) pp"
        )
        found_anywhere = False
        for name, text in _docs():
            for m in pattern.finditer(text):
                quoted = float(m.group(1) or m.group(2))
                found_anywhere = True
                assert quoted == pytest.approx(truth, abs=0.01), (
                    f"{name} quotes {module} as {quoted:+.2f} pp; "
                    f"figures/effects.csv says {truth:+.2f} pp"
                )
        assert found_anywhere, f"{module}'s effect is quoted in neither document"


class TestInteractionClaim:
    """This claim was wrong for several commits: the docs said *every*
    interaction is negative, when 8 of 11 are zero and two of those carry a
    positive float sign."""

    def test_no_document_claims_every_interaction_is_negative(self):
        for name, text in _docs():
            assert "Every interaction term is negative" not in text, (
                f"{name} overstates: only the non-zero interactions are negative"
            )

    def test_non_zero_interactions_really_are_all_negative(self):
        """The weaker claim the docs now make must itself hold."""
        non_zero = [
            (k, v) for k, v in _effects().items() if "x" in k and abs(v) >= 0.005
        ]
        assert non_zero, "expected at least one material interaction"
        assert all(v < 0 for _, v in non_zero), dict(non_zero)

    def test_the_three_named_interactions_match_the_csv(self):
        eff = _effects()
        for name, expected in (("M3xM5", -1.14), ("M2xM5", -0.10), ("M1xM5", -0.02)):
            assert eff[name] == pytest.approx(expected, abs=0.01)


class TestHeadlineReduction:
    def test_full_stack_percentage_matches(self):
        full = max(_rows("ablation.csv"), key=lambda r: float(r["reduction %"]))
        truth = float(full["reduction %"])
        for name, text in _docs():
            for quoted in re.findall(r"Full stack reaches \*\*\+([\d.]+)%\*\*", text):
                assert float(quoted) == pytest.approx(truth, abs=0.1), (
                    f"{name} says full stack is +{quoted}%; "
                    f"figures/ablation.csv says {truth:+.1f}%"
                )


class TestAdditivityShortfall:
    """The headline contribution. Quoted with its interval in both documents."""

    def test_shortfall_and_interval_match_the_report(self):
        report = (FIGURES / "report.md").read_text(encoding="utf-8")
        m = re.search(
            r"Additivity shortfall: ([\d.]+) percentage points "
            r"\(95% CI \[([+-][\d.]+), ([+-][\d.]+)\]\)",
            report,
        )
        assert m, "figures/report.md no longer states the shortfall in the expected form"
        point, lo, hi = (float(g) for g in m.groups())

        quoted_anywhere = False
        for name, text in _docs():
            for q in re.finditer(
                r"([\d.]+) (?:pp|percentage points), 95% CI \[([+-][\d.]+), ([+-][\d.]+)\]",
                text,
            ):
                quoted_anywhere = True
                assert (float(q.group(1)), float(q.group(2)), float(q.group(3))) == (
                    pytest.approx(point, abs=0.01),
                    pytest.approx(lo, abs=0.01),
                    pytest.approx(hi, abs=0.01),
                ), f"{name} quotes a stale shortfall: {q.group(0)}"
        assert quoted_anywhere, "the shortfall is quoted in neither document"

    def test_the_interval_still_excludes_zero(self):
        """If this fails the contribution claim itself is dead, not just the prose."""
        report = (FIGURES / "report.md").read_text(encoding="utf-8")
        lo = float(re.search(r"95% CI \[([+-][\d.]+),", report).group(1))
        assert lo > 0


class TestCountsQuotedInTheReadme:
    def test_adr_count_matches_the_decision_log(self):
        log = (ROOT / "docs" / "03-decision-log.md").read_text(encoding="utf-8")
        actual = len(set(re.findall(r"^#+ ADR-(\d+)", log, re.M)))
        quoted = int(re.search(r"(\d+) ADRs", README.read_text(encoding="utf-8")).group(1))
        assert quoted == actual

    def test_every_doc_linked_from_the_readme_exists(self):
        text = README.read_text(encoding="utf-8")
        for rel in re.findall(r"\]\((docs/[\w.-]+\.md)\)", text):
            assert (ROOT / rel).exists(), f"README links {rel}, which does not exist"

    def test_every_doc_that_exists_is_linked_from_the_readme(self):
        text = README.read_text(encoding="utf-8")
        for doc in sorted((ROOT / "docs").glob("*.md")):
            assert f"docs/{doc.name}" in text, f"docs/{doc.name} is not linked from README"
