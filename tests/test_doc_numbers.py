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


MATERIAL = 0.05  # pp; below this an interaction is noise at this sample size


class TestInteractionClaim:
    """This claim has now been wrong twice, in opposite directions.

    First the docs said *every* interaction is negative when most are zero.
    Then, after the encoder change (ADR-035), M1xM2 came back at +0.02 — so
    even "every non-zero interaction is negative" became false. The defensible
    claim is about MATERIAL interactions, and the threshold is stated rather
    than left to whatever the float signs happen to be.
    """

    def test_no_document_overstates_the_claim(self):
        for name, text in _docs():
            for bad in ("Every interaction term is negative",
                        "every interaction term is negative"):
                assert bad not in text, f"{name} overstates: most interactions are zero"

    def test_material_interactions_are_all_negative(self):
        """The claim the docs actually make must hold, at the stated threshold."""
        material = [
            (k, v) for k, v in _effects().items() if "x" in k and abs(v) >= MATERIAL
        ]
        assert material, "expected at least one material interaction"
        assert all(v < 0 for _, v in material), dict(material)

    def test_documents_quote_material_interactions_correctly(self):
        """Derived from the CSV rather than hardcoded, so a legitimate change
        to the pipeline updates the expectation instead of failing the test."""
        eff = _effects()
        for name, text in _docs():
            for term, quoted in re.findall(r"(M\d×M\d) (−[\d.]+)", text):
                key = term.replace("×", "x")
                actual = eff[key]
                assert float(quoted.replace("−", "-")) == pytest.approx(actual, abs=0.01), (
                    f"{name} quotes {term} as {quoted}; effects.csv says {actual:+.2f}"
                )


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

    @staticmethod
    def _from_report():
        report = (FIGURES / "report.md").read_text(encoding="utf-8")
        m = re.search(
            r"Additivity shortfall: ([\d.]+) percentage points "
            r"\(95% CI \[([+-][\d.]+), ([+-][\d.]+)\]\)",
            report,
        )
        assert m, "figures/report.md no longer states the shortfall in the expected form"
        return tuple(float(g) for g in m.groups())

    def test_the_live_shortfall_is_quoted_somewhere(self):
        """The documents deliberately quote TWO shortfalls — one per encoder
        (ADR-035) — so this checks the CURRENT configuration's value appears,
        not that every quoted triple matches."""
        point, lo, hi = self._from_report()
        needle = (point, lo, hi)
        found = [
            (float(a), float(b), float(c))
            for _, text in _docs()
            for a, b, c in re.findall(
                r"([\d.]+) (?:pp|percentage points), 95% CI \[([+-−][\d.]+), ([+-−][\d.]+)\]",
                text.replace("−", "-"),
            )
        ]
        assert any(
            all(x == pytest.approx(y, abs=0.01) for x, y in zip(triple, needle))
            for triple in found
        ), f"no document quotes the live shortfall {needle}; found {found}"

    def test_documents_do_not_claim_the_interval_excludes_zero_when_it_does_not(self):
        """This replaces a test that asserted the interval DOES exclude zero.

        That was the wrong test to write: it pinned a scientific outcome, so it
        would fail whenever the result legitimately changed — which is exactly
        the pressure that makes someone keep a worse component because it gives
        a better number. ADR-035 improved the encoder, the shortfall fell to
        1.63 pp and the interval reached zero, and that test failed for being
        right. What must actually hold is that the prose does not claim more
        than the interval supports.
        """
        _, lo, _ = self._from_report()
        if lo > 0:
            return
        for name, text in _docs():
            for claim in ("which excludes zero", "excludes zero, so the shortfall is a real effect"):
                assert claim not in text, (
                    f"{name} says the interval excludes zero, but it starts at {lo}"
                )


class TestCountsQuotedInTheReadme:
    def test_adr_count_matches_the_decision_log(self):
        """Checked in BOTH documents. The findings header quoted 29 ADRs while
        the README quoted 32 and the log held 34 — the first version of this
        test only looked at the README, so the findings header drifted
        unnoticed."""
        log = (ROOT / "docs" / "03-decision-log.md").read_text(encoding="utf-8")
        actual = len(set(re.findall(r"^#+ ADR-(\d+)", log, re.M)))
        for name, text in _docs():
            for quoted in re.findall(r"(\d+) ADRs", text):
                assert int(quoted) == actual, f"{name} says {quoted} ADRs; there are {actual}"

    def test_quoted_test_count_is_not_stale(self):
        """Both documents lead with a test count. It is the first number a
        reader sees and the easiest one to leave behind."""
        counts = {
            int(m)
            for _, text in _docs()
            for m in re.findall(r"\*\*(\d+) tests passing", text)
        }
        assert len(counts) <= 1, f"documents disagree on the test count: {sorted(counts)}"

    def test_every_doc_linked_from_the_readme_exists(self):
        text = README.read_text(encoding="utf-8")
        for rel in re.findall(r"\]\((docs/[\w.-]+\.md)\)", text):
            assert (ROOT / rel).exists(), f"README links {rel}, which does not exist"

    def test_every_doc_that_exists_is_linked_from_the_readme(self):
        text = README.read_text(encoding="utf-8")
        for doc in sorted((ROOT / "docs").glob("*.md")):
            assert f"docs/{doc.name}" in text, f"docs/{doc.name} is not linked from README"
