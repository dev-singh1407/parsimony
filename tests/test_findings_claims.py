"""The findings document's claims about the system, against the figures.

Section 12 is the list of what the project has *not* measured, and it had gone
stale in the direction that matters: it still said everything ran on a mock
provider four sections below a table of real prefill timings, and still quoted
a true-hit rate for an encoder the system had stopped shipping. A reader who
took it at its word and then read §8 had to decide which half of one document
to believe.

`test_doc_numbers.py` checks the numbers a document quotes. These check the
CLAIMS -- that a section saying something is unmeasured is not contradicted by
a figure measuring it.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FINDINGS = ROOT / "docs" / "09-findings.md"
FIGURES = ROOT / "figures"


def section(number: int) -> str:
    """One numbered section of the findings, without the next one."""
    text = FINDINGS.read_text(encoding="utf-8")
    start = text.index(f"## {number}. ")
    rest = text[start:]
    nxt = re.search(rf"^## (?!{number}\.)\d+\. ", rest[3:], re.M)
    return rest[: nxt.start() + 3] if nxt else rest


def rows(name: str) -> list[dict]:
    path = FIGURES / name
    if not path.exists():
        pytest.skip(f"{name} not generated; run reproduce.py")
    return list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))


class TestTheUnmeasuredListIsStillTrue:
    """Every claim in §12 that a figure can contradict."""

    def test_latency_is_not_called_unmeasured_while_it_is_measured(self):
        """§8 reads prefill and decode off the runtime's own counters. The list
        said "everything runs on MockProvider" for months afterwards."""
        text = section(12)
        assert "Everything runs on `MockProvider`" not in text
        assert "Latency itself is **not** on" in text, (
            "the section has to say why latency is absent from it, or the next "
            "reader will put it back")

    def test_the_true_hit_rate_quoted_is_the_shipped_encoder(self):
        """It read 22.2% -- the lexical encoder's figure -- long after the
        neural one shipped."""
        text = section(12)
        shipped = [r for r in rows("encoders.csv")
                   if "neural" in r["encoder"] and "verify" in r["design"]]
        assert shipped, "encoders.csv no longer names a verified neural arm"
        true_pct = shipped[0]["true %"]
        assert f"{true_pct}%" in text, (
            f"§12 does not quote the shipped encoder's true-hit rate {true_pct}%")
        # A superseded figure may appear, but only as history. Presented flat it
        # is the same mistake again.
        for line in text.splitlines():
            if "22.2%" in line:
                assert any(w in line for w in
                           ("read", "was", "stopped shipping", "no longer")), (
                    "22.2% appears without being marked as the figure it used to be")

    def test_a_number_it_quotes_belongs_to_a_figure(self):
        """The false-answer rate it cites has to be the one measured."""
        text = section(12)
        shipped = [r for r in rows("encoders.csv")
                   if "neural" in r["encoder"] and "verify" in r["design"]]
        assert f'{shipped[0]["false %"]}%' in text

    @pytest.mark.parametrize("phrase", [
        "adaptive floor",        # ADR-051
        "near-off-topic",        # ADR-052
        "prefilter",             # ADR-052
    ])
    def test_the_list_carries_what_the_recent_entries_opened(self, phrase):
        """A decision that closes one question and opens another has to leave
        the open one somewhere a reader will find it."""
        assert phrase.lower() in section(12).lower(), (
            f"§12 does not mention {phrase!r}, which a recent ADR left open")


class TestTheDocumentDoesNotContradictItself:
    def test_no_section_calls_the_model_simulated_after_section_eight(self):
        """§8 is where the real model arrives. Nothing after it may describe
        the system as unmeasured against one without saying which part."""
        text = FINDINGS.read_text(encoding="utf-8")
        after = text[text.index("## 8. On a real model"):]
        for claim in ("Everything runs on `MockProvider`",
                      "no provider is attached",
                      "until a real model is attached"):
            assert claim not in after, f"{claim!r} appears after §8"


class TestTheRunbookAnswersFromTheSameList:
    """The viva answer to "what's left to do?" is the same list, spoken. It had
    drifted the other way -- still offering escalation as future work three ADRs
    after it was measured and came back negative."""

    RUNBOOK = ROOT / "docs" / "11-demo-runbook.md"

    def _answer(self) -> str:
        if not self.RUNBOOK.exists():
            pytest.skip("runbook not present")
        text = self.RUNBOOK.read_text(encoding="utf-8")
        start = text.index('''**"What's left to do?"**''')
        return text[start:start + 1600]

    def test_it_does_not_offer_a_measured_negative_as_future_work(self):
        answer = self._answer()
        assert "Escalation is *not* on that list" in answer, (
            "escalation was measured (ADR-037) and is a negative result, not work remaining")

    def test_it_says_how_many_and_the_findings_agree(self):
        """Saying "six" while the list holds seven is the mismatch this exists
        to catch; the answer may leave one out, but it has to say so."""
        bullets = len([l for l in section(12).splitlines() if l.startswith("- **")])
        answer = self._answer()
        spoken = {"five": 5, "six": 6, "seven": 7, "eight": 8}
        said = next((n for w, n in spoken.items() if w in answer.lower().split("worth")[0]), None)
        assert said is not None, "the answer no longer says how many things are open"
        assert said <= bullets, f"the answer says {said}, the findings list {bullets}"
        if said < bullets:
            assert "findings carry" in answer or "findings list" in answer, (
                f"the answer names {said} of {bullets} without saying the rest are elsewhere")
