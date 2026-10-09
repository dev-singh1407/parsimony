"""The counts a generated document states about the project itself.

Every document built from this directory quotes two numbers about the work
rather than about the measurements: how many tests the suite runs, and how
many decision records the log holds. Both were typed into the generators as
literals, and both went stale without anything noticing: when this was
written, the Review-2 dossier generator still stated a test total several
hundred tests out of date and a record total nine short, and
`report_content.py` was staler still.

A literal inside a document generator is invisible: nothing reads it until
someone rebuilds the document, and by then the wrong number is in a PDF in
front of a panel. So both are derived here, at build time, from the files
that already hold the truth -- the same rule the rest of the report follows,
that a generated document reads its numbers instead of restating them.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DECISION_LOG = ROOT / "docs" / "03-decision-log.md"
README = ROOT / "README.md"


def _read(path: Path) -> str:
    return io.open(path, encoding="utf-8", newline="").read()


def adr_count() -> int:
    """How many decision records `docs/03-decision-log.md` holds.

    Counted with the same expression `tests/test_doc_numbers.py` uses, over
    the same file, so a generator and the test that guards the prose cannot
    disagree about what counts as a record. Deduplicated because an ADR may
    be picked up again under a later heading.
    """
    return len(set(re.findall(r"^#+ ADR-(\d+)", _read(DECISION_LOG), re.M)))


def test_count() -> int:
    """How many tests the suite runs, as README.md states it.

    Read from the README rather than collected with `pytest -q --co`, for two
    reasons. Collection imports the entire test tree, which is a strange and
    fragile thing to do in the middle of building a Word document -- it can
    fail for reasons that have nothing to do with the document. And it counts
    tests COLLECTED, where every document claims tests PASSING; the suite
    skips a test, so collecting would print a total no run reports as passed.

    The README line is the project's published figure, so reading it means a
    rebuilt document and the README cannot drift apart. It also makes that
    one line the thing that has to stay correct; `test_doc_numbers.py` is
    where that is enforced.
    """
    text = _read(README)
    m = re.search(r"\*\*([\d,]+) tests passing", text)
    if m is None:
        raise RuntimeError(
            f"{README.name} no longer opens with '**N tests passing'. The "
            "document generators read that line; fix the README or this "
            "function rather than typing a count into a document."
        )
    return int(m.group(1).replace(",", ""))
