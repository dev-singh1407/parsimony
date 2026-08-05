"""M1 tier 3 — tokenizer-aware rewriting with negative-yield detection.

THE POINT
---------
Byte-pair merges are context dependent. A word does not cost a fixed number of
tokens: it costs whatever the merge table decides given its neighbours. So
deleting a word, or substituting a shorter phrase, can *raise* the total token
count by breaking a merge that spanned the boundary.

Every published prompt-compression method assumes deletion monotonically
reduces tokens. It does not. Tier 3 therefore re-tokenises every candidate edit
and reverts any that does not actually pay. The rejection rate is itself a
reportable number.

THE COST, AND WHY THE WINDOW IS DANGEROUS
-----------------------------------------
Naively, checking N candidate edits on a prompt of M tokens costs O(N*M)
tokenisation work -- 40 edits on a 2000-token prompt is 80k token-ops per
request, far beyond the 120ms overhead budget.

BPE merges are local, so re-tokenising a window around the edit is almost always
equivalent. "Almost always" is not good enough for a correctness claim, which is
why `tests/golden/test_windowed_retokenisation.py` re-tokenises the full text
for every candidate edit across the entire corpus and asserts the windowed
DECISION matches. If that ever fails the window widens. Do not ship tier 3
without that test green.

Edits are applied right-to-left so that earlier offsets stay valid as later ones
are rewritten.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# (pattern, replacement, rule name). Ordered longest-first so that a specific
# rule wins over a general one at the same position.
DEFAULT_LEXICON: tuple[tuple[str, str, str], ...] = (
    (r"\bat this point in time\b", "now", "verbosity"),
    (r"\bin the event that\b", "if", "verbosity"),
    (r"\bdue to the fact that\b", "because", "verbosity"),
    (r"\bfor the purpose of\b", "for", "verbosity"),
    (r"\bin the near future\b", "soon", "verbosity"),
    (r"\bit is important to note that\b", "", "filler"),
    (r"\bit should be noted that\b", "", "filler"),
    (r"\bplease be aware that\b", "", "filler"),
    (r"\bas a matter of fact\b", "", "filler"),
    (r"\ba large number of\b", "many", "verbosity"),
    (r"\ba small number of\b", "few", "verbosity"),
    (r"\bin order to\b", "to", "verbosity"),
    (r"\bwith regard to\b", "about", "verbosity"),
    (r"\bin relation to\b", "about", "verbosity"),
    (r"\bis able to\b", "can", "verbosity"),
    (r"\bare able to\b", "can", "verbosity"),
    (r"\bhas the ability to\b", "can", "verbosity"),
    (r"\bin spite of the fact that\b", "although", "verbosity"),
    (r"\bon a regular basis\b", "regularly", "verbosity"),
    (r"\bthe majority of\b", "most", "verbosity"),
    (r"\ba sufficient amount of\b", "enough", "verbosity"),
    (r"\bprior to\b", "before", "verbosity"),
    (r"\bsubsequent to\b", "after", "verbosity"),
    (r"\bin the process of\b", "", "filler"),
    (r"\bwould like to\b", "want to", "verbosity"),
)


@dataclass(frozen=True, slots=True)
class EditCandidate:
    start: int
    end: int
    replacement: str
    rule: str
    matched: str

    def apply_to(self, text: str) -> str:
        return text[: self.start] + self.replacement + text[self.end :]


@dataclass(frozen=True, slots=True)
class EditOutcome:
    candidate: EditCandidate
    windowed_delta: int
    applied: bool


_FENCE_SPAN_RE = re.compile(r"```.*?```|`[^`\n]+`", re.DOTALL)


def _protected_spans(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in _FENCE_SPAN_RE.finditer(text)]


def _overlaps(span: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    return any(not (span[1] <= s or span[0] >= e) for s, e in spans)


def find_candidates(
    text: str, lexicon: tuple[tuple[str, str, str], ...] = DEFAULT_LEXICON
) -> list[EditCandidate]:
    """Non-overlapping candidates, longest match first, code spans excluded."""
    protected = _protected_spans(text)
    found: list[EditCandidate] = []
    taken: list[tuple[int, int]] = []

    for pattern, replacement, rule in lexicon:
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            span = m.span()
            if _overlaps(span, protected) or _overlaps(span, taken):
                continue
            taken.append(span)
            found.append(
                EditCandidate(
                    start=span[0],
                    end=span[1],
                    replacement=replacement,
                    rule=rule,
                    matched=m.group(0),
                )
            )
    return sorted(found, key=lambda c: c.start)


def windowed_delta(text: str, edit: EditCandidate, tokenizer, window: int) -> int:
    """Token delta computed on a window around the edit.

    Negative means the edit pays. The window must be wide enough that any merge
    the edit could disturb is fully contained; the golden test is what
    establishes that.
    """
    w_start = max(0, edit.start - window)
    w_end = min(len(text), edit.end + window)
    before = text[w_start:w_end]
    after = text[w_start : edit.start] + edit.replacement + text[edit.end : w_end]
    return tokenizer.count(after) - tokenizer.count(before)


def full_delta(text: str, edit: EditCandidate, tokenizer) -> int:
    """Ground truth: re-tokenise the entire text. Used by the golden test."""
    return tokenizer.count(edit.apply_to(text)) - tokenizer.count(text)


def _tidy(text: str) -> str:
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"^\s*[,;:]\s*", "", text)
    return text.strip()


def rewrite(
    text: str,
    tokenizer,
    window: int = 32,
    lexicon: tuple[tuple[str, str, str], ...] = DEFAULT_LEXICON,
) -> tuple[str, list[EditOutcome]]:
    """Apply every candidate edit that actually reduces the token count.

    Right-to-left, so that applying one edit does not invalidate the offsets of
    the ones still to be considered.
    """
    candidates = find_candidates(text, lexicon)
    outcomes: list[EditOutcome] = []
    result = text

    for edit in sorted(candidates, key=lambda c: c.start, reverse=True):
        delta = windowed_delta(result, edit, tokenizer, window)
        pays = delta < 0
        outcomes.append(EditOutcome(edit, delta, pays))
        if pays:
            result = edit.apply_to(result)

    outcomes.reverse()
    return (_tidy(result) if any(o.applied for o in outcomes) else text), outcomes
