"""Lightweight NLP: invariant extraction, sentence splitting, PII detection.

Sprint 0 uses regex + heuristics rather than spaCy so the project has zero heavy
dependencies before the review. Each component sits behind a protocol, so the
spaCy implementations planned for Sprint 2 drop in without touching callers
(docs/01-pipeline-stages.md Stage 2).

The gate's job is to be *conservative*: over-extracting an invariant makes the
gate stricter (a false revert, which costs a little compression), while
under-extracting lets a real meaning change through. When in doubt, extract.
"""

from __future__ import annotations

import re

from parsimony.core.types import Invariants

# --------------------------------------------------------------------------
# Numbers
# --------------------------------------------------------------------------

_UNITS = (
    r"km/h|mph|kWh|kW|MW|GB|MB|KB|TB|hrs|hr|hours|hour|mins|min|secs|sec|"
    r"kg|km|cm|mm|nm|ml|mg|lbs|lb|oz|ft|yd|mi|ms|ns|"
    r"USD|EUR|INR|°C|°F|%|m|g|s|h|K|C|F"
)
_NUM_RE = re.compile(
    rf"(?<![\w.])[$€£₹]?\d+(?:,\d{{3}})*(?:\.\d+)?(?:\s?(?:{_UNITS})(?![A-Za-z]))?"
)

# --------------------------------------------------------------------------
# Negations — a closed lexicon. Missing one lets a meaning flip through, which
# is exactly the failure the adversarial corpus subset is built to catch.
# --------------------------------------------------------------------------

_NEGATIONS = frozenset(
    {
        "not", "no", "never", "none", "nothing", "neither", "nor", "without",
        "cannot", "can't", "won't", "don't", "doesn't", "didn't", "isn't",
        "aren't", "wasn't", "weren't", "shouldn't", "wouldn't", "couldn't",
        "hasn't", "haven't", "hadn't", "unable", "unlike", "exclude",
        "excluding", "except",
    }
)
_WORD_RE = re.compile(r"[A-Za-z']+")

# --------------------------------------------------------------------------
# Quoted spans
# --------------------------------------------------------------------------

_QUOTED_RE = re.compile(
    r"```(?P<fence>.*?)```"
    r"|`(?P<tick>[^`\n]{1,200})`"
    r"|\"(?P<dq>[^\"\n]{1,200})\""
    r"|(?<![A-Za-z])'(?P<sq>[^'\n]{2,200})'(?![A-Za-z])",
    re.DOTALL,
)

# --------------------------------------------------------------------------
# Entities (Sprint 0 heuristic; spaCy NER in Sprint 2)
# --------------------------------------------------------------------------

_ACRONYM_RE = re.compile(r"\b[A-Z]{2,6}\b")
_PROPER_RE = re.compile(r"\b[A-Z][a-z]{1,}(?:\s+[A-Z][a-z]{1,}){0,3}\b")
_SENT_START_RE = re.compile(r"(?:^|[.!?]\s+|\n\s*)")

# Words that begin sentences constantly and are never entities.
_STOP_PROPER = frozenset(
    {
        "The", "This", "That", "These", "Those", "What", "When", "Where", "Which",
        "Who", "Why", "How", "Is", "Are", "Was", "Were", "Do", "Does", "Did",
        "Can", "Could", "Should", "Would", "Will", "If", "In", "On", "At", "For",
        "And", "But", "Or", "So", "It", "I", "We", "You", "They", "He", "She",
        "Please", "Explain", "Write", "Give", "Tell", "Show", "List", "Summarise",
        "Summarize", "Convert", "Calculate", "Compare", "Describe", "Also",
        "Now", "Then", "Here", "There", "Yes", "No", "Let", "Make", "Find",
    }
)


def _sentence_start_offsets(text: str) -> set[int]:
    starts = {0}
    for m in _SENT_START_RE.finditer(text):
        starts.add(m.end())
    return starts


class RegexInvariantExtractor:
    """Sprint 0 invariant extractor. Swappable for a spaCy-backed one."""

    def extract(self, text: str) -> Invariants:
        if not text:
            return Invariants()

        numbers = frozenset(m.group(0).strip() for m in _NUM_RE.finditer(text))

        negations = frozenset(
            w.group(0).lower()
            for w in _WORD_RE.finditer(text)
            if w.group(0).lower() in _NEGATIONS
        )

        quoted_vals: set[str] = set()
        for m in _QUOTED_RE.finditer(text):
            for group in ("fence", "tick", "dq", "sq"):
                v = m.group(group)
                if v and v.strip():
                    quoted_vals.add(v.strip())
        quoted = frozenset(quoted_vals)

        starts = _sentence_start_offsets(text)
        ents: set[str] = set(_ACRONYM_RE.findall(text))
        for m in _PROPER_RE.finditer(text):
            val = m.group(0)
            first = val.split()[0]
            # Skip a capitalised word that merely starts a sentence, unless the
            # phrase is multi-word (e.g. "New Delhi" at a sentence start is real).
            if m.start() in starts and " " not in val:
                continue
            if first in _STOP_PROPER and " " not in val:
                continue
            ents.add(val)

        return Invariants(
            numbers=numbers,
            entities=frozenset(ents),
            negations=negations,
            quoted=quoted,
        )


# --------------------------------------------------------------------------
# Sentence splitting
# --------------------------------------------------------------------------

_ABBREV = frozenset({"e.g", "i.e", "etc", "vs", "mr", "mrs", "dr", "prof", "fig", "no", "approx"})
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> tuple[str, ...]:
    """Regex sentence splitter with light abbreviation handling.

    Fenced code blocks are kept intact — splitting inside one produces
    meaningless 'sentences' and would let M1 tier 2 delete half a function.
    """
    if not text.strip():
        return ()

    blocks = re.split(r"(```.*?```)", text, flags=re.DOTALL)
    out: list[str] = []
    for block in blocks:
        if not block.strip():
            continue
        if block.startswith("```"):
            out.append(block.strip())
            continue
        for line in block.split("\n"):
            if not line.strip():
                continue
            parts = _SENT_SPLIT_RE.split(line.strip())
            merged: list[str] = []
            for p in parts:
                tail = p.rstrip(".").split()[-1].lower() if p.rstrip(".").split() else ""
                if merged and tail in _ABBREV:
                    merged[-1] = merged[-1] + " " + p
                elif merged and len(p) < 3:
                    merged[-1] = merged[-1] + " " + p
                else:
                    merged.append(p)
            out.extend(s.strip() for s in merged if s.strip())
    return tuple(out)


# --------------------------------------------------------------------------
# PII
# --------------------------------------------------------------------------

_PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+\d{1,3}[\s-]?)?(?:\d{10}|\d{3}[\s-]\d{3}[\s-]\d{4})(?!\d)")),
    ("CARD", re.compile(r"(?<!\d)(?:\d{4}[\s-]?){3}\d{4}(?!\d)")),
    ("IP", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("AADHAAR", re.compile(r"(?<!\d)\d{4}\s\d{4}\s\d{4}(?!\d)")),
)


class RegexPiiDetector:
    def spans(self, text: str) -> list[tuple[int, int, str]]:
        found: list[tuple[int, int, str]] = []
        for label, pat in _PII_PATTERNS:
            for m in pat.finditer(text):
                found.append((m.start(), m.end(), label))
        return sorted(found)

    def redact(self, text: str) -> str:
        """Applied at write boundaries only (ADR-013): the model sees real text,
        the cache and ledger never do."""
        spans = self.spans(text)
        if not spans:
            return text
        out, last = [], 0
        for start, end, label in spans:
            if start < last:
                continue
            out.append(text[last:start])
            out.append(f"[{label}]")
            last = end
        out.append(text[last:])
        return "".join(out)


# --------------------------------------------------------------------------
# Lexical similarity — stands in for embeddings until Sprint 2 (ADR-007).
# --------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def shingles(text: str) -> frozenset[str]:
    return frozenset(_TOKEN_RE.findall(text.lower()))


def jaccard(a: str, b: str) -> float:
    sa, sb = shingles(a), shingles(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)
