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
        # particles and contractions
        "not", "no", "never", "none", "nothing", "neither", "nor", "without",
        "cannot", "can't", "won't", "don't", "doesn't", "didn't", "isn't",
        "aren't", "wasn't", "weren't", "shouldn't", "wouldn't", "couldn't",
        "hasn't", "haven't", "hadn't", "unable", "unlike", "except",
        # standalone negating prefix, as in "non commercial" written open
        "non",
        # LEXICAL negation: verbs that negate without any particle. Measured
        # gap -- "Does aspirin thin the blood?" against "Does aspirin FAIL TO
        # thin the blood?" produced a false hit because no particle appears.
        "fail", "fails", "failed", "failing", "lack", "lacks", "lacking",
        "refuse", "refuses", "prevent", "prevents", "avoid", "avoids",
        "omit", "omits", "exclude", "excludes", "excluded", "excluding",
        "forbid", "forbids", "prohibit", "prohibits", "deny", "denies",
    }
)
_WORD_RE = re.compile(r"[A-Za-z']+")

# --------------------------------------------------------------------------
# Operative modifiers.
#
# Measured gap: with only number, entity and negation checks the verifier let
# 78% of modifier-swapped adversarial pairs through. "What is the MINIMUM
# temperature" and "what is the MAXIMUM temperature" share every other token, so
# lexical overlap stays high, no number changes, no entity changes and no
# negation particle appears -- yet the answers are opposite.
#
# These are the words whose substitution inverts a question while leaving its
# surface almost untouched. The caching literature's verifiers do not check for
# them.
# --------------------------------------------------------------------------
# Opposition PAIRS, not a flat set.
#
# A flat set cannot tell opposition from synonymy: "brief" and "short" are both
# modifiers, so comparing raw term sets rejected that legitimate paraphrase as a
# false hit. Grouping into (side A, side B) collapses synonyms onto the same
# side while keeping genuine inversions apart.
_OPPOSITION_PAIRS: tuple[tuple[frozenset[str], frozenset[str]], ...] = (
    (frozenset({"minimum", "min", "least", "smallest", "lowest", "fewest"}),
     frozenset({"maximum", "max", "most", "largest", "highest", "greatest"})),
    (frozenset({"first", "earliest", "initial", "oldest"}),
     frozenset({"last", "latest", "final", "newest"})),
    (frozenset({"best", "better", "cheapest", "cheaper"}),
     frozenset({"worst", "worse", "expensive", "costliest"})),
    (frozenset({"fastest", "faster", "quickest"}),
     frozenset({"slowest", "slower"})),
    (frozenset({"increase", "increases", "raise", "raises", "grow", "add"}),
     frozenset({"decrease", "decreases", "reduce", "reduces", "shrink", "remove"})),
    (frozenset({"before", "prior", "preceding", "earlier"}),
     frozenset({"after", "subsequent", "following", "later"})),
    (frozenset({"ascending", "asc"}), frozenset({"descending", "desc"})),
    (frozenset({"import", "imports", "input", "inputs", "upload", "download"}),
     frozenset({"export", "exports", "output", "outputs"})),
    (frozenset({"average", "mean"}), frozenset({"median", "mode"})),
    (frozenset({"more", "greater", "higher", "above"}),
     frozenset({"less", "fewer", "lower", "below"})),
    (frozenset({"always", "every", "all"}),
     frozenset({"never", "rarely", "sometimes", "occasionally"})),
    (frozenset({"include", "includes", "enable", "enabled", "allow", "allowed"}),
     frozenset({"exclude", "excludes", "disable", "disabled", "forbid", "blocked"})),
    (frozenset({"open", "opens", "unlock", "decrypt"}),
     frozenset({"close", "closes", "lock", "encrypt"})),
    (frozenset({"start", "starts", "begin", "begins"}),
     frozenset({"stop", "stops", "end", "ends"})),
    (frozenset({"commercial", "commercially", "public", "external"}),
     frozenset({"personal", "private", "internal", "noncommercial"})),
    # Synonyms sit on the SAME side, so "brief" against "short" no longer
    # registers as an opposition.
    (frozenset({"brief", "short", "concise", "summary"}),
     frozenset({"detailed", "long", "comprehensive", "thorough"})),
)

_TERM_TO_POLARITY: dict[str, tuple[int, int]] = {
    term: (group_index, side)
    for group_index, sides in enumerate(_OPPOSITION_PAIRS)
    for side, terms in enumerate(sides)
    for term in terms
}

# Morphological negation: "possible" -> "impossible", "refundable" ->
# "non refundable". No separate negation token appears, so a particle-based
# check misses it entirely.
_NEG_PREFIXES = ("non", "un", "im", "in", "ir", "il", "dis", "anti")

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
# Alphanumeric identifiers: Panda3D, Python3, GPT4, Qwen2, room B4.
# _PROPER_RE cannot match these at all -- its trailing \b fails against the
# digit -- so "pandas" against "Panda3D" produced a false cache hit with both
# queries reporting zero entities.
_ALNUM_ID_RE = re.compile(r"\b[A-Za-z]{2,}\d[A-Za-z0-9]*\b")
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
        ents: set[str] = set(_ACRONYM_RE.findall(text)) | set(_ALNUM_ID_RE.findall(text))
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


def operative_modifiers(text: str) -> frozenset[tuple[int, int]]:
    """(opposition group, side) pairs present in the text.

    Returning the POLARITY rather than the surface word is what lets synonyms
    agree: "give me a brief summary" and "give me a short summary" both yield
    the same (group, side), so comparing two texts by equality no longer
    rejects them, while "minimum" against "maximum" still differs.
    """
    return frozenset(
        _TERM_TO_POLARITY[w]
        for w in _TOKEN_RE.findall(text.lower())
        if w in _TERM_TO_POLARITY
    )


def describe_modifiers(text: str) -> frozenset[str]:
    """Human-readable form, for traces and error messages."""
    return frozenset(w for w in _TOKEN_RE.findall(text.lower()) if w in _TERM_TO_POLARITY)


def morphological_negations(text: str, other: str) -> frozenset[str]:
    """Prefix-negated words in `text` whose stem appears in `other`.

    Requiring the stem to be present in the counterpart query is what keeps this
    precise: "international" is not a negation of "national" in isolation, but
    "impossible" opposite "possible" in an otherwise identical question is.
    """
    other_words = set(_TOKEN_RE.findall(other.lower()))
    found = set()
    for word in _TOKEN_RE.findall(text.lower()):
        for prefix in _NEG_PREFIXES:
            if len(word) > len(prefix) + 3 and word.startswith(prefix):
                stem = word[len(prefix) :]
                if stem in other_words:
                    found.add(word)
                    break
    return frozenset(found)


def jaccard(a: str, b: str) -> float:
    sa, sb = shingles(a), shingles(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)
