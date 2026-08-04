"""M1 — Tokenizer-Aware Compressor.

Registered as three separate stages, all carrying module_id "M1" so the ablation
flag still switches the whole module. Separate stages mean the fidelity gate
sees three small proposals: a bad tier-3 rewrite is reverted without costing
tier 1's safe savings.

Tier 3 is implemented but disabled by default until Sprint 3 — negative-yield
detection must not ship before its golden test (docs/02-module-specs.md M1).
"""

from __future__ import annotations

import re

from parsimony.core.config import ParsimonyConfig
from parsimony.core.proposals import ContextPatch, NoOp, Proposal, TransformKind
from parsimony.core.types import RequestContext
from parsimony.infra.nlp import jaccard

_FENCE_RE = re.compile(r"(```.*?```|`[^`\n]+`)", re.DOTALL)

# --- Tier 1: lossless normalisation ---------------------------------------

_POLITENESS = [
    r"\bi was wondering if you (?:could|can|would)\b",
    r"\bcould you please\b",
    r"\bcan you please\b",
    r"\bwould you please\b",
    r"\bplease could you\b",
    r"\bi would like to know\b",
    r"\bi'd like to know\b",
    r"\bif you don'?t mind\b",
    r"\bthanks in advance\b",
    r"\bthank you very much\b",
    r"\bthanks a lot\b",
    r"\bmany thanks\b",
    r"\bhope you'?re (?:well|doing well)\b",
    r"\bplease\b",
    r"\bkindly\b",
    r"\bthanks\b",
    r"\bthank you\b",
]
_POLITENESS_RE = re.compile("|".join(_POLITENESS), re.IGNORECASE)
_GREETING_RE = re.compile(
    r"^\s*(?:hello|hi|hey|good (?:morning|afternoon|evening))\b[\s,!.-]*", re.IGNORECASE
)
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
_EMPHASIS_RE = re.compile(r"(\*\*|__|\*|_)(?=\S)(.+?)(?<=\S)\1", re.DOTALL)
_BLANKS_RE = re.compile(r"\n{3,}")
_SPACES_RE = re.compile(r"[ \t]{2,}")
_TRAILING_RE = re.compile(r"[ \t]+$", re.MULTILINE)
_PUNCT_SPACE_RE = re.compile(r"\s+([,.;:!?])")
_SENT_PARTS_RE = re.compile(r"([^.!?\n]+[.!?]*)")
_HAS_ALNUM_RE = re.compile(r"[A-Za-z0-9]")


def _protect_code(text: str):
    """Split into (segments, is_code) so no tier touches fenced or inline code."""
    return [(part, bool(i % 2)) for i, part in enumerate(_FENCE_RE.split(text))]


def _clean_sentence(sentence: str) -> str:
    """Strip boilerplate from one sentence, then repair what removal broke.

    Removing a phrase mid-sentence leaves punctuation debris ('Thanks in
    advance!' -> '!', 'Hello, explain...' -> ', explain...'). Working at
    sentence granularity means a sentence that becomes contentless is dropped
    whole rather than leaving a stray token that costs a token and reads as a
    bug in the demo.
    """
    s = _GREETING_RE.sub("", sentence)
    s = _POLITENESS_RE.sub("", s)
    s = _SPACES_RE.sub(" ", s)
    s = _PUNCT_SPACE_RE.sub(r"\1", s)
    s = re.sub(r"([,;:])\s*\1+", r"\1", s)
    s = re.sub(r"^\s*[,;:.\-]+\s*", "", s)
    s = s.strip()
    if not _HAS_ALNUM_RE.search(s):
        return ""
    # Re-capitalise: removal frequently strips the original sentence opener.
    for i, ch in enumerate(s):
        if ch.isalpha():
            return s[:i] + ch.upper() + s[i + 1 :]
    return s


def normalise_lossless(text: str) -> str:
    out: list[str] = []
    for segment, is_code in _protect_code(text):
        if is_code:
            out.append(segment)
            continue
        s = _HEADING_RE.sub("", segment)
        s = _EMPHASIS_RE.sub(r"\2", s)
        s = _TRAILING_RE.sub("", s)
        s = _BLANKS_RE.sub("\n\n", s)

        lines: list[str] = []
        for line in s.split("\n"):
            if not line.strip():
                lines.append("")
                continue
            cleaned = [_clean_sentence(p) for p in _SENT_PARTS_RE.findall(line)]
            lines.append(" ".join(c for c in cleaned if c))
        out.append("\n".join(lines))

    result = "".join(out)
    result = _BLANKS_RE.sub("\n\n", result)
    return result.strip()


class Tier1Normaliser:
    module_id = "M1"
    name = "m1_tier1"
    reads = frozenset({"query", "history"})
    writes = frozenset({"query", "history"})

    def applies_to(self, ctx: RequestContext, cfg: ParsimonyConfig) -> bool:
        return cfg.enables("M1") and cfg.compression.tier1_enabled

    def propose(self, ctx: RequestContext, cfg: ParsimonyConfig) -> Proposal:
        new_query = normalise_lossless(ctx.query)
        new_history = tuple(
            t if t.content == (n := normalise_lossless(t.content)) else _replace_content(t, n)
            for t in ctx.history
        )
        if new_query == ctx.query and new_history == ctx.history:
            return NoOp("no_yield", "already normalised")

        d = ctx.derived
        before = d.token_count(ctx.text_payload()) if d else 0
        return ContextPatch(
            kind=TransformKind.REWRITE,
            fields={"query": new_query, "history": new_history},
            rationale="lossless normalisation: whitespace, markdown, boilerplate",
            evidence={"tier": 1, "tokens_before": before},
        )


class Tier2Deduper:
    """Extractive redundancy removal.

    Sprint 0 scores sentence similarity lexically (Jaccard). Sprint 2 swaps in
    embeddings behind the same call — the selection logic does not change, only
    the similarity function, which is why it is isolated here.
    """

    module_id = "M1"
    name = "m1_tier2"
    reads = frozenset({"query", "history"})
    writes = frozenset({"query", "history"})

    def applies_to(self, ctx: RequestContext, cfg: ParsimonyConfig) -> bool:
        return cfg.enables("M1") and cfg.compression.tier2_enabled

    def propose(self, ctx: RequestContext, cfg: ParsimonyConfig) -> Proposal:
        d = ctx.derived
        if d is None:
            return NoOp("not_applicable", "no derived cache")

        sentences = d.sentences(ctx.query)
        if len(sentences) < 3:
            return NoOp("not_applicable", "fewer than 3 sentences")

        kept: list[str] = []
        dropped = 0
        for sent in sentences:
            if sent.startswith("```"):
                kept.append(sent)
                continue
            if d.token_count(sent) < cfg.compression.min_sentence_tokens:
                kept.append(sent)
                continue
            if any(jaccard(sent, k) >= cfg.compression.dedup_threshold for k in kept):
                dropped += 1
                continue
            kept.append(sent)

        if not dropped:
            return NoOp("no_yield", "no near-duplicate sentences")

        return ContextPatch(
            kind=TransformKind.REWRITE,
            fields={"query": " ".join(kept)},
            rationale=f"removed {dropped} near-duplicate sentence(s)",
            evidence={"tier": 2, "dropped_sentences": dropped, "kept": len(kept)},
        )


class Tier3Rewriter:
    """Tokenizer-aware rewriting with negative-yield detection.

    Disabled by default. Because byte-pair merges are context dependent,
    deleting a word can *raise* the token count by breaking a merge in its
    neighbours, so every candidate edit is re-tokenised and reverted if it does
    not actually pay. Ships in Sprint 3 together with the golden test that
    asserts windowed re-tokenisation matches full re-tokenisation.
    """

    module_id = "M1"
    name = "m1_tier3"
    reads = frozenset({"query"})
    writes = frozenset({"query"})

    LEXICON: tuple[tuple[str, str], ...] = (
        (r"\bin order to\b", "to"),
        (r"\bat this point in time\b", "now"),
        (r"\bdue to the fact that\b", "because"),
        (r"\bin the event that\b", "if"),
        (r"\bfor the purpose of\b", "for"),
        (r"\ba large number of\b", "many"),
        (r"\bis able to\b", "can"),
        (r"\bit is important to note that\b", ""),
    )

    def applies_to(self, ctx: RequestContext, cfg: ParsimonyConfig) -> bool:
        return cfg.enables("M1") and cfg.compression.tier3_enabled

    def skip_reason(self, ctx: RequestContext, cfg: ParsimonyConfig) -> str:
        if cfg.enables("M1") and not cfg.compression.tier3_enabled:
            return "tier 3 off until its golden test lands (Sprint 3)"
        return "not applicable"

    def propose(self, ctx: RequestContext, cfg: ParsimonyConfig) -> Proposal:
        d = ctx.derived
        if d is None:
            return NoOp("not_applicable", "no derived cache")

        text = ctx.query
        applied, rejected = 0, 0
        for pattern, replacement in self.LEXICON:
            candidate = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            if candidate == text:
                continue
            # Negative-yield detection: the edit must actually reduce tokens.
            if d.token_count(candidate) < d.token_count(text):
                text = re.sub(r"\s{2,}", " ", candidate).strip()
                applied += 1
            else:
                rejected += 1

        if not applied:
            return NoOp("no_yield", f"{rejected} edit(s) rejected as negative-yield")

        return ContextPatch(
            kind=TransformKind.REWRITE,
            fields={"query": text},
            rationale=f"{applied} token-reducing rewrite(s), {rejected} rejected",
            evidence={"tier": 3, "applied": applied, "negative_yield_rejected": rejected},
        )


def _replace_content(turn, content: str):
    from dataclasses import replace

    return replace(turn, content=content)


def stages() -> list:
    return [Tier1Normaliser(), Tier2Deduper(), Tier3Rewriter()]
