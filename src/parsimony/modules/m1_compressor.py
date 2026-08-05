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
from parsimony.modules.m1_tier3 import rewrite as tier3_rewrite

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

        use_embeddings = getattr(d, "has_embedder", False)
        if use_embeddings:
            # One batched call for every sentence, not one call per comparison.
            vectors = d.embed(list(sentences))
            threshold = cfg.compression.dedup_threshold
            scorer = "cosine"
        else:
            vectors = None
            threshold = cfg.compression.dedup_threshold_lexical
            scorer = "jaccard"

        kept: list[str] = []
        kept_idx: list[int] = []
        dropped = 0
        for i, sent in enumerate(sentences):
            if sent.startswith("```"):
                kept.append(sent)
                kept_idx.append(i)
                continue
            if d.token_count(sent) < cfg.compression.min_sentence_tokens:
                kept.append(sent)
                kept_idx.append(i)
                continue

            if vectors is not None:
                duplicate = any(
                    float(vectors[i] @ vectors[j]) >= threshold for j in kept_idx
                )
            else:
                duplicate = any(jaccard(sent, k) >= threshold for k in kept)

            if duplicate:
                dropped += 1
                continue
            kept.append(sent)
            kept_idx.append(i)

        if not dropped:
            return NoOp("no_yield", "no near-duplicate sentences")

        return ContextPatch(
            kind=TransformKind.REWRITE,
            fields={"query": " ".join(kept)},
            rationale=f"removed {dropped} near-duplicate sentence(s) [{scorer}]",
            evidence={
                "tier": 2,
                "dropped_sentences": dropped,
                "kept": len(kept),
                "scorer": scorer,
                "threshold": threshold,
            },
        )


class Tier3Rewriter:
    """Tokenizer-aware rewriting with negative-yield detection.

    Because byte-pair merges are context dependent, an edit that removes
    characters can *raise* the token count by breaking a merge that spanned the
    boundary. Every candidate is re-tokenised and reverted if it does not
    actually pay. Implementation in modules/m1_tier3.py; the windowed
    re-tokenisation it relies on is guarded by tests/golden/.
    """

    module_id = "M1"
    name = "m1_tier3"
    reads = frozenset({"query"})
    writes = frozenset({"query"})

    def applies_to(self, ctx: RequestContext, cfg: ParsimonyConfig) -> bool:
        return cfg.enables("M1") and cfg.compression.tier3_enabled

    def skip_reason(self, ctx: RequestContext, cfg: ParsimonyConfig) -> str:
        if cfg.enables("M1") and not cfg.compression.tier3_enabled:
            return "tier 3 disabled in this configuration"
        return "not applicable"

    def propose(self, ctx: RequestContext, cfg: ParsimonyConfig) -> Proposal:
        d = ctx.derived
        if d is None:
            return NoOp("not_applicable", "no derived cache")

        new_text, outcomes = tier3_rewrite(
            ctx.query, _TokenCounter(d), window=cfg.compression.retokenise_window
        )
        applied = [o for o in outcomes if o.applied]
        rejected = [o for o in outcomes if not o.applied]

        if not applied:
            detail = (
                f"{len(rejected)} candidate edit(s) rejected as negative-yield"
                if rejected
                else "no candidate edits"
            )
            return NoOp(
                "no_yield",
                detail,
                {"tier": 3, "candidates": len(outcomes), "negative_yield_rejected": len(rejected)},
            )

        return ContextPatch(
            kind=TransformKind.REWRITE,
            fields={"query": new_text},
            rationale=(
                f"{len(applied)} token-reducing rewrite(s), "
                f"{len(rejected)} rejected as negative-yield"
            ),
            evidence={
                "tier": 3,
                "applied": len(applied),
                "negative_yield_rejected": len(rejected),
                "tokens_saved": -sum(o.windowed_delta for o in applied),
                "rules": sorted({o.candidate.rule for o in applied}),
            },
        )


class _TokenCounter:
    """Adapts DerivedCache to the minimal tokenizer surface tier 3 needs, so the
    rewriting logic stays testable against a bare tokenizer."""

    __slots__ = ("_d",)

    def __init__(self, derived) -> None:
        self._d = derived

    def count(self, text: str) -> int:
        return self._d.token_count(text)


def _replace_content(turn, content: str):
    from dataclasses import replace

    return replace(turn, content=content)


def stages() -> list:
    return [Tier1Normaliser(), Tier2Deduper(), Tier3Rewriter()]
