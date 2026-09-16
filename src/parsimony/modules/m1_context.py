"""M1 context tier — question-aware extractive compression of long context.

WHY THIS TIER EXISTS
--------------------
Tiers 1-3 edit the question, and a question is a dozen tokens: the whole module
saved 0.2% of the corpus because there was nothing there to save. On CPU the
cost is prefill, ~8.5 ms per input token (ADR-034), and the input tokens of a
real request are overwhelmingly CONTEXT -- retrieved passages, a pasted report,
an earlier long answer. Almost none of it bears on any one question about it.

So this tier removes whole sentences from supplied documents and from older
long turns, keeping the ones the current question needs. It never rewrites a
sentence and never touches the question.

HOW IT DECIDES, and what each step is for
-----------------------------------------
Each is there because the obvious version fails without it (ADR-040):

  1. Relevance: Okapi BM25 over the sentences of THIS request, so rarity is
     judged against the context at hand -- "office" is noise in a document
     about six offices and signal in a document about one. Blended with the
     embedder's cosine, which catches morphology BM25 misses.
  2. Anchors: a sentence naming something the question names -- "Berlin",
     "Q3", "ticket 4471", a quoted string -- is boosted, and the best one per
     name is guaranteed. The fidelity gate enforces the guarantee
     independently (TransformKind.EXTRACT).
  3. Document prior: a sentence in a document that is relevant as a whole
     outranks an equally-scored stray sentence elsewhere. Coarse-to-fine, the
     structure LongLLMLingua found essential.
  4. Redundancy: Maximal Marginal Relevance, so the kept budget covers
     different facts rather than three phrasings of the best one.
  5. Budget with a relevance floor: stop at a token budget OR when the next
     sentence is no longer relevant, whichever is first. A fixed ratio pads
     with noise on a narrow question and starves a broad one.
  6. Dependency closure: a kept sentence that opens with "It", "This",
     "However" or "They" is unreadable without the sentence before it, so that
     sentence is kept too. Top-k retrieval returns "It was cancelled in 2024."
     with nothing to say what "it" was.
  7. Original order: kept sentences are emitted in document order, never in
     score order, so the model reads a shortened document rather than a
     shuffled one.

Pure Python over the request's own text: no second model, no network. The
decision costs milliseconds against seconds of prefill it removes.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace

from parsimony.core.config import ParsimonyConfig
from parsimony.core.proposals import ContextPatch, NoOp, Proposal, TransformKind
from parsimony.core.types import RequestContext
from parsimony.infra.embedding import content_terms
from parsimony.infra.nlp import RegexInvariantExtractor, split_sentences

#: A sentence opening with one of these leans on the sentence before it.
_DEPENDENT_OPENING = re.compile(
    r"^\W*(?:it|its|it's|this|that|these|those|they|them|their|he|she|his|her|"
    r"such|there|here|however|therefore|thus|hence|also|instead|otherwise|"
    r"meanwhile|consequently|furthermore|moreover|additionally|similarly|"
    r"likewise|nevertheless|still|then|later|afterwards|as a result|in addition|"
    r"in contrast|by contrast|on the other hand|for example|for instance|"
    r"the latter|the former|the same|both|neither|either|each|which)\b",
    re.IGNORECASE,
)

_EXTRACTOR = RegexInvariantExtractor()

def ranking_terms(text: str, prefix: int = 6) -> list[str]:
    """Content terms cut to a fixed length, for ranking only.

    The shared stemmer is deliberately crude and leaves "employees" as
    "employe" and "employs" as "employ", so "Which office has more employees?"
    found no term in "It employs 58 people" -- the sentence holding the answer.
    Fixed-length truncation is a long-established stemming method in
    retrieval; six characters (CompressionConfig.context_term_prefix)
    conflates inflections without merging "station" with "statistics".
    """
    return [t[:prefix] for t in content_terms(text)]


@dataclass(frozen=True, slots=True)
class Unit:
    """One sentence of one source (a document or a turn)."""

    source: tuple[str, int]      # ("doc", i) or ("turn", i)
    position: int                # sentence index within the source
    line: int                    # line within the source, for re-joining
    text: str
    tokens: int
    terms: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Selection:
    kept: frozenset[int]                 # unit indices
    relevance: tuple[float, ...]         # final score per unit, 0..1
    anchors: dict                        # value -> unit index guaranteed for it
    closure_added: int
    budget: int
    stopped_by: str


def _split_with_lines(content: str) -> list[tuple[int, str]]:
    """Sentences with the line they sit on, matching `split_sentences` exactly.

    The gate re-splits the source with `split_sentences`; any other splitter
    here would produce sentences the gate does not recognise.
    """
    if "```" in content:
        return [(0, s) for s in split_sentences(content)]
    out: list[tuple[int, str]] = []
    for li, line in enumerate(content.split("\n")):
        out.extend((li, s) for s in split_sentences(line))
    return out


def eligible_turns(ctx: RequestContext, cfg: ParsimonyConfig) -> list[int]:
    """Older turns long enough to be worth compressing.

    The most recent exchange is never touched: a follow-up ("and the second
    one?") points into it, and relevance to the follow-up's words cannot see
    what it points at.
    """
    c = cfg.compression
    keep_from = max(0, len(ctx.history) - c.context_keep_recent_turns)
    return [i for i, t in enumerate(ctx.history[:keep_from])
            if t.token_count >= c.context_turn_min_tokens]


def build_units(ctx: RequestContext, turn_ids: list[int], count, prefix: int = 6) -> list[Unit]:
    units: list[Unit] = []
    sources = [(("doc", i), d.content) for i, d in enumerate(ctx.documents)]
    sources += [(("turn", i), ctx.history[i].content) for i in turn_ids]
    for source, content in sources:
        for pos, (line, text) in enumerate(_split_with_lines(content)):
            units.append(Unit(source, pos, line, text, count(text),
                              tuple(ranking_terms(text, prefix))))
    return units


def bm25(query_terms: list[str], units: list[Unit], k1: float, b: float) -> list[float]:
    """Okapi BM25, with document frequency taken over this request's sentences."""
    n = len(units)
    if not n or not query_terms:
        return [0.0] * n
    df: dict[str, int] = {}
    for u in units:
        for t in set(u.terms):
            df[t] = df.get(t, 0) + 1
    avg = sum(len(u.terms) for u in units) / n or 1.0
    wanted = set(query_terms)
    scores = []
    for u in units:
        tf: dict[str, int] = {}
        for t in u.terms:
            if t in wanted:
                tf[t] = tf.get(t, 0) + 1
        s = 0.0
        for t, f in tf.items():
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(u.terms) / avg))
        scores.append(s)
    return scores


def _normalise(values: list[float]) -> list[float]:
    top = max(values, default=0.0)
    return [v / top if top > 0 else 0.0 for v in values]


def _overlap(a: tuple[str, ...], b: tuple[str, ...]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa and sb else 0.0


def anchor_values(query: str) -> list[str]:
    """What the question names: numbers, names, identifiers, quoted strings."""
    inv = _EXTRACTOR.extract(query)
    return sorted(inv.numbers | inv.entities | inv.quoted, key=str.lower)


def _mentions(value: str, text: str) -> bool:
    left = r"\b" if value[:1].isalnum() else ""
    right = r"\b" if value[-1:].isalnum() else ""
    return re.search(f"{left}{re.escape(value)}{right}", text, re.IGNORECASE) is not None


def select(query: str, units: list[Unit], cfg: ParsimonyConfig,
           dense: list[float] | None = None,
           titles: dict[tuple[str, int], str] | None = None) -> Selection:
    c = cfg.compression
    total = sum(u.tokens for u in units)
    budget = max(c.context_min_keep_tokens, math.ceil(c.context_target_ratio * total))

    lexical = _normalise(bm25(ranking_terms(query, c.context_term_prefix), units,
                              c.bm25_k1, c.bm25_b))
    if dense is not None:
        dense_n = _normalise([max(0.0, x) for x in dense])
        w = c.context_dense_weight
        rel = [(1 - w) * lx + w * dn for lx, dn in zip(lexical, dense_n)]
    else:
        rel = lexical
    base = list(rel)

    # A sentence opening "It employs 58 people" is about whatever the sentence
    # before it named. Resolving the pronoun by adjacency -- the sentence
    # inherits its predecessor's mentions -- lets "How many people does the
    # Tallinn office employ?" find it, where string matching never could.
    anchors = anchor_values(query)
    mentions: list[list[str]] = []
    for i, u in enumerate(units):
        hits = [a for a in anchors if _mentions(a, u.text)]
        if (not hits and i > 0 and _DEPENDENT_OPENING.match(u.text)
                and units[i - 1].source == u.source):
            hits = list(mentions[i - 1])
        mentions.append(hits)
    anchored: dict[str, list[int]] = {}
    for i, hits in enumerate(mentions):
        for a in hits:
            anchored.setdefault(a, []).append(i)
        if hits:
            rel[i] += c.context_anchor_bonus * len(hits) / len(anchors)

    # Coarse-to-fine: scale by how relevant the whole source is -- its best
    # sentence, or its title, whichever says more.
    best_in_source: dict[tuple[str, int], float] = {}
    for u, r in zip(units, rel):
        best_in_source[u.source] = max(best_in_source.get(u.source, 0.0), r)
    if titles and c.context_title_weight > 0:
        wanted = set(ranking_terms(query, c.context_term_prefix))
        for source, title in titles.items():
            if source not in best_in_source or not title:
                continue
            terms = set(ranking_terms(title, c.context_term_prefix))
            overlap = len(wanted & terms) / len(wanted) if wanted else 0.0
            named = sum(_mentions(a, title) for a in anchors) / len(anchors) if anchors else 0.0
            score = c.context_title_weight * (overlap + c.context_anchor_bonus * 2 * named)
            best_in_source[source] = max(best_in_source[source], score)
    top_source = max(best_in_source.values(), default=0.0) or 1.0
    dw = c.context_doc_weight
    factor = [(1 - dw) + dw * best_in_source[u.source] / top_source for u in units]
    rel = _normalise([r * f for r, f in zip(rel, factor)])
    gauge = (_normalise([b * f for b, f in zip(base, factor)])
             if c.context_floor_before_bonus else rel)

    kept: set[int] = set()
    used = 0

    def keep(i: int) -> None:
        nonlocal used
        if i not in kept:
            kept.add(i)
            used += units[i].tokens

    # Guarantee: the best sentence for every name the question mentions.
    guaranteed = ({a: max(ids, key=lambda i: rel[i]) for a, ids in anchored.items()}
                  if c.context_anchor_guarantee else {})
    for i in guaranteed.values():
        keep(i)

    stopped_by = "exhausted"
    floor = c.context_relevance_floor
    lam = c.context_mmr_lambda
    remaining = [i for i in range(len(units)) if i not in kept]
    while remaining:
        def mmr(i: int) -> float:
            redundancy = max((_overlap(units[i].terms, units[j].terms) for j in kept),
                             default=0.0)
            return lam * rel[i] - (1 - lam) * redundancy

        best = max(remaining, key=mmr)
        if kept and gauge[best] < floor and rel[best] < floor:
            stopped_by = "relevance floor"
            break
        if rel[best] <= 0.0 and kept:
            stopped_by = "relevance floor"
            break
        remaining.remove(best)
        if used + units[best].tokens > budget:
            if not kept:
                keep(best)          # one sentence always survives
            stopped_by = "budget"
            # A smaller relevant sentence may still fit.
            continue
        keep(best)

    # Dependency closure, walking back through consecutive dependent openings.
    closure_added = 0
    index_of = {(u.source, u.position): i for i, u in enumerate(units)}
    for i in sorted(kept):
        j = i
        for _ in range(c.context_closure_depth):
            if not _DEPENDENT_OPENING.match(units[j].text):
                break
            prev = index_of.get((units[j].source, units[j].position - 1))
            if prev is None:
                break
            if prev not in kept:
                keep(prev)
                closure_added += 1
            j = prev

    return Selection(frozenset(kept), tuple(rel),
                     {a: units[i].text for a, i in guaranteed.items()},
                     closure_added, budget, stopped_by)


def render_source(units: list[Unit], kept: frozenset[int]) -> str:
    """Kept sentences in their original order, re-joined by line."""
    out: list[str] = []
    last_line = None
    for idx, u in enumerate(units):
        if idx not in kept:
            continue
        if last_line is None:
            out.append(u.text)
        elif u.line != last_line:
            out.append("\n" + u.text)
        else:
            out.append(" " + u.text)
        last_line = u.line
    return "".join(out)


class ContextCompressor:
    module_id = "M1"
    name = "m1_context"
    reads = frozenset({"query", "history", "documents"})
    writes = frozenset({"history", "documents"})

    def applies_to(self, ctx: RequestContext, cfg: ParsimonyConfig) -> bool:
        return (cfg.enables("M1") and cfg.compression.context_enabled
                and (bool(ctx.documents) or bool(eligible_turns(ctx, cfg))))

    def skip_reason(self, ctx: RequestContext, cfg: ParsimonyConfig) -> str:
        if cfg.enables("M1") and not cfg.compression.context_enabled:
            return "context compression disabled in this configuration"
        return "no documents and no long earlier turns"

    def propose(self, ctx: RequestContext, cfg: ParsimonyConfig) -> Proposal:
        d = ctx.derived
        if d is None:
            return NoOp("not_applicable", "no derived cache")
        c = cfg.compression

        turn_ids = eligible_turns(ctx, cfg)
        units = build_units(ctx, turn_ids, d.token_count, c.context_term_prefix)
        before = sum(u.tokens for u in units)
        if before < c.context_min_tokens:
            return NoOp("not_applicable",
                        f"context is {before} tokens, below the {c.context_min_tokens}-token "
                        f"threshold where selection pays",
                        {"context_tokens": before})
        if not ranking_terms(ctx.query, c.context_term_prefix):
            return NoOp("not_applicable", "the question has no content words to rank by")

        dense = None
        if getattr(d, "has_embedder", False) and c.context_dense_weight > 0:
            vectors = d.embed([ctx.query] + [u.text for u in units])
            dense = [float(vectors[0] @ v) for v in vectors[1:]]

        titles = {("doc", i): doc.title for i, doc in enumerate(ctx.documents)}
        sel = select(ctx.query, units, cfg, dense, titles)

        by_source: dict[tuple[str, int], list[Unit]] = {}
        index: dict[tuple[str, int], list[int]] = {}
        for i, u in enumerate(units):
            by_source.setdefault(u.source, []).append(u)
            index.setdefault(u.source, []).append(i)

        def rendered(source) -> str:
            local = index.get(source, [])
            kept_local = frozenset(k for k, i in enumerate(local) if i in sel.kept)
            return render_source(by_source.get(source, []), kept_local)

        documents = []
        dropped_docs = 0
        for i, doc in enumerate(ctx.documents):
            text = rendered(("doc", i))
            if text:
                documents.append(replace(doc, content=text) if text != doc.content else doc)
            else:
                dropped_docs += 1

        history = list(ctx.history)
        for i in turn_ids:
            text = rendered(("turn", i))
            if not text:
                # A turn is never emptied: its opening sentence says what it was.
                text = by_source[("turn", i)][0].text
            if text != history[i].content:
                history[i] = replace(history[i], content=text, token_count=d.token_count(text))

        after = sum(d.token_count(doc.content) for doc in documents) + sum(
            d.token_count(history[i].content) for i in turn_ids)
        if before - after < max(c.context_min_saving_tokens, 1):
            return NoOp("no_yield", f"selection would save {before - after} tokens",
                        {"context_tokens": before})

        kept_sentences = len(sel.kept)
        return ContextPatch(
            kind=TransformKind.EXTRACT,
            fields={"documents": tuple(documents), "history": tuple(history)},
            rationale=(f"kept {kept_sentences} of {len(units)} sentences relevant to the "
                       f"question ({before} -> {after} context tokens)"),
            evidence={
                "context_tokens_before": before,
                "context_tokens_after": after,
                "tokens_saved": before - after,
                "sentences": len(units),
                "sentences_kept": kept_sentences,
                "documents": len(ctx.documents),
                "documents_dropped": dropped_docs,
                "turns_compressed": len(turn_ids),
                "anchors": dict(sel.anchors),
                "closure_added": sel.closure_added,
                "budget_tokens": sel.budget,
                "stopped_by": sel.stopped_by,
                "scorer": "bm25+dense" if dense is not None else "bm25",
            },
        )


def stages() -> list:
    return [ContextCompressor()]
