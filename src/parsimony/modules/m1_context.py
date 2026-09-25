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
from enum import Enum
import re
from dataclasses import dataclass, field, replace

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


class StopReason(str, Enum):
    """Why selection ended, as a closed set rather than a sentence.

    Subclasses `str` deliberately: these values are serialised into ledger rows
    and JSON payloads and interpolated into prose, and every one of those
    carried on working when this stopped being a bare string. The type exists
    so the surfaces that TRANSLATE a reason can be checked for covering all of
    them -- see `test_every_stop_reason_is_explained`.
    """

    #: Everything above the floor was kept and there was room for it.
    EXHAUSTED = "exhausted"
    #: The next best sentence scored below the relevance floor. More budget
    #: would change nothing.
    FLOOR = "relevance floor"
    #: Sentences above the floor were left behind for want of room. More budget
    #: would take them, which is what makes this the binding one when both apply.
    BUDGET = "budget"
    #: The question is not about this context at all; one sentence is kept so
    #: the prompt does not read as an instruction with a missing attachment.
    OFF_TOPIC = "nothing in the context bears on the question"

    # Without this, f"{StopReason.FLOOR}" renders as "StopReason.FLOOR" on
    # Python 3.11 and the phrase reaches the terminal and the proof document as
    # a type name. The whole point of subclassing str is that the existing
    # consumers keep working, so it has to format like one.
    __str__ = str.__str__


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
    stopped_by: StopReason
    coverage: float = 1.0                # question terms present in the context
    off_topic: bool = False
    #: The floor this selection actually used, and the reading that produced it.
    #: Constant when `context_adaptive_floor` is off; when it is on, nothing
    #: outside `select()` can recompute it, so it travels with the result.
    floor: "FloorReading" = field(default_factory=lambda: FloorReading(0.0, "not applicable"))
    #: unit index -> (tag, detail). Every unit appears exactly once, so a
    #: surface can state WHY each sentence went or stayed instead of showing a
    #: kept set and leaving the reader to guess. Tags name decisions this
    #: module actually makes -- ANCHOR, MATCH, CLOSURE, FLOOR, BUDGET,
    #: REDUNDANT, OFF-TOPIC -- and nothing else.
    reasons: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class UnitAudit:
    """One sentence, its score, and the decision taken on it."""

    index: int
    source: tuple[str, int]
    source_label: str
    position: int
    text: str
    tokens: int
    score: float
    kept: bool
    tag: str
    detail: str

    @property
    def protected(self) -> bool:
        return self.tag == "PROTECTED"


@dataclass(frozen=True, slots=True)
class ContextAudit:
    """Every sentence the request carried, with what happened to it."""

    units: tuple[UnitAudit, ...]
    tokens_before: int
    tokens_after: int
    budget: int
    coverage: float
    off_topic: bool
    stopped_by: StopReason
    anchors: dict
    applied: bool
    note: str = ""
    #: The floor in force for this request -- see `Selection.floor`.
    floor: "FloorReading" = field(default_factory=lambda: FloorReading(0.0, "not applicable"))

    @property
    def removed_tokens(self) -> int:
        return self.tokens_before - self.tokens_after

    @property
    def removed_pct(self) -> float:
        return 100.0 * self.removed_tokens / self.tokens_before if self.tokens_before else 0.0

    def by_source(self):
        groups: dict[tuple[str, int], list[UnitAudit]] = {}
        for unit in self.units:
            groups.setdefault(unit.source, []).append(unit)
        return groups


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

    "Most recent" means recent in the CONVERSATION, not last in the list this
    stage receives. M3's position-aware arrangement moves the most relevant
    turn to the end of that list, so protecting the last two positions
    protected exactly the turn worth compressing -- on ten conversations whose
    key fact sat in a long earlier answer, the tier fired once. Identifying the
    protected turns by id, from the original history, fixes it (ADR-043).
    """
    c = cfg.compression
    recent = {t.turn_id for t in ctx.original_history[-c.context_keep_recent_turns:]}         if c.context_keep_recent_turns else set()
    return [i for i, t in enumerate(ctx.history)
            if t.turn_id not in recent and t.token_count >= c.context_turn_min_tokens]


def build_units(ctx: RequestContext, turn_ids: list[int], count, prefix: int = 6) -> list[Unit]:
    """Sentences to choose between, in CONVERSATION order.

    Not in the order this stage happens to receive them: M3's position-aware
    arrangement moves the most relevant turn to the end, and the redundancy
    penalty in `select` compares each candidate against what is already kept,
    so ranking in list order made the kept set depend on where M4 was going to
    place things. Two arrangements of the same conversation then sent different
    numbers of tokens (648 against 611), which is exactly the confound ADR-025
    exists to rule out.
    """
    original = {t.turn_id: i for i, t in enumerate(ctx.original_history)}
    ordered = sorted(turn_ids, key=lambda i: original.get(ctx.history[i].turn_id, i))
    units: list[Unit] = []
    sources = [(("doc", i), d.content) for i, d in enumerate(ctx.documents)]
    sources += [(("turn", i), ctx.history[i].content) for i in ordered]
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


@dataclass(frozen=True, slots=True)
class FloorReading:
    """The floor in force for one request, and the reading that produced it."""

    value: float
    #: One sentence, for a reader. Never parsed: everything a surface needs to
    #: draw the reading is a field of its own.
    why: str
    #: 0-based rank the cliff falls after, in score order; -1 when there is none.
    rank: int = -1
    #: How steeply the scores fell there, as a ratio. 0.0 when there is no cliff.
    fall: float = 0.0
    #: True when this came from reading a distribution -- including a reading
    #: that found no cliff and fell back to the constant, which is a result.
    #: False for the configured constant, and for paths that never reach the
    #: floor at all. Surfaces branch on this, never on the wording of `why`.
    read: bool = False


def elbow_floor(rel: list[float], units: list["Unit"], budget: int, c) -> FloorReading:
    """Where the sorted scores fall off a cliff, as a floor.

    The floor only ever decides between sentences the budget could afford, so
    the shape that matters is the shape of that head -- not of the several
    hundred sentences below it, which are near zero on any long document and
    would drag every summary statistic to the same place.

    Within the head, the cliff is the steepest FALL between adjacent ranks --
    a ratio, not a difference. That distinction is the whole rule: measured on
    the development items, `1.00 0.29 0.22 | 0.03` is a collapse into noise
    whose largest additive gap is only 0.19, while `1.00 0.94 0.89 0.73` is a
    flat band of strong sentences whose largest additive gap is 0.25 -- bigger.
    Differences near the top of the range swamp collapses near the bottom,
    which is exactly backwards. A halving is a halving wherever it happens.

    A cliff means the evidence is concentrated and the rest is padding, so the
    floor goes into the middle of the fall and selection stops early, under
    budget -- the middle rather than the lower lip because no score lies
    strictly between two adjacent ranks, so both keep the same sentences and
    the midpoint sits furthest from either. No
    cliff means the scores are a smooth ramp with no natural break in it, and
    the honest reading of that is not "be permissive" but "this says nothing":
    the constant is what gets used, so the rule only ever departs from measured
    behaviour where it has something positive to depart on.

    Two guards keep it from being clever at the wrong moment. The cliff must be
    at rank `context_elbow_min_keep` or later, so one dominant sentence cannot
    collapse the context to itself; and the result is clamped, so an unusual
    distribution cannot produce a floor outside the range the constant was
    ever measured over.
    """
    order = sorted(range(len(rel)), key=lambda i: rel[i], reverse=True)
    head, used = [], 0
    for i in order:
        if used and used + units[i].tokens > budget:
            break
        head.append(rel[i])
        used += units[i].tokens
    lo, hi = c.context_elbow_floor_min, c.context_elbow_floor_max
    fixed = min(hi, max(lo, c.context_relevance_floor))
    if len(head) < c.context_elbow_min_keep + 2:
        # Too few candidates to read a shape from; the constant is as good a
        # guess as any and is what the rest of the system was measured with.
        return FloorReading(fixed, f"only {len(head)} candidates fit the budget, "
                                   f"too few to read a shape from", read=True)
    falls = [(head[k] / head[k + 1] if head[k + 1] > 0 else float("inf"), k)
             for k in range(c.context_elbow_min_keep - 1, len(head) - 1)]
    fall, k = max(falls)
    if fall < c.context_elbow_min_fall:
        return FloorReading(fixed, f"no cliff: the steepest fall over {len(head)} candidates "
                                   f"is {fall:.1f}x, under the {c.context_elbow_min_fall:.1f}x "
                                   f"a break has to be", read=True)
    # The midpoint of the cliff, not its lower lip: no score lies strictly
    # between two adjacent ranks, so this keeps exactly the same sentences
    # while sitting as far from either as it can.
    return FloorReading(min(hi, max(lo, (head[k] + head[k + 1]) / 2.0)),
                        f"the scores fall {fall:.1f}x after rank {k + 1} of {len(head)}",
                        k, fall, read=True)


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
    # A section heading names its sentences' subject once, for all of them.
    # Adjacency inheritance below only rescues sentences that OPEN dependently
    # ("It employs 58 people"); a plain "The site manager is Tomas Aguiar" under
    # "Porto office" reads as dependent to a person and as unrelated to BM25.
    by_title: dict[tuple[str, int], list[str]] = {}
    if c.context_section_anchors and titles:
        for source, title in titles.items():
            if title:
                by_title[source] = [a for a in anchors if _mentions(a, title)]
    mentions: list[list[str]] = []
    for i, u in enumerate(units):
        hits = [a for a in anchors if _mentions(a, u.text)]
        if not hits:
            hits = list(by_title.get(u.source, ()))
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

    # Absolute check, before anything relative: does the context bear on the
    # question at all? Everything below this point ranks sentences against each
    # other, which cannot tell "the best of six relevant sentences" from "the
    # least irrelevant of sixty". Measured separation on the tuning split is
    # wide -- 0.70-1.00 term coverage on-topic against 0.00-0.25 off-topic.
    wanted_terms = set(ranking_terms(query, c.context_term_prefix))
    present = set().union(*(set(u.terms) for u in units)) if units else set()
    coverage = len(wanted_terms & present) / len(wanted_terms) if wanted_terms else 1.0
    best_cosine = max(dense) if dense else None
    off_topic = coverage < c.context_topic_floor and (
        best_cosine is None or best_cosine < c.context_topic_cosine)
    if off_topic:
        # One sentence, not none: an empty context reads to the model as an
        # instruction with a missing attachment, and the gate refuses a context
        # annihilated outright. One sentence costs ~20 tokens and says plainly
        # that the documents were consulted.
        top = max(range(len(units)), key=lambda i: rel[i]) if units else None
        why = {i: ("OFF-TOPIC", f"the question shares {100 * coverage:.0f}% of its words "
                                f"with this context")
               for i in range(len(units))}
        if top is not None:
            why[top] = ("MATCH", "kept as the single closest sentence, so the context is not empty")
        return Selection(frozenset() if top is None else frozenset({top}), tuple(rel), {}, 0,
                         budget, StopReason.OFF_TOPIC, coverage, True,
                         floor=FloorReading(c.context_relevance_floor,
                                            "not reached: the context is off topic"),
                         reasons=why)

    kept: set[int] = set()
    used = 0
    why: dict[int, tuple[str, str]] = {}

    def keep(i: int) -> None:
        nonlocal used
        if i not in kept:
            kept.add(i)
            used += units[i].tokens

    # Guarantee: the best sentence for every name the question mentions.
    guaranteed = ({a: max(ids, key=lambda i: rel[i]) for a, ids in anchored.items()}
                  if c.context_anchor_guarantee else {})
    for name, i in guaranteed.items():
        keep(i)
        why[i] = ("ANCHOR", f"guaranteed: the question names {name!r}")

    stopped_by = StopReason.EXHAUSTED
    refused_for_room = 0
    reading = (elbow_floor(rel, units, budget, c) if c.context_adaptive_floor
               else FloorReading(c.context_relevance_floor, "the configured constant"))
    floor = reading.value

    lam = c.context_mmr_lambda
    remaining = [i for i in range(len(units)) if i not in kept]
    while remaining:
        def mmr(i: int) -> float:
            redundancy = max((_overlap(units[i].terms, units[j].terms) for j in kept),
                             default=0.0)
            return lam * rel[i] - (1 - lam) * redundancy

        best = max(remaining, key=mmr)
        if kept and gauge[best] < floor and rel[best] < floor:
            stopped_by = StopReason.FLOOR
            break
        if rel[best] <= 0.0 and kept:
            stopped_by = StopReason.FLOOR
            break
        remaining.remove(best)
        if used + units[best].tokens > budget:
            if not kept:
                keep(best)          # one sentence always survives
                why[best] = ("MATCH", f"kept anyway: nothing else fit ({rel[best]:.2f})")
            else:
                why.setdefault(best, ("BUDGET", f"scored {rel[best]:.2f} but the "
                                                f"{budget}-token budget was spent"))
                refused_for_room += 1
            # A smaller relevant sentence may still fit, so the loop goes on --
            # which is why the reason is decided after it, not here.
            continue
        keep(best)
        redundancy = max((_overlap(units[best].terms, units[j].terms)
                          for j in kept if j != best), default=0.0)
        why[best] = ("MATCH", f"relevance {rel[best]:.2f}"
                              + (f", overlap {redundancy:.2f} with a kept sentence"
                                 if redundancy >= 0.5 else ""))

    # Which constraint actually bound the result. "Budget" wins over "floor"
    # whenever something above the floor was refused for room, because that is
    # the one where more of it would change the outcome -- the question a reader
    # is really asking when they ask what stopped it.
    if refused_for_room:
        stopped_by = StopReason.BUDGET

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
                why[prev] = ("CLOSURE", "kept so the next sentence's opening word resolves")
                closure_added += 1
            j = prev

    for i in range(len(units)):
        if i in why:
            continue
        if i in kept:
            why[i] = ("MATCH", f"relevance {rel[i]:.2f}")
            continue
        redundancy = max((_overlap(units[i].terms, units[j].terms) for j in kept), default=0.0)
        if stopped_by is StopReason.BUDGET and rel[i] >= floor:
            why[i] = ("BUDGET", f"scored {rel[i]:.2f}; the {budget}-token budget was spent")
        elif redundancy >= 0.5 and rel[i] >= floor:
            why[i] = ("REDUNDANT", f"overlap {redundancy:.2f} with a sentence already kept")
        else:
            why[i] = ("FLOOR", f"relevance {rel[i]:.2f}, under the {floor:.2f} floor")

    return Selection(frozenset(kept), tuple(rel),
                     {a: units[i].text for a, i in guaranteed.items()},
                     closure_added, budget, stopped_by, coverage, False,
                     floor=reading, reasons=why)


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


def audit(ctx: RequestContext, cfg: ParsimonyConfig) -> ContextAudit:
    """Every sentence of the request's context, with the decision taken on it.

    One computation feeding every surface that explains this stage -- the
    terminal marginalia, the live dashboard, the web heatmap and the proof
    document. They were each about to re-derive "why was this dropped" from
    evidence dictionaries, and four re-derivations of one decision is four
    chances to describe the system as it is not.

    Protected turns (the most recent exchange) appear too, tagged PROTECTED, so
    a reader can see the sentences the stage deliberately never considered.
    """
    d = ctx.derived
    c = cfg.compression
    turn_ids = eligible_turns(ctx, cfg)
    count = d.token_count if d is not None else (lambda t: len(t.split()))
    units = build_units(ctx, turn_ids, count, c.context_term_prefix)
    labels: dict[tuple[str, int], str] = {
        ("doc", i): (doc.title or doc.doc_id) for i, doc in enumerate(ctx.documents)}
    labels.update({("turn", i): f"{ctx.history[i].role} turn {i + 1}"
                   for i in range(len(ctx.history))})

    before = sum(u.tokens for u in units)
    rows: list[UnitAudit] = []
    note = ""
    if not units or before < c.context_min_tokens:
        note = (f"context is {before} tokens, below the {c.context_min_tokens}-token threshold "
                f"where selection pays" if units else "no documents and no long earlier turns")
        for i, u in enumerate(units):
            rows.append(UnitAudit(i, u.source, labels.get(u.source, "context"), u.position,
                                  u.text, u.tokens, 1.0, True, "KEPT",
                                  "below the size threshold: nothing is removed"))
        selection = None
    else:
        dense = None
        if getattr(d, "has_embedder", False) and c.context_dense_weight > 0:
            vectors = d.embed([ctx.query] + [u.text for u in units])
            dense = [float(vectors[0] @ v) for v in vectors[1:]]
        titles = {("doc", i): doc.title for i, doc in enumerate(ctx.documents)}
        selection = select(ctx.query, units, cfg, dense, titles)
        for i, u in enumerate(units):
            tag, detail = selection.reasons.get(i, ("FLOOR", ""))
            rows.append(UnitAudit(i, u.source, labels.get(u.source, "context"), u.position,
                                  u.text, u.tokens, selection.relevance[i],
                                  i in selection.kept, tag, detail))

    # The sentences of protected turns: never candidates, and worth showing.
    protected = [i for i in range(len(ctx.history)) if i not in turn_ids]
    index = len(rows)
    for i in protected:
        turn = ctx.history[i]
        if turn.token_count < c.context_turn_min_tokens:
            continue
        for pos, (_line, text) in enumerate(_split_with_lines(turn.content)):
            rows.append(UnitAudit(index, ("turn", i), labels.get(("turn", i), "turn"), pos,
                                  text, count(text), 1.0, True, "PROTECTED",
                                  "in the most recent exchange, which is never compressed"))
            index += 1

    after = sum(u.tokens for u in rows if u.kept and u.tag != "PROTECTED")
    return ContextAudit(tuple(rows), before, after,
                        selection.budget if selection else before,
                        selection.coverage if selection else 1.0,
                        bool(selection and selection.off_topic),
                        selection.stopped_by if selection else "not applicable",
                        dict(selection.anchors) if selection else {},
                        applied=selection is not None, note=note,
                        floor=selection.floor if selection
                        else FloorReading(0.0, "selection did not run"))


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
                # The floor in force and how it was arrived at. Constant unless
                # `context_adaptive_floor` is on, and then not recomputable from
                # anything else in this row (ADR-051).
                "relevance_floor": round(sel.floor.value, 4),
                "floor_why": sel.floor.why,
                "floor_read": sel.floor.read,
                "stopped_by": sel.stopped_by,
                "topical_coverage": round(sel.coverage, 3),
                "off_topic": sel.off_topic,
                "scorer": "bm25+dense" if dense is not None else "bm25",
            },
        )


def stages() -> list:
    return [ContextCompressor()]
