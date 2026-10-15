"""M2 — Two-Tier Semantic Cache.

Tier 0 is an exact hash; tier 1 is cosine over an exact vector index. A single
similarity threshold is replaced by verification: a candidate above the floor
is settled by cheap lexical and invariant agreement rather than by the
embedding alone.

It was a three-zone policy -- accept, verify, reject -- until a better encoder
showed that the accept zone's safety came from the encoder's weakness rather
than from the threshold. MiniLM scores "is X safe" against "is X NOT safe" at
0.996, above any usable tau_hi, where the lexical encoder scored it 0.924 and
the verifier caught it. Verification now runs on every candidate
(`verify_always`), and the zone survives only as a switch so the design the
literature uses stays measurable (ADR-041).

The verifier is where the real work happens, and it is deliberately not a second
neural forward pass. Two questions differing by one operative token — the
adversarial subset — sit at very high cosine similarity under any encoder, and
under a lexical encoder they sit higher still. Nothing in the vector geometry
separates "is X safe" from "is X not safe". A set comparison over numbers,
entities and negations does, in microseconds.

Cache keys include model_id: report 4.6 re-runs the winning configuration on
three models, and without it a Llama-generated answer would be served during
the Qwen run, silently corrupting the generalisation study.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field

import numpy as np

from parsimony.core.config import ParsimonyConfig
from parsimony.core.proposals import NoOp, Proposal, ShortCircuit
from parsimony.core.types import Invariants, RequestContext, RouteTier
from parsimony.infra.embedding import STOPWORDS, ExactIndex
from parsimony.infra.nlp import (
    RegexInvariantExtractor,
    entity_words,
    modifiers_oppose,
    morphological_negations,
    operative_modifiers,
    sanitise,
)

_WS_RE = re.compile(r"\s+")
_VOLATILE_RE = re.compile(
    r"\b(today|now|current|currently|latest|recent|this (week|month|year)|"
    r"price|rate|stock|weather|news)\b",
    re.IGNORECASE,
)


_ALNUM_RE = re.compile(r"[a-z0-9]")


def canonicalise(text: str) -> str:
    return _WS_RE.sub(" ", text.strip().lower()).rstrip("?.! ")


def is_cacheable(query: str) -> bool:
    """Is this query safe to key on?

    Canonicalisation strips trailing punctuation so that "what is X?" and
    "what is X" share a key. That is lossless for real queries and CATASTROPHIC
    for degenerate ones: "   ", "?!...", "!!!" and "" all canonicalise to the
    empty string and therefore to the SAME key. Measured before this guard, six
    distinct inputs collided and served each other's answers.

    The exact-hash tier is the dangerous place for this, because it
    short-circuits BEFORE the three-zone verifier runs — the verifier only
    guards the semantic tier. Hash equality is trusted as semantic equality, so
    the canonical form has to actually carry information.

    This is the collision class the key-collision literature describes
    (docs/03-decision-log.md, ADR-029), reachable here without any adversarial
    search at all.
    """
    return bool(_ALNUM_RE.search(canonicalise(query)))


# ---------------------------------------------------------------------------
# What the cache compares
# ---------------------------------------------------------------------------
#
# What a person types varies in ways that carry no meaning: a greeting, "could
# you please", "tell me", a trailing "thanks!", singular against plural. Live,
# those differences alone sank reuse. "Hello, could you please explain what
# recursion is? Thanks!" scored 0.632 against "What is recursion?" and was
# rejected; "Explain how hash tables work" scored 0.706 against "How does a
# hash table work?", because the shared stemmer reduces "tables" to "tabl"
# while leaving "table" alone.
#
# So the cache compares a KEY TEXT rather than the raw question. The raw text is
# still what gets stored, shown and fidelity-checked -- only the comparison is
# normalised.

_GREETING_RE = re.compile(
    r"^(?:hello|hi|hey|hiya|greetings|good (?:morning|afternoon|evening)|dear \w+)"
    r"\b[\s,!.:;-]*"
)
_WONDER_RE = re.compile(
    r"^(?:i (?:was|am|'m) wondering (?:if|whether)(?: you (?:could|can|would))?"
    r"|i(?: would|'d) like to know|i want to know|i need to know|do you know)\b[\s,]*"
)
_PLEASE_RE = re.compile(r"\b(?:please|kindly)\b")
_SIGNOFF_RE = re.compile(
    r"[\s,.!;:-]*(?:thanks(?: so much| a lot)?(?: in advance)?"
    r"|thank you(?: so much| very much)?(?: in advance)?"
    r"|cheers|much appreciated|regards)[\s,.!]*$"
)
_FRAMING_RE = re.compile(
    r"^(?:(?:can|could|would|will) you\s+)?"
    r"(?:tell me(?: about)?|explain(?: to me)?|describe|define|give me|show me|list"
    r"|summari[sz]e|outline)\b\s*"
)
_LEADING_PHRASE_RE = re.compile(r"^(?:in|for|with|on|using|as for)\s+[^,?]{1,30},\s*")
_AUX_RE = re.compile(
    r"^(?:is|are|was|were|can|could|do|does|did|should|shall|will|would|has|have|had"
    r"|may|might|must)\b"
)
_INTERROGATIVE_RE = re.compile(
    r"^(?:in (?:what|which) (?:year|decade|century|month)|how many|how much"
    r"|how (?:long|far|big|tall|old|heavy|deep|wide|fast|large|high|often)"
    r"|what causes|what is the reason (?:for|why)|how come|why|whom|whose|who|where"
    r"|when|what time|what year|which|what|how)\b\s*"
)

#: Kind of question, in the order the patterns must be tried. Two questions of
#: different kinds are never the same question however similar their words:
#: "What is Java?" and "Where is Java?" both reduce to the single content word
#: "java" and score cosine 1.000 -- an auto-accept, and a wrong answer.
_QUESTION_KINDS = (
    (re.compile(r"^in (?:what|which) (?:year|decade|century|month)\b"), "when"),
    (re.compile(r"^(?:what is|what's|what are) the differences? between\b"
                r"|^how (?:do|does|is|are)\b.{1,60}?\bdiffer\b"
                r"|^(?:compare|contrast)\b"), "compare"),
    (re.compile(r"^(?:how many|how much"
                r"|how (?:long|far|big|tall|old|heavy|deep|wide|fast|large|high|often))\b"
                r"|^(?:what is|what's|what are) the "
                r"(?:distance|length|height|size|weight|age|depth|width|speed|population"
                r"|number|amount|count|cost|price|temperature|area|volume|duration) of\b"),
     "quantity"),
    (re.compile(r"^(?:what causes|what is the reason|how come|why)\b"), "why"),
    (re.compile(r"^(?:who|whom|whose)\b"), "who"),
    (re.compile(r"^where\b"), "where"),
    (re.compile(r"^(?:when|what time|what year|what date)\b"), "when"),
    (re.compile(r"^how\b"), "how"),
    (re.compile(r"^(?:which|what)\b"), "what"),
)


def courtesy_stripped(query: str) -> str:
    """The question with greeting, please/kindly and sign-off removed."""
    s = sanitise(query).lower().strip()
    s = _GREETING_RE.sub("", s)
    s = _WONDER_RE.sub("", s)
    s = _PLEASE_RE.sub(" ", s)
    s = _SIGNOFF_RE.sub("", s)
    return _WS_RE.sub(" ", s).strip(" ,")


def _kind_of(text: str) -> str | None:
    text = _LEADING_PHRASE_RE.sub("", text)
    for pattern, kind in _QUESTION_KINDS:
        if pattern.search(text):
            return kind
    return None


def question_type(query: str) -> str | None:
    """Which kind of question this is, or None when it cannot be told.

    None is a wildcard: an unclassifiable query agrees with anything, so this
    check can only ever refuse a pair whose kinds are both known and different.
    That is the safe direction -- the alternative loses genuine reuse.
    """
    s = courtesy_stripped(query)
    framed = _FRAMING_RE.match(s)
    if framed:
        # "Explain how DNS works" asks a how-question; "Define X" asks a what.
        return _kind_of(s[framed.end():]) or "what"
    kind = _kind_of(s)
    if kind is not None:
        return kind
    return "yesno" if _AUX_RE.match(_LEADING_PHRASE_RE.sub("", s)) else None


def question_types_agree(a: str, b: str, embed=None) -> bool:
    """Whether these two ask for the same kind of thing.

    With an encoder, the comparison is over the kind of answer each wants
    rather than the word that asked for it (ADR-054). Without one it is the
    surface opener, which refuses three rewordings in six and is the safe
    direction to be wrong in.
    """
    ka, kb = question_type(a), question_type(b)
    if ka is None or kb is None or ka == kb:
        return True
    if embed is None:
        return False
    # Only ever consulted to RESCUE a pair the openers refuse, never to overturn
    # one they accept. Without that the resolver's own noise invents
    # disagreements between two "what" questions: "What is machine learning?"
    # and "Define machine learning" agreed on the surface and resolved to "why"
    # and "who", costing three pairs of the free-typing corpus.
    #
    # And not when the opener is the ONLY difference. "What is the deadline?"
    # and "When is the deadline?" have one content word between them and it is
    # the same word, so the opener carries the whole question; a reworded
    # question restates its subject as well. Without this the resolver reads
    # "deadline" as a when-question and serves one for the other.
    if content_words(cache_key_text(a)) == content_words(cache_key_text(b)):
        return False
    ra = resolved_question_kind(a, embed)
    rb = resolved_question_kind(b, embed)
    return ra is None or rb is None or ra == rb


#: Suffixes that only look plural. Stripping the "s" destroys these words:
#: analysis -> analysi, status -> statu, physics -> physic.
_NOT_PLURAL = ("ss", "us", "is", "ous", "ics", "xis", "sis")


def singularise(word: str) -> str:
    """Plural to singular, crudely but SYMMETRICALLY.

    The shared stemmer strips "es" before "s", turning "tables" into "tabl"
    while leaving "table" untouched, so a plural and its own singular stopped
    matching. This runs before the encoder sees either side, so both reduce to
    the same form.
    """
    if len(word) < 5 or word.isdigit() or word.endswith(_NOT_PLURAL):
        return word
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-3] + "y"
    if re.search(r"(?:ch|sh|x|z)es$", word):
        return word[:-2]
    return word[:-1] if word.endswith("s") else word


def cache_key_text(query: str) -> str:
    """The text the cache embeds and compares -- never the text it stores.

    Courtesy, request framing and the interrogative opener come out, and plurals
    fold. The interrogative goes because `question_type` carries it separately:
    leaving "where" in the compared text would raise similarity between every
    pair of where-questions, which is the opposite of what it is for.
    """
    s = courtesy_stripped(query)
    for _ in range(2):  # "could you explain how X works": framing, then "how"
        s = _FRAMING_RE.sub("", s)
        s = _INTERROGATIVE_RE.sub("", s)
    key = " ".join(singularise(w) for w in re.findall(r"[a-z0-9]+", s))
    # A question that is nothing but courtesy still has to key on something.
    return key or canonicalise(query)


#: "What's" without this sheds its verb and keeps its suffix: the interrogative
#: goes and a bare "s" is left at the head of the embedded text.
_CONTRACTION_RE = re.compile(
    r"\b(what|who|where|when|why|how|that|it|there)'s\b", re.IGNORECASE)
_NOT_CONTRACTION_RE = re.compile(r"n't\b", re.IGNORECASE)
#: Only between digits. "x" is a variable everywhere else.
_TIMES_RE = re.compile(r"(?<=\d)\s*[x\u00d7*]\s*(?=\d)")


def embed_text(query: str) -> str:
    """The text handed to the encoder -- not the text compared lexically.

    `cache_key_text` strips the interrogative so that two where-questions do
    not look alike for being where-questions. Under a lexical encoder that is
    right. Under the neural one it costs more than it saves: what is left is a
    fragment beginning with a stranded copula, and paraphrase recall drops from
    8 in 10 to 6 in 10 while the distractors do not move at all.

    So the encoder gets the question as asked, minus the courtesy, with
    contractions expanded and plurals folded the same way both sides fold them.
    The verifier and the overlap floor go on using `cache_key_text`, where
    stripping the opener is still the right call.
    """
    s = _CONTRACTION_RE.sub(lambda m: m.group(1) + " is", query)
    s = _NOT_CONTRACTION_RE.sub(" not", s)
    s = _TIMES_RE.sub(" times ", s)
    s = courtesy_stripped(s)
    key = " ".join(singularise(w) for w in re.findall(r"[a-z0-9]+", s.lower()))
    # A question that is nothing but courtesy still has to embed something.
    return key or canonicalise(query)


def content_words(key_text: str) -> frozenset[str]:
    """Key-text words carrying subject matter, for the overlap floor."""
    return frozenset(w for w in key_text.split() if w not in STOPWORDS)


#: Words that make a question depend on what came before it. A query carrying
#: any of these cannot be answered without the conversation, so its cache entry
#: must be scoped to that conversation. Deliberately over-inclusive: a
#: self-contained question wrongly treated as dependent costs a cache hit, while
#: a dependent one wrongly treated as self-contained serves the wrong answer.
_DEICTIC = frozenset("""
it its it's this that these those they them their there then
he she him her his hers
former latter above below aforementioned
same other another
instead also too again still yet
""".split())

#: Position words that point backwards only when they point AT something --
#: "the second one". "When was Python first released?" uses "first" as an
#: adverb and is perfectly self-contained, but the flat list scoped it to one
#: conversation and cost the hit.
_POSITIONAL = frozenset("first second third last next previous".split())
_DETERMINERS = frozenset("the this that my your our its their".split())

#: "it" is not always a reference. In "why does it rain" and "is it safe to mix
#: bleach and vinegar" it points at nothing; it is grammatical filler. Treating
#: those as follow-ups scoped them to a single conversation and cost every hit
#: -- and worse, it made the adversarial negation pair miss for the wrong
#: reason, so the verifier never ran and the safety property was never shown.
_WEATHER = frozenset("""
rain rains raining rained snow snows snowing snowed hail hails hailing
thunder thundering drizzle drizzling pour pouring
""".split())

_FOLLOW_UP_OPENERS = (
    "and ", "but ", "so ", "then ", "what about", "how about", "why not",
    "ok ", "okay ", "yes ", "no ", "sure ",
)


def _filler_it(tokens: list[str]) -> set[int]:
    """Indices of an "it" (and any complementiser "that") that refer to nothing.

    Conservative by construction, because a referential "it" mistaken for filler
    is the failure that serves a wrong answer. Filler is only recognised when
    what follows actually supplies the subject: "is it safe to mix bleach" has
    an object after the verb, while "is it safe to eat?" does not and stays a
    follow-up.
    """
    drop: set[int] = set()
    for i, word in enumerate(tokens):
        if word not in ("it", "it's"):
            continue
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        prev = tokens[i - 1] if i else ""
        if nxt in _WEATHER or (
            word == "it" and nxt in ("is", "was")
            and i + 2 < len(tokens) and tokens[i + 2] in _WEATHER
        ):
            drop.add(i)
            continue
        copular = (
            (word == "it" and prev in {"is", "was", "isn't", "wasn't", "will", "would"})
            or word == "it's"
            or (word == "it" and nxt in ("is", "was"))
        )
        measuring = word == "it" and prev == "does" and nxt in ("take", "cost")
        if not (copular or measuring):
            continue
        for j in range(i + 1, min(i + 5, len(tokens))):
            if tokens[j] == "to":
                # An object after the verb means the "it" was standing in for
                # the action, not for something said earlier.
                if j + 1 < len(tokens) and any(
                    t not in STOPWORDS for t in tokens[j + 2:]
                ):
                    drop.add(i)
                break
            if tokens[j] == "that" and not measuring:
                if len([t for t in tokens[j + 1:] if t not in STOPWORDS]) >= 2:
                    drop.update({i, j})
                break
    return drop


def dependence_reason(query: str) -> str | None:
    """The word that scopes this question to its conversation, or None.

    Returned rather than a bare boolean so a trace can say WHY a question was
    treated as a follow-up instead of leaving the reader to guess.
    """
    low = query.strip().lower()
    for opener in _FOLLOW_UP_OPENERS:
        if low.startswith(opener):
            return opener.strip()
    tokens = re.findall(r"[a-z']+", low)
    filler = _filler_it(tokens)
    for i, word in enumerate(tokens):
        if i in filler:
            continue
        if word in _POSITIONAL:
            prev = tokens[i - 1] if i else ""
            nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
            if prev in _DETERMINERS or nxt in ("one", "ones"):
                return word
            continue
        if word in _DEICTIC:
            return word
    return None


def is_context_dependent(query: str) -> bool:
    """Does answering this question require the conversation before it?

    'What is 847 * 23?' does not. 'And what about the second one?' does. The
    distinction decides whether the cache entry is scoped to a conversation.
    """
    return dependence_reason(query) is not None


def chain_hash(history: tuple, depth: int, query: str = "") -> str:
    """MeanCache-style context chain, applied only where it is needed.

    Without a chain, 'and what about the second one?' in conversation A can be
    served from conversation B. With a chain on EVERY query the hit rate inside
    a conversation collapses to exactly zero — which it did: the chain is a hash
    of the last `depth` turns, so it changes on every turn, and asking an
    identical question twice in one conversation reported "cache miss (no
    candidates)" rather than a low similarity. The entry was never a candidate,
    because it was filed under a chain that no longer existed.

    That is the wrong trade for a self-contained question. '847 * 23' has the
    same answer whatever preceded it, so scoping it to a conversation buys no
    safety and costs every hit.

    So the chain applies only when the query is context-dependent. The test is
    conservative in the safe direction: a self-contained question mistaken for a
    dependent one merely misses a cache hit, while the reverse serves a wrong
    answer (ADR-039).
    """
    if depth <= 0 or not history:
        return "root"
    if query and not is_context_dependent(query):
        return "root"
    parents = [t.content for t in history[-depth:]]
    return hashlib.blake2b("␟".join(parents).encode(), digest_size=8).hexdigest()


@dataclass(slots=True)
class CacheEntry:
    entry_id: str
    key: str
    query: str
    response: str
    created_at: float
    chain: str
    model_id: str
    invariants: Invariants
    volatile: bool = False
    hits: int = 0


@dataclass(slots=True)
class VerifierResult:
    passed: bool
    jaccard: float
    entity_agree: bool
    number_agree: bool
    negation_agree: bool
    modifier_agree: bool = True
    question_agree: bool = True

    def as_dict(self) -> dict[str, float]:
        return {
            "jaccard": round(self.jaccard, 4),
            "entity_agree": float(self.entity_agree),
            "number_agree": float(self.number_agree),
            "negation_agree": float(self.negation_agree),
            "modifier_agree": float(self.modifier_agree),
            "question_agree": float(self.question_agree),
        }

    def failure(self) -> str:
        if not self.negation_agree:
            return "negation mismatch"
        if not self.modifier_agree:
            return "operative modifier mismatch"
        if not self.number_agree:
            return "number mismatch"
        if not self.entity_agree:
            return "entity mismatch"
        if not self.question_agree:
            return "different kind of question"
        return f"lexical overlap {self.jaccard:.2f} below floor"


#: One prototype per kind the opener vocabulary already distinguishes, written
#: as nouns because a question's subject is a noun phrase. Sentences describing
#: each kind were measured first and are worse: 11 of 15 against 13 of 15, and
#: they put a headcount, a manager's name and a travel policy all in "where".
#: Keys are `question_type`'s OWN vocabulary, and must stay that way: a
#: specific opener keeps its surface kind while a "what" question gets a
#: resolved one, so a resolver that answers "duration" where the surface says
#: "quantity" can never agree with anything. Durations fold into quantity
#: because "how long" already does.
ANSWER_KINDS = {
    "quantity": "amount number budget cost count total period duration length",
    "when": "date day time schedule deadline",
    "who": "person name manager owner",
    "where": "place location address building",
    "why": "reason cause purpose",
    "how": "process procedure method policy steps",
}
_KIND_VECTORS: dict[str, "np.ndarray"] = {}

#: What ends the head of a complement and begins what the question is
#: about. Deliberately short: "in", "at", "to" and "on" turn up inside
#: heads once hyphens are tokenised -- "sign-in process" becomes "sign in
#: process" and cutting at "in" leaves "sign", which resolves to why.
_HEAD_TAIL_RE = re.compile(r"\b(?:of|for|from|about)\b")
_HEAD_LEAD_RE = re.compile(
    r"^(?:is|are|was|were|the|a|an|my|our|your|their|its)\b\s*")


def complement_head(subject: str) -> str:
    """The noun phrase that states the kind, before what it is about.

    "the headcount of the Porto office" is a headcount; "the booking
    process for the roof terrace" is a process. Resolving on all of it
    lets the office and the terrace outvote the word that carries the
    kind, and both resolved to `where`.
    """
    s = subject
    for _ in range(3):
        shorter = _HEAD_LEAD_RE.sub("", s)
        if shorter == s:
            break
        s = shorter
    cut = _HEAD_TAIL_RE.search(s)
    if cut:
        s = s[:cut.start()]
    return s.strip() or subject


def resolved_question_kind(query: str, embed) -> str | None:
    """The kind of answer this question asks for, rather than the word used.

    A specific opener states its own kind and keeps it. "What" states nothing --
    it is the generic interrogative, which is why comparing surface openers
    refuses "what is the annual travel budget" against "how much can they
    spend". Its subject is what fixes the kind, so that is what gets compared.

    Falls back to the surface kind if the encoder is unavailable: a missed
    reuse, never a wrong one.
    """
    kind = question_type(query)
    if kind != "what" or embed is None:
        return kind
    subject = complement_head(cache_key_text(query)).strip()
    if not subject:
        return kind
    try:
        if not _KIND_VECTORS:
            vectors = embed(list(ANSWER_KINDS.values()))
            for name, vec in zip(ANSWER_KINDS, vectors):
                _KIND_VECTORS[name] = vec / (float(np.linalg.norm(vec)) or 1.0)
        v = embed([subject])[0]
        v = v / (float(np.linalg.norm(v)) or 1.0)
    except Exception:
        return kind
    return max(_KIND_VECTORS, key=lambda k: float(np.dot(v, _KIND_VECTORS[k])))


def soft_overlap(
    ca: frozenset[str], cb: frozenset[str], embed, word_tau: float
) -> float:
    """Jaccard where two words count as the same if the encoder says so.

    Exact matches are taken first and cost nothing. What is left is aligned
    greedily, best pair first, each word spent once -- so one word on one side
    cannot pay for three on the other, which is how a soft measure quietly
    becomes a meaningless one.

    `embed` is the Derived batch call, which memoises per word, so a word seen
    on an earlier probe costs a dictionary lookup rather than a forward pass.
    If it is unavailable or fails this falls back to exact counting, which is
    the safe direction: a missed hit, not a wrong answer.
    """
    exact = ca & cb
    ra, rb = sorted(ca - exact), sorted(cb - exact)
    matched = len(exact)
    if ra and rb and embed is not None:
        va = vb = None
        try:
            both = embed(ra + rb)
            unit = [v / (float(np.linalg.norm(v)) or 1.0) for v in both]
            va = dict(zip(ra, unit[:len(ra)]))
            vb = dict(zip(rb, unit[len(ra):]))
        except Exception:
            va = vb = None
        if va and vb:
            pairs = sorted(
                ((float(np.dot(va[x], vb[y])), x, y) for x in ra for y in rb),
                reverse=True)
            used_a: set[str] = set()
            used_b: set[str] = set()
            for score, x, y in pairs:
                if score < word_tau or x in used_a or y in used_b:
                    continue
                used_a.add(x)
                used_b.add(y)
                matched += 1
    union = len(ca) + len(cb) - matched
    return matched / union if union else 0.0


def verify_match(
    a: Invariants, b: Invariants, qa: str, qb: str, jaccard_min: float,
    embed=None, word_tau: float = 0.0, resolve_kind: bool = False,
    oppose_only: bool = False
) -> VerifierResult:
    """Settle a borderline match without a second neural pass.

    Four agreement checks plus a lexical floor. The modifier check was added
    after measurement: with only number, entity and negation checks, 78% of
    modifier-swapped adversarial pairs produced false hits. "minimum" against
    "maximum" changes no number, no entity and no negation particle, and leaves
    lexical overlap high — nothing else in the verifier could see it.

    Negation additionally covers morphological forms ("possible"/"impossible"),
    which a particle-based check misses because no separate negation token
    exists.
    """
    # Overlap is taken over CONTENT words of the key text, not raw tokens.
    # Raw tokens counted "what is" and "define" as subject matter, so
    # "What is machine learning?" against "Define machine learning" scored
    # 0.40 and was refused for a difference that carries no meaning.
    ca = content_words(cache_key_text(qa))
    cb = content_words(cache_key_text(qb))
    jac = (soft_overlap(ca, cb, embed, word_tau) if embed is not None
           else (len(ca & cb) / len(ca | cb) if (ca or cb) else 0.0))

    neg_a = a.negations | morphological_negations(qa, qb)
    neg_b = b.negations | morphological_negations(qb, qa)
    mod_a, mod_b = operative_modifiers(qa), operative_modifiers(qb)

    negation_agree = neg_a == neg_b
    # Presence on one side is wording; an inversion needs both sides.
    modifier_agree = (not modifiers_oppose(qa, qb) if oppose_only
                      else mod_a == mod_b)
    number_agree = a.numbers == b.numbers
    # Compared by the words the spans hold, not the spans: the regex takes
    # runs, so two adjacent names are one span and the same two apart are
    # two (ADR-058).
    entity_agree = entity_words(a.entities) == entity_words(b.entities)
    question_agree = question_types_agree(
        qa, qb, embed if resolve_kind else None)

    return VerifierResult(
        passed=(
            negation_agree
            and modifier_agree
            and number_agree
            and entity_agree
            and question_agree
            and jac >= jaccard_min
        ),
        jaccard=jac,
        entity_agree=entity_agree,
        number_agree=number_agree,
        negation_agree=negation_agree,
        modifier_agree=modifier_agree,
        question_agree=question_agree,
    )


@dataclass(slots=True)
class CacheStats:
    consulted: int = 0
    exact_hits: int = 0
    semantic_hits: int = 0
    verified_hits: int = 0
    verify_rejections: int = 0
    misses: int = 0
    expired: int = 0
    stores: int = 0
    uncacheable: int = 0  # canonical form carried no information to key on
    evicted: int = 0  # dropped to stay under max_entries


class SemanticCache:
    """Cross-request state, so the stage holds it rather than owning it."""

    def __init__(
        self,
        ttl_seconds: int = 86_400,
        embedder=None,
        index=None,
        max_entries: int = 10_000,
    ) -> None:
        # Insertion-ordered dicts double as the LRU: a hit moves its key to the
        # end, so the oldest live key is always the first one.
        self._exact: dict[str, CacheEntry] = {}
        self._entries: dict[str, CacheEntry] = {}
        self._max_entries = max_entries
        self._embedder = embedder
        # `index` is injectable so ADR-004's claim about approximate search can
        # be measured rather than asserted. Default stays exact.
        self._index = index if index is not None else (
            ExactIndex(embedder.dim) if embedder is not None else None
        )
        self._ttl = ttl_seconds
        self._extractor = RegexInvariantExtractor()
        self._inv_memo: dict[str, Invariants] = {}
        self.stats = CacheStats()

    def attach_embedder(self, embedder, full_question: bool = False) -> None:
        """Adopt an embedder if constructed without one.

        A cache built without an embedder silently has no vector index, so
        every semantic lookup returns nothing and the tier looks like it is
        working while measuring zero. The Pipeline calls this so no caller can
        half-configure the cache by construction order.
        """
        if self._embedder is not None or embedder is None:
            return
        self._embedder = embedder
        self._index = ExactIndex(embedder.dim)
        # Backfill anything stored before the embedder arrived.
        if self._entries:
            prepare = embed_text if full_question else cache_key_text
            queries = [prepare(e.query) for e in self._entries.values()]
            for entry, vec in zip(self._entries.values(), embedder.embed(queries)):
                self._index.add(vec, entry.entry_id)

    @property
    def has_embedder(self) -> bool:
        return self._embedder is not None

    # -- keys ---------------------------------------------------------------

    @staticmethod
    def make_key(query: str, chain: str, model_id: str) -> str:
        payload = f"{canonicalise(query)}␟{chain}␟{model_id}"
        return hashlib.blake2b(payload.encode(), digest_size=16).hexdigest()

    def invariants_of(self, query: str) -> Invariants:
        hit = self._inv_memo.get(query)
        if hit is None:
            hit = self._extractor.extract(query)
            if len(self._inv_memo) < 8192:
                self._inv_memo[query] = hit
        return hit

    # -- tier 0 -------------------------------------------------------------

    def lookup(self, key: str, now: float | None = None, query: str | None = None) -> CacheEntry | None:
        self.stats.consulted += 1
        if query is not None and not is_cacheable(query):
            self.stats.uncacheable += 1
            self.stats.misses += 1
            return None
        entry = self._exact.get(key)
        if entry is None:
            self.stats.misses += 1
            return None
        if self._expired(entry, now):
            self.stats.expired += 1
            self.stats.misses += 1
            return None
        entry.hits += 1
        self.stats.exact_hits += 1
        self._touch(key)
        return entry

    # -- tier 1 -------------------------------------------------------------

    def search(
        self, vec: np.ndarray, chain: str, model_id: str, k: int, now: float | None = None
    ) -> list[tuple[CacheEntry, float]]:
        """Top-k live candidates whose context chain and model match.

        Filtering after search rather than maintaining one index per (chain,
        model) keeps a single exact index; at these cache sizes the extra rows
        scanned cost far less than the bookkeeping would.
        """
        if self._index is None or self._index.size() == 0:
            return []
        out: list[tuple[CacheEntry, float]] = []
        for entry_id, score in self._index.search(vec, k * 4):
            entry = self._entries.get(entry_id)
            if entry is None or entry.chain != chain or entry.model_id != model_id:
                continue
            if self._expired(entry, now):
                continue
            out.append((entry, score))
            if len(out) >= k:
                break
        return out

    # -- writes -------------------------------------------------------------

    def store(
        self,
        key: str,
        query: str,
        response: str,
        *,
        chain: str = "root",
        model_id: str = "",
        vec: np.ndarray | None = None,
        now: float | None = None,
    ) -> CacheEntry | None:
        if not is_cacheable(query):
            # Nothing to key on. Storing it would make this entry the answer to
            # every future degenerate query.
            self.stats.uncacheable += 1
            return None
        self.stats.stores += 1
        entry = CacheEntry(
            entry_id=key,
            key=key,
            query=query,
            response=response,
            created_at=now if now is not None else time.time(),
            chain=chain,
            model_id=model_id,
            invariants=self.invariants_of(query),
            volatile=bool(_VOLATILE_RE.search(query)),
        )
        self._exact[key] = entry
        self._entries[key] = entry
        if vec is not None and self._index is not None:
            self._index.add(vec, key)
        self._evict_if_needed()
        return entry

    # -- helpers ------------------------------------------------------------

    def touch(self, key: str) -> None:
        """Mark a key most-recently-used.

        Public because SEMANTIC hits are served from `search()`, which the stage
        drives. Without this, only exact hits would refresh recency and a
        heavily-used paraphrase entry could be evicted while a stale
        exact-matched one survived — LRU that does not see half its traffic.
        """
        self._touch(key)

    def _touch(self, key: str) -> None:
        """Move a key to the end of insertion order (most-recently-used)."""
        for store in (self._exact, self._entries):
            entry = store.pop(key, None)
            if entry is not None:
                store[key] = entry

    def _evict_if_needed(self) -> None:
        """Drop least-recently-used entries down to the cap.

        Evicting from the vector index too is the part that is easy to forget:
        an orphaned vector would keep scoring in `search()` and return an
        entry_id that no longer resolves, which reads as a silent cache miss
        while still costing the similarity computation.
        """
        while len(self._exact) > self._max_entries:
            oldest = next(iter(self._exact))
            self._exact.pop(oldest, None)
            self._entries.pop(oldest, None)
            if self._index is not None:
                self._index.remove(oldest)
            self.stats.evicted += 1

    def _expired(self, entry: CacheEntry, now: float | None) -> bool:
        """Expired entries are filtered at retrieval, not deleted: how often the
        TTL fires is itself a reportable number."""
        if not entry.volatile:
            return False
        now = now if now is not None else time.time()
        return (now - entry.created_at) > self._ttl

    def entry(self, entry_id: str) -> CacheEntry | None:
        """One entry by id, for a surface that wants to show what matched."""
        return self._entries.get(entry_id)

    def entries(self) -> tuple[CacheEntry, ...]:
        """Everything held, oldest first. Display and inspection only."""
        return tuple(self._entries.values())

    def size(self) -> int:
        return len(self._exact)

    def clear(self) -> None:
        self._exact.clear()
        self._entries.clear()
        if self._index is not None:
            self._index.clear()
        self.stats = CacheStats()


def documents_scope(chain: str, documents: tuple) -> str:
    """Fold the supplied documents into the chain.

    "What is the notice period?" has a different answer for every contract it
    is asked about, and the question text alone cannot tell them apart. A
    request that carries documents is therefore only ever matched against
    requests that carried the same documents. Keyed on the documents as
    supplied, so compressing them later in the pipeline cannot change the key.
    """
    if not documents:
        return chain
    material = "\u241e".join(f"{d.doc_id}\u241f{d.title}\u241f{d.content}" for d in documents)
    digest = hashlib.blake2b(material.encode(), digest_size=8).hexdigest()
    return f"{chain}:docs:{digest}"


class CacheLookupStage:
    module_id = "M2"
    name = "m2_cache"
    reads = frozenset({"query", "history", "original_documents"})
    writes = frozenset()

    def __init__(self, cache: SemanticCache, *, probe_only: bool = False,
                 name: str | None = None) -> None:
        self.cache = cache
        # A probe records what the cache WOULD have done without acting on it.
        # cache_lookup_on="BOTH" runs a probe before compression and the real
        # lookup after, so a single request yields a paired observation of the
        # same cache under both orderings — a far stronger design for Gap 3
        # than comparing two independent runs.
        self.probe_only = probe_only
        if name is not None:
            self.name = name

    def applies_to(self, ctx: RequestContext, cfg: ParsimonyConfig) -> bool:
        return cfg.enables("M2") and (cfg.cache.exact_tier or cfg.cache.semantic_tier)

    def chain_for(self, ctx: RequestContext, cfg: ParsimonyConfig) -> str:
        return documents_scope(chain_hash(ctx.history, cfg.cache.chain_depth, ctx.query),
                               ctx.original_documents)

    def key_for(self, ctx: RequestContext, cfg: ParsimonyConfig) -> str:
        return SemanticCache.make_key(ctx.query, self.chain_for(ctx, cfg), cfg.model.name)

    def propose(self, ctx: RequestContext, cfg: ParsimonyConfig) -> Proposal:
        chain = self.chain_for(ctx, cfg)
        key = SemanticCache.make_key(ctx.query, chain, cfg.model.name)
        # Carried on every outcome so a surface can explain a miss caused by
        # scoping rather than by similarity: those look identical otherwise.
        scope = dependence_reason(ctx.query)

        if not is_cacheable(ctx.query):
            return NoOp(
                "not_applicable",
                "query carries no information to key on",
                {"zone": "uncacheable", "top_k": ()},
            )

        if cfg.cache.exact_tier:
            entry = self.cache.lookup(key, query=ctx.query)
            if entry is not None:
                evidence = {"zone": "accept", "tier": "exact", "cache_key": key[:12],
                            "entry_hits": entry.hits, "top_k": (),
                            "probe_only": self.probe_only, "best_entry": entry.entry_id}
                if self.probe_only:
                    return NoOp("no_yield", "probe: exact hit (not acted on)", evidence)
                return ShortCircuit(
                    response=entry.response,
                    served_by=RouteTier.CACHE_EXACT,
                    rationale="exact-hash cache hit",
                    evidence=evidence,
                )

        if not (cfg.cache.semantic_tier and ctx.derived is not None
                and getattr(ctx.derived, "has_embedder", False)):
            return NoOp("no_yield", "cache miss",
                        {"zone": "miss", "top_k": (),
                         "scoped": scope is not None, "scope_reason": scope})

        prepare = embed_text if cfg.cache.embed_full_question else cache_key_text
        vec = ctx.derived.embed_one(prepare(ctx.query))
        candidates = self.cache.search(vec, chain, cfg.model.name, cfg.cache.top_k)
        top_k = tuple((e.entry_id[:12], round(s, 4)) for e, s in candidates)

        if not candidates:
            # A miss because nothing was COMPARABLE reads identically to a miss
            # because nothing was similar, and they need different fixes. Say
            # which, and name the word responsible.
            why = "cache miss (no candidates)" if scope is None else (
                f"cache miss: treated as a follow-up because of the word "
                f"{scope!r}, so only answers from this conversation could match"
            )
            return NoOp("no_yield", why,
                        {"zone": "miss", "top_k": top_k,
                         "scoped": scope is not None, "scope_reason": scope})

        best, score = candidates[0]
        query_inv = self.cache.invariants_of(ctx.query)

        if score >= cfg.cache.tau_hi and not cfg.cache.verify_always:
            # The accept zone: a score high enough to skip verification. Safe
            # only while the encoder never scores an adversarial pair this
            # high, which is a property of a WEAK encoder -- MiniLM puts "is X
            # safe" against "is X NOT safe" at 0.996 and walks it straight
            # through here (ADR-041). `verify_always` (the default) leaves this
            # branch unreachable and is what lets a better encoder be used at
            # all; the branch stays so the unsafe design remains measurable.
            #
            # Even then, a perfect score does not survive a different KIND of
            # question. "What is Java?" and "Where is Java?" both reduce to
            # "java" and score 1.000, so without this the cache answers the
            # wrong question with full confidence.
            if not question_types_agree(ctx.query, best.query):
                self.cache.stats.verify_rejections += 1
                return NoOp(
                    "no_yield",
                    f"different kind of question at cosine {score:.3f}: stored asks "
                    f"{question_type(best.query)}, this asks {question_type(ctx.query)}",
                    {"zone": "accept", "score": round(score, 4), "top_k": top_k,
                     "rejected": True, "best_entry": best.entry_id,
                     "question_type": question_type(ctx.query),
                     "stored_question_type": question_type(best.query),
                     "type_agree": False},
                )
            evidence = {"zone": "accept", "tier": "semantic", "score": round(score, 4),
                        "top_k": top_k, "probe_only": self.probe_only,
                        "best_entry": best.entry_id,
                        "question_type": question_type(ctx.query),
                        "stored_question_type": question_type(best.query),
                        "type_agree": True}
            if self.probe_only:
                return NoOp("no_yield", "probe: semantic hit (not acted on)", evidence)
            self.cache.stats.semantic_hits += 1
            best.hits += 1
            self.cache.touch(best.entry_id)
            return ShortCircuit(
                response=best.response,
                served_by=RouteTier.CACHE_SEMANTIC,
                rationale=f"semantic hit, cosine {score:.3f} >= tau_hi {cfg.cache.tau_hi}",
                evidence=evidence,
            )

        if score >= cfg.cache.tau_lo:
            soft = (cfg.cache.soft_overlap and ctx.derived is not None
                    and getattr(ctx.derived, "has_embedder", False))
            result = verify_match(
                query_inv, best.invariants, ctx.query, best.query,
                cfg.cache.soft_jaccard_min if soft else cfg.cache.jaccard_min,
                embed=(ctx.derived.embed if soft else None),
                word_tau=cfg.cache.word_tau,
                resolve_kind=cfg.cache.resolve_question_kind,
                oppose_only=cfg.cache.modifier_opposition_only)
            if result.passed:
                evidence = {"zone": "verify", "tier": "semantic", "score": round(score, 4),
                            "verifier": result.as_dict(), "top_k": top_k,
                            "probe_only": self.probe_only, "best_entry": best.entry_id,
                            "question_type": question_type(ctx.query),
                            "stored_question_type": question_type(best.query),
                            # The claim worth showing a reader is that these two
                            # are one question in different words. An entry id
                            # does not make it; the question does.
                            "matched_query": best.query}
                if self.probe_only:
                    return NoOp("no_yield", "probe: verified hit (not acted on)", evidence)
                self.cache.stats.semantic_hits += 1
                self.cache.stats.verified_hits += 1
                best.hits += 1
                self.cache.touch(best.entry_id)
                return ShortCircuit(
                    response=best.response,
                    served_by=RouteTier.CACHE_SEMANTIC,
                    rationale=f"verified hit, cosine {score:.3f} in verify zone",
                    evidence=evidence,
                )
            self.cache.stats.verify_rejections += 1
            return NoOp(
                "no_yield",
                f"verify-zone rejection at cosine {score:.3f}: {result.failure()}",
                {"zone": "verify", "score": round(score, 4), "verifier": result.as_dict(),
                 "top_k": top_k, "rejected": True, "best_entry": best.entry_id,
                 "question_type": question_type(ctx.query),
                 "stored_question_type": question_type(best.query)},
            )

        return NoOp(
            "no_yield",
            f"below tau_lo (best cosine {score:.3f})",
            {"zone": "reject", "score": round(score, 4), "top_k": top_k,
             "best_entry": best.entry_id},
        )

    def remember(self, ctx: RequestContext, cfg: ParsimonyConfig, response: str) -> None:
        """Write path, called by the orchestrator after generation."""
        if self.probe_only:
            return  # a probe observes; the authoritative stage owns the write
        chain = self.chain_for(ctx, cfg)
        key = SemanticCache.make_key(ctx.query, chain, cfg.model.name)
        vec = None
        if (cfg.cache.semantic_tier and ctx.derived is not None
                and getattr(ctx.derived, "has_embedder", False)):
            # Stored under the SAME normalisation the lookup uses. Storing the
            # raw text and looking up the normalised form was the asymmetry
            # that made a polite question unreachable by its plain form.
            prepare = (embed_text if cfg.cache.embed_full_question
                       else cache_key_text)
            vec = ctx.derived.embed_one(prepare(ctx.query))
        self.cache.store(key, ctx.query, response, chain=chain,
                         model_id=cfg.model.name, vec=vec)
