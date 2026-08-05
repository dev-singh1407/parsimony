"""M2 — Two-Tier Semantic Cache.

Tier 0 is an exact hash; tier 1 is cosine over an exact vector index. A single
similarity threshold is replaced by a three-zone policy: accept, reject, or
*verify*. Borderline matches are settled by cheap lexical and invariant
agreement rather than by the embedding alone.

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
from parsimony.infra.embedding import ExactIndex
from parsimony.infra.nlp import (
    RegexInvariantExtractor,
    morphological_negations,
    operative_modifiers,
    shingles,
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


def chain_hash(history: tuple, depth: int) -> str:
    """MeanCache-style context chain.

    Without it, 'and what about the second one?' in conversation A can be served
    from conversation B. With an unbounded chain every follow-up key becomes
    unique and the hit rate collapses to zero, which is why depth is bounded
    and ablatable.
    """
    if depth <= 0 or not history:
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

    def as_dict(self) -> dict[str, float]:
        return {
            "jaccard": round(self.jaccard, 4),
            "entity_agree": float(self.entity_agree),
            "number_agree": float(self.number_agree),
            "negation_agree": float(self.negation_agree),
            "modifier_agree": float(self.modifier_agree),
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
        return f"lexical overlap {self.jaccard:.2f} below floor"


def verify_match(
    a: Invariants, b: Invariants, qa: str, qb: str, jaccard_min: float
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
    sa, sb = shingles(qa), shingles(qb)
    jac = len(sa & sb) / len(sa | sb) if (sa or sb) else 0.0

    neg_a = a.negations | morphological_negations(qa, qb)
    neg_b = b.negations | morphological_negations(qb, qa)
    mod_a, mod_b = operative_modifiers(qa), operative_modifiers(qb)

    negation_agree = neg_a == neg_b
    modifier_agree = mod_a == mod_b
    number_agree = a.numbers == b.numbers
    entity_agree = a.entities == b.entities

    return VerifierResult(
        passed=(
            negation_agree
            and modifier_agree
            and number_agree
            and entity_agree
            and jac >= jaccard_min
        ),
        jaccard=jac,
        entity_agree=entity_agree,
        number_agree=number_agree,
        negation_agree=negation_agree,
        modifier_agree=modifier_agree,
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


class SemanticCache:
    """Cross-request state, so the stage holds it rather than owning it."""

    def __init__(self, ttl_seconds: int = 86_400, embedder=None, index=None) -> None:
        self._exact: dict[str, CacheEntry] = {}
        self._entries: dict[str, CacheEntry] = {}
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

    def attach_embedder(self, embedder) -> None:
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
            queries = [e.query for e in self._entries.values()]
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
        return entry

    # -- helpers ------------------------------------------------------------

    def _expired(self, entry: CacheEntry, now: float | None) -> bool:
        """Expired entries are filtered at retrieval, not deleted: how often the
        TTL fires is itself a reportable number."""
        if not entry.volatile:
            return False
        now = now if now is not None else time.time()
        return (now - entry.created_at) > self._ttl

    def size(self) -> int:
        return len(self._exact)

    def clear(self) -> None:
        self._exact.clear()
        self._entries.clear()
        if self._index is not None:
            self._index.clear()
        self.stats = CacheStats()


class CacheLookupStage:
    module_id = "M2"
    name = "m2_cache"
    reads = frozenset({"query", "history"})
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
        return chain_hash(ctx.history, cfg.cache.chain_depth)

    def key_for(self, ctx: RequestContext, cfg: ParsimonyConfig) -> str:
        return SemanticCache.make_key(ctx.query, self.chain_for(ctx, cfg), cfg.model.name)

    def propose(self, ctx: RequestContext, cfg: ParsimonyConfig) -> Proposal:
        chain = self.chain_for(ctx, cfg)
        key = SemanticCache.make_key(ctx.query, chain, cfg.model.name)

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
                            "probe_only": self.probe_only}
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
            return NoOp("no_yield", "cache miss", {"zone": "miss", "top_k": ()})

        vec = ctx.derived.embed_one(ctx.query)
        candidates = self.cache.search(vec, chain, cfg.model.name, cfg.cache.top_k)
        top_k = tuple((e.entry_id[:12], round(s, 4)) for e, s in candidates)

        if not candidates:
            return NoOp("no_yield", "cache miss (no candidates)", {"zone": "miss", "top_k": top_k})

        best, score = candidates[0]
        query_inv = self.cache.invariants_of(ctx.query)

        if score >= cfg.cache.tau_hi:
            evidence = {"zone": "accept", "tier": "semantic", "score": round(score, 4),
                        "top_k": top_k, "probe_only": self.probe_only}
            if self.probe_only:
                return NoOp("no_yield", "probe: semantic hit (not acted on)", evidence)
            self.cache.stats.semantic_hits += 1
            best.hits += 1
            return ShortCircuit(
                response=best.response,
                served_by=RouteTier.CACHE_SEMANTIC,
                rationale=f"semantic hit, cosine {score:.3f} >= tau_hi {cfg.cache.tau_hi}",
                evidence=evidence,
            )

        if score >= cfg.cache.tau_lo:
            result = verify_match(query_inv, best.invariants, ctx.query, best.query,
                                  cfg.cache.jaccard_min)
            if result.passed:
                evidence = {"zone": "verify", "tier": "semantic", "score": round(score, 4),
                            "verifier": result.as_dict(), "top_k": top_k,
                            "probe_only": self.probe_only}
                if self.probe_only:
                    return NoOp("no_yield", "probe: verified hit (not acted on)", evidence)
                self.cache.stats.semantic_hits += 1
                self.cache.stats.verified_hits += 1
                best.hits += 1
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
                 "top_k": top_k, "rejected": True},
            )

        return NoOp(
            "no_yield",
            f"below tau_lo (best cosine {score:.3f})",
            {"zone": "reject", "score": round(score, 4), "top_k": top_k},
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
            vec = ctx.derived.embed_one(ctx.query)
        self.cache.store(key, ctx.query, response, chain=chain,
                         model_id=cfg.model.name, vec=vec)
