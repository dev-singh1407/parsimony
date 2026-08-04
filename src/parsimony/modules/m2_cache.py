"""M2 — Two-Tier Semantic Cache.

Tier 0 (exact hash) is live. Tier 1 (embedding + three-zone verifier) lands in
Sprint 2 with sentence-transformers; its policy is already expressed here so the
threshold fields in ParsimonyConfig are real rather than aspirational.

The cache key includes model_id. Non-obvious and important: report 4.6 re-runs
the winning configuration on three models, and without model_id in the key a
Llama-generated answer would be served during the Qwen run, silently corrupting
the generalisation study.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field

from parsimony.core.config import ParsimonyConfig
from parsimony.core.proposals import NoOp, Proposal, ShortCircuit
from parsimony.core.types import RequestContext, RouteTier

_WS_RE = re.compile(r"\s+")
_VOLATILE_RE = re.compile(
    r"\b(today|now|current|currently|latest|recent|this (week|month|year)|"
    r"price|rate|stock|weather|news)\b",
    re.IGNORECASE,
)


def canonicalise(text: str) -> str:
    return _WS_RE.sub(" ", text.strip().lower()).rstrip("?.! ")


def chain_hash(history: tuple, depth: int) -> str:
    """MeanCache-style context chain.

    Without it, 'and what about the second one?' in conversation A can be served
    from conversation B. With the full chain every follow-up key becomes unique
    and the hit rate collapses, which is why depth is bounded and ablatable.
    """
    if depth <= 0 or not history:
        return "root"
    parents = [t.content for t in history[-depth:]]
    return hashlib.blake2b("␟".join(parents).encode(), digest_size=8).hexdigest()


@dataclass(slots=True)
class CacheEntry:
    key: str
    query: str
    response: str
    created_at: float
    volatile: bool = False
    hits: int = 0


@dataclass(slots=True)
class CacheStats:
    consulted: int = 0
    exact_hits: int = 0
    semantic_hits: int = 0
    misses: int = 0
    expired: int = 0
    stores: int = 0
    top_k: tuple = field(default_factory=tuple)


class SemanticCache:
    """Cross-request state, so it is a service the stage holds, not per-request."""

    def __init__(self, ttl_seconds: int = 86_400) -> None:
        self._exact: dict[str, CacheEntry] = {}
        self._ttl = ttl_seconds
        self.stats = CacheStats()

    @staticmethod
    def make_key(query: str, chain: str, model_id: str) -> str:
        payload = f"{canonicalise(query)}␟{chain}␟{model_id}"
        return hashlib.blake2b(payload.encode(), digest_size=16).hexdigest()

    def lookup(self, key: str, now: float | None = None) -> CacheEntry | None:
        self.stats.consulted += 1
        entry = self._exact.get(key)
        if entry is None:
            self.stats.misses += 1
            return None
        now = now if now is not None else time.time()
        # Expired entries stay in the store: how often TTL fires is a reportable
        # number, so they are filtered at retrieval rather than deleted.
        if entry.volatile and (now - entry.created_at) > self._ttl:
            self.stats.expired += 1
            self.stats.misses += 1
            return None
        entry.hits += 1
        self.stats.exact_hits += 1
        return entry

    def store(self, key: str, query: str, response: str, now: float | None = None) -> None:
        self.stats.stores += 1
        self._exact[key] = CacheEntry(
            key=key,
            query=query,
            response=response,
            created_at=now if now is not None else time.time(),
            volatile=bool(_VOLATILE_RE.search(query)),
        )

    def size(self) -> int:
        return len(self._exact)

    def clear(self) -> None:
        self._exact.clear()
        self.stats = CacheStats()


class CacheLookupStage:
    module_id = "M2"
    name = "m2_cache"
    reads = frozenset({"query", "history"})
    writes = frozenset()

    def __init__(self, cache: SemanticCache) -> None:
        self.cache = cache

    def applies_to(self, ctx: RequestContext, cfg: ParsimonyConfig) -> bool:
        return cfg.enables("M2") and cfg.cache.exact_tier

    def key_for(self, ctx: RequestContext, cfg: ParsimonyConfig) -> str:
        return SemanticCache.make_key(
            ctx.query,
            chain_hash(ctx.history, cfg.cache.chain_depth),
            cfg.model.name,
        )

    def propose(self, ctx: RequestContext, cfg: ParsimonyConfig) -> Proposal:
        entry = self.cache.lookup(self.key_for(ctx, cfg))
        if entry is None:
            return NoOp("no_yield", "cache miss")
        return ShortCircuit(
            response=entry.response,
            served_by=RouteTier.CACHE_EXACT,
            rationale="exact-hash cache hit",
            evidence={"cache_key": entry.key[:12], "entry_hits": entry.hits},
        )
