"""Bounded memory (ADR-031).

Probed under sustained load, nothing evicted: 400 distinct queries produced 400
cache entries, 400 tracked conversations and 415 blobs. The report targets an
8 GB consumer laptop (4.7), and the cache is the one component designed to
accumulate — so unbounded was a memory leak in exactly the wrong place.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from parsimony.core.config import full_stack
from parsimony.infra.embedding import HashingEmbedder
from parsimony.modules.m2_cache import SemanticCache
from parsimony.pipeline.orchestrator import Pipeline


def _cache(max_entries: int, with_embedder: bool = True) -> SemanticCache:
    embedder = HashingEmbedder() if with_embedder else None
    return SemanticCache(embedder=embedder, max_entries=max_entries)


def _store(cache: SemanticCache, query: str) -> str:
    key = SemanticCache.make_key(query, "root", "m")
    vec = cache._embedder.embed([query])[0] if cache._embedder else None
    cache.store(key, query, f"answer to {query}", chain="root", model_id="m", vec=vec)
    return key


class TestEviction:
    def test_cache_stays_under_the_cap(self):
        cache = _cache(10)
        for i in range(50):
            _store(cache, f"question number {i}")
        assert cache.size() == 10
        assert cache.stats.evicted == 40

    def test_the_vector_index_is_evicted_too(self):
        """The easy thing to forget. An orphaned vector keeps scoring in
        search() and returns an entry_id that no longer resolves — a silent miss
        that still costs the similarity computation."""
        cache = _cache(10)
        for i in range(50):
            _store(cache, f"question number {i}")
        assert cache._index.size() == cache.size() == 10

    def test_oldest_entries_go_first(self):
        cache = _cache(3)
        keys = [_store(cache, f"query {i}") for i in range(5)]
        assert cache.lookup(keys[0], query="query 0") is None
        assert cache.lookup(keys[4], query="query 4") is not None

    def test_an_exact_hit_refreshes_recency(self):
        """LRU, not FIFO: a key that is still being used must survive."""
        cache = _cache(3)
        keys = [_store(cache, f"query {i}") for i in range(3)]
        cache.lookup(keys[0], query="query 0")   # touch the oldest
        _store(cache, "query 3")                  # forces one eviction
        assert cache.lookup(keys[0], query="query 0") is not None
        assert cache.lookup(keys[1], query="query 1") is None

    def test_touch_is_public_for_semantic_hits(self):
        """Semantic hits are served from search(), which the stage drives. If
        only exact hits refreshed recency, LRU would not see half its traffic."""
        cache = _cache(3)
        keys = [_store(cache, f"query {i}") for i in range(3)]
        cache.touch(keys[0])
        _store(cache, "query 3")
        assert cache.lookup(keys[0], query="query 0") is not None

    def test_touching_an_absent_key_is_harmless(self):
        cache = _cache(3)
        cache.touch("no-such-key")  # must not raise

    def test_eviction_works_without_an_embedder(self):
        cache = _cache(5, with_embedder=False)
        for i in range(20):
            _store(cache, f"question number {i}")
        assert cache.size() == 5


class TestPipelineBounds:
    def test_conversation_tracking_is_bounded(self):
        """Prefix survival only ever compares against the immediately preceding
        turn, so retaining every conversation forever buys nothing."""
        p = Pipeline(full_stack())
        p._max_tracked_conversations = 20
        for i in range(100):
            p.run(f"Distinct question {i} on subject {i}?", conversation_id=f"c{i}")
        assert len(p._last_prompt_ids) <= 20

    def test_cache_is_bounded_end_to_end(self):
        cfg = full_stack()
        cfg = replace(cfg, cache=replace(cfg.cache, max_entries=25))
        p = Pipeline(cfg)
        for i in range(200):
            p.run(f"Distinct question {i} on subject {i}?", conversation_id=f"c{i}")
        assert p.cache.size() <= 25
        assert p.cache.stats.evicted > 0

    def test_prefix_survival_still_works_within_the_bound(self):
        """The bound must not break the measurement it bounds."""
        from parsimony.core.types import Turn

        p = Pipeline(full_stack())
        hist: list[Turn] = []
        ratios = []
        for i, q in enumerate(["Explain hash tables.", "What is the complexity?",
                               "And in the worst case?"]):
            o = p.run(q, tuple(hist), conversation_id="stable", turn_index=i)
            hist += [Turn(f"u{i}", "user", q), Turn(f"a{i}", "assistant", o.response)]
            ratios.append(o.row.prefix_tokens_survived or 0)
        assert max(ratios) > 0

    def test_the_default_cap_is_generous_enough_for_the_corpus(self):
        """A cap that evicted during a normal sweep would silently depress the
        hit rate and corrupt every M2 result."""
        from parsimony.eval.corpus import load_corpus

        assert full_stack().cache.max_entries > load_corpus().n_requests * 4
