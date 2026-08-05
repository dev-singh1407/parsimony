"""Cache key safety and the paired-observation lookup mode.

Both suites cover bugs that were live in the pushed code:

  * six degenerate queries shared one cache key and served each other's
    answers, because canonicalisation strips trailing punctuation and they all
    reduced to the empty string;
  * cache_lookup_on="BOTH" was accepted by config validation and documented as
    a paired-observation design, but behaved exactly like RAW.
"""

from __future__ import annotations

import pytest

from parsimony.core.config import full_stack, with_cache_lookup
from parsimony.core.types import Mode, RouteTier
from parsimony.modules.m2_cache import (
    SemanticCache,
    canonicalise,
    chain_hash,
    is_cacheable,
)

DEGENERATE = ["", "   ", "   \n\t  ", "?", "?!...", "!!!", "...", ".", " . ! ? "]


class TestUncacheableQueries:
    @pytest.mark.parametrize("query", DEGENERATE)
    def test_degenerate_queries_are_not_cacheable(self, query):
        assert not is_cacheable(query)

    @pytest.mark.parametrize(
        "query",
        ["x", "2+2", "What is recursion?", "hi", "a?", "```code```", "café"],
    )
    def test_real_queries_remain_cacheable(self, query):
        assert is_cacheable(query)

    def test_degenerate_queries_all_share_one_key(self):
        """The underlying collision still exists — the guard is what stops it
        being reachable. Asserting it here documents WHY the guard is needed."""
        keys = {SemanticCache.make_key(q, "root", "m") for q in DEGENERATE}
        assert len(keys) == 1

    def test_store_refuses_an_uncacheable_query(self):
        cache = SemanticCache()
        key = SemanticCache.make_key("?!...", "root", "m")
        assert cache.store(key, "?!...", "an answer") is None
        assert cache.size() == 0
        assert cache.stats.uncacheable == 1

    def test_lookup_refuses_an_uncacheable_query(self):
        """Belt and braces: even if an entry existed, a degenerate query must
        not retrieve it."""
        cache = SemanticCache()
        key = SemanticCache.make_key("real question about python", "root", "m")
        cache.store(key, "real question about python", "the answer")
        assert cache.lookup(key, query="?!...") is None

    def test_degenerate_queries_do_not_serve_each_others_answers(self, make_pipeline):
        """The end-to-end failure, as a regression test. Before the guard these
        three served one another from the exact-hash tier — which the three-zone
        verifier can never catch, because it only guards the semantic tier."""
        p = make_pipeline(full_stack())
        first = p.run("   \n\t  ", conversation_id="a")
        second = p.run("?!...", conversation_id="b")
        third = p.run("!!!", conversation_id="c")
        assert not second.row.cache_hit
        assert not third.row.cache_hit
        assert first.row.route_tier == RouteTier.MODEL_SMALL.name

    def test_a_real_repeat_still_hits(self, make_pipeline):
        """The guard must not cost us legitimate caching."""
        p = make_pipeline(full_stack())
        p.run("What is the capital of France?", conversation_id="a")
        again = p.run("What is the capital of France?", conversation_id="b")
        assert again.row.cache_hit


class TestCacheLookupOrdering:
    def test_raw_places_the_lookup_before_compression(self):
        order = with_cache_lookup(full_stack(), "RAW").stage_order
        assert order.index("m2_cache") < order.index("m1_tier1")
        assert "m2_cache_probe" not in order

    def test_compressed_places_the_lookup_after_compression(self):
        order = with_cache_lookup(full_stack(), "COMPRESSED").stage_order
        assert order.index("m2_cache") > order.index("m1_tier3")
        assert "m2_cache_probe" not in order

    def test_both_runs_two_lookups_straddling_compression(self):
        """The bug: BOTH validated fine and behaved exactly like RAW, while
        docs/00-architecture.md described it as a paired-observation design."""
        order = with_cache_lookup(full_stack(), "BOTH").stage_order
        assert "m2_cache_probe" in order
        assert order.index("m2_cache_probe") < order.index("m1_tier1")
        assert order.index("m2_cache") > order.index("m1_tier3")

    @pytest.mark.parametrize("mode", ["RAW", "COMPRESSED", "BOTH"])
    def test_every_mode_produces_a_valid_stage_order(self, mode, make_pipeline):
        make_pipeline(with_cache_lookup(full_stack(), mode)).run("What is recursion?")

    def test_the_probe_never_short_circuits(self, make_pipeline):
        """A probe records what the cache WOULD have done. If it served the
        request instead, the compressed-arm observation would never happen and
        the pairing would be lost."""
        from dataclasses import replace

        cfg = with_cache_lookup(replace(full_stack(), mode=Mode.EXPERIMENT), "BOTH")
        p = make_pipeline(cfg)
        q = "Please explain recursion."
        p.run(q, conversation_id="a")
        second = p.run(q, conversation_id="b")

        probe = next(t for t in second.traces if t.name == "m2_cache_probe")
        assert probe.outcome.value != "short_circuit"
        assert probe.evidence.get("probe_only") is True

    def test_both_records_a_paired_observation(self, make_pipeline):
        """One request, two observations of the same cache under both
        orderings — the point of the mode."""
        from dataclasses import replace

        cfg = with_cache_lookup(replace(full_stack(), mode=Mode.EXPERIMENT), "BOTH")
        p = make_pipeline(cfg)
        q = "Please explain recursion."
        p.run(q, conversation_id="a")
        second = p.run(q, conversation_id="b")

        names = [t.name for t in second.traces]
        assert "m2_cache_probe" in names
        probe = next(t for t in second.traces if t.name == "m2_cache_probe")
        assert "zone" in probe.evidence

    def test_the_probe_does_not_write_to_the_cache(self, make_pipeline):
        """Two stages sharing one cache must not double-write."""
        from dataclasses import replace

        cfg = with_cache_lookup(replace(full_stack(), mode=Mode.EXPERIMENT), "BOTH")
        p = make_pipeline(cfg)
        p.run("What is the capital of France?", conversation_id="a")
        assert p.cache.stats.stores == 1


class TestChainAndCanonicalisation:
    def test_canonicalisation_is_idempotent(self):
        once = canonicalise("  What IS  Recursion??  ")
        assert canonicalise(once) == once

    def test_chain_depth_zero_is_root(self):
        from parsimony.core.types import Turn

        assert chain_hash((Turn("1", "user", "hello"),), 0) == "root"

    def test_empty_history_is_root_at_any_depth(self):
        assert chain_hash((), 5) == "root"
