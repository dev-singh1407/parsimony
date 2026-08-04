"""Embedder, exact index, MMR, and the semantic cache tier."""

from __future__ import annotations

import numpy as np
import pytest

from parsimony.core.config import CacheConfig, full_stack
from parsimony.core.proposals import NoOp, ShortCircuit
from parsimony.core.types import RouteTier
from parsimony.infra.embedding import ExactIndex, HashingEmbedder, mmr_select
from parsimony.infra.nlp import RegexInvariantExtractor
from parsimony.modules.m2_cache import (
    CacheLookupStage,
    SemanticCache,
    chain_hash,
    verify_match,
)


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


class TestHashingEmbedder:
    def test_vectors_are_unit_length(self, embedder):
        vecs = embedder.embed(["hello world", "something else entirely"])
        assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)

    def test_identical_text_scores_one(self, embedder):
        a, b = embedder.embed(["what is recursion?", "what is recursion?"])
        assert float(a @ b) == pytest.approx(1.0, abs=1e-5)

    def test_unrelated_text_scores_near_zero(self, embedder):
        a, b = embedder.embed(["What is photosynthesis?", "How do I reverse a string?"])
        assert float(a @ b) < 0.3

    def test_is_deterministic_across_instances(self):
        """Python's hash() is salted per process; using it would silently break
        every cross-run cache measurement in the project."""
        a = HashingEmbedder().embed(["stability matters"])[0]
        b = HashingEmbedder().embed(["stability matters"])[0]
        assert np.array_equal(a, b)

    def test_morphological_variants_score_higher_than_unrelated(self, embedder):
        v = embedder.embed(["rain", "rainfall", "database"])
        assert float(v[0] @ v[1]) > float(v[0] @ v[2])

    def test_empty_batch_returns_empty_matrix(self, embedder):
        assert embedder.embed([]).shape == (0, embedder.dim)

    def test_negation_pairs_score_dangerously_high(self, embedder):
        """ADR-024, as a regression test.

        The pair that must NOT be auto-accepted is the most similar pair we
        have. If a future encoder change lowers this below tau_hi the design
        assumption has changed and the thresholds need revisiting.
        """
        a, b = embedder.embed(
            ["Is it safe to mix bleach and vinegar?",
             "Is it not safe to mix bleach and vinegar?"]
        )
        assert float(a @ b) > 0.85
        assert float(a @ b) > CacheConfig().tau_lo  # lands in the verify zone
        assert float(a @ b) < CacheConfig().tau_hi  # and must NOT auto-accept


class TestExactIndex:
    def test_finds_the_nearest_neighbour(self, embedder):
        index = ExactIndex(embedder.dim)
        texts = ["what is recursion", "how to bake bread", "what is a hash table"]
        for text, vec in zip(texts, embedder.embed(texts)):
            index.add(vec, text)
        top = index.search(embedder.embed(["what is recursion?"])[0], 1)
        assert top[0][0] == "what is recursion"

    def test_reports_itself_as_exact(self, embedder):
        assert ExactIndex(embedder.dim).is_exact()

    def test_recall_at_one_is_perfect(self, embedder):
        """Exactness is what lets the false-hit rate be attributed to the
        policy rather than to index recall (ADR-004)."""
        index = ExactIndex(embedder.dim)
        texts = [f"question number {i} about topic {i}" for i in range(60)]
        vecs = embedder.embed(texts)
        for text, vec in zip(texts, vecs):
            index.add(vec, text)
        for text, vec in zip(texts, vecs):
            assert index.search(vec, 1)[0][0] == text

    def test_results_are_sorted_by_descending_score(self, embedder):
        index = ExactIndex(embedder.dim)
        texts = ["alpha beta", "alpha gamma", "totally different"]
        for t, v in zip(texts, embedder.embed(texts)):
            index.add(v, t)
        scores = [s for _, s in index.search(embedder.embed(["alpha beta"])[0], 3)]
        assert scores == sorted(scores, reverse=True)

    def test_remove_takes_an_entry_out(self, embedder):
        index = ExactIndex(embedder.dim)
        for t, v in zip(["a", "b"], embedder.embed(["alpha", "beta"])):
            index.add(v, t)
        index.remove("a")
        assert index.size() == 1
        assert [i for i, _ in index.search(embedder.embed(["alpha"])[0], 2)] == ["b"]

    def test_re_adding_the_same_id_updates_rather_than_duplicates(self, embedder):
        index = ExactIndex(embedder.dim)
        v = embedder.embed(["one", "two"])
        index.add(v[0], "x")
        index.add(v[1], "x")
        assert index.size() == 1

    def test_empty_index_returns_nothing(self, embedder):
        assert ExactIndex(embedder.dim).search(embedder.embed(["q"])[0], 5) == []


class TestMmr:
    def test_picks_the_most_relevant_first(self, embedder):
        texts = ["hash tables and hashing", "baking sourdough bread", "hash map complexity"]
        vecs = embedder.embed(texts)
        query = embedder.embed(["hash table"])[0]
        assert mmr_select(vecs, query, k=1)[0] in (0, 2)

    def test_diversity_beats_redundancy_at_low_lambda(self, embedder):
        texts = ["hash tables explained", "hash tables explained again", "sourdough bread"]
        vecs = embedder.embed(texts)
        query = embedder.embed(["hash tables"])[0]
        picked = mmr_select(vecs, query, k=2, lambda_=0.2)
        assert 2 in picked  # the diverse item is pulled in

    def test_respects_k(self, embedder):
        vecs = embedder.embed(["a", "b", "c"])
        assert len(mmr_select(vecs, vecs[0], k=2)) == 2

    def test_handles_empty_input(self, embedder):
        assert mmr_select(np.zeros((0, embedder.dim)), np.zeros(embedder.dim), k=3) == []


class TestVerifier:
    def _inv(self, text):
        return RegexInvariantExtractor().extract(text)

    def test_rejects_a_negation_mismatch(self):
        a, b = "Is it safe to mix them?", "Is it not safe to mix them?"
        result = verify_match(self._inv(a), self._inv(b), a, b, 0.55)
        assert not result.passed
        assert not result.negation_agree
        assert result.failure() == "negation mismatch"

    def test_rejects_a_number_mismatch(self):
        a, b = "Convert 100 km to miles", "Convert 200 km to miles"
        result = verify_match(self._inv(a), self._inv(b), a, b, 0.55)
        assert not result.passed
        assert not result.number_agree

    def test_rejects_an_entity_mismatch(self):
        a, b = "What is the capital of Australia?", "What is the capital of Austria?"
        result = verify_match(self._inv(a), self._inv(b), a, b, 0.55)
        assert not result.passed
        assert not result.entity_agree

    def test_accepts_a_politeness_only_difference(self):
        a, b = "Explain recursion", "Please explain recursion"
        assert verify_match(self._inv(a), self._inv(b), a, b, 0.55).passed

    def test_rejects_when_lexical_overlap_is_too_low(self):
        a, b = "Explain recursion", "Describe iteration loops in detail"
        assert not verify_match(self._inv(a), self._inv(b), a, b, 0.55).passed


class TestSemanticCacheTier:
    def _stage(self, embedder):
        return CacheLookupStage(SemanticCache(embedder=embedder))

    def test_a_paraphrase_hits_through_the_verify_zone(self, make_pipeline):
        p = make_pipeline(full_stack())
        p.run("What is the capital city of France?")
        again = p.run("What is the capital city of France")  # punctuation only
        assert again.row.cache_hit

    def test_an_adversarial_negation_pair_does_not_hit(self, make_pipeline):
        """The headline safety property: high similarity must not defeat meaning."""
        p = make_pipeline(full_stack())
        p.run("Is it safe to mix bleach and vinegar?")
        second = p.run("Is it not safe to mix bleach and vinegar?")
        assert not second.row.cache_hit

    def test_an_adversarial_number_pair_does_not_hit(self, make_pipeline):
        p = make_pipeline(full_stack())
        p.run("How many days are in 3 weeks?")
        second = p.run("How many days are in 5 weeks?")
        assert not second.row.cache_hit

    def test_an_adversarial_entity_pair_does_not_hit(self, make_pipeline):
        p = make_pipeline(full_stack())
        p.run("What is the capital of Australia?")
        second = p.run("What is the capital of Austria?")
        assert not second.row.cache_hit

    def test_candidates_are_recorded_even_on_a_miss(self, make_pipeline):
        """This is what makes the threshold sweep an offline groupby."""
        p = make_pipeline(full_stack())
        p.run("Is it safe to mix bleach and vinegar?")
        second = p.run("Is it not safe to mix bleach and vinegar?")
        assert second.row.cache_top_k
        assert second.row.cache_zone in ("verify", "reject")
        assert second.row.cache_verifier is not None

    def test_entries_are_scoped_by_context_chain(self, embedder):
        cache = SemanticCache(embedder=embedder)
        vec = embedder.embed(["and the second one?"])[0]
        cache.store("k1", "and the second one?", "answer A", chain="conversationA",
                    model_id="m", vec=vec)
        assert cache.search(vec, "conversationA", "m", 3)
        assert not cache.search(vec, "conversationB", "m", 3)

    def test_entries_are_scoped_by_model(self, embedder):
        cache = SemanticCache(embedder=embedder)
        vec = embedder.embed(["what is recursion?"])[0]
        cache.store("k1", "what is recursion?", "a", chain="root", model_id="llama", vec=vec)
        assert not cache.search(vec, "root", "qwen", 3)
