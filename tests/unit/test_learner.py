"""M7 — conversation mining, counterfactual replay, and warm start."""

from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path

from parsimony.core.config import full_stack
from parsimony.core.types import Mode
from parsimony.infra.embedding import HashingEmbedder
from parsimony.pipeline.warm_start import warm_start
from parsimony.modules.m7_learner import (
    PolicyBundle,
    counterfactual_redundancy,
    learn,
    mine_recurring_questions,
    mine_standing_context,
    mine_templates,
)


class TestMining:
    def test_recurring_questions_need_a_repeat(self):
        questions = ["what is x", "what is x", "asked once"]
        answers = {"what is x": "the answer", "asked once": "other"}
        seeded = mine_recurring_questions(questions, answers, min_count=2)
        assert [q for q, _ in seeded] == ["what is x"]

    def test_recurring_questions_are_deduplicated(self):
        questions = ["repeat me"] * 4
        seeded = mine_recurring_questions(questions, {"repeat me": "a"}, min_count=2)
        assert len(seeded) == 1

    def test_standing_context_needs_repetition_across_sessions(self):
        """A fact stated once is not standing context; a fact restated every
        session is exactly what belongs in the invariant zone."""
        once = mine_standing_context(["I am using Python 3.11"], min_count=2)
        assert once == ""
        twice = mine_standing_context(
            ["I am using Python 3.11", "I am using Python 3.11 for this"], min_count=2
        )
        assert "python 3.11" in twice.lower()

    def test_templates_capture_numeric_slots(self):
        templates = mine_templates(
            ["convert 100 km to miles", "convert 250 km to miles"], min_count=2
        )
        assert any("{n}" in t for t in templates)

    def test_templates_ignore_one_off_shapes(self):
        assert mine_templates(["convert 100 km to miles"], min_count=2) == []


class TestCounterfactualReplay:
    def test_identifies_a_phrase_that_never_changes_the_answer(self):
        """A stable generator means removing the phrase is provably safe."""
        embedder = HashingEmbedder()
        findings = counterfactual_redundancy(
            ["please explain recursion", "please explain hashing"],
            generate=lambda q: "a fixed answer regardless of the question asked",
            embedder=embedder,
        )
        please = next(f for f in findings if f.phrase == "please")
        assert please.safe_rate == 1.0

    def test_identifies_a_phrase_that_does_change_the_answer(self):
        """If the answer moves when the phrase is removed, it is load-bearing
        and must not enter the lexicon."""
        embedder = HashingEmbedder()
        findings = counterfactual_redundancy(
            ["please explain recursion", "please explain hashing"],
            generate=lambda q: f"answer shaped entirely by {q}",
            embedder=embedder,
        )
        please = next(f for f in findings if f.phrase == "please")
        assert please.safe_rate < 1.0

    def test_ignores_phrases_below_the_occurrence_floor(self):
        findings = counterfactual_redundancy(
            ["kindly explain recursion"],
            generate=lambda q: "stable",
            embedder=HashingEmbedder(),
            min_occurrences=2,
        )
        assert all(f.phrase != "kindly" for f in findings)


class TestBundle:
    def _bundle(self):
        return PolicyBundle(
            cache_seed=[("what is x", "the answer")],
            redundancy=["please", "thanks"],
            digest="Standing context:\n- i am using python 3.11",
            templates=["convert {n} km to miles"],
        )

    def test_round_trips_through_disk(self):
        with tempfile.TemporaryDirectory() as d:
            self._bundle().save(Path(d))
            loaded = PolicyBundle.load(Path(d))
            assert loaded.cache_seed == [("what is x", "the answer")]
            assert set(loaded.redundancy) == {"please", "thanks"}
            assert "python 3.11" in loaded.digest
            assert loaded.templates == ["convert {n} km to miles"]

    def test_hash_is_stable_for_identical_content(self):
        assert self._bundle().bundle_hash == self._bundle().bundle_hash

    def test_hash_changes_with_content(self):
        other = self._bundle()
        other.redundancy.append("kindly")
        assert other.bundle_hash != self._bundle().bundle_hash

    def test_writes_a_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            self._bundle().save(Path(d))
            assert (Path(d) / "MANIFEST.sha256").exists()

    def test_redacts_pii_before_writing(self):
        """The bundle persists and may be shared; the cache is the one component
        with a memory (ADR-013)."""
        bundle = PolicyBundle(cache_seed=[("mail me at a.b@c.com", "sure")])
        with tempfile.TemporaryDirectory() as d:
            bundle.save(Path(d))
            written = (Path(d) / "cache_seed.jsonl").read_text(encoding="utf-8")
            assert "a.b@c.com" not in written
            assert "EMAIL" in written


class TestWarmStart:
    def test_seeded_entries_produce_immediate_hits(self, make_pipeline):
        """Gap 6: on day one the cache is empty and the hit rate is zero. Mining
        the user's own logs moves day-one performance up the curve."""
        cfg = replace(full_stack(), mode=Mode.EXPERIMENT)
        cold = make_pipeline(cfg)
        assert not cold.run("what is x", conversation_id="c1").row.cache_hit

        warm = make_pipeline(cfg)
        seeded = warm_start(warm, PolicyBundle(cache_seed=[("what is x", "the seeded answer")]))
        assert seeded == 1
        outcome = warm.run("what is x", conversation_id="c1")
        assert outcome.row.cache_hit
        assert outcome.response == "the seeded answer"
        assert outcome.row.tokens_out == 0

    def test_warm_and_cold_use_the_same_code_path(self, make_pipeline):
        """Warm vs cold must differ only in what the cache already contains, so
        the reported curves cannot diverge for uninteresting reasons."""
        cfg = replace(full_stack(), mode=Mode.EXPERIMENT)
        warm = make_pipeline(cfg)
        warm_start(warm, PolicyBundle(cache_seed=[("what is x", "seeded")]))
        cold = make_pipeline(cfg)
        assert warm.registry.ordered(cfg) is not None
        assert [s.name for s in warm.registry.ordered(cfg)] == [
            s.name for s in cold.registry.ordered(cfg)
        ]

    def test_digest_lands_in_the_invariant_zone(self, make_pipeline):
        """The mined digest lengthens the byte-stable prefix, which is where M4
        turns it into reusable KV cache."""
        cfg = replace(full_stack(), mode=Mode.EXPERIMENT,
                      context_digest="Standing context:\n- uses python 3.11")
        outcome = make_pipeline(cfg).run("what now", conversation_id="c1")
        assert outcome.ctx.assembled is not None
        assert "python 3.11" in outcome.ctx.assembled.invariant_zone

    def test_bundle_hash_is_recorded_in_the_config(self):
        cfg = replace(full_stack(), bundle_hash="abc123")
        assert cfg.bundle_hash == "abc123"
        assert cfg.config_hash != full_stack().config_hash


class TestLearnEndToEnd:
    def test_produces_a_bundle_from_conversations(self):
        conversations = [
            ["please explain recursion", "thanks, what about hashing"],
            ["please explain recursion"],
            ["convert 100 km to miles", "convert 250 km to miles"],
        ]
        bundle = learn(
            conversations,
            generate=lambda q: "a stable answer",
            embedder=HashingEmbedder(),
        )
        assert bundle.source_conversations == 3
        assert any(q == "please explain recursion" for q, _ in bundle.cache_seed)
        assert bundle.templates
        assert "please" in bundle.redundancy
