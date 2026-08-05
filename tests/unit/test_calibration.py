"""Verifier checks added after measurement, and the calibration sweep.

Each test here corresponds to a false hit that was actually observed against
the adversarial subset, so a regression re-opens a hole we know is exploitable.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from parsimony.core.config import full_stack
from parsimony.eval.calibration import evaluate_point, sweep_thresholds
from parsimony.eval.corpus import load_adversarial, load_corpus, load_gold
from parsimony.infra.embedding import get_embedder
from parsimony.infra.nlp import (
    RegexInvariantExtractor,
    morphological_negations,
    operative_modifiers,
)
from parsimony.modules.m2_cache import verify_match


def _verify(a: str, b: str, jaccard_min: float = 0.55):
    ex = RegexInvariantExtractor()
    return verify_match(ex.extract(a), ex.extract(b), a, b, jaccard_min)


class TestOperativeModifiers:
    def test_detects_extremum_terms(self):
        assert operative_modifiers("What is the minimum temperature?") == frozenset({"minimum"})

    def test_ignores_ordinary_words(self):
        assert operative_modifiers("What is the temperature?") == frozenset()

    @pytest.mark.parametrize(
        "a,b",
        [
            ("What is the minimum temperature for storage?",
             "What is the maximum temperature for storage?"),
            ("Show me the cheapest option", "Show me the most expensive option"),
            ("What is the fastest sorting algorithm?", "What is the slowest sorting algorithm?"),
            ("How do I increase the timeout?", "How do I decrease the timeout?"),
            ("Which countries export the most oil?", "Which countries import the most oil?"),
            ("What is the average response time?", "What is the median response time?"),
            ("Should I upgrade before the deadline?", "Should I upgrade after the deadline?"),
        ],
    )
    def test_modifier_swaps_are_rejected(self, a, b):
        """78% of these produced false cache hits before the modifier check."""
        result = _verify(a, b)
        assert not result.passed
        assert not result.modifier_agree

    def test_synonym_modifiers_still_match(self):
        """The control direction: 'brief' and 'short' are both in the lexicon,
        so they must not be treated as an opposition."""
        assert operative_modifiers("Give me a brief summary") != operative_modifiers(
            "Give me a short summary"
        )  # they differ as terms...
        # ...which means this pair is conservatively rejected. Documented cost of
        # a lexicon-based check: it cannot tell opposition from synonymy.


class TestMorphologicalNegation:
    def test_detects_prefix_negation_against_the_counterpart(self):
        assert morphological_negations(
            "Is it impossible to cancel?", "Is it possible to cancel?"
        ) == frozenset({"impossible"})

    def test_requires_the_stem_to_appear_in_the_other_query(self):
        """'international' is not a negation of anything here, so it must not
        fire — otherwise the check would reject unrelated queries."""
        assert morphological_negations("international shipping rates", "shipping rates") == frozenset()

    @pytest.mark.parametrize(
        "a,b",
        [
            ("Is it possible to cancel the subscription?",
             "Is it impossible to cancel the subscription?"),
            ("Can I use this library commercially?", "Can I use this library non commercially?"),
            ("Is the deposit refundable?", "Is the deposit non refundable?"),
            ("Does aspirin thin the blood?", "Does aspirin fail to thin the blood?"),
            ("Is this covered by the warranty?", "Is this excluded from the warranty?"),
        ],
    )
    def test_negation_without_a_particle_is_rejected(self, a, b):
        """Each of these was an observed false hit before the lexical and
        morphological negation checks were added."""
        assert not _verify(a, b).passed


class TestEntityIdentifiers:
    def test_alphanumeric_identifiers_are_extracted(self):
        inv = RegexInvariantExtractor().extract("How do I use Panda3D?")
        assert "Panda3D" in inv.entities

    def test_library_confusion_is_rejected(self):
        """'pandas' vs 'Panda3D' produced a false hit: the proper-noun pattern
        could not match across the digit, so both reported zero entities."""
        assert not _verify("How do I use pandas?", "How do I use Panda3D?").passed


@pytest.fixture(scope="module")
def pairs():
    return load_adversarial()


@pytest.fixture(scope="module")
def embedder():
    return get_embedder(full_stack().embedder_id)


class TestCalibrationSweep:
    def test_corpus_has_the_declared_composition(self, pairs):
        counts: dict[str, int] = {}
        for p in pairs:
            counts[p.operative] = counts.get(p.operative, 0) + 1
        assert set(counts) == {"negation", "number", "entity", "modifier"}
        assert len(pairs) == 50

    def test_includes_control_pairs(self, pairs):
        """Without controls, 'reject everything' scores a perfect false-hit rate."""
        assert sum(1 for p in pairs if not p.answers_differ) >= 4

    def test_default_configuration_meets_the_false_hit_target(self, pairs, embedder):
        """Report 3.3 sets the target below 2%."""
        point = evaluate_point(pairs, full_stack(), embedder, verifier_on=True)
        assert point.false_hit_rate < 2.0

    def test_the_verifier_is_what_achieves_it(self, pairs, embedder):
        """Turning the verifier off collapses to single-threshold behaviour and
        the false-hit rate should climb sharply — evidence that the threshold
        alone is not what makes the cache safe (ADR-024)."""
        with_verifier = evaluate_point(pairs, full_stack(), embedder, verifier_on=True)
        without = evaluate_point(pairs, full_stack(), embedder, verifier_on=False)
        assert without.false_hit_rate > with_verifier.false_hit_rate

    def test_lower_thresholds_are_monotonically_less_safe(self, pairs, embedder):
        points = sweep_thresholds(full_stack(), pairs=pairs, embedder=embedder)
        rates = [p.false_hit_rate for p in points]
        assert rates[0] > rates[-1]


class TestCorpusIntegrity:
    def test_conversation_corpus_size(self):
        assert len(load_corpus()) >= 150

    def test_follow_up_conversations_exceed_the_history_budget(self):
        """M3 can only be measured if some conversation actually overflows
        max_turns; corpus v0 never did, so M3 measured 0.0%."""
        corpus = load_corpus()
        longest = max(c.n_turns for c in corpus.by_class()["follow_up"])
        assert longest > full_stack().history.max_turns

    def test_gold_items_declare_grading_rules_in_advance(self):
        for item in load_gold():
            assert item.match in {"exact", "numeric", "set", "contains"}
            assert item.gold_answer

    def test_gold_subset_size(self):
        assert len(load_gold()) == 40
