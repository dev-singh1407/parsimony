"""The content embedder, and the stopword trap underneath it.

ADR-028 measured tier 2's near-zero contribution as an ENCODER property: the
paraphrases it exists to merge scored barely above unrelated text. ADR-035 is
the fix — stopword removal plus stemming, still purely lexical, no PyTorch.

The most valuable tests here are the ones guarding the mistake made building
it: a stopword list lifted from information retrieval strips "not", and a cache
that cannot see negation is not a cache, it is a hazard.
"""

from __future__ import annotations

import pytest

from parsimony.core.config import full_stack
from parsimony.eval.calibration import sweep_thresholds
from parsimony.eval.corpus import load_adversarial
from parsimony.infra.embedding import (
    _STOPWORDS,
    ContentEmbedder,
    HashingEmbedder,
    get_embedder,
    text_similarity,
)

# The three pairs ADR-028 names as tier 2's intended targets.
ADR028_PAIRS = [
    ("The library will close at 6 PM on weekdays", "The library shuts at 6 PM on weekdays"),
    ("Rent is 1200 per month", "Monthly rent comes to 1200"),
    ("The recipe needs 250 g of flour", "You will need 250 g flour for this"),
]


@pytest.fixture(scope="module")
def new():
    return ContentEmbedder()


@pytest.fixture(scope="module")
def old():
    return HashingEmbedder()


class TestOperativeWordsAreNeverStopwords:
    """The bug this class exists for: with "not" and "no" on the stopword list,
    "Is it safe to mix bleach and vinegar?" and "Is it NOT safe…" embedded to
    cosine 1.000 — bit-identical vectors for opposite questions — putting an
    11.1% false-hit floor under every threshold including 0.99."""

    @pytest.mark.parametrize("word", ["not", "no", "never", "all", "any", "some", "every"])
    def test_negation_and_quantifiers_survive(self, word):
        assert word not in _STOPWORDS

    def test_negation_changes_the_vector(self, new):
        sim = text_similarity(
            "Is it safe to mix bleach and vinegar?",
            "Is it not safe to mix bleach and vinegar?",
            new,
        )
        assert sim < 0.999, "opposite questions must not be bit-identical"

    def test_no_adversarial_pair_embeds_identically(self, new):
        """Any pair at 1.000 is invisible to every threshold at once."""
        collisions = [
            p for p in load_adversarial()
            if p.answers_differ and text_similarity(p.a, p.b, new) >= 0.9999
        ]
        assert not collisions, [p.pair_id for p in collisions]


class TestItFixesWhatADR028Measured:
    @pytest.mark.parametrize("a,b", ADR028_PAIRS)
    def test_paraphrases_score_higher_than_before(self, a, b, old, new):
        assert text_similarity(a, b, new) > text_similarity(a, b, old)

    def test_the_worst_pair_improves_substantially(self, old, new):
        """0.321 -> 0.847: the recipe pair, which the lexical encoder placed
        barely above unrelated text."""
        a, b = ADR028_PAIRS[2]
        assert text_similarity(a, b, old) < 0.4
        assert text_similarity(a, b, new) > 0.7

    def test_stemming_merges_inflections(self, new):
        assert text_similarity("Rent is 1200 per month", "Monthly rent 1200", new) > 0.5

    def test_unrelated_text_stays_unrelated(self, new):
        """A fix that raises every similarity is not a fix."""
        assert text_similarity(
            "How do I reverse a string in Python?",
            "Sourdough bread needs a starter culture.",
            new,
        ) < 0.2


class TestCalibrationImproves:
    """The deliverable is the operating point, not the mean similarity."""

    @pytest.fixture(scope="class")
    @classmethod
    def points(cls):
        return {
            "old": sweep_thresholds(full_stack(), embedder=HashingEmbedder()),
            "new": sweep_thresholds(full_stack(), embedder=ContentEmbedder()),
        }

    def test_a_safe_operating_point_still_exists(self, points):
        assert [p for p in points["new"] if p.false_hit_rate < 2.0]

    def test_more_true_hits_at_the_same_safety(self, points):
        best = {k: max((p for p in v if p.false_hit_rate < 2.0),
                       key=lambda p: p.true_hit_rate)
                for k, v in points.items()}
        assert best["new"].true_hit_rate > best["old"].true_hit_rate

    def test_the_default_threshold_is_safe_with_margin(self, points):
        """0.97 is chosen over 0.92 deliberately. 0.92 is also safe on this
        corpus and buys 2.2 more points, but 0.90 is not (4.4% false), so the
        default keeps a threshold step of margin rather than sitting one
        adversarial pair from the cliff."""
        at_default = next(p for p in points["new"] if p.tau_hi == 0.97)
        assert at_default.false_hit_rate == 0.0


class TestSubstitutability:
    def test_registry_returns_it_by_id(self):
        assert isinstance(get_embedder("content-v1"), ContentEmbedder)

    def test_hashing_is_still_reachable(self):
        e = get_embedder("hashing-v1")
        assert isinstance(e, HashingEmbedder) and not isinstance(e, ContentEmbedder)

    def test_it_is_the_default(self):
        assert full_stack().embedder_id.startswith("content")

    def test_id_is_distinct_so_thresholds_cannot_be_confused(self):
        """Every threshold is calibrated per encoder (Contribution 6). A shared
        id would let one encoder's calibration be read as another's."""
        assert ContentEmbedder().id != HashingEmbedder().id

    def test_all_stopword_input_still_embeds(self, new):
        """"What is it?" is entirely stopwords. Without a fallback these all
        collapse to the zero vector and collide in the index."""
        for text in ("What is it?", "the", "is this that?"):
            v = new.embed([text])[0]
            assert float((v * v).sum()) > 0.0
