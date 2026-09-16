"""The obvious approaches have to fail for the reason the tour says they do.

The guided tour runs each of these beside the real layer and puts the result
on screen as evidence that the layer is not trivial. That is only honest if
the obvious version is built the way a practitioner would build it, and if the
failure is a property of the approach rather than of a contrived input. Each
test below pins one sentence the tour, or naive.py's docstring, says aloud.
"""

from __future__ import annotations

import pytest

from parsimony.core.config import full_stack
from parsimony.core.types import Turn
from parsimony.eval import naive
from parsimony.infra.embedding import get_embedder
from parsimony.infra.tokenization import HeuristicTokenizer, get_tokenizer
from parsimony.modules.m8_fidelity import FidelityGate

BLEACH = "Is it safe to mix bleach and vinegar?"
NOT_BLEACH = "Is it not safe to mix bleach and vinegar?"


@pytest.fixture(scope="module")
def embedder():
    return get_embedder(full_stack().embedder_id)


class TestStopwordRemoval:
    def test_the_standard_list_contains_the_negation(self):
        """Not a list chosen to fail: NLTK's English list really does include these."""
        assert {"not", "no", "nor"} <= naive.NLTK_STOPWORDS

    def test_removing_stopwords_reverses_the_question(self):
        assert naive.stopword_compress(NOT_BLEACH) == "safe mix bleach vinegar"

    def test_the_safety_check_sees_what_was_lost(self):
        gate = FidelityGate()
        cut = naive.stopword_compress(NOT_BLEACH)
        assert gate.invariants_of(NOT_BLEACH).negations - gate.invariants_of(cut).negations == {"not"}

    def test_it_does_shorten_the_prompt(self, tok):
        """The failure is not that it saves nothing -- it saves tokens and loses meaning."""
        assert tok.count(naive.stopword_compress(NOT_BLEACH)) < tok.count(NOT_BLEACH)


class TestAbbreviation:
    def test_rewrites_whole_words_only(self):
        assert naive.abbreviate("Please send information without delay") == \
            "pls send info w/o delay"
        assert naive.abbreviate("misinformation") == "misinformation"

    def test_abbreviations_are_real_ones(self):
        """Every short form is a prefix-style or conventional abbreviation, not a typo."""
        conventional = {"info", "approx", "pls", "w/o", "e.g.", "govt"}
        assert {short for _, short in naive.ABBREVIATIONS} == conventional

    def test_no_abbreviation_saves_a_token_on_the_models_vocabulary(self):
        """What the tour shows: 'saves nothing' or 'costs MORE', never 'saves'."""
        tok = get_tokenizer(full_stack().tokenizer_id)
        if isinstance(tok, HeuristicTokenizer):
            pytest.skip("needs the model's real vocabulary, which is not cached here")
        costs = {long: (tok.count(f" {long}"), tok.count(f" {short}"))
                 for long, short in naive.ABBREVIATIONS}
        assert all(after >= before for before, after in costs.values()), costs
        assert any(after > before for before, after in costs.values()), costs


class TestThresholdCache:
    def test_default_sits_inside_the_commonly_recommended_range(self):
        assert 0.85 <= naive.DEFAULT_CACHE_THRESHOLD <= 0.92

    def test_reuses_the_answer_to_the_opposite_question(self, embedder):
        hit, score = naive.threshold_cache_hit(BLEACH, NOT_BLEACH, embedder)
        assert hit and score >= naive.DEFAULT_CACHE_THRESHOLD

    def test_misses_a_politely_worded_repeat(self, embedder):
        hit, score = naive.threshold_cache_hit(
            "Hello, could you please explain what recursion is? Thanks!",
            "What is recursion?", embedder)
        assert not hit and score < naive.DEFAULT_CACHE_THRESHOLD

    def test_the_opposite_scores_higher_than_the_rewording(self, embedder):
        """The tour's central claim about the memory: no threshold separates these."""
        opposite = naive.cosine(embedder, BLEACH, NOT_BLEACH)
        rewording = naive.cosine(embedder,
                                 "Hello, could you please explain what recursion is? Thanks!",
                                 "What is recursion?")
        assert opposite > rewording

    def test_what_and_where_look_identical_to_similarity(self, embedder):
        hit, _ = naive.threshold_cache_hit("What is Java?", "Where is Java?", embedder)
        assert hit


class TestKeepLast:
    HISTORY = tuple(Turn(f"t{i}", "user" if i % 2 == 0 else "assistant", text)
                    for i, text in enumerate(["My server runs Ubuntu 22.04 with 8 GB of RAM.",
                                              "Noted.", "Name a cat.", "Whiskers.",
                                              "Recommend a film.", "Inception."]))

    def test_drops_the_earliest_message_first(self):
        kept = naive.keep_last(self.HISTORY, 4)
        assert len(kept) == 4
        assert all("Ubuntu" not in t.content for t in kept)

    def test_keeps_the_most_recent_in_order(self):
        assert naive.keep_last(self.HISTORY, 2) == self.HISTORY[-2:]

    def test_zero_keeps_nothing(self):
        assert naive.keep_last(self.HISTORY, 0) == ()
