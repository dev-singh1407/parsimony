"""The semantic cache as a person using `ask` actually experiences it.

Every case here was first reproduced as a live failure. Measured over
`corpus/interactive_pairs.jsonl`, the cache scored 24/37 before this work and
37/37 after, and three of the thirteen wrong decisions were not misses but
WRONG ANSWERS served with full confidence:

  * "Hello, could you please explain what recursion is? Thanks!" stored the
    polite wording, so the later plain "What is recursion?" scored 0.632 and
    was rejected -- the courtesy was being compared as though it were subject
    matter.
  * "Explain how hash tables work" missed "How does a hash table work?" at
    0.706, because the shared stemmer reduces "tables" to "tabl" and leaves
    "table" alone, so a plural never met its own singular.
  * "Why does it rain?" was never compared against anything at all: the word
    "it" marked it a follow-up and scoped it to one conversation. The same
    accident is why the adversarial bleach pair missed -- the verifier never
    ran, so the safety property was never actually demonstrated in a live
    session.
  * "What is Java?" and "Where is Java?" both reduce to the single content word
    "java" and score cosine 1.000. The accept zone does not consult the
    verifier, so the cache answered a where-question with a what-answer.

The corpus file carries the reasoning for each pair; this module is the
executable form of it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from parsimony.core.config import full_stack
from parsimony.core.types import Turn
from parsimony.eval.calibration import _lookup_hits
from parsimony.infra.embedding import get_embedder
from parsimony.modules.m2_cache import (
    cache_key_text,
    content_words,
    dependence_reason,
    is_context_dependent,
    question_type,
    question_types_agree,
    singularise,
)

PAIRS = [
    json.loads(line)
    for line in (Path(__file__).resolve().parents[2] / "corpus" / "interactive_pairs.jsonl")
    .read_text(encoding="utf-8").splitlines()
    if line.strip()
]


@pytest.fixture(scope="module")
def embedder():
    return get_embedder(full_stack().embedder_id)


def converse(pipeline, *questions):
    """Run questions as one conversation, the way `ask` does.

    The history matters: every one of these failures is invisible to a
    single-shot `run()`, which is why the existing pipeline tests passed
    throughout.
    """
    history: list[Turn] = []
    outcomes = []
    for q in questions:
        outcome = pipeline.run(q, tuple(history), conversation_id="c1",
                               turn_index=len(history))
        outcomes.append(outcome)
        history.append(Turn(f"u{len(history)}", "user", q))
        history.append(Turn(f"a{len(history)}", "assistant", outcome.response))
    return outcomes


def cache_trace(outcome):
    return next(t for t in outcome.traces if t.name == "m2_cache")


class TestEveryInteractivePair:
    """The whole corpus, decided by the same code path the calibration uses."""

    @pytest.mark.parametrize("pair", PAIRS, ids=[p["pair_id"] for p in PAIRS])
    def test_the_decision_matches_the_expectation(self, pair, embedder):
        hit = _lookup_hits(pair["b"], pair["a"], embedder, full_stack(), True)
        assert hit == (pair["expect"] == "hit"), (
            f"{pair['category']}: {pair['a']!r} -> {pair['b']!r}\n{pair['notes']}"
        )


class TestTheFailuresThatStartedThis:
    """The same four cases, live, through a real conversation."""

    def test_a_polite_question_is_reused_for_its_plain_form(self, make_pipeline):
        outcomes = converse(
            make_pipeline(full_stack()),
            "Hello, could you please explain what recursion is? Thanks!",
            "What is a pointer?",
            "What is recursion?",
        )
        assert outcomes[-1].row.cache_hit

    def test_a_plural_meets_its_own_singular(self, make_pipeline):
        outcomes = converse(
            make_pipeline(full_stack()),
            "How does a hash table work?",
            "What is a pointer?",
            "Explain how hash tables work",
        )
        assert outcomes[-1].row.cache_hit

    def test_filler_it_does_not_scope_a_self_contained_question(self, make_pipeline):
        outcomes = converse(
            make_pipeline(full_stack()),
            "What causes rain?",
            "What is a pointer?",
            "Why does it rain?",
        )
        assert outcomes[-1].row.cache_hit

    def test_a_where_question_is_never_served_a_what_answer(self, make_pipeline):
        outcomes = converse(
            make_pipeline(full_stack()),
            "What is Java?",
            "What is a pointer?",
            "Where is Java?",
        )
        last = outcomes[-1]
        assert not last.row.cache_hit
        trace = cache_trace(last)
        assert trace.evidence.get("type_agree") is False
        assert "kind of question" in trace.rationale

    def test_the_negation_pair_is_refused_by_the_verifier_not_by_accident(
        self, make_pipeline
    ):
        """Previously this missed because "it" scoped the question away, so the
        verifier never ran. A miss for the wrong reason is not a safety
        property -- it evaporates the moment the wording changes."""
        outcomes = converse(
            make_pipeline(full_stack()),
            "Is it safe to mix bleach and vinegar?",
            "What is a pointer?",
            "Is it not safe to mix bleach and vinegar?",
        )
        last = outcomes[-1]
        assert not last.row.cache_hit
        trace = cache_trace(last)
        assert trace.evidence.get("zone") in ("verify", "accept"), (
            "the candidate must actually be compared, not filtered out first"
        )
        verifier = trace.evidence.get("verifier") or {}
        assert verifier.get("negation_agree") == 0.0
        assert "negation" in trace.rationale


class TestWhatCountsAsAFollowUp:
    @pytest.mark.parametrize("query", [
        "What is 847 * 23?",
        "How do I reverse a string in Python?",
        "Explain recursion.",
        "What is the capital of Australia?",
        "Why does it rain?",
        "Is it safe to mix bleach and vinegar?",
        "Is it not safe to mix bleach and vinegar?",
        "Is it true that water boils at 100 degrees?",
        "When was Python first released?",
        "How long does it take to learn Rust?",
    ])
    def test_self_contained(self, query):
        assert not is_context_dependent(query)

    @pytest.mark.parametrize("query", [
        "And what about the second one?",
        "Why does that affect lookup time?",
        "Can you explain it again?",
        "So how do I fix them?",
        "Summarise this.",
        "Tell me more about it",
        "What about the other one",
        "Is that also true",
        "Do the same again",
        "Is it safe to eat?",
        "How long does it take?",
        "Is the first one faster?",
        "What is the last step?",
    ])
    def test_depends_on_the_conversation(self, query):
        assert is_context_dependent(query)

    def test_the_reason_is_reported_so_a_trace_can_explain_itself(self):
        assert dependence_reason("Summarise this.") == "this"
        assert dependence_reason("And what about the second one?") == "and"
        assert dependence_reason("What is recursion?") is None

    def test_referential_it_still_scopes_because_the_object_is_missing(self):
        """"Is it safe to eat?" needs the conversation to say what "it" is;
        "Is it safe to eat raw eggs?" supplies its own subject."""
        assert is_context_dependent("Is it safe to eat?")
        assert not is_context_dependent("Is it safe to eat raw eggs?")


class TestQuestionKind:
    @pytest.mark.parametrize("query,kind", [
        ("What is recursion?", "what"),
        ("Can you explain recursion?", "what"),
        ("Define machine learning", "what"),
        ("Explain how DNS works", "how"),
        ("In Python, how can I reverse a string?", "how"),
        ("Why does it rain?", "why"),
        ("What causes rain?", "why"),
        ("How many moons does Mars have?", "quantity"),
        ("How long is a marathon in kilometres?", "quantity"),
        ("What is the distance of a marathon in km?", "quantity"),
        ("Where is Java?", "where"),
        ("Who invented the telephone?", "who"),
        ("When was the Eiffel Tower built?", "when"),
        ("In what year did the Berlin Wall fall?", "when"),
        ("Is it safe to mix bleach and vinegar?", "yesno"),
        ("Can you run Docker on Windows?", "yesno"),
        ("What is the difference between a list and a tuple?", "compare"),
        ("How do lists and tuples differ?", "compare"),
        ("Calculate 15 percent of 200", None),
    ])
    def test_classification(self, query, kind):
        assert question_type(query) == kind

    def test_an_unclassifiable_question_agrees_with_anything(self):
        """None is a wildcard, so the check can only refuse a pair whose kinds
        are both known and different. Losing reuse is the worse failure here."""
        assert question_types_agree("Calculate 15 percent of 200", "What is 15% of 200?")

    def test_two_known_and_different_kinds_disagree(self):
        assert not question_types_agree("What is Java?", "Where is Java?")


class TestKeyText:
    def test_courtesy_is_not_subject_matter(self):
        key = cache_key_text("Hello, could you please explain what recursion is? Thanks!")
        assert "recursion" in key
        for noise in ("hello", "please", "thanks", "explain"):
            assert noise not in key

    def test_a_plural_folds_onto_its_singular(self):
        assert content_words(cache_key_text("Explain how hash tables work")) == \
               content_words(cache_key_text("How does a hash table work?"))

    @pytest.mark.parametrize("word,expected", [
        ("tables", "table"), ("classes", "class"), ("boxes", "box"),
        ("causes", "cause"), ("benefits", "benefit"), ("ies", "ies"),
        ("analysis", "analysis"), ("status", "status"), ("physics", "physics"),
        ("this", "this"), ("does", "does"),
    ])
    def test_singularise(self, word, expected):
        assert singularise(word) == expected

    def test_meaning_bearing_words_survive(self):
        """Normalisation must never remove what the verifier relies on."""
        assert "not" in cache_key_text("Can you not run Docker on Windows?")
        assert "100" in cache_key_text("Convert 100 km to miles")
        assert "minimum" in cache_key_text("What is the minimum wage in Kerala?")

    def test_a_question_made_only_of_courtesy_still_keys_on_something(self):
        assert cache_key_text("Thanks!").strip()


class TestTheLedgerStaysFreeOfText:
    def test_normalised_text_never_reaches_the_ledger(self, make_pipeline):
        """Evidence rides on LedgerRow. The key text is display and comparison
        material, not something to persist for every request."""
        outcome = make_pipeline(full_stack()).run(
            "Hello, could you please explain what recursion is? Thanks!"
        )
        assert "recursion" not in repr(outcome.row)


class TestTheBaselineArmIsUntouched:
    def test_with_the_verifier_off_it_stays_a_pure_threshold_cache(self, embedder):
        """The kind-of-question guard is part of verification. Switching the
        verifier off must leave the literature's single-threshold design intact,
        because that is the comparison the calibration exists to make."""
        assert _lookup_hits("Where is Java?", "What is Java?", embedder,
                            full_stack(), verifier_on=False)
        assert not _lookup_hits("Where is Java?", "What is Java?", embedder,
                                full_stack(), verifier_on=True)
