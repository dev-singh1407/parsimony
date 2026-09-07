"""Cache context-chain scoping (ADR-039).

Found by using the demo rather than by reading the code: asking the same
question four times in one conversation never produced a cache hit.
"""

from __future__ import annotations

import pytest

from parsimony.modules.m2_cache import chain_hash, is_context_dependent


class TestChainScopingAppliesOnlyWhereNeeded:
    """The context chain made in-conversation cache hits impossible.

    The chain is a hash of the last `depth` turns, so it changes on every turn.
    Asking an identical question twice in one conversation reported "cache miss
    (no candidates)" — not a low similarity, but no candidate at all, because
    the entry was filed under a chain that no longer existed. Found by using
    the demo: asking the same thing four times never hit (ADR-039).
    """

    @staticmethod
    def _history(n=2):
        from parsimony.core.types import Turn

        return tuple(
            Turn(turn_id=f"t{i}", role="user" if i % 2 == 0 else "assistant",
                 content=f"earlier turn {i}")
            for i in range(n)
        )

    def test_a_self_contained_question_is_not_chain_scoped(self):
        """Its answer does not depend on what came before, so scoping it to a
        conversation buys no safety and costs every hit."""
        assert chain_hash(self._history(), 2, "What is 847 * 23?") == "root"

    def test_a_follow_up_is_chain_scoped(self):
        assert chain_hash(self._history(), 2, "And what about the second one?") != "root"

    def test_a_pronoun_makes_it_dependent(self):
        assert chain_hash(self._history(), 2, "Why does that affect lookup time?") != "root"

    def test_no_history_is_always_root(self):
        assert chain_hash((), 2, "Why does that matter?") == "root"

    def test_omitting_the_query_keeps_the_old_conservative_behaviour(self):
        """Callers that do not pass a query still get the chain, so nothing
        silently loosens."""
        assert chain_hash(self._history(), 2) != "root"

    @pytest.mark.parametrize(
        "query,dependent",
        [
            ("What is 847 * 23?", False),
            ("How do I reverse a string in Python?", False),
            ("Explain recursion.", False),
            ("What is the capital of Australia?", False),
            ("And what about the second one?", True),
            ("Why does that affect lookup time?", True),
            ("Can you explain it again?", True),
            ("So how do I fix them?", True),
            ("Summarise this.", True),
        ],
    )
    def test_dependence_detection(self, query, dependent):
        assert is_context_dependent(query) is dependent

    def test_the_detector_errs_towards_scoping(self):
        """A self-contained question mistaken for a dependent one costs a cache
        hit. The reverse serves a wrong answer. Only one of those is acceptable,
        so ambiguous wording must come back True."""
        for ambiguous in ("Tell me more about it", "What about the other one",
                          "Is that also true", "Do the same again"):
            assert is_context_dependent(ambiguous), ambiguous
