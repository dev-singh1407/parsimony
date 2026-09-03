"""M7's transfer, measured rather than asserted.

The project is titled "a stacked, self-improving optimisation layer". Until
this module existed, "self-improving" rested on code rather than on a number.
"""

from __future__ import annotations

import pytest

from parsimony.core.config import full_stack
from parsimony.eval.corpus import load_corpus
from parsimony.eval.learning import (
    deployment_trace,
    recurrence_rate,
    run_learning_study,
    stratified_split,
)


@pytest.fixture(scope="module")
def corpus():
    return load_corpus()


class TestSplitIsHonest:
    def test_halves_are_disjoint(self, corpus):
        """The whole measurement is void if a conversation is in both halves:
        the bundle would contain the answers it is then tested on."""
        mine, held = stratified_split(corpus)
        assert not ({c.conversation_id for c in mine.conversations}
                    & {c.conversation_id for c in held.conversations})

    def test_nothing_is_lost(self, corpus):
        mine, held = stratified_split(corpus)
        assert len(mine) + len(held) == len(corpus)

    def test_both_halves_carry_every_class(self, corpus):
        mine, held = stratified_split(corpus)
        assert set(mine.by_class()) == set(held.by_class()) == set(corpus.by_class())

    def test_is_deterministic(self, corpus):
        """A bundle mined from a different random half is a different
        experiment; the result must be re-derivable from the corpus alone."""
        a, _ = stratified_split(corpus)
        b, _ = stratified_split(corpus)
        assert [c.conversation_id for c in a.conversations] == \
               [c.conversation_id for c in b.conversations]

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.5, 2.0])
    def test_refuses_a_degenerate_fraction(self, corpus, bad):
        with pytest.raises(ValueError):
            stratified_split(corpus, fraction=bad)


class TestRecurrenceAxisMeansWhatItSays:
    """The first version of the trace generator sampled the tail WITH
    replacement, so the birthday paradox manufactured repeats on its own: a
    trace built at recurrence=0 measured 25% actual recurrence. The x-axis was
    detached from the thing it claimed to vary, which would have made the
    reported curve meaningless."""

    def test_zero_really_means_zero(self, corpus):
        assert recurrence_rate(deployment_trace(corpus, recurrence=0.0)) == 0.0

    def test_axis_is_monotonic(self, corpus):
        rates = [
            recurrence_rate(deployment_trace(corpus, recurrence=r))
            for r in (0.0, 0.2, 0.4, 0.6, 0.8)
        ]
        assert rates == sorted(rates)
        assert rates[-1] > rates[0]

    def test_the_real_corpus_has_almost_no_recurrence(self, corpus):
        """1.9%. The corpus was authored for ablation diversity, which is the
        right shape for M1/M2/M3/M5 and the wrong shape for a module that
        learns from repetition. That is why M7 shows nothing in the headline
        ablation — a property of the corpus, not of the module."""
        assert recurrence_rate(corpus) < 5.0

    def test_trace_is_deterministic(self, corpus):
        a = deployment_trace(corpus, recurrence=0.4)
        b = deployment_trace(corpus, recurrence=0.4)
        assert [c.user_turns for c in a.conversations] == \
               [c.user_turns for c in b.conversations]

    def test_trace_uses_only_real_corpus_questions(self, corpus):
        """Synthetic in its repetition structure, never in its content."""
        real = {q for c in corpus.conversations for q in c.user_turns}
        trace = deployment_trace(corpus, recurrence=0.5)
        assert {q for c in trace.conversations for q in c.user_turns} <= real

    @pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
    def test_refuses_an_impossible_recurrence(self, corpus, bad):
        with pytest.raises(ValueError):
            deployment_trace(corpus, recurrence=bad)


class TestTransfer:
    def test_no_recurrence_gives_exactly_no_transfer(self, corpus):
        """The null that makes the positive results credible. With nothing
        repeated there is nothing to mine, and the bundle must buy nothing."""
        trace = deployment_trace(corpus, recurrence=0.0, n_conversations=40)
        study = run_learning_study(trace, full_stack())
        assert study.extra_cache_hits == 0
        assert study.transfer_pp == pytest.approx(0.0, abs=0.01)
        assert study.verdict == "no transfer"

    def test_repetitive_traffic_transfers(self, corpus):
        """A concentrated hot set — the FAQ shape M7 targets. With the default
        hot set (38 questions) and only 40 conversations the two halves rarely
        share a question at all, so the test measured sampling luck rather than
        transfer."""
        trace = deployment_trace(
            corpus, recurrence=0.7, n_conversations=60, hot_fraction=0.03
        )
        study = run_learning_study(trace, full_stack())
        assert study.extra_cache_hits > 0
        assert study.transfer_pp > 0
        assert study.verdict == "transfers"

    def test_warm_start_never_costs_fidelity(self, corpus):
        """A seeded cache can serve an answer mined from a different question.
        Buying tokens by serving wrong answers is not a win, and the gate is
        where that would show first."""
        trace = deployment_trace(
            corpus, recurrence=0.7, n_conversations=60, hot_fraction=0.03
        )
        study = run_learning_study(trace, full_stack())
        assert study.extra_gate_fires <= 0

    def test_both_arms_see_identical_held_out_work(self, corpus):
        """The delta is attributable to the bundle only if nothing else differs."""
        trace = deployment_trace(corpus, recurrence=0.5, n_conversations=40)
        study = run_learning_study(trace, full_stack())
        assert study.cold.n_requests == study.warm.n_requests
        assert study.cold.tokens_in_baseline == study.warm.tokens_in_baseline
