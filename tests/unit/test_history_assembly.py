"""M3 history management and M4 prefix-stable assembly, including the
interaction between them that Gap 4 is about."""

from __future__ import annotations

from dataclasses import replace

import pytest

from parsimony.core.config import full_stack
from parsimony.core.proposals import ContextPatch, NoOp, TransformKind
from parsimony.core.types import Turn
from parsimony.modules.m3_history import (
    HistoryArranger,
    HistorySelector,
    MmrStrategy,
    RecencyStrategy,
    RelevanceStrategy,
    SummaryStrategy,
)
from parsimony.modules.m4_assembler import assemble_prefix_stable, assemble_volatile_head


def _history(n: int = 6) -> tuple[Turn, ...]:
    topics = [
        "Hash tables store key value pairs using a hash function.",
        "The average lookup complexity is O(1).",
        "Sourdough bread needs a starter culture.",
        "Collisions are resolved by chaining or open addressing.",
        "The weather in Chennai is hot in May.",
        "Load factor determines when a table resizes.",
    ]
    return tuple(
        Turn(turn_id=f"t{i}", role="user" if i % 2 == 0 else "assistant", content=topics[i])
        for i in range(n)
    )


class TestSelectionStrategies:
    def test_recency_keeps_the_most_recent(self, pipeline):
        cfg = replace(full_stack(), history=replace(full_stack().history, max_turns=2))
        ctx = pipeline.build_context("anything", _history())
        keep = RecencyStrategy().select(ctx.history, None, None, cfg, ctx.derived.token_count)
        assert keep == [4, 5]

    def test_relevance_prefers_on_topic_turns(self, pipeline):
        cfg = replace(full_stack(), history=replace(full_stack().history, max_turns=2))
        ctx = pipeline.build_context("Tell me about hash table collisions", _history())
        both = ctx.derived.embed([ctx.query] + [t.content for t in ctx.history])
        keep = RelevanceStrategy().select(
            ctx.history, both[0], both[1:], cfg, ctx.derived.token_count
        )
        kept = " ".join(ctx.history[i].content for i in keep).lower()
        assert "sourdough" not in kept and "weather" not in kept

    def test_mmr_avoids_returning_near_duplicates(self, pipeline):
        cfg = replace(full_stack(), history=replace(full_stack().history, max_turns=3))
        ctx = pipeline.build_context("hash tables", _history())
        both = ctx.derived.embed([ctx.query] + [t.content for t in ctx.history])
        keep = MmrStrategy().select(ctx.history, both[0], both[1:], cfg, ctx.derived.token_count)
        assert len(keep) == len(set(keep)) == 3

    def test_strategies_respect_max_turns(self, pipeline):
        cfg = replace(full_stack(), history=replace(full_stack().history, max_turns=2))
        ctx = pipeline.build_context("q", _history())
        both = ctx.derived.embed([ctx.query] + [t.content for t in ctx.history])
        for strategy in (RecencyStrategy(), RelevanceStrategy(), MmrStrategy()):
            keep = strategy.select(ctx.history, both[0], both[1:], cfg, ctx.derived.token_count)
            assert len(keep) <= 2, strategy.name

    def test_selection_output_is_in_chronological_order(self, pipeline):
        """Selection decides WHICH turns survive; ordering is the arranger's job
        and is measured separately."""
        cfg = full_stack()
        ctx = pipeline.build_context("hash tables", _history())
        both = ctx.derived.embed([ctx.query] + [t.content for t in ctx.history])
        keep = RelevanceStrategy().select(
            ctx.history, both[0], both[1:], cfg, ctx.derived.token_count
        )
        assert keep == sorted(keep)


class TestSelectorStage:
    def test_emits_select_so_the_gate_allows_dropping_turns(self, pipeline):
        cfg = replace(full_stack(), history=replace(full_stack().history, max_turns=2))
        ctx = pipeline.build_context("hash tables", _history())
        proposal = HistorySelector().propose(ctx, cfg)
        assert isinstance(proposal, ContextPatch)
        assert proposal.kind is TransformKind.SELECT

    def test_noop_when_everything_already_fits(self, pipeline):
        cfg = replace(full_stack(), history=replace(full_stack().history, max_turns=50))
        ctx = pipeline.build_context("q", _history(2))
        assert isinstance(HistorySelector().propose(ctx, cfg), NoOp)

    def test_summary_strategy_rewrites_rather_than_selects(self, pipeline):
        """Condensation changes content, so it must face the full invariant
        check rather than the retained-units check (ADR-003)."""
        cfg = replace(full_stack(), history=replace(full_stack().history, strategy="summary"))
        long_turns = tuple(
            Turn(f"t{i}", "user", f"Topic {i} matters. It has several aspects. Aspect one is size.")
            for i in range(4)
        )
        ctx = pipeline.build_context("tell me about topic 0", long_turns)
        proposal = HistorySelector().propose(ctx, cfg)
        if isinstance(proposal, ContextPatch):
            assert proposal.kind is TransformKind.REWRITE


class TestArranger:
    def test_moves_a_relevant_turn_next_to_the_query(self, pipeline):
        """Asserts topicality, not which of two on-topic turns wins.

        This previously pinned the "Collisions are resolved by…" turn
        specifically. Under hashing-v1 that turn led by 0.206 to 0.186 — a
        0.02 margin, i.e. a near-tie — and content-v1 reverses it, preferring
        the turn matching two of the query's three content words (hash, table)
        over the one matching one (collision). Both orderings are defensible,
        so pinning the winner was testing a coin flip. What must hold is that
        an on-topic turn lands adjacent to the query and an off-topic one does
        not.
        """
        cfg = full_stack()
        ctx = pipeline.build_context("Tell me about hash table collisions", _history())
        proposal = HistoryArranger().propose(ctx, cfg)
        assert isinstance(proposal, ContextPatch)

        last = proposal.fields["history"][-1].content.lower()
        assert any(w in last for w in ("hash", "collision", "table")), last

    def test_does_not_move_an_off_topic_turn_next_to_the_query(self, pipeline):
        ctx = pipeline.build_context("Tell me about hash table collisions", _history())
        proposal = HistoryArranger().propose(ctx, full_stack())
        last = proposal.fields["history"][-1].content.lower()
        assert "sourdough" not in last and "weather" not in last

    def test_preserves_every_turn(self, pipeline):
        ctx = pipeline.build_context("hash table collisions", _history())
        proposal = HistoryArranger().propose(ctx, full_stack())
        assert isinstance(proposal, ContextPatch)
        assert sorted(t.turn_id for t in proposal.fields["history"]) == sorted(
            t.turn_id for t in ctx.history
        )

    def test_is_inert_on_short_history(self, pipeline):
        ctx = pipeline.build_context("q", _history(1))
        assert isinstance(HistoryArranger().propose(ctx, full_stack()), NoOp)

    def test_is_inert_when_arrangement_is_chronological(self, pipeline):
        cfg = replace(
            full_stack(), history=replace(full_stack().history, arrangement="chronological")
        )
        ctx = pipeline.build_context("q", _history())
        assert not HistoryArranger().applies_to(ctx, cfg)


class TestAssembly:
    def test_prefix_stable_puts_the_invariant_zone_first(self, pipeline):
        ctx = replace(pipeline.build_context("q", _history(2)), context_digest="User is a student.")
        prompt = assemble_prefix_stable(ctx, ctx.derived.token_count)
        assert prompt.full_text.startswith(prompt.invariant_zone)
        assert prompt.prefix_token_count > 0

    def test_invariant_zone_is_byte_identical_across_turns(self, pipeline):
        a = assemble_prefix_stable(
            pipeline.build_context("first", _history(2)), pipeline.tokenizer.count
        )
        b = assemble_prefix_stable(
            pipeline.build_context("second", _history(4)), pipeline.tokenizer.count
        )
        assert a.invariant_zone == b.invariant_zone

    def test_volatile_head_has_no_invariant_zone(self, pipeline):
        ctx = pipeline.build_context("q", _history(2))
        assert assemble_volatile_head(ctx, ctx.derived.token_count).prefix_token_count == 0

    def test_volatile_head_changes_at_position_zero_every_turn(self, pipeline):
        a = assemble_volatile_head(pipeline.build_context("q", _history(2)), pipeline.tokenizer.count)
        b = assemble_volatile_head(
            replace(pipeline.build_context("q", _history(4)), turn_index=1), pipeline.tokenizer.count
        )
        assert a.full_text.split("\n")[0] != b.full_text.split("\n")[0]


def _run_conversation(pipeline, questions):
    history: list[Turn] = []
    survived: list[int] = []
    for i, q in enumerate(questions):
        out = pipeline.run(q, tuple(history), conversation_id="c1", turn_index=i)
        history += [
            Turn(f"u{i}", "user", q),
            Turn(f"a{i}", "assistant", out.response),
        ]
        survived.append(out.row.prefix_tokens_survived or 0)
    return survived


QUESTIONS = [
    "Explain what a hash table is.",
    "What is its average lookup complexity?",
    "And in the worst case?",
    "Give me an example.",
    "What about collisions?",
]


class TestPrefixSurvival:
    def test_m4_dramatically_increases_prefix_reuse(self, make_pipeline):
        base = full_stack()
        with_m4 = sum(_run_conversation(make_pipeline(base), QUESTIONS))
        without = sum(
            _run_conversation(
                make_pipeline(base.with_modules(base.enabled_modules - {"M4"})), QUESTIONS
            )
        )
        assert with_m4 > without * 3

    def test_position_aware_placement_costs_prefix_reuse(self, make_pipeline):
        """Gap 4 / Contribution 3, as a regression test.

        Chronological and position-aware retain the SAME turns, so they send an
        identical number of tokens. Only the order differs. A token counter
        scores them identically; prefix reuse does not — reordering rewrites the
        volatile zone immediately behind the invariant one.

        This is the conflict the report predicts between prompt optimisation and
        KV reuse, and it is invisible to every metric in the literature.
        """
        base = full_stack()
        chrono_cfg = replace(base, history=replace(base.history, arrangement="chronological"))
        position = sum(_run_conversation(make_pipeline(base), QUESTIONS))
        chronological = sum(_run_conversation(make_pipeline(chrono_cfg), QUESTIONS))
        assert chronological > position

    def test_the_two_arrangements_send_the_same_tokens(self, make_pipeline):
        """The other half of the finding: the cost is invisible to token counts."""
        base = full_stack()
        chrono_cfg = replace(base, history=replace(base.history, arrangement="chronological"))

        def total_in(cfg):
            p = make_pipeline(cfg)
            history: list[Turn] = []
            total = 0
            for i, q in enumerate(QUESTIONS):
                out = p.run(q, tuple(history), conversation_id="c1", turn_index=i)
                history += [Turn(f"u{i}", "user", q), Turn(f"a{i}", "assistant", out.response)]
                total += out.row.tokens_in_final
            return total

        assert total_in(base) == pytest.approx(total_in(chrono_cfg), rel=0.02)
