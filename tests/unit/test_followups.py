"""History management, measured on conversations rather than on token counts.

M3 drops turns and M1's context tier shortens the ones that survive. Until
`corpus/followups.jsonl` there was no case where dropping the wrong turn makes
the last question unanswerable, so both modules were only ever scored on how
many tokens they removed -- a metric that rewards removing the answer.

Every test here is about the fact the final question needs: is it still in the
prompt, and does the arm that loses it lose it for the reason claimed.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from parsimony.core.config import full_stack
from parsimony.eval.followups import (
    ARM_LABELS,
    arms,
    evidence_survives,
    load_followups,
    summarise,
)
from parsimony.infra.providers import MockProvider
from parsimony.pipeline.orchestrator import Pipeline

ITEMS = load_followups()
DEV = [i for i in ITEMS if i.split == "dev"]
TEST = [i for i in ITEMS if i.split == "test"]
#: Conversations whose fact sits inside a LONG earlier answer, so sentence
#: compression has something to do (ADR-043).
LONG = [i for i in ITEMS if i.split.endswith("_long")]


def prompt_for(item, cfg, tok):
    pipe = Pipeline(cfg, provider=MockProvider(), tokenizer=tok)
    return pipe.run(item.question, item.turns, conversation_id=item.conversation_id)


class TestTheCorpus:
    def test_shape(self):
        assert len(ITEMS) == 30
        assert len(DEV) == 6 and len(TEST) == 14
        assert len(LONG) == 10
        assert sum(i.split == "test_long" for i in ITEMS) == 7

    def test_every_conversation_buries_its_fact_behind_other_turns(self):
        for item in ITEMS:
            assert item.evidence_turn in (0, 1)
            assert len(item.turns) >= 8, item.conversation_id
            # The turns after the fact are unrelated: none of them contains the
            # answer, so a system that keeps only recent turns cannot be right
            # by accident.
            later = " ".join(t.content for t in item.turns[item.evidence_turn + 1:]).lower()
            answer = item.gold.gold_answer.lower()
            if answer.isalpha() and len(answer) >= 3:
                assert answer not in later, item.conversation_id

    def test_the_question_is_not_answerable_from_its_own_words(self):
        for item in ITEMS:
            assert item.gold.gold_answer.lower() not in item.question.lower()

    def test_both_roles_are_present_so_the_history_is_realistic(self):
        for item in ITEMS:
            roles = {t.role for t in item.turns}
            assert roles == {"user", "assistant"}, item.conversation_id


class TestWhatEachStrategyKeeps:
    def test_keeping_the_last_few_turns_loses_every_fact(self, tok):
        """The industry default, on the case it is worst at."""
        kept = sum(evidence_survives(i, prompt_for(i, arms()["keep_last_4"], tok).ctx)
                   for i in DEV)
        assert kept == 0

    def test_relevance_keeps_facts_recency_cannot(self, tok):
        relevance = sum(evidence_survives(i, prompt_for(i, arms()["relevance"], tok).ctx)
                        for i in DEV)
        assert relevance >= 4

    def test_sending_everything_always_keeps_it_and_costs_the_most(self, tok):
        everything = arms()["everything"]
        relevance = arms()["relevance"]
        kept = tokens_all = tokens_some = 0
        for item in DEV:
            all_out = prompt_for(item, everything, tok)
            some_out = prompt_for(item, relevance, tok)
            kept += evidence_survives(item, all_out.ctx)
            tokens_all += all_out.row.tokens_in_final
            tokens_some += some_out.row.tokens_in_final
        assert kept == len(DEV)
        assert tokens_some < tokens_all

    def test_the_control_arm_is_a_control(self):
        assert arms()["no_history"] is None
        assert set(ARM_LABELS) == set(arms())


class TestTheGatesForConversations:
    """The context tier's thresholds were set for attached documents."""

    def test_document_gates_never_fire_on_a_chat(self, tok):
        shipped = arms(gates=None)["relevance_sentences"]
        for item in DEV:
            outcome = prompt_for(item, shipped, tok)
            fired = [t for t in outcome.traces
                     if t.name == "m1_context" and t.tokens_after < t.tokens_before]
            assert not fired, "shipped gates should not fire on short turns"

    def test_lower_gates_do_not_lose_the_fact(self, tok):
        tight = arms(gates=(100, 40))["relevance_sentences"]
        for item in DEV:
            assert evidence_survives(item, prompt_for(item, tight, tok).ctx) or \
                not evidence_survives(item, prompt_for(item, arms()["relevance"], tok).ctx)


class TestTheSummary:
    def test_it_pairs_every_arm_against_sending_everything(self):
        rows = [
            {"conversation_id": "a", "split": "test", "arm": "everything", "correct": True,
             "evidence_kept": True, "prompt_tokens": 200, "prefill_ms": 1000.0},
            {"conversation_id": "a", "split": "test", "arm": "keep_last_4", "correct": False,
             "evidence_kept": False, "prompt_tokens": 90, "prefill_ms": 500.0},
            {"conversation_id": "b", "split": "test", "arm": "everything", "correct": True,
             "evidence_kept": True, "prompt_tokens": 200, "prefill_ms": 1000.0},
            {"conversation_id": "b", "split": "test", "arm": "keep_last_4", "correct": False,
             "evidence_kept": False, "prompt_tokens": 90, "prefill_ms": 500.0},
        ]
        summary = {r["arm"]: r for r in summarise(rows)}
        assert summary["everything"]["correct"] == "2/2"
        assert summary["keep_last_4"]["correct"] == "0/2"
        assert summary["keep_last_4"]["lost vs everything"] == 2
        assert float(summary["keep_last_4"]["McNemar p"]) == pytest.approx(0.5)

    def test_a_split_can_be_selected(self):
        rows = [{"conversation_id": "a", "split": "dev", "arm": "everything", "correct": True,
                 "evidence_kept": True, "prompt_tokens": 1, "prefill_ms": 1.0}]
        assert summarise(rows, split="test") == []
        assert summarise(rows, split="dev")[0]["correct"] == "1/1"


class TestALongTurnIsCompressedRatherThanKeptWhole:
    """Where sentence compression can actually act, and the bug that stopped it.

    M3's position-aware arrangement moves the most relevant turn to the END of
    the list it hands on. The context tier protected "the last two turns", so
    it protected precisely the turn worth compressing: on these ten
    conversations it fired once. Protection now means recent in the
    CONVERSATION, identified by turn id (ADR-043).
    """

    def test_the_tier_fires_on_most_of_them(self, tok):
        from parsimony.core.config import full_stack

        fired = 0
        for item in LONG:
            outcome = prompt_for(item, arms(full_stack())["relevance_sentences"], tok)
            fired += any(t.name == "m1_context" and t.tokens_after < t.tokens_before
                         for t in outcome.traces)
        assert fired >= 7, "a long earlier answer is exactly what this tier is for"

    def test_and_shortens_the_prompt(self, tok):
        from parsimony.core.config import full_stack

        plain = sum(prompt_for(i, arms(full_stack())["relevance"], tok).row.tokens_in_final
                    for i in LONG)
        squeezed = sum(prompt_for(i, arms(full_stack())["relevance_sentences"], tok)
                       .row.tokens_in_final for i in LONG)
        assert squeezed < 0.85 * plain

    def test_the_answer_sentence_is_what_survives(self, tok):
        """Scored on the sentence carrying the answer, not the whole turn --
        removing the rest of the turn is the point."""
        from parsimony.core.config import full_stack

        kept = sum(evidence_survives(i, prompt_for(
            i, arms(full_stack())["relevance_sentences"], tok).ctx) for i in LONG)
        assert kept >= 8

    def test_the_most_recent_exchange_is_still_protected_after_reordering(self, tok):
        """The rule is about the conversation, not about list position."""
        from dataclasses import replace

        from parsimony.core.config import full_stack
        from parsimony.modules.m1_context import eligible_turns

        item = LONG[0]
        cfg = full_stack()
        pipe = Pipeline(cfg, provider=MockProvider(), tokenizer=tok)
        ctx = pipe.build_context(item.question, item.turns)
        reordered = replace(ctx, history=(ctx.history[-1], *ctx.history[:-1]))
        protected = {t.turn_id for t in ctx.original_history[-2:]}
        chosen = {reordered.history[i].turn_id for i in eligible_turns(reordered, cfg)}
        assert not (chosen & protected)
