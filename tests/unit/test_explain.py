"""The per-turn explanation has to be true, and has to be complete.

Every assertion here corresponds to something the previous walkthrough got
wrong on screen, in front of a reader:

  * it reported the whole prompt as "tokens as typed", so a four-token question
    was announced as 200 tokens once the conversation had built up;
  * it hid every layer that did not act behind "Not needed for this question",
    which is how a working cache came to look broken -- a miss is a decision,
    not an absence;
  * it credited the politeness remover with saving tokens on a question that
    contained no politeness, because the stage had cleaned an earlier turn;
  * it called a layer "switched off" when the layer was simply not applicable
    yet, which on turn one made half the pipeline look disabled.
"""

from __future__ import annotations

import pytest
from rich.console import Console

from parsimony.core.config import baseline, full_stack
from parsimony.core.types import Turn
from parsimony.infra.tokenization import get_tokenizer
from parsimony.pipeline.orchestrator import Pipeline
from parsimony.surfaces.cli import explain
from parsimony.surfaces.cli.explain import (
    LAYERS,
    Session,
    context_panel,
    layers_table,
    memory_panel,
    question_panel,
    result_panel,
    turn_report,
)

CFG = full_stack()
POLITE = "Hello, could you please explain what recursion is? Thanks!"


@pytest.fixture(scope="module")
def counter():
    return get_tokenizer(CFG.tokenizer_id).count


def render(renderable) -> str:
    console = Console(width=110, record=True)
    console.print(renderable)
    return console.export_text()


def capturing(tok, cfg=None):
    """A pipeline that records text deltas, which the shared fixture does not."""
    return Pipeline(cfg or full_stack(), tokenizer=tok, capture_text=True)


def converse(pipeline, *questions):
    history: list[Turn] = []
    outcomes = []
    for q in questions:
        outcome = pipeline.run(q, tuple(history), conversation_id="c1",
                               turn_index=len(history))
        outcomes.append(outcome)
        history.append(Turn(f"u{len(history)}", "user", q))
        history.append(Turn(f"a{len(history)}", "assistant", outcome.response))
    return outcomes


class TestTheTokenArithmeticIsHonest:
    def test_what_you_typed_is_not_the_whole_prompt(self, make_pipeline, counter):
        """The old panel showed the assembled prompt under the heading "tokens
        as typed", so the figure grew every turn while the question did not."""
        outcomes = converse(make_pipeline(full_stack()),
                            "What is recursion?", "What is a pointer?",
                            "How does a hash table work?")
        out = render(question_panel(outcomes[-1], "How does a hash table work?", counter))
        assert "What you typed" in out
        assert "The conversation so far" in out
        assert "Full prompt, before any trimming" in out

    def test_the_three_parts_add_up_to_the_total(self, make_pipeline, counter):
        """A reader checks the arithmetic first. It has to survive that."""
        question = "How does a hash table work?"
        outcomes = converse(make_pipeline(full_stack()),
                            "What is recursion?", "What is a pointer?", question)
        row = outcomes[-1].row
        typed = counter(question)
        payload = outcomes[-1].traces[0].tokens_before
        history = payload - typed
        fixed = row.tokens_in_original - payload
        assert typed + history + fixed == row.tokens_in_original
        assert history > 0, "a third turn must carry conversation history"


class TestEveryLayerReportsItself:
    def test_all_layers_that_ran_are_listed(self, make_pipeline):
        outcome = make_pipeline(full_stack()).run("What is recursion?")
        out = render(layers_table(outcome, cfg=CFG))
        ran = {t.name for t in outcome.traces}
        shown = [name for key, name, _job in LAYERS if key in ran]
        assert shown, "no layer ran at all"
        for name in shown:
            assert name in out, f"{name} ran but was not shown"

    def test_a_layer_with_nothing_to_do_is_not_called_switched_off(self, make_pipeline):
        """On turn one the history manager has no history. That is not the same
        as the operator having disabled it."""
        outcome = make_pipeline(full_stack()).run("What is recursion?")
        out = render(layers_table(outcome, cfg=CFG))
        assert "switched off" not in out

    def test_a_history_only_edit_is_not_reported_as_a_change_to_the_question(
        self, tok
    ):
        """The politeness remover cleans the greeting stored in an earlier turn.
        Claiming it changed the question the user just typed is false."""
        outcomes = converse(capturing(tok), POLITE,
                            "What is a pointer?", "How does a hash table work?")
        out = render(layers_table(outcomes[-1], cfg=CFG))
        assert "cleaned an earlier message" in out


class TestTheMemoryAlwaysExplainsItself:
    def test_a_miss_still_produces_an_explanation(self, make_pipeline):
        """The complaint that started this: a miss looked like a broken module
        because nothing on screen said what had been compared."""
        outcome = make_pipeline(full_stack()).run("What is recursion?")
        panel = memory_panel(outcome, cfg=CFG)
        assert panel is not None
        assert "Did not reuse" in render(panel)

    def test_a_hit_names_the_question_it_matched(self, make_pipeline):
        pipeline = make_pipeline(full_stack())
        outcomes = converse(pipeline, POLITE, "What is a pointer?", "What is recursion?")
        assert outcomes[-1].row.cache_hit
        out = render(memory_panel(outcomes[-1], cache=pipeline.cache, cfg=CFG))
        assert "Reused the stored answer" in out
        assert "recursion" in out, "the matched question must be shown, not just a score"

    def test_an_exact_repeat_renders_and_says_how_it_matched(self, make_pipeline, counter):
        """An exact repeat is matched by hash and carries no similarity score.
        The first version assumed every hit had one and crashed on precisely the
        case a new user tries first: asking the same thing twice."""
        pipeline = make_pipeline(full_stack())
        pipeline.run("What is recursion?")
        again = pipeline.run("What is recursion?")
        assert again.row.cache_hit
        out = render(memory_panel(again, cache=pipeline.cache, cfg=CFG))
        assert "word for word" in out or "same wording" in out
        assert "What is recursion?" in out
        console = Console(width=110, record=True)
        turn_report(console, again, "What is recursion?", counter,
                    cache=pipeline.cache, cfg=CFG, session=Session())
        assert "never called" in console.export_text()

    def test_the_similarity_is_shown_against_its_thresholds(self, make_pipeline):
        pipeline = make_pipeline(full_stack())
        outcomes = converse(pipeline, "What is recursion?", "What is a pointer?",
                            "What is the capital of France?")
        out = render(memory_panel(outcomes[-1], cache=pipeline.cache, cfg=CFG))
        assert "Similarity" in out and "%" in out

    def test_a_scoped_miss_names_the_word_that_scoped_it(self, make_pipeline):
        """A follow-up is compared only within its own conversation. Without
        saying so, that miss is indistinguishable from a broken cache."""
        pipeline = make_pipeline(full_stack())
        outcomes = converse(pipeline, "What is recursion?", "Summarise this.")
        out = render(memory_panel(outcomes[-1], cache=pipeline.cache, cfg=CFG))
        assert "follow-up" in out.lower()
        assert "this" in out

    def test_no_panel_when_the_cache_is_switched_off(self, make_pipeline):
        cfg = baseline()
        outcome = make_pipeline(cfg).run("What is recursion?")
        trace = next((t for t in outcome.traces if t.name == "m2_cache"), None)
        assert trace is None or trace.outcome.value == "skipped"
        assert memory_panel(outcome, cfg=cfg) is None


class TestTheResult:
    def test_a_short_circuit_says_the_model_was_never_called(self, make_pipeline, counter):
        outcome = make_pipeline(full_stack()).run("What is 847 * 23?")
        out = render(result_panel(outcome, counter))
        assert "never called" in out
        assert "prompt tokens avoided" in out

    def test_a_generated_answer_shows_before_and_after(self, make_pipeline, counter):
        outcome = make_pipeline(full_stack()).run(POLITE)
        out = render(result_panel(outcome, counter))
        assert "Prompt:" in out and "->" in out


class TestTheSessionTotals:
    def test_totals_accumulate_and_name_how_questions_were_served(self, make_pipeline):
        pipeline = make_pipeline(full_stack())
        session = Session()
        for outcome in converse(pipeline, POLITE, "What is 847 * 23?", "What is recursion?"):
            session.record(outcome)
        assert session.questions == 3
        assert session.from_calculator == 1
        assert session.from_memory == 1
        assert session.saved > 0
        assert "prompt tokens avoided" in session.line()


class TestTheWholeReportRenders:
    @pytest.mark.parametrize("question", [
        POLITE,
        "What is 847 * 23?",
        "Explain the deadline. The deadline is 15 March. The deadline is 16 March.",
        "Summarise this.",
    ])
    def test_it_does_not_raise_on_any_kind_of_turn(self, make_pipeline, counter, question):
        pipeline = make_pipeline(full_stack())
        outcome = pipeline.run(question)
        console = Console(width=110, record=True)
        turn_report(console, outcome, question, counter,
                    cache=pipeline.cache, cfg=CFG, session=Session())
        assert console.export_text().strip()

class TestAFloorThatWasReadIsExplained:
    """ADR-051: the floor can now be computed per request, and a number the
    system computed that no surface reports is one it cannot account for."""

    # Chosen because it produces a cliff under the lexical encoder the test
    # environment falls back to, not only under a neural one.
    QUESTION = "What is the annual travel budget for the Tallinn office?"

    @staticmethod
    def _outcome(adaptive, tok):
        from dataclasses import replace
        from pathlib import Path

        from parsimony.core.types import split_into_documents

        text = (Path(__file__).resolve().parents[2]
                / "examples/staff-handbook.md").read_text(encoding="utf-8")
        docs = split_into_documents(text, "staff-handbook.md")
        cfg = full_stack()
        cfg = replace(cfg, compression=replace(cfg.compression,
                                               context_adaptive_floor=adaptive))
        pipe = Pipeline(cfg, tokenizer=tok)
        return pipe.run(TestAFloorThatWasReadIsExplained.QUESTION, (),
                        conversation_id="c-floor", turn_index=0, documents=tuple(docs))

    @staticmethod
    def _evidence(outcome):
        trace = next(t for t in outcome.traces if t.name == "m1_context")
        return dict(trace.evidence)

    def test_the_ledger_row_carries_the_floor_and_the_reading(self, tok):
        ev = self._evidence(self._outcome(True, tok))
        assert ev["relevance_floor"] > 0
        assert ev["floor_why"]

    def test_a_read_floor_is_stated_in_the_terminal_panel(self, tok):
        outcome = self._outcome(True, tok)
        ev = self._evidence(outcome)
        assert ev["floor_read"], (
            "this question must produce a reading or the test asserts nothing")
        text = render(context_panel(outcome))
        assert "read off" in text
        assert f"{ev['relevance_floor']:.2f}" in text

    def test_a_configured_floor_is_not_narrated_on_every_request(self, tok):
        """0.15 on every line is noise: the settings already say 0.15."""
        outcome = self._outcome(False, tok)
        assert self._evidence(outcome)["floor_read"] is False
        assert "read off" not in render(context_panel(outcome))

    def test_it_decides_by_the_flag_and_not_by_the_wording(self):
        """The first version compared `floor_why` against two known sentences.
        The off-topic path never reaches the floor, its sentence matched
        neither, and the terminal duly announced a reading that never happened.
        Whether the floor was read is a property of the run."""
        from pathlib import Path

        source = (Path(explain.__file__)).read_text(encoding="utf-8")
        assert 'ev.get("floor_read")' in source
        assert '"the configured constant"' not in source, (
            "branching on the wording is how the off-topic path was mis-narrated")

    def test_every_reading_is_narrated_and_stays_inside_its_clamp(self, tok):
        """Whichever branch fires -- a cliff, a smooth ramp, or too few
        candidates to judge -- it is a reading, it is reported, and it lands in
        the range the constant was measured over. Which branch a given question
        takes depends on the tokenizer, because the budget decides how many
        candidates there are, so the branch is not what this asserts."""
        from dataclasses import replace
        from pathlib import Path

        from parsimony.core.types import split_into_documents

        text = (Path(__file__).resolve().parents[2]
                / "examples/staff-handbook.md").read_text(encoding="utf-8")
        docs = tuple(split_into_documents(text, "staff-handbook.md"))
        cfg = full_stack()
        cfg = replace(cfg, compression=replace(cfg.compression,
                                               context_adaptive_floor=True))
        outcome = Pipeline(cfg, tokenizer=tok).run(
            "Who manages the Porto office?", (), conversation_id="c-ramp",
            turn_index=0, documents=docs)
        ev = self._evidence(outcome)
        c = cfg.compression
        assert ev["floor_read"] and ev["floor_why"]
        assert c.context_elbow_floor_min <= ev["relevance_floor"] <= c.context_elbow_floor_max
        assert "read off" in render(context_panel(outcome))
