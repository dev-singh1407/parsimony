"""The dashboard, the audit behind it, and the counters it shows.

Every number on this screen is either measured on the spot or derived from
something the runtime reports, and the ones that are derived must show their
arithmetic. A HUD that mixes the two silently is worse than no HUD, so that is
what these tests pin -- along with the tags, which must name branches of
`m1_context.select` rather than invented ones.
"""

from __future__ import annotations

import io

import pytest
from rich.console import Console

from parsimony.core.config import full_stack
from parsimony.core.types import Document, Turn, split_into_documents
from parsimony.infra.providers import MockProvider, OllamaProvider, kv_bytes_per_token
from parsimony.modules.m1_context import audit
from parsimony.pipeline.orchestrator import Pipeline
from parsimony.surfaces.cli.live import LiveTurn, run_live

TAGS = {"ANCHOR", "MATCH", "CLOSURE", "PROTECTED", "BUDGET", "REDUNDANT", "FLOOR",
        "OFF-TOPIC", "KEPT"}
HANDBOOK = "examples/staff-handbook.md"


@pytest.fixture(scope="module")
def documents():
    from pathlib import Path

    text = (Path(__file__).resolve().parents[2] / HANDBOOK).read_text(encoding="utf-8")
    return split_into_documents(text, "staff-handbook.md")


def context_for(question, documents, tok=None):
    pipe = Pipeline(full_stack(), provider=MockProvider(), tokenizer=tok)
    return pipe.build_context(question, documents=documents), pipe


def rendered(renderable, width=110, height=40) -> str:
    console = Console(file=io.StringIO(), width=width, height=height, force_terminal=False)
    console.print(renderable)
    return console.file.getvalue()


class TestTheAudit:
    def test_every_sentence_carries_a_decision(self, documents, tok):
        ctx, _ = context_for("What is the annual travel budget for the Tallinn office?",
                             documents, tok)
        report = audit(ctx, full_stack())
        assert report.units
        for unit in report.units:
            assert unit.tag in TAGS, unit.tag
            assert unit.detail, "a decision with no reason beside it explains nothing"
            assert 0.0 <= unit.score <= 1.0

    def test_the_kept_set_matches_what_the_stage_would_send(self, documents, tok):
        from parsimony.core.proposals import ContextPatch
        from parsimony.modules.m1_context import ContextCompressor

        question = "How many people does the Tallinn office employ?"
        ctx, _ = context_for(question, documents, tok)
        cfg = full_stack()
        report = audit(ctx, cfg)
        proposal = ContextCompressor().propose(ctx, cfg)
        assert isinstance(proposal, ContextPatch)
        sent = "\n".join(d.content for d in proposal.fields["documents"])
        for unit in report.units:
            if unit.tag == "PROTECTED":
                continue
            assert (unit.text in sent) == unit.kept, unit.text[:40]

    def test_an_anchor_is_named_in_its_reason(self, documents, tok):
        ctx, _ = context_for("How many people does the Tallinn office employ?", documents, tok)
        report = audit(ctx, full_stack())
        anchored = [u for u in report.units if u.tag == "ANCHOR"]
        assert anchored
        assert any("Tallinn" in u.detail for u in anchored)

    def test_an_off_topic_question_says_so_on_every_sentence(self, documents, tok):
        ctx, _ = context_for("What is the capital of Peru?", documents, tok)
        report = audit(ctx, full_stack())
        assert report.off_topic
        tags = {u.tag for u in report.units if u.tag != "PROTECTED"}
        assert tags <= {"OFF-TOPIC", "MATCH"}
        assert sum(u.kept for u in report.units if u.tag != "PROTECTED") == 1

    def test_protected_turns_are_shown_as_protected(self, tok):
        long_turn = " ".join(f"Point {i} concerns the Leeds calibration lab." for i in range(30))
        history = (Turn("t0", "user", "Tell me about Leeds."),
                   Turn("t1", "assistant", long_turn),
                   Turn("t2", "user", "And the lab?"),
                   Turn("t3", "assistant", long_turn))
        pipe = Pipeline(full_stack(), provider=MockProvider(), tokenizer=tok)
        ctx = pipe.build_context("Which point mentions the schedule?", history)
        report = audit(ctx, full_stack())
        assert any(u.tag == "PROTECTED" for u in report.units)
        assert all(u.kept for u in report.units if u.tag == "PROTECTED")


class TestTheMarginalia:
    def test_it_states_a_verdict_and_a_number_for_each_sentence(self, documents, tok):
        from parsimony.surfaces.cli.main import marginalia

        ctx, _ = context_for("What is the annual travel budget for the Tallinn office?",
                             documents, tok)
        text = rendered(marginalia(audit(ctx, full_stack())), width=120)
        assert "[KEEP: " in text and "[DROP: " in text
        assert "relevance 0." in text

    def test_the_tags_are_branches_of_the_selector_not_prose(self):
        from parsimony.surfaces.cli.main import TAG_STYLE

        assert set(TAG_STYLE) <= TAGS


class TestTheHud:
    def test_derived_figures_show_their_arithmetic(self, documents, tok):
        pipe = Pipeline(full_stack(), provider=MockProvider(), tokenizer=tok, capture_text=True)
        console = Console(file=io.StringIO(), width=110, height=40, force_terminal=False)
        outcome, view = run_live(console, pipe, "What is the travel budget for Tallinn?",
                                 documents=documents)
        view.kv_per_token = (28672, "2 x 28 blocks x 2 kv-heads x 128 dims x 2 bytes (fp16)")
        hud = rendered(view.hud_pane(40))
        assert "MB" in hud and "never allocated" in hud
        assert "kv-heads" in hud, "a megabyte figure with no derivation is decoration"

    def test_it_reports_the_encoder_cost_beside_what_it_bought(self, documents, tok):
        pipe = Pipeline(full_stack(), provider=MockProvider(), tokenizer=tok, capture_text=True)
        console = Console(file=io.StringIO(), width=110, height=40, force_terminal=False)
        _, view = run_live(console, pipe, "What is the travel budget for Tallinn?",
                           documents=documents)
        hud = rendered(view.hud_pane(40))
        assert "removed" in hud and "not read" in hud
        assert "selector" in hud or "encoder" in hud

    def test_a_simulated_model_is_never_reported_as_a_rate(self, documents, tok):
        pipe = Pipeline(full_stack(), provider=MockProvider(), tokenizer=tok, capture_text=True)
        console = Console(file=io.StringIO(), width=110, height=40, force_terminal=False)
        _, view = run_live(console, pipe, "What is the travel budget for Tallinn?",
                           documents=documents)
        hud = rendered(view.hud_pane(40))
        assert "tokens/s" not in hud, "the mock has no generation speed to report"

    def test_the_dashboard_renders_at_any_size(self, documents, tok):
        pipe = Pipeline(full_stack(), provider=MockProvider(), tokenizer=tok, capture_text=True)
        console = Console(file=io.StringIO(), width=110, height=40, force_terminal=False)
        _, view = run_live(console, pipe, "What is the travel budget for Tallinn?",
                           documents=documents)
        for width, height in ((80, 24), (110, 40), (200, 60)):
            text = rendered(view.dashboard(height, width), width=width, height=height)
            assert "Measured" in text and "Context" in text


class TestTheKvDerivation:
    def test_it_is_absent_rather_than_guessed_when_the_model_cannot_say(self):
        assert kv_bytes_per_token(MockProvider()) is None

    @pytest.mark.skipif(not OllamaProvider.available(), reason="no model server")
    def test_it_comes_from_the_runtimes_own_metadata(self):
        result = kv_bytes_per_token(OllamaProvider())
        assert result is not None
        per_token, how = result
        # A 1.5B model with 2 kv-heads: tens of kilobytes per token, not megabytes.
        assert 1_000 < per_token < 1_000_000
        assert "blocks" in how and "kv-heads" in how

    def test_the_request_records_what_the_encoder_cost(self, documents, tok):
        pipe = Pipeline(full_stack(), provider=MockProvider(), tokenizer=tok)
        ctx = pipe.build_context("What is the travel budget for Tallinn?", documents=documents)
        audit(ctx, full_stack())
        assert ctx.derived.embed_ns >= 0


class TestWhichConstraintBound:
    """stopped_by must name the BINDING constraint, not the last branch taken.

    The loop sets "budget" when a sentence does not fit and then continues -- a
    smaller one may still fit, which is right -- and used to end on the floor
    break, overwriting the reason. A run that refused 44 sentences for want of
    room reported "relevance floor", and at a 5% budget with the floor at zero
    it still did. The question a reader is asking is which one, given more,
    would change the answer.
    """

    def _audit(self, question, documents, tok, **over):
        from dataclasses import replace
        from parsimony.core.config import full_stack
        from parsimony.infra.providers import MockProvider
        from parsimony.modules.m1_context import audit
        from parsimony.pipeline.orchestrator import Pipeline

        cfg = full_stack()
        if over:
            cfg = replace(cfg, compression=replace(cfg.compression, **over))
        pipe = Pipeline(cfg, provider=MockProvider(), tokenizer=tok)
        return audit(pipe.build_context(question, documents=documents), cfg)

    def test_a_tight_budget_with_a_slack_floor_reports_the_budget(self, documents, tok):
        report = self._audit('What is the annual travel budget for the Tallinn office?',
                             documents, tok,
                             context_relevance_floor=0.0, context_target_ratio=0.05)
        assert any(u.tag == 'BUDGET' for u in report.units)
        assert report.stopped_by == 'budget', (
            'sentences above the floor were refused for room, so room is what binds')

    def test_a_slack_budget_with_a_live_floor_reports_the_floor(self, documents, tok):
        report = self._audit('What is the annual travel budget for the Tallinn office?',
                             documents, tok,
                             context_relevance_floor=0.15, context_target_ratio=0.80)
        assert not any(u.tag == 'BUDGET' for u in report.units)
        assert report.stopped_by == 'relevance floor'

    def test_raising_a_slack_budget_changes_nothing(self, documents, tok):
        """The finding the two dials exist to show (ADR-050)."""
        a = self._audit('What is the annual travel budget for the Tallinn office?',
                        documents, tok, context_target_ratio=0.35)
        b = self._audit('What is the annual travel budget for the Tallinn office?',
                        documents, tok, context_target_ratio=0.80)
        assert a.tokens_after == b.tokens_after
        assert a.stopped_by == b.stopped_by == 'relevance floor'

    def test_lowering_the_binding_floor_does_change_things(self, documents, tok):
        a = self._audit('What is the annual travel budget for the Tallinn office?',
                        documents, tok, context_relevance_floor=0.15)
        b = self._audit('What is the annual travel budget for the Tallinn office?',
                        documents, tok, context_relevance_floor=0.05)
        assert b.tokens_after > a.tokens_after


class TestTheStopReasonIsAClosedSet:
    """It was a sentence travelling across three surfaces as if it were a type.

    Five places consume it and two key dictionaries on the exact phrase: the
    terminal's per-layer line and the web script's "which dial is biting" note.
    `select()` branched on `stopped_by == "budget"` to tag sentences. Rewording
    "relevance floor" would have degraded all four to a default with nothing
    failing, which is the failure these tests exist to make impossible.
    """

    def test_it_still_behaves_as_the_string_it_replaced(self):
        """Subclassing str is the whole compatibility story: ledger rows, JSON
        payloads and interpolated prose all carried on working."""
        import json

        from parsimony.modules.m1_context import StopReason

        assert isinstance(StopReason.FLOOR, str)
        assert f"{StopReason.FLOOR}" == "relevance floor"
        assert str(StopReason.BUDGET) == "budget"
        assert json.loads(json.dumps({"s": StopReason.BUDGET}))["s"] == "budget"
        assert {"budget": 1}[StopReason.BUDGET] == 1
        assert StopReason("budget") is StopReason.BUDGET

    def test_every_reason_is_explained_by_the_terminal(self):
        """`explain.py` maps the reason to a sentence for the layer table. A
        member it does not know falls through to an empty string, silently."""
        import re
        from pathlib import Path

        from parsimony.modules.m1_context import StopReason

        source = (Path(__file__).resolve().parents[2]
                  / "src/parsimony/surfaces/cli/explain.py").read_text(encoding="utf-8")
        block = re.search(r"stopped_by[^\n]*\n?", source)
        assert block, "explain.py no longer translates stopped_by"
        for reason in StopReason:
            assert reason.value in source, (
                f"{reason.name} has no explanation in explain.py")

    def test_every_reason_is_explained_by_the_page(self):
        """The web script's lookup falls back to printing the raw phrase, which
        is not wrong but is not an explanation either."""
        from pathlib import Path

        from parsimony.modules.m1_context import StopReason

        script = (Path(__file__).resolve().parents[2]
                  / "src/parsimony/surfaces/web_app.js").read_text(encoding="utf-8")
        for reason in (StopReason.FLOOR, StopReason.BUDGET, StopReason.EXHAUSTED):
            assert reason.value in script, (
                f"{reason.name} has no explanation on the page")

    def test_the_selector_matches_on_the_type_not_the_words(self):
        from pathlib import Path

        source = (Path(__file__).resolve().parents[2]
                  / "src/parsimony/modules/m1_context.py").read_text(encoding="utf-8")
        assert 'stopped_by == "budget"' not in source
        assert "stopped_by is StopReason.BUDGET" in source

    def test_an_audit_carries_the_type_through(self, documents, tok):
        from parsimony.core.config import full_stack
        from parsimony.infra.providers import MockProvider
        from parsimony.modules.m1_context import StopReason, audit
        from parsimony.pipeline.orchestrator import Pipeline

        pipe = Pipeline(full_stack(), provider=MockProvider(), tokenizer=tok)
        ctx = pipe.build_context("What is the travel budget for Tallinn?",
                                 documents=documents)
        assert audit(ctx, full_stack()).stopped_by in set(StopReason)
