"""What the context selector does when no encoder answers.

The off-topic rule asks two things: is term coverage below the floor, and did
the encoder fail to rescue it anyway. A missing cosine used to answer the second
with "yes, it failed to rescue" regardless of why it was missing -- and it is
missing both when there is no dense term configured at all and when Ollama is
simply down.

The first is the designed rule and must not change. The second is an outage,
where nothing is known, and reading it as a negative verdict discards the whole
document on exactly the questions the cosine arm was added to rescue (ADR-059).

The control in this suite matters more than the fix: a question that is
genuinely off topic must STILL be caught during an outage, because the
zero-coverage rule is lexical and owes nothing to an encoder.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from parsimony.core.config import full_stack
from parsimony.modules.m1_context import encoder_was_expected

HANDBOOK = Path(__file__).resolve().parents[2] / "examples" / "staff-handbook.md"


@pytest.fixture(scope="module")
def handbook():
    return HANDBOOK.read_text(encoding="utf-8")


class _NoEncoder:
    has_embedder = False


class _WithEncoder:
    has_embedder = True


class TestTellingTheTwoCausesApart:
    def test_no_dense_term_configured_is_not_an_outage(self):
        """Coverage alone is the designed rule when there is no dense term."""
        from dataclasses import replace
        cfg = full_stack()
        cfg = replace(cfg, compression=replace(cfg.compression, context_dense_weight=0.0))
        assert not encoder_was_expected(_NoEncoder(), cfg)

    def test_a_dense_term_with_no_encoder_is_an_outage(self):
        cfg = full_stack()
        assert cfg.compression.context_dense_weight > 0, "fixture assumes a dense term"
        assert encoder_was_expected(_NoEncoder(), cfg)

    def test_an_encoder_that_answers_is_never_an_outage(self):
        assert not encoder_was_expected(_WithEncoder(), full_stack())


class TestAnOutageDoesNotDiscardTheDocument:
    """Driven through the visualiser, which is the path the web surface and
    the floor-mode tests both take."""

    @staticmethod
    def _compress(monkeypatch, question, text, *, encoder):
        import parsimony.infra.derived as derived_mod
        from parsimony.surfaces import web

        if not encoder:
            monkeypatch.setattr(derived_mod.DerivedCache, "has_embedder",
                                property(lambda self: False))
        vis = web.Visualiser(provider_name="mock")
        return vis.compress(question, text, "handbook", None, None, False)

    def test_a_question_the_document_can_answer_survives_an_outage(
            self, monkeypatch, handbook):
        q = "When does the office close for Christmas?"
        out = self._compress(monkeypatch, q, handbook, encoder=False)
        assert not out["off_topic"], (
            "the document was discarded because the encoder could not be asked")
        assert sum(1 for u in out["units"] if u["kept"]) > 1

    def test_a_genuinely_off_topic_question_is_still_caught(
            self, monkeypatch, handbook):
        """The control. Zero term coverage is decisive on its own (ADR-052) and
        owes nothing to an encoder, so an outage must not smuggle this through."""
        out = self._compress(monkeypatch, "What is the capital of Peru?",
                             handbook, encoder=False)
        assert out["off_topic"]

    def test_with_the_encoder_present_nothing_changes(self, handbook):
        from parsimony.surfaces import web
        vis = web.Visualiser(provider_name="mock")
        out = vis.compress("When does the office close for Christmas?", handbook,
                           "handbook", None, None, False)
        assert not out["off_topic"]

    def test_the_old_behaviour_is_still_reachable(self, monkeypatch, handbook):
        """Kept measurable rather than remembered, as ADR-041's accept zone was."""
        from dataclasses import replace
        import parsimony.infra.derived as derived_mod
        from parsimony.surfaces import web

        monkeypatch.setattr(derived_mod.DerivedCache, "has_embedder",
                            property(lambda self: False))
        vis = web.Visualiser(provider_name="mock")
        vis.cfg = replace(vis.cfg, compression=replace(
            vis.cfg.compression, context_offtopic_without_encoder=True))
        out = vis.compress("When does the office close for Christmas?", handbook,
                           "handbook", None, None, False)
        assert out["off_topic"]
