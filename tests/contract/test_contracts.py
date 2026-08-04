"""Contract suites.

Each protocol has ONE suite, run against every implementation. This is what
makes "components are replaceable" a checked claim rather than a hope: adding
FaissIndex or OllamaProvider means adding it to a parametrize list, and if it
does not behave like the thing it replaces, the suite says so.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from parsimony.core.config import full_stack
from parsimony.core.ledger import LedgerRow
from parsimony.core.proposals import ContextPatch, NoOp, ShortCircuit
from parsimony.core.types import GenParams
from parsimony.infra.providers import MockProvider
from parsimony.infra.storage import JsonlSink, MemorySink, SqliteSink, read_jsonl
from parsimony.infra.tokenization import HeuristicTokenizer
from parsimony.modules.m2_cache import SemanticCache
from parsimony.pipeline.registry import default_registry


# ------------------------------------------------------------ Stage ---------
def all_stages():
    reg = default_registry(SemanticCache())
    return list(reg._stages.values())


@pytest.mark.parametrize("stage", all_stages(), ids=lambda s: s.name)
class TestStageContract:
    def test_declares_required_attributes(self, stage):
        assert isinstance(stage.module_id, str) and stage.module_id
        assert isinstance(stage.name, str) and stage.name
        assert isinstance(stage.reads, frozenset)
        assert isinstance(stage.writes, frozenset)

    def test_propose_returns_a_valid_proposal(self, stage, pipeline):
        ctx = pipeline.build_context("Please explain the 42 rule. Thanks!")
        result = stage.propose(ctx, full_stack())
        assert isinstance(result, (ContextPatch, ShortCircuit, NoOp))

    def test_propose_does_not_mutate_the_context(self, stage, pipeline):
        """Modules propose; the orchestrator commits (ADR-001)."""
        ctx = pipeline.build_context("Please explain the 42 rule. Thanks!")
        before = (ctx.query, ctx.history, ctx.output_budget, ctx.response_class)
        stage.propose(ctx, full_stack())
        assert (ctx.query, ctx.history, ctx.output_budget, ctx.response_class) == before

    def test_patches_only_touch_declared_writes(self, stage, pipeline):
        """A stage writing an undeclared field breaks DAG validation, and would
        do so silently at hour nine of an unattended sweep."""
        ctx = pipeline.build_context("Please explain the 42 rule. Thanks in advance!")
        result = stage.propose(ctx, full_stack())
        if isinstance(result, ContextPatch):
            assert set(result.fields) <= set(stage.writes)

    def test_is_inert_when_its_module_is_disabled(self, stage, pipeline):
        from parsimony.core.config import baseline

        ctx = pipeline.build_context("What is 2+2?")
        assert not stage.applies_to(ctx, baseline())

    def test_handles_an_empty_query(self, stage, pipeline):
        ctx = pipeline.build_context("")
        stage.propose(ctx, full_stack())  # must not raise


# --------------------------------------------------------- LLMProvider ------
@pytest.mark.parametrize("provider", [MockProvider()], ids=["mock"])
class TestProviderContract:
    def test_exposes_identity(self, provider):
        assert provider.model_name and provider.model_digest and provider.quantisation

    def test_digest_is_stable(self, provider):
        assert provider.model_digest == provider.model_digest

    def test_streams_monotonic_indices(self, provider):
        events = list(provider.generate("hello", GenParams(num_predict=32)))
        assert [e.index for e in events] == list(range(len(events)))

    def test_timestamps_are_non_decreasing(self, provider):
        events = list(provider.generate("hello world", GenParams(num_predict=32)))
        stamps = [e.emitted_at_ns for e in events]
        assert stamps == sorted(stamps)

    def test_respects_num_predict(self, provider):
        events = list(provider.generate("explain everything", GenParams(num_predict=5)))
        assert len(events) <= 5

    def test_is_deterministic(self, provider):
        a = "".join(e.text for e in provider.generate("q", GenParams(num_predict=64)))
        b = "".join(e.text for e in provider.generate("q", GenParams(num_predict=64)))
        assert a == b

    def test_handles_an_empty_prompt(self, provider):
        list(provider.generate("", GenParams(num_predict=8)))  # must not raise


# ----------------------------------------------------------- Tokenizer ------
@pytest.mark.parametrize("tokenizer", [HeuristicTokenizer("t")], ids=["heuristic"])
class TestTokenizerContract:
    def test_count_matches_encode_length(self, tokenizer):
        for text in ["hello world", "a", "", "3.5 kg of stuff", "```code```"]:
            assert tokenizer.count(text) == len(tokenizer.encode(text))

    def test_offsets_stay_inside_the_string(self, tokenizer):
        text = "the quick brown fox"
        assert all(0 <= s <= e <= len(text) for s, e in tokenizer.offsets(text))

    def test_empty_string_costs_nothing(self, tokenizer):
        assert tokenizer.count("") == 0

    def test_count_is_monotonic_under_concatenation(self, tokenizer):
        assert tokenizer.count("hello world") >= tokenizer.count("hello")

    def test_exposes_an_id(self, tokenizer):
        assert isinstance(tokenizer.id, str) and tokenizer.id


# ---------------------------------------------------------- LedgerSink ------
def _row(request_id: str = "r1") -> LedgerRow:
    return LedgerRow(
        request_id=request_id,
        conversation_id="c1",
        turn_index=0,
        config_hash="abc123",
        run_id="run1",
        tokens_in_original=10,
        tokens_in_final=7,
        tokens_out=5,
        route_tier="MODEL_SMALL",
    )


class TestSinkContract:
    def test_memory_sink_round_trips(self):
        sink = MemorySink()
        sink.write(_row())
        assert sink.rows[0].request_id == "r1"

    def test_jsonl_sink_round_trips(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ledger.jsonl"
            with JsonlSink(path) as sink:
                sink.write(_row("a"))
                sink.write(_row("b"))
            rows = list(read_jsonl(path))
            assert [r["request_id"] for r in rows] == ["a", "b"]
            assert rows[0]["tokens_in_final"] == 7

    def test_jsonl_survives_a_truncated_final_line(self):
        """An interrupted sweep must cost one row, not the whole run."""
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ledger.jsonl"
            with JsonlSink(path) as sink:
                sink.write(_row("a"))
            with path.open("a", encoding="utf-8") as fh:
                fh.write('{"request_id": "b", "tok')  # power cut mid-write
            assert [r["request_id"] for r in read_jsonl(path)] == ["a"]

    def test_sqlite_sink_round_trips(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ledger.db"
            with SqliteSink(path) as sink:
                sink.write(_row("a"))
                sink.flush()
                cur = sink._conn.execute("SELECT request_id, tokens_in_final FROM ledger")
                assert cur.fetchone() == ("a", 7)

    def test_sqlite_is_idempotent_on_replay(self):
        """A resumed sweep re-writing a completed row must not duplicate it."""
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ledger.db"
            with SqliteSink(path) as sink:
                sink.write(_row("a"))
                sink.write(_row("a"))
                sink.flush()
                count = sink._conn.execute("SELECT COUNT(*) FROM ledger").fetchone()[0]
                assert count == 1
