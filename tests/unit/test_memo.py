"""Generation memoisation and sweep resumability.

The load-bearing test here is bit-exactness. The memo is only defensible as a
pure compute optimisation if a memoised run produces byte-identical answers to
an unmemoised one; if it does not, every result computed with it is suspect.
"""

from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

from parsimony.core.config import ParsimonyConfig, full_stack
from parsimony.core.types import GenParams, Mode
from parsimony.infra.memo import (
    CompletionLog,
    GenerationMemo,
    MemoEntry,
    gen_params_hash,
    memo_key,
)


def _experiment(cfg: ParsimonyConfig | None = None) -> ParsimonyConfig:
    return replace(cfg or full_stack(), mode=Mode.EXPERIMENT)


class TestMemoKey:
    def test_same_inputs_give_the_same_key(self):
        cfg = full_stack()
        p = GenParams(num_predict=128)
        a = memo_key("prompt", "digest", gen_params_hash(p, cfg))
        b = memo_key("prompt", "digest", gen_params_hash(p, cfg))
        assert a == b

    def test_prompt_change_changes_the_key(self):
        h = gen_params_hash(GenParams(), full_stack())
        assert memo_key("a", "d", h) != memo_key("b", "d", h)

    def test_model_digest_change_changes_the_key(self):
        """A silent `ollama pull` must not serve the old model's answers."""
        h = gen_params_hash(GenParams(), full_stack())
        assert memo_key("p", "digest-v1", h) != memo_key("p", "digest-v2", h)

    def test_num_predict_change_changes_the_key(self):
        cfg = full_stack()
        assert gen_params_hash(GenParams(num_predict=64), cfg) != gen_params_hash(
            GenParams(num_predict=128), cfg
        )

    def test_early_stop_setting_changes_the_key(self):
        """Two cells can share num_predict while differing in whether the stop
        rule runs, and the stopped output differs. Keying on num_predict alone
        would serve one cell's truncation to the other."""
        cfg = full_stack()
        without = replace(cfg, budget=replace(cfg.budget, early_stop=False))
        assert gen_params_hash(GenParams(), cfg) != gen_params_hash(GenParams(), without)


class TestMemoStore:
    def test_round_trips_in_memory(self):
        memo = GenerationMemo()
        memo.put("k", MemoEntry("hello", False))
        assert memo.get("k") == MemoEntry("hello", False)

    def test_miss_returns_none(self):
        assert GenerationMemo().get("absent") is None

    def test_tracks_hit_rate(self):
        memo = GenerationMemo()
        memo.put("k", MemoEntry("x", False))
        memo.get("k")
        memo.get("absent")
        assert memo.hit_rate == pytest.approx(50.0)

    def test_persists_across_instances(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "memo.db"
            first = GenerationMemo(path)
            first.put("k", MemoEntry("persisted", True))
            first.close()

            second = GenerationMemo(path)
            entry = second.get("k")
            second.close()
            assert entry == MemoEntry("persisted", True)


class TestBitExactness:
    def test_memoised_run_produces_identical_answers(self, make_pipeline):
        """The claim the whole optimisation rests on."""
        from parsimony.core.types import Turn

        questions = [
            "Explain what a hash table is.",
            "What is its average lookup complexity?",
            "Summarise how vaccines work.",
        ]

        def run(memo):
            p = make_pipeline(_experiment())
            p.memo = memo
            history: list[Turn] = []
            out = []
            for i, q in enumerate(questions):
                r = p.run(q, tuple(history), conversation_id="c", turn_index=i)
                history += [Turn(f"u{i}", "user", q), Turn(f"a{i}", "assistant", r.response)]
                out.append(r.response)
            return out

        cold = run(None)
        memo = GenerationMemo()
        run(memo)  # populate
        warm = run(memo)
        assert warm == cold
        assert memo.hits > 0

    @staticmethod
    def _no_cache():
        """M2 off. With the semantic cache enabled the second identical request
        short-circuits before generation, so the memo is never consulted — which
        is correct behaviour, but makes it the wrong setup for testing the memo."""
        cfg = _experiment()
        return cfg.with_modules(cfg.enabled_modules - {"M2"}, label="no-cache")

    def test_memo_hits_are_flagged_in_the_ledger(self, make_pipeline, sink):
        p = make_pipeline(self._no_cache(), sink=sink)
        p.memo = GenerationMemo()
        p.run("Summarise how vaccines work.", conversation_id="a")
        p.run("Summarise how vaccines work.", conversation_id="b")
        assert sink.rows[0].generation_memoised is False
        assert sink.rows[1].generation_memoised is True

    def test_memoised_rows_carry_no_timing(self, make_pipeline, sink):
        """A memo hit has no meaningful TTFT, so the field is left None rather
        than fabricated — latency analysis filters on generation_memoised, and a
        plausible-looking number would defeat that."""
        p = make_pipeline(self._no_cache(), sink=sink)
        p.memo = GenerationMemo()
        p.run("Summarise how vaccines work.", conversation_id="a")
        p.run("Summarise how vaccines work.", conversation_id="b")
        assert sink.rows[1].generation_memoised is True
        assert sink.rows[1].ttft_ns is None

    def test_the_cache_short_circuits_before_the_memo(self, make_pipeline, sink):
        """Documents the interaction: a cache hit never reaches generation, so
        it is neither a memo hit nor a memo miss."""
        p = make_pipeline(_experiment(), sink=sink)
        p.memo = GenerationMemo()
        p.run("Summarise how vaccines work.", conversation_id="a")
        p.run("Summarise how vaccines work.", conversation_id="b")
        assert sink.rows[1].cache_hit
        assert sink.rows[1].generation_memoised is False

    def test_memo_is_ignored_in_serve_mode(self, make_pipeline):
        """A served request must never receive a memoised answer."""
        p = make_pipeline(replace(full_stack(), mode=Mode.SERVE))
        assert p.memo is None


class TestCompletionLog:
    def test_marks_and_reports_completion(self):
        with tempfile.TemporaryDirectory() as d:
            with CompletionLog(Path(d) / "done.log") as log:
                assert not log.is_done("cfg", 0, "conv1")
                log.mark("cfg", 0, "conv1")
                assert log.is_done("cfg", 0, "conv1")

    def test_survives_a_restart(self):
        """An interruption at hour 14 should cost one conversation, not 14 hours."""
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "done.log"
            with CompletionLog(path) as log:
                log.mark("cfg", 0, "conv1")
                log.mark("cfg", 0, "conv2")
            with CompletionLog(path) as resumed:
                assert resumed.is_done("cfg", 0, "conv1")
                assert resumed.is_done("cfg", 0, "conv2")
                assert not resumed.is_done("cfg", 0, "conv3")

    def test_distinguishes_cells_and_seeds(self):
        with tempfile.TemporaryDirectory() as d:
            with CompletionLog(Path(d) / "done.log") as log:
                log.mark("cfgA", 0, "conv1")
                assert not log.is_done("cfgB", 0, "conv1")
                assert not log.is_done("cfgA", 1, "conv1")

    def test_marking_twice_is_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "done.log"
            with CompletionLog(path) as log:
                log.mark("cfg", 0, "conv1")
                log.mark("cfg", 0, "conv1")
            assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 1
