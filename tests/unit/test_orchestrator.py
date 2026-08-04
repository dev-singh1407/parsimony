"""Orchestrator behaviour — the properties the whole ablation rests on."""

from __future__ import annotations

from dataclasses import replace

import pytest

from parsimony.core.config import ParsimonyConfig, baseline, full_stack, with_cache_lookup
from parsimony.core.errors import ConfigError
from parsimony.core.ledger import StageOutcome
from parsimony.core.types import Mode, RouteTier
from parsimony.pipeline.registry import StageRegistry


class TestAblation:
    def test_disabled_modules_are_skipped_not_run(self, baseline_pipeline):
        outcome = baseline_pipeline.run("Could you please explain recursion? Thanks!")
        applied = [t for t in outcome.traces if t.outcome is StageOutcome.APPLIED]
        assert applied == []

    def test_baseline_leaves_the_query_untouched(self, baseline_pipeline):
        query = "Could you please explain recursion? Thanks!"
        outcome = baseline_pipeline.run(query)
        assert outcome.ctx.query == query

    def test_enabling_m1_reduces_input_tokens(self, make_pipeline):
        query = "Hello! Could you please explain to me what recursion is? Thanks in advance!"
        base = make_pipeline(baseline()).run(query)
        opt = make_pipeline(full_stack()).run(query)
        assert opt.row.tokens_in_final < base.row.tokens_in_final

    def test_every_configured_stage_appears_in_the_trace(self, pipeline):
        outcome = pipeline.run("What is the capital of France?")
        traced = {t.name for t in outcome.traces}
        assert traced == set(pipeline.cfg.stage_order)

    def test_unimplemented_stages_are_visible_rather_than_absent(self, pipeline):
        outcome = pipeline.run("hello")
        planned = [t for t in outcome.traces if t.outcome is StageOutcome.NOT_IMPLEMENTED]
        assert {t.name for t in planned} == {"m3_history", "m4_assembler", "m6b_router"}


class TestShortCircuit:
    def test_arithmetic_bypasses_the_model_entirely(self, pipeline):
        outcome = pipeline.run("What is 847 * 23?")
        assert outcome.response == "19481"
        assert outcome.row.route_tier == RouteTier.DETERMINISTIC.name
        assert outcome.row.tokens_out == 0
        assert outcome.row.tokens_in_final == 0
        assert not outcome.generated

    def test_stages_after_a_short_circuit_do_not_run(self, pipeline):
        outcome = pipeline.run("What is 2+2?")
        names = [t.name for t in outcome.traces]
        assert names == ["m6a_deterministic"]

    def test_an_identical_repeat_is_served_from_cache(self, pipeline):
        query = "What is the capital of France?"
        first = pipeline.run(query)
        second = pipeline.run(query)
        assert first.row.route_tier == RouteTier.MODEL_SMALL.name
        assert second.row.route_tier == RouteTier.CACHE_EXACT.name
        assert second.row.tokens_out == 0

    def test_a_cache_hit_returns_the_same_answer(self, pipeline):
        query = "What is the capital of France?"
        assert pipeline.run(query).response == pipeline.run(query).response


class TestFidelityIntegration:
    def test_a_reverted_patch_leaves_the_context_unchanged(self, pipeline):
        query = "Explain the deadline. The deadline is 15 March. The deadline is 16 March."
        outcome = pipeline.run(query)
        assert outcome.row.gate_fired
        assert outcome.ctx.query == query

    def test_the_reverting_stage_is_recorded_as_reverted(self, pipeline):
        outcome = pipeline.run(
            "Explain the deadline. The deadline is 15 March. The deadline is 16 March."
        )
        reverted = [t for t in outcome.traces if t.outcome is StageOutcome.REVERTED]
        assert len(reverted) == 1
        assert reverted[0].gate_events[0].invariant_class == "number"

    def test_a_revert_does_not_discard_earlier_savings(self, make_pipeline):
        """Separate proposals per tier mean a bad tier-3 edit never costs
        tier-1's safe reduction (docs/02-module-specs.md M1)."""
        query = ("Hello! Please explain the deadline. The deadline is 15 March. "
                 "The deadline is 16 March. Thanks!")
        outcome = make_pipeline(full_stack()).run(query)
        assert outcome.row.gate_fired
        assert outcome.row.tokens_in_final < outcome.row.tokens_in_original


class TestLedger:
    def test_writes_one_row_per_request(self, make_pipeline, sink):
        p = make_pipeline(full_stack(), sink=sink)
        p.run("What is 2+2?")
        p.run("What is the capital of France?")
        assert len(sink.rows) == 2

    def test_row_carries_experiment_identity(self, make_pipeline, sink):
        cfg = full_stack()
        make_pipeline(cfg, sink=sink).run("hello")
        row = sink.rows[0]
        assert row.config_hash == cfg.config_hash
        assert row.schema_version == 1
        assert row.model_digest == "mock:v1"

    def test_row_serialises_to_json_safe_types(self, make_pipeline, sink):
        import json

        make_pipeline(full_stack(), sink=sink).run("Explain recursion please.")
        json.dumps(sink.rows[0].to_dict())  # must not raise

    def test_middleware_time_excludes_generation(self, make_pipeline, sink):
        make_pipeline(full_stack(), sink=sink).run("What is the capital of France?")
        row = sink.rows[0]
        assert 0 <= row.middleware_ns <= row.total_ns

    def test_ledger_failure_is_fatal_in_experiment_mode(self, make_pipeline):
        class BrokenSink:
            def write(self, row):
                raise OSError("disk full")

            def flush(self): ...

            def close(self): ...

        from parsimony.core.errors import LedgerError

        p = make_pipeline(replace(full_stack(), mode=Mode.EXPERIMENT), sink=BrokenSink())
        with pytest.raises(LedgerError):
            p.run("hello")

    def test_ledger_failure_is_survivable_in_serve_mode(self, make_pipeline):
        class BrokenSink:
            def write(self, row):
                raise OSError("disk full")

            def flush(self): ...

            def close(self): ...

        p = make_pipeline(replace(full_stack(), mode=Mode.SERVE), sink=BrokenSink())
        assert p.run("hello").response  # request still succeeds


class TestErrorIsolation:
    def test_a_raising_module_does_not_fail_the_request(self, make_pipeline):
        pipeline = make_pipeline(full_stack())

        class Exploding:
            module_id, name = "M1", "m1_tier1"
            reads, writes = frozenset({"query"}), frozenset({"query"})

            def applies_to(self, ctx, cfg):
                return True

            def propose(self, ctx, cfg):
                raise RuntimeError("boom")

        pipeline.registry._stages["m1_tier1"] = Exploding()
        outcome = pipeline.run("What is the capital of France?")
        assert outcome.response
        errors = [t for t in outcome.traces if t.outcome is StageOutcome.ERROR]
        assert len(errors) == 1 and "boom" in errors[0].rationale


class TestStageOrdering:
    def test_reordering_the_cache_changes_where_it_runs(self, make_pipeline):
        raw = make_pipeline(with_cache_lookup(full_stack(), "RAW"))
        comp = make_pipeline(with_cache_lookup(full_stack(), "COMPRESSED"))
        raw_names = [t.name for t in raw.run("Please explain recursion.").traces]
        comp_names = [t.name for t in comp.run("Please explain recursion.").traces]
        assert raw_names.index("m2_cache") < raw_names.index("m1_tier1")
        assert comp_names.index("m2_cache") > comp_names.index("m1_tier1")

    def test_compressed_lookup_collapses_politeness_only_paraphrases(self, make_pipeline):
        """The Gap 3 effect, as a regression test: with the cache behind the
        compressor, 'Explain X.' and 'Please explain X.' become one key."""
        p = make_pipeline(with_cache_lookup(full_stack(), "COMPRESSED"))
        p.run("Explain recursion.")
        second = p.run("Please explain recursion.")
        assert second.row.cache_hit

    def test_raw_lookup_does_not(self, make_pipeline):
        p = make_pipeline(with_cache_lookup(full_stack(), "RAW"))
        p.run("Explain recursion.")
        assert not p.run("Please explain recursion.").row.cache_hit

    def test_boot_fails_on_an_impossible_order(self):
        reg = StageRegistry()

        class NeedsAssembled:
            module_id, name = "MX", "needs_assembled"
            reads, writes = frozenset({"assembled"}), frozenset()

            def applies_to(self, ctx, cfg):
                return True

            def propose(self, ctx, cfg): ...

        reg.register(NeedsAssembled())
        cfg = ParsimonyConfig(stage_order=("needs_assembled",))
        with pytest.raises(ConfigError, match="reads"):
            reg.validate(cfg)


class TestDeterminism:
    def test_the_same_query_yields_the_same_response(self, make_pipeline):
        a = make_pipeline(baseline()).run("Explain recursion")
        b = make_pipeline(baseline()).run("Explain recursion")
        assert a.response == b.response

    def test_identical_configs_hash_identically_across_pipelines(self, make_pipeline, sink):
        make_pipeline(full_stack(), sink=sink).run("hello")
        make_pipeline(full_stack(), sink=sink).run("hello")
        assert sink.rows[0].config_hash == sink.rows[1].config_hash
