"""M6b escalation router: features, complexity, and tier selection."""

from __future__ import annotations

from dataclasses import replace

import pytest

from parsimony.core.config import full_stack
from parsimony.core.proposals import ContextPatch
from parsimony.core.types import ResponseClass, RouteTier
from parsimony.infra.providers import MockProvider
from parsimony.modules.m6_router import (
    EscalationRouterStage,
    complexity_score,
    routing_features,
)


def _escalating_cfg(threshold: float = 0.4):
    base = full_stack()
    return replace(
        base, router=replace(base.router, escalation_tier=True, escalation_complexity=threshold)
    )


class TestRoutingFeatures:
    def test_counts_words_and_questions(self, pipeline):
        ctx = pipeline.build_context("Why is this slow? What can I do?")
        f = routing_features(ctx)
        assert f["n_questions"] == 2
        assert f["n_words"] == 8

    def test_detects_reasoning_markers(self, pipeline):
        ctx = pipeline.build_context("Explain why this happens and compare the trade-offs")
        assert routing_features(ctx)["reason_markers"] >= 2

    def test_counts_invariants(self, pipeline):
        ctx = pipeline.build_context("Is Paris not 500 km away?")
        f = routing_features(ctx)
        assert f["n_numbers"] >= 1
        assert f["n_negations"] >= 1

    def test_every_feature_is_logged_even_when_unused(self, pipeline):
        """A learned router can be fitted from the ledger later only if the
        features were being recorded all along."""
        ctx = pipeline.build_context("hello")
        proposal = EscalationRouterStage().propose(ctx, full_stack())
        assert isinstance(proposal, ContextPatch)
        for key in routing_features(ctx):
            assert key in proposal.evidence


class TestComplexity:
    def test_is_bounded(self, pipeline):
        for query in ["hi", "Explain why " * 40, "What is 2+2?"]:
            ctx = pipeline.build_context(query)
            assert 0.0 <= complexity_score(routing_features(ctx)) <= 1.0

    def test_reasoning_scores_higher_than_a_lookup(self, pipeline):
        simple = complexity_score(routing_features(pipeline.build_context("Capital of France?")))
        hard = complexity_score(
            routing_features(
                pipeline.build_context(
                    "Explain why quicksort degrades and compare the trade-offs step by step"
                )
            )
        )
        assert hard > simple


class TestTierSelection:
    def test_stays_small_when_escalation_is_disabled(self, pipeline):
        ctx = pipeline.build_context("Explain why this fails and compare the trade-offs")
        proposal = EscalationRouterStage().propose(ctx, full_stack())
        assert proposal.fields["route_tier"] is RouteTier.MODEL_SMALL

    def test_escalates_a_complex_query_when_enabled(self, pipeline):
        ctx = replace(
            pipeline.build_context(
                "Explain why quicksort degrades and compare the trade-offs step by step"
            ),
            response_class=ResponseClass.REASONING,
        )
        proposal = EscalationRouterStage().propose(ctx, _escalating_cfg(0.4))
        assert proposal.fields["route_tier"] is RouteTier.MODEL_LARGE

    def test_does_not_escalate_a_simple_query(self, pipeline):
        ctx = pipeline.build_context("Capital of France?")
        proposal = EscalationRouterStage().propose(ctx, _escalating_cfg(0.4))
        assert proposal.fields["route_tier"] is RouteTier.MODEL_SMALL

    def test_decide_patches_never_touch_text(self, pipeline):
        from parsimony.core.proposals import TransformKind

        ctx = pipeline.build_context("anything")
        assert EscalationRouterStage().propose(ctx, full_stack()).kind is TransformKind.DECIDE


class TestProviderSelection:
    def test_escalated_requests_use_the_large_provider(self, tok):
        from parsimony.modules.m2_cache import SemanticCache
        from parsimony.pipeline.orchestrator import Pipeline
        from parsimony.pipeline.registry import default_registry

        cfg = _escalating_cfg(0.3)
        cache = SemanticCache(cfg.cache.ttl_seconds)
        large = MockProvider()
        large.__class__ = type("BigMock", (MockProvider,), {"model_name": property(lambda s: "mock-3b")})
        p = Pipeline(
            cfg,
            provider=MockProvider(),
            provider_large=large,
            tokenizer=tok,
            cache=cache,
            registry=default_registry(cache),
        )
        out = p.run("Explain why quicksort degrades and compare the trade-offs step by step")
        assert out.row.route_tier == RouteTier.MODEL_LARGE.name
        assert out.row.model_name == "mock-3b"

    def test_falls_back_to_the_small_provider_when_none_is_configured(self, make_pipeline):
        """The tier is still recorded honestly; model_name shows what actually ran,
        so the ledger cannot claim an escalation that did not happen."""
        p = make_pipeline(_escalating_cfg(0.3))
        out = p.run("Explain why quicksort degrades and compare the trade-offs step by step")
        assert out.row.route_tier == RouteTier.MODEL_LARGE.name
        assert out.row.model_name == "mock-1b"
