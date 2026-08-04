"""Shared fixtures.

Tests use HeuristicTokenizer rather than the real Hugging Face one so the suite
is hermetic: a test run must never depend on network access or a warm HF cache.
Token *counts* differ from production, which is fine — these tests assert
behaviour and relative direction, never absolute vocabulary-specific numbers.
"""

from __future__ import annotations

import pytest

from parsimony.core.config import ParsimonyConfig, baseline, full_stack
from parsimony.infra.providers import MockProvider
from parsimony.infra.storage import MemorySink
from parsimony.infra.tokenization import HeuristicTokenizer
from parsimony.modules.m2_cache import SemanticCache
from parsimony.pipeline.orchestrator import Pipeline
from parsimony.pipeline.registry import default_registry


@pytest.fixture
def tok() -> HeuristicTokenizer:
    return HeuristicTokenizer("test")


@pytest.fixture
def cfg() -> ParsimonyConfig:
    return full_stack()


@pytest.fixture
def make_pipeline(tok):
    def _make(config: ParsimonyConfig | None = None, sink=None) -> Pipeline:
        config = config or full_stack()
        cache = SemanticCache(config.cache.ttl_seconds)
        return Pipeline(
            config,
            provider=MockProvider(),
            tokenizer=tok,
            cache=cache,
            registry=default_registry(cache),
            sink=sink,
        )

    return _make


@pytest.fixture
def pipeline(make_pipeline) -> Pipeline:
    return make_pipeline()


@pytest.fixture
def baseline_pipeline(make_pipeline) -> Pipeline:
    return make_pipeline(baseline())


@pytest.fixture
def sink() -> MemorySink:
    return MemorySink()
