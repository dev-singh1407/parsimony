"""Prefill/decode measurement, and the guards that keep it honest.

Research gap 2 is the prefill/decode split. Both measurements here produced a
plausible-looking flat line before their guards existed — a flat line is what
this measurement looks like when it is silently wrong, which is why the guards
are tested harder than the happy path.

The arithmetic and the guards are tested with a scripted fake provider so they
run everywhere. The real numbers are only measurable with Ollama up, and those
tests skip when it is not.
"""

from __future__ import annotations

import pytest

from parsimony.core.errors import ProviderError
from parsimony.eval.latency import (
    PrefillPoint,
    prefill_scaling,
    prefix_reuse,
)
from parsimony.infra.providers import DEFAULT_OLLAMA_MODEL, MockProvider, OllamaProvider


class FakeOllama:
    """Speaks `probe`, with a controllable cost model."""

    def __init__(self, ms_per_token=8.5, cap=None, cached_after=None):
        self.ms_per_token = ms_per_token
        self.cap = cap                  # simulate context truncation
        self.cached_after = cached_after  # simulate KV prefix reuse
        self.num_ctx = None
        self.seen: list[str] = []

    def probe(self, prompt: str, *, num_predict: int = 4) -> dict:
        tokens = max(1, len(prompt) // 4)
        if self.cap is not None:
            tokens = min(tokens, self.cap)
        cached = self.cached_after is not None and len(self.seen) >= self.cached_after
        self.seen.append(prompt)
        return {
            "prompt_eval_count": tokens,
            "prompt_eval_duration": int((5 if cached else tokens * self.ms_per_token) * 1e6),
            "eval_count": num_predict,
            "eval_duration": int(130 * 1e6),
        }


class TestArithmetic:
    def test_ms_per_prompt_token(self):
        assert PrefillPoint(1000, 8500.0, 130.0, 4).ms_per_prompt_token == pytest.approx(8.5)

    def test_prefill_share(self):
        assert PrefillPoint(1000, 9000.0, 1000.0, 4).prefill_share == pytest.approx(90.0)

    def test_zero_tokens_does_not_divide_by_zero(self):
        assert PrefillPoint(0, 0.0, 0.0, 0).ms_per_prompt_token == 0.0
        assert PrefillPoint(0, 0.0, 0.0, 0).prefill_share == 0.0


class TestTruncationGuard:
    """An unset context window silently cut a 14,000-token prompt to 2,050 and
    reported 0.03 ms/token — a cache hit wearing a measurement's clothes,
    because truncation removed the unique head and made two prompts identical."""

    def test_truncated_points_are_dropped_not_reported(self):
        fake = FakeOllama(cap=300)
        points = prefill_scaling(fake, target_tokens=(128, 256, 512, 1024))
        assert all(p.prompt_tokens <= 300 for p in points)
        assert len(points) < 4, "points beyond the cap must be dropped"

    def test_targets_beyond_the_window_are_never_attempted(self):
        fake = FakeOllama()
        prefill_scaling(fake, target_tokens=(128, 8192), num_ctx=4096)
        assert not any(len(p) > 30000 for p in fake.seen)

    def test_the_window_is_set_and_then_restored(self):
        fake = FakeOllama()
        fake.num_ctx = "original"
        prefill_scaling(fake, target_tokens=(128,), num_ctx=4096)
        assert fake.num_ctx == "original", "must not mutate the caller's provider"


class TestPrefixCacheIsNotMeasuredByAccident:
    def test_each_prompt_carries_a_distinct_head(self):
        """Consecutive prompts sharing a prefix let the server skip prefill, so
        the measurement reports the cache instead of prefill."""
        fake = FakeOllama()
        prefill_scaling(fake, target_tokens=(128, 256, 512))
        heads = [p[:40] for p in fake.seen]
        assert len(set(heads)) == len(heads)

    def test_scaling_is_recovered_from_a_well_behaved_server(self):
        fake = FakeOllama(ms_per_token=8.5)
        points = prefill_scaling(fake, target_tokens=(128, 256, 512, 1024))
        assert [p.prompt_tokens for p in points] == sorted(p.prompt_tokens for p in points)
        for p in points:
            assert p.ms_per_prompt_token == pytest.approx(8.5, abs=0.01)


class TestPrefixReuse:
    def test_stable_and_volatile_arms_are_both_reported(self):
        arms = prefix_reuse(FakeOllama(), words=20, repeats=2)
        assert [a.label for a in arms] == ["stable prefix (M4)", "volatile head"]

    def test_reuse_is_detected_when_the_server_caches(self):
        arms = prefix_reuse(FakeOllama(cached_after=1), words=20, repeats=2)
        assert arms[0].reuse_pct > 90

    def test_no_reuse_when_the_server_never_caches(self):
        arms = prefix_reuse(FakeOllama(), words=20, repeats=2)
        assert arms[0].reuse_pct == pytest.approx(0.0, abs=1.0)

    def test_token_counts_are_comparable_between_arms(self):
        """The claim is that placement costs time at equal token count. If the
        arms differed materially in size the comparison would be confounded."""
        arms = prefix_reuse(FakeOllama(), words=60, repeats=1)
        a, b = arms[0].prompt_tokens, arms[1].prompt_tokens
        assert abs(a - b) / max(a, b) < 0.05


class TestRequiresARealProvider:
    def test_mock_is_refused_with_a_reason(self):
        """MockProvider synthesises TTFT and TPOT from two constants, so it
        cannot answer a question about the split. Silently accepting it would
        produce numbers that look real."""
        with pytest.raises(ProviderError, match="MockProvider synthesises"):
            prefill_scaling(MockProvider(), target_tokens=(128,))


@pytest.mark.skipif(
    not OllamaProvider.available(model=DEFAULT_OLLAMA_MODEL),
    reason="ollama is not running",
)
class TestAgainstTheRealModel:
    def test_prefill_dominates_on_cpu(self):
        """The empirical case for the whole project: on CPU the prompt side is
        the expensive half, so removing input tokens buys the expensive half."""
        points = prefill_scaling(OllamaProvider(), target_tokens=(256, 512))
        assert points, "expected at least one usable measurement"
        assert all(p.prefill_share > 80 for p in points)

    def test_prefill_grows_with_prompt_length(self):
        points = prefill_scaling(OllamaProvider(), target_tokens=(256, 1024))
        assert len(points) == 2
        assert points[1].prefill_ms > points[0].prefill_ms * 2

    def test_a_volatile_head_destroys_prefix_reuse(self):
        """ADR-025 measured this in prefix-tokens-reused, a proxy nobody
        outside this project reports. It has a wall-clock price."""
        stable, volatile = prefix_reuse(OllamaProvider(), words=120, repeats=2)
        assert stable.reuse_pct > 80
        assert volatile.reuse_pct < 30
        assert stable.steady_ms < volatile.steady_ms
