"""Where the time actually goes on CPU-only hardware.

Research gap 2 is the prefill/decode split. Under MockProvider it was
unanswerable — the mock synthesises TTFT and TPOT from two constants — so the
whole project could only argue in token counts and assert that tokens stand in
for time. With a real model that assertion becomes measurable, and it turns out
to be understated rather than merely unverified.

Two measurements live here:

`prefill_scaling` — how prefill cost grows with prompt length. On CPU this is
the dominant term by an order of magnitude, which is the empirical case for the
entire project: input reduction buys the expensive half.

`prefix_reuse` — what a volatile token at the head of a prompt costs. M4
assembles prompts so the invariant part comes first (ADR-025); the payoff was
previously expressed as "prefix tokens reused", a proxy nobody outside this
project measures. It has a wall-clock price.

Both read Ollama's server-reported timings rather than wall-clock TTFT, because
TTFT conflates prefill with HTTP and scheduling overhead, and the gap is
specifically about the split.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass

from parsimony.core.errors import ProviderError
from parsimony.core.types import GenParams

# One short generation: we are measuring the prompt side, and every decoded
# token adds noise to a measurement that is not about decoding.
_PROBE = GenParams(num_predict=4, temperature=0.0, seed=0)

_FILLER = "The system processes each request in the order it was received. "


def _nonce() -> str:
    """A fresh marker for every run.

    Ollama's KV cache outlives the process. With fixed prompts, the first run
    measures prefill and every run after it measures the cache: a re-run
    reported 0.37 ms/token against the first run's 8.25, and turned the
    prefix-reuse arms into nonsense (-10,092% "reuse"). A measurement that is
    only valid the first time it is ever executed is not a measurement, and it
    would have failed in front of a reviewer running the demo twice.
    """
    return secrets.token_hex(6)


@dataclass(frozen=True, slots=True)
class PrefillPoint:
    prompt_tokens: int
    prefill_ms: float
    decode_ms: float
    decoded_tokens: int

    @property
    def ms_per_prompt_token(self) -> float:
        return self.prefill_ms / self.prompt_tokens if self.prompt_tokens else 0.0

    @property
    def prefill_share(self) -> float:
        total = self.prefill_ms + self.decode_ms
        return (self.prefill_ms / total * 100) if total else 0.0


def _measure(provider, prompt: str) -> tuple[float, float, int, int]:
    probe = getattr(provider, "probe", None)
    if probe is None:
        raise ProviderError(
            "provider reports no server-side timings, so prefill and decode "
            "cannot be separated. This measurement requires OllamaProvider; "
            "MockProvider synthesises both from constants."
        )
    s = probe(prompt, num_predict=_PROBE.num_predict)
    if "prompt_eval_duration" not in s:
        raise ProviderError(f"ollama returned no prefill timing: {sorted(s)}")
    return (
        s["prompt_eval_duration"] / 1e6,
        s.get("eval_duration", 0) / 1e6,
        s.get("prompt_eval_count", 0),
        s.get("eval_count", 0),
    )


def prefill_scaling(
    provider,
    *,
    target_tokens=(128, 256, 512, 1024, 1536),
    num_ctx: int = 4096,
    tolerance: float = 0.15,
    nonce: str | None = None,
    progress=None,
) -> list[PrefillPoint]:
    """Prefill cost against prompt length.

    Two failure modes are guarded here, both of which silently produce a
    plausible-looking flat line rather than an error:

    1. KV prefix reuse. Consecutive prompts sharing a prefix let the server
       skip prefill entirely, so this measures the cache instead. Each prompt
       therefore carries a distinct marker at position 0.
    2. Context truncation. A prompt longer than num_ctx is silently cut, which
       both caps the measurement and — because the cut removes the head — can
       make two different prompts identical. An unset window truncated 7,000
       and 14,000-token prompts to the same 2,050 tokens and reported 0.03
       ms/token for the second, which is a cache hit wearing a measurement's
       clothes. The window is now explicit, and any prompt whose measured
       token count falls short of the request is reported rather than kept.
    """
    note = progress or (lambda _: None)
    run = nonce or _nonce()
    original_ctx = getattr(provider, "num_ctx", None)
    provider.num_ctx = num_ctx
    points, dropped = [], []
    try:
        for i, target in enumerate(target_tokens):
            if target > num_ctx * 0.9:
                dropped.append((target, f"exceeds 90% of num_ctx={num_ctx}"))
                continue
            note(f"~{target} tokens")
            # ~1.3 tokens per word for this filler; overshoot then let the
            # measured count be the truth.
            words = int(target / 1.3)
            prompt = f"Probe {run}-{i} marker {i * 7919}. " + _FILLER * (words // 11 + 1)
            prefill_ms, decode_ms, in_tok, out_tok = _measure(provider, prompt)
            if in_tok < target * (1 - tolerance):
                dropped.append((target, f"server counted only {in_tok} tokens — truncated"))
                continue
            points.append(PrefillPoint(in_tok, prefill_ms, decode_ms, out_tok))
    finally:
        provider.num_ctx = original_ctx

    if dropped:
        note("dropped: " + "; ".join(f"{t} ({why})" for t, why in dropped))
    return points


@dataclass(frozen=True, slots=True)
class PrefixArm:
    label: str
    prompt_tokens: int
    first_ms: float
    repeat_ms: list[float]

    @property
    def steady_ms(self) -> float:
        return sum(self.repeat_ms) / len(self.repeat_ms) if self.repeat_ms else self.first_ms

    @property
    def saved_ms(self) -> float:
        return self.first_ms - self.steady_ms

    @property
    def reuse_pct(self) -> float:
        return (self.saved_ms / self.first_ms * 100) if self.first_ms else 0.0


def prefix_reuse(
    provider, *, words: int = 120, repeats: int = 3,
    nonce: str | None = None, progress=None,
) -> list[PrefixArm]:
    """What a volatile token at the head of the prompt costs.

    Two arms over the same content and near-identical token counts. The stable
    arm keeps the shared context first and varies only the tail, which is what
    M4's prefix-stable assembly produces. The volatile arm prepends a turn
    counter — the "turn N of M" preamble that looks free in a token count and
    is exactly what `assemble_volatile_head` models.
    """
    note = progress or (lambda _: None)
    run = nonce or _nonce()
    # The nonce sits INSIDE the shared body, so both arms carry it and the
    # comparison stays fair — it makes each run cold without making the two
    # arrangements differ.
    body = f"Context {run}. " + (_FILLER * words).strip()
    arms = []

    note("stable prefix")
    first = None
    repeat_ms = []
    for i in range(repeats + 1):
        prefill_ms, _, in_tok, _ = _measure(provider, f"{body}\nQuestion {i}: what is this?")
        if i == 0:
            first, tokens = prefill_ms, in_tok
        else:
            repeat_ms.append(prefill_ms)
    arms.append(PrefixArm("stable prefix (M4)", tokens, first, repeat_ms))

    note("volatile head")
    first = None
    repeat_ms = []
    for i in range(repeats + 1):
        prompt = f"Session {i} started at turn {i}. {body}\nQuestion: what is this?"
        prefill_ms, _, in_tok, _ = _measure(provider, prompt)
        if i == 0:
            first, tokens = prefill_ms, in_tok
        else:
            repeat_ms.append(prefill_ms)
    arms.append(PrefixArm("volatile head", tokens, first, repeat_ms))

    return arms
