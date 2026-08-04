"""Model providers.

ADR-007: MockProvider is built first so the orchestrator, ledger timing fields,
the budgeter's early-stop rule and the whole ablation harness are complete and
tested before any model is installed. OllamaProvider lands in Sprint 2 as ~60
lines against this same interface, which already has a contract-test suite.

Every number produced through MockProvider is pipeline-correctness evidence,
NOT performance evidence. The ledger makes that unmistakable: model_digest is
literally "mock:v1".
"""

from __future__ import annotations

import hashlib
import time
from typing import Iterator

from parsimony.core.types import GenParams, TokenEvent

# Realistic for a 1B Q4_K_M model on an i5-class CPU. Used to synthesise
# timestamps, not to actually sleep (a sweep that really waited would take days).
MOCK_TTFT_NS = 120_000_000  # 120 ms to first token
MOCK_TPOT_NS = 65_000_000  # ~15 tokens/second

_CANNED: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        ("boiling", "water"),
        "Water boils at 100 degrees Celsius at sea level, where atmospheric "
        "pressure is 101.3 kPa. The boiling point falls as altitude increases "
        "because the surrounding pressure drops.",
    ),
    (
        ("capital", "australia"),
        "The capital of Australia is Canberra. It was chosen as a compromise "
        "between Sydney and Melbourne and became the seat of government in 1927.",
    ),
    (
        ("python", "list", "tuple"),
        "A list is mutable and a tuple is immutable. Lists use square brackets "
        "and support item assignment; tuples use parentheses, are hashable when "
        "their contents are, and can therefore be dictionary keys.",
    ),
    (
        ("photosynthesis",),
        "Photosynthesis is the process by which plants convert light energy into "
        "chemical energy. Chlorophyll absorbs light, water is split, and carbon "
        "dioxide is fixed into glucose, releasing oxygen as a by-product.",
    ),
)

_FILLER = (
    "The answer depends on the specific context you are working in.",
    "There are several factors that influence the outcome here.",
    "In most practical cases the standard approach is sufficient.",
    "It is worth considering the trade-offs before deciding.",
    "This behaviour is consistent across the common implementations.",
)


class MockProvider:
    """Deterministic fake model.

    Given identical prompt bytes and parameters it returns identical output,
    which mirrors the temperature-0 property that generation memoisation relies
    on (ADR-019) and makes the whole test suite reproducible.

    It deliberately becomes repetitive in its tail, so that M5's trigram-novelty
    early-stop rule has something real to fire on before Ollama exists.
    """

    def __init__(self, realtime: bool = False, verbosity: int = 6) -> None:
        self._realtime = realtime
        self._verbosity = verbosity

    @property
    def model_name(self) -> str:
        return "mock-1b"

    @property
    def quantisation(self) -> str:
        return "none"

    @property
    def model_digest(self) -> str:
        return "mock:v1"

    @property
    def tokenizer_id(self) -> str:
        return "mock"

    def _body(self, prompt: str) -> str:
        low = prompt.lower()
        for keys, answer in _CANNED:
            if all(k in low for k in keys):
                return answer
        seed = int(hashlib.blake2b(prompt.encode(), digest_size=4).hexdigest(), 16)
        n = 3 + (seed % max(1, self._verbosity - 2))
        picked = [_FILLER[(seed + i) % len(_FILLER)] for i in range(n)]
        # Repetitive tail: small models restate. This is what M5 stops.
        picked.append(picked[0])
        picked.append(picked[0])
        return " ".join(picked)

    def generate(self, prompt: str, params: GenParams) -> Iterator[TokenEvent]:
        text = self._body(prompt)
        pieces = _split_pieces(text)
        start = time.perf_counter_ns()
        for i, piece in enumerate(pieces):
            if i >= params.num_predict:
                break
            if params.stop and any(s in piece for s in params.stop):
                break
            if self._realtime:
                time.sleep((MOCK_TTFT_NS if i == 0 else MOCK_TPOT_NS) / 1e9)
                emitted = time.perf_counter_ns()
            else:
                emitted = start + MOCK_TTFT_NS + i * MOCK_TPOT_NS
            yield TokenEvent(text=piece, index=i, emitted_at_ns=emitted)


def _split_pieces(text: str) -> list[str]:
    """Approximate token pieces: words with their leading space, as a BPE
    tokenizer would emit them."""
    out: list[str] = []
    for i, word in enumerate(text.split(" ")):
        out.append(word if i == 0 else " " + word)
    return out
