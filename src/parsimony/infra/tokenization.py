"""Tokenisation.

Token counts are measured under the *target model's own* vocabulary, never in
words and never with a generic tokenizer. The report's tokenizer-aware claim
(and all of M1 tier 3) depends on this being exact.

HF `tokenizers` is preferred over tiktoken because it exposes offset mapping,
which M1's windowed re-tokenisation and M4's prefix-survival measurement both
need (ADR-011).
"""

from __future__ import annotations

import re
import warnings

_MISSING = object()


class HFTokenizer:
    """Wraps a Hugging Face fast tokenizer. Counts are memoised: count() is on
    the hot path and the same strings recur constantly within a request."""

    def __init__(self, model_id: str) -> None:
        from tokenizers import Tokenizer as _HF  # imported lazily: infra, not core

        self._tok = _HF.from_pretrained(model_id)
        self._model_id = model_id
        self._count_cache: dict[str, int] = {}

    @property
    def id(self) -> str:
        return self._model_id

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False).ids

    def count(self, text: str) -> int:
        hit = self._count_cache.get(text, _MISSING)
        if hit is not _MISSING:
            return hit  # type: ignore[return-value]
        n = len(self._tok.encode(text, add_special_tokens=False).ids)
        if len(self._count_cache) < 50_000:
            self._count_cache[text] = n
        return n

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return list(self._tok.encode(text, add_special_tokens=False).offsets)


_WORDISH = re.compile(r"\s+|[A-Za-z]+|\d+|[^\sA-Za-z\d]")


class HeuristicTokenizer:
    """Offline fallback. Deterministic, but NOT the target model's vocabulary.

    Used only when the real tokenizer cannot be fetched. Any run using this is
    marked in the ledger via tokenizer_id, because token counts from it are not
    comparable with counts from a real vocabulary and must never be mixed into
    a result table.
    """

    def __init__(self, model_id: str = "heuristic") -> None:
        self._model_id = f"heuristic:{model_id}"

    @property
    def id(self) -> str:
        return self._model_id

    def _pieces(self, text: str) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        for m in _WORDISH.finditer(text):
            s, e = m.span()
            if text[s:e].isspace():
                continue
            # Approximate BPE: long words split into ~4-character pieces.
            length = e - s
            if length <= 4:
                out.append((s, e))
            else:
                for i in range(s, e, 4):
                    out.append((i, min(i + 4, e)))
        return out

    def encode(self, text: str) -> list[int]:
        return [hash(text[s:e]) & 0xFFFF for s, e in self._pieces(text)]

    def count(self, text: str) -> int:
        return len(self._pieces(text))

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return self._pieces(text)


_CACHE: dict[str, object] = {}


def get_tokenizer(model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"):
    """Return a real tokenizer, falling back to the heuristic one if offline.

    The fallback is loud: a silent degradation here would corrupt every token
    count in the project.
    """
    if model_id in _CACHE:
        return _CACHE[model_id]
    try:
        tok: object = HFTokenizer(model_id)
    except Exception as exc:  # network down, model gated, hub error
        warnings.warn(
            f"Could not load tokenizer {model_id!r} ({type(exc).__name__}); "
            "falling back to HeuristicTokenizer. Token counts will NOT be "
            "comparable with real-vocabulary runs.",
            RuntimeWarning,
            stacklevel=2,
        )
        tok = HeuristicTokenizer(model_id)
    _CACHE[model_id] = tok
    return tok


def common_prefix_tokens(a_ids: list[int], b_ids: list[int]) -> int:
    """Longest common token prefix — the unit KV-cache reuse actually works in.

    A byte-level measure over-reports across multi-byte boundaries and
    under-reports when a byte edit leaves the token sequence unchanged (ADR-011).
    """
    n = 0
    for x, y in zip(a_ids, b_ids):
        if x != y:
            break
        n += 1
    return n
