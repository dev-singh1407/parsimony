"""Per-request memo (ADR-006).

The naive design embeds the query in M2, again in M3, again in M1, and again for
classification — four MiniLM passes, ~20ms, a sixth of the whole overhead budget
spent recomputing the same 384 floats. Everything derived goes through here and
is computed at most once.

Lazy by design: a request short-circuited at router tier 0 pays for none of it.
"""

from __future__ import annotations

import numpy as np

from parsimony.core.errors import FeatureNotAvailable
from parsimony.infra.nlp import split_sentences


class DerivedCache:
    def __init__(self, tokenizer, embedder=None) -> None:
        self._tok = tokenizer
        self._embedder = embedder
        self._tokens: dict[str, int] = {}
        self._sentences: dict[str, tuple[str, ...]] = {}
        self._vectors: dict[str, np.ndarray] = {}
        self._counts = {
            "token_count": 0,
            "token_count_hits": 0,
            "sentences": 0,
            "embed_calls": 0,
            "embed_texts": 0,
            "embed_hits": 0,
        }

    def token_count(self, text: str) -> int:
        cached = self._tokens.get(text)
        if cached is not None:
            self._counts["token_count_hits"] += 1
            return cached
        self._counts["token_count"] += 1
        n = self._tok.count(text)
        self._tokens[text] = n
        return n

    def sentences(self, text: str) -> tuple[str, ...]:
        cached = self._sentences.get(text)
        if cached is not None:
            return cached
        self._counts["sentences"] += 1
        s = split_sentences(text)
        self._sentences[text] = s
        return s

    def embed(self, texts: list[str]) -> np.ndarray:
        """One batched call for whatever is not already memoised.

        The naive design embeds the query in M2, again in M3, again in M1 and
        again for classification. Here the second through fourth requests are
        dictionary lookups, and any genuinely new texts go to the encoder
        together rather than one at a time (ADR-006).
        """
        if self._embedder is None:
            raise FeatureNotAvailable(
                "This pipeline was constructed without an embedder. Pass one to "
                "Pipeline(embedder=...) to enable the semantic cache tier, MMR "
                "history selection and embedding-based deduplication."
            )
        if not texts:
            return np.zeros((0, self._embedder.dim), dtype=np.float32)

        missing = [t for t in dict.fromkeys(texts) if t not in self._vectors]
        self._counts["embed_hits"] += len(texts) - len(missing)
        if missing:
            self._counts["embed_calls"] += 1
            self._counts["embed_texts"] += len(missing)
            computed = self._embedder.embed(missing)
            for text, vec in zip(missing, computed):
                self._vectors[text] = vec
        return np.vstack([self._vectors[t] for t in texts])

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    @property
    def has_embedder(self) -> bool:
        return self._embedder is not None

    def stats(self) -> dict[str, int]:
        return dict(self._counts)
