"""Per-request memo (ADR-006).

The naive design embeds the query in M2, again in M3, again in M1, and again for
classification — four MiniLM passes, ~20ms, a sixth of the whole overhead budget
spent recomputing the same 384 floats. Everything derived goes through here and
is computed at most once.

Lazy by design: a request short-circuited at router tier 0 pays for none of it.
"""

from __future__ import annotations

from parsimony.core.errors import FeatureNotAvailable
from parsimony.infra.nlp import split_sentences


class DerivedCache:
    def __init__(self, tokenizer, embedder=None) -> None:
        self._tok = tokenizer
        self._embedder = embedder
        self._tokens: dict[str, int] = {}
        self._sentences: dict[str, tuple[str, ...]] = {}
        self._query_emb = None
        self._turn_emb = None
        self._counts = {"token_count": 0, "token_count_hits": 0, "sentences": 0, "embed": 0}

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

    def query_embedding(self):
        raise FeatureNotAvailable(
            "Embeddings land in Sprint 2 with sentence-transformers (ADR-007). "
            "M1 tier 2 and M2's semantic tier use lexical similarity until then."
        )

    def turn_embeddings(self):
        raise FeatureNotAvailable("Embeddings land in Sprint 2 (ADR-007).")

    def stats(self) -> dict[str, int]:
        return dict(self._counts)
