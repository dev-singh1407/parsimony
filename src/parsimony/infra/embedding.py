"""Embeddings and the vector index.

WHAT THIS IS, HONESTLY
----------------------
HashingEmbedder is a *lexical* embedder: hashed character n-grams plus word
unigrams, sublinear term weighting, L2 normalised. It is not a semantic model.
It will score "rain" against "rainfall" highly (shared character n-grams) and
"car" against "automobile" at roughly zero.

It is the default because sentence-transformers pulls PyTorch — about 2 GB, a
long install, and real memory pressure on the target 8 GB CPU machine — for a
capability the pipeline does not yet need to be *correct*. Everything that
depends on having a vector space (the semantic cache tier, the three-zone
verifier, MMR history selection, the threshold sweep, the adversarial study)
becomes real and measurable with this in place, and swapping in MiniLM later is
one class behind the same protocol.

Two consequences that must be stated in the report rather than discovered:

1. Every similarity threshold calibrated against this embedder MUST be
   recalibrated when the encoder changes. That is not a wart — it is precisely
   the claim of Contribution 6 (a calibration table per configuration rather
   than one universal number), demonstrated on our own stack.

2. A lexical embedder is the *hard* case for the adversarial subset: two
   questions differing by one operative token score near 1.0, so the three-zone
   verifier has to do all the work. If the verifier holds up here it is not
   relying on the encoder to separate them for it.

The index is exact brute force (ADR-004): approximate search would make the
measured false-hit rate a mixture of policy error and recall error, and those
are not separable after the fact.
"""

from __future__ import annotations

import math
import re
import zlib
from collections import Counter

import numpy as np

_WORD_RE = re.compile(r"[a-z0-9]+")
_NORM_RE = re.compile(r"[^a-z0-9 ]+")


def _stable_hash(token: str) -> int:
    """CRC32, not Python's hash().

    hash() is salted per process, so an index written in one run would not match
    a query embedded in the next — silently, and only for string keys. That
    would corrupt every cross-run cache measurement in the project.
    """
    return zlib.crc32(token.encode("utf-8"))


class HashingEmbedder:
    def __init__(
        self,
        dim: int = 384,
        char_ngrams: tuple[int, ...] = (3, 4, 5),
        use_words: bool = True,
        word_weight: float = 1.5,
    ) -> None:
        self._dim = dim
        self._char_ngrams = char_ngrams
        self._use_words = use_words
        self._word_weight = word_weight
        self._cache: dict[str, np.ndarray] = {}

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def id(self) -> str:
        grams = "".join(str(n) for n in self._char_ngrams)
        return f"hashing-v1-d{self._dim}-c{grams}{'-w' if self._use_words else ''}"

    def _features(self, text: str) -> Counter[str]:
        norm = _NORM_RE.sub(" ", text.lower())
        norm = re.sub(r"\s+", " ", norm).strip()
        feats: Counter[str] = Counter()

        if self._use_words:
            for word in _WORD_RE.findall(norm):
                feats[f"w:{word}"] += self._word_weight

        padded = f" {norm} "
        for n in self._char_ngrams:
            if len(padded) < n:
                continue
            for i in range(len(padded) - n + 1):
                feats[f"c{n}:{padded[i : i + n]}"] += 1
        return feats

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        for feature, count in self._features(text).items():
            h = _stable_hash(feature)
            index = h % self._dim
            sign = 1.0 if (h >> 31) & 1 else -1.0  # signed hashing reduces collision bias
            vec[index] += sign * (1.0 + math.log(count))
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    def embed(self, texts: list[str]) -> np.ndarray:
        """Batched by contract (ADR-006). There is deliberately no embed_one()."""
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        out = np.empty((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            cached = self._cache.get(text)
            if cached is None:
                cached = self._embed_one(text)
                if len(self._cache) < 20_000:
                    self._cache[text] = cached
            out[i] = cached
        return out


class SentenceTransformerEmbedder:
    """Drop-in swap for HashingEmbedder. Requires the optional 'models' extra.

    Deliberately not the default: it pulls PyTorch. Install with
    `pip install -e ".[models]"` and set cfg.embedder_id to a model name.
    """

    def __init__(self, model_id: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        self._model = SentenceTransformer(model_id)
        self._model_id = model_id

    @property
    def dim(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    @property
    def id(self) -> str:
        return self._model_id

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.asarray(
            self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )


def get_embedder(embedder_id: str = "hashing-v1"):
    if embedder_id.startswith("hashing"):
        return HashingEmbedder()
    return SentenceTransformerEmbedder(embedder_id)


class ExactIndex:
    """Exact brute-force cosine search over L2-normalised vectors.

    At the cache sizes this project reaches (10^3-10^5) a full matmul is well
    under a millisecond, and exactness is what lets the false-hit rate be
    attributed to the similarity policy rather than to index recall (ADR-004).
    """

    def __init__(self, dim: int) -> None:
        self._dim = dim
        self._ids: list[str] = []
        self._rows: list[np.ndarray] = []
        self._matrix: np.ndarray | None = None
        self._position: dict[str, int] = {}

    def is_exact(self) -> bool:
        return True

    def size(self) -> int:
        return len(self._ids)

    def add(self, vec: np.ndarray, entry_id: str) -> None:
        if vec.shape != (self._dim,):
            raise ValueError(f"expected shape ({self._dim},), got {vec.shape}")
        existing = self._position.get(entry_id)
        if existing is not None:
            self._rows[existing] = vec
        else:
            self._position[entry_id] = len(self._ids)
            self._ids.append(entry_id)
            self._rows.append(vec)
        self._matrix = None

    def remove(self, entry_id: str) -> None:
        pos = self._position.pop(entry_id, None)
        if pos is None:
            return
        self._ids.pop(pos)
        self._rows.pop(pos)
        self._position = {eid: i for i, eid in enumerate(self._ids)}
        self._matrix = None

    def search(self, vec: np.ndarray, k: int) -> list[tuple[str, float]]:
        if not self._ids or k <= 0:
            return []
        if self._matrix is None:
            self._matrix = np.vstack(self._rows)
        scores = self._matrix @ vec
        k = min(k, len(self._ids))
        # argpartition is O(n); a full sort would be O(n log n) for no benefit
        # since only the top k are ever inspected.
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self._ids[i], float(scores[i])) for i in top]

    def clear(self) -> None:
        self._ids.clear()
        self._rows.clear()
        self._position.clear()
        self._matrix = None


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def mmr_select(
    candidate_vecs: np.ndarray,
    query_vec: np.ndarray,
    k: int,
    lambda_: float = 0.7,
) -> list[int]:
    """Maximal marginal relevance.

    argmax_i [ lambda * sim(c_i, q) - (1 - lambda) * max_j sim(c_i, s_j) ]

    One implementation, shared by M1 tier 2 and M3 — one lambda to explain in
    the report and one place for it to be wrong.
    """
    n = len(candidate_vecs)
    if n == 0 or k <= 0:
        return []
    k = min(k, n)
    relevance = candidate_vecs @ query_vec
    selected: list[int] = []
    remaining = set(range(n))

    while len(selected) < k and remaining:
        best_i, best_score = None, -math.inf
        for i in remaining:
            if selected:
                redundancy = float(np.max(candidate_vecs[selected] @ candidate_vecs[i]))
            else:
                redundancy = 0.0
            score = lambda_ * float(relevance[i]) - (1.0 - lambda_) * redundancy
            if score > best_score:
                best_i, best_score = i, score
        assert best_i is not None
        selected.append(best_i)
        remaining.discard(best_i)
    return selected
