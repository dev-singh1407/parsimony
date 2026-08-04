"""Structural interfaces. Every one of these has a contract-test suite in
tests/contract/ that runs against *all* implementations — that is what makes
'components are replaceable' a checked claim rather than a hope.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, Protocol, runtime_checkable

from parsimony.core.proposals import Proposal
from parsimony.core.types import GenParams, RequestContext, TokenEvent

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

    from parsimony.core.config import ParsimonyConfig
    from parsimony.core.ledger import LedgerRow


@runtime_checkable
class Stage(Protocol):
    """The only contract a module implements."""

    module_id: str
    name: str
    reads: frozenset[str]
    writes: frozenset[str]

    def applies_to(self, ctx: RequestContext, cfg: "ParsimonyConfig") -> bool: ...

    def propose(self, ctx: RequestContext, cfg: "ParsimonyConfig") -> Proposal: ...


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...

    def count(self, text: str) -> int: ...

    def offsets(self, text: str) -> list[tuple[int, int]]: ...

    @property
    def id(self) -> str: ...


class Embedder(Protocol):
    """Batched by contract.

    There is deliberately no embed_one(): a convenience method would be called
    in a loop by someone in week 7 and quietly cost 5x (ADR-006). Removing the
    temptation is cheaper than policing it.
    """

    def embed(self, texts: list[str]) -> "np.ndarray": ...

    @property
    def dim(self) -> int: ...

    @property
    def id(self) -> str: ...


class VectorIndex(Protocol):
    def add(self, vec: "np.ndarray", entry_id: str) -> None: ...

    def search(self, vec: "np.ndarray", k: int) -> list[tuple[str, float]]: ...

    def remove(self, entry_id: str) -> None: ...

    def size(self) -> int: ...

    def is_exact(self) -> bool:
        """False disqualifies this index from false-hit-rate measurement (ADR-004).

        With an approximate index the measured false-hit rate is a mixture of
        'the policy was wrong' and 'the index missed the true neighbour', and
        those are not separable after the fact. The analysis layer refuses
        rather than silently producing a contaminated number.
        """
        ...


class LLMProvider(Protocol):
    def generate(self, prompt: str, params: GenParams) -> Iterator[TokenEvent]: ...

    @property
    def tokenizer_id(self) -> str: ...

    @property
    def model_digest(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    @property
    def quantisation(self) -> str: ...


class LedgerSink(Protocol):
    def write(self, row: "LedgerRow") -> None: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


class InvariantExtractor(Protocol):
    def extract(self, text: str) -> object: ...  # -> Invariants


class PiiDetector(Protocol):
    def spans(self, text: str) -> list[tuple[int, int, str]]: ...


class Derived(Protocol):
    """Per-request memo (ADR-006).

    Lazy: a request short-circuited at router tier 0 never pays for sentence
    splitting or embeddings. Modules must go through this rather than calling
    infrastructure directly.
    """

    def token_count(self, text: str) -> int: ...

    def sentences(self, text: str) -> tuple[str, ...]: ...

    def embed(self, texts: list[str]) -> "np.ndarray":
        """Memoised and batched.

        Keyed on the text itself rather than on 'the query' or 'the turns',
        because modules rewrite those mid-pipeline: a binding captured at
        ingestion would be stale by the time M1 has run. Text-keyed memoisation
        stays correct under rewriting and still collapses the four passes the
        naive design would make.
        """
        ...

    def stats(self) -> dict[str, int]: ...
