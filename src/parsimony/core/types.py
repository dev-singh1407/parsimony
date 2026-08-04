"""L0 domain types. Standard library only — see docs/00-architecture.md 2.

RequestContext is frozen. Modules never mutate it; they return proposals and the
orchestrator commits them with dataclasses.replace. That is what makes revert
free (ADR-001): reverting is simply not assigning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:  # pragma: no cover
    from parsimony.core.protocols import Derived

Role = Literal["system", "user", "assistant"]


class ResponseClass(Enum):
    FACTUAL = "factual"
    ARITHMETIC = "arithmetic"
    REASONING = "reasoning"
    CODE = "code"
    SUMMARISATION = "summarisation"
    FOLLOW_UP = "follow_up"


class RouteTier(Enum):
    """Ordered by cost. Tiers 0-2 consume zero model tokens."""

    DETERMINISTIC = 0
    CACHE_EXACT = 1
    CACHE_SEMANTIC = 2
    MODEL_SMALL = 3
    MODEL_LARGE = 4

    @property
    def uses_model(self) -> bool:
        return self.value >= RouteTier.MODEL_SMALL.value


class InvariantClass(Enum):
    NUMBER = "number"
    ENTITY = "entity"
    NEGATION = "negation"
    QUOTED = "quoted"


class Mode(Enum):
    SERVE = "serve"
    EXPERIMENT = "experiment"


@dataclass(frozen=True, slots=True)
class Turn:
    turn_id: str
    role: Role
    content: str
    created_at: float = 0.0
    token_count: int = 0

    def with_tokens(self, n: int) -> Turn:
        return Turn(self.turn_id, self.role, self.content, self.created_at, n)


def _escape_boundary(value: str) -> str:
    """Word-boundary pattern that also works for values starting/ending in punctuation.

    re.escape + \\b fails for '3.5kg' (ends alphanumeric, fine) but also for
    values like '"quoted"'. Fall back to a plain containment test when the value
    has no word character at the relevant edge.
    """
    esc = re.escape(value)
    left = r"\b" if value[:1].isalnum() else ""
    right = r"\b" if value[-1:].isalnum() else ""
    return f"{left}{esc}{right}"


def _present(value: str, text: str) -> bool:
    if not value:
        return True
    return re.search(_escape_boundary(value), text, flags=re.IGNORECASE) is not None


@dataclass(frozen=True, slots=True)
class Invariants:
    """The fidelity fingerprint.

    Extracted exactly once per request (docs/01-pipeline-stages.md Stage 2) so
    that every gate check is a set difference rather than a re-parse. With up to
    seven proposals per request, re-extracting per check would cost ~70ms of the
    120ms overhead budget.
    """

    numbers: frozenset[str] = frozenset()
    entities: frozenset[str] = frozenset()
    negations: frozenset[str] = frozenset()
    quoted: frozenset[str] = frozenset()

    def missing_from(self, text: str) -> dict[InvariantClass, frozenset[str]]:
        """Invariant values present in self but absent from `text`.

        An empty dict means fidelity is preserved.
        """
        lost: dict[InvariantClass, frozenset[str]] = {}
        for cls, values in (
            (InvariantClass.NUMBER, self.numbers),
            (InvariantClass.ENTITY, self.entities),
            (InvariantClass.NEGATION, self.negations),
            (InvariantClass.QUOTED, self.quoted),
        ):
            gone = frozenset(v for v in values if not _present(v, text))
            if gone:
                lost[cls] = gone
        return lost

    def union(self, other: Invariants) -> Invariants:
        return Invariants(
            numbers=self.numbers | other.numbers,
            entities=self.entities | other.entities,
            negations=self.negations | other.negations,
            quoted=self.quoted | other.quoted,
        )

    def total(self) -> int:
        return len(self.numbers) + len(self.entities) + len(self.negations) + len(self.quoted)


@dataclass(frozen=True, slots=True)
class AssembledPrompt:
    """Two-zone prompt. Only M4 may write the invariant zone (docs/02-module-specs.md M4)."""

    invariant_zone: str
    volatile_zone: str
    full_text: str
    prefix_token_count: int
    total_token_count: int


@dataclass(frozen=True, slots=True)
class GenParams:
    num_predict: int = 256
    temperature: float = 0.0
    stop: tuple[str, ...] = ()
    seed: int = 0


@dataclass(frozen=True, slots=True)
class TokenEvent:
    """One streamed token. emitted_at_ns is perf_counter_ns at receipt, which is
    what makes TTFT and TPOT measurable (docs/01-pipeline-stages.md Stage 8)."""

    text: str
    index: int
    emitted_at_ns: int


@dataclass(frozen=True, slots=True)
class RequestContext:
    request_id: str
    conversation_id: str

    # Immutable reference. Written once in ingestion; the fidelity gate's ground truth.
    original_query: str
    original_history: tuple[Turn, ...]
    invariants: Invariants

    # Working state. Only the orchestrator writes, and only via committed patches.
    query: str
    history: tuple[Turn, ...]
    system_prompt: str = ""
    context_digest: str = ""

    turn_index: int = 0
    config_hash: str = ""
    corpus_hash: str | None = None

    # Decisions accumulated by stages.
    response_class: ResponseClass | None = None
    complexity: float | None = None
    output_budget: int | None = None
    route_tier: RouteTier | None = None
    assembled: AssembledPrompt | None = None

    derived: "Derived | None" = field(default=None, compare=False, repr=False)

    def text_payload(self) -> str:
        """The concatenation a REWRITE proposal is fidelity-checked against.

        The system prompt and context digest are excluded: they are the invariant
        zone and no module may rewrite them, so including them would only dilute
        the check.
        """
        parts = [t.content for t in self.history]
        parts.append(self.query)
        return "\n".join(parts)

    def original_payload(self) -> str:
        parts = [t.content for t in self.original_history]
        parts.append(self.original_query)
        return "\n".join(parts)
