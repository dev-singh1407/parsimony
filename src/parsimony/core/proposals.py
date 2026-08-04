"""What a module returns. Modules propose; the orchestrator commits (ADR-001)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Mapping, Union

from parsimony.core.types import RouteTier


class TransformKind(Enum):
    """Scopes the fidelity gate (ADR-003).

    M1 rewriting a sentence must preserve every number; M3 dropping a turn is
    *supposed* to remove its content. A single 'did the text change' check would
    either veto M3 permanently or wave M1 through. One enum field makes one gate
    implementation correct for all eight modules.
    """

    REWRITE = "rewrite"  # same information, new surface form -> full invariant check
    SELECT = "select"  # deliberate removal of whole units  -> retained units byte-identical
    AUGMENT = "augment"  # adds content, removes nothing      -> no check
    DECIDE = "decide"  # sets a decision field, no text edit -> no check


@dataclass(frozen=True, slots=True)
class ContextPatch:
    kind: TransformKind
    fields: Mapping[str, Any]
    rationale: str = ""
    evidence: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ShortCircuit:
    """Terminates the pipeline with an answer that cost no model tokens."""

    response: str
    served_by: RouteTier
    rationale: str = ""
    evidence: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class NoOp:
    reason: Literal["disabled", "not_applicable", "no_yield", "error"]
    detail: str = ""


Proposal = Union[ContextPatch, ShortCircuit, NoOp]
