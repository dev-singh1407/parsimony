"""L0 — core contracts. Standard library only (docs/00-architecture.md 2)."""

from parsimony.core.config import ParsimonyConfig, baseline, factorial_cells, full_stack
from parsimony.core.ledger import GateEvent, LedgerRow, StageOutcome, StageTrace
from parsimony.core.proposals import ContextPatch, NoOp, Proposal, ShortCircuit, TransformKind
from parsimony.core.types import (
    AssembledPrompt,
    GenParams,
    Invariants,
    InvariantClass,
    Mode,
    RequestContext,
    ResponseClass,
    RouteTier,
    TokenEvent,
    Turn,
)

__all__ = [
    "AssembledPrompt",
    "ContextPatch",
    "GateEvent",
    "GenParams",
    "InvariantClass",
    "Invariants",
    "LedgerRow",
    "Mode",
    "NoOp",
    "ParsimonyConfig",
    "Proposal",
    "RequestContext",
    "ResponseClass",
    "RouteTier",
    "ShortCircuit",
    "StageOutcome",
    "StageTrace",
    "TokenEvent",
    "TransformKind",
    "Turn",
    "baseline",
    "factorial_cells",
    "full_stack",
]
