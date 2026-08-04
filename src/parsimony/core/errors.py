"""Core exception hierarchy.

Design rule (docs/00-architecture.md 7.3): optimisation modules are optional by
construction, so a module raising must never fail the request. The orchestrator
catches ModuleError at its boundary and degrades to NoOp("error"). Only
ProviderError is fatal to a request.
"""


class ParsimonyError(Exception):
    """Base for everything raised by this package."""


class ConfigError(ParsimonyError):
    """Configuration is internally inconsistent or violates the stage DAG."""


class ModuleError(ParsimonyError):
    """A module failed. Non-fatal: the orchestrator degrades to NoOp."""


class ProviderError(ParsimonyError):
    """The model provider is unreachable or misbehaving. Fatal to the request."""


class LedgerError(ParsimonyError):
    """A ledger write failed.

    Non-fatal in Mode.SERVE, fatal in Mode.EXPERIMENT — in an experiment the
    ledger *is* the result (ADR-005).
    """


class FeatureNotAvailable(ParsimonyError):
    """A capability scheduled for a later sprint was requested.

    Raised rather than silently returning a degraded answer, so that a missing
    dependency can never masquerade as a measured result.
    """
