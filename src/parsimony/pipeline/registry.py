"""Stage registry and boot-time DAG validation.

Stage order is a configuration value (ADR-002), which is what makes Gap 3
answerable. The price of that flexibility is that a misconfigured order must
fail loudly and immediately: a bad sweep cell should fail in the first second,
not after six unattended CPU-hours.
"""

from __future__ import annotations

from dataclasses import dataclass

from parsimony.core.config import PLANNED_STAGES, ParsimonyConfig
from parsimony.core.errors import ConfigError

# Fields present on RequestContext before any stage runs.
INITIAL_FIELDS: frozenset[str] = frozenset(
    {
        "query",
        "history",
        "system_prompt",
        "context_digest",
        "invariants",
        "original_query",
        "original_history",
        "conversation_id",
        "turn_index",
    }
)


@dataclass(frozen=True, slots=True)
class PlannedStage:
    """Placeholder for a stage declared in stage_order but not yet built.

    Surfacing these in the trace means every run shows the roadmap, rather than
    a later sprint's absence being silently invisible.
    """

    name: str
    module_id: str = "--"
    reads: frozenset[str] = frozenset()
    writes: frozenset[str] = frozenset()


class StageRegistry:
    def __init__(self) -> None:
        self._stages: dict[str, object] = {}

    def register(self, stage) -> None:
        if stage.name in self._stages:
            raise ConfigError(f"stage already registered: {stage.name}")
        self._stages[stage.name] = stage

    def register_all(self, stages) -> None:
        for s in stages:
            self.register(s)

    def get(self, name: str):
        return self._stages.get(name)

    def ordered(self, cfg: ParsimonyConfig) -> list:
        out: list = []
        for name in cfg.stage_order:
            stage = self._stages.get(name)
            if stage is not None:
                out.append(stage)
            elif name in PLANNED_STAGES:
                out.append(PlannedStage(name=name))
            else:
                raise ConfigError(
                    f"stage {name!r} is in stage_order but is neither registered "
                    f"nor declared in PLANNED_STAGES"
                )
        return out

    def validate(self, cfg: ParsimonyConfig) -> None:
        """Topological check: a stage may only read fields that already exist."""
        available = set(INITIAL_FIELDS)
        for stage in self.ordered(cfg):
            missing = set(stage.reads) - available
            if missing:
                raise ConfigError(
                    f"stage {stage.name!r} reads {sorted(missing)} which no earlier "
                    f"stage produces — check stage_order"
                )
            available |= set(stage.writes)


def default_registry(cache=None) -> StageRegistry:
    """Wire the stages that exist today. Later sprints append here."""
    from parsimony.modules.m1_compressor import stages as m1_stages
    from parsimony.modules.m2_cache import CacheLookupStage, SemanticCache
    from parsimony.modules.m3_history import stages as m3_stages
    from parsimony.modules.m4_assembler import stages as m4_stages
    from parsimony.modules.m5_budgeter import OutputBudgeter
    from parsimony.modules.m6_router import DeterministicRouterStage

    reg = StageRegistry()
    reg.register(DeterministicRouterStage())
    reg.register(CacheLookupStage(cache if cache is not None else SemanticCache()))
    reg.register_all(m3_stages())
    reg.register_all(m1_stages())
    reg.register_all(m4_stages())
    reg.register(OutputBudgeter())
    return reg
