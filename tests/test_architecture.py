"""Architectural constraints, enforced rather than documented.

docs/00-architecture.md claims dependencies point strictly downward through the
L0-L5 layering, and that L0 is standard-library-only. Those claims were true
when written and would silently rot the first time someone reached upward for a
convenient helper — which is exactly what happened (M7 imported from eval).

This is deliberately hand-rolled over `ast` rather than pulling in
import-linter: it is forty lines, it runs inside the existing suite with no new
dependency, and the failure message can explain *why* the rule exists.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "parsimony"

# Lower number = lower layer. A module may import its own layer and below.
#
# `eval` sits BELOW `surfaces`, which corrects the ordering in the original
# architecture sketch. The CLI legitimately drives benchmarks, calibration and
# mining, so it depends on the evaluation layer; nothing in eval depends on a
# surface. Dependency direction, not conceptual importance, decides the order.
LAYERS: dict[str, int] = {
    "core": 0,
    "infra": 1,
    "modules": 2,
    "pipeline": 3,
    "eval": 4,
    "surfaces": 5,
}

# Third-party packages L0 may not import at runtime. Keeping core at
# stdlib-only is what makes a unit test of the compressor free of a web
# framework, and swapping the vector index a non-core change.
THIRD_PARTY = {"numpy", "tokenizers", "typer", "rich", "pytest", "sentence_transformers",
               "spacy", "httpx", "faiss", "pydantic", "fastapi", "scipy", "pandas"}


def _python_files() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def _layer_of(path: Path) -> tuple[str, int] | None:
    rel = path.relative_to(SRC)
    if not rel.parts or rel.parts[0] not in LAYERS:
        return None
    return rel.parts[0], LAYERS[rel.parts[0]]


def _imports(path: Path, runtime_only: bool = True) -> list[tuple[str, int]]:
    """(module, lineno) for each import. TYPE_CHECKING blocks are excluded when
    runtime_only, since those cost nothing at run time and exist precisely so
    L0 can name types it must not depend on."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    guarded: set[int] = set()
    if runtime_only:
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                test = ast.dump(node.test)
                if "TYPE_CHECKING" in test:
                    for child in ast.walk(node):
                        guarded.add(getattr(child, "lineno", -1))

    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if node.lineno in guarded:
            continue
        if isinstance(node, ast.Import):
            found += [(a.name, node.lineno) for a in node.names]
        elif node.module and node.level == 0:
            found.append((node.module, node.lineno))
    return found


class TestLayering:
    def test_dependencies_point_downward(self):
        violations = []
        for path in _python_files():
            layer = _layer_of(path)
            if layer is None:
                continue
            name, level = layer
            for module, lineno in _imports(path):
                if not module.startswith("parsimony."):
                    continue
                parts = module.split(".")
                if len(parts) < 2 or parts[1] not in LAYERS:
                    continue
                target, target_level = parts[1], LAYERS[parts[1]]
                if target_level > level:
                    violations.append(
                        f"{path.relative_to(SRC)}:{lineno} — {name}(L{level}) imports "
                        f"{target}(L{target_level})"
                    )
        assert not violations, (
            "Upward imports break the layering in docs/00-architecture.md. A module that "
            "can reach the evaluation layer can be changed by it, and the ablation stops "
            "being independent:\n  " + "\n  ".join(violations)
        )

    def test_core_imports_no_third_party_at_runtime(self):
        violations = []
        for path in sorted((SRC / "core").rglob("*.py")):
            for module, lineno in _imports(path):
                root = module.split(".")[0]
                if root in THIRD_PARTY:
                    violations.append(f"{path.relative_to(SRC)}:{lineno} — {module}")
        assert not violations, (
            "L0 must stay standard-library-only, so swapping an implementation is never a "
            "core change and unit-testing a module never drags in a framework:\n  "
            + "\n  ".join(violations)
        )

    def test_modules_do_not_import_each_other(self):
        """Modules coordinate through the orchestrator, never directly.

        A module that calls another cannot be ablated independently: disabling
        it would not remove its effect, and the factorial cell would be a lie.
        Same-family imports (m1_compressor -> m1_tier3) are fine.
        """
        violations = []
        for path in sorted((SRC / "modules").rglob("*.py")):
            family = path.stem.split("_")[0]
            for module, lineno in _imports(path):
                if not module.startswith("parsimony.modules."):
                    continue
                other = module.split(".")[2]
                if other.split("_")[0] != family:
                    violations.append(
                        f"{path.relative_to(SRC)}:{lineno} — {path.stem} imports {other}"
                    )
        assert not violations, (
            "Modules must not import each other; coordination belongs to the orchestrator "
            "(ADR-001):\n  " + "\n  ".join(violations)
        )

    def test_every_source_file_sits_in_a_known_layer(self):
        stray = [
            str(p.relative_to(SRC))
            for p in _python_files()
            if p.name != "__init__.py" and _layer_of(p) is None
        ]
        assert not stray, f"files outside the layer scheme: {stray}"


class TestModuleContracts:
    @pytest.mark.parametrize("stage_file", sorted((SRC / "modules").glob("m*.py")))
    def test_thresholds_live_in_config_not_in_modules(self, stage_file):
        """ADR-008: every threshold lives in ParsimonyConfig and nowhere else.

        A hard-coded float makes M7's output un-loadable and the per-model
        calibration table impossible to assemble. Only a small set of
        structural constants is allowed.
        """
        allowed = {0.0, 1.0, 0.5, 2.0, 3.0, 100.0, 60.0, 0.25, 1000.0}
        tree = ast.parse(stage_file.read_text(encoding="utf-8"))
        offenders = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, float):
                if node.value not in allowed:
                    offenders.append(f"{stage_file.name}:{node.lineno} — {node.value}")
        assert not offenders, (
            "Float literals in a module are almost always a threshold that belongs in "
            "ParsimonyConfig (ADR-008):\n  " + "\n  ".join(offenders)
        )
