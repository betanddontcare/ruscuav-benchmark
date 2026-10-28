#!/usr/bin/env python3
"""Extract the legacy environment code, verbatim, with its dependency closure.

The legacy notebook code (src/ruscuav_benchmark/legacy/definitions.py plus the
submission-repair cells) defines the cargo-UAV environments as a chain of classes
spread over 25,000 lines. This tool finds every top-level statement the requested
root names depend on (functions, classes, constants and imports, followed
transitively through the names they load) and writes them, unchanged and in their
original order, to one module. It was the first step of the environment refactor:
the closure lists everything the notebook environments depend on, and the
extracted module, run as a candidate of env_parity.py, behaves exactly like the
notebook code. The notebook-cell statements restored by the legacy runtime
(NOTEBOOK_STATE_PATCHES) are included.

Usage: python scripts/migration/extract_legacy_env.py OUTPUT.py
"""

from __future__ import annotations

import argparse
import ast
import builtins
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from ruscuav_benchmark.legacy.runtime import apply_notebook_state_patches  # noqa: E402

SOURCES = [
    REPO / "src" / "ruscuav_benchmark" / "legacy" / "definitions.py",
    REPO / "code" / "submission_repair" / "RUSCUAV_submission_repair_cells.py",
]
ROOTS = [
    # environment classes
    "RandomizedCurriculumEnvV14", "RealisticUrbanEnvConfig", "RealisticUrbanCargoUAVEnv",
    "RealisticUrbanCargoUAVEnvM4", "RealisticUrbanCargoUAVEnvM11Dynamic",
    "RealisticUrbanCargoUAVEnvM11DynamicV2", "RealisticUrbanCargoUAVEnvM11CDynamicObs",
    "M11DynamicRiskConfig", "M11DynamicRiskField",
    # factories used by the reported evaluations
    "make_ruscuav_env_m4", "make_ruscuav_env_m4_mode", "make_ruscuav_m8_ablation_env_m4_mode",
    "make_ruscuav_geo_baseline_env_m4_mode",
    "make_ruscuav_env_m11_dynamic_mode", "make_ruscuav_env_m11c_dynamic_obs_mode",
    "make_m12c_real_vienna_env_mode", "make_geo_eval_env", "make_static_eval_env",
    # evaluation regimes and reported configurations
    "EvaluationRegime", "PRIMARY_NO_TERMINAL", "SECONDARY_OPERATIONAL_STOP",
    "DEFAULT_BUDGETS", "REPORTED_DYNAMIC_CONFIG", "reported_dynamic_config",
    # bounds used at call time
    "MIN_X", "MAX_X", "MIN_Y", "MAX_Y", "MIN_Z", "MAX_Z",
]
BUILTINS = set(dir(builtins))


def bound_names(node: ast.stmt) -> set[str]:
    """Names a top-level statement binds (or, for a method call, mutates) in the module namespace."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return {(a.asname or a.name).split(".")[0] for a in node.names}
    if isinstance(node, ast.Expr):  # e.g. M8_VARIANTS["reward_only"].setdefault(...)
        target = node.value.func.value
        while isinstance(target, (ast.Attribute, ast.Subscript)):
            target = target.value
        return {target.id} if isinstance(target, ast.Name) else set()
    names = set()
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            for sub in ast.walk(target):
                if isinstance(sub, ast.Name):
                    names.add(sub.id)
    return names


def loaded_names(node: ast.AST) -> set[str]:
    """Names a top-level statement may read from the module namespace.

    For functions and classes, names assigned anywhere inside (and parameters) are
    local in Python, so they are excluded; e.g. a local ``ox`` must not pull in
    ``import osmnx as ox``. The result can still over-approximate, never under.
    """
    loads = {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        local = set()
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, (ast.Store, ast.Del)):
                local.add(sub.id)
            elif isinstance(sub, ast.arg):
                local.add(sub.arg)
            elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and sub is not node:
                local.add(sub.name)
        declared_global = {n for sub in ast.walk(node) if isinstance(sub, ast.Global) for n in sub.names}
        loads -= local - declared_global
        if isinstance(node, ast.ClassDef):  # base classes and decorators are evaluated at module level
            for expr in node.bases + node.decorator_list:
                loads |= {n.id for n in ast.walk(expr) if isinstance(n, ast.Name)}
    return loads - BUILTINS


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("output", type=Path, help="module file to write")
    output = ap.parse_args().output
    statements = []  # (source_index, node, text)
    for index, path in enumerate(SOURCES):
        text = path.read_text(encoding="utf-8")
        if index == 0:
            text = apply_notebook_state_patches(text)
        tree = ast.parse(text)
        for node in tree.body:
            mutating_call = (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                             and isinstance(node.value.func, ast.Attribute))
            if mutating_call or isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import,
                                                  ast.ImportFrom, ast.Assign, ast.AnnAssign)):
                segment = ast.get_source_segment(text, node, padded=False)
                decorators = getattr(node, "decorator_list", [])
                if decorators:  # include decorator lines
                    lines = text.splitlines()
                    segment = "\n".join(lines[decorators[0].lineno - 1: node.end_lineno])
                statements.append((index, node, segment))

    definers: dict[str, list[int]] = {}
    for i, (_, node, _) in enumerate(statements):
        for name in bound_names(node):
            definers.setdefault(name, []).append(i)

    needed: set[int] = set()
    queue = list(ROOTS)
    seen_names: set[str] = set()
    while queue:
        name = queue.pop()
        if name in seen_names:
            continue
        seen_names.add(name)
        if name not in definers:
            continue
        for i in definers[name]:
            if i in needed:
                continue
            needed.add(i)
            queue.extend(loaded_names(statements[i][1]))

    missing_roots = [r for r in ROOTS if r not in definers]
    if missing_roots:
        raise SystemExit(f"roots not defined anywhere: {missing_roots}")

    header = (
        '"""Legacy cargo-UAV environment code, extracted verbatim by\n'
        "scripts/migration/extract_legacy_env.py from the archived notebook definitions and\n"
        'submission-repair cells. Do not edit by hand; regenerate instead."""\n'
        "# flake8: noqa\n"
    )
    parts = [header]
    for i in sorted(needed):
        source_index, node, segment = statements[i]
        parts.append(f"# --- {SOURCES[source_index].name}:{node.lineno}\n{segment}\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts), encoding="utf-8")

    uses_globals = [statements[i][1].lineno for i in sorted(needed)
                    if "globals()" in statements[i][2]]
    print(f"wrote {output}: {len(needed)} statements, {sum(len(statements[i][2].splitlines()) for i in needed)} lines")
    print("statements calling globals():", uses_globals)
    return 0


if __name__ == "__main__":
    sys.exit(main())
