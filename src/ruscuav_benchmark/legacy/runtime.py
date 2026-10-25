"""Run the archived TOP_UAV notebook code outside Google Colab.

The legacy RUSCUAV code was developed as notebook cells that share one global
namespace and read and write a mounted Google Drive
(``/content/drive/MyDrive/RUSCUAV``). This module recreates that runtime locally:

* the legacy definitions and, optionally, the submission-repair cells are executed
  in order inside one module namespace, as the notebook executed its cells, so
  ``globals()`` lookups in the legacy code behave as they did in Colab;
* hard-coded Drive and Colab paths are rewritten to a local "virtual drive" before
  execution (the source files on disk are not modified);
* the virtual drive exposes the archived inputs from an extracted Zenodo archive in
  the directory layout the notebook expected, and receives all new outputs.

The archived inputs are only read; nothing is written into the archive.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Mapping, Optional

#: Drive folder hard-coded throughout the notebook code.
DRIVE_LITERAL = "/content/drive/MyDrive/RUSCUAV"
#: Colab-local folders used by a few early notebook helpers.
COLAB_LOCAL_LITERALS = ("/content/ruscuav", "/content/data")

LEGACY_DIR = Path(__file__).resolve().parent
DEFINITIONS_PATH = LEGACY_DIR / "definitions.py"
#: Location of the submission-repair cells in a source checkout of the repository.
DEFAULT_REPAIR_CELLS_PATH = (
    LEGACY_DIR.parents[2] / "code" / "submission_repair" / "RUSCUAV_submission_repair_cells.py"
)

#: Archived files (relative to the extracted Zenodo archive) and where the notebook
#: expected them on Drive (relative to the Drive root).
ARCHIVE_TO_DRIVE: Mapping[str, str] = {
    "worlds/synthetic/AOI_FAKE_001_urban_world_layers.npz": "worlds/AOI_FAKE_001_urban_world_layers.npz",
}


#: Statements from notebook execution cells that set module-level state which later
#: definitions rely on (for example as default argument values). The definitions-only
#: extraction dropped them, so they are re-inserted, in memory, directly before the
#: first definition that needs them.
#: Entries: (line at column 0 to insert before, code to insert, notebook cell of origin).
NOTEBOOK_STATE_PATCHES = (
    (
        "def rasterize_mask(",
        "OUT_SHAPE = raster_shape()\nLOCAL_TRANSFORM = local_affine_transform()\n",
        "M12 REAL GIS AOI / BLOCK 5 (rasterization helpers)",
    ),
    (
        "def rollout_m13_no_terminal_with_telemetry(",
        "m13_planner_cfg = M10RecedingHorizonPlannerConfig(plan_res_m=5.0, planner_agl_m=30.0, "
        "risk_lambda=0.35, use_astar=True, sensing_radius_m=75.0, replan_every_steps=5, "
        "prior_risk_mode='perfect', hard_blockers_known=True, waypoint_radius_m=8.0, "
        "target_speed_mps=12.0, max_steps=1000)\n",
        "M13 / BLOCK 2 (telemetry rollouts for gate candidate mining)",
    ),
)


def apply_notebook_state_patches(source: str) -> str:
    """Re-insert the dropped notebook-cell statements listed in NOTEBOOK_STATE_PATCHES.

    The statements replace the blank line just above the anchor, joined with ``;``,
    so line numbers in tracebacks still match ``definitions.py`` on disk.
    """
    for anchor, code, origin in NOTEBOOK_STATE_PATCHES:
        position = source.find("\n\n" + anchor)
        if position < 0:
            raise ValueError(
                f"Patch anchor {anchor!r} ({origin}) not found after a blank line in the legacy source"
            )
        statements = "; ".join(line for line in code.splitlines() if line.strip())
        source = (
            source[: position + 1]
            + f"{statements}  # restored notebook state: {origin}"
            + source[position + 1 :]
        )
    return source


def prepare_virtual_drive(archive_root: str | Path, drive_root: str | Path) -> Path:
    """Create the Drive layout the notebook expected, linking archived inputs read-only."""
    archive_root = Path(archive_root).resolve()
    drive_root = Path(drive_root).resolve()
    drive_root.mkdir(parents=True, exist_ok=True)
    for archive_rel, drive_rel in ARCHIVE_TO_DRIVE.items():
        source = archive_root / archive_rel
        if not source.exists():
            raise FileNotFoundError(
                f"Archived input not found: {source}. Is {archive_root} an extracted "
                "RUSCUAV v2.0.0 Zenodo archive?"
            )
        target = drive_root / drive_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink() or target.exists():
            if target.resolve() == source:
                continue
            raise FileExistsError(f"{target} exists and does not point to {source}")
        target.symlink_to(source)
    return drive_root


def rewrite_paths(source: str, drive_root: Path, colab_local_root: Path) -> str:
    """Replace hard-coded Drive and Colab paths in legacy source text."""
    source = source.replace(DRIVE_LITERAL, drive_root.as_posix())
    for literal in COLAB_LOCAL_LITERALS:
        source = source.replace(literal, (colab_local_root / Path(literal).name).as_posix())
    return source


def load_legacy_namespace(
    archive_root: str | Path,
    work_dir: str | Path,
    include_repair_cells: bool = True,
    repair_cells_path: Optional[str | Path] = None,
    module_name: str = "ruscuav_legacy_notebook",
) -> types.ModuleType:
    """Execute the legacy notebook code in one namespace and return it as a module.

    ``work_dir`` receives the virtual drive (``work_dir/drive``) and Colab-local
    folders (``work_dir/colab_local``). Legacy functions are then called as
    attributes of the returned module, e.g. ``ns.evaluate_m9_static_planner_drive``.
    """
    work_dir = Path(work_dir).resolve()
    drive_root = prepare_virtual_drive(archive_root, work_dir / "drive")
    colab_local_root = work_dir / "colab_local"
    colab_local_root.mkdir(parents=True, exist_ok=True)

    sources = [DEFINITIONS_PATH]
    if include_repair_cells:
        repair = Path(repair_cells_path) if repair_cells_path else DEFAULT_REPAIR_CELLS_PATH
        if not repair.exists():
            raise FileNotFoundError(
                f"Submission-repair cells not found: {repair}. Pass repair_cells_path."
            )
        sources.append(repair)

    module = types.ModuleType(module_name)
    module.__file__ = str(DEFINITIONS_PATH)
    # Dataclasses and pickling look classes up through sys.modules.
    sys.modules[module_name] = module
    for path in sources:
        text = rewrite_paths(path.read_text(encoding="utf-8"), drive_root, colab_local_root)
        if path == DEFINITIONS_PATH:
            text = apply_notebook_state_patches(text)
        exec(compile(text, str(path), "exec"), module.__dict__)

    module.RUSCUAV_VIRTUAL_DRIVE = drive_root
    module.RUSCUAV_ARCHIVE_ROOT = Path(archive_root).resolve()
    return module
