"""Run the archived TOP_UAV notebook code outside Google Colab.

The legacy RUSCUAV code was developed as notebook cells that share one global
namespace and read and write a mounted Google Drive
(``/content/drive/MyDrive/RUSCUAV``). This module recreates that runtime locally:

* the legacy definitions and, optionally, the submission-repair cells are executed
  in order inside one module namespace, as the notebook executed its cells, so
  ``globals()`` lookups in the legacy code behave as they did in Colab;
* hard-coded Drive and Colab paths are rewritten to a local "virtual drive" before
  execution (the source files on disk are not modified);
* the virtual drive holds copies of the archived inputs from an extracted Zenodo
  archive, in the directory layout the notebook expected, and receives all new
  outputs. Because inputs are copied, legacy code that writes to Drive cannot
  change the archive.
"""

from __future__ import annotations

import filecmp
import re
import shutil
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
#: definitions rely on (for example as default argument values or registry entries).
#: The definitions-only extraction dropped them, so they are re-inserted, in memory,
#: where the notebook ran them: directly before the next definition in notebook order.
#: Entries: (text at column 0 to insert before, which must occur exactly once after a
#: blank line; code to insert, one simple statement per line; notebook cell of origin).
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
    (
        # The second definition of the factory, which reads urban_risk_lambda_mult.
        'def make_ruscuav_m8_ablation_env_m4_mode(\n    world,\n    variant_name,\n    stage="A",\n'
        '    mode="fast_train",\n    risk_profile="balanced",\n    n_obstacles=0,\n    seed=None,\n'
        "):\n    if variant_name not in M8_VARIANTS:",
        'M8_VARIANTS["risk_only"] = {"include_urban_vector_obs": False, "use_urban_risk_costs": True, '
        '"urban_risk_lambda_mult": 1.0, "green_bonus_lambda": 0.0, "urban_lidar_enabled": True, '
        '"urban_action_shield_enabled": True, "urban_geometric_shield_enabled": True, '
        '"description": "No semantic vector; semantic risk penalty active; green bonus disabled."}\n'
        'M8_VARIANTS["green_only"] = {"include_urban_vector_obs": False, "use_urban_risk_costs": True, '
        '"urban_risk_lambda_mult": 0.0, "green_bonus_lambda": 0.03, "urban_lidar_enabled": True, '
        '"urban_action_shield_enabled": True, "urban_geometric_shield_enabled": True, '
        '"description": "No semantic vector; semantic risk penalty disabled; green bonus active."}\n'
        # The cell's `if "urban_risk_lambda_mult" not in ...: ... = 1.0`, as one simple statement.
        'M8_VARIANTS["reward_only"].setdefault("urban_risk_lambda_mult", 1.0)\n',
        "ANALYSIS 5 / 6.1 (reward-factor variants risk_only and green_only)",
    ),
    (
        "def ma_pick_reward_model_path():",
        'M8_VARIANTS["reward_only_full_stack_diag"] = {"include_urban_vector_obs": False, '
        '"use_urban_risk_costs": True, "urban_risk_lambda_mult": 1.0, "green_bonus_lambda": 0.03, '
        '"urban_lidar_enabled": True, "urban_action_shield_enabled": True, '
        '"urban_geometric_shield_enabled": True, "description": "Reward-only policy eval; full shield stack."}\n'
        'M8_VARIANTS["reward_only_no_action_shield_diag"] = {"include_urban_vector_obs": False, '
        '"use_urban_risk_costs": True, "urban_risk_lambda_mult": 1.0, "green_bonus_lambda": 0.03, '
        '"urban_lidar_enabled": True, "urban_action_shield_enabled": False, '
        '"urban_geometric_shield_enabled": True, "description": "Reward-only policy eval; action shield disabled."}\n'
        'M8_VARIANTS["reward_only_no_geom_shield_diag"] = {"include_urban_vector_obs": False, '
        '"use_urban_risk_costs": True, "urban_risk_lambda_mult": 1.0, "green_bonus_lambda": 0.03, '
        '"urban_lidar_enabled": True, "urban_action_shield_enabled": True, '
        '"urban_geometric_shield_enabled": False, "description": "Reward-only policy eval; geometric shield disabled."}\n'
        'M8_VARIANTS["reward_only_no_shields_diag"] = {"include_urban_vector_obs": False, '
        '"use_urban_risk_costs": True, "urban_risk_lambda_mult": 1.0, "green_bonus_lambda": 0.03, '
        '"urban_lidar_enabled": True, "urban_action_shield_enabled": False, '
        '"urban_geometric_shield_enabled": False, '
        '"description": "Reward-only policy eval; both action and geometric shield disabled."}\n',
        "ANALYSIS 6 (shield-diagnostic variants of the reward-only policy)",
    ),
)


def apply_notebook_state_patches(source: str) -> str:
    """Re-insert the dropped notebook-cell statements listed in NOTEBOOK_STATE_PATCHES.

    The statements replace the blank line just above the anchor, joined with ``;``,
    so line numbers in tracebacks still match ``definitions.py`` on disk.
    """
    for anchor, code, origin in NOTEBOOK_STATE_PATCHES:
        occurrences = source.count("\n\n" + anchor)
        if occurrences != 1:
            raise ValueError(
                f"Patch anchor {anchor.splitlines()[0]!r} ({origin}) found {occurrences} times after a "
                "blank line in the legacy source; expected exactly once"
            )
        position = source.find("\n\n" + anchor)
        statements = "; ".join(line for line in code.splitlines() if line.strip())
        source = (
            source[: position + 1]
            + f"{statements}  # restored notebook state: {origin}"
            + source[position + 1 :]
        )
    return source


def prepare_virtual_drive(
    archive_root: str | Path,
    drive_root: str | Path,
    extra_inputs: Optional[Mapping[str | Path, str]] = None,
) -> Path:
    """Create the Drive layout the notebook expected, with copies of the archived inputs.

    ``extra_inputs`` maps further local files to Drive-relative paths, e.g. processed
    real-GIS worlds that the Zenodo archive does not distribute:
    ``{"/data/vienna.npz": "worlds/vienna_prater_leopoldstadt_osm_1km_v1_urban_world_layers.npz"}``.
    """
    archive_root = Path(archive_root).resolve()
    drive_root = Path(drive_root).resolve()
    drive_root.mkdir(parents=True, exist_ok=True)
    inputs = [(archive_root / rel, drive_rel) for rel, drive_rel in ARCHIVE_TO_DRIVE.items()]
    inputs += [(Path(src).resolve(), drive_rel) for src, drive_rel in (extra_inputs or {}).items()]
    for source, drive_rel in inputs:
        if not source.exists():
            raise FileNotFoundError(
                f"Input not found: {source}. Is {archive_root} an extracted RUSCUAV v2.0.0 Zenodo archive?"
            )
        target = drive_root / drive_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            target.unlink()
        if target.exists() and filecmp.cmp(source, target, shallow=False):
            continue
        shutil.copyfile(source, target)  # a copy, so writes in the drive never reach the archive
    return drive_root


def rewrite_paths(source: str, drive_root: Path, colab_local_root: Path) -> str:
    """Replace hard-coded Drive and Colab paths in legacy source text.

    All literals are replaced in one pass, so a replacement path that itself contains
    a literal (for example a work folder under /content/ruscuav_work) is left alone.
    A literal only matches as a whole path component: /content/data does not match
    /content/database.
    """
    replacements = {DRIVE_LITERAL: drive_root.as_posix()}
    for literal in COLAB_LOCAL_LITERALS:
        replacements[literal] = (colab_local_root / Path(literal).name).as_posix()
    pattern = re.compile(
        "(?:" + "|".join(re.escape(k) for k in sorted(replacements, key=len, reverse=True)) + r")(?![\w.-])"
    )
    return pattern.sub(lambda m: replacements[m.group(0)], source)


def load_legacy_namespace(
    archive_root: str | Path,
    work_dir: str | Path,
    include_repair_cells: bool = True,
    repair_cells_path: Optional[str | Path] = None,
    module_name: str = "ruscuav_legacy_notebook",
    extra_inputs: Optional[Mapping[str | Path, str]] = None,
) -> types.ModuleType:
    """Execute the legacy notebook code in one namespace and return it as a module.

    ``work_dir`` receives the virtual drive (``work_dir/drive``) and Colab-local
    folders (``work_dir/colab_local``). Legacy functions are then called as
    attributes of the returned module, e.g. ``ns.evaluate_m9_static_planner_drive``.
    ``extra_inputs`` is passed to :func:`prepare_virtual_drive`.
    """
    work_dir = Path(work_dir).resolve()
    drive_root = prepare_virtual_drive(archive_root, work_dir / "drive", extra_inputs)
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
        # dont_inherit: do not pass this module's `from __future__ import annotations`
        # to the legacy code, which ran without it in Colab.
        exec(compile(text, str(path), "exec", dont_inherit=True), module.__dict__)

    module.RUSCUAV_VIRTUAL_DRIVE = drive_root
    module.RUSCUAV_ARCHIVE_ROOT = Path(archive_root).resolve()
    return module
