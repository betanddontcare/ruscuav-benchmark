"""Command-line interface: ``ruscuav <command>``.

Commands:
  validate   check world NPZ files against the ruscuav-world schema
  evaluate   fly a planner of the paper or a policy on a world, or score flown trajectories
             (same as ``ruscuav-evaluate``)
  score      the paper's tables and paired tests for per-episode results
  protocol   list, show or create evaluation protocols
  build-aoi  build a world NPZ from OpenStreetMap data
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any, Dict, List, Optional

from . import __version__


def _cmd_validate(args: argparse.Namespace) -> int:
    from .validate import validate_world_npz

    reports = [validate_world_npz(path) for path in args.paths]
    if args.json:
        print(json.dumps([r.to_dict() for r in reports], indent=2))
    else:
        print("\n\n".join(r.format() for r in reports))
    return 0 if all(r.ok for r in reports) else 1


def _config_changes(items: List[str], kind: str) -> Dict[str, Any]:
    """``--set FIELD=VALUE`` items, parsed by the type of each configuration field."""
    from dataclasses import fields

    from .planners import PlannerConfig, RecedingPlannerConfig

    config_class = PlannerConfig if kind == "static" else RecedingPlannerConfig
    types = {f.name: (f.type if isinstance(f.type, str) else f.type.__name__) for f in fields(config_class)}
    if kind == "dynamic":
        types["dynamic_prior_mode"] = "str"
    changes: Dict[str, Any] = {}
    for item in items:
        key, sep, text = (part.strip() for part in item.partition("="))
        if not sep or not key:
            raise ValueError(f"--set expects FIELD=VALUE, got {item!r}")
        if key not in types:
            raise ValueError(f"--set {key}: not a setting of this planner; settings: {', '.join(types)}")
        kind_of = types[key]
        try:
            if kind_of == "bool":
                lowered = text.lower()
                if lowered not in ("true", "false", "1", "0", "yes", "no"):
                    raise ValueError
                value: Any = lowered in ("true", "1", "yes")
            elif kind_of == "int":
                value = int(text)
            elif kind_of == "float":
                value = float(text)
            else:
                value = text
        except ValueError:
            expected = "true or false" if kind_of == "bool" else f"a {kind_of}"
            raise ValueError(f"--set {key}={text!r}: expected {expected}") from None
        changes[key] = value
    return changes


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from .scenarios import missions_from_csv
    from .world import UrbanWorldLayers

    sources = [name for name, value in (("--policy", args.policy), ("--checkpoint", args.checkpoint),
                                        ("--evaluation", args.evaluation), ("--trajectories", args.trajectories))
               if value is not None]
    policy_mode = bool(sources) and sources[0] != "--trajectories"
    kind = "trajectories" if args.trajectories is not None else "policy" if policy_mode else "planner"
    try:
        if len(sources) > 1:
            raise ValueError(f"give one of --policy, --checkpoint, --evaluation and --trajectories, not "
                             f"{' and '.join(sources)}")
        if sources and args.planner is not None:
            raise ValueError(f"give --planner or {sources[0]}, not both")
        if sources and args.set:
            raise ValueError("--set changes planner settings; it does not apply to policies or trajectories")
        if kind == "planner" and (args.method or args.format or args.label or args.stochastic or args.archive):
            raise ValueError("--method, --format, --label, --stochastic and --archive apply to "
                             "--policy, --checkpoint and --evaluation")
        if args.protocol is not None:
            given = [flag for flag, value in (("--scenarios", args.scenarios), ("--seed0", args.seed0),
                                              ("--regime", args.regime), ("--profile", args.profile))
                     if value is not None]
            if given:
                raise ValueError(f"{', '.join(given)}: the protocol fixes the missions, the regime and the risk "
                                 "profile")
        if kind == "trajectories":
            name, result = _evaluate_trajectories(args)
        elif args.evaluation is not None:
            if args.protocol is not None:
                raise ValueError("--protocol does not apply to --evaluation, which re-runs an archived evaluation "
                                 "with its own settings")
            name, result = _evaluate_archived(args)
        else:
            missions = None
            if args.protocol is not None:
                from .protocols import load_protocol, protocol_world

                world = protocol_world(load_protocol(args.protocol), args.world)
            else:
                if args.world is None:
                    raise ValueError("--world is required (or --protocol, with the protocol's world)")
                if args.seed0 is None:
                    args.seed0 = 5000
                world = UrbanWorldLayers.load_npz(args.world)
                if args.scenarios is not None:
                    missions = missions_from_csv(args.scenarios, world, seed0=args.seed0)
                    if not missions:
                        print(f"{args.scenarios} has no missions for world {world.aoi_id}", file=sys.stderr)
                        return 2
            if policy_mode:
                name, result = _evaluate_policy(args, world, missions)
            else:
                name, result = _evaluate_planner(args, world, missions)
    except (ValueError, TypeError, OSError, KeyError, ModuleNotFoundError) as exc:
        print(f"ruscuav evaluate: {exc}", file=sys.stderr)
        return 2
    s = result.summary
    keys = ["n_episodes", "success_rate", "mission_acceptability_rate", "acceptable_given_success_rate",
            "mean_steps", "termination_counts", "cargo_health_estimated", "episodes_outside_speed_limits",
            "episodes_outside_workspace", "episodes_with_hazard_segments"]
    regime = s["regime"]["name"] if isinstance(s.get("regime"), dict) else s.get("regime")
    out = {kind if kind != "trajectories" else "label": name, "world_id": s["world_id"], "regime": regime}
    if isinstance(s.get("protocol"), dict):
        out["protocol"] = s["protocol"]["name"]
        out["protocol_complete"] = s["protocol"]["complete"]
    print(json.dumps({**out, **{k: s[k] for k in keys if k in s},
                      "csv_path": None if result.csv_path is None else str(result.csv_path)}, indent=2))
    if kind == "trajectories":
        unflyable = {k: s[k] for k in ("episodes_outside_speed_limits", "episodes_outside_workspace",
                                        "episodes_with_hazard_segments") if s.get(k)}
        if unflyable:
            counts = ", ".join(f"{k}: {v}" for k, v in unflyable.items())
            print(f"note: some trajectories could not have been flown by the environment's vehicle ({counts}); "
                  "their scores are not comparable with the environment's methods (docs/TRAJECTORIES.md)",
                  file=sys.stderr)
    return 0


def _evaluate_trajectories(args: argparse.Namespace):
    from .trajectories import evaluate_trajectories

    if args.protocol is None:
        raise ValueError("--trajectories needs --protocol: trajectories are scored on a protocol's missions")
    given = [flag for flag, value in (("--method", args.method), ("--format", args.format), ("--archive", args.archive))
             if value is not None]
    given += ["--stochastic"] if args.stochastic else []
    given += ["--save-paths"] if args.save_paths else []
    if given:
        raise ValueError(f"{', '.join(given)}: not used when scoring trajectories")
    label = args.label or "TRAJECTORY"
    result = evaluate_trajectories(args.trajectories, args.protocol, args.world, label=label,
                                   n_episodes=args.episodes, output_dir=args.output_dir, prefix=args.prefix)
    return label, result


def _progress(args: argparse.Namespace, name: str):
    def progress(index: int, total: int, row: dict) -> None:
        if not args.quiet and ((index + 1) % 10 == 0 or index == 0 or index + 1 == total):
            print(f"[{name}] episode {index + 1}/{total}: seed {row['seed']}, {row['termination']}, "
                  f"{row['steps']} steps", file=sys.stderr, flush=True)

    return progress


def _evaluate_planner(args: argparse.Namespace, world, missions):
    from .planners import evaluate_planner, get_planner

    spec = get_planner(args.planner or "FULL_RH_Astar")
    spec = spec.with_config(**_config_changes(args.set, spec.kind))
    result = evaluate_planner(
        world, spec, regime=args.regime if args.protocol is None else None, missions=missions,
        n_episodes=args.episodes, seed0=5000 if args.seed0 is None else args.seed0, risk_profile=args.profile,
        protocol=args.protocol, output_dir=args.output_dir, prefix=args.prefix, save_paths=args.save_paths,
        progress=_progress(args, spec.name),
    )
    return spec.name, result


def _evaluate_policy(args: argparse.Namespace, world, missions):
    from .env import METHOD_PRESETS
    from .policies import ARCHIVED_CHECKPOINTS, archived_checkpoint, checkpoint_env_overrides, evaluate_policy

    overrides = None
    if args.checkpoint is not None:
        if args.archive is None:
            raise ValueError("--checkpoint needs --archive (the extracted v2.0.0 archive)")
        if args.checkpoint not in ARCHIVED_CHECKPOINTS:
            raise ValueError(f"unknown checkpoint {args.checkpoint!r}; available: {', '.join(sorted(ARCHIVED_CHECKPOINTS))}")
        policy = archived_checkpoint(args.checkpoint, args.archive)
        method = args.method or ARCHIVED_CHECKPOINTS[args.checkpoint].method
        overrides = checkpoint_env_overrides(args.checkpoint)
        name = args.checkpoint
    else:
        if args.archive is not None:
            raise ValueError("--archive applies to --checkpoint and --evaluation")
        if args.method is None:
            raise ValueError("--policy needs --method, the method preset whose observation the policy was trained with")
        policy, method, name = args.policy, args.method, args.policy
    if method not in METHOD_PRESETS:
        raise ValueError(f"unknown method {method!r}; available: {', '.join(METHOD_PRESETS)}")
    result = evaluate_policy(
        world, policy, method, regime=args.regime if args.protocol is None else None, missions=missions,
        n_episodes=args.episodes, seed0=5000 if args.seed0 is None else args.seed0, row_format=args.format,
        label=args.label, deterministic=not args.stochastic, risk_profile=args.profile, env_overrides=overrides,
        protocol=args.protocol, output_dir=args.output_dir, prefix=args.prefix, save_paths=args.save_paths,
        progress=_progress(args, name),
    )
    return name, result


def _evaluate_archived(args: argparse.Namespace):
    from .policies import ARCHIVED_EVALUATIONS, evaluate_archived
    from .world import UrbanWorldLayers

    if args.archive is None:
        raise ValueError("--evaluation needs --archive (the extracted v2.0.0 archive)")
    if args.evaluation not in ARCHIVED_EVALUATIONS:
        raise ValueError(f"unknown archived evaluation {args.evaluation!r}; available: "
                         f"{', '.join(ARCHIVED_EVALUATIONS)}")
    given = [flag for flag, value in (("--method", args.method), ("--format", args.format), ("--label", args.label),
                                      ("--scenarios", args.scenarios)) if value is not None]
    given += ["--stochastic"] if args.stochastic else []
    given += ["--regime"] if args.regime is not None else []
    given += ["--profile"] if args.profile is not None else []
    if given:
        raise ValueError(f"{', '.join(given)}: the archived evaluation fixes these settings")
    world = None if args.world is None else UrbanWorldLayers.load_npz(args.world)
    kwargs = {"seed0": args.seed0} if args.seed0 is not None else {}
    if args.episodes is not None:
        kwargs["n_episodes"] = args.episodes
    result = evaluate_archived(args.evaluation, args.archive, world=world, output_dir=args.output_dir,
                               prefix=args.prefix, save_paths=args.save_paths,
                               progress=_progress(args, args.evaluation), **kwargs)
    return args.evaluation, result


def _run_spec(spec: str):
    """``[LABEL[@SEED]=]PATH`` -> (label or None, training seed or None, path)."""
    from pathlib import Path

    if "=" in spec and not Path(spec).exists():
        left, path = spec.split("=", 1)
        label, _, seed = left.partition("@")
        try:
            train_seed = float(seed) if seed else None
        except ValueError:
            raise ValueError(f"{spec!r}: the training seed after @ must be a number") from None
        return (label or None), train_seed, path
    return None, None, spec


def _sha256(path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _recorded_settings(csv_path: Optional[str]) -> Optional[Dict[str, Any]]:
    """The settings an evaluation of this package recorded next to its
    ``<prefix>_episodes.csv`` (``<prefix>_config.json`` or ``<prefix>_summary.json``)."""
    from pathlib import Path

    if not csv_path or not str(csv_path).endswith("_episodes.csv"):
        return None
    path = Path(csv_path)
    stem = path.name[: -len("_episodes.csv")]
    for suffix in ("_config.json", "_summary.json"):
        candidate = path.with_name(stem + suffix)
        if candidate.is_file():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(data, dict):
                return data
    return None


def _check_against_protocol(run, protocol) -> Dict[str, Any]:
    """Refuse a run that was not evaluated under ``protocol``: dynamic risk where the
    protocol has none (or none where it has), another regime in the rows, another
    protocol, world or regime recorded next to the CSV, or seeds that are not the
    protocol's missions. Returns the protocol check of the run for score.json."""
    from .tables import is_missing
    from .world import canonical_aoi_id

    what = f"run {run.label!r}"
    dynamic_rows = any("dynamic_cumulative_risk" in r for r in run.rows)
    if protocol.dynamic and not dynamic_rows:
        raise ValueError(f"{what} has no dynamic risk; protocol {protocol.name!r} scores dynamic acceptability")
    if dynamic_rows and not protocol.dynamic:
        raise ValueError(f"{what} has dynamic risk; protocol {protocol.name!r} has none (score it under a dynamic "
                         "protocol such as canonical-dynamic)")
    regimes = sorted({str(r["evaluation_regime"]) for r in run.rows if not is_missing(r.get("evaluation_regime"))})
    if regimes and regimes != [protocol.regime]:
        raise ValueError(f"{what} was evaluated in the regime {', '.join(regimes)}; protocol {protocol.name!r} has "
                         f"{protocol.regime}")
    recorded = _recorded_settings(run.path)
    recorded_protocol = None
    if recorded is not None:
        record = recorded.get("protocol")
        if isinstance(record, dict) and record.get("digest"):
            recorded_protocol = record.get("name")
            if record["digest"] != protocol.digest():
                raise ValueError(f"{what} was flown under the protocol {record.get('name')!r} (digest "
                                 f"{str(record['digest'])[:12]}...), not under {protocol.name!r} (digest "
                                 f"{protocol.digest()[:12]}...)")
        else:
            world_id = recorded.get("world_id")
            if world_id is not None and canonical_aoi_id(str(world_id)) != protocol.world_id:
                raise ValueError(f"{what} was flown in the world {world_id}; protocol {protocol.name!r} is defined for "
                                 f"{protocol.world_id}")
            regime = recorded.get("regime")
            regime = regime.get("name") if isinstance(regime, dict) else regime
            if regime is not None and regime != protocol.regime:
                raise ValueError(f"{what} was flown in the regime {regime}; protocol {protocol.name!r} has "
                                 f"{protocol.regime}")
    check = protocol.check_seeds([r.get("seed") for r in run.rows], what=what)
    return {"label": run.label, "train_seed": run.train_seed, "complete": check["complete"],
            "missing_missions": len(check["missing"]), "recorded_protocol": recorded_protocol}


def _cmd_score(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .posthoc import DT_S
    from .scoring import PAIRED_METRICS, load_run, read_comparisons, read_registry, score_runs
    from .tables import is_missing, jsonable

    try:
        protocol = None
        if args.protocol is not None:
            from .protocols import load_protocol

            protocol = load_protocol(args.protocol)
        runs = read_registry(args.registry, exact_floats=args.exact_floats) if args.registry else []
        for spec in args.runs:
            label, seed, path = _run_spec(spec)
            run = load_run(path, label, train_seed=seed, exact_floats=args.exact_floats,
                           scope=args.scope if args.scope is not None else (protocol.name if protocol else ""))
            regimes = {str(r["evaluation_regime"]) for r in run.rows if not is_missing(r.get("evaluation_regime"))}
            run.regime = regimes.pop() if len(regimes) == 1 else (protocol.regime if protocol and not regimes else "")
            runs.append(run)
        if not runs:
            raise ValueError("give per-episode CSVs or --registry")
        checks = []
        if protocol is not None:
            for run in runs:
                checks.append(_check_against_protocol(run, protocol))
                # checked: the rows' regime, where they record one, is the protocol's
                run.dynamic, run.regime = protocol.dynamic, protocol.regime
        comparisons = []
        for item in args.compare:
            if Path(item).is_file():
                comparisons += read_comparisons(item)
                continue
            parts = item.split(":")
            if len(parts) != 3 or parts[2] not in PAIRED_METRICS:
                raise ValueError(f"--compare {item!r}: expected a CSV file or A:B:METRIC with METRIC one of "
                                 f"{', '.join(PAIRED_METRICS)}")
            a, b, metric = parts
            comparisons.append({"family": "comparisons", "comparison_id": f"{a}_vs_{b}_{metric}", "method_a": a,
                                "method_b": b, "metric": metric, "claim_driving": False})
        tables = score_runs(runs, comparisons or None, budgets=None if protocol is None else protocol.budgets,
                            dt_s=DT_S if protocol is None else protocol.dt_s)
        written = tables.write(args.output_dir)
        report = {
            "protocol": None if protocol is None else {"name": protocol.name, "digest": protocol.digest(),
                                                       "acceptability": protocol.acceptability},
            "runs": [{"label": r.label, "train_seed": r.train_seed, "csv_path": r.source, "read_from": r.path,
                      "episodes": len(r.rows), "dynamic": r.dynamic,
                      "sha256": _sha256(r.path) if r.path and Path(r.path).is_file() else None} for r in runs],
            "protocol_checks": checks,
            "tables": {name: str(path) for name, path in written.items()},
            "summaries": tables.summaries,
            "paired_tests": tables.paired_tests,
        }
        out = Path(args.output_dir) / "score.json"
        out.write_text(json.dumps(jsonable(report), indent=2) + "\n", encoding="utf-8")
    except (ValueError, TypeError, KeyError, OSError, ModuleNotFoundError) as exc:
        print(f"ruscuav score: {exc}", file=sys.stderr)
        return 2

    def pct(value):
        return "   -  " if value is None or value != value else f"{100 * value:5.1f}%"

    print(f"{'run':40s} {'episodes':>8s} {'success':>8s} {'MAR':>8s} {'AGS':>8s}")
    for row in tables.summaries:
        seed = "" if row["train_seed"] is None else f"@{row['train_seed']:g}"
        print(f"{(row['label'] + seed)[:40]:40s} {row['n_episodes']:8d} {pct(row['success_rate']):>8s} "
              f"{pct(row['mission_acceptability_rate']):>8s} {pct(row['acceptable_given_success_rate']):>8s}")
    for row in tables.paired_tests:
        print(f"{row['comparison_id']}: {row['n_ab']} vs {row['n_ba']} discordant missions, p = {row['p_raw']:.3g}, "
              f"Holm p = {row['p_holm']:.3g}{' (significant)' if row['significant_holm_005'] else ''}")
    incomplete = [c for c in checks if not c["complete"]]
    if incomplete:
        print(f"note: {len(incomplete)} runs do not cover all missions of {protocol.name}", file=sys.stderr)
    print(f"tables and score.json written to {args.output_dir}")
    return 0


def _parse_center(text: str):
    parts = [p.strip() for p in str(text).split(",")]
    try:
        lat, lon = (float(p) for p in parts)
    except ValueError:
        raise ValueError(f"--center {text!r}: expected LAT,LON in degrees, e.g. 48.2152,16.3975") from None
    return lat, lon


def _cmd_build_aoi(args: argparse.Namespace) -> int:
    import os
    from pathlib import Path

    from .validate import validate_world_npz

    aoi = None
    try:
        from .gis import PAPER_AOIS, Aoi, build_world

        preset = dict(PAPER_AOIS[args.preset]) if args.preset else {}
        if args.center is not None:
            lat, lon = _parse_center(args.center)
        elif preset:
            lat, lon = preset["center_lat"], preset["center_lon"]
        else:
            raise ValueError("give --center LAT,LON or --preset")
        aoi = Aoi(lat, lon, args.size if args.size is not None else preset.get("size_m", 1000.0),
                  args.resolution if args.resolution is not None else preset.get("resolution_m", 1.0))
        if args.bbox_only:
            from .gis import EXTRACT_MARGIN_M, aoi_geometry, extract_bbox

            geo = aoi_geometry(aoi)
            west, south, east, north = extract_bbox(aoi)
            print(json.dumps({"center": [aoi.center_lat, aoi.center_lon], "size_m": aoi.size_m,
                              "resolution_m": aoi.resolution_m, **geo.to_dict(),
                              "extract_margin_m": EXTRACT_MARGIN_M, "extract_bbox": {
                                  "north": north, "south": south, "east": east, "west": west},
                              "osmium_extract": f"osmium extract -s smart -b {west!r},{south!r},{east!r},{north!r} "
                                                "REGION.osm.pbf -o area.osm"}, indent=2))
            return 0
        if args.output is None:
            raise ValueError("give --output, the world file to write")
        moved = preset and any(v is not None for v in (args.center, args.size, args.resolution))
        if moved and not args.aoi_id:
            raise ValueError("--center, --size and --resolution change the area of --preset: give --aoi-id, the "
                             "world's identifier (the preset's is the paper's area)")
        aoi_id = args.aoi_id or preset.get("aoi_id")
        if not aoi_id:
            raise ValueError("give --aoi-id, the world's identifier")
        if (args.osm is None) == (not args.download):
            raise ValueError("give the OpenStreetMap data: --osm FILE (OSM XML) or --download")
        if not args.download and (args.osm_date or args.cache):
            raise ValueError("--osm-date and --cache apply to --download")
        if args.dem is None and args.terrain_datum is not None:
            raise ValueError("--terrain-datum applies to --dem")
        if args.protocol is not None and args.episodes < 1:
            raise ValueError(f"--episodes must be at least 1, got {args.episodes}")
        if args.protocol is not None and args.seed0 < 0:
            raise ValueError(f"--seed0 must not be negative, got {args.seed0}")
        if args.protocol is not None:  # the "env" extra, before the world is built
            import importlib.util

            for module in ("gymnasium", "scipy"):
                if importlib.util.find_spec(module) is None:
                    raise ModuleNotFoundError(f"No module named {module!r}")
        out = Path(args.output)
        if not out.name.endswith(".npz"):  # as numpy.savez names the file
            out = out.with_name(out.name + ".npz")
        built = build_world(aoi, aoi_id, osm_file=args.osm, download=args.download, date=args.osm_date,
                            cache_folder=args.cache, dem=args.dem, dsm=args.dsm,
                            terrain_datum=args.terrain_datum or "relative", sidewalks=args.sidewalks)
        out = Path(built.world.save_npz(out))
        report = validate_world_npz(out)
        meta = built.meta
        result = {
            "aoi_id": built.world.aoi_id, "file": str(out), "layers_sha256": built.world.layers_sha256(),
            "utm_crs": built.geometry.utm_crs, "wgs_bbox": built.geometry.wgs_bbox,
            "features_in_area": meta["osm"]["features_in_area"], "terrain": meta["terrain"]["source"],
            "terrain_offset_m": meta["terrain"]["offset_m"],
            "sidewalks": meta["sidewalks"], "coverage": {k: round(v, 4) for k, v in meta["coverage"].items()},
            "valid": report.ok, "errors": report.errors, "warnings": built.warnings + report.warnings,
        }
        if args.protocol is not None and not report.ok:
            result["protocol"] = None
            print("ruscuav build-aoi: no protocol recorded, the world is invalid", file=sys.stderr)
        elif args.protocol is not None:
            from .protocols import make_protocol

            protocol_path = Path(args.protocol)
            try:
                world_file = Path(os.path.relpath(out.resolve(), protocol_path.resolve().parent)).as_posix()
            except ValueError:  # another drive (Windows)
                world_file = out.resolve().as_posix()
            protocol = make_protocol(built.world, args.protocol_name or built.world.aoi_id,
                                     n_episodes=args.episodes, seed0=args.seed0, world_file=world_file)
            protocol.save(protocol_path)
            result["protocol"] = {"name": protocol.name, "file": str(protocol_path), "missions": protocol.n_episodes,
                                  "digest": protocol.digest()}
    except ModuleNotFoundError as exc:
        print(f"ruscuav build-aoi: {exc}. Building worlds needs the \"gis\" extra (and recording a protocol the "
              "\"env\" extra): pip install \"ruscuav-benchmark[gis,env]\"", file=sys.stderr)
        return 2
    except MemoryError:
        grid = "" if aoi is None else f" for a grid of {aoi.shape[0]} x {aoi.shape[1]} cells"
        print(f"ruscuav build-aoi: not enough memory{grid}", file=sys.stderr)
        return 2
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        print(f"ruscuav build-aoi: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0 if report.ok else 1


def _cmd_protocol_list(args: argparse.Namespace) -> int:
    from .protocols import find_world, list_protocols, load_protocol

    for name in list_protocols():
        p = load_protocol(name)
        world = "world found" if find_world(p) else ("world not found" if p.world_distributed else
                                                     "world not distributed")
        print(f"{name:22s} {p.n_episodes:4d} missions  {p.world_id}  {p.regime}  {p.acceptability} acceptability  "
              f"({world})")
    return 0


def _cmd_protocol_show(args: argparse.Namespace) -> int:
    from .protocols import find_world, load_protocol

    try:
        p = load_protocol(args.protocol)
    except (ValueError, KeyError, OSError) as exc:
        print(f"ruscuav protocol: {exc}", file=sys.stderr)
        return 2
    data = p.to_dict()
    missions = data.pop("missions")
    data["n_missions"] = len(missions)
    data["seeds"] = [missions[0]["seed"], missions[-1]["seed"]]
    data["digest"] = p.digest()
    found = find_world(p)
    data["world"]["found"] = None if found is None else str(found)
    if args.missions:
        data["missions"] = missions
    print(json.dumps(data, indent=2))
    return 0


def _cmd_protocol_create(args: argparse.Namespace) -> int:
    import os
    from pathlib import Path

    from .protocols import make_protocol

    try:
        # the world file is recorded relative to the protocol file's folder, where find_world looks first
        try:
            world_file = Path(os.path.relpath(Path(args.world).resolve(), Path(args.output).resolve().parent)).as_posix()
        except ValueError:  # another drive (Windows)
            world_file = Path(args.world).resolve().as_posix()
        p = make_protocol(args.world, args.name, n_episodes=args.episodes, seed0=args.seed0, regime=args.regime,
                          dynamic=args.dynamic, risk_profile=args.profile, description=args.description,
                          world_file=world_file)
        path = p.save(args.output)
    except (ValueError, KeyError, OSError, RuntimeError, ModuleNotFoundError) as exc:
        print(f"ruscuav protocol create: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"protocol": p.name, "file": str(path), "world_id": p.world_id, "world_file": p.world_file,
                      "world_layers_sha256": p.world_layers_sha256, "missions": p.n_episodes,
                      "seeds": [p.seeds[0], p.seeds[-1]], "regime": p.regime, "acceptability": p.acceptability,
                      "digest": p.digest()}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    from .planners import PLANNER_PRESETS

    parser = argparse.ArgumentParser(prog="ruscuav", description="RUSCUAV benchmark tools.")
    parser.add_argument("--version", action="version", version=f"ruscuav {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate", help="validate world NPZ files (exit code 1 if any is invalid)")
    p.add_argument("paths", nargs="+", help="world NPZ files")
    p.add_argument("--json", action="store_true", help="print machine-readable reports")
    p.set_defaults(func=_cmd_validate)

    p = sub.add_parser(
        "evaluate", help="fly a planner of the paper or a policy on a world, or score flown trajectories (needs the "
                         "'env' extra)",
        description="Fly a planner of the paper, or a PPO policy, on a world, one episode per mission, and write "
                    "the per-episode CSV and the summary. Missions come from --protocol, from --scenarios, or are "
                    "sampled by the environment from seeds --seed0, --seed0+1, ... Policies need the 'rl' extra. "
                    "With --trajectories, positions flown by any other method are scored on a protocol's missions "
                    "(docs/TRAJECTORIES.md).",
    )
    p.add_argument("--world", default=None,
                   help="world NPZ file (with --protocol: needed when the protocol's world is not distributed; for "
                        "--evaluation: only for the Vienna evaluations, whose world is not in the archive)")
    p.add_argument("--protocol", default=None,
                   help="evaluation protocol, a name (canonical-primary, canonical-secondary, canonical-dynamic, "
                        "vienna-primary; see `ruscuav protocol list`) or a protocol file: its world, missions, "
                        "regime, risk profile and dynamic risk (docs/PROTOCOLS.md)")
    p.add_argument("--trajectories", default=None,
                   help="score flown trajectories instead of flying a method: a folder of .npy files, an .npz or a "
                        "CSV (docs/TRAJECTORIES.md); needs --protocol")
    p.add_argument("--planner", default=None, choices=list(PLANNER_PRESETS),
                   help="planner (default FULL_RH_Astar unless a policy is given)")
    p.add_argument("--policy", default=None, help="Stable-Baselines3 PPO checkpoint (.zip) to fly; needs --method")
    p.add_argument("--checkpoint", default=None,
                   help="archived checkpoint to fly, e.g. RISK+GREEN-RL/4001 (see docs/POLICIES.md); needs --archive")
    p.add_argument("--evaluation", default=None,
                   help="re-run an archived policy evaluation, e.g. static/RISK+GREEN-RL/4001 or "
                        "dynamic/DYN-OBS-RL/1001 (see docs/POLICIES.md); needs --archive")
    p.add_argument("--archive", default=None, help="extracted RUSCUAV v2.0.0 archive, for --checkpoint and --evaluation")
    p.add_argument("--method", default=None,
                   help="method preset of the policy's environment, e.g. RISK+GREEN-RL (default for --checkpoint: "
                        "the preset it was trained in)")
    p.add_argument("--format", default=None, choices=["posthoc", "dynamic", "transfer"],
                   help="row format of a policy evaluation (default posthoc; dynamic for DYN-* presets)")
    p.add_argument("--label", default=None,
                   help="method label written in the rows of a policy evaluation or of scored trajectories")
    p.add_argument("--stochastic", action="store_true", help="sample the policy's actions (default deterministic)")
    p.add_argument("--scenarios", default=None,
                   help="scenario CSV with start/target columns and, optionally, seed and world_id (rows of "
                        "this world are flown; rows sharing a seed share the dynamic risk field)")
    p.add_argument("--episodes", type=int, default=None,
                   help="number of episodes (default: every mission of the protocol or scenario file, or 200 seeds)")
    p.add_argument("--seed0", type=int, default=None,
                   help="first seed when no scenario file is given, or for rows without a seed (default 5000; for "
                        "--evaluation that of the archived run)")
    p.add_argument("--regime", default=None,
                   choices=["primary_no_terminal", "secondary_operational_stopping"],
                   help="evaluation regime (default primary_no_terminal)")
    p.add_argument("--profile", default=None, help="risk profile (default balanced)")
    p.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE",
                   help="change a planner setting, e.g. --set planner_agl_m=30 (repeatable; see docs/PLANNERS.md)")
    p.add_argument("--output-dir", required=True, help="directory for <prefix>_episodes.csv, _summary.json, ...")
    p.add_argument("--prefix", default=None,
                   help="file name prefix (default <planner or method>_<world>_<regime>; for planners plus a "
                        "configuration checksum when settings differ from the preset)")
    p.add_argument("--save-paths", action="store_true", help="also save the flown (and planned) paths")
    p.add_argument("--quiet", action="store_true", help="no progress messages")
    p.set_defaults(func=_cmd_evaluate)

    p = sub.add_parser(
        "score", help="the paper's tables and paired tests for per-episode results",
        description="Summaries (success, post-hoc mission acceptability, conditional and normalized risk), "
                    "training-seed intervals, causes of non-acceptability, the dynamic-budget sweep and, with "
                    "--compare, exact paired tests with Holm's correction, for per-episode CSVs (docs/SCORING.md).",
    )
    p.add_argument("runs", nargs="*", metavar="[LABEL[@SEED]=]CSV",
                   help="per-episode CSV of one run; LABEL defaults to the rows' method, SEED is the training seed "
                        "of a learned method")
    p.add_argument("--registry", default=None,
                   help="CSV listing runs (columns label, csv_path and optionally dynamic, train_seed, scope, regime)")
    p.add_argument("--protocol", default=None,
                   help="check that every run was evaluated under the protocol (its missions, regime and dynamic "
                        "risk, and the protocol, world or regime recorded next to the CSV) and score with its "
                        "budgets and acceptability")
    p.add_argument("--compare", action="append", default=[], metavar="CSV|A:B:METRIC",
                   help="paired comparisons: a CSV with family, comparison_id, method_a, method_b, metric "
                        "(success, MAR_static, DynOK, MAR_dynamic), or A:B:METRIC (repeatable; one family)")
    p.add_argument("--scope", default=None, help="scope of the runs given on the command line (default: the protocol)")
    p.add_argument("--exact-floats", action="store_true",
                   help="read numbers correctly rounded (default: as pandas.read_csv reads them, as the archive did)")
    p.add_argument("--output-dir", required=True, help="directory for the tables and score.json")
    p.set_defaults(func=_cmd_score)

    p = sub.add_parser(
        "build-aoi", help="build a world NPZ from OpenStreetMap data (needs the 'gis' extra)",
        description="Build a world from the OpenStreetMap features of a square area with the pipeline of the paper's "
                    "real-GIS worlds (buildings with heights, roads, sidewalks, green areas, canopy, sensitive zones, "
                    "terrain), validate it and optionally record a protocol for it (docs/BUILDING_WORLDS.md). "
                    "Worlds built from OpenStreetMap carry its licence (ODbL); see docs/OSM_ATTRIBUTION.md.",
    )
    p.add_argument("--preset", default=None, choices=["vienna", "warsaw"],
                   help="the area of one of the paper's worlds (centre, size, resolution and identifier)")
    p.add_argument("--center", default=None, metavar="LAT,LON",
                   help="centre of the area in WGS 84 degrees, e.g. 48.2152,16.3975 or -33.9249,18.4241")
    p.add_argument("--size", type=float, default=None, help="side of the square area in metres (default 1000)")
    p.add_argument("--resolution", type=float, default=None, help="cell size in metres (default 1)")
    p.add_argument("--aoi-id", default=None, help="the world's identifier (footway-only worlds get '_footway_only')")
    p.add_argument("--osm", default=None, help="OSM XML file of the area (.osm or .osm.bz2)")
    p.add_argument("--download", action="store_true", help="download the features from the Overpass API")
    p.add_argument("--osm-date", default=None, metavar="ISO-TIME",
                   help="with --download: the OpenStreetMap data as of this time, e.g. 2026-07-21T00:00:00Z")
    p.add_argument("--cache", default=None, help="with --download: folder for the Overpass responses (read again "
                                                   "offline by the same command)")
    p.add_argument("--dem", default=None, help="GeoTIFF of the terrain (default: the smooth fallback surface)")
    p.add_argument("--dsm", default=None, help="GeoTIFF of the surface (terrain, buildings and canopy), with --dem")
    p.add_argument("--terrain-datum", default=None, choices=["relative", "absolute"],
                   help="heights of --dem and --dsm: relative to the area's lowest whole metre (default; the "
                        "environment flies between 0 and 150 m) or as the files have them")
    p.add_argument("--sidewalks", default="footway-only", choices=["footway-only", "proxy"],
                   help="sidewalk layer: the footways (default, as the paper's evaluations) or the footways and a "
                        "4 m proxy along roads")
    p.add_argument("--output", default=None, help="world NPZ file to write (.npz is added if missing)")
    p.add_argument("--bbox-only", action="store_true",
                   help="only print the area's UTM bounds and WGS 84 bounding box, and an osmium command that cuts "
                        "an OSM extract for it")
    p.add_argument("--protocol", default=None, help="also record a protocol for the world in this file (needs the "
                                                    "'env' extra)")
    p.add_argument("--protocol-name", default=None, help="the protocol's name (default: the world's identifier)")
    p.add_argument("--episodes", type=int, default=200, help="missions of the protocol (default 200)")
    p.add_argument("--seed0", type=int, default=5000, help="seed of the protocol's first mission (default 5000)")
    p.set_defaults(func=_cmd_build_aoi)

    p = sub.add_parser("protocol", help="list, show or create evaluation protocols",
                       description="Evaluation protocols fix the world, the missions, the regime, the dynamic risk "
                                   "and the budgets that results are compared under (docs/PROTOCOLS.md).")
    actions = p.add_subparsers(dest="action", required=True)
    q = actions.add_parser("list", help="the protocols of the package")
    q.set_defaults(func=_cmd_protocol_list)
    q = actions.add_parser("show", help="a protocol's settings")
    q.add_argument("protocol", help="protocol name or file")
    q.add_argument("--missions", action="store_true", help="also print the missions")
    q.set_defaults(func=_cmd_protocol_show)
    q = actions.add_parser("create", help="record a protocol for a world (needs the 'env' extra)")
    q.add_argument("--world", required=True, help="world NPZ file")
    q.add_argument("--name", required=True, help="protocol name")
    q.add_argument("--output", required=True, help="protocol file to write (.json)")
    q.add_argument("--episodes", type=int, default=200, help="number of missions (default 200)")
    q.add_argument("--seed0", type=int, default=5000, help="seed of the first mission (default 5000)")
    q.add_argument("--regime", default="primary_no_terminal",
                   choices=["primary_no_terminal", "secondary_operational_stopping"])
    q.add_argument("--dynamic", action="store_true",
                   help="with the paper's dynamic risk field and dynamic acceptability")
    q.add_argument("--profile", default="balanced", help="risk profile (default balanced)")
    q.add_argument("--description", default="", help="description written into the protocol")
    q.set_defaults(func=_cmd_protocol_create)
    return parser


#: Options whose value can start with a minus sign (a southern latitude), which
#: argparse would take for an option: main() passes them as ``--center=VALUE``.
_SIGNED_VALUE_OPTIONS = ("--center",)


def _join_signed_values(argv: List[str]) -> List[str]:
    out, i = [], 0
    while i < len(argv):
        if argv[i] in _SIGNED_VALUE_OPTIONS and i + 1 < len(argv) and re.match(r"-\.?\d", argv[i + 1]):
            out.append(f"{argv[i]}={argv[i + 1]}")
            i += 2
        else:
            out.append(argv[i])
            i += 1
    return out


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    args = build_parser().parse_args(_join_signed_values(argv))
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
