#!/usr/bin/env python3
"""Write the package's protocol files (src/ruscuav_benchmark/protocols/*.json).

The missions are those recorded in scenarios/ (scripts/materialize_legacy_scenarios.py),
with the world checksums of the JSON files next to them; the regimes, the dynamic
risk and the budgets are the paper's. tests/test_scoring.py checks that the files
match the scenario CSVs.

    python scripts/make_protocols.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ruscuav_benchmark.env.dynamic_risk import REPORTED_DYNAMIC_CONFIG  # noqa: E402
from ruscuav_benchmark.posthoc import DEFAULT_BUDGETS  # noqa: E402
from ruscuav_benchmark.protocols import Mission, Protocol  # noqa: E402
from ruscuav_benchmark.scenarios import read_scenarios  # noqa: E402

OUT = REPO / "src" / "ruscuav_benchmark" / "protocols"
CANONICAL = "scenarios/canonical/AOI_SYNTH_000_seed5000_n200.csv"
VIENNA = "scenarios/real_gis/vienna_prater_leopoldstadt_osm_1km_v1_footway_only_seed7000_n200.csv"

PRIMARY = ("the primary no-terminal regime (stage B, strict sensing): hard physical and low-altitude "
           "sensitive-zone constraints are active, road, pedestrian, building and dynamic budget exceedances are "
           "logged without ending the episode, and mission acceptability is computed post hoc with the paper's "
           "budgets")


def missions(csv_name: str):
    rows = read_scenarios(REPO / csv_name)
    meta = json.loads((REPO / csv_name).with_suffix(".json").read_text(encoding="utf-8"))
    out = tuple(Mission(int(r["seed"]), tuple(float(r[f"start_{a}"]) for a in "xyz"),
                        tuple(float(r[f"target_{a}"]) for a in "xyz")) for r in rows)
    source = {"file": csv_name, "sha256": hashlib.sha256((REPO / csv_name).read_bytes()).hexdigest(),
              "recorded_from": meta.get("environment")}
    return out, source, meta


def main() -> int:
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()  # no options; --help
    canonical, canonical_source, canonical_meta = missions(CANONICAL)
    vienna, vienna_source, vienna_meta = missions(VIENNA)
    world = {"world_id": "AOI_SYNTH_000", "world_layers_sha256": canonical_meta["world_layers_sha256"],
             "world_file": "worlds/synthetic/AOI_SYNTH_000/urban_world_layers.npz", "world_distributed": True}
    protocols = [
        Protocol(name="canonical-primary",
                 description="The paper's primary evaluation: 200 recorded missions (seeds 5000-5199) in the "
                             f"canonical synthetic world AOI_SYNTH_000, {PRIMARY}.",
                 missions=canonical, missions_source=canonical_source, **world),
        Protocol(name="canonical-secondary",
                 description="The paper's secondary evaluation: the canonical-primary missions in the operational-"
                             "stopping regime (stage C, strict sensing), in which collisions, low-altitude "
                             "sensitive-zone violations and exposure above the stopping budgets (30 s over roads, "
                             "10 s over pedestrian ways, 20 s over buildings, any time over sensitive zones) end the "
                             "episode. Its success rate is strict operational completion, not to be pooled with "
                             "primary success rates.",
                 missions=canonical, missions_source=canonical_source, regime="secondary_operational_stopping",
                 **world),
        Protocol(name="canonical-dynamic",
                 description="The paper's dynamic evaluation: the canonical-primary missions with the reported "
                             "dynamic risk field (16 time-varying hotspots over roads and sidewalks, drawn from each "
                             f"mission seed), {PRIMARY}; mission acceptability also requires a cumulative dynamic "
                             "risk of at most 10.",
                 missions=canonical, missions_source=canonical_source, dynamic_risk=asdict(REPORTED_DYNAMIC_CONFIG),
                 acceptability="dynamic", **world),
        Protocol(name="vienna-primary",
                 description="The paper's real-GIS transfer evaluation: 200 recorded missions (seeds 7000-7199) in "
                             "the processed Vienna Prater/Leopoldstadt footway-only world (OpenStreetMap data, not "
                             f"distributed: docs/OSM_ATTRIBUTION.md), {PRIMARY}. The missions were sampled in that "
                             "world and are derived from OpenStreetMap data: © OpenStreetMap contributors, available "
                             "under the Open Database License (ODbL 1.0).",
                 missions=vienna, missions_source=vienna_source, world_id=vienna_meta["world_id"],
                 world_layers_sha256=vienna_meta["world_layers_sha256"],
                 world_file=vienna_meta["world_file"]["name"], world_distributed=False),
    ]
    assert canonical[0].seed == 5000 and len(canonical) == 200 and vienna[0].seed == 7000 and len(vienna) == 200
    assert all(p.budgets == DEFAULT_BUDGETS for p in protocols)
    for protocol in protocols:
        path = protocol.save(OUT / f"{protocol.name}.json")
        print(f"{path.relative_to(REPO)}: {protocol.n_episodes} missions, digest {protocol.digest()[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
