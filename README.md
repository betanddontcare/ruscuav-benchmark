# RUSCUAV — semantic mission-acceptability benchmark for urban cargo UAVs

**RUSCUAV** (Reproducible Urban Semantic Cargo-UAV) is a benchmark for low-altitude
urban cargo-UAV navigation that scores missions by **benchmark-defined semantic
mission acceptability**, not by target-reaching success alone. This repository holds
`ruscuav-benchmark` 2.1, the Python package of the benchmark. It accompanies the
manuscript

> **RUSCUAV: A Semantic Mission-Acceptability Benchmark for Low-Altitude Urban Cargo UAV Navigation**
>
> Igor Betkier and Mateusz Oszczypała

whose evaluations, trained weights and paper-ready tables are archived on Zenodo
(RUSCUAV v2.0.0, see "The Zenodo archive" below). The package reproduces them and
lets any method be evaluated the same way.

## What the package does

- **Worlds.** Raster worlds of an urban area (terrain, surface, buildings, roads,
  pedestrian ways, green areas, canopy, sensitive zones) in a documented NPZ schema;
  `ruscuav validate` checks any world file. Four synthetic worlds come with the
  repository, and `ruscuav build-aoi` builds the world of any city from OpenStreetMap
  with the pipeline of the paper's real-GIS worlds.
- **Environment.** `ruscuav_benchmark.env.RuscuavEnv`, a Gymnasium environment with the
  paper's method presets, curriculum stages and evaluation regimes; it reproduces the
  research notebook's environments step for step.
- **Methods.** The paper's risk-cost A* planners (static, receding-horizon, dynamic)
  and PPO policies (the archived checkpoints, and training with the paper's
  curriculum).
- **Protocols.** Fixed missions, regime, dynamic risk and budgets under which results
  are compared; the paper's four protocols come with the package, and
  `ruscuav protocol create` records one for any world.
- **Scoring.** Trajectories of any method (another simulator, a learned policy,
  recorded flights) scored with the environment's accounting, and `ruscuav score`:
  success, mission acceptability, training-seed confidence intervals and exact paired
  tests with Holm's correction, computed as in the paper.

## Installation

Python 3.10 to 3.13. From a checkout of the repository:

```bash
pip install -e ".[env]"          # environment, planners, protocols, trajectory scoring
pip install -e ".[env,gis]"      # and worlds from OpenStreetMap
pip install -e ".[all,test]"     # and the PPO policies (Stable-Baselines3, PyTorch), and the tests
```

| Extra | Adds | For |
|---|---|---|
| (none) | NumPy, SciPy, Matplotlib | `ruscuav validate`, `ruscuav score`, `ruscuav protocol list` and `show` |
| `env` | Gymnasium, NumPy 2 | the environment, the planners, `ruscuav evaluate`, `ruscuav protocol create` |
| `rl` | Stable-Baselines3 | the PPO policies |
| `gis` | OSMnx, GeoPandas, Shapely, Rasterio, PyProj | `ruscuav build-aoi` |
| `all` | `env`, `rl` and `gis` | everything |
| `test` | pytest, pandas | the test suite |

`requirements-reproducibility.txt` pins the versions of the archived environment.

## Quick start

From the repository root (the canonical world of the protocols is in
`worlds/synthetic/`):

```bash
ruscuav protocol list

# a planner of the paper on the 200 missions of the primary protocol
ruscuav evaluate --protocol canonical-primary --planner FULL_RH_Astar --output-dir runs/full_rh

# your method, from the positions it flew on each mission (docs/TRAJECTORIES.md)
ruscuav evaluate --protocol canonical-primary --trajectories my_flights.csv --label MINE --output-dir runs/mine

# both, compared mission by mission
ruscuav score --protocol canonical-primary MINE=runs/mine/MINE_canonical-primary_episodes.csv \
    FULL_RH=runs/full_rh/FULL_RH_Astar_canonical-primary_episodes.csv \
    --compare MINE:FULL_RH:success --compare MINE:FULL_RH:MAR_static --output-dir runs/score
```

A trajectory is scored as far as the benchmark's vehicle could have flown it: a step
faster than its speed limits, outside its workspace or through a hazard fails the
mission (docs/TRAJECTORIES.md). `--episodes 20` flies the first 20 missions only
(the files then end in `_n20_episodes.csv`). A city of your own:

```bash
ruscuav build-aoi --center 50.0614,19.9366 --aoi-id krakow --download \
    --output worlds/krakow.npz --protocol protocols/krakow.json
ruscuav evaluate --protocol protocols/krakow.json --planner FULL_RH_Astar --output-dir runs/krakow
```

In Python, the environment:

```python
from ruscuav_benchmark import UrbanWorldLayers
from ruscuav_benchmark.env import make_env

world = UrbanWorldLayers.load_npz("worlds/synthetic/AOI_SYNTH_000/urban_world_layers.npz")
env = make_env(world, "RISK+GREEN-RL", regime="primary_no_terminal")
obs, info = env.reset(seed=5000)          # the canonical mission of seed 5000
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

## The benchmark

```text
Success = target-region completion
Mission acceptability = Success
                        AND cargo health >= 0.80
                        AND road time <= 20 s
                        AND pedestrian time <= 5 s
                        AND building time <= 15 s
                        AND sensitive-zone time <= 0 s
                        AND hard-violation steps = 0
```

In the primary no-terminal regime, budget exceedances are recorded but do not end
the mission, and acceptability is computed after it; the secondary
operational-stopping regime ends missions at operational budgets. The dynamic regime
adds ground-risk hotspots that rise and fade on roads and pedestrian ways during the
mission, and requires the cumulative dynamic risk to stay within the budget of 10. These thresholds are benchmark parameters, not regulatory
approval criteria or validated societal-preference thresholds.
`BENCHMARK_SPEC.md` gives the specification.

## The paper's results

The primary protocol (`canonical-primary`: the synthetic world AOI_SYNTH_000,
missions of seeds 5000–5199, primary no-terminal regime):

| Method | Success | Mission acceptability |
|---|---|---|
| `STATIC_Astar` | 92.5 % | 91.5 % |
| `LOCAL_RH_Astar` | 98.5 % | 95.0 % |
| `FULL_RH_Astar` | 98.5 % | 97.0 % |
| `GEO-RL` (PPO, five training seeds) | 79.3 % [76.4, 82.2] | 71.0 % [65.9, 76.1] |
| `OBS-RL` (five training seeds) | 82.5 % [79.0, 86.0] | 72.1 % [67.9, 76.3] |
| `RISK+GREEN-RL` (five training seeds) | 84.4 % [80.2, 88.6] | 75.8 % [70.5, 81.1] |
| `RISK+OBS-RL` (one checkpoint) | 88.0 % | 78.5 % |

PPO rows give the mean over training seeds with 95 % t-intervals. The dynamic,
Vienna and Warsaw evaluations and the statistical families are in the archive's
tables (docs/TABLE_FIGURE_ARTIFACT_MAP.md); docs/REPRODUCIBILITY.md shows how the
package reproduces them.

## Documentation

docs/README.md lists the guides: the world schema and validation, building worlds
from OpenStreetMap, the environment, planners, policies, protocols, trajectory
scoring, the tables and statistics, and the reproduction of the archive.

## Repository and archive

```text
src/ruscuav_benchmark/   the package (legacy/: the research notebook's code, kept verbatim)
tests/                   the test suite
docs/                    guides
scripts/                 tools and checks against the archive (scripts/README.md)
worlds/synthetic/        the four synthetic worlds (AOI_SYNTH_000, AOI_SYNTH_001-003)
worlds/real_gis/         metadata of the Vienna and Warsaw worlds (the worlds are not in the repository)
scenarios/               the recorded missions of the canonical and Vienna evaluations
configs/                 the settings of the archived experiments (configs/README.md)
code/submission_repair/  the code of the archived one-click repair runs
examples/                a quickstart script; softwarex/: the examples of the SoftwareX paper
                         (examples/softwarex/README.md)
```

### The Zenodo archive

The RUSCUAV v2.0.0 archive on Zenodo (https://doi.org/10.5281/zenodo.21331402, the DOI
of all versions of the record; v1.0.0 is 10.5281/zenodo.21331403) holds what the
repository does not: the per-episode CSVs and summaries of the
evaluations and the trained weights (`experiments/`; `models/` lists the weights and
holds the Vienna fine-tuning checkpoints), the paper-ready tables, JSON and figures
(`paper_ready_v2/`),
the manuscript and supplement (`paper/`), and provenance, environment snapshots and
manifests. The package's reproduction checks take the extracted archive
(`--archive`), and from its root:

```bash
python scripts/verify_v2_release.py --root .
sha256sum -c SHA256SUMS
```

## Real-GIS worlds and OpenStreetMap

The processed real-GIS worlds (Vienna, Warsaw, Berlin) are not in this repository. The
Vienna world of the paper's evaluations and the Berlin world of the SoftwareX article's
Example 2 (examples/softwarex/README.md) are published apart, as one dataset under the
ODbL, with the release (docs/RELEASING.md); the Warsaw footway-only world of the
archived runs was never saved. `ruscuav build-aoi --preset vienna` (or `warsaw`) builds a world of the
same area from OpenStreetMap data, and `scenarios/real_gis/` records the missions and
the layers checksum of the paper's Vienna world (docs/BUILDING_WORLDS.md,
worlds/real_gis/README.md). OpenStreetMap data are © OpenStreetMap contributors under
the ODbL; worlds built from them are derived databases (docs/OSM_ATTRIBUTION.md).

## Tests

```bash
pip install -e ".[env,gis,test]"
pytest
```

Tests that compare the package with the archive run when `RUSCUAV_ARCHIVE` points
to the extracted archive (and `RUSCUAV_VIENNA_WORLD` to the Vienna world); they are
skipped otherwise.

## Licences

- **Source code and executable scripts:** MIT License (`LICENSE_CODE_MIT`).
- **Synthetic worlds, trained weights, results, tables, figures, documentation and
  paper files:** CC BY 4.0 (`LICENSE_DATA_CC_BY_4_0`); `LICENSE_DATA_AND_ASSETS.md`
  allocates the files.
- **OpenStreetMap-derived metadata:** released with attribution and subject to the
  upstream OpenStreetMap/ODbL terms described in `docs/OSM_ATTRIBUTION.md`.

## Authors

- **Igor Betkier** — ORCID: 0000-0002-5437-7669; corresponding author; igor.betkier@wat.edu.pl
- **Mateusz Oszczypała** — ORCID: 0000-0002-1194-6913; mateusz.oszczypala@wat.edu.pl

Faculty of Civil Engineering and Geodesy, Military University of Technology, gen. Sylwestra Kaliskiego Street 2, 00-908 Warsaw, Poland.

## Citation

Cite the associated manuscript and the version of the software you used;
`CITATION.cff` and `codemeta.json` hold the machine-readable metadata. Version 2.1.0 is
archived on Zenodo, https://doi.org/10.5281/zenodo.23825176. The changes of each
version are in `CHANGELOG.md`.
