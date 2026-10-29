# The examples of the SoftwareX paper

The scripts behind Section 3 of the SoftwareX paper on RUSCUAV (Examples 1 and 2), its Fig. 2 and the data of
its Fig. 3. Run them from the root of a checkout of the repository with the package installed
(`pip install -e ".[env]"`); they write everything, a world built from OpenStreetMap too, to the
git-ignored `runs/softwarex/`.

```bash
bash examples/softwarex/example1.sh                         # Example 1, about 4 minutes on two cores
bash examples/softwarex/example2.sh --center LAT,LON        # Example 2: a world built from OpenStreetMap
bash examples/softwarex/example2.sh --world FILE.npz        #   ... or a world you already have
bash examples/softwarex/figures.sh CITY_WORLD.npz [MISSION [TITLE]]   # Fig. 2 and the data of Fig. 3
python examples/softwarex/listing_sac.py 2000 3 runs/softwarex/sac   # checks that Listing 1 runs ("rl" extra)
```

Building a world needs the `gis` extra and access to the Overpass API.

| File | What it does |
|---|---|
| `geometric_planner.py` | GEO_SHORTEST, a planner from outside the package (NumPy and SciPy only): the shortest path at 30 m above the terrain around what it cannot overfly, flown at 12 m/s; writes the trajectory CSV that `ruscuav evaluate --trajectories` scores |
| `example1.sh` | FULL_RH_Astar, STATIC_Astar and GEO_SHORTEST on `canonical-primary`, their score with the paired tests, and the budget sensitivity |
| `example2.sh` | a protocol recorded for a city world (seeds 9000–9199), the two planners at 30 m, GEO_SHORTEST and the score |
| `budget_sensitivity.py` | acceptability with another pedestrian-way budget |
| `figures.sh`, `fig2_map.py`, `fig3_data.py` | Fig. 2 (one mission per world, the two paths over the semantic layers) and the counts of Fig. 3 |
| `listing_sac.py` | Listing 1, shortened to 2,000 training steps and 3 missions |

## Results

Each run has 200 missions; success and mission acceptability.

| World (protocol) | FULL_RH_Astar | STATIC_Astar | GEO_SHORTEST |
|---|---|---|---|
| AOI_SYNTH_000 (`canonical-primary`) | 98.5% / 97.0% | 92.5% / 91.5% | 100.0% / 69.5% |
| Vienna, Prater–Leopoldstadt (`city-example`, planners at 30 m) | 97.5% / 63.0% | 97.5% / 64.5% | 97.0% / 20.0% |

GEO_SHORTEST against FULL_RH_Astar, exact two-sided McNemar tests with Holm's correction over the two
comparisons of each world (discordant missions, p, Holm-adjusted p):

| World | Success | Acceptability |
|---|---|---|
| AOI_SYNTH_000 | 3 vs 0, 0.25, 0.25 | 3 vs 58, 3.3e-14, 6.6e-14 |
| Vienna | 0 vs 1, 1.0, 1.0 | 2 vs 88, 6.6e-24, 1.3e-23 |

- **Exposure.** All 61 unacceptable missions of GEO_SHORTEST in the synthetic world exceed the 5 s budget over
  pedestrian ways.
- **Budget sensitivity.** With a pedestrian budget of 10 s (the stopping budget of `canonical-secondary`),
  95.5% of its missions would be acceptable, against 98.5% for FULL_RH_Astar.
- **Fig. 2.** On mission 5010 GEO_SHORTEST spends 20.0 s over pedestrian ways and FULL_RH_Astar 3.5 s. On
  mission 9042 in Vienna, 11.0 s against 3.0 s.
- **Vienna failures.** GEO_SHORTEST fails 6 Vienna missions: their targets cannot be reached at its cruise
  altitude, so its trajectory stays at the start.
- **Trajectories.** All its trajectories keep to the vehicle's speed and acceleration limits and its
  workspace, and none enters a hazard: none ends as infeasible.

The rates of the planners in the synthetic world are the archived results of the paper's study.

The Vienna rows come from the study's processed Vienna world (`vienna_prater_leopoldstadt_osm_1km_v1_footway_only`,
layers checksum `fa99c571…`). That world is not in the repository (docs/OSM_ATTRIBUTION.md); it is published
apart, under the ODbL, with the release (docs/RELEASING.md), and `example2.sh --world` with it gives these rows.
`ruscuav build-aoi --preset vienna` builds a world of the same area, which gives them only if it has the same
checksum.
For another city, choose the mission of Fig. 2 (b) and give it to `figures.sh`; a world built with
`example2.sh --center` is `runs/softwarex/city/world.npz`.
