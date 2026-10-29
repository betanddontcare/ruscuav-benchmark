#!/usr/bin/env bash
# Example 1 of the SoftwareX paper: the reference planners and a planner from outside the package on the
# packaged protocol canonical-primary (200 missions in the synthetic world AOI_SYNTH_000).
#
# Run from the root of a checkout of the repository with the package installed (pip install -e ".[env]"):
#
#     bash examples/softwarex/example1.sh
#
# Everything is written to runs/softwarex/ (git-ignored). About 4 minutes on two cores.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
W0=worlds/synthetic/AOI_SYNTH_000/urban_world_layers.npz
[ -f "$W0" ] || {
  echo "run this from the root of a checkout" >&2
  exit 2
}
OUT=runs/softwarex
mkdir -p $OUT

# The listing of Example 1, with STATIC_Astar beside FULL_RH_Astar (quoted in the text and in Fig. 3)
ruscuav evaluate --protocol canonical-primary --world $W0 --planner FULL_RH_Astar --output-dir $OUT/full \
    --save-paths --quiet > $OUT/full.json &
full=$!
ruscuav evaluate --protocol canonical-primary --world $W0 --planner STATIC_Astar --output-dir $OUT/static \
    --quiet > $OUT/static.json &
static=$!
wait $full
wait $static
python "$HERE/geometric_planner.py" --protocol canonical-primary --world $W0 --output $OUT/geo_paths.csv
ruscuav evaluate --protocol canonical-primary --world $W0 --trajectories $OUT/geo_paths.csv --label GEO_SHORTEST \
    --output-dir $OUT/geo > $OUT/geo.json
ruscuav score --protocol canonical-primary \
    GEO_SHORTEST=$OUT/geo/GEO_SHORTEST_canonical-primary_episodes.csv \
    FULL_RH_Astar=$OUT/full/FULL_RH_Astar_canonical-primary_episodes.csv \
    --compare GEO_SHORTEST:FULL_RH_Astar:success --compare GEO_SHORTEST:FULL_RH_Astar:MAR_static \
    --output-dir $OUT/scores > $OUT/scores.txt

# The three runs together, for Fig. 3
ruscuav score --protocol canonical-primary \
    GEO_SHORTEST=$OUT/geo/GEO_SHORTEST_canonical-primary_episodes.csv \
    STATIC_Astar=$OUT/static/STATIC_Astar_canonical-primary_episodes.csv \
    FULL_RH_Astar=$OUT/full/FULL_RH_Astar_canonical-primary_episodes.csv \
    --output-dir $OUT/scores_all > $OUT/scores_all.txt

# The last sentence of Example 1: the verdicts with a pedestrian budget of 10 s
python "$HERE/budget_sensitivity.py" --ped-s 10 \
    GEO_SHORTEST=$OUT/geo/GEO_SHORTEST_canonical-primary_episodes.csv \
    FULL_RH_Astar=$OUT/full/FULL_RH_Astar_canonical-primary_episodes.csv > $OUT/budget_sensitivity.txt
cat $OUT/scores.txt $OUT/budget_sensitivity.txt
