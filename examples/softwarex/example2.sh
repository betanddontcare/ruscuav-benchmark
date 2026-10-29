#!/usr/bin/env bash
# Example 2 of the SoftwareX paper: a world of another city, a protocol recorded for it, the reference planners
# and GEO_SHORTEST under that protocol.
#
# Run from the root of a checkout of the repository with the package installed (pip install -e ".[env]";
# building a world also needs ".[gis]" and access to the Overpass API):
#
#     bash examples/softwarex/example2.sh --center LAT,LON   build the world from OpenStreetMap
#     bash examples/softwarex/example2.sh --world FILE.npz    use a world you already have
#
# Everything is written to runs/softwarex/city/ (git-ignored), the world built with --center too: worlds built
# from OpenStreetMap are databases derived from it, under the ODbL 1.0 (docs/OSM_ATTRIBUTION.md).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
usage() {
  echo "usage: $0 --center LAT,LON | --world FILE.npz" >&2
  exit 2
}
[ $# -eq 2 ] || usage
OUT=runs/softwarex/city
P=$OUT/city-example.json
mkdir -p $OUT
case "$1" in
  --center)
    W=$OUT/world.npz
    ruscuav build-aoi --center "$2" --aoi-id city --download --output $W \
        --protocol $P --protocol-name city-example --seed0 9000 > $OUT/build-aoi.json ;;
  --world)
    W=$2
    ruscuav validate "$W" > $OUT/validate.txt || {
      cat $OUT/validate.txt >&2
      exit 1
    }
    ruscuav protocol create --world "$W" --name city-example --output $P --episodes 200 --seed0 9000 \
        --description "Example protocol: 200 missions (seeds 9000-9199) in the world $(basename "$W")" \
        > $OUT/protocol.txt ;;
  *) usage ;;
esac

# The planners plan 30 m above the ground instead of 40 m, as FULL_RH_Astar did in the paper's Vienna evaluation
pids=()
for planner in STATIC_Astar FULL_RH_Astar; do
  ruscuav evaluate --protocol $P --world "$W" --planner $planner --set planner_agl_m=30 \
      --prefix ${planner}_30m --output-dir $OUT --save-paths --quiet > $OUT/$planner.json &
  pids+=($!)
done
for pid in "${pids[@]}"; do
  wait "$pid"
done
python "$HERE/geometric_planner.py" --protocol $P --world "$W" --output $OUT/geo_paths.csv
ruscuav evaluate --protocol $P --world "$W" --trajectories $OUT/geo_paths.csv --label GEO_SHORTEST \
    --output-dir $OUT > $OUT/GEO_SHORTEST.json
ruscuav score --protocol $P \
    GEO_SHORTEST=$OUT/GEO_SHORTEST_city-example_episodes.csv \
    STATIC_Astar=$OUT/STATIC_Astar_30m_episodes.csv \
    FULL_RH_Astar=$OUT/FULL_RH_Astar_30m_episodes.csv \
    --compare GEO_SHORTEST:FULL_RH_Astar:success --compare GEO_SHORTEST:FULL_RH_Astar:MAR_static \
    --output-dir $OUT/scores > $OUT/scores.txt
cat $OUT/scores.txt
