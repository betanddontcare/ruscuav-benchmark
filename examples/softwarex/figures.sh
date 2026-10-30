#!/usr/bin/env bash
# Fig. 2 and the data of Fig. 3 of the SoftwareX paper, from the outputs of example1.sh and example2.sh. Run them
# first, from the root of the same checkout, which is where this script runs too:
#
#     bash examples/softwarex/figures.sh CITY_WORLD.npz [MISSION [TITLE]]
#
# CITY_WORLD is the world of Example 2 (runs/softwarex/city/world.npz when example2.sh built it); MISSION is the
# seed of the mission of panel (b), 9035 in the paper's Berlin world (choose another for another city). Writes
# runs/softwarex/fig2_paths.png, .pdf and runs/softwarex/fig3_data.csv.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
CITY=${1:?give the world of Example 2}
MISSION=${2:-9035}
TITLE=${3:-"(b) Berlin, Alt-Treptow–Treptower Park, mission $MISSION"}
OUT=runs/softwarex
target() {  # the target of a mission of a protocol, as X,Y
  python -c "import sys
from ruscuav_benchmark.protocols import load_protocol
m = next(m for m in load_protocol(sys.argv[1]).missions if m.seed == int(sys.argv[2]))
print(f'{m.target[0]},{m.target[1]}')" "$1" "$2"
}
TARGET_A=$(target canonical-primary 5010)
TARGET_B=$(target $OUT/city/city-example.json "$MISSION")
python "$HERE/fig2_map.py" \
    --panel worlds/synthetic/AOI_SYNTH_000/urban_world_layers.npz $OUT/full/FULL_RH_Astar_canonical-primary_paths \
            $OUT/geo_paths.csv 5010 "$TARGET_A" "(a) Synthetic world AOI_SYNTH_000, mission 5010" \
    --panel "$CITY" $OUT/city/FULL_RH_Astar_30m_paths $OUT/city/geo_paths.csv "$MISSION" "$TARGET_B" "$TITLE" \
    --output $OUT/fig2_paths
python "$HERE/fig3_data.py" \
    synthetic:FULL_RH_Astar=$OUT/full/FULL_RH_Astar_canonical-primary_episodes.csv \
    synthetic:STATIC_Astar=$OUT/static/STATIC_Astar_canonical-primary_episodes.csv \
    synthetic:GEO_SHORTEST=$OUT/geo/GEO_SHORTEST_canonical-primary_episodes.csv \
    city:FULL_RH_Astar=$OUT/city/FULL_RH_Astar_30m_episodes.csv \
    city:STATIC_Astar=$OUT/city/STATIC_Astar_30m_episodes.csv \
    city:GEO_SHORTEST=$OUT/city/GEO_SHORTEST_city-example_episodes.csv > $OUT/fig3_data.csv
cat $OUT/fig3_data.csv
