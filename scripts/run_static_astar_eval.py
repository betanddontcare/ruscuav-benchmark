#!/usr/bin/env python3
"""Fly the static risk-cost A* planner (STATIC_Astar) on a world.

The same as ``ruscuav evaluate --planner STATIC_Astar ...``; run with ``--help`` for
the options and see docs/PLANNERS.md.
"""

import sys

from ruscuav_benchmark.cli import main

if __name__ == "__main__":
    sys.exit(main(["evaluate", "--planner", "STATIC_Astar", *sys.argv[1:]]))
