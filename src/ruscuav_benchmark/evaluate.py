"""``ruscuav-evaluate``: the same as ``ruscuav evaluate`` (planner or policy evaluation).

The planners and their evaluation are in :mod:`ruscuav_benchmark.planners`, the
policies in :mod:`ruscuav_benchmark.policies`.
"""

from __future__ import annotations

import sys
from typing import List, Optional

from .scenarios import read_scenarios

__all__ = ["read_scenarios", "main"]


def main(argv: Optional[List[str]] = None) -> int:
    from .cli import main as cli_main

    return cli_main(["evaluate", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    sys.exit(main())
