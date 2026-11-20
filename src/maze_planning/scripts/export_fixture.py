#!/usr/bin/env python3
"""Write a level's CAD map (Contract 1) and its designed route as a YAML test fixture for the C++
planner tests, which cannot import the Python CAD map builder.

    PYTHONPATH=src/maze_cad_map python3 src/maze_planning/scripts/export_fixture.py \
        --cad-scripts maze_CAD/scripts --level L1 --out src/maze_planning/test/data/L1.yaml
"""

import argparse
import sys
from pathlib import Path

import yaml

from maze_cad_map.cad_map import build_map, path_cells_board


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cad-scripts", required=True, help="maze_CAD/scripts")
    ap.add_argument("--level", default="L1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    m = build_map(args.cad_scripts, args.level)
    fixture = {
        "level": args.level,
        "resolution": m.resolution,
        "width": m.width,
        "height": m.height,
        "origin": list(m.origin),
        "data": m.occupancy_data(),
        "holes": [[h.x, h.y, h.radius] for h in m.holes],
        "start": list(m.start),
        "goal": [m.goal.x, m.goal.y, m.goal.radius],
        "route": [list(p) for p in path_cells_board(args.cad_scripts, args.level)],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = (f"# Generated from maze_CAD ({args.level}) by src/maze_planning/scripts/export_fixture.py;"
              " do not edit. Regenerate with the command in that script's docstring.\n")
    out.write_text(header + yaml.safe_dump(fixture, default_flow_style=None, width=120))
    return 0


if __name__ == "__main__":
    sys.exit(main())
