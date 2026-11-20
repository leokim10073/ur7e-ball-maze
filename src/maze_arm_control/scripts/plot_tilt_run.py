#!/usr/bin/env python3
"""Plot the CSV runs test_sim_tilt.py writes (TILT_RUN_DIR): commanded vs measured tilt (sim ground
truth) and how far the board center moved.
    python3 plot_tilt_run.py RUN_DIR OUT.png [run ...]     # default runs: step_+x step_+y sine_x sine_y
"""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DEG = 180.0 / np.pi


def load(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    return {k: np.array([float(r[i]) for r in rows[1:]]) for i, k in enumerate(rows[0])}


def main():
    run_dir, out = Path(sys.argv[1]), sys.argv[2]
    runs = sys.argv[3:] or ["step_+x", "step_+y", "sine_x", "sine_y"]
    fig, axes = plt.subplots(2, len(runs), figsize=(4.2 * len(runs), 6), sharex="col", squeeze=False)
    for col, name in enumerate(runs):
        d = load(run_dir / f"{name}.csv")
        axis = "y" if name.endswith("y") else "x"
        ax = axes[0, col]
        ax.plot(d["t"], d[f"cmd_{axis}"] * DEG, color="#888888", lw=1.5, label="tilt_cmd")
        ax.plot(d["t"], d[f"tilt_{axis}"] * DEG, color="#1f6feb", lw=1.5, label="board (sim ground truth)")
        ax.plot(d["t"], d[f"fk_{axis}"] * DEG, color="#d97706", lw=1, ls="--", label="tilt_state (FK)")
        ax.set_title(f"{name}")
        ax.set_ylabel(f"tilt_{axis} (deg)")
        ax.grid(alpha=0.3)
        err = d[f"tilt_{axis}"] - d[f"cmd_{axis}"]
        ax2 = axes[1, col]
        ax2.plot(d["t"], np.abs(err) * DEG, color="#1f6feb", lw=1.2, label="|tilt error| (deg)")
        ax2.plot(d["t"], 1000 * np.sqrt(d["dx"] ** 2 + d["dy"] ** 2 + d["dz"] ** 2), color="#16a34a", lw=1.2,
                 label="board center moved (mm)")
        ax2.axhline(0.5, color="#1f6feb", lw=0.8, ls=":")
        ax2.axhline(5.0, color="#16a34a", lw=0.8, ls=":")
        ax2.set_xlabel("sim time (s)")
        ax2.set_ylim(0, 6)
        ax2.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8, loc="lower right")
    axes[1, 0].legend(fontsize=8, loc="upper right")
    fig.suptitle("maze_arm_control in the MuJoCo sim: tilt tracking about the board center "
                 "(limits dotted: 0.5 deg, 5 mm)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
