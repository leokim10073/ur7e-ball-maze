#!/usr/bin/env python3
"""Sweep the MPC's corner-speed reference (v_ref / v_turn / a_ref) in the offline sim.

    python3 scripts/compare/sweep_corner_speed.py [--runs 20] [--jobs 4]

The check behind the defaults in config/mpc.yaml: per setting, the share of
randomized runs that reach the goal without leaving the 5 mm corridor, the
median time to the goal and the worst distance outside, on the test paths of
common/scenarios.py plus two bends the planner produces: a 90 deg bend
straightened into two 45 deg bends (1 cm chamfer) and a 45 deg bend.

Keep --jobs below the core count: solve_budget is wall-clock time, so a loaded
CPU turns slow solves into fallbacks.
"""

import argparse
import multiprocessing as mp
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)

import compare_controllers as cc  # noqa: E402
from maze_control.common.ball_sim import BallSim, run_closed_loop  # noqa: E402
from maze_control.common.config import load_params  # noqa: E402
from maze_control.common.scenarios import SCENARIOS, corridors_for  # noqa: E402
from maze_control.mpc.controller import MpcPathController  # noqa: E402

_D = 0.01 / np.sqrt(2)
PATHS = dict(SCENARIOS)
PATHS['L chamfer'] = (np.array([[-0.18, 0.15], [-0.18, -0.15 + _D], [-0.18 + _D, -0.15],
                                [0.18, -0.15]]), (-0.18, 0.15), 14.0)
PATHS['45 deg'] = (np.array([[-0.20, -0.10], [0.0, -0.10], [0.15, 0.05]]), (-0.20, -0.10), 10.0)
SWEEP_PATHS = ('line', 'L', 'U', 'L chamfer', '45 deg')
# (label, v_ref, v_turn, a_ref); v_turn = v_ref is a constant reference speed
CONFIGS = [('const 0.10', 0.10, 0.10, 0.30), ('const 0.15', 0.15, 0.15, 0.30),
           ('0.20/0.08/0.30', 0.20, 0.08, 0.30), ('0.20/0.10/0.30', 0.20, 0.10, 0.30),
           ('0.20/0.12/0.30', 0.20, 0.12, 0.30), ('0.20/0.08/0.50', 0.20, 0.08, 0.50),
           ('0.25/0.08/0.30', 0.25, 0.08, 0.30), ('0.30/0.08/0.30', 0.30, 0.08, 0.30)]


def run(job):
    label, v_ref, v_turn, a_ref, scen, sim_kw, dp0 = job
    p = load_params('mpc.yaml')
    p['mpc'].update(v_ref=v_ref, v_turn=v_turn, a_ref=a_ref)
    path, p0, T = PATHS[scen]
    ctrl = MpcPathController.from_params(p)
    arc, corr, pc = corridors_for(path, cc.HALF_WIDTH)
    ctrl.on_route(path, arc, corr, pc, plan_id=1)
    sim = BallSim.from_params(load_params('ball_sim.yaml'), pos0=np.asarray(p0) + dp0, **sim_kw)
    log = run_closed_loop(ctrl, sim, path, T)
    v = cc.violation(log['pos'], corr)
    reached = np.array([s == 'reached' for s in log['status']])
    t = float(log['t'][np.argmax(reached)]) if reached.any() else np.nan
    return label, scen, t, float(v.max()), bool(reached.any() and v.max() <= cc.LEAVE_TOL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=int, default=20)
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--seed', type=int, default=2)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    draws = {s: [(dict(tau=rng.uniform(0.06, 0.10), delay=rng.uniform(0.04, 0.07),
                       c=rng.uniform(0.2, 0.4), noise_pos=3e-4, noise_vel=3e-3,
                       seed=int(rng.integers(1 << 30))), rng.uniform(-0.003, 0.003, 2))
                  for _ in range(a.runs)] for s in SWEEP_PATHS}
    jobs = [(*c, s, kw, dp) for c in CONFIGS for s in SWEEP_PATHS for kw, dp in draws[s]]
    with mp.get_context('spawn').Pool(a.jobs) as pool:
        res = pool.map(run, jobs, chunksize=2)
    print(f'{a.runs} randomized runs per path, ±{cc.HALF_WIDTH * 1000:.0f} mm corridors. '
          'Cells: in-corridor %, median time to goal, worst mm outside.')
    print(f"{'v_ref/v_turn/a_ref':18s}" + ''.join(f'{s:>20s}' for s in SWEEP_PATHS))
    for c in CONFIGS:
        row = f'{c[0]:18s}'
        for s in SWEEP_PATHS:
            r = [x for x in res if x[0] == c[0] and x[1] == s]
            row += (f"{100 * np.mean([x[4] for x in r]):6.0f}% "
                    f"{np.nanmedian([x[2] for x in r]):5.2f}s "
                    f"{max(x[3] for x in r) * 1000:5.1f}")
        print(row, flush=True)


if __name__ == '__main__':
    main()
