#!/usr/bin/env python3
"""Run every scenario in the offline sim and plot it (no ROS needed).

    python3 scripts/plot_trajectories.py [-o out.png]   # default: docs/media/pid_trajectories.png
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)

from maze_control.common.ball_sim import BallSim, run_closed_loop  # noqa: E402
from maze_control.common.config import load_params  # noqa: E402
from maze_control.pid.controller import PidPathController  # noqa: E402
from maze_control.common.scenarios import SCENARIOS  # noqa: E402

BLUE, ORANGE = '#2a78d6', '#eb6834'
INK, INK2, GRID, PATH = '#0b0b0b', '#52514e', '#e4e3df', '#9a9893'
TITLES = {'point': '1. Single point', 'line': '2. Straight line',
          'L': '3a. L path', 'U': '3b. U path'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--out',
                    default=os.path.join(PKG, '..', '..', 'docs', 'media',
                                         'pid_trajectories.png'))
    args = ap.parse_args()

    P, SP = load_params('pid.yaml'), load_params('ball_sim.yaml')
    max_tilt = float(np.degrees(P['limits']['max_tilt']))   # plotted in deg
    goal_tol = P['tracker']['goal_tol']

    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': GRID,
                         'axes.labelcolor': INK2, 'xtick.color': INK2,
                         'ytick.color': INK2, 'axes.titlecolor': INK,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(3, 4, figsize=(15, 10.5),
                             gridspec_kw={'height_ratios': [2.2, 1, 1]})

    for col, (name, (path, p0, T)) in enumerate(SCENARIOS.items()):
        ctrl = PidPathController.from_params(P)
        sim = BallSim.from_params(SP, pos0=p0)
        log = run_closed_loop(ctrl, sim, path, T)
        t, pos, cmd = log['t'], log['pos'], np.degrees(log['cmd'])
        d_goal = np.linalg.norm(pos - path[-1], axis=1)
        reached = d_goal < goal_tol
        t_reach = t[np.argmax(reached)] if reached.any() else np.nan
        max_xt = log['cross_track'].max()

        # Row 1: board-frame trajectory
        ax = axes[0, col]
        if len(path) > 1:
            ax.plot(path[:, 0] * 100, path[:, 1] * 100, '--', color=PATH,
                    lw=2, label='path')
        ax.add_patch(plt.Circle(path[-1] * 100, goal_tol * 100, fill=False,
                                ec=PATH, lw=1, ls=':'))
        ax.plot(pos[:, 0] * 100, pos[:, 1] * 100, color=BLUE, lw=2,
                label='ball')
        ax.plot(*(np.array(p0) * 100), 'o', ms=8, mfc='white', mec=INK2,
                mew=1.5, label='start')
        ax.plot(*(path[-1] * 100), '*', ms=13, color=INK, label='goal')
        ax.set_xlim(-25, 25), ax.set_ylim(-25, 25), ax.set_aspect('equal')
        ax.grid(color=GRID, lw=0.6)
        ax.set_xlabel('x [cm]'), ax.set_ylabel('y [cm]')
        sub = (f'final err {d_goal[-1] * 1000:.2f} mm' if name == 'point'
               else f'max cross-track {max_xt * 1000:.1f} mm')
        ax.set_title(f'{TITLES[name]}\nwithin 2 cm at {t_reach:.2f} s · {sub}',
                     loc='left')
        if col == 0:
            ax.legend(loc='lower right', frameon=False, fontsize=8)

        # Row 2: tilt commands with the limits
        ax = axes[1, col]
        for lim in (-max_tilt, max_tilt):
            ax.axhline(lim, color=INK2, lw=0.8, ls=':')
        ax.plot(t, cmd[:, 0], color=BLUE, lw=1.5, label='tilt_x cmd')
        ax.plot(t, cmd[:, 1], color=ORANGE, lw=1.5, label='tilt_y cmd')
        ax.set_ylim(-max_tilt * 1.25, max_tilt * 1.25)
        ax.set_ylabel('tilt [deg]'), ax.grid(color=GRID, lw=0.6)
        if col == 0:
            ax.legend(loc='upper right', frameon=False, fontsize=8, ncol=2)
            ax.text(0.01, max_tilt, f' ±{max_tilt:.1f}° limit', va='bottom', color=INK2,
                    fontsize=7.5, transform=ax.get_yaxis_transform())

        # Row 3: distance to goal and cross-track error
        ax = axes[2, col]
        ax.plot(t, d_goal * 100, color=BLUE, lw=1.5, label='to goal')
        if len(path) > 1:
            ax.plot(t, log['cross_track'] * 100, color=ORANGE, lw=1.5,
                    label='cross-track')
        ax.axhline(goal_tol * 100, color=INK2, lw=0.8, ls=':')
        ax.set_xlabel('time [s]'), ax.set_ylabel('error [cm]')
        ax.grid(color=GRID, lw=0.6)
        ax.legend(loc='upper right', frameon=False, fontsize=8)

    fig.suptitle('PID path tracking in the offline sim  '
                 f'(L={P["tracker"]["lookahead"] * 100:.0f} cm, '
                 f'kp={P["pid"]["kp"]}, kd={P["pid"]["kd"]}, ki={P["pid"]["ki"]}; '
                 f'τ={SP["tau"] * 1000:.0f} ms, delay={SP["delay"] * 1000:.0f} ms, '
                 f'c={SP["c"]} 1/s)', x=0.01, ha='left', fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=130, facecolor='white')
    print(f'wrote {os.path.normpath(args.out)}')


if __name__ == '__main__':
    main()
