#!/usr/bin/env python3
"""PID vs LQR vs MPC in the offline sim (no ROS; MPC needs casadi).

    python3 scripts/compare/compare_controllers.py [--runs 50] [--jobs 4] [--out-dir DIR]
        [--ctrls PID,LQR,MPC,...]

Writes controller_trajectories.png and controller_comparison.png to
results/compare/ at the repo root (git-ignored) and prints a summary table. Corridors are 5 mm either side of the path (ball centre, wall
clearance already taken off, contract 3); PID and LQR do not see them, MPC
keeps its prediction inside them.

"Left the corridor" = the true ball centre was more than 1 mm outside every
corridor box (into the planner's wall margin). A run succeeds when the ball
reaches the goal (within goal_tol) inside the scenario's time limit and never
left the corridor.

MPC variants (all mpc.yaml otherwise; corridors as above):
  MPC           mpc.yaml as is
  MPC 0.1       constant 0.1 m/s reference (no corner slowing)
  MPC no P      input_ref zero, no terminal cost (mpc.yaml before the terminal cost)
  MPC u_ss      input cost r |u - u_ss|^2, u_ss the tilt that holds the reference speed
  MPC u_ss+P    MPC u_ss with the LQR's Riccati P as terminal cost
  MPC LQR cost  the LQR's cost: every horizon step tracks the LQR's target (the
                tracker's lookahead point) with zero reference velocity, + P
"""

import argparse
import copy
import multiprocessing as mp
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PKG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PKG)

from maze_control.common.ball_sim import BallSim, run_closed_loop  # noqa: E402
from maze_control.common.config import load_params  # noqa: E402
from maze_control.common.scenarios import SCENARIOS, corridors_for  # noqa: E402
from maze_control.lqr.controller import LqrPathController  # noqa: E402
from maze_control.mpc.controller import MpcPathController  # noqa: E402
from maze_control.pid.controller import PidPathController  # noqa: E402

# categorical slots of the dataviz reference palette, fixed per controller
COLORS = {'PID': '#2a78d6', 'LQR': '#eb6834', 'MPC': '#1baf7a', 'MPC 0.1': '#eda100',
          'MPC no P': '#008300', 'MPC u_ss': '#e87ba4', 'MPC u_ss+P': '#6250d6', 'MPC LQR cost': '#e34948'}
INK, INK2, GRID, PATH, CORR = '#0b0b0b', '#52514e', '#e4e3df', '#9a9893', '#ecebe7'
CTRLS = ['PID', 'LQR', 'MPC 0.1', 'MPC']     # --ctrls replaces it (in every worker)
CONST_V_REF = 0.1     # 'MPC 0.1': mpc.yaml with a constant reference speed (no corner slowing)
PATHS = ('line', 'L', 'U')
HALF_WIDTH = 0.005
LEAVE_TOL = 0.001


def params():
    return {'PID': load_params('pid.yaml'), 'LQR': load_params('lqr.yaml'),
            'MPC': load_params('mpc.yaml')}


class LqrCostMpc(MpcPathController):
    """The LQR's cost in the MPC: every step of the horizon tracks the LQR's
    target (the tracker's lookahead point, self.target) at zero reference
    velocity; the corridors still follow the path."""

    def _law_step(self, dt):
        ref = self.horizon_ref()
        ref.R[:] = np.asarray(self.target, float).reshape(2, 1)
        ref.V[:] = 0.0
        return self.law.step(ref, self.pos, self.vel, dt)


MPC_OVERRIDES = {'MPC': {}, 'MPC 0.1': {'v_ref': CONST_V_REF, 'v_turn': CONST_V_REF},
                 'MPC no P': {'input_ref': 'zero', 'terminal_cost': 'none'},
                 'MPC u_ss': {'input_ref': 'steady', 'terminal_cost': 'none'},
                 'MPC u_ss+P': {'input_ref': 'steady', 'terminal_cost': 'lqr'},
                 'MPC LQR cost': {'input_ref': 'zero', 'terminal_cost': 'lqr'}}


def make(name, P, path, half_width):
    if name == 'PID':
        return PidPathController.from_params(P['PID'])
    if name == 'LQR':
        return LqrPathController.from_params(P['LQR'])
    p = copy.deepcopy(P['MPC'])
    p['mpc'].update(MPC_OVERRIDES[name])
    cls = LqrCostMpc if name == 'MPC LQR cost' else MpcPathController
    ctrl = cls.from_params(p)
    arc, corr, pc = corridors_for(path, half_width)
    ctrl.on_route(path, arc, corr, pc, plan_id=1)
    return ctrl


def violation(pos, corr):
    pos = np.asarray(pos)[:, None, :]
    lo, hi = corr[None, :, [0, 2]], corr[None, :, [1, 3]]
    out = np.maximum(lo - pos, 0.0) + np.maximum(pos - hi, 0.0)
    return np.linalg.norm(out, axis=2).min(axis=1)


def run(job):
    """One closed loop. job = (ctrl, scenario, half_width, sim_overrides, p0_offset)."""
    name, scen, hw, sim_kw, dp0 = job
    P, SP = params(), load_params('ball_sim.yaml')
    path, p0, T = SCENARIOS[scen]
    p0 = np.asarray(p0) + np.asarray(dp0)
    ctrl = make(name, P, path, hw)
    log = run_closed_loop(ctrl, BallSim.from_params(SP, pos0=p0, **sim_kw), path, T)
    _, corr, _ = corridors_for(path, hw)
    v = violation(log['pos'], corr)
    reached = np.array([s == 'reached' for s in log['status']])
    t_reach = float(log['t'][np.argmax(reached)]) if reached.any() else np.nan
    solve = list(ctrl.law.solve_times) if name.startswith('MPC') else []
    return dict(ctrl=name, scen=scen, hw=hw, sim=sim_kw, t=log['t'], pos=log['pos'],
                cmd=log['cmd'], viol=v, t_reach=t_reach, max_viol=float(v.max()),
                frac_out=float(np.mean(v > LEAVE_TOL)), max_xt=float(log['cross_track'].max()),
                tilt_rms=float(np.sqrt(np.mean(log['cmd'] ** 2))), solve=solve,
                reached=bool(reached.any()),
                success=bool(reached.any() and v.max() <= LEAVE_TOL))


def monte_carlo_jobs(n, seed=0):
    rng = np.random.default_rng(seed)
    jobs = []
    for scen in PATHS:
        for i in range(n):
            sim_kw = dict(tau=rng.uniform(0.06, 0.10), delay=rng.uniform(0.04, 0.07),
                          c=rng.uniform(0.2, 0.4), noise_pos=3e-4, noise_vel=3e-3,
                          seed=int(rng.integers(1 << 30)))
            dp0 = rng.uniform(-0.003, 0.003, 2)
            for name in CTRLS:
                jobs.append((name, scen, HALF_WIDTH, sim_kw, dp0))
    return jobs


def style():
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2,
                         'xtick.color': INK2, 'ytick.color': INK2, 'axes.titlecolor': INK,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titlelocation': 'left'})


def plot_trajectories(nominal, out):
    fig, axes = plt.subplots(3, 3, figsize=(15, 11.5),
                             gridspec_kw={'height_ratios': [2.3, 1, 1]})
    for col, scen in enumerate(PATHS):
        path = SCENARIOS[scen][0]
        _, corr, _ = corridors_for(path, HALF_WIDTH)
        ax = axes[0, col]
        for c in corr:
            ax.add_patch(plt.Rectangle((c[0] * 100, c[2] * 100), (c[1] - c[0]) * 100,
                                       (c[3] - c[2]) * 100, fc=CORR, ec='none', zorder=0))
        ax.plot(path[:, 0] * 100, path[:, 1] * 100, '--', color=PATH, lw=1, zorder=1)
        lines = []
        for name in CTRLS:
            r = nominal[(name, scen)]
            ln, = ax.plot(r['pos'][:, 0] * 100, r['pos'][:, 1] * 100, color=COLORS[name],
                          lw=2, zorder=2, label=name)
            lines.append(ln)
        ax.plot(*(path[-1] * 100), '*', ms=12, color=INK, zorder=3)
        ax.set_aspect('equal'), ax.grid(color=GRID, lw=0.6)
        ax.set_xlim(-25, 25), ax.set_ylim(-22, 22)
        ax.set_xlabel('x [cm]'), ax.set_ylabel('y [cm]')
        sub = '\n'.join(f"{n}: {nominal[(n, scen)]['t_reach']:.2f} s, "
                         f"{nominal[(n, scen)]['max_viol'] * 1000:.1f} mm outside" for n in CTRLS)
        # more than four controllers: the per-controller lines crowd the axes out
        # (the same numbers are printed)
        ax.set_title(f'{scen} path (grey: ±5 mm corridor)' + (f'\n{sub}' if len(CTRLS) <= 4 else ''),
                     fontsize=8.5)
        if col == 0:
            ax.legend(handles=lines, loc='lower right', frameon=False)
        # inset at the corner(s): where the controllers differ
        if scen != 'line':
            corner = path[1]
            ins = ax.inset_axes([0.58, 0.58, 0.4, 0.4])
            for c in corr:
                ins.add_patch(plt.Rectangle((c[0] * 100, c[2] * 100), (c[1] - c[0]) * 100,
                                            (c[3] - c[2]) * 100, fc=CORR, ec='none'))
            for name in CTRLS:
                r = nominal[(name, scen)]
                ins.plot(r['pos'][:, 0] * 100, r['pos'][:, 1] * 100, color=COLORS[name], lw=1.5)
            ins.set_xlim(corner[0] * 100 - 2, corner[0] * 100 + 3)
            ins.set_ylim(corner[1] * 100 - 2, corner[1] * 100 + 3)
            ins.set_aspect('equal'), ins.set_xticks([]), ins.set_yticks([])
            ins.set_title('first corner, zoom', fontsize=7.5, color=INK2)
            for s in ins.spines.values():
                s.set_color(GRID)

        ax = axes[1, col]
        for name in CTRLS:
            r = nominal[(name, scen)]
            ax.plot(r['t'], r['viol'] * 1000, color=COLORS[name], lw=1.5, label=name)
        ax.axhline(LEAVE_TOL * 1000, color=INK2, lw=0.8, ls=':')
        ax.set_ylabel('outside corridor [mm]'), ax.grid(color=GRID, lw=0.6)
        if col == 0:
            ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.22 if len(CTRLS) <= 4 else 1.32),
                      frameon=False, ncol=min(len(CTRLS), 4), fontsize=8)
            ax.text(0.01, LEAVE_TOL * 1000, ' 1 mm: "left the corridor"', va='bottom',
                    color=INK2, fontsize=7.5, transform=ax.get_yaxis_transform())

        ax = axes[2, col]
        for name in CTRLS:
            r = nominal[(name, scen)]
            ax.plot(r['t'], np.degrees(np.abs(r['cmd']).max(axis=1)), color=COLORS[name],
                    lw=1.2, label=name)
        ax.axhline(np.degrees(0.08), color=INK2, lw=0.8, ls=':')
        ax.set_ylabel('|tilt| (larger axis) [deg]'), ax.set_xlabel('time [s]')
        ax.grid(color=GRID, lw=0.6)
    fig.suptitle(' vs '.join(CTRLS) + ', offline sim, nominal plant '
                 '(τ = 80 ms, delay = 50 ms, c = 0.3 /s, no noise)',
                 x=0.01, ha='left', fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=130, facecolor='white')


def bar_group(ax, values, ylabel, fmt, title, ylim=None):
    x = np.arange(len(PATHS))
    w = 0.8 / len(CTRLS)
    for i, name in enumerate(CTRLS):
        vals = [values[(name, s)] for s in PATHS]
        bars = ax.bar(x + (i - (len(CTRLS) - 1) / 2) * w, vals, w - 0.02, color=COLORS[name],
                      label=name)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() if np.isfinite(v) else 0,
                    fmt.format(v) if np.isfinite(v) else 'none', ha='center', va='bottom',
                    fontsize=6.5 if len(CTRLS) > 4 else 7.5, color=INK2,
                    rotation=0 if np.isfinite(v) and len(CTRLS) <= 4 else 90)
    ax.set_xticks(x, [f'{s} path' for s in PATHS])
    ax.set_ylabel(ylabel), ax.set_title(title, fontsize=9.5)
    ax.grid(axis='y', color=GRID, lw=0.6), ax.set_axisbelow(True)
    if ylim:
        ax.set_ylim(*ylim)


def plot_comparison(mc, width_sweep, delay_sweep, n, out, solve_ms):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    succ = {(c, s): 100 * np.mean([r['success'] for r in mc if r['ctrl'] == c and r['scen'] == s])
            for c in CTRLS for s in PATHS}
    t_mean = {}
    for c in CTRLS:
        for s in PATHS:
            ts = [r['t_reach'] for r in mc if r['ctrl'] == c and r['scen'] == s and r['success']]
            t_mean[(c, s)] = float(np.mean(ts)) if ts else np.nan
    reach = {(c, s): 100 * np.mean([r['reached'] for r in mc if r['ctrl'] == c and r['scen'] == s])
             for c in CTRLS for s in PATHS}
    viol = {(c, s): 1000 * np.median([r['max_viol'] for r in mc if r['ctrl'] == c and r['scen'] == s])
            for c in CTRLS for s in PATHS}
    bar_group(axes[0, 0], succ, 'success rate [%]', '{:.0f}',
              f'Success rate, {n} randomized runs per path:\nreached goal and never > 1 mm outside '
              f'(goal reached in {min(reach.values()):.0f}% of runs)',
              ylim=(0, 112 if len(CTRLS) <= 4 else 135))
    axes[0, 0].legend(loc='upper left', frameon=False, ncol=3, fontsize=7.5)
    bar_group(axes[0, 1], t_mean, 'time to goal [s]', '{:.1f}',
              'Mean time to goal (successful runs only; none = no successful run)')
    bar_group(axes[0, 2], viol, 'max outside corridor [mm]', '{:.1f}',
              'Median of each run\'s max distance outside the corridor')

    ax = axes[1, 0]
    for name in CTRLS:
        hw = sorted({k[1] for k in width_sweep if k[0] == name})
        ax.plot(np.array(hw) * 1000, [width_sweep[(name, h)] * 1000 for h in hw], 'o-',
                color=COLORS[name], lw=2, ms=6, label=name)
    ax.axhline(LEAVE_TOL * 1000, color=INK2, lw=0.8, ls=':')
    ax.set_xlabel('corridor half-width [mm]'), ax.set_ylabel('max outside corridor [mm]')
    ax.set_title('Corridor width (U path, nominal plant)', fontsize=9.5)
    ax.grid(color=GRID, lw=0.6), ax.legend(frameon=False)

    ax = axes[1, 1]
    for name in CTRLS:
        ds = sorted({k[1] for k in delay_sweep if k[0] == name})
        ax.plot(np.array(ds) * 1000, [delay_sweep[(name, d)]['max_viol'] * 1000 for d in ds],
                'o-', color=COLORS[name], lw=2, ms=6, label=name)
    ymax = 12.0
    for name in CTRLS:
        for d in sorted({k[1] for k in delay_sweep if k[0] == name}):
            v = delay_sweep[(name, d)]['max_viol'] * 1000
            if v > ymax:
                ax.annotate(f'{name}: {v:.0f} mm', (d * 1000, ymax), xytext=(-4, -2),
                            textcoords='offset points', ha='right', va='top', fontsize=7.5,
                            color=INK2, arrowprops=dict(arrowstyle='-|>', color=COLORS[name]))
    ax.set_ylim(-0.5, ymax)
    ax.axhline(LEAVE_TOL * 1000, color=INK2, lw=0.8, ls=':')
    ax.axvline(50, color=INK2, lw=0.8, ls=':')
    ax.text(50, 0.97, ' modelled delay', color=INK2, fontsize=7.5,
            transform=ax.get_xaxis_transform(), va='top')
    ax.set_xlabel('true plant delay [ms]'), ax.set_ylabel('max outside corridor [mm]')
    ax.set_title('Unmodelled delay (U path, ±5 mm; controllers assume 50 ms)', fontsize=9.5)
    ax.grid(color=GRID, lw=0.6), ax.legend(frameon=False, loc='center left')

    ax = axes[1, 2]
    st = solve_ms
    ax.hist(st, bins=np.linspace(0, max(17.5, st.max() * 1.05), 50), color=COLORS['MPC'])
    for x, lab, va in ((12, 'solve_budget 12 ms', 0.97), (16.7, 'period 16.7 ms', 0.85)):
        ax.axvline(x, color=INK2, lw=0.8, ls=':')
        ax.text(x, va, f' {lab}', color=INK2, fontsize=7.5,
                transform=ax.get_xaxis_transform(), va='top')
    ax.set_yscale('log')
    ax.set_xlabel('MPC solve time [ms]'), ax.set_ylabel('solves (log)')
    ax.set_title(f'MPC solve time, one process (randomized U runs)\n'
                 f'median {np.median(st):.1f} ms, p99 {np.percentile(st, 99):.1f} ms, '
                 f'max {st.max():.1f} ms', fontsize=9.5)
    ax.grid(color=GRID, lw=0.6)
    fig.suptitle('Randomized plant: τ 60-100 ms, delay 40-70 ms, c 0.2-0.4 /s, '
                 'noise 0.3 mm / 3 mm/s, start ±3 mm', x=0.01, ha='left', fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=130, facecolor='white')
    return succ, t_mean, viol, reach


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=int, default=50)
    ap.add_argument('--out-dir', default=os.path.join(PKG, '..', '..', 'results', 'compare'))
    ap.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--ctrls', default=','.join(CTRLS),
                    help=f'comma-separated, from {", ".join(COLORS)}')
    a = ap.parse_args()
    ctrls = a.ctrls.split(',')
    unknown = [c for c in ctrls if c not in COLORS]
    if unknown:
        ap.error(f'unknown controllers {unknown}')
    CTRLS[:] = ctrls
    style()
    os.makedirs(a.out_dir, exist_ok=True)

    nominal_jobs = [(c, s, HALF_WIDTH, {}, (0.0, 0.0)) for s in PATHS for c in CTRLS]
    widths = (0.003, 0.005, 0.01, 0.02)
    width_jobs = [(c, 'U', h, {}, (0.0, 0.0)) for h in widths for c in CTRLS]
    delays = (0.03, 0.05, 0.07, 0.09, 0.11)
    delay_jobs = [(c, 'U', HALF_WIDTH, {'delay': d}, (0.0, 0.0)) for d in delays for c in CTRLS]
    mc_jobs = monte_carlo_jobs(a.runs)

    with mp.get_context('spawn').Pool(a.jobs) as pool:
        nominal = {(r['ctrl'], r['scen']): r for r in pool.map(run, nominal_jobs)}
        width_sweep = {(r['ctrl'], r['hw']): r['max_viol'] for r in pool.map(run, width_jobs)}
        delay_sweep = {(r['ctrl'], r['sim']['delay']): r for r in pool.map(run, delay_jobs)}
        mc = pool.map(run, mc_jobs, chunksize=4)
    # solve times in one process: the pool's parallel runs compete for the CPU
    timing = [run(j) for j in mc_jobs if j[0] == 'MPC' and j[1] == 'U'][:20]
    solve_ms = np.array([t for r in timing for t in r['solve']]) * 1000

    plot_trajectories(nominal, os.path.join(a.out_dir, 'controller_trajectories.png'))
    succ, t_mean, viol, reach = plot_comparison(
        mc, width_sweep, delay_sweep, a.runs,
        os.path.join(a.out_dir, 'controller_comparison.png'), solve_ms)

    print('\nnominal plant, ±5 mm corridors')
    print(f"{'path':5} {'ctrl':12} {'t_reach':>8} {'max_out':>8} {'%t_out':>7} {'max_xt':>7} {'tilt_rms':>8}")
    for s in PATHS:
        for c in CTRLS:
            r = nominal[(c, s)]
            print(f"{s:5} {c:12} {r['t_reach']:7.2f}s {r['max_viol'] * 1000:6.1f}mm "
                  f"{100 * r['frac_out']:6.1f}% {r['max_xt'] * 1000:5.1f}mm "
                  f"{np.degrees(r['tilt_rms']):6.2f}deg")
    print(f'\nrandomized, {a.runs} runs per path: reached % / success % / mean t [s] / median max-out [mm]')
    for s in PATHS:
        print(f'{s:5} ' + '  '.join(f'{c} {reach[(c, s)]:3.0f}% {succ[(c, s)]:5.1f}% {t_mean[(c, s)]:5.2f}s '
                                    f'{viol[(c, s)]:4.1f}mm' for c in CTRLS))
    print('\ncorridor width sweep (U, max outside mm):')
    for h in widths:
        print(f'  ±{h * 1000:4.1f} mm  ' + '  '.join(f'{c} {width_sweep[(c, h)] * 1000:5.1f}'
                                               for c in CTRLS))
    print('delay sweep (U, ±5 mm: max outside mm / t_reach s):')
    for d in delays:
        print(f'  {d * 1000:4.0f} ms  ' + '  '.join(
            f"{c} {delay_sweep[(c, d)]['max_viol'] * 1000:5.1f}/{delay_sweep[(c, d)]['t_reach']:5.2f}"
            for c in CTRLS))
    st = solve_ms
    print(f'MPC solve (one process, {len(timing)} U runs): median {np.median(st):.2f} ms, '
          f'p99 {np.percentile(st, 99):.2f} ms, max {st.max():.2f} ms, > 12 ms: {100 * np.mean(st > 12):.2f}%')


if __name__ == '__main__':
    main()
