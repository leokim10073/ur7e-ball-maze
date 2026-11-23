"""MPC controller: the CFTOC (limits, corridors, agreement with the LQR,
solve time), offline and without ROS."""

import copy
import time

import numpy as np
import pytest

from maze_control.lqr.lqr_law import ball_model, c2d, design_gain
from maze_control.common.scenarios import SCENARIOS, corridors_for
from maze_control.lqr.controller import LqrPathController
from maze_control.common.ball_sim import BallSim, run_closed_loop
from maze_control.mpc.controller import MpcPathController
from maze_control.mpc.mpc_law import CFTOC, HorizonRef, MPCController

EPS = 1e-6
INF = np.inf


def make_cftoc(p, horizon=30, **overrides):
    m, w, lim = p['model'], p['lqr'], p['limits']
    dt = 1.0 / p['rate_hz']
    Ad, Bd = c2d(*ball_model(m['c'], m['tau'], m['g']), dt)
    kw = dict(horizon=horizon, q_pos=w['q_pos'], q_vel=w['q_vel'], r=w['r'],
              r_du=0.0, rho_lin=1e4, rho_quad=1e4, max_tilt=lim['max_tilt'],
              max_rate=lim['max_rate'],
              osqp_opts={'max_iter': 4000, 'eps_abs': 1e-5, 'eps_rel': 1e-5,
                         'polish': True})
    kw.update(overrides)
    return CFTOC(Ad, Bd, dt, **kw)


@pytest.fixture
def cftoc(lqr_params):
    return make_cftoc(lqr_params)


def ref_const(N, R=(0.0, 0.0), lo=(-INF, -INF), hi=(INF, INF)):
    col = lambda v: np.tile(np.asarray(v, float).reshape(2, 1), (1, N))  # noqa: E731
    return HorizonRef(R=col(R), V=np.zeros((2, N)), lo=col(lo), hi=col(hi))


def state(px=0.0, vx=0.0, py=0.0, vy=0.0):
    return np.array([px, vx, 0.0, py, vy, 0.0])


def check_plan_limits(plan, u_prev, max_tilt, max_rate, dt):
    U = np.asarray(plan.U)
    assert np.abs(U).max() <= max_tilt + EPS
    du = np.diff(np.hstack([np.asarray(u_prev, float).reshape(2, 1), U]), axis=1)
    assert np.abs(du).max() <= max_rate * dt + EPS


# ------------------------------------------------------------------ CFTOC
def test_plan_respects_tilt_and_rate_limits(cftoc, lqr_params):
    lim = lqr_params['limits']
    u_prev = np.zeros(2)
    plan = cftoc.solve(state(px=-0.15), ref_const(cftoc.N), u_prev)
    assert plan.ok
    check_plan_limits(plan, u_prev, lim['max_tilt'], lim['max_rate'], cftoc.dt)
    # far from the reference, the first steps ramp at the rate limit
    assert plan.U[0, 0] == pytest.approx(lim['max_rate'] * cftoc.dt, abs=1e-6)


def test_at_rest_on_reference_commands_zero(cftoc):
    plan = cftoc.solve(state(), ref_const(cftoc.N), np.zeros(2))
    assert plan.ok
    assert np.abs(plan.U).max() < 1e-5


def test_unconstrained_long_horizon_matches_lqr(lqr_params):
    m, w = lqr_params['model'], lqr_params['lqr']
    dt = 1.0 / lqr_params['rate_hz']
    K, _, _ = design_gain(m['c'], m['tau'], dt, w['q_pos'], w['q_vel'],
                          w['q_tilt'], w['r'], q_int=0.0, g=m['g'])
    mpc = make_cftoc(lqr_params, horizon=200)
    ex, ey = 5e-4, -5e-4
    plan = mpc.solve(state(px=ex, py=ey), ref_const(mpc.N), np.zeros(2))
    assert plan.ok
    u_lqr = -K[:3] @ np.array([[ex, ey], [0.0, 0.0], [0.0, 0.0]])
    assert plan.U[:, 0] == pytest.approx(u_lqr, rel=0.02)


def test_corridor_bounds_the_prediction(cftoc):
    """The reference lies past a wall at x = 0.03: free, the plan goes to it;
    with the corridor, the predicted ball stops at the wall."""
    x0, R = state(vx=0.1), (0.06, 0.0)   # already rolling towards the wall
    free = cftoc.solve(x0, ref_const(cftoc.N, R=R), np.zeros(2))
    assert free.ok and free.X[0].max() > 0.035, \
        f'precondition: unconstrained peak {free.X[0].max():.4f} m'
    plan = cftoc.solve(x0, ref_const(cftoc.N, R=R, hi=(0.03, INF)), np.zeros(2))
    assert plan.ok
    assert plan.X[0, 1:].max() <= 0.03 + 1e-4
    assert plan.S.max() < 1e-4


def test_outside_corridor_still_solves(cftoc, lqr_params):
    lim = lqr_params['limits']
    plan = cftoc.solve(state(px=0.05), ref_const(cftoc.N, hi=(0.03, INF)),
                       np.zeros(2))
    assert plan.ok
    assert plan.S.max() > 0.0
    check_plan_limits(plan, np.zeros(2), lim['max_tilt'], lim['max_rate'],
                      cftoc.dt)


def test_first_solve_is_fast_after_warmup(lqr_params):
    mpc = make_cftoc(lqr_params)
    ref = ref_const(mpc.N, R=(0.05, 0.02), lo=(-0.03, -0.03), hi=(0.1, 0.1))
    first = mpc.solve(state(), ref, np.zeros(2))
    assert first.t_solve < 0.0167
    times = []
    for k in range(50):
        plan = mpc.solve(state(px=0.001 * k), ref, np.zeros(2))
        times.append(plan.t_solve)
    assert np.median(times) < 0.008


# ------------------------------------------------------------ MPCController
MPC_BLOCK = dict(horizon=30, q_pos=2500.0, q_vel=100.0, r=156.0, r_du=0.0,
                 rho_lin=1e4, rho_quad=1e4, solve_budget=0.012, max_fallback=5,
                 osqp={'max_iter': 4000, 'eps_abs': 1e-5, 'eps_rel': 1e-5,
                       'polish': True})


@pytest.fixture
def mpc_ctrl(lqr_params):
    p = lqr_params
    return MPCController.from_params(p['model'], dict(MPC_BLOCK), p['limits'],
                                     p['rate_hz'], g=p['model']['g'])


def far_ref(N, R=(0.15, -0.1)):
    return ref_const(N, R=R)


def check_rate(outputs, dt, max_tilt, max_rate):
    cmd = np.asarray(outputs)
    assert np.abs(cmd).max() <= max_tilt + EPS
    rate = np.abs(np.diff(np.vstack([np.zeros(2), cmd]), axis=0)) / dt
    assert rate.max() <= max_rate + 1e-6


def test_step_output_is_limited_and_tracked_by_observer(mpc_ctrl, lqr_params):
    ctrl, lim = mpc_ctrl, lqr_params['limits']
    dt = ctrl.dt
    lag = ctrl.lag
    outs, theta = [], np.zeros(2)
    for _ in range(30):
        u = ctrl.step(far_ref(ctrl.N), np.zeros(2), np.zeros(2), dt)
        outs.append(u)
        k = len(outs) - 1 - ctrl.n_delay
        acting = outs[k] if k >= 0 else np.zeros(2)
        theta = lag * theta + (1.0 - lag) * acting
        assert ctrl.theta == pytest.approx(theta, abs=1e-12)
    assert ctrl.mode == 'ok'
    check_rate(outs, dt, lim['max_tilt'], lim['max_rate'])
    assert np.abs(outs[-1]).max() > 0.0


def test_delay_prediction_uses_commands_in_flight(mpc_ctrl):
    ctrl = mpc_ctrl
    assert ctrl.n_delay == 3                       # 0.05 s at 60 Hz
    for _ in range(3):
        ctrl.step(far_ref(ctrl.N), np.zeros(2), np.zeros(2), ctrl.dt)
    pending = [u.copy() for u in ctrl._pending]
    assert np.abs(np.array(pending)).max() > 0.0
    pos, vel, theta = np.array([0.01, -0.02]), np.array([0.05, 0.03]), ctrl.theta.copy()

    seen = {}
    solve = ctrl.cftoc.solve

    def spy(x0, ref, u_prev):
        seen['x0'] = np.array(x0)
        return solve(x0, ref, u_prev)
    ctrl.cftoc.solve = spy
    ctrl.step(far_ref(ctrl.N), pos, vel, ctrl.dt)

    x = np.vstack([pos, vel, theta])               # 3 x 2, column per axis
    for u in pending:
        x = ctrl.Ad @ x + ctrl.Bd @ u.reshape(1, 2)
    expected = np.concatenate([x[:, 0], x[:, 1]])  # [px, vx, thx, py, vy, thy]
    assert seen['x0'] == pytest.approx(expected, abs=1e-12)


def fail_qp(args):
    return None, False


def test_failure_applies_shifted_plan_then_levels(mpc_ctrl, lqr_params):
    ctrl = mpc_ctrl
    du = lqr_params['limits']['max_rate'] * ctrl.dt
    ref = far_ref(ctrl.N)
    ctrl.step(ref, np.zeros(2), np.zeros(2), ctrl.dt)
    good = ctrl.last_plan
    assert ctrl.mode == 'ok' and good is not None
    ctrl.cftoc._solve_qp = fail_qp
    max_fb = MPC_BLOCK['max_fallback']
    for j in range(1, max_fb + 1):
        u_before = ctrl.u.copy()
        u = ctrl.step(ref, np.zeros(2), np.zeros(2), ctrl.dt)
        expected = u_before + np.clip(good.U[:, j] - u_before, -du, du)
        assert ctrl.mode == 'fallback' and ctrl.n_fail == j
        assert u == pytest.approx(expected, abs=1e-12)
    u_before = ctrl.u.copy()
    u = ctrl.step(ref, np.zeros(2), np.zeros(2), ctrl.dt)
    assert ctrl.mode == 'level'
    assert np.all(np.abs(u) <= np.abs(u_before) + 1e-12)
    assert np.abs(u - u_before).max() <= du + 1e-12


def test_slow_solve_counts_as_failure(mpc_ctrl):
    ctrl = mpc_ctrl
    ref = far_ref(ctrl.N)
    ctrl.step(ref, np.zeros(2), np.zeros(2), ctrl.dt)
    real = ctrl.cftoc._solve_qp

    def slow(args):
        time.sleep(MPC_BLOCK['solve_budget'] + 0.005)
        return real(args)
    ctrl.cftoc._solve_qp = slow
    ctrl.step(ref, np.zeros(2), np.zeros(2), ctrl.dt)
    assert ctrl.mode == 'fallback' and ctrl.n_fail == 1


def test_nan_state_is_a_failure_without_solving(mpc_ctrl):
    ctrl = mpc_ctrl
    ref = far_ref(ctrl.N)
    ctrl.step(ref, np.zeros(2), np.zeros(2), ctrl.dt)
    calls = []
    real = ctrl.cftoc._solve_qp
    ctrl.cftoc._solve_qp = lambda args: (calls.append(1), real(args))[1]
    u = ctrl.step(ref, np.array([np.nan, 0.0]), np.zeros(2), ctrl.dt)
    assert not calls and ctrl.mode == 'fallback'
    assert np.all(np.isfinite(u))


def test_reset_clears_plan_and_failures(mpc_ctrl):
    ctrl = mpc_ctrl
    ref = far_ref(ctrl.N)
    ctrl.step(ref, np.zeros(2), np.zeros(2), ctrl.dt)
    ctrl.cftoc._solve_qp = fail_qp
    ctrl.step(ref, np.zeros(2), np.zeros(2), ctrl.dt)
    assert ctrl.mode == 'fallback'
    ctrl.reset()
    assert ctrl.n_fail == 0 and ctrl.last_plan is None
    ctrl.step(ref, np.zeros(2), np.zeros(2), ctrl.dt)
    assert ctrl.mode == 'level'


# ------------------------------------------------------------ path controller
def started(params, name, pos=None, vel=(0.0, 0.0), half_width=0.02, plan_id=1):
    """A path controller on a scenario's path, one update done."""
    path, p0, _ = SCENARIOS[name]
    ctrl = MpcPathController.from_params(params)
    arc, corr, pose_corr = corridors_for(path, half_width)
    assert ctrl.on_route(path, arc, corr, pose_corr, plan_id)
    ctrl.on_state(0.0, p0 if pos is None else pos, vel)
    ctrl.update(0.0)
    return ctrl, path


def arc_of(ctrl, pts):
    """Arc length of points that lie on the tracker's path."""
    return np.array([ctrl.tracker._project(np.asarray(p))[0] for p in pts.T])


def test_mpc_params_within_contract_4(mpc_params):
    lim = mpc_params['limits']
    assert lim['max_tilt'] <= 0.087 and lim['max_rate'] <= 0.5
    assert mpc_params['mpc']['osqp']['polish'] is True


def test_horizon_reference_advances_and_stops_at_goal(mpc_params):
    m = mpc_params['mpc']
    ctrl, path = started(mpc_params, 'line')
    dt, nd = ctrl.law.dt, ctrl.law.n_delay
    ref = ctrl.horizon_ref()
    s = arc_of(ctrl, ref.R)
    s0 = ctrl.tracker.progress
    assert s[0] == pytest.approx(s0 + m['v_ref'] * dt * (nd + 1), abs=1e-9)
    assert np.diff(s) == pytest.approx(m['v_ref'] * dt, abs=1e-9)
    t = ctrl.tracker.tangent_at(s0)
    assert ref.V[:, 0] == pytest.approx(m['v_ref'] * t, abs=1e-9)

    # near the goal: the spacing shrinks and the reference never passes it
    L = ctrl.tracker.length
    near = path[0] + (L - 0.03) / L * (path[-1] - path[0])
    ctrl.tracker.progress = L - 0.03
    ctrl.on_state(0.1, near, (0.0, 0.0))
    ctrl.update(0.1)
    s = arc_of(ctrl, ctrl.horizon_ref().R)
    assert s.max() <= L + 1e-9
    d = np.diff(s)
    assert np.all(d >= -1e-12) and d[-1] < d[0]


def test_corridor_lookup_follows_arc_length(mpc_params):
    params = copy.deepcopy(mpc_params)
    params['mpc']['v_ref'] = 1.0           # the horizon reaches past the corner
    params['mpc']['v_turn'] = 1.0          # without slowing for it
    ctrl, path = started(params, 'L', pos=(-0.18, -0.05))
    ref = ctrl.horizon_ref()
    _, corr, _ = corridors_for(path, 0.02)
    s = arc_of(ctrl, ref.R)
    corner = 0.30
    first, second = s < corner - 1e-9, s > corner + 1e-9
    assert first.any() and second.any()
    assert np.allclose(ref.lo[:, first], [[corr[0, 0]], [corr[0, 2]]])
    assert np.allclose(ref.hi[:, first], [[corr[0, 1]], [corr[0, 3]]])
    assert np.allclose(ref.lo[:, second], [[corr[1, 0]], [corr[1, 2]]])
    assert np.allclose(ref.hi[:, second], [[corr[1, 1]], [corr[1, 3]]])


def test_same_points_new_plan_id_starts_new_run(mpc_params):
    ctrl, path = started(mpc_params, 'L')
    arc, corr, pose_corr = corridors_for(path, 0.02)
    ctrl.law.cftoc._solve_qp = fail_qp
    ctrl.update(1 / 60)
    assert ctrl.law.n_fail == 1
    ctrl.stopped = True
    assert not ctrl.on_route(path, arc, corr, pose_corr, plan_id=1)
    assert ctrl.stopped
    assert ctrl.on_route(path, arc, corr, pose_corr, plan_id=2)
    assert not ctrl.stopped and ctrl.law.n_fail == 0 and ctrl.law.last_plan is None


def test_single_point_path_has_zero_velocity_reference(mpc_params):
    ctrl, path = started(mpc_params, 'point')
    ref = ctrl.horizon_ref()
    assert np.all(np.isfinite(ref.R)) and np.all(np.isfinite(ref.V))
    assert np.abs(ref.V).max() == 0.0
    assert ref.R == pytest.approx(np.tile(path[0].reshape(2, 1), (1, ctrl.law.N)))


def test_empty_route_levels_the_board(mpc_params):
    ctrl, path = started(mpc_params, 'L')
    for k in range(1, 20):                    # build up some tilt
        ctrl.on_state(k / 60, SCENARIOS['L'][1], (0.0, 0.0))
        u = ctrl.update(k / 60)
    assert np.abs(u).max() > 0.0
    assert ctrl.on_route([], [], [], [], plan_id=7)
    t, du = 20 / 60, mpc_params['limits']['max_rate'] / 60
    ctrl.on_state(t, SCENARIOS['L'][1], (0.0, 0.0))
    u2 = ctrl.update(t)
    assert ctrl.status == 'no_path'
    assert np.all(np.abs(u2) <= np.abs(u) + 1e-12)
    assert np.abs(u2 - u).max() <= du + 1e-12


# ------------------------------------------------------------ closed loop
def dist_to_goal(log, path):
    return np.linalg.norm(log['pos'] - path[-1], axis=1)


def corridor_violation(pos, corr):
    """Distance of each position outside the union of the boxes [m]."""
    pos = np.asarray(pos)[:, None, :]                       # T x 1 x 2
    lo, hi = corr[None, :, [0, 2]], corr[None, :, [1, 3]]   # 1 x M x 2
    out = np.maximum(lo - pos, 0.0) + np.maximum(pos - hi, 0.0)
    return np.linalg.norm(out, axis=2).min(axis=1)


# 'point' runs without corridors: its one box is around the goal and does not
# hold the start, which a planner's route always does (contract 3).
@pytest.mark.parametrize('name, half_width', [('point', None), ('line', 0.02),
                                              ('L', 0.02), ('U', 0.02)])
def test_mpc_reaches_goal_on_all_scenarios(run_mpc, mpc_params, name, half_width):
    log, path, ctrl = run_mpc(name, half_width=half_width)
    d = dist_to_goal(log, path)
    assert d[-1] < 0.01, f'{name}: final error {d[-1] * 1000:.1f} mm'
    lim = mpc_params['limits']
    check_rate(log['cmd'], 1 / mpc_params['rate_hz'], lim['max_tilt'],
               lim['max_rate'])


HALF_WIDTH_L = 0.005   # LQR (lqr.yaml) leaves this corridor at the L's corner (by ~4 mm)


def test_mpc_stays_in_corridor_on_L(run_mpc, lqr_params, sim_params):
    path, p0, T = SCENARIOS['L']
    _, corr, _ = corridors_for(path, HALF_WIDTH_L)
    lqr = LqrPathController.from_params(lqr_params)
    lqr_log = run_closed_loop(lqr, BallSim.from_params(sim_params, pos0=p0),
                              path, T)
    lqr_v = corridor_violation(lqr_log['pos'], corr).max()
    log, _, _ = run_mpc('L', half_width=HALF_WIDTH_L)
    mpc_v = corridor_violation(log['pos'], corr).max()
    assert lqr_v > 0.003, f'precondition: LQR leaves by {lqr_v * 1000:.1f} mm'
    assert mpc_v < 0.003, f'MPC leaves the corridor by {mpc_v * 1000:.1f} mm'


def test_mpc_with_noise_and_delay(run_mpc):
    log, path, ctrl = run_mpc('L', half_width=0.02, noise_pos=3e-4, delay=0.08)
    assert dist_to_goal(log, path)[-1] < 0.01
    assert np.median(ctrl.law.solve_times) < 0.008


# ------------------------------------------------------------ review fixes
def test_unconverged_solve_is_not_ok(lqr_params):
    """An OSQP exit at max_iter is a failed solve, even when its U happens to
    lie inside the limits."""
    mpc = make_cftoc(lqr_params, osqp_opts={'max_iter': 5, 'eps_abs': 1e-5,
                                            'eps_rel': 1e-5, 'polish': True})
    # near the reference: 5 iterations leave U inside the limits, X0 off by ~1.5 mm
    plan = mpc.solve(state(px=0.002), ref_const(mpc.N), np.zeros(2))
    assert not plan.ok


def test_resting_ball_just_outside_corridor_is_controlled(mpc_params, sim_params):
    """Noise-free ball at rest 0.5 mm outside its corridor: slow slack-active
    solves must not keep the board level (no stored plan to fall back on)."""
    path, _, _ = SCENARIOS['L']
    arc, corr, pose_corr = corridors_for(path, 0.02)
    p0 = (corr[0, 1] + 0.0005, 0.15)     # every solve here took > 12 ms (review I1)
    ctrl = MpcPathController.from_params(mpc_params)
    ctrl.on_route(path, arc, corr, pose_corr, plan_id=1)
    log = run_closed_loop(ctrl, BallSim.from_params(sim_params, pos0=p0), path, 2.0)
    assert np.abs(log['cmd']).max() > 0.01
    assert np.linalg.norm(log['pos'][-1] - np.array(p0)) > 0.02


# --------------------------------------------------- corner speed profile
def corner_params(mpc_params, v_ref=0.15, v_turn=0.06, a_ref=0.25, window=0.02):
    p = copy.deepcopy(mpc_params)
    p['mpc'].update(v_ref=v_ref, v_turn=v_turn, a_ref=a_ref, corner_window=window)
    return p


def profile_on(params, path):
    ctrl = MpcPathController.from_params(params)
    ctrl.on_path(np.asarray(path, float))
    return ctrl, ctrl.speed_profile


def test_profile_off_is_constant_v_ref(mpc_params):
    p = corner_params(mpc_params, v_ref=0.1, v_turn=0.1)
    _, (s, v) = profile_on(p, SCENARIOS['L'][0])
    assert np.all(v == 0.1)


def test_profile_straight_runs_at_v_ref(mpc_params):
    _, (s, v) = profile_on(corner_params(mpc_params), SCENARIOS['line'][0])
    assert v == pytest.approx(0.15)


def test_profile_slows_to_v_turn_at_corner(mpc_params):
    v_turn, a = 0.06, 0.25
    _, (s, v) = profile_on(corner_params(mpc_params, v_turn=v_turn, a_ref=a),
                           SCENARIOS['L'][0])
    corner = 0.30
    assert np.interp(corner, s, v) == pytest.approx(v_turn, abs=1e-3)
    bound = np.sqrt(v_turn ** 2 + 2 * a * np.maximum(np.abs(s - corner) - 0.02, 0.0))
    assert np.all(v <= bound + 1e-6)          # brakes before, speeds up after, at a_ref
    assert np.interp(corner - 0.15, s, v) == pytest.approx(0.15)
    assert np.interp(corner + 0.15, s, v) == pytest.approx(0.15)


def test_profile_u_turn_bounded_by_both_corners(mpc_params):
    v_turn, a = 0.06, 0.05
    _, (s, v) = profile_on(corner_params(mpc_params, v_turn=v_turn, a_ref=a),
                           SCENARIOS['U'][0])
    c1, c2 = 0.33, 0.48
    mid = (c1 + c2) / 2
    w = 0.02                                  # v_turn holds within w of a corner
    expect = np.sqrt(v_turn ** 2 + 2 * a * (mid - c1 - w))
    assert np.interp(mid, s, v) == pytest.approx(expect, abs=2e-3)
    assert np.all(np.abs(np.diff(v ** 2)) <= 2 * a * np.diff(s) + 1e-9)


def test_profile_staircase_is_not_a_corner(mpc_params):
    # 8-connected 5 mm cells along a 30 deg line, as A* leaves them
    d = np.array([np.cos(np.pi / 6), np.sin(np.pi / 6)])
    pts = [np.zeros(2)]
    for _ in range(60):
        steps = [np.array([0.005, 0.0]), np.array([0.005, 0.005])]
        nxt = [pts[-1] + st for st in steps]
        off = [abs(q[0] * d[1] - q[1] * d[0]) for q in nxt]
        pts.append(nxt[int(np.argmin(off))])
    _, (s, v) = profile_on(corner_params(mpc_params), np.array(pts))
    assert v.min() >= 0.85 * 0.15


def test_horizon_velocity_follows_profile(mpc_params):
    p = corner_params(mpc_params)
    ctrl, path = started(p, 'L')
    ctrl.tracker.progress = 0.25              # 5 cm before the corner
    ctrl.on_state(0.1, (-0.18, -0.10), (0.0, 0.0))
    ctrl.update(0.1)
    ref = ctrl.horizon_ref()
    speed = np.linalg.norm(ref.V, axis=0)
    expect = [ctrl._speed(s) for s in arc_of(ctrl, ref.R)]
    assert speed == pytest.approx(expect, abs=1e-6)
    assert speed.min() < 0.1                  # the horizon reaches the corner


def test_profile_degenerate_paths(mpc_params):
    p = corner_params(mpc_params)
    for path in (SCENARIOS['point'][0], [[0.0, 0.0], [0.01, 0.0]],
                 [[0.0, 0.0], [0.01, 0.0], [0.01, 0.01]]):
        ctrl, prof = profile_on(p, path)
        if prof is not None:
            assert np.all(np.isfinite(prof[1])) and np.all(prof[1] > 0)
        assert np.isfinite(ctrl._speed(0.0))
    ctrl, _ = profile_on(p, np.zeros((0, 2)))
    assert ctrl.speed_profile is None


def test_profile_chamfered_corner_slows_to_v_turn(mpc_params):
    # a 90 deg bend straightened into two 45 deg bends joined by a 1 cm diagonal
    d = 0.01 / np.sqrt(2)
    path = [[-0.18, 0.15], [-0.18, -0.15 + d], [-0.18 + d, -0.15], [0.18, -0.15]]
    _, (s, v) = profile_on(corner_params(mpc_params, v_turn=0.06), path)
    assert v.min() == pytest.approx(0.06, abs=1e-3)


def test_profile_build_is_fast(mpc_params):
    # a 4 m route of 5 mm points with a bend every 10 cm, as the planner sends
    legs = [np.array([0.1, 0.0]), np.array([0.0, 0.1])]
    corners = np.cumsum([np.zeros(2)] + [legs[i % 2] for i in range(40)], axis=0)
    pts = np.vstack([np.linspace(a, b, 20, endpoint=False)
                     for a, b in zip(corners[:-1], corners[1:])] + [corners[-1:]])
    ctrl = MpcPathController.from_params(corner_params(mpc_params))
    t0 = time.perf_counter()
    ctrl.on_path(pts)
    assert time.perf_counter() - t0 < 0.05
    assert ctrl.speed_profile[1].min() == pytest.approx(0.06, abs=1e-3)


@pytest.mark.parametrize('key, value', [('v_turn', 0.0), ('v_turn', -0.05),
                                        ('a_ref', 0.0), ('a_ref', -1.0),
                                        ('corner_window', 0.0), ('corner_window', -0.01)])
def test_profile_rejects_nonpositive_settings(mpc_params, key, value):
    # v_turn 0 would stall the reference at a corner for good; a_ref and
    # corner_window <= 0 would silently give a wrong profile
    p = copy.deepcopy(mpc_params)
    p['mpc'][key] = value
    with pytest.raises(ValueError, match=key):
        MpcPathController.from_params(p)
