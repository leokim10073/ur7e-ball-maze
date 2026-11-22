"""MPC controller: the CFTOC (limits, corridors, agreement with the LQR,
solve time), offline and without ROS."""

import time

import numpy as np
import pytest

from maze_control.lqr.lqr_law import ball_model, c2d, design_gain
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
