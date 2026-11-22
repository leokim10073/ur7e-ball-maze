"""MPC controller: the CFTOC (limits, corridors, agreement with the LQR,
solve time), offline and without ROS."""

import time

import numpy as np
import pytest

from maze_control.lqr.lqr_law import ball_model, c2d, design_gain
from maze_control.mpc.mpc_law import CFTOC, HorizonRef

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
