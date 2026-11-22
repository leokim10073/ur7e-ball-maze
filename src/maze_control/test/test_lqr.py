"""LQR controller: design math, closed loop on the offline sim, limits, delay
predictor and watchdog (no ROS)."""

import copy

import numpy as np
import pytest

from maze_control.lqr.controller import LqrPathController
from maze_control.lqr.lqr_law import ball_model, c2d, design_gain, dlqr, expm

EPS = 1e-9


def dist_to_goal(log, path):
    return np.linalg.norm(log['pos'] - path[-1], axis=1)


def check_limits(cmd, dt, max_tilt, max_rate):
    cmd = np.asarray(cmd)
    assert np.abs(cmd).max() <= max_tilt + EPS
    rate = np.abs(np.diff(np.vstack([np.zeros(2), cmd]), axis=0)) / dt
    assert rate.max() <= max_rate + 1e-6


def gain(p):
    m, w = p['model'], p['lqr']
    return design_gain(m['c'], m['tau'], 1.0 / p['rate_hz'], w['q_pos'],
                       w['q_vel'], w['q_tilt'], w['r'], w['q_int'], g=m['g'])


# ------------------------------------------------------------ design math
def test_expm_matches_closed_forms():
    assert np.allclose(expm([[0.7]]), [[np.exp(0.7)]])
    w = 2.0
    R = expm([[0.0, -w], [w, 0.0]])
    assert np.allclose(R, [[np.cos(w), -np.sin(w)], [np.sin(w), np.cos(w)]])


def test_zoh_of_the_lag_is_exact():
    A, B = ball_model(c=0.3, tau=0.08)
    Ad, Bd = c2d(A, B, 1 / 60)
    a = np.exp(-(1 / 60) / 0.08)
    assert Ad[2, 2] == pytest.approx(a)
    assert Bd[2, 0] == pytest.approx(1 - a)


def test_riccati_solution_and_stable_closed_loop(lqr_params):
    m, w = lqr_params['model'], lqr_params['lqr']
    A, B = ball_model(m['c'], m['tau'], m['g'])
    Ad, Bd = c2d(A, B, 1 / lqr_params['rate_hz'])
    Q = np.diag([w['q_pos'], w['q_vel'], w['q_tilt']])
    R = np.array([[w['r']]])
    K, P = dlqr(Ad, Bd, Q, R)
    # P solves the discrete algebraic Riccati equation
    rhs = Q + Ad.T @ P @ Ad - Ad.T @ P @ Bd @ np.linalg.solve(
        R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
    assert np.allclose(P, rhs, rtol=1e-8, atol=1e-6)
    assert np.all(np.abs(np.linalg.eigvals(Ad - Bd @ K)) < 1.0)


def test_gain_signs(lqr_params):
    """u = -K x: ball ahead of the target (e > 0) or moving +x tilts towards
    -x, so the position and velocity gains are positive."""
    K, _, _ = gain(lqr_params)
    assert K[0] > 0 and K[1] > 0


def test_integral_adds_a_fourth_gain(lqr_params):
    p = copy.deepcopy(lqr_params)
    p['lqr']['q_int'] = 2000.0
    K, _, _ = gain(p)
    assert len(K) == 4 and K[3] > 0


# ------------------------------------------------------------ closed loop
def test_single_point_settles_on_target(run_lqr):
    log, path = run_lqr('point')
    d = dist_to_goal(log, path)
    assert d[-1] < 0.002, f'final error {d[-1] * 1000:.2f} mm'
    assert np.linalg.norm(log['vel'][-1]) < 0.005
    assert d[30] < d[0], 'ball moved away from the target: sign error?'


@pytest.mark.parametrize('key', ['sign_x', 'sign_y'])
def test_wrong_plant_sign_is_detected(run_lqr, key):
    """An arm reading one BoardTilt axis backwards must make the point test fail."""
    log, path = run_lqr('point', **{key: -1.0})
    assert dist_to_goal(log, path)[-1] > 0.05


@pytest.mark.parametrize('name', ['line', 'L', 'U'])
def test_paths_reach_end(run_lqr, name):
    log, path = run_lqr(name)
    d = dist_to_goal(log, path)
    max_xt = log['cross_track'].max()
    t_reach = log['t'][np.argmax(d < 0.02)] if (d < 0.02).any() else None
    print(f'\n[LQR {name}] max cross-track error = {max_xt * 1000:.1f} mm, '
          f'goal reached at t = {t_reach} s, final error = {d[-1] * 1000:.2f} mm')
    assert t_reach is not None
    assert d[-1] < 0.002
    assert max_xt < 0.015


@pytest.mark.parametrize('name', ['line', 'L', 'U'])
def test_tolerates_noise_and_extra_lag(run_lqr, name):
    """Same robustness case as the PID: noise plus a slower arm with more
    delay than the model assumes."""
    log, path = run_lqr(name, tau=0.12, delay=0.1, noise_pos=0.001,
                        noise_vel=0.01)
    assert dist_to_goal(log, path)[-1] < 0.01
    assert log['cross_track'].max() < 0.015


def test_integral_variant_still_tracks(run_lqr, lqr_params):
    p = copy.deepcopy(lqr_params)
    p['lqr']['q_int'] = 2000.0
    log, path = run_lqr('L', ctrl_params=p)
    assert dist_to_goal(log, path)[-1] < 0.002


# ------------------------------------------------------------ limits
@pytest.mark.parametrize('name', ['point', 'line', 'L', 'U'])
def test_limits_in_closed_loop(run_lqr, lqr_params, name):
    log, _ = run_lqr(name)
    lim = lqr_params['limits']
    check_limits(log['cmd'], 1.0 / lqr_params['rate_hz'],
                 lim['max_tilt'], lim['max_rate'])


def test_limits_under_random_states(lqr_params):
    rng = np.random.default_rng(1)
    ctrl = LqrPathController.from_params(lqr_params)
    dt = 1.0 / lqr_params['rate_hz']
    lim = lqr_params['limits']
    cmds = []
    for k in range(6000):
        t = k * dt
        if k % 500 == 0:
            ctrl.on_path(rng.uniform(-0.25, 0.25, (rng.integers(1, 6), 2)))
        if rng.random() > 0.05:
            ctrl.on_state(t, rng.uniform(-0.3, 0.3, 2), rng.uniform(-2, 2, 2))
        cmds.append(ctrl.update(t))
    check_limits(cmds, dt, lim['max_tilt'], lim['max_rate'])


# ------------------------------------------------------------ delay predictor
def test_predictor_handles_a_long_delay(run_lqr, lqr_params):
    """150 ms dead time: with the model's delay matched, the L path is still
    tracked; with the predictor off (model delay 0) the loop goes unstable."""
    on = copy.deepcopy(lqr_params)
    on['model']['delay'] = 0.15
    log, path = run_lqr('L', ctrl_params=on, delay=0.15)
    assert dist_to_goal(log, path)[-1] < 0.002
    assert log['cross_track'].max() < 0.015

    off = copy.deepcopy(lqr_params)
    off['model']['delay'] = 0.0
    log, path = run_lqr('L', ctrl_params=off, delay=0.15)
    assert dist_to_goal(log, path)[-1] > 0.05


def test_observer_follows_the_commands(lqr_params):
    """The estimated tilt is the sent command passed through delay + lag."""
    law = LqrPathController.from_params(lqr_params).law
    dt = 1.0 / lqr_params['rate_hz']
    for _ in range(600):                      # 10 s pushing towards +x, +y
        law.step((0.1, 0.1), (0.0, 0.0), (0.0, 0.0), dt)
    assert np.allclose(law.theta, law.u, atol=1e-6)
    assert np.all(law.u > 0)


# ------------------------------------------------------------ watchdog
def test_tracking_loss_levels_the_board(run_lqr, lqr_params):
    lost_at = 0.8
    log, _ = run_lqr('point', state_until=lost_at)
    t, cmd = log['t'], log['cmd']
    timeout = lqr_params['watchdog']['state_timeout']
    lim = lqr_params['limits']
    ramp = lim['max_tilt'] / lim['max_rate']
    assert all(s != 'timeout' and s != 'stopped'
               for s in log['status'][t <= lost_at + timeout])
    zero = t > lost_at + timeout + ramp + 2.0 / lqr_params['rate_hz']
    assert np.all(cmd[zero] == 0.0), 'tilt must be exactly 0 after the ramp'


def test_no_path_outputs_zero(lqr_params):
    ctrl = LqrPathController.from_params(lqr_params)
    ctrl.on_state(0.0, (0.1, 0.1), (0.0, 0.0))
    assert np.all(ctrl.update(0.0) == 0.0)
    assert ctrl.status == 'no_path'


# ------------------------------------------------------------ lqr_sim.yaml
@pytest.fixture
def lqr_sim_params():
    from maze_control.common.config import load_params
    return load_params('lqr_sim.yaml')


def test_sim_config_same_keys_as_default(lqr_params, lqr_sim_params):
    def keys(d, prefix=''):
        out = set()
        for k, v in d.items():
            out |= keys(v, f'{prefix}{k}.') if isinstance(v, dict) else {prefix + k}
        return out
    assert keys(lqr_sim_params) == keys(lqr_params)


def test_sim_config_tracks_on_a_fast_arm(run_lqr, lqr_sim_params):
    """The MuJoCo arm as identified: ~10 ms lag, ~30 ms dead time, rolling loss ~1/s."""
    log, path = run_lqr('L', ctrl_params=lqr_sim_params, tau=0.01, delay=0.03, c=1.0,
                        noise_pos=3e-4, noise_vel=0.01)
    assert dist_to_goal(log, path)[-1] < 0.005
    assert log['cross_track'].max() < 0.015


def test_sim_config_is_not_for_a_slow_arm(run_lqr, lqr_params, lqr_sim_params):
    """Documents lqr_sim.yaml's limit (see its header): on the offline sim's slower
    arm (80 ms lag, 50 ms delay) a large error (the 24 cm 'point' case) makes it
    diverge, while the default lqr.yaml settles. If this starts passing for
    lqr_sim.yaml, update the warning in the config."""
    log, path = run_lqr('point', ctrl_params=lqr_sim_params)
    assert dist_to_goal(log, path)[-1] > 0.05
    log, path = run_lqr('point', ctrl_params=lqr_params)
    assert dist_to_goal(log, path)[-1] < 0.002
