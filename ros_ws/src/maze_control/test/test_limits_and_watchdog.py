"""Tests 4-5: tilt / rate limits and tracking-loss watchdog (no ROS)."""

import math

import numpy as np
import pytest

from maze_control.controller import MazeController
from maze_control.path_tracker import PathTracker

DEG = math.pi / 180.0
EPS = 1e-9


def check_limits(cmd, dt, max_tilt, max_rate):
    cmd = np.asarray(cmd)
    assert np.abs(cmd).max() <= max_tilt + EPS
    rate = np.abs(np.diff(np.vstack([np.zeros(2), cmd]), axis=0)) / dt
    assert rate.max() <= max_rate + 1e-6


# ---------------------------------------------------------------- 4. limits
@pytest.mark.parametrize('name', ['point', 'line', 'L', 'U'])
def test_limits_in_closed_loop(run, pid_params, name):
    log, _ = run(name)
    lim = pid_params['limits']
    check_limits(log['cmd'], 1.0 / pid_params['rate_hz'],
                 lim['max_tilt_deg'] * DEG, lim['max_rate_deg_s'] * DEG)
    # The tight cases actually hit the tilt limit, so it is exercised.
    if name == 'point':
        assert np.isclose(np.abs(log['cmd']).max(), lim['max_tilt_deg'] * DEG)


def test_limits_under_random_states(pid_params):
    """Fuzz: wild states, jumping positions, random paths and dropouts."""
    rng = np.random.default_rng(1)
    ctrl = MazeController.from_params(pid_params)
    dt = 1.0 / pid_params['rate_hz']
    lim = pid_params['limits']
    cmds = []
    for k in range(6000):
        t = k * dt
        if k % 500 == 0:
            ctrl.on_path(rng.uniform(-0.25, 0.25, (rng.integers(1, 6), 2)))
        if rng.random() > 0.05:     # occasional missing states
            ctrl.on_state(t, rng.uniform(-0.3, 0.3, 2), rng.uniform(-2, 2, 2))
        cmds.append(ctrl.update(t))
    check_limits(cmds, dt, lim['max_tilt_deg'] * DEG,
                 lim['max_rate_deg_s'] * DEG)


# ------------------------------------------------------- 5. tracking loss
def test_tracking_loss_returns_to_zero(run, pid_params):
    lost_at = 0.8                           # ball still moving fast here
    log, _ = run('point', state_until=lost_at)
    t, cmd = log['t'], log['cmd']
    timeout = pid_params['watchdog']['state_timeout']
    lim = pid_params['limits']
    ramp = lim['max_tilt_deg'] / lim['max_rate_deg_s']   # worst-case ramp time

    before = (t > lost_at - 0.2) & (t <= lost_at + timeout)
    assert np.abs(cmd[before]).max() > 0.5 * DEG, 'should still be tilting'
    assert all(s != 'timeout' for s in log['status'][t <= lost_at + timeout])
    assert all(s == 'timeout' for s in log['status'][t > lost_at + timeout + 1e-9])

    zero = t > lost_at + timeout + ramp + 2.0 / pid_params['rate_hz']
    assert np.all(cmd[zero] == 0.0), 'tilt must be exactly 0 after the ramp'


def test_recovers_when_states_resume(pid_params):
    ctrl = MazeController.from_params(pid_params)
    ctrl.on_path([[0.1, 0.0]])
    dt = 1.0 / pid_params['rate_hz']
    ctrl.on_state(0.0, (0.0, 0.0), (0.0, 0.0))
    for k in range(60):                     # 1 s without new states
        ctrl.update(k * dt)
    assert ctrl.status == 'timeout'
    assert np.allclose(ctrl.update(60 * dt), 0.0)
    ctrl.on_state(61 * dt, (0.0, 0.0), (0.0, 0.0))
    ctrl.update(61 * dt)
    assert ctrl.status == 'tracking'


def test_no_path_outputs_zero(pid_params):
    ctrl = MazeController.from_params(pid_params)
    ctrl.on_state(0.0, (0.1, 0.1), (0.0, 0.0))
    assert np.all(ctrl.update(0.0) == 0.0)
    assert ctrl.status == 'no_path'


# ----------------------------------------------------------- path tracker
def test_tracker_lookahead_and_goal_hold():
    trk = PathTracker(lookahead=0.05, goal_tol=0.02)
    trk.set_path([[0.0, 0.0], [0.3, 0.0]])
    assert np.allclose(trk.update((0.10, 0.01)), (0.15, 0.0))
    assert np.allclose(trk.update((0.27, 0.0)), (0.30, 0.0))   # clamped at end
    assert np.allclose(trk.update((0.29, 0.0)), (0.30, 0.0))
    assert trk.reached


def test_tracker_does_not_jump_across_u_turn():
    trk = PathTracker(lookahead=0.05, search_ahead=0.15)
    trk.set_path([[0.0, 0.2], [0.0, 0.0], [0.05, 0.0], [0.05, 0.2]])
    # Ball near the top of the left leg, i.e. right next to the end of the
    # right leg: the target must stay on the left leg.
    tgt = trk.update((0.03, 0.19))
    assert tgt[0] == pytest.approx(0.0)
    assert trk.progress < 0.1
