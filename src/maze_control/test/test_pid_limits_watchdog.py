"""PID tests 4-5: tilt / rate limits and tracking-loss watchdog (no ROS)."""

import numpy as np
import pytest

from maze_control.pid.controller import PidPathController

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
                 lim['max_tilt'], lim['max_rate'])
    # The tight cases actually hit the tilt limit, so it is exercised.
    if name == 'point':
        assert np.isclose(np.abs(log['cmd']).max(), lim['max_tilt'])


def test_limits_under_random_states(pid_params):
    """Fuzz: wild states, jumping positions, random paths and dropouts."""
    rng = np.random.default_rng(1)
    ctrl = PidPathController.from_params(pid_params)
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
    check_limits(cmds, dt, lim['max_tilt'], lim['max_rate'])


# ------------------------------------------------------- 5. tracking loss
def test_tracking_loss_returns_to_zero(run, pid_params):
    lost_at = 0.8                           # ball still moving fast here
    log, _ = run('point', state_until=lost_at)
    t, cmd = log['t'], log['cmd']
    timeout = pid_params['watchdog']['state_timeout']
    lim = pid_params['limits']
    ramp = lim['max_tilt'] / lim['max_rate']   # worst-case ramp time

    before = (t > lost_at - 0.2) & (t <= lost_at + timeout)
    assert np.abs(cmd[before]).max() > np.radians(0.5), 'should still be tilting'
    assert all(s != 'timeout' for s in log['status'][t <= lost_at + timeout])
    assert all(s == 'timeout' for s in log['status'][t > lost_at + timeout + 1e-9])

    zero = t > lost_at + timeout + ramp + 2.0 / pid_params['rate_hz']
    assert np.all(cmd[zero] == 0.0), 'tilt must be exactly 0 after the ramp'


def test_recovers_when_states_resume(pid_params):
    ctrl = PidPathController.from_params(pid_params)
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


def test_republished_path_keeps_progress(pid_params):
    ctrl = PidPathController.from_params(pid_params)
    path = [[0.0, 0.0], [0.3, 0.0]]
    assert ctrl.on_path(path)
    ctrl.on_state(0.0, (0.1, 0.0), (0.0, 0.0))
    ctrl.update(0.0)
    s = ctrl.tracker.progress
    assert s == pytest.approx(0.1)
    assert not ctrl.on_path(list(path))        # same path again: ignored
    assert ctrl.tracker.progress == s
    assert ctrl.on_path([[0.0, 0.0], [0.0, 0.3]])   # new path: restart
    assert ctrl.tracker.progress == 0.0


def test_no_path_outputs_zero(pid_params):
    ctrl = PidPathController.from_params(pid_params)
    ctrl.on_state(0.0, (0.1, 0.1), (0.0, 0.0))
    assert np.all(ctrl.update(0.0) == 0.0)
    assert ctrl.status == 'no_path'
