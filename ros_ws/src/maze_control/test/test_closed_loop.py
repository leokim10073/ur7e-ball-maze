"""Closed-loop tests 1-3: point, straight line, L and U paths (no ROS)."""

import copy

import numpy as np
import pytest


def dist_to_goal(log, path):
    return np.linalg.norm(log['pos'] - path[-1], axis=1)


# ---------------------------------------------------------------- 1. point
def test_single_point_settles_on_target(run):
    log, path = run('point')
    d = dist_to_goal(log, path)
    assert d[-1] < 0.002, f'final error {d[-1] * 1000:.2f} mm'
    assert np.linalg.norm(log['vel'][-1]) < 0.005
    # Monotone-ish approach in the first second: confirms the sign convention.
    assert d[30] < d[0], 'ball moved away from the target: sign error?'


@pytest.mark.parametrize('axis', ['theta_y_to_ax', 'theta_x_to_ay'])
def test_wrong_sign_is_detected(run, pid_params, axis):
    """Flipping one controller sign must make the point test fail."""
    bad = copy.deepcopy(pid_params)
    bad['sign'][axis] *= -1
    log, path = run('point', ctrl_params=bad)
    assert dist_to_goal(log, path)[-1] > 0.05


# ----------------------------------------------------------- 2. straight line
def test_straight_line_reaches_end(run):
    log, path = run('line')
    d = dist_to_goal(log, path)
    assert (d < 0.02).any(), 'never got within goal_tol'
    assert d[-1] < 0.002
    assert log['cross_track'].max() < 0.015


# ------------------------------------------------------------- 3. L and U
@pytest.mark.parametrize('name', ['L', 'U'])
def test_turns_reach_end(run, name, record_property):
    log, path = run(name)
    d = dist_to_goal(log, path)
    max_xt = log['cross_track'].max()
    t_reach = log['t'][np.argmax(d < 0.02)] if (d < 0.02).any() else None
    record_property('max_cross_track_m', max_xt)
    print(f'\n[{name}] max cross-track error = {max_xt * 1000:.1f} mm, '
          f'goal reached at t = {t_reach} s, final error = {d[-1] * 1000:.2f} mm')
    assert t_reach is not None
    assert d[-1] < 0.002
    assert max_xt < 0.015


@pytest.mark.parametrize('name', ['line', 'L', 'U'])
def test_tolerates_noise_and_extra_lag(run, name):
    """Robustness: KF-like noise plus a slower arm with more delay."""
    log, path = run(name, tau=0.12, delay=0.1, noise_pos=0.001,
                    noise_vel=0.01)
    assert dist_to_goal(log, path)[-1] < 0.01
    assert log['cross_track'].max() < 0.025
