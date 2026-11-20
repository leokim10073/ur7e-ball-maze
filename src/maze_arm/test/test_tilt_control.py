import numpy as np
import pytest

from maze_arm.kinematics import board_jacobian, board_offset, fk_board
from maze_arm.tilt_control import (TiltController, level_home, load_arm_config, rotation_log,
                                   tilt_from_rotation, tilt_rotation)

DT = 0.01   # controller step, s


@pytest.fixture
def offset(grasp, handle_center):
    return board_offset(*grasp, handle_center)


@pytest.fixture
def ctrl(arm_cfg, offset, initial_joints, grasp, handle_center):
    home = level_home(fk_board(initial_joints, *grasp, handle_center))
    c = TiltController(arm_cfg, offset, home)
    c.reset(initial_joints)
    return c


def simulate(ctrl, q0, cmd, seconds):
    """Kinematic loop: the arm follows the commanded joint velocities exactly."""
    q = np.array(q0, float)
    centers, tilts, speeds = [], [], []
    for k in range(int(round(seconds / DT))):
        qdot = ctrl.step(q, cmd(k * DT) if callable(cmd) else cmd, DT)
        q = q + qdot * DT
        centers.append(ctrl.board_pose(q)[:3, 3])
        tilts.append(ctrl.measured_tilt(q))
        speeds.append(qdot)
    return q, np.array(centers), np.array(tilts), np.array(speeds)


# --- tilt convention (Contract 4) ---

def test_tilt_x_lowers_plus_x_edge():
    R = tilt_rotation(0.05, 0.0)
    assert (R @ [0.1, 0, 0])[2] < 0
    assert abs((R @ [0, 0.1, 0])[2]) < 1e-12


def test_tilt_y_lowers_plus_y_edge():
    R = tilt_rotation(0.0, 0.05)
    assert (R @ [0, 0.1, 0])[2] < 0
    assert abs((R @ [0.1, 0, 0])[2]) < 1e-12


def test_tilt_round_trip():
    for tx, ty in [(0.05, 0.0), (0.0, -0.03), (0.06, 0.04), (-0.087, 0.087)]:
        np.testing.assert_allclose(tilt_from_rotation(tilt_rotation(tx, ty)), [tx, ty], atol=1e-12)


def test_rotation_log_small_and_large():
    np.testing.assert_allclose(rotation_log(np.eye(3)), 0, atol=1e-12)
    R = tilt_rotation(0.0, -0.7)   # +0.7 rad about x
    np.testing.assert_allclose(rotation_log(R), [0.7, 0, 0], atol=1e-12)


# --- kinematics ---

def test_jacobian_matches_finite_differences(initial_joints, offset):
    from maze_arm.kinematics import fk_tool0

    q = np.array(initial_joints)
    J = board_jacobian(q, offset)
    T0 = fk_tool0(q) @ offset
    eps = 1e-6
    for i in range(6):
        dq = np.zeros(6)
        dq[i] = eps
        T1 = fk_tool0(q + dq) @ offset
        np.testing.assert_allclose((T1[:3, 3] - T0[:3, 3]) / eps, J[:3, i], atol=1e-5)
        np.testing.assert_allclose(rotation_log(T1[:3, :3] @ T0[:3, :3].T) / eps, J[3:, i], atol=1e-5)


def test_home_is_level_at_start(ctrl, initial_joints):
    np.testing.assert_allclose(ctrl.T_home[:3, 2], [0, 0, 1])
    np.testing.assert_allclose(ctrl.measured_tilt(initial_joints), 0, atol=1e-3)


# --- closed loop (kinematic) ---

@pytest.mark.parametrize("cmd", [(0.05, 0.0), (0.0, 0.05), (0.0, -0.05), (0.04, -0.03)])
def test_step_reaches_tilt_and_center_stays(ctrl, initial_joints, arm_cfg, cmd):
    _, centers, tilts, speeds = simulate(ctrl, initial_joints, cmd, 1.0)
    np.testing.assert_allclose(tilts[-1], cmd, atol=1e-3)
    drift_mm = np.max(np.linalg.norm(centers - ctrl.T_home[:3, 3], axis=1)) * 1000
    assert drift_mm < 1.0, f"board center moved {drift_mm:.2f} mm"
    assert np.max(np.abs(speeds)) <= arm_cfg.max_joint_speed + 1e-9


def test_tilt_rate_limited(ctrl, initial_joints, arm_cfg):
    _, _, tilts, _ = simulate(ctrl, initial_joints, (0.087, 0.0), 0.5)
    rate = np.max(np.linalg.norm(np.diff(tilts, axis=0), axis=1)) / DT
    assert rate <= arm_cfg.max_tilt_rate * 1.05


def test_tilt_clamped(ctrl, initial_joints, arm_cfg):
    _, _, tilts, _ = simulate(ctrl, initial_joints, (0.3, -0.3), 1.5)
    np.testing.assert_allclose(tilts[-1], [arm_cfg.max_tilt, -arm_cfg.max_tilt], atol=1e-3)


def test_sine_tracking(ctrl, initial_joints):
    amp, f = 0.04, 1.0   # rad, Hz: peak rate 0.25 rad/s, under the rate limit
    cmd = lambda t: (amp * np.sin(2 * np.pi * f * t), 0.0)  # noqa: E731
    _, centers, tilts, _ = simulate(ctrl, initial_joints, cmd, 3.0)
    t = np.arange(1, len(tilts) + 1) * DT
    late = t > 1.0
    err = np.abs(tilts[late, 0] - amp * np.sin(2 * np.pi * f * t[late]))
    assert np.max(err) < 0.005
    assert np.max(np.linalg.norm(centers - ctrl.T_home[:3, 3], axis=1)) < 0.001


def test_tilt_y_is_mostly_wrist_3(ctrl, initial_joints):
    """Tilt about board x is wrist_3's own axis (it passes through the board center)."""
    _, _, _, speeds = simulate(ctrl, initial_joints, (0.0, 0.05), 0.5)
    peak = np.max(np.abs(speeds), axis=0)
    assert np.all(peak[:5] < 0.05 * peak[5])


def test_sag_is_pulled_back(ctrl, initial_joints):
    """With zero commands the sim's arm sags; the loop must bring the center back.
    q: /joint_states read from the sim after sagging (wrist_3 put back at its start value)."""
    q = [-2.74984, -1.68144, -2.25004, 0.77801, 1.18374, initial_joints[5]]
    sag_mm = np.linalg.norm(ctrl.board_pose(q)[:3, 3] - ctrl.T_home[:3, 3]) * 1000
    assert sag_mm > 3, f"sag only {sag_mm:.1f} mm"
    _, centers, tilts, _ = simulate(ctrl, q, (0.0, 0.0), 1.0)
    assert np.linalg.norm(centers[-1] - ctrl.T_home[:3, 3]) * 1000 < 0.1
    np.testing.assert_allclose(tilts[-1], 0, atol=1e-3)


# --- config ---

def test_config_rejects_missing_key(tmp_path):
    p = tmp_path / "arm.yaml"
    p.write_text("max_tilt: 0.087\n")
    with pytest.raises(ValueError, match="missing key"):
        load_arm_config(p)


def test_config_rejects_non_positive(tmp_path, arm_cfg):
    from dataclasses import asdict

    values = asdict(arm_cfg) | {"damping": 0.0}
    p = tmp_path / "arm.yaml"
    p.write_text("\n".join(f"{k}: {v}" for k, v in values.items()))
    with pytest.raises(ValueError, match="damping"):
        load_arm_config(p)
