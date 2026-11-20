"""The sim arm has the real robot's link lengths: at random joint angles, MuJoCo's tool0 matches
the UR's DH forward kinematics, with the DH parameters from ur_description's
default_kinematics.yaml for robot.ur_type. Menagerie's UR5e rounds them (0.163 vs 0.1625 m, ...),
which put tool0 up to 1.5 mm off."""

import mujoco
import numpy as np
import pytest

from maze_sim_description.scene_builder import UR_JOINTS, ur_dh_parameters

POSES = 15


def _dh(a, alpha, d, theta):
    ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
    return np.array([[ct, -st * ca, st * sa, a * ct],
                     [st, ct * ca, -ct * sa, a * st],
                     [0.0, sa, ca, d],
                     [0.0, 0.0, 0.0, 1.0]])


def _fk_tool0(q, dh):
    """world (= ROS base_link) -> tool0. base_link is the DH base rotated by pi about z."""
    d = [dh["d1"], 0.0, 0.0, dh["d4"], dh["d5"], dh["d6"]]
    a = [0.0, -dh["a2"], -dh["a3"], 0.0, 0.0, 0.0]
    alpha = [np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0]
    T = np.diag([-1.0, -1.0, 1.0, 1.0])
    for i in range(6):
        T = T @ _dh(a[i], alpha[i], d[i], q[i])
    return T


def test_ur7e_dh_parameters(cfg):
    dh = ur_dh_parameters(cfg.robot.ur_type)
    assert cfg.robot.ur_type == "ur7e"
    assert dh == pytest.approx(dict(d1=0.1625, a2=0.425, a3=0.3922, d4=0.1333, d5=0.0997, d6=0.0996))


def test_tool0_matches_dh_at_random_poses(model, cfg):
    dh = ur_dh_parameters(cfg.robot.ur_type)
    data = mujoco.MjData(model)
    qadr = [model.joint(j).qposadr[0] for j in UR_JOINTS]
    rng = np.random.default_rng(0)
    for _ in range(POSES):
        q = np.asarray(cfg.robot.initial_joints) + rng.uniform(-1.0, 1.0, 6)
        data.qpos[qadr] = q
        mujoco.mj_kinematics(model, data)
        T = _fk_tool0(q, dh)
        np.testing.assert_allclose(data.site("tool0").xpos, T[:3, 3], atol=1e-6)
        np.testing.assert_allclose(data.site("tool0").xmat.reshape(3, 3), T[:3, :3], atol=1e-9)
