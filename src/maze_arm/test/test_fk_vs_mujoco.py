"""fk_board against the MuJoCo scene at random arm poses (the sim has the UR7e's link lengths
since #25). The board pose is where the grasp weld holds it: wrist_3_link * the weld's relpose."""
from pathlib import Path

import mujoco
import numpy as np
import pytest

from maze_arm.kinematics import UR_JOINTS, fk_board
from maze_sim_description.config import load_sim_config
from maze_sim_description.scene_builder import build_scene

POSES = 15
SIM_PKG = Path(__file__).resolve().parents[2] / "maze_sim_description"


@pytest.fixture(scope="module")
def model():
    cfg = load_sim_config(SIM_PKG / "config" / "sim.yaml")
    return mujoco.MjModel.from_xml_string(build_scene(cfg, SIM_PKG / "meshes" / "board"))


def _pose(pos, mat):
    T = np.eye(4)
    T[:3, :3] = np.asarray(mat).reshape(3, 3)
    T[:3, 3] = pos
    return T


def mujoco_board_pose(model, data):
    eq = model.eq("grasp_weld")
    R = np.empty(9)
    mujoco.mju_quat2Mat(R, eq.data[6:10])
    rel = _pose(eq.data[3:6], R)            # pose of obj2 in obj1's frame
    w3 = model.body("wrist_3_link").id
    T_w3 = _pose(data.xpos[w3], data.xmat[w3])
    return T_w3 @ rel if eq.obj1id[0] == w3 else T_w3 @ np.linalg.inv(rel)


def test_fk_board_matches_mujoco_at_random_poses(model, initial_joints, grasp, handle_center):
    data = mujoco.MjData(model)
    qadr = [model.joint(j).qposadr[0] for j in UR_JOINTS]
    rng = np.random.default_rng(0)
    for _ in range(POSES):
        q = np.asarray(initial_joints) + rng.uniform(-1.0, 1.0, 6)
        data.qpos[qadr] = q
        mujoco.mj_kinematics(model, data)
        T_sim = mujoco_board_pose(model, data)
        T = fk_board(q, *grasp, handle_center)
        np.testing.assert_allclose(T[:3, 3], T_sim[:3, 3], atol=1e-4)   # 0.1 mm
        np.testing.assert_allclose(T[:3, :3], T_sim[:3, :3], atol=1e-6)
