import numpy as np

from maze_arm.kinematics import fk_board

# sim.yaml: initial_joints hold the board level with its handle at (0.60, 0, 0.41)
HANDLE_AT_START = [0.60, 0.0, 0.41]


def test_board_is_level_at_start(initial_joints, grasp, handle_center):
    T = fk_board(initial_joints, *grasp, handle_center)
    np.testing.assert_allclose(T[:3, 2], [0.0, 0.0, 1.0], atol=1e-3)


def test_handle_where_sim_yaml_says(initial_joints, grasp, handle_center):
    T = fk_board(initial_joints, *grasp, handle_center)
    handle = T[:3, 3] + T[:3, :3] @ handle_center
    np.testing.assert_allclose(handle, HANDLE_AT_START, atol=0.005)
