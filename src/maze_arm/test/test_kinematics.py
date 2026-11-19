import numpy as np

from maze_arm.kinematics import fk_board

# From sim.yaml and maze_CAD/levels/L1/sim/parts_report.json
INITIAL_JOINTS = [-2.7498, -1.6799, -2.2485, 0.7868, 1.1837, -1.5708]
GRASP_XYZ = [0.0, 0.0, 0.15]
GRASP_RPY = [3.141593, -1.570796, 0.0]
HANDLE_CENTER = np.array([-0.175, 0.0, 0.0])


def test_board_is_level_at_start():
    T = fk_board(INITIAL_JOINTS, GRASP_XYZ, GRASP_RPY, HANDLE_CENTER)
    np.testing.assert_allclose(T[:3, 2], [0.0, 0.0, 1.0], atol=1e-3)


def test_handle_where_sim_yaml_says():
    # sim.yaml: initial_joints hold the handle at (0.60, 0, 0.41)
    T = fk_board(INITIAL_JOINTS, GRASP_XYZ, GRASP_RPY, HANDLE_CENTER)
    handle = T[:3, 3] + T[:3, :3] @ HANDLE_CENTER
    np.testing.assert_allclose(handle, [0.60, 0.0, 0.41], atol=0.005)
