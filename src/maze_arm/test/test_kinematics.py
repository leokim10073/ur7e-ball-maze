import numpy as np
import pytest

from maze_arm.kinematics import UR_JOINTS, fk_board, joints_from_state




def test_board_is_level_at_start(initial_joints, grasp, handle_center):
    T = fk_board(initial_joints, *grasp, handle_center)
    np.testing.assert_allclose(T[:3, 2], [0.0, 0.0, 1.0], atol=1e-3)




def test_joints_from_state_any_order(initial_joints, grasp, handle_center):
    # the real UR driver lists shoulder_pan_joint last
    order = [1, 2, 3, 4, 5, 0]
    names = [UR_JOINTS[i] for i in order]
    positions = [initial_joints[i] for i in order]
    q = joints_from_state(names, positions)
    np.testing.assert_allclose(fk_board(q, *grasp, handle_center),
                               fk_board(initial_joints, *grasp, handle_center))


def test_joints_from_state_missing_joint():
    with pytest.raises(ValueError, match="wrist_3_joint"):
        joints_from_state(list(UR_JOINTS[:5]), [0.0] * 5)