"""Look-ahead path tracker (shared by all controllers, no ROS)."""

import numpy as np
import pytest

from maze_control.common.path_tracker import PathTracker


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


def test_tangent_at():
    trk = PathTracker()
    trk.set_path([[-0.18, 0.15], [-0.18, -0.15], [0.18, -0.15]])
    assert np.allclose(trk.tangent_at(0.1), (0.0, -1.0))
    assert np.allclose(trk.tangent_at(0.4), (1.0, 0.0))
    assert np.allclose(trk.tangent_at(10.0), (1.0, 0.0))     # clamped at the end
    trk.set_path([[0.1, 0.06]])
    assert np.allclose(trk.tangent_at(0.0), (0.0, 0.0))
