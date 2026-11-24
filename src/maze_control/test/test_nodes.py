"""Node-level smoke tests: run pid_node, lqr_node and mpc_node in-process with
their shipped config, feed them a route and ball states over ROS, and check
that they publish tilts and survive a full run.

The core tests never start a node, so errors in the ROS wrappers (logging,
QoS, message conversion) only showed up in the sim. Needs a sourced workspace
(rclpy + maze_interfaces); skipped otherwise.
"""

import os
import time

import pytest

rclpy = pytest.importorskip('rclpy')
msgs = pytest.importorskip('maze_interfaces.msg')

from geometry_msgs.msg import PoseStamped  # noqa: E402
from nav_msgs.msg import Path  # noqa: E402
from rclpy.executors import SingleThreadedExecutor  # noqa: E402

from maze_control.lqr.node import LqrNode  # noqa: E402
from maze_control.pid.node import PATH_QOS, STATE_QOS, TILT_QOS, PidNode  # noqa: E402

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'config')
START, END = (-0.1, 0.0), (0.1, 0.0)          # straight path along +x [m]


@pytest.fixture
def ros(monkeypatch):
    """rclpy in its own ROS domain, so a running sim cannot interfere."""
    monkeypatch.setenv('ROS_DOMAIN_ID', str(150 + os.getpid() % 80))
    contexts = []

    def init(config):
        rclpy.init(args=['--ros-args', '--params-file',
                         os.path.join(CONFIG_DIR, config)])
        contexts.append(True)
    yield init
    if contexts:
        rclpy.try_shutdown()


def _path_msg():
    path = Path()
    path.header.frame_id = 'maze_board'
    n = 21
    for i in range(n):
        ps = PoseStamped()
        ps.header.frame_id = 'maze_board'
        ps.pose.position.x = START[0] + (END[0] - START[0]) * i / (n - 1)
        ps.pose.position.y = START[1]
        path.poses.append(ps)
    return path


def _route_msg():
    route = msgs.PlannedRoute()
    route.header.frame_id = 'maze_board'
    route.plan_id = 1
    route.status = msgs.PlannedRoute.STATUS_OK
    route.path = _path_msg()
    xs = [ps.pose.position.x for ps in route.path.poses]
    route.arc_length = [x - xs[0] for x in xs]
    route.corridors = [msgs.Corridor(x_min=START[0] - 0.01, x_max=END[0] + 0.01,
                                     y_min=-0.01, y_max=0.01)]
    route.pose_corridor = [0] * len(xs)
    return route


class Harness:
    """A second node standing in for perception and the planner."""

    def __init__(self, ctrl_node, route):
        self.ctrl_node = ctrl_node
        self.io = rclpy.create_node('test_harness')
        if route:
            pub = self.io.create_publisher(msgs.PlannedRoute, '/maze/route', PATH_QOS)
            pub.publish(_route_msg())
        else:
            pub = self.io.create_publisher(Path, '/maze/path', PATH_QOS)
            pub.publish(_path_msg())
        self._path_pub = pub                   # keep the latched publisher alive
        self.state_pub = self.io.create_publisher(msgs.BallState, '/ball/state', STATE_QOS)
        self.tilts = []
        self.io.create_subscription(msgs.BoardTilt, '/board/tilt_cmd',
                                    self.tilts.append, TILT_QOS)
        self.ex = SingleThreadedExecutor()
        self.ex.add_node(ctrl_node)
        self.ex.add_node(self.io)

    def spin(self, seconds, each=None):
        """Spin both nodes for `seconds` of wall time, publishing a ball held
        at the path start at ~60 Hz. Exceptions in callbacks propagate."""
        t_end = time.monotonic() + seconds
        t_pub = 0.0
        while time.monotonic() < t_end:
            if time.monotonic() >= t_pub:
                s = msgs.BallState()
                s.header.stamp = self.io.get_clock().now().to_msg()
                s.header.frame_id = 'maze_board'
                s.x, s.y = START
                s.tracking_ok = True
                self.state_pub.publish(s)
                t_pub = time.monotonic() + 1.0 / 60
            self.ex.spin_once(timeout_sec=0.005)
            if each:
                each()

    def close(self):
        self.ex.shutdown()
        self.io.destroy_node()
        self.ctrl_node.destroy_node()


def _check_tracks(h):
    assert h.ctrl_node.ctrl.status == 'tracking'
    assert len(h.tilts) > 30, 'tilt_cmd should come at the 60 Hz timer rate'
    assert all(t.header.frame_id == 'maze_board' for t in h.tilts)
    # Ball held at the start of a path along +x: the board tilts it towards +x.
    assert max(t.tilt_x for t in h.tilts) > 0.0


@pytest.mark.parametrize('node_cls, config', [(PidNode, 'pid.yaml'),
                                              (LqrNode, 'lqr.yaml')])
def test_path_node_runs(ros, node_cls, config):
    ros(config)
    h = Harness(node_cls(), route=False)
    try:
        h.spin(1.5)
        _check_tracks(h)
    finally:
        h.close()


def test_mpc_node_runs_and_survives_fallback(ros):
    """mpc_node on its shipped config, then with every solve counted as too
    slow, so it switches between ok / late and fallback. The first fallback
    after an ok used to crash the node (Humble: "Logger severity cannot be
    changed between calls")."""
    pytest.importorskip('casadi')
    from maze_control.mpc.node import MpcNode

    ros('mpc.yaml')
    h = Harness(MpcNode(), route=True)
    law = h.ctrl_node.ctrl.law
    modes = []
    record = lambda: modes.append(h.ctrl_node._last_mode)  # noqa: E731
    try:
        h.spin(2.0, each=record)
        _check_tracks(h)
        assert 'ok' in modes

        law.solve_budget = 0.0                 # every solve is now "too slow"
        n_before = len(h.tilts)
        h.spin(1.0, each=record)
        first_ok = modes.index('ok')
        assert 'fallback' in modes[first_ok:], 'the ok -> fallback switch was not hit'
        assert len(h.tilts) - n_before > 30, 'the node stopped publishing'
    finally:
        h.close()
