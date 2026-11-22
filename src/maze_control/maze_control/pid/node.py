"""ROS 2 wrapper around PidPathController. All logic lives in the core modules;
all parameters come from config/pid.yaml (passed in by the launch file)."""

import numpy as np
import rclpy
from maze_interfaces.msg import BallState, BoardTilt
from nav_msgs.msg import Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)

from ..common.config import unflatten
from .controller import PidPathController

# QoS from docs/interfaces.md.
# Contract 2: /ball/state is best effort, keep last 1. A reliable subscriber
# would be incompatible with it and receive nothing.
STATE_QOS = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                       reliability=ReliabilityPolicy.BEST_EFFORT)
# Contract 3: the route is latched, so a restarted controller gets the current path.
PATH_QOS = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                      reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL)
# Contract 4a: /board/tilt_cmd is reliable, keep last 1.
TILT_QOS = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                      reliability=ReliabilityPolicy.RELIABLE)


class PidNode(Node):
    """Path-following ball controller node. Other controllers (lqr/node.py)
    reuse it with their own node name, controller class and config file."""

    NODE_NAME = 'maze_pid_controller'
    CONTROLLER = PidPathController
    CONFIG = 'config/pid.yaml'

    def __init__(self):
        super().__init__(self.NODE_NAME,
                         automatically_declare_parameters_from_overrides=True)
        flat = {name: p.value for name, p in
                self.get_parameters_by_prefix('').items()}
        params = unflatten(flat)
        try:
            self.ctrl = self.CONTROLLER.from_params(params)
        except KeyError as e:
            raise RuntimeError(f'missing parameter {e}; launch with {self.CONFIG}')
        topics = params['topics']
        self.frame_id = params['frame_id']

        self.create_subscription(BallState, topics['state'], self.on_state,
                                 STATE_QOS)
        self._subscribe_path(topics)
        self.pub = self.create_publisher(BoardTilt, topics['tilt_cmd'], TILT_QOS)
        self.create_timer(1.0 / params['rate_hz'], self.on_timer)
        self._last_status = None

    def _subscribe_path(self, topics):
        """The path input; controllers that read the full route override it."""
        self.create_subscription(Path, topics['path'], self.on_path, PATH_QOS)

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_state(self, msg):
        # tracking_ok = false: x, y are only the filter's prediction. Such a
        # state does not feed the watchdog, so after state_timeout (0.2 s,
        # contract 2) without a real detection the board goes level, and
        # after abort_timeout (2 s) the run stops.
        if not msg.tracking_ok:
            return
        # Receive time drives the watchdog, so camera/robot clock offsets
        # cannot hide a stalled estimator.
        self.ctrl.on_state(self.now_s(), (msg.x, msg.y), (msg.vx, msg.vy))

    def on_path(self, msg):
        pts = np.array([(ps.pose.position.x, ps.pose.position.y)
                        for ps in msg.poses])
        if self.ctrl.on_path(pts):
            self.get_logger().info(f'new path: {len(pts)} points, '
                                   f'{self.ctrl.tracker.length:.3f} m')

    def on_timer(self):
        tilt = self.ctrl.update(self.now_s())
        out = BoardTilt()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self.frame_id
        out.tilt_x, out.tilt_y = float(tilt[0]), float(tilt[1])
        self.pub.publish(out)
        if self.ctrl.status != self._last_status:
            self.get_logger().info(f'status: {self.ctrl.status}')
            self._last_status = self.ctrl.status


def run(node_cls, args=None):
    rclpy.init(args=args)
    node = node_cls()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def main(args=None):
    run(PidNode, args)


if __name__ == '__main__':
    main()
