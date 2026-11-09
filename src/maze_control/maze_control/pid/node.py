"""ROS 2 wrapper around PidPathController. All logic lives in the core modules;
all parameters come from config/pid.yaml (passed in by the launch file)."""

import numpy as np
import rclpy
from geometry_msgs.msg import Vector3Stamped
from nav_msgs.msg import Odometry, Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from ..common.config import unflatten
from .controller import PidPathController


class PidNode(Node):
    def __init__(self):
        super().__init__('maze_pid_controller',
                         automatically_declare_parameters_from_overrides=True)
        flat = {name: p.value for name, p in
                self.get_parameters_by_prefix('').items()}
        params = unflatten(flat)
        try:
            self.ctrl = PidPathController.from_params(params)
        except KeyError as e:
            raise RuntimeError(f'missing parameter {e}; launch with config/pid.yaml')
        topics = params['topics']
        self.frame_id = params['frame_id']

        self.create_subscription(Odometry, topics['state'], self.on_state, 10)
        self.create_subscription(Path, topics['path'], self.on_path, 10)
        self.pub = self.create_publisher(Vector3Stamped, topics['tilt_cmd'], 10)
        self.create_timer(1.0 / params['rate_hz'], self.on_timer)
        self._last_status = None

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_state(self, msg):
        # Receive time drives the watchdog, so camera/robot clock offsets
        # cannot hide a stalled estimator.
        p, v = msg.pose.pose.position, msg.twist.twist.linear
        self.ctrl.on_state(self.now_s(), (p.x, p.y), (v.x, v.y))

    def on_path(self, msg):
        pts = np.array([(ps.pose.position.x, ps.pose.position.y)
                        for ps in msg.poses])
        if self.ctrl.on_path(pts):
            self.get_logger().info(f'new path: {len(pts)} points, '
                                   f'{self.ctrl.tracker.length:.3f} m')

    def on_timer(self):
        theta = self.ctrl.update(self.now_s())
        out = Vector3Stamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self.frame_id
        out.vector.x, out.vector.y = float(theta[0]), float(theta[1])
        self.pub.publish(out)
        if self.ctrl.status != self._last_status:
            self.get_logger().info(f'status: {self.ctrl.status}')
            self._last_status = self.ctrl.status


def main(args=None):
    rclpy.init(args=args)
    node = PidNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
