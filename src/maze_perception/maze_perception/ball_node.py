"""ball_tracker: the ball's state on /ball/state (contract 2), plus everything board_tracker does.

One node for both, so each camera frame's ArUco markers are found once: board pose (P1), then
the ball in that frame (P3, ball.py), then the Kalman filter (P4, ball_filter.py).

Subscribes as board_tracker (see board_node.py), and
  /maze/map    maze_interfaces/MazeMap (contract 1, latched): the walls, which the ball finder leaves
               out (a wall end or a blurred wall line is never the ball). Until it arrives the
               ball is not looked for (/ball/state goes on, tracking_ok = false).

Publishes as board_tracker, and
  /ball/state  maze_interfaces/BallState  every camera frame (after the first camera_info), stamped
               with the image's capture time, frame_id `board_frame` (default maze_board). When the
               ball or the board is not found, the filter's prediction with tracking_ok = false;
               before the ball was ever seen (or after `tracking.lost_reset` without it), zeros with
               an all-zero covariance (unknown). Best effort, keep last 1.

Parameter ball_radius (m, default 0 = perception.yaml's): the ball's radius, for the real ball before
the config follows it.

Every 10 s it logs, besides board_tracker's numbers, the share of frames with the ball and the
capture-to-publish latency (median and worst), which a controller has to predict across.
"""

from dataclasses import replace

import numpy as np
import rclpy
from maze_interfaces.msg import BallState, MazeMap
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from maze_perception.ball import WallGrid, locate_ball
from maze_perception.ball_filter import BallFilter
from maze_perception.board_node import SENSOR_QOS, STATS_PERIOD_S, BoardTracker

# contract 1: /maze/map is published once, latched
MAP_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)


def walls_from_map(msg: MazeMap) -> WallGrid:
    """The walls of a contract-1 map: wall (100) and unknown (-1) cells are walls."""
    g = msg.walls
    cells = np.asarray(g.data, np.int16).reshape(g.info.height, g.info.width)
    return WallGrid(cells=cells != 0, origin=(g.info.origin.position.x, g.info.origin.position.y),
                    resolution=g.info.resolution)


class BallTracker(BoardTracker):
    def __init__(self, **kwargs):
        super().__init__(node_name="ball_tracker", **kwargs)
        radius = float(self.declare_parameter("ball_radius", 0.0).value)
        if radius > 0:  # the real ball, while perception.yaml still has the sim's (#19)
            self.cfg = replace(self.cfg, ball=replace(self.cfg.ball, radius=radius))
            self.get_logger().info(f"ball radius {1000 * radius:.1f} mm (ball_radius parameter)")
        self.filter = BallFilter(self.cfg.tracking)
        self.walls = None
        self.create_subscription(MazeMap, "/maze/map", self._on_map, MAP_QOS)
        self.state_pub = self.create_publisher(BallState, "/ball/state", SENSOR_QOS)
        self._ball_stats = {"frames": 0, "seen": 0, "latency": []}
        self.create_timer(STATS_PERIOD_S, self._log_ball_stats)

    def _on_map(self, msg: MazeMap):
        self.walls = walls_from_map(msg)
        self.get_logger().info(f"maze map {msg.source!r}: {int(self.walls.cells.sum())} wall cells")

    def _on_frame(self, msg: Image, img, pose):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        ball = None
        if self.walls is None:
            self.get_logger().warn("no /maze/map yet: not looking for the ball (ros2 launch maze_cad_map "
                                   "cad_map.launch.py, or perception.launch.py with_map:=true)",
                                   throttle_duration_sec=5.0)
        elif pose is not None:
            found = locate_ball(img, pose, self.camera, self.cfg.ball, self.cfg.rectified, self.walls)
            if found is not None:
                ball = (found.x, found.y)
        est = self.filter.step(stamp, ball)
        if est is None:  # not newer than the last frame
            self.get_logger().warn("camera frame older than the last one, skipped", throttle_duration_sec=5.0)
            return
        out = BallState()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.board_frame
        out.x, out.y, out.vx, out.vy = est.x, est.y, est.vx, est.vy
        out.position_covariance = list(est.position_covariance)
        out.tracking_ok = est.tracking_ok
        self.state_pub.publish(out)

        s = self._ball_stats
        s["frames"] += 1
        s["seen"] += est.tracking_ok
        s["latency"].append((self.get_clock().now().nanoseconds - (msg.header.stamp.sec * 10 ** 9
                                                                    + msg.header.stamp.nanosec)) * 1e-9)

    def _log_ball_stats(self):
        s = self._ball_stats
        if s["frames"]:
            lat = np.array(s["latency"]) * 1000
            self.get_logger().info(f"ball tracked in {100 * s['seen'] / s['frames']:.0f}% of frames, "
                                   f"capture-to-publish {np.median(lat):.1f} ms median, {lat.max():.1f} ms worst")
        self._ball_stats = {"frames": 0, "seen": 0, "latency": []}


def main(args=None):
    rclpy.init(args=args)
    node = BallTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
