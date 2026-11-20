"""ball_tracker: the ball's state on /ball/state (contract 2), plus everything board_tracker does.

One node for both, so each camera frame's ArUco markers are found once: board pose (P1), then
the ball in that frame (P3, ball.py), then the Kalman filter (P4, ball_filter.py).

Subscribes / publishes as board_tracker (see board_node.py), and
  /ball/state  maze_interfaces/BallState  every camera frame (after the first camera_info), stamped
               with the image's capture time, frame_id `board_frame` (default maze_board). When the
               ball or the board is not found, the filter's prediction with tracking_ok = false;
               before the ball was ever seen (or after `tracking.lost_reset` without it), zeros with
               an all-zero covariance (unknown). Best effort, keep last 1.

Every 10 s it logs, besides board_tracker's numbers, the share of frames with the ball and the
capture-to-publish latency (median and worst), which a controller has to predict across.
"""

import numpy as np
import rclpy
from maze_interfaces.msg import BallState
from sensor_msgs.msg import Image

from maze_perception.ball import locate_ball
from maze_perception.ball_filter import BallFilter
from maze_perception.board_node import SENSOR_QOS, STATS_PERIOD_S, BoardTracker


class BallTracker(BoardTracker):
    def __init__(self, **kwargs):
        super().__init__(node_name="ball_tracker", **kwargs)
        self.filter = BallFilter(self.cfg.tracking)
        self.state_pub = self.create_publisher(BallState, "/ball/state", SENSOR_QOS)
        self._ball_stats = {"frames": 0, "seen": 0, "latency": []}
        self.create_timer(STATS_PERIOD_S, self._log_ball_stats)

    def _on_frame(self, msg: Image, img, pose):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        ball = None
        if pose is not None:
            found = locate_ball(img, pose, self.camera, self.cfg.ball, self.cfg.rectified)
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
