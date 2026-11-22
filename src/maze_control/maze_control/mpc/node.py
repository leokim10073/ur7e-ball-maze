"""ROS 2 node for the MPC controller: the PID node's state input, tilt output,
QoS and watchdog handling, with the planner's full route (contract 3:
/maze/route, path + corridors) instead of /maze/path. Parameters come from
config/mpc.yaml."""

import numpy as np
from maze_interfaces.msg import PlannedRoute

from ..pid.node import PATH_QOS, PidNode, run
from .controller import MpcPathController


class MpcNode(PidNode):
    NODE_NAME = 'maze_mpc_controller'
    CONTROLLER = MpcPathController
    CONFIG = 'config/mpc.yaml'

    def __init__(self):
        super().__init__()
        self._last_mode = None
        self._last_logged = None
        self.create_timer(1.0, self.log_solve_time)

    def _subscribe_path(self, topics):
        self.create_subscription(PlannedRoute, topics['route'], self.on_route,
                                 PATH_QOS)

    def on_route(self, msg):
        if msg.status == PlannedRoute.STATUS_NO_PATH:
            pts, arc, corr, pose_corr = [], [], [], []
        else:
            pts = np.array([(ps.pose.position.x, ps.pose.position.y)
                            for ps in msg.path.poses]).reshape(-1, 2)
            arc = np.array(msg.arc_length, float)
            corr = np.array([(c.x_min, c.x_max, c.y_min, c.y_max)
                             for c in msg.corridors]).reshape(-1, 4)
            pose_corr = np.array(msg.pose_corridor, int)
        if self.ctrl.on_route(pts, arc, corr, pose_corr, msg.plan_id):
            self.get_logger().info(
                f'new route (plan {msg.plan_id}): {len(pts)} points, '
                f'{self.ctrl.tracker.length:.3f} m, {len(corr)} corridors')

    def on_timer(self):
        super().on_timer()
        mode = self.ctrl.law.mode
        if mode != self._last_mode:
            log = self.get_logger().warn if mode == 'fallback' else self.get_logger().info
            log(f'mpc: {mode} (failed solves in a row: {self.ctrl.law.n_fail})')
            self._last_mode = mode

    def log_solve_time(self):
        t = self.ctrl.law.solve_times
        if t and (len(t), t[-1]) != self._last_logged:   # only after new solves
            self._last_logged = (len(t), t[-1])
            self.get_logger().info(f'solve time: median {1000 * np.median(t):.1f} ms, '
                                    f'max {1000 * max(t):.1f} ms (last {len(t)})')


def main(args=None):
    run(MpcNode, args)


if __name__ == '__main__':
    main()
