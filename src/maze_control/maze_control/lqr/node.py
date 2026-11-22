"""ROS 2 node for the LQR controller: the PID node's topics, QoS and watchdog
handling with LqrPathController. Parameters come from config/lqr.yaml."""

from ..pid.node import PidNode, run
from .controller import LqrPathController


class LqrNode(PidNode):
    NODE_NAME = 'maze_lqr_controller'
    CONTROLLER = LqrPathController
    CONFIG = 'config/lqr.yaml'

    def __init__(self):
        super().__init__()
        K = self.ctrl.law.K
        self.get_logger().info('LQR gain per axis: ' + ', '.join(
            f'{n}={k:.3f}' for n, k in zip(('pos', 'vel', 'tilt', 'int'), K)))


def main(args=None):
    run(LqrNode, args)


if __name__ == '__main__':
    main()
