"""Tracker + LQR + state watchdog, independent of ROS.

The run logic (look-ahead target on the path, tracking-loss watchdog, stop and
restart rules of contract 2) is the PID's: LqrPathController only swaps the
control law, which has the same step / step_zero / reset interface. A change
to that logic in pid/controller.py applies to both controllers.
"""

from ..common.path_tracker import PathTracker
from ..pid.controller import PidPathController
from .lqr_law import LQRController


class LqrPathController(PidPathController):
    @classmethod
    def from_params(cls, p):
        """Build from the nested ``ros__parameters`` dict of config/lqr.yaml."""
        trk = p['tracker']
        tracker = PathTracker(lookahead=trk['lookahead'],
                              goal_tol=trk['goal_tol'],
                              search_ahead=trk['search_ahead'],
                              search_back=trk['search_back'])
        law = LQRController.from_params(p['model'], p['lqr'], p['limits'],
                                        p['rate_hz'], g=p['model']['g'])
        return cls(tracker, law, rate_hz=p['rate_hz'],
                   state_timeout=p['watchdog']['state_timeout'],
                   abort_timeout=p['watchdog']['abort_timeout'])

    # the base class calls its control law "pid"; give it a neutral name too
    @property
    def law(self):
        return self.pid
