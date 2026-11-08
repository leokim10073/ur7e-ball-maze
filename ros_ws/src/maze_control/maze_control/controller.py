"""Tracker + PID + state watchdog, independent of ROS.

The ROS node feeds it ``on_path`` / ``on_state`` from its subscriptions and
calls ``update`` from a 60 Hz timer. Tests drive it the same way with
simulated time.
"""

import math

import numpy as np

from .path_tracker import PathTracker
from .pid_controller import PIDController


class MazeController:
    def __init__(self, tracker, pid, rate_hz=60.0, state_timeout=0.2):
        self.tracker = tracker
        self.pid = pid
        self.rate_hz = float(rate_hz)
        self.state_timeout = float(state_timeout)
        self.pos = None
        self.vel = None
        self.t_state = None
        self.t_last = None
        self.target = None
        self.status = 'no_state'
        self._raw_path = None

    @classmethod
    def from_params(cls, p):
        """Build from the nested ``ros__parameters`` dict of config/pid.yaml."""
        lim, pid, trk, sign = p['limits'], p['pid'], p['tracker'], p['sign']
        deg = math.pi / 180.0
        tracker = PathTracker(lookahead=trk['lookahead'],
                              goal_tol=trk['goal_tol'],
                              search_ahead=trk['search_ahead'],
                              search_back=trk['search_back'])
        ctrl = PIDController(kp=pid['kp'], ki=pid['ki'], kd=pid['kd'],
                             i_zone=pid['i_zone'],
                             i_max=pid['i_max_deg'] * deg,
                             max_tilt=lim['max_tilt_deg'] * deg,
                             max_rate=lim['max_rate_deg_s'] * deg,
                             sign_ax=sign['theta_y_to_ax'],
                             sign_ay=sign['theta_x_to_ay'])
        return cls(tracker, ctrl, rate_hz=p['rate_hz'],
                   state_timeout=p['watchdog']['state_timeout'])

    def on_path(self, points):
        """Load a path. Returns False if it equals the current one (a
        republished path must not reset the progress along it)."""
        pts = np.asarray(points, float).reshape(-1, 2)
        old = self._raw_path
        if old is not None and pts.shape == old.shape \
                and np.allclose(pts, old, atol=1e-6):
            return False
        self._raw_path = pts
        self.tracker.set_path(pts)
        self.pid.reset()
        return True

    def on_state(self, t, pos, vel):
        self.pos = np.asarray(pos, float)
        self.vel = np.asarray(vel, float)
        self.t_state = float(t)

    def update(self, t):
        """Return np.array([theta_x, theta_y]) (rad) for time t (s)."""
        nominal = 1.0 / self.rate_hz
        dt = nominal if self.t_last is None else t - self.t_last
        dt = min(max(dt, 0.0), 2.0 * nominal)   # bound the step after a stall
        self.t_last = t

        if self.t_state is None:
            self.status = 'no_state'
        elif t - self.t_state > self.state_timeout:
            self.status = 'timeout'
        elif not self.tracker.has_path:
            self.status = 'no_path'
        else:
            self.target = self.tracker.update(self.pos)
            self.status = 'reached' if self.tracker.reached else 'tracking'
            return self.pid.step(self.target, self.pos, self.vel, dt)
        return self.pid.step_zero(dt)
