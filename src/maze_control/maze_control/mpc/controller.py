"""Tracker + MPC + state watchdog, independent of ROS.

The run logic (progress along the path, tracking-loss watchdog, stop and
restart rules of contract 2) is the PID's (pid/controller.py). This class adds
the planner's route (contract 3): the corridors that bound the predicted ball,
and the horizon reference the MPC tracks.
"""

import numpy as np

from ..common.path_tracker import PathTracker
from ..pid.controller import PidPathController
from .mpc_law import HorizonRef, MPCController


class MpcPathController(PidPathController):
    def __init__(self, tracker, law, v_ref, slow_dist, **kw):
        super().__init__(tracker, law, **kw)
        self.v_ref = float(v_ref)
        self.slow_dist = max(float(slow_dist), 1e-9)
        self.plan_id = None
        self._route = None        # (arc_length (N,), corridors (M, 4), pose_corridor (N,))

    @classmethod
    def from_params(cls, p):
        """Build from the nested ``ros__parameters`` dict of config/mpc.yaml."""
        trk, m = p['tracker'], p['mpc']
        tracker = PathTracker(lookahead=trk['lookahead'],
                              goal_tol=trk['goal_tol'],
                              search_ahead=trk['search_ahead'],
                              search_back=trk['search_back'])
        law = MPCController.from_params(p['model'], m, p['limits'], p['rate_hz'],
                                        g=p['model']['g'])
        return cls(tracker, law, m['v_ref'], m['slow_dist'], rate_hz=p['rate_hz'],
                   state_timeout=p['watchdog']['state_timeout'],
                   abort_timeout=p['watchdog']['abort_timeout'])

    @property
    def law(self):
        return self.pid

    # ------------------------------------------------------------------ input
    def on_path(self, points):
        """A path without corridors (or the route's own path, unchanged)."""
        changed = super().on_path(points)
        if changed:
            self._route = None
        return changed

    def on_route(self, points, arc_length, corridors, pose_corridor, plan_id):
        """Load a PlannedRoute. Returns True when it starts a new run: a new
        plan_id or different points. Empty points clear the path."""
        pts = np.asarray(points, float).reshape(-1, 2)
        if plan_id != self.plan_id:
            self._raw_path = None             # same points, new plan: new run
        self.plan_id = plan_id
        changed = self.on_path(pts)
        corr = np.asarray(corridors, float).reshape(-1, 4)
        if len(pts) and len(corr):
            self._route = (np.asarray(arc_length, float).ravel(), corr,
                           np.asarray(pose_corridor, int).ravel())
        else:
            self._route = None
        return changed

    # -------------------------------------------------------------- reference
    def _speed(self, s):
        return self.v_ref * float(np.clip((self.tracker.length - s) / self.slow_dist,
                                          0.0, 1.0))

    def _bounds(self, s):
        """(lo, hi) of the corridor at arc length s: that of the last route
        pose at or before s."""
        if self._route is None:
            return np.full(2, -np.inf), np.full(2, np.inf)
        arc, corr, pose_corr = self._route
        i = int(np.clip(np.searchsorted(arc, s + 1e-12, side='right') - 1,
                        0, len(pose_corr) - 1))
        c = corr[pose_corr[i]]
        return c[[0, 2]], c[[1, 3]]

    def horizon_ref(self):
        """HorizonRef from the tracker's progress: the reference moves along the
        path at v_ref (slowing over the last slow_dist), starting n_delay steps
        ahead because the MPC's first state is predicted past the delay."""
        law, trk = self.law, self.tracker
        N, dt = law.N, law.dt
        R, V = np.empty((2, N)), np.empty((2, N))
        lo, hi = np.empty((2, N)), np.empty((2, N))
        if trk.reached:
            g = trk.goal
            R[:], V[:] = g.reshape(2, 1), 0.0
            for k in range(N):
                lo[:, k], hi[:, k] = self._bounds(trk.length)
            return HorizonRef(R, V, lo, hi)
        s = trk.progress
        for _ in range(law.n_delay):
            s += dt * self._speed(s)
        for k in range(N):
            s += dt * self._speed(s)
            R[:, k] = trk.point_at(s)
            V[:, k] = self._speed(s) * trk.tangent_at(s)
            lo[:, k], hi[:, k] = self._bounds(s)
        return HorizonRef(R, V, lo, hi)

    def _law_step(self, dt):
        ref = self.horizon_ref()
        self.target = ref.R[:, 0].copy()
        return self.law.step(ref, self.pos, self.vel, dt)
