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
    PROFILE_DS = 1e-3         # arc-length grid of the speed profile [m]

    def __init__(self, tracker, law, v_ref, slow_dist, v_turn=None, a_ref=np.inf,
                 corner_window=0.02, **kw):
        self.v_ref = float(v_ref)
        self.v_turn = self.v_ref if v_turn is None else min(float(v_turn), self.v_ref)
        self.a_ref = float(a_ref)
        self.corner_window = float(corner_window)
        self.slow_dist = max(float(slow_dist), 1e-9)
        self.plan_id = None
        self._route = None        # (arc_length (N,), corridors (M, 4), pose_corridor (N,))
        self.speed_profile = None  # (s (n,), v (n,)): reference speed before the goal ramp
        super().__init__(tracker, law, **kw)

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
        return cls(tracker, law, m['v_ref'], m['slow_dist'],
                   v_turn=m.get('v_turn'), a_ref=m.get('a_ref', np.inf),
                   corner_window=m.get('corner_window', 0.02), rate_hz=p['rate_hz'],
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
            self.speed_profile = self._build_profile()
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
    def _build_profile(self):
        """Reference speed over arc length: v_ref on straights, v_turn at a turn
        of 90 deg or more (the angle between the chords corner_window before and
        after each point, so a 5 mm staircase is not a corner), and at most a_ref
        of speed change, so it brakes before a corner and speeds up after it."""
        trk = self.tracker
        if not trk.has_path:
            return None
        L = trk.length
        n = max(int(np.ceil(L / self.PROFILE_DS)) + 1, 2)
        s = np.linspace(0.0, L, n)
        v = np.full(n, self.v_ref)
        if self.v_turn >= self.v_ref or L == 0.0:
            return s, v
        w = self.corner_window
        for i, si in enumerate(s):
            p = trk.point_at(si)
            back = p - trk.point_at(si - w)
            fwd = trk.point_at(si + w) - p
            nb, nf = np.linalg.norm(back), np.linalg.norm(fwd)
            if nb > 1e-9 and nf > 1e-9:
                cos_phi = float(np.dot(back, fwd)) / (nb * nf)
                v[i] = self.v_turn + (self.v_ref - self.v_turn) * max(cos_phi, 0.0)
        if np.isfinite(self.a_ref):
            dv2 = 2.0 * self.a_ref * (s[1] - s[0])
            for i in range(n - 2, -1, -1):           # brake before corners
                v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + dv2))
            for i in range(1, n):                    # speed up after them
                v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + dv2))
        return s, v

    def _speed(self, s):
        ramp = float(np.clip((self.tracker.length - s) / self.slow_dist, 0.0, 1.0))
        if self.speed_profile is None:
            return self.v_ref * ramp
        grid, v = self.speed_profile
        return float(np.interp(s, grid, v)) * ramp

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
