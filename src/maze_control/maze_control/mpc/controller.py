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
        for name, val in (('v_turn', v_turn), ('a_ref', a_ref),
                          ('corner_window', corner_window)):
            if val is not None and not float(val) > 0.0:
                raise ValueError(f'mpc.{name} must be > 0, got {val}')
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
        """Reference speed over arc length: v_ref on straights, v_turn at a 90 deg
        turn, and at most a_ref of speed change, so it brakes before a corner and
        speeds up after it.

        The turn at s is the angle phi between the chords [s - 2w, s - w] and
        [s + w, s + 2w] (w = corner_window): they skip the corner itself, so a
        90 deg bend straightened into two 45 deg bends (a diagonal shorter than
        2w) still reads as 90 deg, and a 5 mm staircase averages out. The cap
        keeps the velocity change of the turn, 2 v sin(phi / 2), at that of
        v_turn through 90 deg: v_cap = v_turn sqrt(2) / (2 sin(phi / 2))."""
        trk = self.tracker
        if not trk.has_path:
            return None
        L = trk.length
        n = max(int(np.ceil(L / self.PROFILE_DS)) + 1, 2)
        s = np.linspace(0.0, L, n)
        if self.v_turn >= self.v_ref or L == 0.0:
            return s, np.full(n, self.v_ref)

        def at(q):
            q = np.clip(q, 0.0, L)
            return np.stack([np.interp(q, trk._s, trk.points[:, k]) for k in (0, 1)], 1)

        w = self.corner_window
        back = at(s - w) - at(s - 2 * w)
        fwd = at(s + 2 * w) - at(s + w)
        nb, nf = np.linalg.norm(back, axis=1), np.linalg.norm(fwd, axis=1)
        ok = (nb > 1e-9) & (nf > 1e-9)
        cos_phi = np.ones(n)
        cos_phi[ok] = np.einsum('ij,ij->i', back[ok], fwd[ok]) / (nb[ok] * nf[ok])
        sin_half = np.sqrt(np.clip((1.0 - cos_phi) / 2.0, 0.0, 1.0))
        with np.errstate(divide='ignore'):
            v = np.minimum(self.v_ref, self.v_turn * np.sqrt(2.0) / (2.0 * sin_half))
        if np.isfinite(self.a_ref):
            dv2 = 2.0 * self.a_ref * (s[1] - s[0])
            vl = v.tolist()
            for i in range(n - 2, -1, -1):           # brake before corners
                vl[i] = min(vl[i], (vl[i + 1] ** 2 + dv2) ** 0.5)
            for i in range(1, n):                    # speed up after them
                vl[i] = min(vl[i], (vl[i - 1] ** 2 + dv2) ** 0.5)
            v = np.array(vl)
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
