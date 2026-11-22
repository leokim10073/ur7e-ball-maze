"""Look-ahead path tracker (pure numpy, no ROS).

The path is a polyline in the board frame (metres). Each update projects the
ball onto the path, inside a window around the current progress so that a
U-turn cannot make the projection jump to the other leg, and returns the
point ``lookahead`` metres further along the arc as the target.
"""

import numpy as np


class PathTracker:
    def __init__(self, lookahead=0.05, goal_tol=0.02,
                 search_ahead=0.15, search_back=0.02):
        self.lookahead = float(lookahead)
        self.goal_tol = float(goal_tol)
        self.search_ahead = float(search_ahead)
        self.search_back = float(search_back)
        self.points = None
        self.progress = 0.0   # arc length of the furthest projection so far
        self.reached = False

    # ------------------------------------------------------------------ path
    def set_path(self, points):
        """Load a new path (N x 2, N >= 1) and restart from its beginning."""
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        if len(pts) == 0:
            self.points = None
            return
        # Drop consecutive duplicates so every segment has non-zero length.
        keep = np.ones(len(pts), dtype=bool)
        keep[1:] = np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-9
        self.points = pts[keep]
        seg = np.diff(self.points, axis=0)
        self._seg = seg
        self._seg_len = np.linalg.norm(seg, axis=1)
        self._s = np.concatenate([[0.0], np.cumsum(self._seg_len)])
        self.progress = 0.0
        self.reached = False

    @property
    def has_path(self):
        return self.points is not None

    @property
    def length(self):
        return float(self._s[-1]) if self.has_path else 0.0

    @property
    def goal(self):
        return self.points[-1].copy()

    def point_at(self, s):
        """Point on the path at arc length s (clamped to [0, length])."""
        if len(self.points) == 1:
            return self.points[0].copy()
        s = float(np.clip(s, 0.0, self.length))
        i = int(np.searchsorted(self._s, s, side='right') - 1)
        i = min(max(i, 0), len(self._seg) - 1)
        r = (s - self._s[i]) / self._seg_len[i]
        return self.points[i] + r * self._seg[i]

    def tangent_at(self, s):
        """Unit direction of the path at arc length s (clamped to the path);
        zeros for a single-point path."""
        if len(self.points) == 1:
            return np.zeros(2)
        s = float(np.clip(s, 0.0, self.length))
        i = int(np.searchsorted(self._s, s, side='right') - 1)
        i = min(max(i, 0), len(self._seg) - 1)
        return self._seg[i] / self._seg_len[i]

    # ------------------------------------------------------------ projection
    def _project(self, p, s_lo=-np.inf, s_hi=np.inf):
        """Closest point to p on the part of the path with s in [s_lo, s_hi].

        Returns (s, distance)."""
        if len(self.points) == 1:
            return 0.0, float(np.linalg.norm(p - self.points[0]))
        best_s, best_d = 0.0, np.inf
        for i in range(len(self._seg)):
            a, b = self._s[i], self._s[i + 1]
            lo, hi = max(a, s_lo), min(b, s_hi)
            if lo > hi:
                continue
            r = np.dot(p - self.points[i], self._seg[i]) / self._seg_len[i] ** 2
            s = float(np.clip(a + r * self._seg_len[i], lo, hi))
            d = float(np.linalg.norm(p - self.point_at(s)))
            if d < best_d:
                best_s, best_d = s, d
        return best_s, best_d

    def distance_to_path(self, p):
        """Distance from p to the whole path (cross-track error)."""
        return self._project(np.asarray(p, dtype=float))[1]

    # ---------------------------------------------------------------- update
    def update(self, pos):
        """Return the target point for a ball at ``pos`` (None if no path)."""
        if not self.has_path:
            return None
        p = np.asarray(pos, dtype=float)
        goal = self.goal
        # Close to the goal: stop advancing and hold the goal itself.
        self.reached = bool(np.linalg.norm(p - goal) < self.goal_tol)
        if self.reached:
            return goal
        s_near, _ = self._project(p, self.progress - self.search_back,
                                  self.progress + self.search_ahead)
        self.progress = max(self.progress, s_near)
        return self.point_at(self.progress + self.lookahead)
