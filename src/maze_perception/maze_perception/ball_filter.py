"""The ball's state from the per-frame measurements (P4): a constant-velocity Kalman filter in the
board frame, run at each image's capture time (contract 2: the state is the one at header.stamp;
controllers predict forward from it themselves).

State [x, y, vx, vy] (m, m/s). Between frames the ball keeps its velocity up to a random
acceleration (white, `accel_noise` m/s^2: a 5 deg tilt rolls the ball at about 0.6 m/s^2, a wall
stops it within a frame). A measurement is the ball center from ball.py with `measurement_noise`
(m) per axis.

- A measurement farther from the prediction than `max_jump` plus 3 standard deviations is not
  used: one misdetection elsewhere on the board does not throw the state off. A gate on distance,
  not on Mahalanobis distance alone: a ball stopped by a wall lands a few mm off the prediction,
  which a tight statistical gate would throw away. After `max_rejects` such frames in a row the
  ball is taken to be there after all and the filter restarts from it.
- No measurement (ball or board not found): the state is the prediction and tracking is not ok.
  After `lost_reset` seconds without one, the filter forgets the ball (no state, covariance
  unknown) instead of extrapolating further.
- A frame not newer than the last one is ignored (no going back in time).
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from maze_perception.config import TrackingSettings


@dataclass(frozen=True)
class BallEstimate:
    """One contract-2 message without ROS types."""

    stamp: float  # s, capture time of the image
    x: float  # m, board frame
    y: float
    vx: float  # m/s
    vy: float
    position_covariance: Tuple[float, float, float, float]  # 2x2 row-major, m^2; zeros if unknown
    tracking_ok: bool  # this frame's measurement was used


class BallFilter:
    def __init__(self, settings: TrackingSettings):
        self.s = settings
        self.state: Optional[np.ndarray] = None  # [x, y, vx, vy]
        self.P: Optional[np.ndarray] = None
        self.t: Optional[float] = None  # time of the state
        self.last_seen: Optional[float] = None  # time of the last measurement used
        self.rejects = 0  # gated measurements in a row

    @property
    def initialized(self) -> bool:
        return self.state is not None

    def reset(self):
        self.state = self.P = self.last_seen = None
        self.rejects = 0

    def _start(self, t: float, z: np.ndarray):
        r2, v = self.s.measurement_noise ** 2, self.s.initial_speed
        self.state = np.array([z[0], z[1], 0.0, 0.0])
        self.P = np.diag([r2, r2, v ** 2, v ** 2])
        self.last_seen = t
        self.rejects = 0

    def _predict(self, dt: float):
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        q = self.s.accel_noise ** 2
        # white acceleration: position and velocity noise of a random constant push over dt
        Q1 = q * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]])
        Q = np.zeros((4, 4))
        Q[np.ix_([0, 2], [0, 2])] = Q1
        Q[np.ix_([1, 3], [1, 3])] = Q1
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + Q

    def step(self, t: float, measurement: Optional[Tuple[float, float]]) -> Optional[BallEstimate]:
        """Advance to capture time t (s) with this frame's ball center (board x, y in m) or None
        when it was not found. None (and no change) for a frame not newer than the last."""
        if self.t is not None and t <= self.t:
            return None
        dt = None if self.t is None else t - self.t
        self.t = t
        if self.initialized:
            self._predict(dt)
            if self.last_seen is not None and t - self.last_seen > self.s.lost_reset and measurement is None:
                self.reset()

        used = False
        if measurement is not None:
            z = np.asarray(measurement, float)
            if not self.initialized:
                self._start(t, z)
                used = True
            else:
                H = np.zeros((2, 4))
                H[0, 0] = H[1, 1] = 1.0
                S = H @ self.P @ H.T + np.eye(2) * self.s.measurement_noise ** 2
                innov = z - H @ self.state
                if np.linalg.norm(innov) <= self.s.max_jump + 3 * np.sqrt(np.trace(S) / 2):
                    K = self.P @ H.T @ np.linalg.inv(S)
                    self.state = self.state + K @ innov
                    I_KH = np.eye(4) - K @ H
                    # Joseph form: stays symmetric and positive
                    self.P = I_KH @ self.P @ I_KH.T + K @ (np.eye(2) * self.s.measurement_noise ** 2) @ K.T
                    self.last_seen = t
                    self.rejects = 0
                    used = True
                else:
                    self.rejects += 1
                    if self.rejects >= self.s.max_rejects:
                        self._start(t, z)
                        used = True
        return self.estimate(used)

    def estimate(self, tracking_ok: bool) -> BallEstimate:
        if not self.initialized:
            return BallEstimate(self.t if self.t is not None else 0.0, 0.0, 0.0, 0.0, 0.0, (0.0, 0.0, 0.0, 0.0), False)
        x, y, vx, vy = (float(v) for v in self.state)
        cov = self.P[:2, :2]
        return BallEstimate(self.t, x, y, vx, vy, tuple(float(v) for v in cov.ravel()), tracking_ok)
