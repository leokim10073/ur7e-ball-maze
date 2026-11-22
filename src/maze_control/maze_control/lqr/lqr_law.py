"""Two-axis discrete LQR tilt controller (pure numpy, no ROS).

Plant model per axis (the axes are decoupled; docs/interfaces.md contract 4
defines tilt_x / tilt_y by their effect on the ball, so both axes are the same
model with a positive sign):

    p' = v
    v' = k * theta - c * v              k = (5/7) g  (rolling solid sphere,
                                                     sin(theta) ~ theta)
    theta' = (u_delayed - theta) / tau  first-order arm lag
    u_delayed(t) = u(t - delay)         camera + filter + arm dead time

State x = [p - p_ref, v, theta] (+ the integral of p - p_ref when
``q_int > 0``), input u = commanded tilt. The gain K comes from the discrete
algebraic Riccati equation for the model discretized at the control period
(zero-order hold), and u = -K x.

* theta is not measured: it is estimated by running the lag model on the
  commands this controller sent (open-loop observer).
* Dead time: the commands already sent but not yet acting are known, so the
  state is predicted ``delay`` ahead through the model before applying K
  (a Smith-predictor style prediction). ``delay = 0`` turns this off.
* The output goes through the same limits as the PID: clipped to
  +-max_tilt, then rate limited to +-max_rate. The applied (limited)
  command is what the observer and the predictor see.
* Integral: optional, with the PID's anti-windup rules (only within
  ``i_zone`` of the target, frozen while the output is limited, capped).
"""

import collections

import numpy as np

G = 9.81


# ------------------------------------------------------------------ linear algebra
def expm(M):
    """Matrix exponential (scaling and squaring with a Taylor series).
    Exact enough for the small, well-scaled matrices used here."""
    M = np.asarray(M, float)
    norm = np.linalg.norm(M, np.inf)
    s = max(0, int(np.ceil(np.log2(norm))) + 1) if norm > 0 else 0
    A = M / (2 ** s)
    E, term = np.eye(len(M)), np.eye(len(M))
    for k in range(1, 20):
        term = term @ A / k
        E = E + term
    for _ in range(s):
        E = E @ E
    return E


def c2d(A, B, dt):
    """Zero-order-hold discretization of x' = A x + B u."""
    n, m = B.shape
    M = np.zeros((n + m, n + m))
    M[:n, :n], M[:n, n:] = A, B
    E = expm(M * dt)
    return E[:n, :n], E[:n, n:]


def dlqr(Ad, Bd, Q, R, tol=1e-12, max_iter=100000):
    """Infinite-horizon discrete LQR gain K (u = -K x) and the Riccati
    solution P, by iterating the Riccati recursion to convergence."""
    P = np.array(Q, float)
    for _ in range(max_iter):
        BtP = Bd.T @ P
        K = np.linalg.solve(R + BtP @ Bd, BtP @ Ad)
        P_new = Q + Ad.T @ P @ (Ad - Bd @ K)
        if np.max(np.abs(P_new - P)) <= tol * max(1.0, np.max(np.abs(P))):
            P = P_new
            break
        P = P_new
    else:
        raise RuntimeError('discrete Riccati iteration did not converge')
    BtP = Bd.T @ P
    K = np.linalg.solve(R + BtP @ Bd, BtP @ Ad)
    return K, P


# ------------------------------------------------------------------ model
def ball_model(c, tau, g=G):
    """Continuous (A, B) of one axis with state [p, v, theta]."""
    k = 5.0 / 7.0 * g
    A = np.array([[0.0, 1.0, 0.0],
                  [0.0, -c, k],
                  [0.0, 0.0, -1.0 / tau]])
    B = np.array([[0.0], [0.0], [1.0 / tau]])
    return A, B


def design_gain(c, tau, dt, q_pos, q_vel, q_tilt, r, q_int=0.0, g=G):
    """LQR gain for one axis.

    Returns (K, Ad, Bd): K has 3 entries [pos, vel, tilt], or 4 with the
    integral appended when q_int > 0; Ad, Bd are the 3-state model used by the
    observer and the predictor."""
    A, B = ball_model(c, tau, g)
    Ad, Bd = c2d(A, B, dt)
    if q_int > 0.0:
        # integral state z_{k+1} = z_k + dt * p_k (forward Euler)
        Aa = np.zeros((4, 4))
        Aa[:3, :3], Aa[3, 0], Aa[3, 3] = Ad, dt, 1.0
        Ba = np.vstack([Bd, [[0.0]]])
        Q = np.diag([q_pos, q_vel, q_tilt, q_int])
        K, _ = dlqr(Aa, Ba, Q, np.array([[r]]))
    else:
        Q = np.diag([q_pos, q_vel, q_tilt])
        K, _ = dlqr(Ad, Bd, Q, np.array([[r]]))
    return K.ravel(), Ad, Bd


# ------------------------------------------------------------------ controller
class LQRController:
    """Same interface as pid_law.PIDController: step / step_zero / reset."""

    def __init__(self, K, Ad, Bd, dt, delay, max_tilt, max_rate,
                 i_zone=0.02, i_max=0.0):
        self.K = np.asarray(K, float).ravel()
        self.Ad, self.Bd = np.asarray(Ad, float), np.asarray(Bd, float)
        self.dt = float(dt)
        self.n_delay = int(round(float(delay) / self.dt))
        self.max_tilt, self.max_rate = float(max_tilt), float(max_rate)
        self.i_zone, self.i_max = float(i_zone), float(i_max)
        self.has_int = len(self.K) == 4
        self.integral = np.zeros(2)
        self.u = np.zeros(2)          # last output (tilt_x, tilt_y)
        self.theta = np.zeros(2)      # estimated actual tilt per axis
        # commands sent but not yet acting on the board, oldest first
        self._pending = collections.deque([np.zeros(2)] * self.n_delay)
        # lag pole of the model, for the observer
        self._lag = float(self.Ad[2, 2])

    @classmethod
    def from_params(cls, model, weights, limits, rate_hz, g=G):
        dt = 1.0 / float(rate_hz)
        K, Ad, Bd = design_gain(model['c'], model['tau'], dt,
                                weights['q_pos'], weights['q_vel'],
                                weights['q_tilt'], weights['r'],
                                q_int=weights['q_int'], g=g)
        return cls(K, Ad, Bd, dt, model['delay'], limits['max_tilt'],
                   limits['max_rate'], i_zone=weights['i_zone'],
                   i_max=weights['i_max'])

    def reset(self):
        """Clear the integrator (the output and the observer keep going)."""
        self.integral[:] = 0.0

    # -------------------------------------------------------------- internals
    def _limit(self, u_des, dt):
        u_sat = np.clip(u_des, -self.max_tilt, self.max_tilt)
        step = self.max_rate * max(dt, 0.0)
        return self.u + np.clip(u_sat - self.u, -step, step)

    def _advance(self, u_out):
        """Send u_out: the command that has waited out the delay starts acting,
        and the observed tilt follows it for one period."""
        if self.n_delay > 0:
            self._pending.append(u_out.copy())
            acting = self._pending.popleft()
        else:
            acting = u_out
        self.theta = self._lag * self.theta + (1.0 - self._lag) * acting
        self.u = u_out

    def _predict(self, e, v):
        """State [e, v, theta] per axis, predicted n_delay steps ahead with the
        commands already in flight. Returns a 3 x 2 array (rows: state)."""
        x = np.vstack([e, v, self.theta])
        for u in self._pending:
            x = self.Ad @ x + self.Bd @ u.reshape(1, 2)
        return x

    # -------------------------------------------------------------- interface
    def step(self, target, pos, vel, dt):
        """One control step. Returns np.array([tilt_x, tilt_y]) in rad."""
        e = np.asarray(pos, float) - np.asarray(target, float)
        v = np.asarray(vel, float)
        x = self._predict(e, v)

        in_zone = np.abs(e) < self.i_zone
        u_des = -(self.K[:3] @ x)
        if self.has_int:
            integral_new = np.where(in_zone, self.integral + e * dt, 0.0)
            if self.i_max > 0.0 and abs(self.K[3]) > 0.0:
                lim = self.i_max / abs(self.K[3])
                integral_new = np.clip(integral_new, -lim, lim)
            u_des = u_des - self.K[3] * integral_new

        u_out = self._limit(u_des, dt)

        if self.has_int:
            # freeze an axis whose output is limited when integrating would
            # drive it further into the limit (same rule as the PID)
            limited = np.abs(u_des - u_out) > 1e-12
            drive = -self.K[3] * e                  # sign of the integral's push
            winding = limited & (np.sign(drive) == np.sign(u_des - u_out))
            self.integral = np.where(winding & in_zone, self.integral,
                                     integral_new)

        self._advance(u_out)
        return u_out.copy()

    def step_zero(self, dt):
        """Ramp the tilt back to zero (rate limited) and clear the integrator."""
        self.reset()
        u_out = self._limit(np.zeros(2), dt)
        self._advance(u_out)
        return u_out.copy()
