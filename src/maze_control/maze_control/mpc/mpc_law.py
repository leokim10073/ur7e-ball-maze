"""Constrained tracking MPC for the ball (CasADi + OSQP, no ROS).

Design: docs/superpowers/specs/2026-11-22-mpc-design.md. Each axis is the
LQR's model (lqr_law.ball_model, discretized by lqr_law.c2d), state
[p, v, theta]; the two axes stack into x = [px, vx, thx, py, vy, thy],
u = [tilt_x, tilt_y].

CFTOC solved every control period (k = 0 .. N-1, p = (px, py), v = (vx, vy)):

    min  sum_k  q_pos |p_{k+1} - R_k|^2 + q_vel |v_{k+1} - V_k|^2
              + r |u_k|^2 + r_du |u_k - u_{k-1}|^2
              + rho_lin sum(S_k) + rho_quad |S_k|^2
    s.t. X_0 = x0,  X_{k+1} = Ad X_k + Bd U_k
         |U_k| <= max_tilt,  |U_k - U_{k-1}| <= max_rate dt   (U_{-1} = u_prev)
         lo_k - S_k <= p_{k+1} <= hi_k + S_k,  S_k >= 0

The corridor constraints are soft (slack S with an exact linear penalty), so
the QP stays feasible when a noisy state puts the ball outside a corridor.
"""

import dataclasses
import time

import casadi as ca
import numpy as np

# Infinite corridor bounds are replaced by this (m): far outside any board.
BIG = 1.0e3

# A plan whose inputs break a limit by more than this (rad) is a failed solve;
# smaller breaks (OSQP's accuracy, ~1e-5 when a corridor slack is active) are
# projected back onto the limits.
LIMIT_TOL = 1.0e-4

# rows of p and v in the stacked state
P_ROWS, V_ROWS = [0, 3], [1, 4]


@dataclasses.dataclass
class HorizonRef:
    """References and corridor bounds over the horizon, each 2 x N
    (row 0: x, row 1: y). lo / hi may be +-inf (unconstrained)."""
    R: np.ndarray
    V: np.ndarray
    lo: np.ndarray
    hi: np.ndarray


@dataclasses.dataclass
class Plan:
    U: np.ndarray        # 2 x N inputs
    X: np.ndarray        # 6 x (N+1) predicted states
    S: np.ndarray        # 2 x N corridor slacks
    ok: bool
    t_solve: float       # wall time of the QP solve [s]


class CFTOC:
    """The finite-time optimal control problem, built once, solved per step."""

    def __init__(self, Ad, Bd, dt, horizon, q_pos, q_vel, r, r_du, rho_lin,
                 rho_quad, max_tilt, max_rate, osqp_opts=None):
        self.dt = float(dt)
        self.N = N = int(horizon)
        self.max_tilt, self.max_rate = float(max_tilt), float(max_rate)
        Ad2 = np.kron(np.eye(2), np.asarray(Ad, float))
        Bd2 = np.kron(np.eye(2), np.asarray(Bd, float))

        opti = ca.Opti('conic')
        X, U, S = opti.variable(6, N + 1), opti.variable(2, N), opti.variable(2, N)
        x0, R, V = opti.parameter(6), opti.parameter(2, N), opti.parameter(2, N)
        lo, hi, u_prev = opti.parameter(2, N), opti.parameter(2, N), opti.parameter(2)

        du_max = self.max_rate * self.dt
        J = 0
        opti.subject_to(X[:, 0] == x0)
        for k in range(N):
            opti.subject_to(X[:, k + 1] == Ad2 @ X[:, k] + Bd2 @ U[:, k])
            p, v = X[P_ROWS, k + 1], X[V_ROWS, k + 1]
            du = U[:, k] - (u_prev if k == 0 else U[:, k - 1])
            J += (q_pos * ca.sumsqr(p - R[:, k]) + q_vel * ca.sumsqr(v - V[:, k])
                  + r * ca.sumsqr(U[:, k]) + r_du * ca.sumsqr(du)
                  + rho_lin * ca.sum1(S[:, k]) + rho_quad * ca.sumsqr(S[:, k]))
            opti.subject_to(opti.bounded(-self.max_tilt, U[:, k], self.max_tilt))
            opti.subject_to(opti.bounded(-du_max, du, du_max))
            opti.subject_to(p >= lo[:, k] - S[:, k])
            opti.subject_to(p <= hi[:, k] + S[:, k])
            opti.subject_to(S[:, k] >= 0)
        opti.minimize(J)
        # OSQP's own warm_start is rejected by CasADi (impure); use its options.
        opti.solver('osqp', {'warm_start_primal': True, 'warm_start_dual': True,
                             'error_on_fail': False,
                             'osqp': {'verbose': False, **(osqp_opts or {})}})
        self._fn = opti.to_function('cftoc', [x0, R, V, lo, hi, u_prev, X, U, S],
                                    [X, U, S])
        self._guess = None
        # The first call is several times slower than the rest; pay it here.
        z = np.zeros((2, N))
        self.solve(np.zeros(6), HorizonRef(z, z, z - 1.0, z + 1.0), np.zeros(2))
        self._guess = None

    # -------------------------------------------------------------- internals
    def _solve_qp(self, args):
        """Call the CasADi function. Returns ([X, U, S], ok)."""
        try:
            out = self._fn(*args)
        except RuntimeError:
            return None, False
        out = [np.array(o, float) for o in out]
        ok = all(np.all(np.isfinite(o)) for o in out)
        stats = self._fn.stats()
        if ok and 'success' in stats:
            ok = bool(stats['success'])
        return out, ok

    def _initial_guess(self):
        if self._guess is None:
            N = self.N
            return np.zeros((6, N + 1)), np.zeros((2, N)), np.zeros((2, N))
        X, U, S = self._guess
        shift = lambda M: np.hstack([M[:, 1:], M[:, -1:]])  # noqa: E731
        return shift(X), shift(U), shift(S)

    def _project_limits(self, U, u_prev):
        """Clip U onto the tilt and rate limits, step by step. Returns
        (U_projected, largest change made)."""
        du_max = self.max_rate * self.dt
        out, prev = np.empty_like(U), u_prev
        for k in range(U.shape[1]):
            u = np.clip(U[:, k], -self.max_tilt, self.max_tilt)
            prev = out[:, k] = prev + np.clip(u - prev, -du_max, du_max)
        return out, float(np.abs(out - U).max())

    # -------------------------------------------------------------- interface
    def solve(self, x0, ref, u_prev):
        """Solve from state x0 (6,) with HorizonRef ref; u_prev (2,) is the
        command applied last period. Returns a Plan."""
        u_prev = np.asarray(u_prev, float).reshape(2)
        lo = np.where(np.isfinite(ref.lo), ref.lo, -BIG)
        hi = np.where(np.isfinite(ref.hi), ref.hi, BIG)
        args = [np.asarray(x0, float).reshape(6), ref.R, ref.V, lo, hi, u_prev,
                *self._initial_guess()]
        t0 = time.perf_counter()
        out, ok = self._solve_qp(args)
        t_solve = time.perf_counter() - t0
        if out is None:
            N = self.N
            return Plan(np.zeros((2, N)), np.zeros((6, N + 1)), np.zeros((2, N)),
                        False, t_solve)
        X, U, S = out
        U, moved = self._project_limits(U, u_prev)
        ok = ok and moved <= LIMIT_TOL
        if ok:
            self._guess = (X, U, S)
        return Plan(U, X, S, ok, t_solve)
