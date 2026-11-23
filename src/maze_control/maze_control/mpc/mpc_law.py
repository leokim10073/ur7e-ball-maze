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

MPCController wraps the CFTOC with the LQR's estimation (lqr_law.LQRController):
theta is not measured, so an open-loop observer runs the lag model on the
commands sent, and the state is predicted ``delay`` ahead through the commands
already in flight. When a solve fails or runs past ``solve_budget``, it plays
the next step of the last good plan; after ``max_fallback`` failures in a row
(or with no plan) it levels the board, except that a plan which was solved but
late is used rather than levelling (mode 'late'): it was computed for the
current state, and levelling under a slow but solvable problem never recovers.
"""

import collections
import dataclasses
import time

import casadi as ca
import numpy as np

from ..lqr.lqr_law import G, ball_model, c2d

# Infinite corridor bounds are replaced by this (m): far outside any board.
BIG = 1.0e3

# A plan whose inputs break a limit by more than this (rad) is a failed solve;
# smaller breaks (OSQP's accuracy, ~1e-5 when a corridor slack is active) are
# projected back onto the limits.
LIMIT_TOL = 1.0e-4

# OSQP's status does not come out of opti.to_function (stats() is empty), so a
# plan whose dynamics residual |X_0 - x0|, |X_{k+1} - Ad X_k - Bd U_k| exceeds
# this is taken as unconverged (a max_iter exit). Converged plans: ~1e-5.
RESID_TOL = 5.0e-4

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
        self._Ad2 = Ad2 = np.kron(np.eye(2), np.asarray(Ad, float))
        self._Bd2 = Bd2 = np.kron(np.eye(2), np.asarray(Bd, float))

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

    def clear_guess(self):
        """Drop the warm start (a new run starts from zeros)."""
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
        x0 = args[0]
        resid = max(np.abs(X[:, 0] - x0).max(),
                    np.abs(X[:, 1:] - self._Ad2 @ X[:, :-1] - self._Bd2 @ U).max())
        U, moved = self._project_limits(U, u_prev)
        ok = ok and resid <= RESID_TOL and moved <= LIMIT_TOL
        if ok:
            self._guess = (X, U, S)
        return Plan(U, X, S, ok, t_solve)


class MPCController:
    """Same interface as LQRController (step / step_zero / reset), except that
    step takes a HorizonRef instead of a single target point."""

    def __init__(self, cftoc, Ad, Bd, delay, solve_budget, max_fallback):
        self.cftoc = cftoc
        self.N, self.dt = cftoc.N, cftoc.dt
        self.max_tilt, self.max_rate = cftoc.max_tilt, cftoc.max_rate
        self.Ad, self.Bd = np.asarray(Ad, float), np.asarray(Bd, float)
        self.n_delay = int(round(float(delay) / self.dt))
        self.solve_budget = float(solve_budget)
        self.max_fallback = int(max_fallback)
        self.lag = float(self.Ad[2, 2])         # lag pole, for the observer
        self.u = np.zeros(2)                    # last output (tilt_x, tilt_y)
        self.theta = np.zeros(2)                # estimated actual tilt per axis
        # commands sent but not yet acting on the board, oldest first
        self._pending = collections.deque([np.zeros(2)] * self.n_delay)
        self.mode = 'level'
        self.n_fail = 0
        self.last_plan = None
        self.solve_times = collections.deque(maxlen=600)   # last 10 s at 60 Hz

    @classmethod
    def from_params(cls, model, mpc, limits, rate_hz, g=G):
        dt = 1.0 / float(rate_hz)
        Ad, Bd = c2d(*ball_model(model['c'], model['tau'], g), dt)
        cftoc = CFTOC(Ad, Bd, dt, mpc['horizon'], mpc['q_pos'], mpc['q_vel'],
                      mpc['r'], mpc['r_du'], mpc['rho_lin'], mpc['rho_quad'],
                      limits['max_tilt'], limits['max_rate'],
                      osqp_opts=dict(mpc.get('osqp', {})))
        return cls(cftoc, Ad, Bd, model['delay'], mpc['solve_budget'],
                   mpc['max_fallback'])

    def reset(self):
        """New run: forget the stored plan, the failures and the warm start
        (the output and the observer keep going)."""
        self.last_plan = None
        self.n_fail = 0
        self.cftoc.clear_guess()

    # -------------------------------------------------------------- internals
    # _limit / _advance / _predict follow LQRController's (lqr_law.py).
    def _limit(self, u_des, dt):
        u_sat = np.clip(u_des, -self.max_tilt, self.max_tilt)
        step = self.max_rate * max(dt, 0.0)
        return self.u + np.clip(u_sat - self.u, -step, step)

    def _advance(self, u_out):
        if self.n_delay > 0:
            self._pending.append(u_out.copy())
            acting = self._pending.popleft()
        else:
            acting = u_out
        self.theta = self.lag * self.theta + (1.0 - self.lag) * acting
        self.u = u_out

    def _predict(self, pos, vel):
        """Stacked state [px, vx, thx, py, vy, thy] n_delay steps ahead."""
        x = np.vstack([pos, vel, self.theta])
        for u in self._pending:
            x = self.Ad @ x + self.Bd @ u.reshape(1, 2)
        return np.concatenate([x[:, 0], x[:, 1]])

    def _fallback(self):
        """Desired command after a failed solve."""
        self.n_fail += 1
        plan = self.last_plan
        if plan is None or self.n_fail > self.max_fallback \
                or self.n_fail >= plan.U.shape[1]:
            self.mode = 'level'
            return np.zeros(2)
        self.mode = 'fallback'
        return plan.U[:, self.n_fail]

    # -------------------------------------------------------------- interface
    def step(self, ref, pos, vel, dt):
        """One control step. Returns np.array([tilt_x, tilt_y]) in rad."""
        pos, vel = np.asarray(pos, float), np.asarray(vel, float)
        if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(vel))):
            u_des = self._fallback()
        else:
            plan = self.cftoc.solve(self._predict(pos, vel), ref, self.u)
            self.solve_times.append(plan.t_solve)
            if plan.ok and plan.t_solve <= self.solve_budget:
                self.last_plan, self.n_fail, self.mode = plan, 0, 'ok'
                u_des = plan.U[:, 0]
            else:
                u_des = self._fallback()
                if self.mode == 'level' and plan.ok:
                    self.last_plan, self.n_fail, self.mode = plan, 0, 'late'
                    u_des = plan.U[:, 0]
        u_out = self._limit(u_des, dt)
        self._advance(u_out)
        return u_out.copy()

    def step_zero(self, dt):
        """Ramp the tilt back to zero (rate limited) and start over."""
        self.reset()
        self.mode = 'level'
        u_out = self._limit(np.zeros(2), dt)
        self._advance(u_out)
        return u_out.copy()
