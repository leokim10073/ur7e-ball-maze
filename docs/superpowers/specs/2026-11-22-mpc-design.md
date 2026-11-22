# C5: Constrained tracking MPC (design)

Issue #10. A model predictive controller that tilts the board so the ball follows the planned path while the *predicted* ball centre stays inside the planner's safe corridors (Contract 3, `docs/interfaces.md`). It is the third controller in the proposal's comparison (PID, LQR, constrained MPC, MPCC): the comparison tests whether wall constraints give a higher completion rate and fewer wall impacts than PID and LQR.

## Goals

- Solve a constrained finite-time optimal control problem (CFTOC) every control period (60 Hz) with CasADi.
- Hard limits on tilt and tilt rate, the same values as PID/LQR (below arm control's limits, Contract 4).
- Corridor constraints on the predicted ball centre, taken from `PlannedRoute.corridors`.
- Warm start, and a fallback when the solver fails or runs late.
- A ROS 2 node, launch file and config with the same interface as PID and LQR.

**Done when** (issue #10): MPC reaches the goal in ≥ 80 % of runs, with a higher completion rate and fewer wall impacts than PID and LQR (#8). This design covers the controller; the 50-run comparison is #8.

## Non-goals

- No progress reward or time-optimal objective (that is MPCC, #11).
- No wall-impact model (#12).
- No nonlinear model: the plant is the LQR's linear model.

## Constraints

- Python, in `src/maze_control`, next to `pid/` and `lqr/`. The control law has no ROS dependency, so tests run offline against `common/ball_sim.py`.
- Solver: CasADi (the user's choice). `casadi==3.6.7` is pinned in `docker/Containerfile`. It runs on the container's numpy 1.21.5 (checked) and its wheel includes the OSQP and qpOASES plugins.
- Budget: one solve must finish well inside 16.7 ms.

## Approaches

| | Approach | Trade-off |
|---|---|---|
| **Picked** | `casadi.Opti('conic')` → `qpsol` with the **OSQP** backend, wrapped once with `opti.to_function`, warm-started with the previous solution | Sparse, so it scales with the horizon. Measured in the container (N = 30, both axes, corridor slacks): median 3.4 ms, worst 11.6 ms (first call). |
| Rejected | qpOASES backend | Dense active-set solver. Same problem: median 33 ms, worst 1.3 s. Too slow at 60 Hz, so it is not offered as a config option. |
| Rejected | `nlpsol('ipopt')` | An interior-point NLP solver used on a QP. Slow; it is kept for MPCC's nonlinear model. |
| Rejected (earlier) | Hand-written ADMM in numpy | No dependency, but it would re-implement OSQP. |

## Model (reused from LQR)

Each axis is the model in `lqr/lqr_law.py`: state `[p, v, θ]`, `ball_model(c, tau)` discretized by `c2d` at `dt = 1/rate_hz`. The two axes stack into a block-diagonal system: state `x ∈ R⁶ = [px, vx, θx, py, vy, θy]`, input `u ∈ R² = [tilt_x, tilt_y]`.

- θ is not measured. It is estimated by running the lag model on the commands actually sent, the LQR's open-loop observer.
- Dead time: the commands in flight (`delay / dt` of them) are applied to the measured state through the model. The prediction gives `x0`, the state at the time the first new command starts acting. This is the LQR's `_predict`.

## CFTOC (solved each period)

Decision variables: `X` (6 × N+1), `U` (2 × N), corridor slacks `S` (2 × N) ≥ 0.

Parameters: `x0` (6), the position reference `R` (2 × N), the velocity reference `V` (2 × N), the corridor bounds `lo`, `hi` (2 × N), and the previously applied command `u_prev` (2).

```
min   Σ_{k=0}^{N-1} [ q_pos ‖p_{k+1} − R_k‖² + q_vel ‖v_{k+1} − V_k‖² + r ‖u_k‖² + r_du ‖u_k − u_{k−1}‖²
                      + ρ₁ Σ S_k + ρ₂ ‖S_k‖² ]
s.t.  X_0 = x0
      X_{k+1} = Ad X_k + Bd U_k
      −max_tilt ≤ U_k ≤ max_tilt
      −max_rate·dt ≤ U_k − U_{k−1} ≤ max_rate·dt      (U_{−1} = u_prev)
      lo_k − S_k ≤ p_{k+1} ≤ hi_k + S_k,   S_k ≥ 0
```

`p` is `(px, py)` and `v` is `(vx, vy)`. The corridor constraints are soft. Large ρ₁ (exact penalty) and ρ₂ keep the QP feasible when a noisy state puts the ball just outside a corridor. The applied command is `U_0`, passed through the same clip and rate limit as PID/LQR as a guard. The observer and the in-flight queue record the applied command, as in the LQR.

## Reference and corridors along the horizon

- **Progress.** `PathTracker` keeps its windowed projection, so a U-turn cannot jump to the other leg. Its progress `s0` is the ball's arc length on the path.
- **Reference.** `s_k = s0 + v_ref·(k+1)·dt`, clamped to the path length, so the reference stops at the goal. `R_k = point_at(s_k)`, and `V_k = v_ref_k · tangent(s_k)`. Within `slow_dist` of the goal, `v_ref_k` drops linearly to 0, so the ball arrives at rest.
- **Corridors.** At step k, the corridor of the path pose nearest `s_k`, from `pose_corridor` and `arc_length`.
- **Without corridors** (a path with no route, or an empty `corridors`), the bounds are ±∞ and the problem is plain tracking.
- **Reached.** Inside `goal_tol` of the goal the reference is the goal itself and `V = 0`, the same as the tracker's `reached`.

## Warm start and fallback

- **Warm start.** Before each solve, the previous `X, U, S` are shifted one step (the last column repeated) and passed to the solve as its initial guess. OSQP warm starts through CasADi's `warm_start_primal`/`warm_start_dual` options. OSQP's own `warm_start` is rejected by CasADi.
- **Failure.** The solver returns not-solved (`return_status` not success), or the solve takes longer than `solve_budget` (wall time, a config value). The controller then applies the next step of the last good plan, `U_{j+1}` from the plan made j periods ago.
- **Consecutive failures.** After `max_fallback` failures in a row, or with no stored plan, it calls `step_zero` (levels the board, rate limited). The fallback state is logged.
- **Solver limits.** OSQP's `max_iter` and `time_limit` are bounded so a solve cannot block the timer.

## Code layout

| File | What |
|---|---|
| `maze_control/mpc/mpc_law.py` | `MPCController`: builds the Opti problem once (`to_function`), plus the observer, delay prediction, warm start and fallback. Same interface as `LQRController` (`step`, `step_zero`, `reset`, `from_params`), except `step` takes the horizon references. |
| `maze_control/mpc/controller.py` | `MpcPathController(PidPathController)`: run logic and watchdog unchanged. Adds `on_route(points, arc_length, corridors, pose_corridor, plan_id)` and builds `R, V, lo, hi` from the tracker. |
| `maze_control/mpc/node.py` | `MpcNode(PidNode)`: subscribes to `/maze/route` (`PlannedRoute`, latched QoS) instead of `/maze/path`. Logs the solve time and the fallback count. |
| `config/mpc.yaml` | Robot config. Same keys as `lqr.yaml` plus `mpc: {horizon, q_pos, q_vel, r, r_du, rho_lin, rho_quad, v_ref, slow_dist, solve_budget, max_fallback, osqp: {...}}`. |
| `config/mpc_sim.yaml` | Only if the sim needs different values (repo rule). |
| `launch/mpc.launch.py`, `setup.py` entry point `mpc_node` | Same as LQR. |
| `test/test_mpc.py` | Offline tests against `ball_sim`. |
| `docker/Containerfile` | `casadi==3.6.7`. |
| `package.xml` | `casadi` noted as a pip dependency (no rosdep key). |

A new run (`on_path` / a changed `plan_id`) clears the stored plan and the fallback counter, as well as the integrator state.

## Tests (offline, `ball_sim`)

- **CFTOC.** The solution satisfies the tilt and rate limits. At rest on the reference, U ≈ 0. With no active constraints, a long horizon matches the LQR step (`q_int = 0`) within tolerance.
- **Corridor.** In a straight corridor narrower than the unconstrained overshoot, MPC keeps the ball inside, and LQR with the same weights does not.
- **Fallback.** A forced solver failure applies the shifted plan, then levels the board after `max_fallback` failures.
- **Closed loop.** On the L-path and U-turn scenarios in `common/scenarios.py`, with the configured delay and noise, the ball reaches the goal.
- **Timing.** The median solve is under a few ms at the configured N. The test asserts a loose bound, so CI stays stable.
- **Config.** `mpc.yaml` loads through `from_params`, and its limits do not exceed Contract 4.

## Left to the builder

- Initial weights, from LQR's: `q_pos` 2500, `q_vel` 100, `r` 156, plus `r_du` and `ρ` chosen so a slack of 1 mm costs more than any tracking gain. Also `v_ref` (start near 0.1 m/s) and `slow_dist`. Tune offline, then in the sim.
- How to pull the solver status out of `to_function` (e.g. `opti.stats()` vs a qpsol function built directly). Any way that detects failure without raising works.
- Whether `step` takes the references as arrays or a small dataclass.
