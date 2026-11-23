# C5: MPC reference that slows before corners (design)

Follow-up to `2026-11-22-mpc-design.md`. Small change.

## Why

The MPC's horizon reference moves along the path at one constant `v_ref`. In the offline comparison (±5 mm corridors) `v_ref = 0.1 m/s` stays in the corridor in 100/92/98 % of randomized runs (line/L/U) but is ~12 % slower than the LQR; `v_ref = 0.15` is ~30 % faster but stays in only 100/40/20 %. All the loss is at corners: the tilt (0.08 rad, ~0.56 m/s²) and tilt-rate limits cannot turn a fast ball inside the corridor, and the 50 ms dead time costs more at speed. A reference that is fast on straights and slow at corners should get most of both.

## Goals

- `v_ref` becomes the straight-line speed; the reference slows to `v_turn` at a 90° (or sharper) turn, and changes speed at most at `a_ref`, so it brakes before the corner and speeds up after it.
- Robust to the planner's paths (Contract 3, PL3): points every 5 mm, straight segments, staircases kept in narrow places. A staircase is not a corner.
- Off switch: `v_turn = v_ref` gives exactly the current reference; existing tests unchanged.

## Non-goals

- No change to the QP, constraints, fallback or node.
- No time-optimal or progress-maximising reference (MPCC, #11).
- The goal approach keeps the existing linear `slow_dist` ramp.

## Approaches

| | Approach | Trade-off |
|---|---|---|
| **Picked** | Speed profile from the turn angle between the chords over a window `corner_window` before and after each point, then a backward (brake) and forward (accelerate) pass with `a_ref` | Cheap (once per path), handles dense and staircase paths |
| Rejected | Turn angle at each vertex | A 5 mm staircase reads as a chain of 45° corners |
| Rejected | Time-optimal profile / MPCC | Much more work; that is #11 |

## Design (`mpc/controller.py`, `MpcPathController`)

- On every path change, build `self._profile = (s_grid, v)` on a 1 mm arc grid:
  - turn angle φ(s) = angle between chord `p(s) − p(s − w)` and chord `p(s + w) − p(s)` (`w = corner_window`, clamped to the path; a zero chord gives φ = 0);
  - cap `v_cap = v_turn + (v_ref − v_turn) · max(cos φ, 0)`;
  - backward pass `v[i] = min(v_cap[i], √(v[i+1]² + 2·a_ref·ds))`, then forward pass `v[i] = min(v[i], √(v[i−1]² + 2·a_ref·ds))`; both ends start from `v_cap`.
- `_speed(s) = interp(s, profile) · clip((L − s)/slow_dist, 0, 1)`.
- Off (`v_turn ≥ v_ref`): the profile is the constant `v_ref`.
- `config/mpc.yaml` `mpc:` gains `v_turn`, `a_ref`, `corner_window`; `from_params` reads them with the off defaults when absent.

## Defaults

Chosen by a sweep in `scripts/compare_controllers.py` (randomized runs, ±5 mm corridors): goal is time close to `v_ref = 0.15` with the in-corridor rate of `v_ref = 0.1`.

## Tests

Straight path runs at `v_ref`; at a 90° corner the speed is `v_turn`; braking before and accelerating after are bounded by `a_ref`; a 5 mm staircase along a straight line stays near `v_ref`; `v_turn = v_ref` keeps the constant reference; the horizon reference's velocity follows the profile.
