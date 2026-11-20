# Planning PL4: planner ROS node with replanning

Work package: [#16 · Planning](https://github.com/leokim10073/ur7e-ball-maze/issues/16), task PL4. Builds on PL1+PL2 (`docs/superpowers/specs/2026-11-19-planning-grid-path-design.md`) and PL3 (`docs/superpowers/specs/2026-11-19-planning-route-corridors-design.md`).

## Why

PL1–PL3 are a ROS-free library. Ball control (PID #11, MPC, MPCC) needs Contract 3 (`docs/interfaces.md`) on the wire: a route on `/maze/route` and `/maze/path`, published when the map and the ball are known, and again whenever the ball has left the route. Without this node nothing downstream can run on a planned path.

## Goals

- A C++ node `planner` in `maze_planning` that subscribes to `/maze/map` (Contract 1) and `/ball/state` (Contract 2) and publishes `/maze/route` (`maze_interfaces/PlannedRoute`) and `/maze/path` (`nav_msgs/Path`) per Contract 3.
- Plan once when a map and a usable ball state are both available; replan when the ball has left the route, when a new map arrives, and (after `STATUS_NO_PATH`) periodically until a path exists.
- Replanning from wherever the ball is must work, including a ball pushed into a spot whose nearest safe cell by straight line is behind a wall (the PL2 lead-in fix).
- Every threshold in a YAML file; no magic numbers in code.

## Non-goals

- Velocity profile, Bézier/spline smoothing (decided: revisit after PL5), merging identical consecutive corridors (decided: not worth it).
- Detecting that the ball reached the goal, stopping a run, or reacting to lost tracking (`tracking_timeout` is the controllers' rule in Contract 2).
- Tracking the ball's progress along the route (controllers do this; see Known risks).

## Constraints

- C++17, rclcpp, in the existing `maze_planning` ament_cmake package. The algorithm stays in the ROS-free `maze_planning_core` library; the node only converts messages and decides when to plan.
- Topics, types, frames and QoS exactly as `docs/interfaces.md`: `/maze/map` latched (reliable, transient local, depth 1); `/ball/state` best effort, keep last 1; `/maze/route` latched; `/maze/path` latched too (a latched publisher also serves volatile and best-effort subscribers, so `pid_node` can subscribe either way). Frame `maze_board`, SI units, `use_sim_time` from the launch file.
- Build and test inside the `maze-sim` distrobox (ROS 2 Humble).

## Approaches (replan trigger)

1. **Outside every corridor of the current route (chosen).** The ball counts as on the route while it is inside any corridor, inflated by `outside_margin`. Simplest, and it asks exactly "is the ball still in safe space on the route?".
2. Only the corridors near the ball's progress. Stricter (also catches a ball rolling far backwards), but the planner would keep its own progress state, which the controllers already keep (MPCC's arc length).
3. Replan on a timer regardless of the ball. Simple, but every replan resets the controllers' progress (`plan_id` changes), so needless replans hurt tracking.

## Design

### Library changes (`maze_planning_core`)

**Lead-in by shortest walkable distance (PL2 change).** When the start cell is blocked but not a wall, `plan_grid_path` finds the nearest allowed cell by shortest path through passable cells: Dijkstra on 8 neighbours with step lengths 1 and √2, a diagonal step allowed only when neither side cell is impassable (this is the "BFS" discussed; Dijkstra because diagonal steps are longer). Figure: `docs/figures/planning-pl4/bfs_lead_in.png`.

Passable = not a wall or unknown cell, and its center not inside any hole (distance to the hole center < `hole.radius`; the `hole_margin` keep-out is not used here). *My call:* a pushed ball must never be routed over a trap hole, but it may be pushed into a hole's margin ring and has to be able to leave it; using the full keep-out would strand it there with `STATUS_NO_PATH`. A start cell inside a hole is treated like a wall start (`found = false`). *Final-review fix:* the lead-in stops only at an allowed cell from which A\* reaches the goal (flood fill from the goal), so it also runs for a ball on allowed cells cut off by a too-narrow gap; and a step may enter a cell with clearance below `ball_radius − resolution/2` only if that is not tighter than the cell it leaves, so the ball never squeezes into a gap it cannot pass. `GridPath.lead_in` counts the lead-in cells. The lead-in is that path's cells (start cell first, the allowed cell last, which A\* then starts from). No allowed cell reachable → `found = false`. This replaces the straight-line nearest cell and its "line must not cross a wall" check. A start cell that is a wall, or outside the grid, still gives `found = false`.

**Lead-in corridors (PL3 change).** The lead-in is no longer one straight segment. In `build_route`:

- Lead-in points go from the exact start through the lead-in cells' centers to the first allowed cell's center, sampled every `point_spacing` like the rest.
- Lead-in corridors: walk the lead-in cells (plus the first allowed cell) and group consecutive cells greedily: a group grows while the box of its cells contains only passable cells (as above: no wall, unknown or hole cell). Each group's corridor spans its cells' full extent (centers ± resolution/2), not grown, and the first one also takes in the exact start. Consecutive groups share their boundary cell, so they overlap. These corridors may contain blocked-but-not-wall cells: they are the recovery region, as in PL3.
- A single 8-step without corner cutting through an impassable cell always has a passable box, so grouping always succeeds.

**Replan monitor (new, ROS-free).** A class `ReplanMonitor` in the core library holds the decision logic so it can be unit-tested without ROS:

```cpp
struct ReplanParams { double outside_margin = 0.003; int outside_frames = 3; double no_path_retry_period = 0.5; };
class ReplanMonitor {
 public:
  explicit ReplanMonitor(const ReplanParams&);          // validates: margin >= 0, frames >= 1, period > 0
  void route_planned(const Route& route, double now);   // route.points empty = STATUS_NO_PATH
  bool should_replan(double x, double y, double now);   // one call per usable ball state
};
```

- Ball on the route = inside at least one corridor inflated by `outside_margin` on each side. Inside → the outside count resets to 0. Outside → count + 1; at `outside_frames` → `should_replan` returns true (the node replans and calls `route_planned`, which resets the count).
- After a `STATUS_NO_PATH` result: `should_replan` returns true once `now − last attempt ≥ no_path_retry_period`, whatever the ball's position.
- Comment at the inside check (user's request): *Known risk: only "outside every corridor" triggers a replan. A ball that rolls backwards into an earlier corridor of the same route is still "inside" and does not replan; controllers own progress. If this causes bugs, switch to checking only corridors near the ball's progress.*

### Node (`planner`, `src/planner_node.cpp`)

- **Time:** "now" is the node's clock (`use_sim_time` in the sim), used for `header.stamp` and the no-path retry period; the ball state's `header.stamp` is not used.
- **On `/maze/map`:** convert `walls` (OccupancyGrid → `Grid`: width, height, resolution, origin x/y, data), `holes` and `goal` to library types; keep it. A map whose `walls` fail `validate(Grid)` is logged and ignored (the previous map stays). If a usable ball state is known, plan now (a new map always replans).
- **On `/ball/state`:** ignore it when `tracking_ok` is false (the position is only a prediction): it neither plans nor counts toward the outside count. Otherwise remember it; if a map is known and nothing was planned yet, plan; else ask the monitor and plan if it says so.
- **Plan:** `plan_grid_path` from the ball's (x, y) to the map's goal with the map's holes, then `build_route`. `plan_id` = previous + 1 (first plan 1). Publish `/maze/route` with `header.stamp` = now, `frame_id` = `maze_board`, `status`, `path` (poses at z = 0, orientation w = 1, `path.header` same as the route's), `arc_length`, `corridors`, `pose_corridor`; and the same `path` on `/maze/path`. No path → `STATUS_NO_PATH` with all arrays empty.
- **No-path retries:** a retry that is still no path is not published (`plan_id` stays), so controllers do not see a new plan every 0.5 s. A retry that finds a path publishes normally.
- Planner exceptions (`std::invalid_argument` from bad params at startup) stop the node with a clear log message; a bad map at runtime is logged and skipped, as above.

### Config and launch

`src/maze_planning/config/planner.yaml`, every key required (the node declares parameters without usable defaults and fails at startup if one is missing):

```yaml
planner:
  ros__parameters:
    ball_radius: 0.0125          # 25 mm marble (#19)
    wall_margin: 0.003
    hole_margin: 0.003
    preferred_clearance: 0.020
    clearance_weight: 2.0
    point_spacing: 0.005
    replan:
      outside_margin: 0.003      # ball must be this far outside every corridor ...
      outside_frames: 3          # ... for this many consecutive usable ball states (~0.05 s at 60 Hz)
      no_path_retry_period: 0.5  # s, between retries after STATUS_NO_PATH
```

`launch/planner.launch.py`: arguments `use_sim_time` (default true, like `cad_map.launch.py`) and `params_file` (default the installed `planner.yaml`). The config and launch directories are installed.

### Contract update (`docs/interfaces.md`, Contract 3)

- Lead-in corridors: when the ball starts too close to a wall, the first corridors cover the ball's way out to safe space at whole-cell size, and they are not reduced by the wall clearance (the ball is already inside it); every later corridor is.
- Replan rule in numbers: the ball is outside every corridor by more than `replan.outside_margin` for `replan.outside_frames` consecutive states with `tracking_ok`; a new map also replans; after `STATUS_NO_PATH` the planner retries every `replan.no_path_retry_period` and publishes only when a path is found.
- The known risk above, one line.

## Tests

- **Library (gtest):**
  - Dead end (the layout of `docs/figures/planning-pl4/bfs_lead_in.png`, default params): a 34 × 30 grid (width × height), walled border, wall cells in column 14 rows 0–16 (the divider) and column 7 rows 0–16, so columns 8–13 form a 30 mm-wide dead end open at the top; with the default 15.5 mm clearance every cell of the dead end is blocked. Start = center of cell (row 3, col 13), against the divider. The straight-line nearest allowed cell is across the divider at (4, 18); before the change `plan_grid_path` returns `found = false` (the line hits the divider). After: `found = true`, no lead-in cell is a wall, every lead-in cell has column < 14 (it leaves through the top of the dead end), and the path reaches the goal. Goal: center of cell (24, 20).
  - Hole: a free room with a hole whose center is between a pushed start and the nearest allowed cell → the lead-in never has a cell whose center is inside the hole.
  - The existing `StartAgainstWallLeavesIt` and L1 tests still pass; `L1Properties` skips every lead-in corridor (not only corridor 0) when checking that corridors hold only allowed cells.
  - `build_route` with a bent lead-in (the dead end above): every lead-in corridor contains only passable cells, the start is inside corridor 0, consecutive corridors overlap, every point is inside its `pose_corridor`.
- **ReplanMonitor (gtest):** inside → never replans; outside by less than the margin → never; outside by more for `outside_frames − 1` states then inside → no replan and the count resets; `outside_frames` in a row → replans once, then the count restarts after `route_planned`; after a no-path route, replans only once the retry period has passed; bad params throw.
- **Node (gtest with rclcpp, in-process):** publish the L1 map (from the test fixture) and a ball state at the start → one `/maze/route` with `plan_id` 1, `STATUS_OK`, poses = `/maze/path` poses; ball states outside every corridor for 3 frames → `plan_id` 2; the same with `tracking_ok` false → no new plan; a ball state before any map → nothing published.

## Known risks

- Backwards roll is not detected (Approach 1): see the comment above. Owner of the fix if it bites: planning, by switching to approach 2.
- A ball resting just outside a narrow corridor edge (one-row corridors in tight passages are 0–5 mm wide) can trigger replans more often; `outside_margin` is the knob.

## Left to the builder

- File split inside `maze_planning` (e.g. `replan_monitor.hpp/.cpp`, `planner_node.cpp`, a small `ros_convert` helper), the node test's spinning and waiting mechanics, how required parameters are declared.
