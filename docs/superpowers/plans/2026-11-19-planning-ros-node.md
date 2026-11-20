# Planning PL4: Planner ROS Node Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put Contract 3 on the wire: a C++ node `planner` that plans from `/maze/map` and `/ball/state` and publishes `/maze/route` and `/maze/path`, replanning when the ball leaves every corridor, when a new map arrives, and after `STATUS_NO_PATH`.

**Architecture:** Two library changes in the ROS-free `maze_planning_core` (Dijkstra lead-in, lead-in corridors), a ROS-free `ReplanMonitor` holding the replan decision, then a thin rclcpp node that converts messages, asks the monitor and publishes. Config in YAML, launch file next to it.

**Tech Stack:** C++17, ament_cmake, rclcpp (ROS 2 Humble), maze_interfaces, nav_msgs, ament_cmake_gtest, yaml-cpp (tests).

**Spec:** `docs/superpowers/specs/2026-11-19-planning-ros-node-design.md`

## Global Constraints

- Everything in `docs/superpowers/plans/2026-11-19-planning-grid-path.md` → Global Constraints still holds (grid format, SI units, no ROS in `maze_planning_core`, compile flags, `[planning]` commit prefix, no attribution), except the build command below.
- **BUILD+TEST** (the node needs `maze_interfaces` built): `distrobox enter maze-sim -- bash -c "source /opt/ros/humble/setup.bash && colcon build --packages-up-to maze_planning && source install/setup.bash && colcon test --packages-select maze_planning --event-handlers console_direct+ && colcon test-result --verbose"`, run with the sandbox disabled.
- Passable cell = not `is_obstacle` (wall or unknown) and its center not inside any hole (distance to the hole center < `hole.radius`; `hole_margin` not used).
- QoS: `/maze/map` reliable + transient local + depth 1; `/ball/state` best effort, keep last 1; `/maze/route` and `/maze/path` reliable + transient local + depth 1. Frame `maze_board`.
- YAML keys (node name `planner`, all required, no usable defaults in code): `ball_radius 0.0125`, `wall_margin 0.003`, `hole_margin 0.003`, `preferred_clearance 0.020`, `clearance_weight 2.0`, `point_spacing 0.005`, `replan.outside_margin 0.003`, `replan.outside_frames 3`, `replan.no_path_retry_period 0.5`.
- "now" = the node's clock (`use_sim_time` from the launch file).
- After code changes, run `graphify update .` before committing.

## Review Focus

- A params file missing one key (e.g. `replan.outside_frames`): the node fails at startup with a message naming the key, never runs on a default → Task 4 test `RequiredParamMissingThrows`.
- A malformed map at runtime (data size ≠ width × height): logged and ignored, the previous map stays and nothing crashes → Task 4 test `BadMapIgnored`.
- A ball state outside the grid or with a non-finite x/y: outside the grid gives `STATUS_NO_PATH`; non-finite is ignored like `tracking_ok = false` → Task 4 tests `BallOutsideGridGivesNoPath`, `NonFiniteBallIgnored`.
- A ball state before the map: nothing published until the map arrives, then one plan at once → Task 4 test `BallBeforeMapPlansWhenMapArrives`.
- `STATUS_NO_PATH` followed by many ball states: retries are not republished, `plan_id` stays → Task 3 monitor test plus Task 4 `BallOutsideGridGivesNoPath` (no second message within 1.2 s).

---

### Task 1: Lead-in by shortest walkable distance

**Files:**
- Modify: `src/maze_planning/src/geometry.hpp`, `src/maze_planning/src/grid_planner.cpp` (replace the straight-line nearest cell), `src/maze_planning/include/maze_planning/grid_planner.hpp` (doc comment)
- Test: `src/maze_planning/test/test_grid_planner.cpp`, `src/maze_planning/test/grids.hpp`

**Interfaces:**
- Produces (`geometry.hpp`, `maze_planning::detail`): `std::vector<uint8_t> passable_cells(const Grid&, const std::vector<Circle>& holes)` — 1 = passable per Global Constraints.
- Produces (`grids.hpp`): `Grid dead_end()` — the spec's 34 × 30 layout (walled border, wall cells col 14 rows 0–16 and col 7 rows 0–16).
- `plan_grid_path` signature unchanged.

- [ ] **Step 1: Write the failing tests** (`GridPlannerTest`)
  - `DeadEndLeadInLeavesThroughTop`: `dead_end()`, default params, start = `cell_center(g, 3, 13)`, goal `goal_at(g, 24, 20)`. `ASSERT_TRUE(found)`; `cells.back() == {24, 20}`; for every leading cell that is blocked (`blocked_cells`): not `is_obstacle` and column < 14.
  - `LeadInAvoidsHole`: `free_grid(40, 21)` + `wall_border`, default params; start = `cell_center(g, 1, 10)` (pushed against the bottom wall); hole `{cell_center(g,2,10) x, y, 0.004}` — between the start and the straight-line nearest allowed cell (row 4, col 10). `found`; no lead-in cell's center has distance < 0.004 to the hole center.
  - `StartInHoleFails`: same room, start = `cell_center(g, 1, 10)`, hole centered there with radius 0.004 → `found == false`.
  - Existing `StartAgainstWallLeavesIt` and `StartInWallFails` unchanged.
- [ ] **Step 2: Run to verify they fail** — BUILD+TEST. Expected: `DeadEndLeadInLeavesThroughTop` fails (`found == false`), `LeadInAvoidsHole` fails (a lead-in cell inside the hole), `StartInHoleFails` fails (`found == true`).
- [ ] **Step 3: Implement** `passable_cells`; in `plan_grid_path`, when the start cell is blocked: return not found if the start cell is not passable; else Dijkstra from it over passable cells, 8 neighbours, step 1 / √2, diagonal only when both side cells are passable; stop at the first popped cell with `blocked == 0`; lead-in = parent chain start → that cell, minus that cell (A\* starts from it). None reached → not found. Update the header comment.
- [ ] **Step 4: Run to verify they pass** — BUILD+TEST. Expected: every suite passes (42 earlier tests included).
- [ ] **Step 5: Commit** `[planning] lead a pushed ball out by the shortest walkable way, never over a hole`

### Task 2: Lead-in corridors

**Files:**
- Modify: `src/maze_planning/include/maze_planning/route.hpp`, `src/maze_planning/src/route.cpp`
- Test: `src/maze_planning/test/test_route.cpp`

**Interfaces:**
- Consumes: `passable_cells` (Task 1), `dead_end()` (Task 1).
- Produces: `Route` gains `uint32_t lead_in_corridors = 0;` — how many leading corridors are lead-in corridors (not published; for tests and the node's logs).

- [ ] **Step 1: Write the failing tests** (`RouteTest`)
  - `BentLeadInCorridors`: `dead_end()`, start (3, 13) center, goal (24, 20): `lead_in_corridors >= 2`; every cell whose center lies inside a lead-in corridor (bounds ± 1e-9) is passable; `points.front() == start` and inside `corridors[0]`; consecutive corridors overlap; every point inside `corridors[pose_corridor[i]]` (± 1e-9); consecutive points ≤ `point_spacing` + 1e-12 apart.
  - `PushedStartInsideFirstCorridor`: also `EXPECT_EQ(lead_in_corridors, 1u)` (straight lead-in in the channel stays one group).
  - `L1Properties`: replace the "skip corridor 0" with "skip the first `lead_in_corridors`".
- [ ] **Step 2: Run to verify they fail** — BUILD+TEST. Expected: `lead_in_corridors` not a member (build error).
- [ ] **Step 3: Implement** in `build_route`: lead-in cells L = leading blocked cells plus the first allowed cell. Groups: from index i, grow j while the cell box of L[i..j] holds only passable cells; corridor = that box's cell centers ± resolution/2, the first one also including the exact start; the next group starts at j (shared cell). Lead-in points: exact start, then each L[k] center (k ≥ 1), sampled every `point_spacing`; a point's `pose_corridor` = the group of the segment it ends (shared cell → earlier group). The A\* part is unchanged and starts at the first allowed cell.
- [ ] **Step 4: Run to verify they pass** — BUILD+TEST. Expected: every suite passes.
- [ ] **Step 5: Commit** `[planning] lead-in corridors follow a bent way out in passable boxes`

### Task 3: ReplanMonitor

**Files:**
- Create: `src/maze_planning/include/maze_planning/replan_monitor.hpp`, `src/maze_planning/src/replan_monitor.cpp`
- Modify: `src/maze_planning/CMakeLists.txt` (core source, new test)
- Test: `src/maze_planning/test/test_replan_monitor.cpp`

**Interfaces:**
- Consumes: `Route`, `Corridor` (`route.hpp`).
- Produces: exactly the spec's `ReplanParams` and `ReplanMonitor` (`route_planned(const Route&, double now)`, `bool should_replan(double x, double y, double now)`). Before any `route_planned`, `should_replan` returns false. Throws `std::invalid_argument` unless `outside_margin` finite ≥ 0, `outside_frames` ≥ 1, `no_path_retry_period` finite > 0. The spec's known-risk comment sits at the inside check, verbatim.

- [ ] **Step 1: Write the failing tests** (`ReplanMonitorTest`, a route with corridors {0,0.1,0,0.02} and {0.08,0.1,0,0.2}, default params)
  - `InsideNeverReplans`: 10 calls at (0.05, 0.01) → all false.
  - `WithinMarginNeverReplans`: 10 calls at (0.05, 0.022) (2 mm out) → all false.
  - `ShortExcursionResets`: 2 calls at (0.05, 0.03), 1 inside, 2 outside → all false.
  - `ReplansAfterFramesThenRestarts`: 3 outside → false, false, true; `route_planned(same, t)`; next 2 outside false, 3rd true.
  - `NoPathRetriesAfterPeriod`: `route_planned(Route{}, 1.0)`; `should_replan(0.05, 0.01, 1.4)` false; at 1.5 true; `route_planned(Route{}, 1.5)`; at 1.6 false.
  - `NothingPlannedNeverReplans`: fresh monitor, 5 outside calls → false.
  - `RejectsBadParams`: margin −0.001, frames 0, period 0 → each throws.
- [ ] **Step 2: Run to verify they fail** — BUILD+TEST. Expected: `replan_monitor.hpp` not found.
- [ ] **Step 3: Implement** `replan_monitor.cpp` per the spec.
- [ ] **Step 4: Run to verify they pass** — BUILD+TEST. Expected: every suite passes.
- [ ] **Step 5: Commit** `[planning] decide replans: ball outside every corridor, or a no-path retry`

### Task 4: The planner node, config, launch and contract

**Files:**
- Create: `src/maze_planning/src/ros_convert.hpp`, `src/maze_planning/src/ros_convert.cpp`, `src/maze_planning/src/planner_node.hpp`, `src/maze_planning/src/planner_node.cpp`, `src/maze_planning/src/planner_main.cpp`, `src/maze_planning/config/planner.yaml`, `src/maze_planning/launch/planner.launch.py`
- Modify: `src/maze_planning/CMakeLists.txt`, `src/maze_planning/package.xml`, `docs/interfaces.md` (Contract 3)
- Test: `src/maze_planning/test/test_planner_node.cpp`

**Interfaces:**
- Consumes: `plan_grid_path`, `build_route` (Tasks 1–2), `ReplanMonitor` (Task 3), `validate(Grid)`.
- Produces (`ros_convert.hpp`, namespace `maze_planning`): `Grid to_grid(const nav_msgs::msg::OccupancyGrid&)`; `Circle to_circle(const maze_interfaces::msg::Circle&)`; `maze_interfaces::msg::PlannedRoute to_msg(const Route&, uint32_t plan_id, const rclcpp::Time& stamp)` (status `STATUS_NO_PATH` when `points` empty; poses z = 0, w = 1; frame `maze_board` on the route and its path).
- Produces (`planner_node.hpp`): `class PlannerNode : public rclcpp::Node { explicit PlannerNode(const rclcpp::NodeOptions& = {}); }` — node name `planner`; reads every Global Constraints key in the constructor (declared without a default; a missing one throws, naming the key).
- Build: a static `maze_planning_node` library (ros_convert + planner_node, not installed) linked by the `planner` executable and the test. `planner` installed to `lib/maze_planning`; `config/` and `launch/` installed to `share/maze_planning`. `package.xml` adds `depend` rclcpp, maze_interfaces, nav_msgs, geometry_msgs, std_msgs and `exec_depend` launch, launch_ros.

- [ ] **Step 1: Write the failing tests** (`PlannerNodeTest`; `rclcpp::init` once per suite; a test node with the documented QoS subscribes to `/maze/route` and `/maze/path` and publishes `/maze/map` and `/ball/state`; a `wait_for(pred, 3 s)` spins both nodes on one executor; node options set every key to the YAML values and `use_sim_time` false; the CMake test sets `ENV ROS_DOMAIN_ID=77 ROS_LOCALHOST_ONLY=1`). The map is the L1 fixture turned into a `MazeMap` (walls frame `maze_board`).
  - `PlansOnceWithMapAndBall`: map, ball at `fixture.start` → one route, `plan_id` 1, `STATUS_OK`, `header.frame_id == "maze_board"`, route path poses == `/maze/path` poses, every pose z 0 and w 1, `arc_length.size() == poses.size()`.
  - `ReplansWhenOutsideEveryCorridor`: after the first plan, 3 ball states at a point outside every corridor by > 3 mm but in free floor (pick the fixture's start shifted to a cell whose center is outside all corridors; compute it in the test from the received corridors) → a route with `plan_id` 2.
  - `TrackingLostDoesNotReplan`: the same 3 states with `tracking_ok` false → no `plan_id` 2 within 1 s.
  - `NewMapReplans`: after the first plan, republish the map → `plan_id` 2.
  - `BallBeforeMapPlansWhenMapArrives`: ball first, nothing within 0.5 s; then the map → `plan_id` 1.
  - `BallOutsideGridGivesNoPath`: map, ball at (1.0, 1.0) → `STATUS_NO_PATH`, empty path/corridors/arc_length/pose_corridor; 20 more states over 1.2 s → no second message.
  - `NonFiniteBallIgnored`: map, ball x = NaN → nothing within 0.5 s.
  - `BadMapIgnored`: first plan on L1; publish a map whose `walls.data` is one element short, then 3 outside states → `plan_id` 2 planned on L1 (route still reaches the L1 goal), node alive.
  - `RequiredParamMissingThrows`: options without `replan.outside_frames` → constructing `PlannerNode` throws.
  - `ConvertGridRoundTrip`: fixture grid → `OccupancyGrid` → `to_grid` equals the fixture grid field by field.
- [ ] **Step 2: Run to verify they fail** — BUILD+TEST. Expected: `planner_node.hpp` not found.
- [ ] **Step 3: Implement** `ros_convert`, `PlannerNode` (behaviour exactly as the spec's Node section; a ball state with `tracking_ok` false or non-finite x/y is ignored; an exception from a bad map is logged and the map dropped), `planner_main.cpp` (catches the constructor's exception, logs it, returns 1), `planner.yaml` with the spec's comments, `planner.launch.py` (args `use_sim_time` default true, `params_file` default the installed `planner.yaml`, module docstring with usage like `cad_map.launch.py`).
- [ ] **Step 4: Run to verify they pass** — BUILD+TEST. Expected: every suite passes.
- [ ] **Step 5: Smoke run** in the distrobox: `ros2 launch maze_cad_map cad_map.launch.py use_sim_time:=false` and `ros2 launch maze_planning planner.launch.py use_sim_time:=false`, publish one `/ball/state` at the L1 start with `ros2 topic pub --once`; Expected: `ros2 topic echo --once /maze/route` shows `plan_id: 1`, `status: 0`.
- [ ] **Step 6: Contract update** in `docs/interfaces.md` Contract 3, the spec's three bullets (lead-in corridors, replan rule in numbers, known risk); the **Rate** line points at the replan rule.
- [ ] **Step 7: Commit** `[planning] planner node: Contract 3 on /maze/route and /maze/path with replanning`
