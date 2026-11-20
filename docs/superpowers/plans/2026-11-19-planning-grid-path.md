# Planning PL1 + PL2: Grid Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A C++ library that turns a MazeMap grid into a start→goal cell path keeping the ball clear of walls and holes.

**Architecture:** New ament_cmake package `src/maze_planning` with a ROS-free library `maze_planning_core` (clearance map → blocked cells → A\*), tested with gtest on synthetic grids and on an L1 fixture exported from `maze_cad_map`. The ROS node is PL4 and not part of this plan.

**Tech Stack:** C++17, ament_cmake, ament_cmake_gtest, yaml-cpp (tests only), Python 3 for the fixture export script.

**Spec:** `docs/superpowers/specs/2026-11-19-planning-grid-path-design.md`

## Global Constraints

- Grid format is Contract 1 (`docs/interfaces.md`): 0 free, 100 wall, -1 unknown (treated as wall); `data[row * width + col]`; row 0 at the bottom (−y); origin = lower-left corner; cell centre `x = origin_x + (col + 0.5) * resolution`, same for y with row.
- SI units, frame `maze_board`. No magic numbers in the library: ball radius and margins come from `PlannerParams`.
- Default parameters (a `PlannerParams` default-constructed): `ball_radius` 0.0125, `wall_margin` 0.003, `hole_margin` 0.003, `preferred_clearance` 0.020, `clearance_weight` 2.0.
- The library has no ROS dependency (no rclcpp, no msgs).
- `add_compile_options(-Wall -Wextra -Wpedantic)`, `cxx_std_17`, like `src/maze_sim_nodes/CMakeLists.txt`.
- Build and test inside the `maze-sim` distrobox: `distrobox enter maze-sim -- bash -c "source /opt/ros/humble/setup.bash && colcon build --packages-select maze_planning && colcon test --packages-select maze_planning --event-handlers console_direct+ && colcon test-result --verbose"` (called **BUILD+TEST** below).
- Commit messages start with `[planning]`; no attribution lines.

## Review Focus

- Non-square grids (width ≠ height): a row/column swap must not go unnoticed → tests use a 40 × 20 grid (Tasks 1, 3).
- `data.size() != width * height`, or `resolution <= 0`: throw `std::invalid_argument`, never read out of bounds → Task 1 test.
- Goal cell blocked by clearance (goal pocket too tight for the ball): `found = false`, not a path that ends next to the goal → Task 3 test.
- Start already in the goal cell: `found = true`, one cell, points = [start, goal centre] → Task 3 test.
- Map with no walls at all (all free): clearance is +infinity, planning still works → Task 1 test.

---

### Task 1: Package and clearance map

**Files:**
- Create: `src/maze_planning/package.xml`, `src/maze_planning/CMakeLists.txt`
- Create: `src/maze_planning/include/maze_planning/grid.hpp`, `src/maze_planning/include/maze_planning/clearance.hpp`, `src/maze_planning/src/clearance.cpp`
- Test: `src/maze_planning/test/test_clearance.cpp`

**Interfaces:**
- Produces (`grid.hpp`, namespace `maze_planning`):
  - `struct Grid { int width, height; double resolution, origin_x, origin_y; std::vector<int8_t> data; }`
  - `struct Circle { double x, y, radius; }`
  - `struct PlannerParams` with the five fields and defaults from Global Constraints
  - `struct GridPath { bool found = false; std::vector<std::array<int, 2>> cells; /* {row, col} */ std::vector<std::array<double, 2>> points; /* {x, y} */ }`
  - `bool is_obstacle(int8_t v)` (true for 100 and −1; any value other than 0 counts as obstacle)
  - `std::array<double, 2> cell_center(const Grid&, int row, int col)`
  - `void validate(const Grid&)` throws `std::invalid_argument` if `width <= 0`, `height <= 0`, `resolution <= 0` or `data.size() != width * height`
- Produces (`clearance.hpp`): `std::vector<double> clearance_map(const Grid&)` — per cell, metres from the cell centre to the nearest obstacle cell's edge = (centre-to-centre distance × resolution) − resolution / 2; 0 for obstacle cells; +infinity when the grid has no obstacle. Calls `validate`.

- [ ] **Step 1: Write the package files and the failing tests**

`package.xml`: format 3, name `maze_planning`, version 0.1.0, maintainer Leo Chen `leotcchen@berkeley.edu`, MIT, buildtool `ament_cmake`, `test_depend` `ament_cmake_gtest` and `yaml_cpp_vendor`. `CMakeLists.txt`: library `maze_planning_core` (STATIC, `src/clearance.cpp`; later tasks add sources), public include dir `include/`, install headers and library, `if(BUILD_TESTING)` add `ament_add_gtest(test_clearance test/test_clearance.cpp)` linked to the library.

Tests in `test_clearance.cpp` (res = 0.005, origin (0, 0) unless stated):
- `ClearanceTest.SingleWallNeighbours`: 5 × 5, wall at (2, 2). Clearance at (2, 3) = 0.0025, at (3, 3) = √2·0.005 − 0.0025, at (2, 2) = 0, at (0, 0) = √8·0.005 − 0.0025 (all `EXPECT_NEAR` 1e-12).
- `ClearanceTest.UnknownIsObstacle`: 5 × 5, cell (2, 2) = −1; clearance at (2, 3) = 0.0025.
- `ClearanceTest.NoObstacleIsInfinite`: 4 × 4 all 0; every value is `std::isinf` and > 0.
- `ClearanceTest.MatchesBruteForceNonSquare`: 40 wide × 20 high, deterministic pseudo-random walls (`std::mt19937 rng(7)`, wall with probability 0.15); compare every cell to a brute-force minimum over all obstacle cells, `EXPECT_NEAR` 1e-12.
- `ClearanceTest.RejectsBadGrid`: `data.size()` 24 for a 5 × 5 grid → `EXPECT_THROW(..., std::invalid_argument)`; resolution 0 → throws.

- [ ] **Step 2: Run to verify they fail**

Run: BUILD+TEST. Expected: build fails, `clearance_map` not defined (or link error).

- [ ] **Step 3: Implement `grid.hpp` helpers and `clearance_map`**

Exact squared Euclidean distance transform of Felzenszwalb–Huttenlocher (1-D lower-envelope pass along columns, then along rows) on cell indices, with obstacle cells as sources; take the square root, multiply by `resolution`, subtract `resolution / 2`, set obstacle cells to 0. Infinite when no source exists in the grid.

- [ ] **Step 4: Run to verify they pass**

Run: BUILD+TEST. Expected: `test_clearance` 5/5 pass.

- [ ] **Step 5: Commit**

```bash
git add src/maze_planning
git commit -m "[planning] maze_planning package and exact clearance map"
```

### Task 2: Blocked cells

**Files:**
- Create: `src/maze_planning/include/maze_planning/blocked.hpp`, `src/maze_planning/src/blocked.cpp`
- Modify: `src/maze_planning/CMakeLists.txt` (add source and `test_blocked`)
- Test: `src/maze_planning/test/test_blocked.cpp`

**Interfaces:**
- Consumes: `Grid`, `Circle`, `PlannerParams`, `cell_center`, `clearance_map` (Task 1).
- Produces: `std::vector<uint8_t> blocked_cells(const Grid&, const std::vector<double>& clearance, const std::vector<Circle>& holes, const PlannerParams&)` — 1 where the cell is an obstacle, or `clearance < ball_radius + wall_margin`, or the cell centre is closer than `hole.radius + hole_margin` to a hole centre; else 0.

- [ ] **Step 1: Write the failing tests**

- `BlockedTest.ChannelBlocksNearWalls`: 20 wide × 11 high, rows 0 and 10 walls (a horizontal channel 9 cells tall), res 0.005, params `ball_radius` 0.0125, `wall_margin` 0.003 (need 0.0155). Rows 4–6 are free (clearance 0.0175–0.0225), rows 1–3 and 7–9 are blocked (clearance ≤ 0.0125).
- `BlockedTest.HoleKeepOut`: 21 × 21 all free, hole at the centre cell's centre with radius 0.010, `hole_margin` 0.003: a cell whose centre is 0.010 away is blocked, one 0.015 away is free.
- `BlockedTest.ObstacleAlwaysBlocked`: an isolated wall cell in open space is blocked even with `ball_radius` 0 and `wall_margin` 0.

- [ ] **Step 2: Run to verify they fail**

Run: BUILD+TEST. Expected: `blocked_cells` not defined.

- [ ] **Step 3: Implement `blocked_cells`**

- [ ] **Step 4: Run to verify they pass**

Run: BUILD+TEST. Expected: `test_clearance` and `test_blocked` pass.

- [ ] **Step 5: Commit**

```bash
git add src/maze_planning
git commit -m "[planning] blocked cells from ball clearance and hole keep-out"
```

### Task 3: A\* grid path

**Files:**
- Create: `src/maze_planning/include/maze_planning/grid_planner.hpp`, `src/maze_planning/src/grid_planner.cpp`
- Modify: `src/maze_planning/CMakeLists.txt` (add source and `test_grid_planner`)
- Test: `src/maze_planning/test/test_grid_planner.cpp`

**Interfaces:**
- Consumes: `clearance_map` (Task 1), `blocked_cells` (Task 2).
- Produces: `GridPath plan_grid_path(const Grid&, const std::vector<Circle>& holes, std::array<double, 2> start, const Circle& goal, const PlannerParams& = PlannerParams{})`.

Behaviour (from the spec, Design steps 3–5):
- 8-connected; a diagonal step needs both orthogonal side cells unblocked.
- Step cost = step length (1 or √2, in cells) × (1 + `clearance_weight` × max(0, `preferred_clearance` − clearance(to-cell)) / `preferred_clearance`). Heuristic: Euclidean distance in cells to the goal cell.
- Start outside the grid or in an obstacle cell → `found = false`. Goal cell blocked or outside → `found = false`.
- Start cell blocked (but not an obstacle): the path begins at the start cell, walks to the nearest unblocked cell (smallest Euclidean distance in cells, ties by lowest row then col) along a 4/8-connected line of non-obstacle cells (cells of a Bresenham line; if it crosses an obstacle, `found = false`), then A\* continues from there.
- `points` = cell centres, with `points.front()` = the exact start and `points.back()` = the goal centre (`goal.x, goal.y`). Start in the goal cell → cells = [goal cell], points = [start, goal centre].

- [ ] **Step 1: Write the failing tests**

All with res 0.005 and default params unless stated.
- `GridPlannerTest.StraightChannelFollowsCentre`: 40 wide × 11 high, rows 0 and 10 walls; start at centre of (5, 2), goal centre of (5, 37), r 0.01. `found`; every path cell has row 5.
- `GridPlannerTest.TooNarrowHasNoPath`: same channel but rows 0 and 6 walls (5 free rows: max clearance 0.0125 < 0.0155) → `found == false`.
- `GridPlannerTest.AvoidsHole`: 40 × 21 all free with a walled border; hole of radius 0.010 at the centre of (10, 20); start (10, 3), goal (10, 36). `found`; every path point is ≥ 0.013 from the hole centre.
- `GridPlannerTest.NoCornerCutting`: 7 × 7 all free, `ball_radius` 0, `wall_margin` 0, walls at (3, 2) and (2, 3); start (2, 2), goal (3, 3). `found`; no consecutive cells (2, 2)→(3, 3).
- `GridPlannerTest.StartAgainstWallLeavesIt`: the 40 × 11 channel of the first test, start at the centre of (1, 2) (blocked, not an obstacle); `found`; `cells.front() == {1, 2}`; `points.front()` equals the start exactly; the path reaches row 5 before column 10.
- `GridPlannerTest.StartInWallFails`: start in a wall cell → `found == false`; start at x = −1 (outside) → `found == false`.
- `GridPlannerTest.GoalBlockedFails`: the channel of the first test with the goal centre in row 2 (blocked by clearance) → `found == false`.
- `GridPlannerTest.StartInGoalCell`: start inside the goal cell → `found`, `cells.size() == 1`, `points.size() == 2`, `points.back()` == goal centre.
- `GridPlannerTest.NonSquareRowsAndCols`: 40 wide × 20 high, all free with a walled border, `ball_radius` 0, `wall_margin` 0; start centre of (2, 2), goal centre of (17, 37). `found`; `cells.back() == {17, 37}`; every point lies inside the grid's x range [0, 0.2] and y range [0, 0.1].

- [ ] **Step 2: Run to verify they fail**

Run: BUILD+TEST. Expected: `plan_grid_path` not defined.

- [ ] **Step 3: Implement `plan_grid_path`**

`std::priority_queue` on (f, index) with a `g` array initialised to +infinity and a parent array; reconstruct by walking parents.

- [ ] **Step 4: Run to verify they pass**

Run: BUILD+TEST. Expected: all three test executables pass.

- [ ] **Step 5: Commit**

```bash
git add src/maze_planning
git commit -m "[planning] A* grid path with clearance cost"
```

### Task 4: L1 fixture and end-to-end test

**Files:**
- Create: `src/maze_planning/scripts/export_fixture.py`
- Create: `src/maze_planning/test/data/L1.yaml` (generated, committed)
- Modify: `src/maze_planning/CMakeLists.txt` (`test_l1`, linked to yaml-cpp, with `MAZE_PLANNING_TEST_DATA` defined as the absolute path of `test/data`)
- Test: `src/maze_planning/test/test_l1.cpp`

**Interfaces:**
- Consumes: `plan_grid_path`, `clearance_map`, `cell_center` (Tasks 1–3); `maze_cad_map.cad_map.build_map` and `path_cells_board` (existing, `src/maze_cad_map/maze_cad_map/cad_map.py`).
- Produces: `test/data/L1.yaml` with keys `level`, `resolution`, `width`, `height`, `origin` [x, y], `data` (flat list, row 0 first), `holes` (list of [x, y, radius]), `start` [x, y], `goal` [x, y, radius], `route` (list of [x, y], the designed route cell centres).

- [ ] **Step 1: Write `export_fixture.py` and generate the fixture**

Usage: `python3 src/maze_planning/scripts/export_fixture.py --cad-scripts maze_CAD/scripts --level L1 --out src/maze_planning/test/data/L1.yaml`, run in the container with `PYTHONPATH=src/maze_cad_map`. A header comment in the YAML says it is generated and names the command.

Expected: the file exists; `width` and `height` are 60; `route` has 18 entries.

- [ ] **Step 2: Write the failing test**

`L1Test.RouteReachesGoalWithClearance`: load the fixture with yaml-cpp; `plan_grid_path` with default params. `found`; `points.back()` equals the goal centre; for every designed route centre, some path point lies within 0.021 m of it; for every path cell, `clearance_map` ≥ 0.0155 − 1e-9.

- [ ] **Step 3: Run to verify it fails**

Run: BUILD+TEST. Expected: `test_l1` fails to build until it is added to CMake, then fails on the first assertion if the planner is wrong (with Tasks 1–3 done it should pass; if it passes immediately, temporarily set `ball_radius` 0.02 in the test and check it fails with `found == false`, then restore).

- [ ] **Step 4: Run to verify it passes**

Run: BUILD+TEST. Expected: all four test executables pass.

- [ ] **Step 5: Commit**

```bash
git add src/maze_planning
git commit -m "[planning] L1 fixture from the CAD map and end-to-end grid path test"
```
