# Planning PL3: Route and Corridors Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the PL1+PL2 `GridPath` into a Contract 3 route: straightened points, arc length, safe axis-aligned corridors and the point-to-corridor index.

**Architecture:** Additions to the ROS-free `maze_planning_core` library: a shared internal geometry header, `shortcut_cells` (approach 2 straightening), `segment_corridors` (split and grow), and `build_route` tying them together with sampling. gtest on synthetic grids and the L1 fixture.

**Tech Stack:** C++17, ament_cmake, ament_cmake_gtest, yaml-cpp (tests).

**Spec:** `docs/superpowers/specs/2026-11-19-planning-route-corridors-design.md`

## Global Constraints

- Everything in `docs/superpowers/plans/2026-11-19-planning-grid-path.md` → Global Constraints still holds (grid format, SI units, no ROS in the library, compile flags, BUILD+TEST command, `[planning]` commit prefix, no attribution).
- `PlannerParams` gains `double point_spacing = 0.005;` and `validate(PlannerParams)` rejects `point_spacing` that is not finite and > 0.
- "Allowed" cell = `blocked_cells(...)[i] == 0`.
- Corridor bounds are cell centers of the outermost cells of the box.

## Review Focus

- `GridPath` with a single cell (start in the goal cell): one corridor, points = [start, goal], arc_length = [0, d] → Task 3 test.
- Start exactly on a shared vertex / zero-length segments (start at the goal center): no NaN in sampling, no duplicate points with equal arc length → Task 3 test (start == goal center).
- Corridor growth reaching the grid edge (no walls at the border): stops at the edge without reading out of range → Task 2 test (grid with no walls).
- A `GridPath` whose cells and points sizes disagree, or with `found == false`: empty Route, no crash → Task 3 test.
- `point_spacing` 0 or negative: throws `std::invalid_argument` → Task 3 test.

---

### Task 1: Shared geometry helpers and straightening

**Files:**
- Create: `src/maze_planning/src/geometry.hpp` (internal, not installed), `src/maze_planning/include/maze_planning/route.hpp`, `src/maze_planning/src/shortcut.cpp`
- Modify: `src/maze_planning/src/grid_planner.cpp` (use `geometry.hpp` instead of its local `cell_of` / `line_cells`), `src/maze_planning/include/maze_planning/grid.hpp` (`point_spacing`), `src/maze_planning/src/grid.cpp` (validate it), `src/maze_planning/CMakeLists.txt`
- Test: `src/maze_planning/test/test_shortcut.cpp`

**Interfaces:**
- Produces (`geometry.hpp`, namespace `maze_planning::detail`): `using Cell = std::array<int, 2>;` `std::optional<Cell> cell_of(const Grid&, double x, double y);` `std::vector<Cell> line_cells(Cell a, Cell b);` (moved unchanged from `grid_planner.cpp`); `bool line_allowed(const Grid&, const std::vector<uint8_t>& blocked, Cell a, Cell b);` — every cell of `line_cells(a, b)` allowed and, at each diagonal step, both side cells allowed.
- Produces (`route.hpp`): `struct Corridor`, `struct Route` exactly as in the spec; `std::vector<std::array<int, 2>> shortcut_cells(const Grid&, const std::vector<double>& clearance, const std::vector<uint8_t>& blocked, const std::vector<std::array<int, 2>>& cells, const PlannerParams&)` — vertex cells, first and last cell of `cells` always included; from vertex i take the largest j > i with `line_allowed` and every line cell's clearance ≥ min(`preferred_clearance`, min clearance over `cells[i..j]`) − 1e-12. Assumes every cell of `cells` is allowed (the caller strips the lead-in).

- [ ] **Step 1: Write the failing tests**
  - `ShortcutTest.OpenRoomDiagonalIsOneSegment`: 40 × 40, walled border, default params; `plan_grid_path` from cell (10, 10) to (29, 29); `shortcut_cells` on its cells returns exactly 2 vertices.
  - `ShortcutTest.StraightChannelIsOneSegment`: the 40 × 11 channel (walls rows 0 and 10), path (5, 2)→(5, 37): 2 vertices.
  - `ShortcutTest.KeepsAStarClearance`: L1 fixture (load as in `test_l1.cpp`; move its loader to `test/l1_fixture.hpp`); for every pair of consecutive vertices, every cell of `line_cells` has clearance ≥ min(0.020, min clearance of the A\* cells between them) − 1e-9.
  - `ShortcutTest.LShapeKeepsCorner`: 30 × 30 walled border plus a wall block filling rows 0–19, cols 10–29 (an L-shaped free area: a left leg, cols 1–9, and a top leg, rows 20–28); path from (5, 5) in the left leg to (25, 25) in the top leg; the result has ≥ 3 vertices and every consecutive pair is `line_allowed`.
- [ ] **Step 2: Run to verify they fail** — BUILD+TEST. Expected: `shortcut_cells` not defined.
- [ ] **Step 3: Implement** `geometry.hpp` helpers (move `cell_of`/`line_cells`, add `line_allowed`), `point_spacing` + its validation, and `shortcut_cells`.
- [ ] **Step 4: Run to verify they pass** — BUILD+TEST. Expected: all suites pass (24 earlier tests unchanged).
- [ ] **Step 5: Commit** `[planning] straighten the grid path where it keeps the A* clearance`

### Task 2: Corridors

**Files:**
- Create: `src/maze_planning/src/corridors.cpp`
- Modify: `src/maze_planning/include/maze_planning/route.hpp`, `src/maze_planning/CMakeLists.txt`
- Test: `src/maze_planning/test/test_corridors.cpp`

**Interfaces:**
- Consumes: `Cell`, `line_cells` (Task 1).
- Produces: `std::vector<std::array<int, 4>> split_segments(const Grid&, const std::vector<uint8_t>& blocked, const std::vector<std::array<int, 2>>& vertices)` — returns segments as `{r0, c0, r1, c1}` after recursive splitting at the middle cell of `line_cells` until each segment's cell box has no blocked cell; `Corridor grow_corridor(const Grid&, const std::vector<uint8_t>& blocked, int r0, int c0, int r1, int c1)` — the box spanned by the two cells, grown round-robin (x_max, y_max, x_min, y_min order) one row/column at a time while the new row/column is all allowed and inside the grid; bounds are cell centers.

- [ ] **Step 1: Write the failing tests**
  - `CorridorTest.StraightChannelSpansFreeRows`: 40 × 11 channel, default params; `grow_corridor` for (5, 2)–(5, 37) gives y_min = center y of row 4 (0.0225), y_max = row 6 (0.0325), x_min = col 0 center (0.0025), x_max = col 39 center (0.1975), all `EXPECT_NEAR` 1e-12.
  - `CorridorTest.GrowsToGridEdgeWithoutWalls`: 10 × 10 all free, `ball_radius` 0, `wall_margin` 0; growing (5, 5)–(5, 5) gives the whole grid's cell centers (0.0025 … 0.0475 both axes).
  - `CorridorTest.SplitsDiagonalAroundBlock`: 30 × 30 all free with `ball_radius` 0, `wall_margin` 0 and a wall block at rows 12–17, cols 3–8; vertices (2, 2) and (27, 27) → `split_segments` returns ≥ 2 segments, consecutive segments share an end cell, and no segment's box contains a blocked cell.
- [ ] **Step 2: Run to verify they fail** — BUILD+TEST. Expected: not defined.
- [ ] **Step 3: Implement** `split_segments` and `grow_corridor`.
- [ ] **Step 4: Run to verify they pass** — BUILD+TEST.
- [ ] **Step 5: Commit** `[planning] split segments into free boxes and grow safe corridors`

### Task 3: build_route

**Files:**
- Create: `src/maze_planning/src/route.cpp`
- Modify: `src/maze_planning/include/maze_planning/route.hpp`, `src/maze_planning/CMakeLists.txt`
- Test: `src/maze_planning/test/test_route.cpp`

**Interfaces:**
- Consumes: `clearance_map`, `blocked_cells`, `plan_grid_path` (PL1+PL2); `shortcut_cells` (Task 1); `split_segments`, `grow_corridor` (Task 2).
- Produces: `Route build_route(const Grid&, const std::vector<Circle>& holes, const GridPath&, const PlannerParams& = PlannerParams{})` per the spec's steps 1–7. Lead-in = the leading blocked cells of `path.cells`; its corridor is the cell box of the lead-in plus the first allowed cell, not grown.

- [ ] **Step 1: Write the failing tests**
  - `RouteTest.StraightChannel`: channel, start (5, 2) center, goal (5, 37) center: `corridors.size() == 1`; consecutive points ≤ 0.005 + 1e-12 apart; `arc_length.front() == 0`; `arc_length.back()` ≈ distance start→goal (1e-9); `pose_corridor` all 0; points.front() == start, points.back() == goal center.
  - `RouteTest.PushedStartInsideFirstCorridor`: channel, start = center of (1, 2): `points.front()` == start and lies inside `corridors[pose_corridor[0]]`.
  - `RouteTest.NotFoundGivesEmpty`: a `GridPath` with `found = false` → all four vectors empty; a `GridPath` with `found = true` but `cells.size() != points.size()` → empty.
  - `RouteTest.StartInGoalCell`: start 1 mm from the goal center in its cell: `points.size() == 2`, `corridors.size() == 1`, `arc_length` == {0, 0.001} (1e-12).
  - `RouteTest.StartAtGoalCenter`: start == goal center: no NaN in points or arc_length; `arc_length` is non-decreasing and has no two equal entries.
  - `RouteTest.RejectsBadSpacing`: `point_spacing` 0 → throws `std::invalid_argument`.
  - `RouteTest.L1Properties`: L1 fixture, default params: every point inside its `pose_corridor`; consecutive corridors overlap (x and y ranges intersect); every corridor after the lead-in contains only allowed cells; `arc_length` starts at 0 and strictly increases; route length ≤ A\* path length (sum of `GridPath.points` segment lengths); every point's cell clearance ≥ min(0.020, min A\* cell clearance) − 1e-9; `corridors.size() < path.cells.size()`.
- [ ] **Step 2: Run to verify they fail** — BUILD+TEST. Expected: `build_route` not defined.
- [ ] **Step 3: Implement** `build_route`. For `StartAtGoalCenter`, drop a sample whose distance from the previous point is < 1e-12.
- [ ] **Step 4: Run to verify they pass** — BUILD+TEST. Expected: every suite passes.
- [ ] **Step 5: Commit** `[planning] build the Contract 3 route: points, arc length and corridors`
