# Planning PL3: smoothed route, arc length and safe corridors

Work package: [#16 · Planning](https://github.com/leokim10073/ur7e-ball-maze/issues/16), task PL3. Builds on PL1+PL2 (`docs/superpowers/specs/2026-11-19-planning-grid-path-design.md`).

## Why

The PL1+PL2 path is a 5 mm staircase of cells. Ball control needs what Contract 3 (`docs/interfaces.md`) promises in `PlannedRoute`: waypoints PID can follow without jitter (#6), an arc length MPCC can use as progress (#11), and convex safe corridors for the ball center that the constrained MPC uses as constraints (#10), with the wall clearance already taken off.

## Goals

- Turn a `GridPath` into a `Route`: points from the ball to the goal, `arc_length` per point, axis-aligned `corridors` in path order with consecutive ones overlapping, and `pose_corridor` per point.
- Straighten the staircase where that does not bring the ball closer to walls than the A\* path already did.
- Every corridor is safe for the ball center: built only from allowed cells (clearance ≥ `ball_radius + wall_margin`, outside hole keep-outs).

## Non-goals

- ROS node, messages, replanning trigger (PL4). Velocity profile. A curved (spline) path: segments stay straight.

## Approaches (straightening)

1. Straighten wherever the line avoids blocked cells: shortest, but hugs the 15.5 mm boundary, undoing the A\* centering.
2. **Straighten only where the line keeps the clearance the A\* path had (chosen).** A line from vertex i to j is accepted when every cell on it is allowed and has clearance ≥ min(`preferred_clearance`, the smallest clearance on the A\* cells i..j). Open rooms become straight lines; narrow places keep the centered path.
3. No straightening: simplest, but many tiny corridors and a jittery path for PID.

## Design

Library additions (`maze_planning`, C++17, no ROS):

```cpp
struct Corridor { double x_min, x_max, y_min, y_max; };  // ball-center bounds in maze_board, m
struct Route {
  std::vector<std::array<double, 2>> points;  // start to goal
  std::vector<double> arc_length;             // per point, m, from 0
  std::vector<Corridor> corridors;            // path order, consecutive ones overlap
  std::vector<uint32_t> pose_corridor;        // per point, index into corridors
};
// PlannerParams gains: double point_spacing = 0.005;
Route build_route(const Grid&, const std::vector<Circle>& holes, const GridPath&, const PlannerParams&);
```

Steps:

1. **Lead-in.** Leading cells of the `GridPath` that are blocked (a pushed ball, PL2's lead-in) are kept as they are, as one straight segment from the exact start to the first allowed cell.
2. **Straighten** the rest greedily: from the current vertex take the farthest later cell whose line (the same Bresenham line as PL2, with no corner cutting) satisfies approach 2. The vertices' points are cell centers, except the exact start and the goal center from `GridPath.points`.
3. **Split for corridors.** A segment whose bounding box (of its cell centers) contains a blocked cell is split at its middle cell, recursively, until every box is free. A single 8-connected step without corner cutting always has a free box.
4. **Grow** each segment's box one row or column at a time (round robin over the four sides) while the new row or column is all allowed cells, up to the grid edge. Bounds are the centers of the outermost cells (conservative by half a cell; the 3 mm margin covers the ~1 mm discretization). The lead-in segment's corridor is the box of its cells, not grown, and may contain blocked-but-not-wall cells: it is the recovery region a pushed ball is in, so it must not trigger another replan (PL4).
5. **Overlap.** Consecutive segments share a vertex cell, which lies in both boxes, so consecutive corridors overlap.
6. **Points.** Each segment is sampled every `point_spacing` (both ends included, no duplicate at shared vertices); `arc_length` is cumulative; `pose_corridor` is the segment's corridor (a shared vertex belongs to the earlier segment; it lies in both).
7. A `GridPath` with `found == false` gives an empty `Route` (PL4 publishes `STATUS_NO_PATH`).

## Tests (gtest)

- A straight channel gives one corridor spanning the free rows (bounds at the cell centers of rows 4 and 6) and the full free length; points spaced ≤ `point_spacing`; last `arc_length` = start-to-goal distance.
- An open room with start and goal on a diagonal gives one straight segment.
- A diagonal blocked by an obstacle block gets split; every corridor's cells are allowed.
- A pushed start: `points.front()` is the exact start and lies inside corridor 0.
- A not-found `GridPath` gives an empty `Route`.
- L1 properties: every point inside its `pose_corridor`; consecutive corridors overlap; every corridor (after the lead-in) contains only allowed cells; `arc_length` starts at 0 and strictly increases; the route is no longer than the A\* path; every point's cell clearance ≥ min(`preferred_clearance`, the A\* path's smallest clearance) − 1e-9; the route has fewer corridors than the A\* path has cells.

## Left to the builder

- Internal helpers and their files; reusing PL2's line and cell helpers (move them out of `grid_planner.cpp` into a shared internal header).
