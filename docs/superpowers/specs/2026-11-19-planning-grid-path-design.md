# Planning PL1 + PL2: grid path with ball clearance

Work package: [#16 · Planning](https://github.com/leokim10073/ur7e-ball-maze/issues/16), tasks PL1 (A\* on the grid map) and PL2 (wall clearance from the ball radius, keep-out around holes).

## Why

Ball control needs a route from the ball to the goal that a real ball can actually roll along. A plain shortest path through free cells hugs the walls, and with the 25 mm marble ([#19](https://github.com/leokim10073/ur7e-ball-maze/issues/19)) in 42 mm channels, a route that ignores the ball's size runs it into the walls. This step produces that route on the grid; smoothing and corridors (PL3) and the ROS node (PL4) build on it.

## Goals

- From a `MazeMap` grid (Contract 1 in `docs/interfaces.md`), the holes, a start point and the goal, return a path of cells and board-frame points that keeps the ball's center at least `ball_radius + wall_margin` from every wall and `hole_radius + hole_margin` from every hole centre.
- Prefer the middle of a channel when there is room.
- Report "no path" when the ball cannot get through.
- Start from a ball that is already too close to a wall (pushed off), so replanning works.

## Non-goals

- Smoothing, arc length, corridors (PL3), ROS node, topics and config loading (PL4), velocity profile.

## Constraints

- **C++** (team decision for the planner), ament_cmake package `src/maze_planning`.
- The algorithm is a library with no ROS dependency; the PL4 node will only convert messages.
- Grid format exactly as Contract 1: `nav_msgs/OccupancyGrid` conventions, 0 free, 100 wall, -1 unknown (treated as wall), row 0 at the bottom (-y), origin at the lower-left corner, 5 mm cells on L1.
- Units SI, frame `maze_board`. No magic numbers: ball radius and margins are parameters.
- Measured on the L1 CAD map: with ball radius 12.5 mm and a 3 mm margin every cell of the designed route has 17.5–23.5 mm clearance, start and goal are connected, and the grid clearance is within about 1 mm of the CAD geometry. A 5 mm grid is good enough.

## Approaches

1. **Inflate on the grid, then A\* with a clearance cost (chosen).** A distance transform gives every cell's distance to the nearest wall; cells closer than the clearance are blocked; A\* pays extra near walls. Simple, fast, and works on perception's map unchanged.
2. A finer internal grid (1 mm): smoother but 25× the cells, and no more accurate than the 5 mm map it comes from.
3. Geometric planning on wall polygons: exact, but perception only gives a grid, so the grid would have to be turned back into polygons.

Language: Python measured 7.5 ms per plan on L1, well inside one camera frame, but the team chose C++ for the planner.

## Design

Interface (`maze_planning/grid_planner.hpp`, namespace `maze_planning`):

```cpp
struct Grid   { int width, height; double resolution, origin_x, origin_y; std::vector<int8_t> data; };
struct Circle { double x, y, radius; };
struct PlannerParams { double ball_radius, wall_margin, hole_margin, clearance_weight, preferred_clearance; };
struct GridPath { bool found; std::vector<std::array<int,2>> cells; std::vector<std::array<double,2>> points; };

GridPath plan_grid_path(const Grid&, const std::vector<Circle>& holes,
                        std::array<double,2> start, const Circle& goal, const PlannerParams&);
```

Steps:

1. **Clearance.** Exact Euclidean distance transform (Felzenszwalb–Huttenlocher, two separable passes) from every cell centre to the nearest wall or unknown cell centre, times the resolution, minus half a cell. Written in the library, so no OpenCV dependency.
2. **Blocked cells.** Clearance below `ball_radius + wall_margin`, or centre within `hole_radius + hole_margin` of a hole centre. The goal is not a hole.
3. **A\*.** 8-connected; a diagonal step needs both side cells free (no corner cutting). Step cost = step length × (1 + `clearance_weight` × max(0, `preferred_clearance` − clearance) / `preferred_clearance`). Heuristic: Euclidean distance to the goal cell (admissible, since the cost factor is ≥ 1).
4. **Start in a blocked cell.** If the start cell is blocked but not a wall, the path begins there and steps to the nearest free cell (by distance), then A\* continues. A start in a wall cell, or outside the grid, returns `found = false`.
5. **Output.** `cells` from start to goal; `points` the cell centres in `maze_board`, with the first point replaced by the exact start and the last by the goal centre.

Default parameters: `ball_radius` 0.0125, `wall_margin` 0.003, `hole_margin` 0.003, `preferred_clearance` 0.020, `clearance_weight` 2.0. The weight and preferred clearance are my call, tuned in PL5.

## Tests (gtest)

- Synthetic grids: a straight channel (path stays on the centre line); a channel narrower than `2 × (ball_radius + wall_margin)` (`found = false`); a hole in the way (path keeps out); a start against a wall (path starts there and leaves); unknown cells block like walls; no corner cutting.
- L1: a fixture `test/data/L1.yaml` (grid, start, goal and the designed route cells), exported by a small script from `maze_cad_map` and committed with the script. The path reaches the goal, passes within half a maze cell (21 mm) of every designed route cell's centre, and every path cell has clearance ≥ 15.5 mm.

## Left to the builder

- Data structures inside A\* (priority queue, closed set) and the YAML reader for the fixture (yaml-cpp is available through ROS).
