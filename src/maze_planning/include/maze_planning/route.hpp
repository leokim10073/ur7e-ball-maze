// The route PL4 publishes as Contract 3's PlannedRoute (docs/interfaces.md): straightened points,
// arc length and axis-aligned safe corridors for the ball center, all in maze_board, m.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "maze_planning/grid.hpp"

namespace maze_planning
{

struct Corridor
{
  double x_min = 0.0;  // ball-center bounds, wall clearance already taken off
  double x_max = 0.0;
  double y_min = 0.0;
  double y_max = 0.0;
};

struct Route
{
  std::vector<std::array<double, 2>> points;  // start to goal
  std::vector<double> arc_length;             // per point, from 0
  std::vector<Corridor> corridors;            // path order, consecutive ones overlap
  std::vector<uint32_t> pose_corridor;        // per point, index into corridors
};

// Vertices of the straightened path, a subsequence of cells that keeps the first and the last.
// From each vertex the farthest later cell is taken whose straight line has only allowed cells,
// cuts no corner and stays as far from walls as A* did there: every line cell's clearance is at
// least min(preferred_clearance, the smallest clearance on the A* cells in between).
// Every cell of cells must be allowed (the caller leaves out the lead-in).
std::vector<std::array<int, 2>> shortcut_cells(
  const Grid & grid, const std::vector<double> & clearance, const std::vector<uint8_t> & blocked,
  const std::vector<std::array<int, 2>> & cells, const PlannerParams & params);

// Segments {r0, c0, r1, c1} between consecutive vertices, each split at the middle cell of its
// line, recursively, until the box spanned by its two end cells has no blocked cell.
// Consecutive segments share an end cell.
std::vector<std::array<int, 4>> split_segments(
  const Grid & grid, const std::vector<uint8_t> & blocked,
  const std::vector<std::array<int, 2>> & vertices);

// The box spanned by cells (r0, c0) and (r1, c1), grown one row or column at a time, round robin
// (+x, +y, -x, -y), while the new row or column is all allowed and inside the grid. Its bounds
// are the centers of the outermost cells. The starting box must be free.
Corridor grow_corridor(
  const Grid & grid, const std::vector<uint8_t> & blocked, int r0, int c0, int r1, int c1);

}  // namespace maze_planning
