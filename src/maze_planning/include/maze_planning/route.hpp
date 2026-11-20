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

}  // namespace maze_planning
