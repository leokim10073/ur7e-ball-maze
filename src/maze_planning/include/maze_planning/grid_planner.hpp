#pragma once

#include <array>
#include <vector>

#include "maze_planning/grid.hpp"

namespace maze_planning
{

// A* from the ball at `start` to the goal pocket's center over the cells the ball's center may
// occupy (blocked_cells), 8-connected without cutting corners, with steps near walls costing extra.
// A start that is already too close to a wall (a pushed ball) first walks straight to the nearest
// allowed cell. found = false when the start is outside the grid or in an obstacle, the goal cell is
// blocked or outside, or no route exists.
GridPath plan_grid_path(
  const Grid & grid, const std::vector<Circle> & holes, std::array<double, 2> start,
  const Circle & goal, const PlannerParams & params = PlannerParams{});

}  // namespace maze_planning
