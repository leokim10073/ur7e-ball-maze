#pragma once

#include <cstdint>
#include <vector>

#include "maze_planning/grid.hpp"

namespace maze_planning
{

// 1 where the ball's center may not be: an obstacle cell, a cell closer than
// ball_radius + wall_margin to an obstacle (clearance from clearance_map), or a cell whose center is
// closer than radius + hole_margin to a hole's center. Else 0.
std::vector<uint8_t> blocked_cells(
  const Grid & grid, const std::vector<double> & clearance, const std::vector<Circle> & holes,
  const PlannerParams & params);

}  // namespace maze_planning
