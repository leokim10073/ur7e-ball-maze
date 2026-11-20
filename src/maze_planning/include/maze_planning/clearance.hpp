#pragma once

#include <vector>

#include "maze_planning/grid.hpp"

namespace maze_planning
{

// For every cell, the distance in m from its center to the nearest obstacle cell's edge
// (center-to-center distance minus half a cell). 0 on obstacle cells, +infinity when the grid has
// no obstacle. Exact Euclidean distance transform (Felzenszwalb & Huttenlocher).
std::vector<double> clearance_map(const Grid & grid);

}  // namespace maze_planning
