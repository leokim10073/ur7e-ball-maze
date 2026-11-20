#include "maze_planning/grid.hpp"

#include <cmath>
#include <stdexcept>

namespace maze_planning
{

bool is_obstacle(int8_t value)
{
  return value != 0;
}

std::array<double, 2> cell_center(const Grid & grid, int row, int col)
{
  return {grid.origin_x + (col + 0.5) * grid.resolution, grid.origin_y + (row + 0.5) * grid.resolution};
}

void validate(const Grid & grid)
{
  if (grid.width <= 0 || grid.height <= 0) {
    throw std::invalid_argument("grid has no cells");
  }
  if (!(std::isfinite(grid.resolution) && grid.resolution > 0.0)) {
    throw std::invalid_argument("grid resolution must be finite and positive");
  }
  if (!std::isfinite(grid.origin_x) || !std::isfinite(grid.origin_y)) {
    throw std::invalid_argument("grid origin must be finite");
  }
  if (grid.data.size() != static_cast<size_t>(grid.width) * static_cast<size_t>(grid.height)) {
    throw std::invalid_argument("grid data size does not match width * height");
  }
}

void validate(const PlannerParams & params)
{
  for (double v : {params.ball_radius, params.wall_margin, params.hole_margin, params.clearance_weight}) {
    if (!(std::isfinite(v) && v >= 0.0)) {
      throw std::invalid_argument("ball radius, margins and clearance weight must be finite and >= 0");
    }
  }
  if (!(std::isfinite(params.preferred_clearance) && params.preferred_clearance > 0.0)) {
    throw std::invalid_argument("preferred clearance must be finite and > 0");
  }
  if (!(std::isfinite(params.point_spacing) && params.point_spacing > 0.0)) {
    throw std::invalid_argument("point spacing must be finite and > 0");
  }
}

}  // namespace maze_planning
