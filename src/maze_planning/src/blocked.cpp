#include "maze_planning/blocked.hpp"

#include <cmath>
#include <stdexcept>

namespace maze_planning
{

std::vector<uint8_t> blocked_cells(
  const Grid & grid, const std::vector<double> & clearance, const std::vector<Circle> & holes,
  const PlannerParams & params)
{
  validate(grid);
  validate(params);
  if (clearance.size() != grid.data.size()) {
    throw std::invalid_argument("clearance map size does not match the grid");
  }
  const double wall_clearance = params.ball_radius + params.wall_margin;
  std::vector<uint8_t> blocked(grid.data.size(), 0);
  for (int r = 0; r < grid.height; ++r) {
    for (int c = 0; c < grid.width; ++c) {
      const size_t i = static_cast<size_t>(r * grid.width + c);
      if (is_obstacle(grid.data[i]) || clearance[i] < wall_clearance) {
        blocked[i] = 1;
        continue;
      }
      const auto p = cell_center(grid, r, c);
      for (const auto & hole : holes) {
        if (std::hypot(p[0] - hole.x, p[1] - hole.y) < hole.radius + params.hole_margin) {
          blocked[i] = 1;
          break;
        }
      }
    }
  }
  return blocked;
}

}  // namespace maze_planning
