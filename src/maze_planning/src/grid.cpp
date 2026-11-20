#include "maze_planning/grid.hpp"

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
  if (!(grid.resolution > 0.0)) {
    throw std::invalid_argument("grid resolution must be positive");
  }
  if (grid.data.size() != static_cast<size_t>(grid.width) * static_cast<size_t>(grid.height)) {
    throw std::invalid_argument("grid data size does not match width * height");
  }
}

}  // namespace maze_planning
