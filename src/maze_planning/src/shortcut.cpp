#include <algorithm>

#include "geometry.hpp"
#include "maze_planning/route.hpp"

namespace maze_planning
{

std::vector<std::array<int, 2>> shortcut_cells(
  const Grid & grid, const std::vector<double> & clearance, const std::vector<uint8_t> & blocked,
  const std::vector<std::array<int, 2>> & cells, const PlannerParams & params)
{
  std::vector<std::array<int, 2>> vertices;
  if (cells.empty()) {
    return vertices;
  }
  const size_t n = cells.size();
  size_t i = 0;
  vertices.push_back(cells[0]);
  std::vector<double> run_min(n);  // smallest A* clearance on cells[i..j]
  while (i + 1 < n) {
    run_min[i] = clearance[detail::index(grid, cells[i])];
    for (size_t j = i + 1; j < n; ++j) {
      run_min[j] = std::min(run_min[j - 1], clearance[detail::index(grid, cells[j])]);
    }
    size_t next = i + 1;  // one A* step always qualifies
    for (size_t j = n - 1; j > i + 1; --j) {
      const double needed = std::min(params.preferred_clearance, run_min[j]) - 1e-12;
      if (!detail::line_allowed(grid, blocked, cells[i], cells[j])) {
        continue;
      }
      const auto line = detail::line_cells(cells[i], cells[j]);
      if (std::all_of(line.begin(), line.end(), [&](const detail::Cell & c) {
          return clearance[detail::index(grid, c)] >= needed;
        }))
      {
        next = j;
        break;
      }
    }
    vertices.push_back(cells[next]);
    i = next;
  }
  return vertices;
}

}  // namespace maze_planning
