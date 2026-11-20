// Internal cell and line helpers shared by the planner, the straightening and the corridors.
// Not installed: the public headers are under include/.
#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <vector>

#include "maze_planning/grid.hpp"

namespace maze_planning::detail
{

using Cell = std::array<int, 2>;  // {row, col}

inline size_t index(const Grid & g, Cell c)
{
  return static_cast<size_t>(c[0]) * static_cast<size_t>(g.width) + static_cast<size_t>(c[1]);
}

inline std::optional<Cell> cell_of(const Grid & g, double x, double y)
{
  const double col = std::floor((x - g.origin_x) / g.resolution);
  const double row = std::floor((y - g.origin_y) / g.resolution);
  if (!(row >= 0 && row < g.height && col >= 0 && col < g.width)) {
    return std::nullopt;
  }
  return Cell{static_cast<int>(row), static_cast<int>(col)};
}

// the cells of a straight (Bresenham) line from a to b, both included
inline std::vector<Cell> line_cells(Cell a, Cell b)
{
  std::vector<Cell> cells;
  const int dr = std::abs(b[0] - a[0]), dc = std::abs(b[1] - a[1]);
  const int sr = a[0] < b[0] ? 1 : -1, sc = a[1] < b[1] ? 1 : -1;
  int err = dc - dr;
  Cell p = a;
  while (true) {
    cells.push_back(p);
    if (p == b) {
      return cells;
    }
    const int e2 = 2 * err;
    if (e2 > -dr) {
      err -= dr;
      p[1] += sc;
    }
    if (e2 < dc) {
      err += dc;
      p[0] += sr;
    }
  }
}

// every cell of the line from a to b is allowed, and no diagonal step cuts a blocked corner
inline bool line_allowed(const Grid & g, const std::vector<uint8_t> & blocked, Cell a, Cell b)
{
  const auto cells = line_cells(a, b);
  for (size_t i = 0; i < cells.size(); ++i) {
    if (blocked[index(g, cells[i])]) {
      return false;
    }
    if (i > 0 && cells[i][0] != cells[i - 1][0] && cells[i][1] != cells[i - 1][1] &&
      (blocked[index(g, {cells[i][0], cells[i - 1][1]})] ||
      blocked[index(g, {cells[i - 1][0], cells[i][1]})]))
    {
      return false;
    }
  }
  return true;
}

}  // namespace maze_planning::detail
