#include <algorithm>

#include "geometry.hpp"
#include "maze_planning/route.hpp"

namespace maze_planning
{

namespace
{

using detail::Cell;

// no blocked cell in rows r0..r1 x cols c0..c1 (inclusive, any order)
bool box_free(const Grid & g, const std::vector<uint8_t> & blocked, int r0, int c0, int r1, int c1)
{
  for (int r = std::min(r0, r1); r <= std::max(r0, r1); ++r) {
    for (int c = std::min(c0, c1); c <= std::max(c0, c1); ++c) {
      if (blocked[detail::index(g, {r, c})]) {
        return false;
      }
    }
  }
  return true;
}

// split line[i..j] at its middle cell until the box of its ends is free; the split cells stay on
// the original line, whose cells and corners were checked when it was straightened
void split(
  const Grid & g, const std::vector<uint8_t> & blocked, const std::vector<Cell> & line, size_t i,
  size_t j, std::vector<std::array<int, 4>> & out)
{
  const Cell a = line[i], b = line[j];
  // a single step has no middle cell to split at; without corner cutting its box is free anyway
  if (j - i <= 1 || box_free(g, blocked, a[0], a[1], b[0], b[1])) {
    out.push_back({a[0], a[1], b[0], b[1]});
    return;
  }
  const size_t mid = i + (j - i) / 2;
  split(g, blocked, line, i, mid, out);
  split(g, blocked, line, mid, j, out);
}

}  // namespace

std::vector<std::array<int, 4>> split_segments(
  const Grid & grid, const std::vector<uint8_t> & blocked,
  const std::vector<std::array<int, 2>> & vertices)
{
  std::vector<std::array<int, 4>> out;
  for (size_t k = 1; k < vertices.size(); ++k) {
    const auto line = detail::line_cells(vertices[k - 1], vertices[k]);
    split(grid, blocked, line, 0, line.size() - 1, out);
  }
  return out;
}

Corridor grow_corridor(
  const Grid & grid, const std::vector<uint8_t> & blocked, int r0, int c0, int r1, int c1)
{
  int rlo = std::min(r0, r1), rhi = std::max(r0, r1);
  int clo = std::min(c0, c1), chi = std::max(c0, c1);
  bool grew = true;
  while (grew) {
    grew = false;
    if (chi + 1 < grid.width && box_free(grid, blocked, rlo, chi + 1, rhi, chi + 1)) {
      ++chi;
      grew = true;
    }
    if (rhi + 1 < grid.height && box_free(grid, blocked, rhi + 1, clo, rhi + 1, chi)) {
      ++rhi;
      grew = true;
    }
    if (clo > 0 && box_free(grid, blocked, rlo, clo - 1, rhi, clo - 1)) {
      --clo;
      grew = true;
    }
    if (rlo > 0 && box_free(grid, blocked, rlo - 1, clo, rlo - 1, chi)) {
      --rlo;
      grew = true;
    }
  }
  const auto lo = cell_center(grid, rlo, clo);
  const auto hi = cell_center(grid, rhi, chi);
  return {lo[0], hi[0], lo[1], hi[1]};
}

}  // namespace maze_planning
