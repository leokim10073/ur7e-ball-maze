// Straightening the A* staircase only where the straight line keeps the clearance A* had.
#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <random>

#include "geometry.hpp"
#include "grids.hpp"
#include "l1_fixture.hpp"
#include "maze_planning/blocked.hpp"
#include "maze_planning/clearance.hpp"
#include "maze_planning/grid_planner.hpp"
#include "maze_planning/route.hpp"

using namespace maze_planning;

namespace
{

struct Maps
{
  std::vector<double> clearance;
  std::vector<uint8_t> blocked;
};

Maps maps(const Grid & g, const std::vector<Circle> & holes, const PlannerParams & p)
{
  Maps m;
  m.clearance = clearance_map(g);
  m.blocked = blocked_cells(g, m.clearance, holes, p);
  return m;
}

size_t at(const Grid & g, std::array<int, 2> c)
{
  return static_cast<size_t>(c[0] * g.width + c[1]);
}

}  // namespace

TEST(ShortcutTest, OpenRoomDiagonalIsOneSegment)
{
  Grid g = free_grid(40, 40);
  wall_border(g);
  const PlannerParams p;
  const auto m = maps(g, {}, p);
  const auto path = plan_grid_path(g, {}, cell_center(g, 10, 10), goal_at(g, 29, 29), p);
  ASSERT_TRUE(path.found);
  EXPECT_EQ(shortcut_cells(g, m.clearance, m.blocked, path.cells, p).size(), 2u);
}

TEST(ShortcutTest, StraightChannelIsOneSegment)
{
  const Grid g = channel();
  const PlannerParams p;
  const auto m = maps(g, {}, p);
  const auto path = plan_grid_path(g, {}, cell_center(g, 5, 2), goal_at(g, 5, 37), p);
  ASSERT_TRUE(path.found);
  const auto v = shortcut_cells(g, m.clearance, m.blocked, path.cells, p);
  ASSERT_EQ(v.size(), 2u);
  EXPECT_EQ(v.front(), path.cells.front());
  EXPECT_EQ(v.back(), path.cells.back());
}

TEST(ShortcutTest, KeepsAStarClearance)
{
  const Fixture f = load_fixture("L1");
  const PlannerParams p;
  const auto m = maps(f.grid, f.holes, p);
  const auto path = plan_grid_path(f.grid, f.holes, f.start, f.goal, p);
  ASSERT_TRUE(path.found);
  std::vector<std::array<int, 2>> cells;
  for (const auto & c : path.cells) {
    if (cells.empty() && m.blocked[at(f.grid, c)]) {
      continue;  // the lead-in is not straightened
    }
    cells.push_back(c);
  }
  const auto v = shortcut_cells(f.grid, m.clearance, m.blocked, cells, p);
  ASSERT_GE(v.size(), 2u);
  EXPECT_LT(v.size(), cells.size());
  size_t from = 0;
  for (size_t k = 1; k < v.size(); ++k) {
    const auto it = std::find(cells.begin() + static_cast<long>(from), cells.end(), v[k]);
    ASSERT_NE(it, cells.end()) << "vertex " << k << " is not a later A* cell";
    const size_t to = static_cast<size_t>(it - cells.begin());
    double a_star_min = 1e9;
    for (size_t i = from; i <= to; ++i) {
      a_star_min = std::min(a_star_min, m.clearance[at(f.grid, cells[i])]);
    }
    for (const auto & c : detail::line_cells(v[k - 1], v[k])) {
      EXPECT_GE(m.clearance[at(f.grid, c)], std::min(p.preferred_clearance, a_star_min) - 1e-9)
        << "segment " << k;
    }
    from = to;
  }
}

TEST(ShortcutTest, LShapeKeepsCorner)
{
  Grid g = free_grid(30, 30);
  wall_border(g);
  wall_block(g, 0, 19, 10, 29);  // free: the left leg (cols 1-9) and the top leg (rows 20-28)
  const PlannerParams p;
  const auto m = maps(g, {}, p);
  const auto path = plan_grid_path(g, {}, cell_center(g, 5, 5), goal_at(g, 25, 25), p);
  ASSERT_TRUE(path.found);
  const auto v = shortcut_cells(g, m.clearance, m.blocked, path.cells, p);
  ASSERT_GE(v.size(), 3u);
  for (size_t k = 1; k < v.size(); ++k) {
    EXPECT_TRUE(detail::line_allowed(g, m.blocked, v[k - 1], v[k])) << "segment " << k;
  }
}

TEST(ShortcutTest, CornerCellsKeepAStarClearance)
{
  // rooms with random wall blocks (fixed seed): wherever a straightened line takes a diagonal
  // step, the two corner cells it crosses are as far from walls as A* came in between
  std::mt19937 rng(7);
  int checked = 0;
  for (int trial = 0; trial < 200; ++trial) {
    Grid g = free_grid(40, 40);
    wall_border(g);
    for (int b = 0; b < 6; ++b) {
      const int r = 4 + static_cast<int>(rng() % 30), c = 4 + static_cast<int>(rng() % 30);
      wall_block(g, r, r + static_cast<int>(rng() % 3), c, c + static_cast<int>(rng() % 3));
    }
    const PlannerParams p;
    const auto m = maps(g, {}, p);
    const auto path = plan_grid_path(g, {}, cell_center(g, 6, 6), goal_at(g, 33, 33), p);
    if (!path.found || m.blocked[at(g, path.cells.front())]) {
      continue;
    }
    const auto v = shortcut_cells(g, m.clearance, m.blocked, path.cells, p);
    size_t from = 0;
    for (size_t k = 1; k < v.size(); ++k) {
      const auto it = std::find(path.cells.begin() + static_cast<std::ptrdiff_t>(from), path.cells.end(), v[k]);
      ASSERT_NE(it, path.cells.end());
      const size_t to = static_cast<size_t>(it - path.cells.begin());
      double a_star_min = 1e9;
      for (size_t i = from; i <= to; ++i) {  // A*'s cells and the corner cells its steps cross
        const auto & c = path.cells[i];
        a_star_min = std::min(a_star_min, m.clearance[at(g, c)]);
        if (i > from) {
          const auto & b = path.cells[i - 1];
          a_star_min = std::min({a_star_min, m.clearance[at(g, {c[0], b[1]})], m.clearance[at(g, {b[0], c[1]})]});
        }
      }
      const double needed = std::min(p.preferred_clearance, a_star_min) - 1e-9;
      const auto line = detail::line_cells(v[k - 1], v[k]);
      for (size_t i = 1; i < line.size(); ++i) {
        EXPECT_GE(m.clearance[at(g, {line[i][0], line[i - 1][1]})], needed) << "trial " << trial;
        EXPECT_GE(m.clearance[at(g, {line[i - 1][0], line[i][1]})], needed) << "trial " << trial;
      }
      from = to;
    }
    ++checked;
  }
  EXPECT_GT(checked, 100);
}
