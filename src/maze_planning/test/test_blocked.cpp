#include <gtest/gtest.h>

#include "maze_planning/blocked.hpp"
#include "maze_planning/clearance.hpp"

using namespace maze_planning;

namespace
{

Grid free_grid(int width, int height)
{
  Grid g;
  g.width = width;
  g.height = height;
  g.resolution = 0.005;
  g.data.assign(static_cast<size_t>(width * height), 0);
  return g;
}

uint8_t at(const std::vector<uint8_t> & b, const Grid & g, int row, int col)
{
  return b[static_cast<size_t>(row * g.width + col)];
}

}  // namespace

TEST(BlockedTest, ChannelBlocksNearWalls)
{
  Grid g = free_grid(20, 11);
  for (int c = 0; c < g.width; ++c) {
    g.data[c] = 100;
    g.data[10 * g.width + c] = 100;
  }
  PlannerParams p;
  p.ball_radius = 0.0125;
  p.wall_margin = 0.003;
  const auto b = blocked_cells(g, clearance_map(g), {}, p);
  for (int r = 0; r < g.height; ++r) {
    const uint8_t expected = (r >= 4 && r <= 6) ? 0 : 1;
    EXPECT_EQ(at(b, g, r, 10), expected) << "row " << r;
  }
}

TEST(BlockedTest, HoleKeepOut)
{
  const Grid g = free_grid(21, 21);
  const auto center = cell_center(g, 10, 10);
  PlannerParams p;
  p.hole_margin = 0.003;
  const auto b = blocked_cells(g, clearance_map(g), {{center[0], center[1], 0.010}}, p);
  EXPECT_EQ(at(b, g, 10, 12), 1);  // 0.010 from the hole center
  EXPECT_EQ(at(b, g, 10, 13), 0);  // 0.015
}

TEST(BlockedTest, ObstacleAlwaysBlocked)
{
  Grid g = free_grid(9, 9);
  g.data[4 * 9 + 4] = 100;
  PlannerParams p;
  p.ball_radius = 0.0;
  p.wall_margin = 0.0;
  const auto b = blocked_cells(g, clearance_map(g), {}, p);
  EXPECT_EQ(at(b, g, 4, 4), 1);
  EXPECT_EQ(at(b, g, 4, 5), 0);
}
