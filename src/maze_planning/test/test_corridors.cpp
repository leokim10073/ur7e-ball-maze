// Splitting straight segments into boxes free of blocked cells and growing them into corridors.
#include <gtest/gtest.h>

#include <algorithm>

#include "grids.hpp"
#include "maze_planning/blocked.hpp"
#include "maze_planning/clearance.hpp"
#include "maze_planning/route.hpp"

using namespace maze_planning;

namespace
{

std::vector<uint8_t> blocked_of(const Grid & g, const PlannerParams & p)
{
  return blocked_cells(g, clearance_map(g), {}, p);
}

PlannerParams point_ball()
{
  PlannerParams p;
  p.ball_radius = 0.0;
  p.wall_margin = 0.0;
  return p;
}

bool box_free(const Grid & g, const std::vector<uint8_t> & blocked, const std::array<int, 4> & s)
{
  for (int r = std::min(s[0], s[2]); r <= std::max(s[0], s[2]); ++r) {
    for (int c = std::min(s[1], s[3]); c <= std::max(s[1], s[3]); ++c) {
      if (blocked[static_cast<size_t>(r * g.width + c)]) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

TEST(CorridorTest, StraightChannelSpansFreeRows)
{
  const Grid g = channel();
  const auto blocked = blocked_of(g, PlannerParams{});
  const Corridor k = grow_corridor(g, blocked, 5, 2, 5, 37);
  EXPECT_NEAR(k.y_min, 0.0225, 1e-12);
  EXPECT_NEAR(k.y_max, 0.0325, 1e-12);
  EXPECT_NEAR(k.x_min, 0.0025, 1e-12);
  EXPECT_NEAR(k.x_max, 0.1975, 1e-12);
}

TEST(CorridorTest, GrowsToGridEdgeWithoutWalls)
{
  const Grid g = free_grid(10, 10);
  const auto blocked = blocked_of(g, point_ball());
  const Corridor k = grow_corridor(g, blocked, 5, 5, 5, 5);
  EXPECT_NEAR(k.x_min, 0.0025, 1e-12);
  EXPECT_NEAR(k.x_max, 0.0475, 1e-12);
  EXPECT_NEAR(k.y_min, 0.0025, 1e-12);
  EXPECT_NEAR(k.y_max, 0.0475, 1e-12);
}

TEST(CorridorTest, SplitsDiagonalAroundBlock)
{
  Grid g = free_grid(30, 30);
  wall_block(g, 12, 17, 3, 8);
  const auto blocked = blocked_of(g, point_ball());
  const auto segments = split_segments(g, blocked, {{2, 2}, {27, 27}});
  ASSERT_GE(segments.size(), 2u);
  EXPECT_EQ(segments.front()[0], 2);
  EXPECT_EQ(segments.front()[1], 2);
  EXPECT_EQ(segments.back()[2], 27);
  EXPECT_EQ(segments.back()[3], 27);
  for (size_t k = 0; k < segments.size(); ++k) {
    EXPECT_TRUE(box_free(g, blocked, segments[k])) << "segment " << k;
    if (k > 0) {
      EXPECT_EQ(segments[k - 1][2], segments[k][0]) << "segment " << k;
      EXPECT_EQ(segments[k - 1][3], segments[k][1]) << "segment " << k;
    }
  }
}
