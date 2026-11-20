#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

#include "maze_planning/grid_planner.hpp"

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

void wall_row(Grid & g, int row)
{
  for (int c = 0; c < g.width; ++c) {
    g.data[static_cast<size_t>(row * g.width + c)] = 100;
  }
}

void wall_col(Grid & g, int col, int row_from, int row_to)
{
  for (int r = row_from; r <= row_to; ++r) {
    g.data[static_cast<size_t>(r * g.width + col)] = 100;
  }
}

void wall_border(Grid & g)
{
  wall_row(g, 0);
  wall_row(g, g.height - 1);
  wall_col(g, 0, 0, g.height - 1);
  wall_col(g, g.width - 1, 0, g.height - 1);
}

// 40 x 11, walls along rows 0 and 10: rows 4-6 are open to a 25 mm ball
Grid channel()
{
  Grid g = free_grid(40, 11);
  wall_row(g, 0);
  wall_row(g, 10);
  return g;
}

Circle goal_at(const Grid & g, int row, int col)
{
  const auto c = cell_center(g, row, col);
  return {c[0], c[1], 0.01};
}

// two open rooms joined by a gap in a wall at column 20, rows gap_from..gap_to
Grid two_rooms(int gap_from, int gap_to)
{
  Grid g = free_grid(40, 21);
  wall_border(g);
  wall_col(g, 20, 0, gap_from - 1);
  wall_col(g, 20, gap_to + 1, 20);
  return g;
}

}  // namespace

TEST(GridPlannerTest, StraightChannelFollowsCentre)
{
  const Grid g = channel();
  const auto path = plan_grid_path(g, {}, cell_center(g, 5, 2), goal_at(g, 5, 37));
  ASSERT_TRUE(path.found);
  for (const auto & cell : path.cells) {
    EXPECT_EQ(cell[0], 5);
  }
}

TEST(GridPlannerTest, TooNarrowHasNoPath)
{
  const Grid narrow = two_rooms(9, 11);  // 15 mm gap: the 25 mm ball does not fit
  EXPECT_FALSE(plan_grid_path(narrow, {}, cell_center(narrow, 10, 5), goal_at(narrow, 10, 35)).found);
  const Grid wide = two_rooms(5, 15);   // 55 mm gap
  EXPECT_TRUE(plan_grid_path(wide, {}, cell_center(wide, 10, 5), goal_at(wide, 10, 35)).found);
}

TEST(GridPlannerTest, AvoidsHole)
{
  Grid g = free_grid(40, 21);
  wall_border(g);
  const auto h = cell_center(g, 10, 20);
  const Circle hole{h[0], h[1], 0.010};
  const auto path = plan_grid_path(g, {hole}, cell_center(g, 10, 5), goal_at(g, 10, 34));
  ASSERT_TRUE(path.found);
  for (const auto & p : path.points) {
    EXPECT_GE(std::hypot(p[0] - hole.x, p[1] - hole.y), 0.013);
  }
}

TEST(GridPlannerTest, NoCornerCutting)
{
  Grid g = free_grid(7, 7);
  g.data[3 * 7 + 2] = 100;
  g.data[2 * 7 + 3] = 100;
  PlannerParams p;
  p.ball_radius = 0.0;
  p.wall_margin = 0.0;
  const auto path = plan_grid_path(g, {}, cell_center(g, 2, 2), goal_at(g, 3, 3), p);
  ASSERT_TRUE(path.found);
  for (size_t i = 1; i < path.cells.size(); ++i) {
    const bool cut = path.cells[i - 1] == std::array<int, 2>{2, 2} && path.cells[i] == std::array<int, 2>{3, 3};
    EXPECT_FALSE(cut);
  }
}

TEST(GridPlannerTest, StartAgainstWallLeavesIt)
{
  const Grid g = channel();
  const auto start = cell_center(g, 1, 2);
  const auto path = plan_grid_path(g, {}, start, goal_at(g, 5, 37));
  ASSERT_TRUE(path.found);
  EXPECT_EQ(path.cells.front(), (std::array<int, 2>{1, 2}));
  EXPECT_EQ(path.points.front(), start);
  bool centre_by_col_10 = false;
  for (const auto & cell : path.cells) {
    if (cell[1] >= 10) {
      break;
    }
    centre_by_col_10 = centre_by_col_10 || cell[0] == 5;
  }
  EXPECT_TRUE(centre_by_col_10);
}

TEST(GridPlannerTest, StartInWallFails)
{
  const Grid g = channel();
  EXPECT_FALSE(plan_grid_path(g, {}, cell_center(g, 0, 2), goal_at(g, 5, 37)).found);
  EXPECT_FALSE(plan_grid_path(g, {}, {-1.0, 0.0275}, goal_at(g, 5, 37)).found);
}

TEST(GridPlannerTest, GoalBlockedFails)
{
  const Grid g = channel();
  EXPECT_FALSE(plan_grid_path(g, {}, cell_center(g, 5, 2), goal_at(g, 2, 37)).found);
}

TEST(GridPlannerTest, StartInGoalCell)
{
  const Grid g = channel();
  const Circle goal = goal_at(g, 5, 37);
  const auto path = plan_grid_path(g, {}, {goal.x + 0.001, goal.y}, goal);
  ASSERT_TRUE(path.found);
  EXPECT_EQ(path.cells.size(), 1u);
  ASSERT_EQ(path.points.size(), 2u);
  EXPECT_EQ(path.points.back(), (std::array<double, 2>{goal.x, goal.y}));
}

TEST(GridPlannerTest, NonSquareRowsAndCols)
{
  Grid g = free_grid(40, 20);
  wall_border(g);
  PlannerParams p;
  p.ball_radius = 0.0;
  p.wall_margin = 0.0;
  const auto path = plan_grid_path(g, {}, cell_center(g, 2, 2), goal_at(g, 17, 37), p);
  ASSERT_TRUE(path.found);
  EXPECT_EQ(path.cells.back(), (std::array<int, 2>{17, 37}));
  for (const auto & pt : path.points) {
    EXPECT_GE(pt[0], 0.0);
    EXPECT_LE(pt[0], 0.2);
    EXPECT_GE(pt[1], 0.0);
    EXPECT_LE(pt[1], 0.1);
  }
}

TEST(GridPlannerTest, RejectsBadParams)
{
  const Grid g = channel();
  const auto start = cell_center(g, 5, 2);
  const Circle goal = goal_at(g, 5, 37);
  PlannerParams zero_preferred;
  zero_preferred.preferred_clearance = 0.0;
  EXPECT_THROW(plan_grid_path(g, {}, start, goal, zero_preferred), std::invalid_argument);
  PlannerParams negative_weight;
  negative_weight.clearance_weight = -1.0;
  EXPECT_THROW(plan_grid_path(g, {}, start, goal, negative_weight), std::invalid_argument);
  PlannerParams negative_radius;
  negative_radius.ball_radius = -0.01;
  EXPECT_THROW(plan_grid_path(g, {}, start, goal, negative_radius), std::invalid_argument);
}
