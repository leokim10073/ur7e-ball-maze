// The Contract 3 route: straightened points, arc length, corridors and the point-to-corridor index.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "grids.hpp"
#include "l1_fixture.hpp"
#include "maze_planning/blocked.hpp"
#include "maze_planning/clearance.hpp"
#include "maze_planning/grid_planner.hpp"
#include "maze_planning/route.hpp"

using namespace maze_planning;

namespace
{

constexpr double kEps = 1e-12;

bool inside(const Corridor & k, const std::array<double, 2> & p)
{
  return p[0] >= k.x_min - kEps && p[0] <= k.x_max + kEps && p[1] >= k.y_min - kEps &&
         p[1] <= k.y_max + kEps;
}

double dist(const std::array<double, 2> & a, const std::array<double, 2> & b)
{
  return std::hypot(a[0] - b[0], a[1] - b[1]);
}

void expect_consistent(const Route & r)
{
  ASSERT_EQ(r.arc_length.size(), r.points.size());
  ASSERT_EQ(r.pose_corridor.size(), r.points.size());
  for (size_t i = 0; i < r.points.size(); ++i) {
    ASSERT_LT(r.pose_corridor[i], r.corridors.size());
    EXPECT_TRUE(inside(r.corridors[r.pose_corridor[i]], r.points[i])) << "point " << i;
    EXPECT_FALSE(std::isnan(r.points[i][0]) || std::isnan(r.points[i][1]) || std::isnan(r.arc_length[i]));
    if (i > 0) {
      EXPECT_GT(r.arc_length[i], r.arc_length[i - 1]) << "point " << i;
      EXPECT_NEAR(r.arc_length[i] - r.arc_length[i - 1], dist(r.points[i], r.points[i - 1]), 1e-12);
      EXPECT_GE(r.pose_corridor[i], r.pose_corridor[i - 1]);
    }
  }
  if (!r.arc_length.empty()) {
    EXPECT_EQ(r.arc_length.front(), 0.0);
  }
}

}  // namespace

TEST(RouteTest, StraightChannel)
{
  const Grid g = channel();
  const auto start = cell_center(g, 5, 2);
  const Circle goal = goal_at(g, 5, 37);
  const auto path = plan_grid_path(g, {}, start, goal);
  ASSERT_TRUE(path.found);
  const Route r = build_route(g, {}, path);
  expect_consistent(r);
  EXPECT_EQ(r.corridors.size(), 1u);
  for (size_t i = 1; i < r.points.size(); ++i) {
    EXPECT_LE(dist(r.points[i], r.points[i - 1]), 0.005 + kEps);
  }
  EXPECT_NEAR(r.arc_length.back(), dist(start, {goal.x, goal.y}), 1e-9);
  for (auto k : r.pose_corridor) {
    EXPECT_EQ(k, 0u);
  }
  EXPECT_EQ(r.points.front(), start);
  EXPECT_EQ(r.points.back(), (std::array<double, 2>{goal.x, goal.y}));
}

TEST(RouteTest, PushedStartInsideFirstCorridor)
{
  const Grid g = channel();
  const auto start = cell_center(g, 1, 2);  // 2.5 mm from the wall
  const auto path = plan_grid_path(g, {}, start, goal_at(g, 5, 37));
  ASSERT_TRUE(path.found);
  const Route r = build_route(g, {}, path);
  expect_consistent(r);
  ASSERT_FALSE(r.points.empty());
  EXPECT_EQ(r.points.front(), start);
  EXPECT_TRUE(inside(r.corridors[r.pose_corridor[0]], start));
  // the recovery region must not be a bare line: a ball 1 mm to either side is still in it
  EXPECT_TRUE(inside(r.corridors[0], {start[0] + 0.001, start[1]}));
  EXPECT_TRUE(inside(r.corridors[0], {start[0] - 0.001, start[1]}));
}

TEST(RouteTest, NotFoundGivesEmpty)
{
  const Grid g = channel();
  GridPath none;
  const Route a = build_route(g, {}, none);
  EXPECT_TRUE(a.points.empty() && a.arc_length.empty() && a.corridors.empty() && a.pose_corridor.empty());

  GridPath bad;
  bad.found = true;
  bad.cells = {{5, 2}, {5, 3}, {5, 4}};
  bad.points = {cell_center(g, 5, 2), cell_center(g, 5, 4)};
  const Route b = build_route(g, {}, bad);
  EXPECT_TRUE(b.points.empty() && b.arc_length.empty() && b.corridors.empty() && b.pose_corridor.empty());
}

TEST(RouteTest, StartInGoalCell)
{
  const Grid g = channel();
  const Circle goal = goal_at(g, 5, 20);
  const std::array<double, 2> start{goal.x + 0.001, goal.y};
  const auto path = plan_grid_path(g, {}, start, goal);
  ASSERT_TRUE(path.found);
  const Route r = build_route(g, {}, path);
  expect_consistent(r);
  ASSERT_EQ(r.points.size(), 2u);
  EXPECT_EQ(r.corridors.size(), 1u);
  EXPECT_NEAR(r.arc_length[0], 0.0, 1e-12);
  EXPECT_NEAR(r.arc_length[1], 0.001, 1e-12);
}

TEST(RouteTest, StartAtGoalCenter)
{
  const Grid g = channel();
  const Circle goal = goal_at(g, 5, 20);
  const auto path = plan_grid_path(g, {}, {goal.x, goal.y}, goal);
  ASSERT_TRUE(path.found);
  const Route r = build_route(g, {}, path);
  expect_consistent(r);  // no NaN, strictly increasing arc length
  ASSERT_FALSE(r.points.empty());
  EXPECT_FALSE(r.corridors.empty());
}

TEST(RouteTest, RejectsBadSpacing)
{
  const Grid g = channel();
  const auto path = plan_grid_path(g, {}, cell_center(g, 5, 2), goal_at(g, 5, 37));
  PlannerParams p;
  p.point_spacing = 0.0;
  EXPECT_THROW(build_route(g, {}, path, p), std::invalid_argument);
  p.point_spacing = -0.005;
  EXPECT_THROW(build_route(g, {}, path, p), std::invalid_argument);
}

TEST(RouteTest, L1Properties)
{
  const Fixture f = load_fixture("L1");
  const PlannerParams p;
  const auto path = plan_grid_path(f.grid, f.holes, f.start, f.goal, p);
  ASSERT_TRUE(path.found);
  const Route r = build_route(f.grid, f.holes, path, p);
  expect_consistent(r);
  ASSERT_GE(r.points.size(), 2u);
  EXPECT_EQ(r.points.front(), f.start);
  EXPECT_EQ(r.points.back(), (std::array<double, 2>{f.goal.x, f.goal.y}));

  const auto clearance = clearance_map(f.grid);
  const auto blocked = blocked_cells(f.grid, clearance, f.holes, p);
  auto at = [&](int row, int col) {return static_cast<size_t>(row * f.grid.width + col);};
  const bool lead_in = blocked[at(path.cells[0][0], path.cells[0][1])];

  for (size_t k = 0; k < r.corridors.size(); ++k) {
    const Corridor & c = r.corridors[k];
    if (k > 0) {
      const Corridor & b = r.corridors[k - 1];
      EXPECT_TRUE(std::max(b.x_min, c.x_min) <= std::min(b.x_max, c.x_max) + kEps &&
                  std::max(b.y_min, c.y_min) <= std::min(b.y_max, c.y_max) + kEps)
        << "corridors " << k - 1 << " and " << k << " do not overlap";
    }
    if (k == 0 && lead_in) {
      continue;
    }
    for (int row = 0; row < f.grid.height; ++row) {
      for (int col = 0; col < f.grid.width; ++col) {
        const auto m = cell_center(f.grid, row, col);
        if (inside(c, m)) {
          EXPECT_FALSE(blocked[at(row, col)]) << "corridor " << k << " has blocked cell " << row << "," << col;
        }
      }
    }
  }

  double a_star_length = 0.0, a_star_min = 1e9;
  for (size_t i = 1; i < path.points.size(); ++i) {
    a_star_length += dist(path.points[i], path.points[i - 1]);
  }
  // how close A* came to a wall: its cells, and the corner cells its diagonal steps cross
  for (size_t i = 0; i < path.cells.size(); ++i) {
    const auto & c = path.cells[i];
    a_star_min = std::min(a_star_min, clearance[at(c[0], c[1])]);
    if (i > 0 && !blocked[at(path.cells[i - 1][0], path.cells[i - 1][1])]) {
      const auto & b = path.cells[i - 1];
      a_star_min = std::min({a_star_min, clearance[at(c[0], b[1])], clearance[at(b[0], c[1])]});
    }
  }
  EXPECT_LE(r.arc_length.back(), a_star_length + 1e-12);
  for (size_t i = 0; i < r.points.size(); ++i) {
    const int col = static_cast<int>(std::floor((r.points[i][0] - f.grid.origin_x) / f.grid.resolution));
    const int row = static_cast<int>(std::floor((r.points[i][1] - f.grid.origin_y) / f.grid.resolution));
    EXPECT_GE(clearance[at(row, col)], std::min(p.preferred_clearance, a_star_min) - 1e-9) << "point " << i;
  }
  EXPECT_LT(r.corridors.size(), path.cells.size());
}
