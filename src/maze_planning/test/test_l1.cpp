// The planner on the real L1 maze: the CAD map exported by scripts/export_fixture.py.
#include <gtest/gtest.h>

#include <cmath>

#include "maze_planning/clearance.hpp"
#include "maze_planning/grid_planner.hpp"
#include "l1_fixture.hpp"

using namespace maze_planning;

TEST(L1Test, RouteReachesGoalWithClearance)
{
  const Fixture f = load_fixture("L1");
  const PlannerParams params;
  const auto path = plan_grid_path(f.grid, f.holes, f.start, f.goal, params);
  ASSERT_TRUE(path.found);
  EXPECT_EQ(path.points.back(), (std::array<double, 2>{f.goal.x, f.goal.y}));

  // levels.py's route is the line drawn on the schematic, not the shortest one: it zigzags through
  // the top row. The planner must go through the same wall openings, never take a longer way.
  // Route cells 6 = (2, 5) and 11 = (3, 0) are L1's only gaps in its two long walls.
  for (size_t i : {6u, 11u}) {
    double nearest = 1e9;
    for (const auto & p : path.points) {
      nearest = std::min(nearest, std::hypot(p[0] - f.route[i][0], p[1] - f.route[i][1]));
    }
    EXPECT_LE(nearest, 0.021) << "route cell " << i;
  }
  auto length = [](const std::vector<std::array<double, 2>> & pts) {
      double sum = 0.0;
      for (size_t i = 1; i < pts.size(); ++i) {
        sum += std::hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]);
      }
      return sum;
    };
  EXPECT_LE(length(path.points), length(f.route));

  // the 25 mm ball never comes closer than its radius plus the margin to a wall
  const auto clearance = clearance_map(f.grid);
  for (const auto & c : path.cells) {
    EXPECT_GE(clearance[static_cast<size_t>(c[0] * f.grid.width + c[1])],
              params.ball_radius + params.wall_margin - 1e-9);
  }
}
