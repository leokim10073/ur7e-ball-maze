#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

#include "maze_planning/clearance.hpp"

using maze_planning::Grid;
using maze_planning::clearance_map;

namespace
{

constexpr double kRes = 0.005;

Grid free_grid(int width, int height)
{
  Grid g;
  g.width = width;
  g.height = height;
  g.resolution = kRes;
  g.data.assign(static_cast<size_t>(width * height), 0);
  return g;
}

double at(const std::vector<double> & c, const Grid & g, int row, int col)
{
  return c[static_cast<size_t>(row * g.width + col)];
}

}  // namespace

TEST(ClearanceTest, SingleWallNeighbours)
{
  Grid g = free_grid(5, 5);
  g.data[2 * 5 + 2] = 100;
  const auto c = clearance_map(g);
  EXPECT_NEAR(at(c, g, 2, 3), 0.0025, 1e-12);
  EXPECT_NEAR(at(c, g, 3, 3), std::sqrt(2.0) * kRes - 0.0025, 1e-12);
  EXPECT_NEAR(at(c, g, 2, 2), 0.0, 1e-12);
  EXPECT_NEAR(at(c, g, 0, 0), std::sqrt(8.0) * kRes - 0.0025, 1e-12);
}

TEST(ClearanceTest, UnknownIsObstacle)
{
  Grid g = free_grid(5, 5);
  g.data[2 * 5 + 2] = -1;
  const auto c = clearance_map(g);
  EXPECT_NEAR(at(c, g, 2, 3), 0.0025, 1e-12);
}

TEST(ClearanceTest, NoObstacleIsInfinite)
{
  const Grid g = free_grid(4, 4);
  for (double v : clearance_map(g)) {
    EXPECT_TRUE(std::isinf(v));
    EXPECT_GT(v, 0.0);
  }
}

TEST(ClearanceTest, MatchesBruteForceNonSquare)
{
  Grid g = free_grid(40, 20);
  std::mt19937 rng(7);
  std::bernoulli_distribution wall(0.15);
  for (auto & v : g.data) {
    v = wall(rng) ? 100 : 0;
  }
  const auto c = clearance_map(g);
  for (int r = 0; r < g.height; ++r) {
    for (int col = 0; col < g.width; ++col) {
      double best = std::numeric_limits<double>::infinity();
      for (int r2 = 0; r2 < g.height; ++r2) {
        for (int c2 = 0; c2 < g.width; ++c2) {
          if (g.data[static_cast<size_t>(r2 * g.width + c2)] != 0) {
            best = std::min(best, std::hypot(r - r2, col - c2));
          }
        }
      }
      const double expected = best == 0.0 ? 0.0 : best * kRes - kRes / 2;
      EXPECT_NEAR(at(c, g, r, col), expected, 1e-12) << "row " << r << " col " << col;
    }
  }
}

TEST(ClearanceTest, RejectsBadGrid)
{
  Grid g = free_grid(5, 5);
  g.data.resize(24);
  EXPECT_THROW(clearance_map(g), std::invalid_argument);
  Grid h = free_grid(5, 5);
  h.resolution = 0.0;
  EXPECT_THROW(clearance_map(h), std::invalid_argument);
}
