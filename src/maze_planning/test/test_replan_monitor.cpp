// When the planner replans: the ball outside every corridor for a few frames, or a no-path retry.
#include <gtest/gtest.h>

#include <stdexcept>

#include "maze_planning/replan_monitor.hpp"

using namespace maze_planning;

namespace
{

Route two_corridors()
{
  Route r;
  r.points = {{0.01, 0.01}, {0.09, 0.15}};
  r.arc_length = {0.0, 0.16};
  r.corridors = {{0.0, 0.1, 0.0, 0.02}, {0.08, 0.1, 0.0, 0.2}};
  r.pose_corridor = {0, 1};
  return r;
}

constexpr double kOutX = 0.05, kOutY = 0.03;  // 10 mm above corridor 0, left of corridor 1

}  // namespace

TEST(ReplanMonitorTest, InsideNeverReplans)
{
  ReplanMonitor m(ReplanParams{});
  m.route_planned(two_corridors(), 0.0);
  for (int i = 0; i < 10; ++i) {
    EXPECT_FALSE(m.should_replan(0.05, 0.01, 0.1 * i));
  }
}

TEST(ReplanMonitorTest, WithinMarginNeverReplans)
{
  ReplanMonitor m(ReplanParams{});
  m.route_planned(two_corridors(), 0.0);
  for (int i = 0; i < 10; ++i) {
    EXPECT_FALSE(m.should_replan(0.05, 0.022, 0.1 * i));  // 2 mm out, margin 3 mm
  }
}

TEST(ReplanMonitorTest, ShortExcursionResets)
{
  ReplanMonitor m(ReplanParams{});
  m.route_planned(two_corridors(), 0.0);
  EXPECT_FALSE(m.should_replan(kOutX, kOutY, 0.1));
  EXPECT_FALSE(m.should_replan(kOutX, kOutY, 0.2));
  EXPECT_FALSE(m.should_replan(0.05, 0.01, 0.3));
  EXPECT_FALSE(m.should_replan(kOutX, kOutY, 0.4));
  EXPECT_FALSE(m.should_replan(kOutX, kOutY, 0.5));
}

TEST(ReplanMonitorTest, ReplansAfterFramesThenRestarts)
{
  ReplanMonitor m(ReplanParams{});
  m.route_planned(two_corridors(), 0.0);
  EXPECT_FALSE(m.should_replan(kOutX, kOutY, 0.1));
  EXPECT_FALSE(m.should_replan(kOutX, kOutY, 0.2));
  EXPECT_TRUE(m.should_replan(kOutX, kOutY, 0.3));
  m.route_planned(two_corridors(), 0.3);
  EXPECT_FALSE(m.should_replan(kOutX, kOutY, 0.4));
  EXPECT_FALSE(m.should_replan(kOutX, kOutY, 0.5));
  EXPECT_TRUE(m.should_replan(kOutX, kOutY, 0.6));
}

TEST(ReplanMonitorTest, NoPathRetriesAfterPeriod)
{
  ReplanMonitor m(ReplanParams{});
  m.route_planned(Route{}, 1.0);
  EXPECT_FALSE(m.should_replan(0.05, 0.01, 1.4));
  EXPECT_TRUE(m.should_replan(0.05, 0.01, 1.5));
  m.route_planned(Route{}, 1.5);
  EXPECT_FALSE(m.should_replan(0.05, 0.01, 1.6));
}

TEST(ReplanMonitorTest, NothingPlannedNeverReplans)
{
  ReplanMonitor m(ReplanParams{});
  for (int i = 0; i < 5; ++i) {
    EXPECT_FALSE(m.should_replan(kOutX, kOutY, 0.1 * i));
  }
}

TEST(ReplanMonitorTest, RejectsBadParams)
{
  ReplanParams margin;
  margin.outside_margin = -0.001;
  EXPECT_THROW(ReplanMonitor{margin}, std::invalid_argument);
  ReplanParams frames;
  frames.outside_frames = 0;
  EXPECT_THROW(ReplanMonitor{frames}, std::invalid_argument);
  ReplanParams period;
  period.no_path_retry_period = 0.0;
  EXPECT_THROW(ReplanMonitor{period}, std::invalid_argument);
}
