// The planner node in-process: L1 map and ball states in, Contract 3 out (/maze/route, /maze/path).
#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "l1_fixture.hpp"
#include "planner_node.hpp"
#include "ros_convert.hpp"

using namespace maze_planning;
using maze_interfaces::msg::BallState;
using maze_interfaces::msg::MazeMap;
using maze_interfaces::msg::PlannedRoute;
using namespace std::chrono_literals;

namespace
{

std::vector<rclcpp::Parameter> yaml_params(const std::string & skip = "")
{
  std::vector<rclcpp::Parameter> all{
    {"ball_radius", 0.0125}, {"wall_margin", 0.003}, {"hole_margin", 0.003},
    {"preferred_clearance", 0.020}, {"clearance_weight", 2.0}, {"point_spacing", 0.005},
    {"replan.outside_margin", 0.003}, {"replan.outside_frames", 3},
    {"replan.no_path_retry_period", 0.5}, {"use_sim_time", false}};
  std::vector<rclcpp::Parameter> out;
  for (const auto & p : all) {
    if (p.get_name() != skip) {
      out.push_back(p);
    }
  }
  return out;
}

nav_msgs::msg::OccupancyGrid to_occupancy(const Grid & g)
{
  nav_msgs::msg::OccupancyGrid o;
  o.header.frame_id = "maze_board";
  o.info.width = static_cast<uint32_t>(g.width);
  o.info.height = static_cast<uint32_t>(g.height);
  o.info.resolution = static_cast<float>(g.resolution);
  o.info.origin.position.x = g.origin_x;
  o.info.origin.position.y = g.origin_y;
  o.info.origin.orientation.w = 1.0;
  o.data = g.data;
  return o;
}

MazeMap make_map(const Fixture & f)
{
  MazeMap m;
  m.header.frame_id = "maze_board";
  m.source = "cad:L1";
  m.walls = to_occupancy(f.grid);
  for (const auto & h : f.holes) {
    maze_interfaces::msg::Circle c;
    c.x = h.x;
    c.y = h.y;
    c.radius = h.radius;
    m.holes.push_back(c);
  }
  m.start.x = f.start[0];
  m.start.y = f.start[1];
  m.goal.x = f.goal.x;
  m.goal.y = f.goal.y;
  m.goal.radius = f.goal.radius;
  return m;
}

BallState ball(double x, double y, bool ok = true)
{
  BallState b;
  b.header.frame_id = "maze_board";
  b.x = x;
  b.y = y;
  b.tracking_ok = ok;
  return b;
}

class PlannerNodeTest : public ::testing::Test
{
protected:
  static void SetUpTestSuite() {rclcpp::init(0, nullptr);}
  static void TearDownTestSuite() {rclcpp::shutdown();}

  void SetUp() override
  {
    fixture = load_fixture("L1");
    planner = std::make_shared<PlannerNode>(rclcpp::NodeOptions().parameter_overrides(yaml_params()));
    tester = std::make_shared<rclcpp::Node>("planner_tester");
    const auto latched = rclcpp::QoS(1).reliable().transient_local();
    map_pub = tester->create_publisher<MazeMap>("/maze/map", latched);
    ball_pub = tester->create_publisher<BallState>("/ball/state", rclcpp::QoS(1).best_effort());
    route_sub = tester->create_subscription<PlannedRoute>(
      "/maze/route", latched, [this](PlannedRoute::ConstSharedPtr m) {routes.push_back(*m);});
    path_sub = tester->create_subscription<nav_msgs::msg::Path>(
      "/maze/path", latched, [this](nav_msgs::msg::Path::ConstSharedPtr m) {paths.push_back(*m);});
    executor.add_node(planner);
    executor.add_node(tester);
  }

  void TearDown() override
  {
    executor.remove_node(planner);
    executor.remove_node(tester);
  }

  // spins until pred holds or the timeout passes; `each` runs every 50 ms (e.g. publish a ball state)
  bool wait_for(
    const std::function<bool()> & pred, std::chrono::milliseconds timeout,
    const std::function<void()> & each = [] {})
  {
    const auto end = std::chrono::steady_clock::now() + timeout;
    auto next = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() < end) {
      if (std::chrono::steady_clock::now() >= next) {
        each();
        next += 50ms;
      }
      executor.spin_some(10ms);
      if (pred()) {
        return true;
      }
      std::this_thread::sleep_for(5ms);
    }
    return pred();
  }

  // the first plan on L1 from the fixture's start
  void plan_once()
  {
    map_pub->publish(make_map(fixture));
    ASSERT_TRUE(wait_for([this] {return !routes.empty();}, 3s,
      [this] {ball_pub->publish(ball(fixture.start[0], fixture.start[1]));}));
    ASSERT_EQ(routes.back().plan_id, 1u);
  }

  // a free floor cell's center outside every corridor of the route by more than 4 mm
  std::array<double, 2> outside_point(const PlannedRoute & r) const
  {
    const Grid & g = fixture.grid;
    for (int row = 0; row < g.height; ++row) {
      for (int col = 0; col < g.width; ++col) {
        if (is_obstacle(g.data[static_cast<size_t>(row * g.width + col)])) {
          continue;
        }
        const auto p = cell_center(g, row, col);
        bool in = false;
        for (const auto & k : r.corridors) {
          in = in || (p[0] >= k.x_min - 0.004 && p[0] <= k.x_max + 0.004 && p[1] >= k.y_min - 0.004 &&
            p[1] <= k.y_max + 0.004);
        }
        if (!in) {
          return p;
        }
      }
    }
    ADD_FAILURE() << "no free cell outside the corridors";
    return {0.0, 0.0};
  }

  Fixture fixture;
  rclcpp::executors::SingleThreadedExecutor executor;
  std::shared_ptr<PlannerNode> planner;
  rclcpp::Node::SharedPtr tester;
  rclcpp::Publisher<MazeMap>::SharedPtr map_pub;
  rclcpp::Publisher<BallState>::SharedPtr ball_pub;
  rclcpp::Subscription<PlannedRoute>::SharedPtr route_sub;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub;
  std::vector<PlannedRoute> routes;
  std::vector<nav_msgs::msg::Path> paths;
};

}  // namespace

TEST_F(PlannerNodeTest, PlansOnceWithMapAndBall)
{
  plan_once();
  const PlannedRoute & r = routes.back();
  EXPECT_EQ(r.status, PlannedRoute::STATUS_OK);
  EXPECT_EQ(r.header.frame_id, "maze_board");
  EXPECT_EQ(r.path.header.frame_id, "maze_board");
  ASSERT_FALSE(r.path.poses.empty());
  EXPECT_EQ(r.arc_length.size(), r.path.poses.size());
  EXPECT_EQ(r.pose_corridor.size(), r.path.poses.size());
  EXPECT_FALSE(r.corridors.empty());
  for (const auto & p : r.path.poses) {
    EXPECT_EQ(p.pose.position.z, 0.0);
    EXPECT_EQ(p.pose.orientation.w, 1.0);
  }
  EXPECT_NEAR(r.path.poses.back().pose.position.x, fixture.goal.x, 1e-12);
  EXPECT_NEAR(r.path.poses.back().pose.position.y, fixture.goal.y, 1e-12);
  ASSERT_TRUE(wait_for([this] {return !paths.empty();}, 2s));
  ASSERT_EQ(paths.back().poses.size(), r.path.poses.size());
  for (size_t i = 0; i < r.path.poses.size(); ++i) {
    EXPECT_EQ(paths.back().poses[i].pose.position.x, r.path.poses[i].pose.position.x);
    EXPECT_EQ(paths.back().poses[i].pose.position.y, r.path.poses[i].pose.position.y);
  }
}

TEST_F(PlannerNodeTest, ReplansWhenOutsideEveryCorridor)
{
  plan_once();
  const auto p = outside_point(routes.back());
  EXPECT_TRUE(wait_for([this] {return routes.back().plan_id == 2u;}, 3s,
    [&] {ball_pub->publish(ball(p[0], p[1]));}));
}

TEST_F(PlannerNodeTest, TrackingLostDoesNotReplan)
{
  plan_once();
  const auto p = outside_point(routes.back());
  EXPECT_FALSE(wait_for([this] {return routes.back().plan_id != 1u;}, 1s,
    [&] {ball_pub->publish(ball(p[0], p[1], false));}));
}

TEST_F(PlannerNodeTest, NewMapReplans)
{
  plan_once();
  map_pub->publish(make_map(fixture));
  EXPECT_TRUE(wait_for([this] {return routes.back().plan_id == 2u;}, 3s));
}

TEST_F(PlannerNodeTest, BallBeforeMapPlansWhenMapArrives)
{
  EXPECT_FALSE(wait_for([this] {return !routes.empty();}, 500ms,
    [this] {ball_pub->publish(ball(fixture.start[0], fixture.start[1]));}));
  map_pub->publish(make_map(fixture));
  ASSERT_TRUE(wait_for([this] {return !routes.empty();}, 3s));
  EXPECT_EQ(routes.back().plan_id, 1u);
  EXPECT_EQ(routes.back().status, PlannedRoute::STATUS_OK);
}

TEST_F(PlannerNodeTest, BallOutsideGridGivesNoPath)
{
  map_pub->publish(make_map(fixture));
  ASSERT_TRUE(wait_for([this] {return !routes.empty();}, 3s, [this] {ball_pub->publish(ball(1.0, 1.0));}));
  const PlannedRoute & r = routes.back();
  EXPECT_EQ(r.status, PlannedRoute::STATUS_NO_PATH);
  EXPECT_EQ(r.plan_id, 1u);
  EXPECT_TRUE(r.path.poses.empty());
  EXPECT_TRUE(r.corridors.empty());
  EXPECT_TRUE(r.arc_length.empty());
  EXPECT_TRUE(r.pose_corridor.empty());
  // retries every 0.5 s are still no path: nothing new is published
  EXPECT_FALSE(wait_for([this] {return routes.size() > 1;}, 1200ms, [this] {ball_pub->publish(ball(1.0, 1.0));}));
}

TEST_F(PlannerNodeTest, NonFiniteBallIgnored)
{
  map_pub->publish(make_map(fixture));
  EXPECT_FALSE(wait_for([this] {return !routes.empty();}, 500ms,
    [this] {ball_pub->publish(ball(std::numeric_limits<double>::quiet_NaN(), fixture.start[1]));}));
}

TEST_F(PlannerNodeTest, BadMapIgnored)
{
  plan_once();
  MazeMap bad = make_map(fixture);
  bad.walls.data.pop_back();
  map_pub->publish(bad);
  wait_for([] {return false;}, 300ms);
  const auto p = outside_point(routes.back());
  ASSERT_TRUE(wait_for([this] {return routes.back().plan_id == 2u;}, 3s,
    [&] {ball_pub->publish(ball(p[0], p[1]));}));
  const PlannedRoute & r = routes.back();
  ASSERT_EQ(r.status, PlannedRoute::STATUS_OK);
  EXPECT_NEAR(r.path.poses.back().pose.position.x, fixture.goal.x, 1e-12);
  EXPECT_NEAR(r.path.poses.back().pose.position.y, fixture.goal.y, 1e-12);
}

TEST_F(PlannerNodeTest, RequiredParamMissingThrows)
{
  EXPECT_ANY_THROW(PlannerNode(rclcpp::NodeOptions().parameter_overrides(yaml_params("replan.outside_frames"))));
  EXPECT_ANY_THROW(PlannerNode(rclcpp::NodeOptions().parameter_overrides(yaml_params("ball_radius"))));
}

TEST_F(PlannerNodeTest, ConvertGridRoundTrip)
{
  const Grid g = to_grid(to_occupancy(fixture.grid));
  EXPECT_EQ(g.width, fixture.grid.width);
  EXPECT_EQ(g.height, fixture.grid.height);
  EXPECT_NEAR(g.resolution, fixture.grid.resolution, 1e-9);  // OccupancyGrid stores it as float32
  EXPECT_EQ(g.origin_x, fixture.grid.origin_x);
  EXPECT_EQ(g.origin_y, fixture.grid.origin_y);
  EXPECT_EQ(g.data, fixture.grid.data);
}
