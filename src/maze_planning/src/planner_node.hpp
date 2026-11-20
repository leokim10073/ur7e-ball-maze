// The planner node (Contract 3): plans from /maze/map and /ball/state, publishes /maze/route and
// /maze/path, and replans when the ball leaves every corridor, a new map arrives, or (after
// STATUS_NO_PATH) every replan.no_path_retry_period until a path exists.
#pragma once

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <maze_interfaces/msg/ball_state.hpp>
#include <maze_interfaces/msg/maze_map.hpp>
#include <maze_interfaces/msg/planned_route.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>

#include "maze_planning/grid.hpp"
#include "maze_planning/replan_monitor.hpp"

namespace maze_planning
{

class PlannerNode : public rclcpp::Node
{
public:
  // every parameter of config/planner.yaml is required: a missing one throws std::runtime_error
  // naming it, as do invalid values (std::invalid_argument)
  explicit PlannerNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  struct Map
  {
    Grid grid;
    std::vector<Circle> holes;
    Circle goal;
  };

  template<typename T>
  T required(const std::string & name);
  void on_map(const maze_interfaces::msg::MazeMap & msg);
  void on_ball(const maze_interfaces::msg::BallState & msg);
  void plan(bool retry);

  PlannerParams params_;
  std::unique_ptr<ReplanMonitor> monitor_;
  std::optional<Map> map_;
  std::optional<std::array<double, 2>> ball_;
  uint32_t plan_id_ = 0;
  bool last_no_path_ = false;

  rclcpp::Subscription<maze_interfaces::msg::MazeMap>::SharedPtr map_sub_;
  rclcpp::Subscription<maze_interfaces::msg::BallState>::SharedPtr ball_sub_;
  rclcpp::Publisher<maze_interfaces::msg::PlannedRoute>::SharedPtr route_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
};

}  // namespace maze_planning
