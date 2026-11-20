#include "planner_node.hpp"

#include <cmath>
#include <stdexcept>

#include "maze_planning/grid_planner.hpp"
#include "maze_planning/route.hpp"
#include "ros_convert.hpp"

namespace maze_planning
{

template<typename T>
T PlannerNode::required(const std::string & name)
{
  try {
    declare_parameter<T>(name);  // no default: the value must come from the params file
    return get_parameter(name).get_value<T>();
  } catch (const rclcpp::exceptions::ParameterUninitializedException &) {
  } catch (const rclcpp::exceptions::UninitializedStaticallyTypedParameterException &) {
  }
  throw std::runtime_error("missing required parameter '" + name + "' (see config/planner.yaml)");
}

PlannerNode::PlannerNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("planner", options)
{
  params_.ball_radius = required<double>("ball_radius");
  params_.wall_margin = required<double>("wall_margin");
  params_.hole_margin = required<double>("hole_margin");
  params_.preferred_clearance = required<double>("preferred_clearance");
  params_.clearance_weight = required<double>("clearance_weight");
  params_.point_spacing = required<double>("point_spacing");
  validate(params_);
  ReplanParams replan;
  replan.outside_margin = required<double>("replan.outside_margin");
  replan.outside_frames = static_cast<int>(required<int64_t>("replan.outside_frames"));
  replan.no_path_retry_period = required<double>("replan.no_path_retry_period");
  monitor_ = std::make_unique<ReplanMonitor>(replan);

  const auto latched = rclcpp::QoS(1).reliable().transient_local();
  route_pub_ = create_publisher<maze_interfaces::msg::PlannedRoute>("/maze/route", latched);
  path_pub_ = create_publisher<nav_msgs::msg::Path>("/maze/path", latched);
  map_sub_ = create_subscription<maze_interfaces::msg::MazeMap>(
    "/maze/map", latched, [this](const maze_interfaces::msg::MazeMap & m) {on_map(m);});
  ball_sub_ = create_subscription<maze_interfaces::msg::BallState>(
    "/ball/state", rclcpp::QoS(1).best_effort(),
    [this](const maze_interfaces::msg::BallState & m) {on_ball(m);});
}

void PlannerNode::on_map(const maze_interfaces::msg::MazeMap & msg)
{
  Map map;
  map.grid = to_grid(msg.walls);
  for (const auto & h : msg.holes) {
    map.holes.push_back(to_circle(h));
  }
  map.goal = to_circle(msg.goal);
  try {
    validate(map.grid);
    for (const auto & c : map.holes) {
      if (!std::isfinite(c.x) || !std::isfinite(c.y) || !(std::isfinite(c.radius) && c.radius >= 0.0)) {
        throw std::invalid_argument("a hole is not finite or has a negative radius");
      }
    }
    if (!std::isfinite(map.goal.x) || !std::isfinite(map.goal.y)) {
      throw std::invalid_argument("the goal is not finite");
    }
  } catch (const std::invalid_argument & e) {
    RCLCPP_ERROR(get_logger(), "ignoring map from '%s': %s", msg.source.c_str(), e.what());
    return;  // the previous map stays
  }
  map_ = map;
  RCLCPP_INFO(get_logger(), "map from '%s': %d x %d cells", msg.source.c_str(), map.grid.width, map.grid.height);
  if (ball_) {
    plan(false);  // a new map always replans
  }
}

void PlannerNode::on_ball(const maze_interfaces::msg::BallState & msg)
{
  if (!msg.tracking_ok || !std::isfinite(msg.x) || !std::isfinite(msg.y)) {
    return;  // only a prediction (or garbage): neither plans nor counts as outside
  }
  ball_ = std::array<double, 2>{msg.x, msg.y};
  if (!map_) {
    return;
  }
  if (plan_id_ == 0 && !last_no_path_) {
    plan(false);  // the first plan
  } else if (monitor_->should_replan(msg.x, msg.y, now().seconds())) {
    plan(last_no_path_);
  }
}

void PlannerNode::plan(bool retry)
{
  const auto path = plan_grid_path(map_->grid, map_->holes, *ball_, map_->goal, params_);
  const Route route = build_route(map_->grid, map_->holes, path, params_);
  const rclcpp::Time stamp = now();
  monitor_->route_planned(route, stamp.seconds());
  const bool no_path = route.points.empty();
  if (retry && no_path) {
    return;  // still no path: controllers already have STATUS_NO_PATH, plan_id stays
  }
  last_no_path_ = no_path;
  const auto msg = to_msg(route, ++plan_id_, stamp);
  route_pub_->publish(msg);
  path_pub_->publish(msg.path);
  if (no_path) {
    RCLCPP_WARN(get_logger(), "plan %u: no path from (%.3f, %.3f)", plan_id_, (*ball_)[0], (*ball_)[1]);
  } else {
    RCLCPP_INFO(get_logger(), "plan %u: %zu poses, %zu corridors (%u lead-in), %.3f m", plan_id_,
      route.points.size(), route.corridors.size(), route.lead_in_corridors, route.arc_length.back());
  }
}

}  // namespace maze_planning
