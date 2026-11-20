#include "ros_convert.hpp"

namespace maze_planning
{

Grid to_grid(const nav_msgs::msg::OccupancyGrid & msg)
{
  Grid g;
  g.width = static_cast<int>(msg.info.width);
  g.height = static_cast<int>(msg.info.height);
  g.resolution = msg.info.resolution;
  g.origin_x = msg.info.origin.position.x;
  g.origin_y = msg.info.origin.position.y;
  g.data = msg.data;
  return g;
}

Circle to_circle(const maze_interfaces::msg::Circle & msg)
{
  return {msg.x, msg.y, msg.radius};
}

maze_interfaces::msg::PlannedRoute to_msg(const Route & route, uint32_t plan_id, const rclcpp::Time & stamp)
{
  maze_interfaces::msg::PlannedRoute msg;
  msg.header.stamp = stamp;
  msg.header.frame_id = "maze_board";
  msg.plan_id = plan_id;
  msg.path.header = msg.header;
  if (route.points.empty()) {
    msg.status = maze_interfaces::msg::PlannedRoute::STATUS_NO_PATH;
    return msg;
  }
  msg.status = maze_interfaces::msg::PlannedRoute::STATUS_OK;
  for (const auto & p : route.points) {
    geometry_msgs::msg::PoseStamped pose;
    pose.header = msg.header;
    pose.pose.position.x = p[0];
    pose.pose.position.y = p[1];
    pose.pose.position.z = 0.0;
    pose.pose.orientation.w = 1.0;
    msg.path.poses.push_back(pose);
  }
  msg.arc_length = route.arc_length;
  for (const auto & k : route.corridors) {
    maze_interfaces::msg::Corridor c;
    c.x_min = k.x_min;
    c.x_max = k.x_max;
    c.y_min = k.y_min;
    c.y_max = k.y_max;
    msg.corridors.push_back(c);
  }
  msg.pose_corridor = route.pose_corridor;
  return msg;
}

}  // namespace maze_planning
