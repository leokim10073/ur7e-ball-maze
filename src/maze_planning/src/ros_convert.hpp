// Contract 1 and 3 messages <-> the ROS-free planner types.
#pragma once

#include <cstdint>

#include <maze_interfaces/msg/circle.hpp>
#include <maze_interfaces/msg/planned_route.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <rclcpp/time.hpp>

#include "maze_planning/grid.hpp"
#include "maze_planning/route.hpp"

namespace maze_planning
{

// origin orientation is ignored: Contract 1's grid is axis-aligned with maze_board
Grid to_grid(const nav_msgs::msg::OccupancyGrid & msg);

Circle to_circle(const maze_interfaces::msg::Circle & msg);

// an empty route (no points) gives STATUS_NO_PATH with all arrays empty; poses at z = 0, orientation
// w = 1; the route and its path in frame maze_board, both stamped `stamp`
maze_interfaces::msg::PlannedRoute to_msg(const Route & route, uint32_t plan_id, const rclcpp::Time & stamp);

}  // namespace maze_planning
