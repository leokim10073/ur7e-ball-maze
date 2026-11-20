#include <exception>
#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "planner_node.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  int code = 0;
  try {
    rclcpp::spin(std::make_shared<maze_planning::PlannerNode>());
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("planner"), "%s", e.what());
    code = 1;
  }
  rclcpp::shutdown();
  return code;
}
