// tilt_control: the node, with its own SIGINT handling so the last thing it sends is zero velocities
// (the velocity controller keeps the last command it got).
#include <atomic>
#include <csignal>

#include "tilt_control_node.hpp"

namespace
{
std::atomic<bool> g_stop{false};
void on_signal(int) {g_stop = true;}
}  // namespace

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv, rclcpp::InitOptions(), rclcpp::SignalHandlerOptions::None);
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  std::shared_ptr<maze_arm_control::TiltControlNode> node;
  try {
    node = std::make_shared<maze_arm_control::TiltControlNode>();
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("tilt_control"), "%s", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node);
  while (rclcpp::ok() && !g_stop && !node->failed()) {
    exec.spin_once(std::chrono::milliseconds(50));
  }
  const bool failed = node->failed();
  if (rclcpp::ok()) {
    for (int i = 0; i < 3; ++i) {node->stop();}
    rclcpp::shutdown();
  }
  return failed ? 1 : 0;
}
