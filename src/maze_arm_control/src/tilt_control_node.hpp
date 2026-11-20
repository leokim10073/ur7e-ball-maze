// tilt_control: /board/tilt_cmd -> UR7e joint velocities, the board tilting about its own center;
// /board/tilt_state from FK of /joint_states (Contract 4).
//
// Start-up (on a steady timer): fresh joint states -> the board within home_tolerance of the home pose
// (else refuse: error, shut down) -> controllers switched with STRICT, the result checked -> running,
// the reference starting from the measured tilt.
// Running: one control step per joint state. The watchdog runs on the steady timer: no tilt_cmd for
// command_timeout or joint states older than state_timeout -> zero velocities (the velocity
// controller has no timeout of its own). A joint jump resyncs; /sim/reset_done re-runs start-up;
// the velocity controller found inactive (a reset, the robot's controller_stopper) -> halted, zeros.
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "controller_manager_msgs/srv/list_controllers.hpp"
#include "controller_manager_msgs/srv/switch_controller.hpp"
#include "maze_arm_control/kinematics.hpp"
#include "maze_arm_control/reference.hpp"
#include "maze_arm_control/tilt_controller.hpp"
#include "maze_arm_control/watchdog.hpp"
#include "maze_interfaces/msg/board_tilt.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/empty.hpp"
#include "std_msgs/msg/float64.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

namespace maze_arm_control
{

extern const std::vector<std::string> kUrJoints;

// Joint positions in kUrJoints order from a JointState (any order, extra joints ignored);
// nullopt if one is missing or the message is malformed.
std::optional<Joints> joints_from_state(const sensor_msgs::msg::JointState & msg);

class TiltControlNode : public rclcpp::Node
{
public:
  enum class Phase {kWaitState, kCheckHome, kSwitch, kRunning, kHalted, kFailed};

  explicit TiltControlNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

  Phase phase() const {return phase_;}
  bool failed() const {return phase_ == Phase::kFailed;}
  // Zero velocities, e.g. before the process exits.
  void stop();

private:
  template<typename T>
  T required(const std::string & name);

  void on_joint_state(const sensor_msgs::msg::JointState & msg);
  void on_tilt_cmd(const maze_interfaces::msg::BoardTilt & msg);
  void on_reset_done();
  void on_timer();

  void check_home();
  void request_switch();
  void on_list(const controller_manager_msgs::srv::ListControllers::Response & res);
  void start_running();
  void poll_controller();
  void resync(const char * why);
  void fail(const std::string & why);
  void publish_velocity(const Joints & qdot);
  double now_steady() const;
  Eigen::Vector2d measured_tilt() const;
  bool speed_scaling_ok() const;

  // parameters
  std::string velocity_controller_;
  std::vector<std::string> deactivate_controllers_;
  double home_pos_tol_, home_rot_tol_, jump_threshold_, yaw_warn_, period_;
  double state_timeout_;
  bool use_speed_scaling_;
  double speed_scaling_full_;

  std::unique_ptr<ArmKinematics> kin_;
  std::unique_ptr<ReferenceShaper> ref_;
  std::unique_ptr<TiltController> ctrl_;
  std::unique_ptr<Watchdog> watchdog_;
  Eigen::Isometry3d home_;

  Phase phase_ = Phase::kWaitState;
  bool request_pending_ = false;
  bool commanding_ = false;  // last cycle sent a control command (not zeros from the watchdog)
  std::optional<Joints> q_;
  std::optional<rclcpp::Time> q_stamp_;
  std::optional<Eigen::Vector2d> last_cmd_;
  std::optional<double> speed_scaling_;
  std::optional<double> state_received_;  // steady time of the last joint state
  double last_poll_ = 0.0;
  bool warned_ = false;  // the watchdog warning for the current stop was logged

  rclcpp::Clock steady_{RCL_STEADY_TIME};
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr vel_pub_;
  rclcpp::Publisher<maze_interfaces::msg::BoardTilt>::SharedPtr state_pub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
  rclcpp::Subscription<maze_interfaces::msg::BoardTilt>::SharedPtr cmd_sub_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr reset_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr scaling_sub_;
  rclcpp::Client<controller_manager_msgs::srv::ListControllers>::SharedPtr list_cli_;
  rclcpp::Client<controller_manager_msgs::srv::SwitchController>::SharedPtr switch_cli_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace maze_arm_control
