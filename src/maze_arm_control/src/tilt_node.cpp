// Arm control for the maze (Contract 4, #14): /board/tilt_cmd -> UR7e joint velocities.
//
//   subscribes  /board/tilt_cmd (BoardTilt), /joint_states (JointState)
//   publishes   /forward_velocity_controller/commands (Float64MultiArray), /board/tilt_state (BoardTilt)
//
// At start it waits for /joint_states, takes the board pose there (made level) as home, and
// switches to the velocity controller. Each cycle it holds the board center at home and turns
// the board toward the commanded tilt (TiltController). No command for command_timeout, or stale
// joint states: zero joint velocities and a warning. Ctrl-C sends zero velocities before exiting,
// since the velocity controller keeps the last command forever.
// C++ because it runs FK and a 6x6 solve every control cycle.

#include <atomic>
#include <chrono>
#include <csignal>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <controller_manager_msgs/srv/switch_controller.hpp>
#include <maze_interfaces/msg/board_tilt.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include "maze_arm_control/kinematics.hpp"
#include "maze_arm_control/tilt_controller.hpp"

using maze_arm_control::Vector6d;
using maze_interfaces::msg::BoardTilt;
using sensor_msgs::msg::JointState;
using std_msgs::msg::Float64MultiArray;
using SwitchController = controller_manager_msgs::srv::SwitchController;

namespace
{
std::atomic<bool> g_stop{ false };

Eigen::Vector3d vec3_param(rclcpp::Node& node, const std::string& name)
{
  const auto v = node.declare_parameter<std::vector<double>>(name, std::vector<double>{});
  if (v.size() != 3)
  {
    throw std::runtime_error("parameter '" + name + "' must be 3 numbers (see launch/tilt_node.launch.py)");
  }
  return { v[0], v[1], v[2] };
}
}  // namespace

class TiltNode : public rclcpp::Node
{
public:
  TiltNode() : Node("tilt_node")
  {
    const std::string share = ament_index_cpp::get_package_share_directory("maze_arm_control");
    cfg_ = maze_arm_control::load_arm_config(declare_parameter("config_file", share + "/config/arm.yaml"));
    kin_ = maze_arm_control::load_ur_kinematics(declare_parameter(
        "kinematics_file",
        ament_index_cpp::get_package_share_directory("ur_description") + "/config/ur7e/default_kinematics.yaml"));
    const Eigen::Vector3d grasp_xyz = vec3_param(*this, "grasp_xyz");
    const Eigen::Vector3d grasp_rpy = vec3_param(*this, "grasp_rpy");
    const Eigen::Vector3d handle_center = vec3_param(*this, "handle_center");
    T_tool0_board_ = maze_arm_control::board_offset(grasp_xyz, grasp_rpy, handle_center);
    const double rate = declare_parameter("rate", 100.0);
    dt_ = 1.0 / rate;
    switch_ = declare_parameter("switch_controllers", true);
    velocity_controller_ = declare_parameter("velocity_controller", std::string("forward_velocity_controller"));
    stop_controllers_ = declare_parameter("stop_controllers", std::vector<std::string>{ "scaled_joint_trajectory_controller" });

    cmd_pub_ = create_publisher<Float64MultiArray>("/" + velocity_controller_ + "/commands", 10);
    state_pub_ = create_publisher<BoardTilt>("/board/tilt_state", rclcpp::QoS(1).best_effort());
    joint_sub_ = create_subscription<JointState>("/joint_states", rclcpp::SensorDataQoS(),
                                                 [this](JointState::ConstSharedPtr msg) { on_joint_state(*msg); });
    cmd_sub_ = create_subscription<BoardTilt>("/board/tilt_cmd", rclcpp::QoS(1).reliable(),
                                              [this](BoardTilt::ConstSharedPtr msg) {
                                                cmd_ = Eigen::Vector2d(msg->tilt_x, msg->tilt_y);
                                                cmd_time_ = now();
                                              });
    switch_client_ = create_client<SwitchController>("/controller_manager/switch_controller");
    timer_ = rclcpp::create_timer(this, get_clock(), rclcpp::Duration::from_seconds(dt_), [this] { tick(); });
    RCLCPP_INFO(get_logger(), "tilt_node: max tilt %.3f rad, %.2f rad/s, %.0f Hz; waiting for /joint_states",
                cfg_.max_tilt, cfg_.max_tilt_rate, rate);
  }

  void stop_arm()
  {
    cmd_pub_->publish(Float64MultiArray().set__data(std::vector<double>(6, 0.0)));
  }

private:
  enum class State { WAIT_JOINTS, SWITCHING, RUNNING, FAILED };

  void on_joint_state(const JointState& msg)
  {
    try
    {
      q_ = maze_arm_control::joints_from_state(msg.name, msg.position);
      q_stamp_ = msg.header.stamp;
      q_time_ = now();
    }
    catch (const std::runtime_error& e)
    {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "%s", e.what());
    }
  }

  void start()
  {
    ctrl_.emplace(cfg_, kin_, T_tool0_board_,
                  maze_arm_control::level_home(maze_arm_control::fk_tool0(kin_, *q_) * T_tool0_board_));
    const Eigen::Vector3d c = ctrl_->home().translation();
    RCLCPP_INFO(get_logger(), "home: board center (%.4f, %.4f, %.4f) m in base_link", c.x(), c.y(), c.z());
    if (!switch_)
    {
      state_ = State::RUNNING;
      return;
    }
    if (!switch_client_->service_is_ready())
    {
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000, "waiting for /controller_manager/switch_controller");
      return;
    }
    auto req = std::make_shared<SwitchController::Request>();
    req->activate_controllers = { velocity_controller_ };
    req->deactivate_controllers = stop_controllers_;
    req->strictness = SwitchController::Request::BEST_EFFORT;  // already active / inactive is fine
    req->activate_asap = true;
    state_ = State::SWITCHING;
    switch_client_->async_send_request(req, [this](rclcpp::Client<SwitchController>::SharedFuture f) {
      if (f.get()->ok)
      {
        RCLCPP_INFO(get_logger(), "switched to %s", velocity_controller_.c_str());
        state_ = State::RUNNING;
      }
      else
      {
        RCLCPP_ERROR(get_logger(), "could not switch to %s; not moving the arm", velocity_controller_.c_str());
        state_ = State::FAILED;
      }
    });
  }

  void tick()
  {
    if (!q_)
    {
      return;
    }
    if (state_ == State::WAIT_JOINTS)
    {
      start();
    }
    if (state_ != State::RUNNING)
    {
      return;
    }
    const rclcpp::Time t = now();

    BoardTilt state;
    state.header.stamp = q_stamp_;
    state.header.frame_id = "maze_board";
    const Eigen::Vector2d tilt = ctrl_->measured_tilt(*q_);
    state.tilt_x = tilt[0];
    state.tilt_y = tilt[1];
    state_pub_->publish(state);

    if ((t - q_time_).seconds() > cfg_.command_timeout)
    {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "no /joint_states for %.0f ms: stopping the arm",
                           cfg_.command_timeout * 1000);
      halt();
      return;
    }
    if (!cmd_time_ || (t - *cmd_time_).seconds() > cfg_.command_timeout)
    {
      if (cmd_time_)
      {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "no /board/tilt_cmd for %.0f ms: stopping the arm",
                             cfg_.command_timeout * 1000);
      }
      halt();
      return;
    }
    if (!tracking_)
    {
      ctrl_->reset(*q_);  // pick up from where the board is now
      tracking_ = true;
    }
    const Vector6d qdot = ctrl_->step(*q_, cmd_, dt_);
    if (!qdot.allFinite())
    {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 1000, "non-finite joint velocities: stopping the arm");
      halt();
      return;
    }
    cmd_pub_->publish(Float64MultiArray().set__data(std::vector<double>(qdot.data(), qdot.data() + 6)));
  }

  void halt()
  {
    stop_arm();
    tracking_ = false;
  }

  maze_arm_control::ArmConfig cfg_{};
  maze_arm_control::UrKinematics kin_;
  Eigen::Isometry3d T_tool0_board_;
  double dt_;
  bool switch_;
  std::string velocity_controller_;
  std::vector<std::string> stop_controllers_;

  State state_ = State::WAIT_JOINTS;
  std::optional<maze_arm_control::TiltController> ctrl_;
  std::optional<Vector6d> q_;
  builtin_interfaces::msg::Time q_stamp_;
  rclcpp::Time q_time_;
  Eigen::Vector2d cmd_ = Eigen::Vector2d::Zero();
  std::optional<rclcpp::Time> cmd_time_;
  bool tracking_ = false;

  rclcpp::Publisher<Float64MultiArray>::SharedPtr cmd_pub_;
  rclcpp::Publisher<BoardTilt>::SharedPtr state_pub_;
  rclcpp::Subscription<JointState>::SharedPtr joint_sub_;
  rclcpp::Subscription<BoardTilt>::SharedPtr cmd_sub_;
  rclcpp::Client<SwitchController>::SharedPtr switch_client_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv)
{
  // Handle Ctrl-C here, not in rclcpp, so the node can still publish zero velocities.
  rclcpp::init(argc, argv, rclcpp::InitOptions(), rclcpp::SignalHandlerOptions::SigTerm);
  std::signal(SIGINT, [](int) { g_stop = true; });
  int code = 0;
  try
  {
    auto node = std::make_shared<TiltNode>();
    rclcpp::executors::SingleThreadedExecutor exec;
    exec.add_node(node);
    while (rclcpp::ok() && !g_stop)
    {
      exec.spin_once(std::chrono::milliseconds(50));
    }
    if (rclcpp::ok())
    {
      node->stop_arm();
      std::this_thread::sleep_for(std::chrono::milliseconds(100));  // let the zero command go out
    }
  }
  catch (const std::exception& e)
  {
    RCLCPP_FATAL(rclcpp::get_logger("tilt_node"), "%s", e.what());
    code = 1;
  }
  rclcpp::shutdown();
  return code;
}
