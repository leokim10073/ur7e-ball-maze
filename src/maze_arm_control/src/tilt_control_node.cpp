#include "tilt_control_node.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "maze_arm_control/so3.hpp"

namespace maze_arm_control
{

using controller_manager_msgs::srv::ListControllers;
using controller_manager_msgs::srv::SwitchController;

const std::vector<std::string> kUrJoints = {
  "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"};

namespace
{
constexpr double kPollPeriod = 1.0;   // s, how often the velocity controller is checked while running
constexpr double kMaxStepDt = 0.02;   // s, a longer gap between joint states is not integrated over
constexpr double kDeg = M_PI / 180.0;

template<int N>
Eigen::Matrix<double, N, 1> to_vec(const std::vector<double> & v, const std::string & name)
{
  if (v.size() != N) {
    throw std::runtime_error(
            "parameter '" + name + "' needs " + std::to_string(N) + " values, got " + std::to_string(v.size()));
  }
  return Eigen::Map<const Eigen::Matrix<double, N, 1>>(v.data());
}
}  // namespace

std::optional<Joints> joints_from_state(const sensor_msgs::msg::JointState & msg)
{
  if (msg.name.size() != msg.position.size()) {
    return std::nullopt;
  }
  Joints q;
  for (size_t j = 0; j < kUrJoints.size(); ++j) {
    const auto it = std::find(msg.name.begin(), msg.name.end(), kUrJoints[j]);
    if (it == msg.name.end()) {
      return std::nullopt;
    }
    q[j] = msg.position[it - msg.name.begin()];
  }
  if (!q.allFinite()) {
    return std::nullopt;
  }
  return q;
}

template<typename T>
T TiltControlNode::required(const std::string & name)
{
  try {
    declare_parameter<T>(name);  // no default: the value must come from the params file
    return get_parameter(name).get_value<T>();
  } catch (const rclcpp::exceptions::ParameterUninitializedException &) {
  } catch (const rclcpp::exceptions::UninitializedStaticallyTypedParameterException &) {
  }
  throw std::runtime_error("missing required parameter '" + name + "' (see config/tilt_control.yaml)");
}

TiltControlNode::TiltControlNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("tilt_control", options)
{
  using DV = std::vector<double>;
  const auto topic_cmd = required<std::string>("topics.tilt_cmd");
  const auto topic_state = required<std::string>("topics.tilt_state");
  const auto topic_joints = required<std::string>("topics.joint_states");
  const auto topic_vel = required<std::string>("topics.velocity_command");
  const auto topic_reset = required<std::string>("topics.reset_done");
  const auto topic_scaling = required<std::string>("topics.speed_scaling");
  speed_scaling_full_ = required<double>("speed_scaling_full");
  velocity_controller_ = required<std::string>("controllers.velocity");
  deactivate_controllers_ = required<std::vector<std::string>>("controllers.deactivate");

  DhParams dh;
  const auto d = to_vec<6>(required<DV>("kinematics.d"), "kinematics.d");
  const auto a = to_vec<6>(required<DV>("kinematics.a"), "kinematics.a");
  const auto alpha = to_vec<6>(required<DV>("kinematics.alpha"), "kinematics.alpha");
  for (int i = 0; i < 6; ++i) {
    dh.d[i] = d[i];
    dh.a[i] = a[i];
    dh.alpha[i] = alpha[i];
  }
  BoardMount mount{
    to_vec<3>(required<DV>("mount.grasp_xyz"), "mount.grasp_xyz"),
    to_vec<3>(required<DV>("mount.grasp_rpy"), "mount.grasp_rpy"),
    to_vec<3>(required<DV>("mount.handle_center"), "mount.handle_center")};
  kin_ = std::make_unique<ArmKinematics>(dh, mount);
  home_ = kin_->fk_board(to_vec<6>(required<DV>("home_joints"), "home_joints"));
  home_pos_tol_ = required<double>("home_tolerance.position");
  home_rot_tol_ = required<double>("home_tolerance.rotation");

  ReferenceParams rp{required<double>("limits.max_tilt"), required<double>("limits.max_rate"),
    required<double>("reference.smoothing_tau")};
  ControllerParams cp;
  cp.k_pos = required<double>("gains.k_pos");
  cp.k_rot = required<double>("gains.k_rot");
  cp.sigma_min = required<double>("dls.sigma_min");
  cp.lambda_max = required<double>("dls.lambda_max");
  cp.max_linear_speed = required<double>("twist_cap.linear");
  cp.max_angular_speed = required<double>("twist_cap.angular");
  cp.max_joint_speed = to_vec<6>(required<DV>("joint_limits.max_speed"), "joint_limits.max_speed");
  cp.max_joint_accel = to_vec<6>(required<DV>("joint_limits.max_accel"), "joint_limits.max_accel");
  const double cmd_timeout = required<double>("watchdog.command_timeout");
  state_timeout_ = required<double>("watchdog.state_timeout");
  period_ = required<double>("watchdog.period");
  jump_threshold_ = required<double>("joint_jump_threshold");
  yaw_warn_ = required<double>("yaw_warn");

  const bool positive = rp.max_tilt > 0 && rp.max_rate > 0 && rp.smoothing_tau >= 0 && cp.k_pos > 0 &&
    cp.k_rot > 0 && cp.sigma_min > 0 && cp.lambda_max >= 0 && cp.max_linear_speed > 0 &&
    cp.max_angular_speed > 0 && (cp.max_joint_speed.array() > 0).all() &&
    (cp.max_joint_accel.array() > 0).all() && cmd_timeout > 0 && state_timeout_ > 0 && period_ > 0 &&
    jump_threshold_ > 0 && home_pos_tol_ > 0 && home_rot_tol_ > 0 && speed_scaling_full_ > 0;
  if (!positive) {
    throw std::runtime_error("tilt_control: a limit, gain, tolerance or period is not positive");
  }
  ref_ = std::make_unique<ReferenceShaper>(rp);
  ctrl_ = std::make_unique<TiltController>(*kin_, cp, home_);
  watchdog_ = std::make_unique<Watchdog>(cmd_timeout, state_timeout_);

  vel_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(topic_vel, 10);
  state_pub_ = create_publisher<maze_interfaces::msg::BoardTilt>(topic_state, rclcpp::QoS(1).best_effort());
  joint_sub_ = create_subscription<sensor_msgs::msg::JointState>(
    topic_joints, rclcpp::SensorDataQoS(),
    [this](const sensor_msgs::msg::JointState & m) {on_joint_state(m);});
  cmd_sub_ = create_subscription<maze_interfaces::msg::BoardTilt>(
    topic_cmd, rclcpp::QoS(1).reliable(), [this](const maze_interfaces::msg::BoardTilt & m) {on_tilt_cmd(m);});
  if (!topic_reset.empty()) {
    reset_sub_ = create_subscription<std_msgs::msg::Empty>(
      topic_reset, rclcpp::QoS(1).reliable().transient_local(),
      [this](const std_msgs::msg::Empty &) {on_reset_done();});
  }
  use_speed_scaling_ = !topic_scaling.empty();
  if (use_speed_scaling_) {
    scaling_sub_ = create_subscription<std_msgs::msg::Float64>(
      topic_scaling, rclcpp::QoS(1), [this](const std_msgs::msg::Float64 & m) {speed_scaling_ = m.data;});
  }
  list_cli_ = create_client<ListControllers>("/controller_manager/list_controllers");
  switch_cli_ = create_client<SwitchController>("/controller_manager/switch_controller");
  timer_ = create_wall_timer(std::chrono::duration<double>(period_), [this] {on_timer();});
  RCLCPP_INFO(get_logger(), "home board center (%.4f, %.4f, %.4f) m; waiting for joint states",
    home_.translation().x(), home_.translation().y(), home_.translation().z());
}

double TiltControlNode::now_steady() const
{
  return const_cast<rclcpp::Clock &>(steady_).now().seconds();
}

Eigen::Vector2d TiltControlNode::measured_tilt() const
{
  return tilt_of(home_.linear(), kin_->fk_board(*q_).linear());
}

bool TiltControlNode::speed_scaling_ok() const
{
  return !use_speed_scaling_ || (speed_scaling_ && *speed_scaling_ >= 0.999 * speed_scaling_full_);
}

void TiltControlNode::publish_velocity(const Joints & qdot)
{
  std_msgs::msg::Float64MultiArray msg;
  msg.data.assign(qdot.data(), qdot.data() + 6);
  vel_pub_->publish(msg);
}

void TiltControlNode::stop()
{
  if (vel_pub_) {
    publish_velocity(Joints::Zero());
  }
}

void TiltControlNode::on_tilt_cmd(const maze_interfaces::msg::BoardTilt & msg)
{
  const Eigen::Vector2d cmd(msg.tilt_x, msg.tilt_y);
  if (!cmd.allFinite()) {
    RCLCPP_WARN_THROTTLE(get_logger(), steady_, 1000, "ignoring a non-finite tilt_cmd");
    return;
  }
  last_cmd_ = cmd;
  watchdog_->on_command(now_steady());
  if (phase_ == Phase::kRunning) {
    ref_->set_command(cmd);
  }
}

void TiltControlNode::on_reset_done()
{
  if (phase_ == Phase::kFailed) {
    return;
  }
  if (phase_ == Phase::kRunning || phase_ == Phase::kHalted) {
    RCLCPP_INFO(get_logger(), "sim reset: re-running start-up");
  }
  publish_velocity(Joints::Zero());
  phase_ = Phase::kWaitState;
}

void TiltControlNode::on_joint_state(const sensor_msgs::msg::JointState & msg)
{
  const auto q = joints_from_state(msg);
  if (!q) {
    RCLCPP_WARN_THROTTLE(get_logger(), steady_, 2000, "joint state without the 6 UR joints; ignored");
    return;
  }
  const rclcpp::Time stamp(msg.header.stamp, get_clock()->get_clock_type());
  const bool jumped = q_ && joint_jump(*q_, *q, jump_threshold_);
  double dt = 0.0;
  if (q_stamp_ && q_stamp_->get_clock_type() == stamp.get_clock_type()) {
    dt = (stamp - *q_stamp_).seconds();
  }
  q_ = q;
  q_stamp_ = stamp;
  state_received_ = now_steady();
  watchdog_->on_state(*state_received_);

  // Contract 4b: measured tilt from FK, stamped with the joint state
  const Eigen::Isometry3d board = kin_->fk_board(*q);
  const Eigen::Vector2d tilt = tilt_of(home_.linear(), board.linear());
  maze_interfaces::msg::BoardTilt state;
  state.header.stamp = msg.header.stamp;
  state.header.frame_id = "maze_board";
  state.tilt_x = tilt.x();
  state.tilt_y = tilt.y();
  state_pub_->publish(state);

  if (phase_ != Phase::kRunning) {
    return;
  }
  if (jumped) {
    resync("joint jump");
    publish_velocity(Joints::Zero());
    return;
  }
  const double t = now_steady();
  if (!watchdog_->ok(t) || !speed_scaling_ok()) {
    return;  // the timer sends the zeros and the warning
  }
  if (!commanding_) {
    resync("commands resumed");  // start from where the arm stopped
  }
  if (!(dt > 0.0) || dt > kMaxStepDt) {
    return;  // first, duplicate or late sample: next one
  }
  ref_->step(dt);
  const Joints qdot = ctrl_->step(*q, ref_->value(), ref_->rate(), dt);
  publish_velocity(qdot);
  commanding_ = true;

  const auto & dg = ctrl_->diag();
  if (std::abs(yaw_of(home_.linear(), board.linear())) > yaw_warn_) {
    RCLCPP_WARN_THROTTLE(get_logger(), steady_, 2000, "board yaw drifted %.2f deg from home",
      yaw_of(home_.linear(), board.linear()) / kDeg);
  }
  RCLCPP_DEBUG_THROTTLE(get_logger(), steady_, 1000,
    "pos err %.2f mm, rot err %.3f deg, sigma_min %.3f%s%s%s", dg.pos_err * 1e3, dg.rot_err / kDeg,
    dg.sigma_min, dg.twist_capped ? ", twist capped" : "", dg.vel_scaled ? ", speed scaled" : "",
    dg.accel_limited ? ", accel limited" : "");
}

void TiltControlNode::resync(const char * why)
{
  ref_->reset(measured_tilt());
  if (last_cmd_) {
    ref_->set_command(*last_cmd_);
  }
  ctrl_->reset();
  RCLCPP_INFO(get_logger(), "resync (%s): reference from measured tilt (%.2f, %.2f) deg", why,
    ref_->value().x() / kDeg, ref_->value().y() / kDeg);
}

void TiltControlNode::fail(const std::string & why)
{
  publish_velocity(Joints::Zero());
  phase_ = Phase::kFailed;
  RCLCPP_ERROR(get_logger(), "refusing to run: %s", why.c_str());  // main() exits on failed()
}

void TiltControlNode::on_timer()
{
  const double t = now_steady();
  const bool fresh = state_received_ && t - *state_received_ <= state_timeout_;
  switch (phase_) {
    case Phase::kWaitState:
      if (fresh) {
        phase_ = Phase::kCheckHome;
        check_home();
      }
      return;
    case Phase::kCheckHome:
      check_home();
      return;
    case Phase::kSwitch:
      request_switch();
      return;
    case Phase::kRunning:
      break;
    case Phase::kHalted:
    case Phase::kFailed:
      publish_velocity(Joints::Zero());
      return;
  }

  // running: the watchdog, independent of the joint-state callback
  std::string why = watchdog_->reason(t);
  if (why.empty() && !speed_scaling_ok()) {
    why = speed_scaling_ ? "speed scaling below full speed" : "no speed scaling yet";
  }
  if (!why.empty()) {
    publish_velocity(Joints::Zero());
    ctrl_->set_last_command(Joints::Zero());
    if (commanding_ || !warned_) {
      RCLCPP_WARN(get_logger(), "holding the arm (zero velocities): %s", why.c_str());
      warned_ = true;
    }
    commanding_ = false;
  } else {
    warned_ = false;
  }
  if (t - last_poll_ >= kPollPeriod) {
    last_poll_ = t;
    poll_controller();
  }
}

void TiltControlNode::check_home()
{
  const auto c = home_check(home_, kin_->fk_board(*q_), home_pos_tol_, home_rot_tol_);
  if (!c.ok) {
    char buf[200];
    std::snprintf(buf, sizeof(buf),
      "the board is %.1f mm and %.2f deg from the home pose (limits %.1f mm, %.2f deg); "
      "move the arm to home_joints first", c.pos_err * 1e3, c.rot_err / kDeg, home_pos_tol_ * 1e3,
      home_rot_tol_ / kDeg);
    fail(buf);
    return;
  }
  RCLCPP_INFO(get_logger(), "at home (%.2f mm, %.3f deg off); switching controllers", c.pos_err * 1e3,
    c.rot_err / kDeg);
  phase_ = Phase::kSwitch;
  request_switch();
}

void TiltControlNode::request_switch()
{
  if (request_pending_) {
    return;
  }
  if (!list_cli_->service_is_ready()) {
    RCLCPP_INFO_THROTTLE(get_logger(), steady_, 5000, "waiting for /controller_manager");
    return;
  }
  request_pending_ = true;
  list_cli_->async_send_request(
    std::make_shared<ListControllers::Request>(),
    [this](rclcpp::Client<ListControllers>::SharedFuture f) {
      request_pending_ = false;
      if (phase_ == Phase::kSwitch) {
        on_list(*f.get());
      }
    });
}

void TiltControlNode::on_list(const ListControllers::Response & res)
{
  bool velocity_active = false, velocity_known = false;
  std::vector<std::string> deactivate;
  for (const auto & c : res.controller) {
    if (c.name == velocity_controller_) {
      velocity_known = true;
      velocity_active = c.state == "active";
    } else if (c.state == "active" &&
      std::find(deactivate_controllers_.begin(), deactivate_controllers_.end(), c.name) !=
      deactivate_controllers_.end())
    {
      deactivate.push_back(c.name);
    }
  }
  if (!velocity_known) {
    fail("controller '" + velocity_controller_ + "' is not loaded");
    return;
  }
  if (velocity_active && deactivate.empty()) {
    start_running();
    return;
  }
  auto req = std::make_shared<SwitchController::Request>();
  if (!velocity_active) {
    req->activate_controllers = {velocity_controller_};
  }
  req->deactivate_controllers = deactivate;
  req->strictness = SwitchController::Request::STRICT;
  request_pending_ = true;
  switch_cli_->async_send_request(
    req, [this](rclcpp::Client<SwitchController>::SharedFuture f) {
      request_pending_ = false;
      if (phase_ != Phase::kSwitch) {
        return;
      }
      if (!f.get()->ok) {
        fail("switch_controller (STRICT) to '" + velocity_controller_ + "' failed");
        return;
      }
      request_switch();  // list again: running once the velocity controller shows active
    });
}

void TiltControlNode::start_running()
{
  publish_velocity(Joints::Zero());
  phase_ = Phase::kRunning;
  commanding_ = false;
  warned_ = false;
  last_poll_ = now_steady();
  resync("start");
  RCLCPP_INFO(get_logger(), "running: '%s' active, following tilt_cmd", velocity_controller_.c_str());
}

void TiltControlNode::poll_controller()
{
  if (request_pending_ || !list_cli_->service_is_ready()) {
    return;
  }
  request_pending_ = true;
  list_cli_->async_send_request(
    std::make_shared<ListControllers::Request>(),
    [this](rclcpp::Client<ListControllers>::SharedFuture f) {
      request_pending_ = false;
      if (phase_ != Phase::kRunning) {
        return;
      }
      for (const auto & c : f.get()->controller) {
        if (c.name == velocity_controller_ && c.state == "active") {
          return;
        }
      }
      publish_velocity(Joints::Zero());
      phase_ = Phase::kHalted;
      RCLCPP_ERROR(get_logger(), "'%s' is no longer active: halted until restart (or /sim/reset_done)",
        velocity_controller_.c_str());
    });
}

}  // namespace maze_arm_control
