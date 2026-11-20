// The node in-process against a fake arm (integrates the velocity commands) and a fake controller
// manager: parameters, start-up, the STRICT switch, tracking, the watchdog and the home check.
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <memory>
#include <thread>

#include "maze_arm_control/so3.hpp"
#include "test_support.hpp"
#include "tilt_control_node.hpp"

using maze_arm_control::Joints;
using maze_arm_control::TiltControlNode;
using controller_manager_msgs::srv::ListControllers;
using controller_manager_msgs::srv::SwitchController;
using namespace std::chrono_literals;

namespace
{

rclcpp::NodeOptions options_from(const std::string & params_file)
{
  return rclcpp::NodeOptions().arguments({"--ros-args", "--params-file", params_file});
}

// The shipped config with one key removed (dotted path), written to a temporary file.
std::string config_without(const std::string & dotted)
{
  YAML::Node root = YAML::LoadFile(MAZE_ARM_CONTROL_CONFIG);
  YAML::Node node = root["tilt_control"]["ros__parameters"];
  std::string rest = dotted;
  for (size_t dot; (dot = rest.find('.')) != std::string::npos; rest = rest.substr(dot + 1)) {
    node.reset(node[rest.substr(0, dot)]);
  }
  node.remove(rest);
  const std::string path = testing::TempDir() + "tilt_control_missing.yaml";
  std::ofstream(path) << root;
  return path;
}

// A UR arm that does exactly what the velocity controller is told, and its controller manager.
class FakeArm : public rclcpp::Node
{
public:
  explicit FakeArm(const Joints & q0)
  : rclcpp::Node("fake_arm"), q(q0)
  {
    pub_ = create_publisher<sensor_msgs::msg::JointState>("/joint_states", 10);
    sub_ = create_subscription<std_msgs::msg::Float64MultiArray>(
      "/forward_velocity_controller/commands", 10, [this](const std_msgs::msg::Float64MultiArray & m) {
        commands++;
        if (m.data.size() == 6 && states["forward_velocity_controller"] == "active") {
          qdot = Eigen::Map<const Joints>(m.data.data());
        }
      });
    list_ = create_service<ListControllers>(
      "/controller_manager/list_controllers",
      [this](const ListControllers::Request::SharedPtr, ListControllers::Response::SharedPtr res) {
        for (const auto & [name, state] : states) {
          controller_manager_msgs::msg::ControllerState c;
          c.name = name;
          c.state = state;
          res->controller.push_back(c);
        }
      });
    switch_ = create_service<SwitchController>(
      "/controller_manager/switch_controller",
      [this](const SwitchController::Request::SharedPtr req, SwitchController::Response::SharedPtr res) {
        last_switch = *req;
        switches++;
        res->ok = switch_ok;
        if (switch_ok) {
          for (const auto & n : req->deactivate_controllers) {states[n] = "inactive";}
          for (const auto & n : req->activate_controllers) {states[n] = "active";}
        }
      });
    timer_ = create_wall_timer(2ms, [this] {
        q += qdot * 0.002;
        sensor_msgs::msg::JointState js;
        js.header.stamp = now();
        js.name = {"wrist_3_joint", "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint",
          "wrist_2_joint", "gripper_right_driver_joint"};  // any order, an extra joint
        js.position = {q[5], q[0], q[1], q[2], q[3], q[4], 0.5};
        pub_->publish(js);
      });
  }

  Joints q, qdot = Joints::Zero();
  std::map<std::string, std::string> states = {
    {"scaled_joint_trajectory_controller", "active"}, {"forward_velocity_controller", "inactive"},
    {"forward_position_controller", "inactive"}, {"joint_state_broadcaster", "active"}};
  bool switch_ok = true;
  int switches = 0, commands = 0;
  SwitchController::Request last_switch;

private:
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr pub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr sub_;
  rclcpp::Service<ListControllers>::SharedPtr list_;
  rclcpp::Service<SwitchController>::SharedPtr switch_;
  rclcpp::TimerBase::SharedPtr timer_;
};

class NodeTest : public ::testing::Test
{
protected:
  static void SetUpTestSuite() {rclcpp::init(0, nullptr);}
  static void TearDownTestSuite() {rclcpp::shutdown();}

  void start(const Joints & q0)
  {
    arm = std::make_shared<FakeArm>(q0);
    node = std::make_shared<TiltControlNode>(options_from(MAZE_ARM_CONTROL_CONFIG));
    helper = std::make_shared<rclcpp::Node>("helper");
    cmd_pub = helper->create_publisher<maze_interfaces::msg::BoardTilt>("/board/tilt_cmd", rclcpp::QoS(1).reliable());
    state_sub = helper->create_subscription<maze_interfaces::msg::BoardTilt>(
      "/board/tilt_state", rclcpp::QoS(1).best_effort(),
      [this](const maze_interfaces::msg::BoardTilt & m) {tilt_state = m;});
    exec.add_node(arm);
    exec.add_node(node);
    exec.add_node(helper);
  }

  void TearDown() override
  {
    exec.cancel();
    for (auto n : std::vector<rclcpp::Node::SharedPtr>{arm, node, helper}) {
      if (n) {exec.remove_node(n);}
    }
  }

  // Spin for `seconds`, publishing `cmd` at 60 Hz if given.
  void run(double seconds, std::optional<Eigen::Vector2d> cmd = std::nullopt)
  {
    const auto end = std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);
    auto next_cmd = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() < end) {
      if (cmd && std::chrono::steady_clock::now() >= next_cmd) {
        maze_interfaces::msg::BoardTilt m;
        m.tilt_x = cmd->x();
        m.tilt_y = cmd->y();
        cmd_pub->publish(m);
        next_cmd += std::chrono::microseconds(16667);
      }
      exec.spin_some(1ms);
      std::this_thread::sleep_for(200us);
    }
  }

  rclcpp::executors::SingleThreadedExecutor exec;
  std::shared_ptr<FakeArm> arm;
  std::shared_ptr<TiltControlNode> node;
  rclcpp::Node::SharedPtr helper;
  rclcpp::Publisher<maze_interfaces::msg::BoardTilt>::SharedPtr cmd_pub;
  rclcpp::Subscription<maze_interfaces::msg::BoardTilt>::SharedPtr state_sub;
  std::optional<maze_interfaces::msg::BoardTilt> tilt_state;
};

}  // namespace

TEST_F(NodeTest, MissingParameterIsNamed)
{
  for (const std::string key : {"gains.k_pos", "watchdog.command_timeout", "home_joints", "topics.tilt_cmd"}) {
    try {
      TiltControlNode n(options_from(config_without(key)));
      ADD_FAILURE() << "no error without " << key;
    } catch (const std::runtime_error & e) {
      EXPECT_NE(std::string(e.what()).find("'" + key + "'"), std::string::npos) << e.what();
    }
  }
}

TEST_F(NodeTest, JointsFromStateByName)
{
  sensor_msgs::msg::JointState js;
  js.name = {"elbow_joint", "x", "shoulder_pan_joint", "shoulder_lift_joint", "wrist_1_joint", "wrist_2_joint",
    "wrist_3_joint"};
  js.position = {3, 9, 1, 2, 4, 5, 6};
  const auto q = maze_arm_control::joints_from_state(js);
  ASSERT_TRUE(q);
  EXPECT_EQ(*q, (Joints() << 1, 2, 3, 4, 5, 6).finished());
  js.name.pop_back();
  js.position.pop_back();
  EXPECT_FALSE(maze_arm_control::joints_from_state(js));  // wrist_3 missing
  js.name.push_back("wrist_3_joint");
  EXPECT_FALSE(maze_arm_control::joints_from_state(js));  // sizes differ
}

TEST_F(NodeTest, StartsTracksAndStopsOnTimeout)
{
  start(test_support::home_joints());
  run(0.5);  // no command yet: switched and running, but holding still
  ASSERT_EQ(node->phase(), TiltControlNode::Phase::kRunning);
  EXPECT_EQ(arm->last_switch.strictness, SwitchController::Request::STRICT);
  EXPECT_EQ(arm->last_switch.activate_controllers, std::vector<std::string>{"forward_velocity_controller"});
  EXPECT_EQ(arm->last_switch.deactivate_controllers,
    std::vector<std::string>{"scaled_joint_trajectory_controller"});
  EXPECT_EQ(arm->qdot, Joints::Zero());

  const auto kin = test_support::sim_kinematics();
  const Eigen::Isometry3d home = kin.fk_board(test_support::home_joints());
  run(1.0, Eigen::Vector2d(0.035, -0.02));
  const Eigen::Isometry3d board = kin.fk_board(arm->q);
  const Eigen::Vector2d tilt = maze_arm_control::tilt_of(home.linear(), board.linear());
  EXPECT_NEAR(tilt.x(), 0.035, 0.002);
  EXPECT_NEAR(tilt.y(), -0.02, 0.002);
  EXPECT_LT((board.translation() - home.translation()).norm(), 0.001);
  ASSERT_TRUE(tilt_state);
  EXPECT_NEAR(tilt_state->tilt_x, 0.035, 0.002);
  EXPECT_EQ(tilt_state->header.frame_id, "maze_board");

  // the command stream stops: zero velocities within the 200 ms timeout (+ a timer period)
  arm->qdot.setConstant(0.01);  // as if the arm were still moving: the node must overwrite it
  run(0.25);
  EXPECT_EQ(arm->qdot, Joints::Zero());
  EXPECT_EQ(node->phase(), TiltControlNode::Phase::kRunning);

  // the stopped arm drifts 1.7 deg of tilt_x about the board center; when commands resume the board
  // eases back (acceleration-limited reference and joints, board-rate cap) instead of jumping
  {
    Eigen::Matrix<double, 6, 1> twist = Eigen::Matrix<double, 6, 1>::Zero();
    twist.tail<3>() = kin.fk_board(arm->q).linear() * Eigen::Vector3d(0, 0.03, 0);  // about board +y
    const auto J = kin.jacobian_board(arm->q);
    arm->q += J.colPivHouseholderQr().solve(twist);
  }
  const Eigen::Vector2d before = maze_arm_control::tilt_of(home.linear(), kin.fk_board(arm->q).linear());
  run(0.03, Eigen::Vector2d(0.035, -0.02));
  const Eigen::Vector2d after = maze_arm_control::tilt_of(home.linear(), kin.fk_board(arm->q).linear());
  EXPECT_LT((after - before).cwiseAbs().maxCoeff(), 0.005);
}

TEST_F(NodeTest, IgnoresNonFiniteCommand)
{
  start(test_support::home_joints());
  run(0.3, Eigen::Vector2d(0.0, 0.0));
  run(0.3, Eigen::Vector2d(std::nan(""), 0.0));
  EXPECT_TRUE(arm->q.allFinite());
  EXPECT_EQ(arm->qdot, Joints::Zero());  // NaN does not count as a command: the watchdog stopped it
}

TEST_F(NodeTest, RefusesAwayFromHome)
{
  Joints q = test_support::home_joints();
  q[4] += 0.05;  // wrist 2: ~3 deg of board rotation
  start(q);
  run(0.3);
  EXPECT_EQ(node->phase(), TiltControlNode::Phase::kFailed);
  EXPECT_EQ(arm->switches, 0);
  EXPECT_EQ(arm->states["forward_velocity_controller"], "inactive");
}

TEST_F(NodeTest, FailedSwitchRefuses)
{
  start(test_support::home_joints());
  arm->switch_ok = false;
  run(0.3);
  EXPECT_EQ(node->phase(), TiltControlNode::Phase::kFailed);
  EXPECT_EQ(arm->switches, 1);
}

TEST_F(NodeTest, SimResetResyncsWithoutHomeCheck)
{
  start(test_support::home_joints());
  auto reset_pub = helper->create_publisher<std_msgs::msg::Empty>(
    "/sim/reset_done", rclcpp::QoS(1).reliable().transient_local());
  run(0.3, Eigen::Vector2d(0.0, 0.0));
  ASSERT_EQ(node->phase(), TiltControlNode::Phase::kRunning);
  // a reset: the controller goes away for a moment, the arm comes back drooped (> home_tolerance)
  arm->states["forward_velocity_controller"] = "inactive";
  run(1.2, Eigen::Vector2d(0.0, 0.0));
  ASSERT_EQ(node->phase(), TiltControlNode::Phase::kHalted);
  arm->states["forward_velocity_controller"] = "active";
  arm->q[1] += 0.01;  // shoulder lift: ~8 mm at the board
  reset_pub->publish(std_msgs::msg::Empty());
  run(0.5, Eigen::Vector2d(0.0, 0.0));
  EXPECT_EQ(node->phase(), TiltControlNode::Phase::kRunning);
  const auto kin = test_support::sim_kinematics();
  const double off = (kin.fk_board(arm->q).translation() -
    kin.fk_board(test_support::home_joints()).translation()).norm();
  EXPECT_LT(off, 0.0005);  // pulled back to home
}

TEST_F(NodeTest, LostServiceReplyFails)
{
  // a controller manager that takes the request and never answers: no silent hang in start-up
  start(test_support::home_joints());
  exec.remove_node(arm);
  arm.reset();  // services gone with the arm: replaced by ones that never reply
  auto silent = std::make_shared<rclcpp::Node>("silent_cm");
  std::vector<std::shared_ptr<rmw_request_id_t>> held;
  auto list = silent->create_service<ListControllers>(
    "/controller_manager/list_controllers",
    [&held](std::shared_ptr<rmw_request_id_t> id, const ListControllers::Request::SharedPtr) {held.push_back(id);});
  exec.add_node(silent);
  auto js_pub = helper->create_publisher<sensor_msgs::msg::JointState>("/joint_states", 10);
  const Joints q = test_support::home_joints();
  const auto end = std::chrono::steady_clock::now() + 6.5s;
  while (std::chrono::steady_clock::now() < end && node->phase() != TiltControlNode::Phase::kFailed) {
    sensor_msgs::msg::JointState js;
    js.header.stamp = helper->now();
    js.name = maze_arm_control::kUrJoints;
    js.position.assign(q.data(), q.data() + 6);
    js_pub->publish(js);
    exec.spin_some(1ms);
    std::this_thread::sleep_for(2ms);
  }
  EXPECT_FALSE(held.empty());
  EXPECT_EQ(node->phase(), TiltControlNode::Phase::kFailed);
  exec.remove_node(silent);
}

TEST_F(NodeTest, HaltsWhenVelocityControllerGoesAway)
{
  start(test_support::home_joints());
  run(0.3, Eigen::Vector2d(0.0, 0.0));
  ASSERT_EQ(node->phase(), TiltControlNode::Phase::kRunning);
  arm->states["forward_velocity_controller"] = "inactive";  // e.g. controller_stopper on the robot
  run(1.3, Eigen::Vector2d(0.03, 0.0));
  EXPECT_EQ(node->phase(), TiltControlNode::Phase::kHalted);
}
