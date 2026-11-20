// Kinematic-only closed loop of the controller (perfect joint velocity tracking, 500 Hz), the C++
// version of the plan's validation sweep: steps and sines on both axes from the home pose.
#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <string>

#include "maze_arm_control/reference.hpp"
#include "maze_arm_control/so3.hpp"
#include "maze_arm_control/tilt_controller.hpp"
#include "test_support.hpp"

using namespace maze_arm_control;

namespace
{

constexpr double kDt = 0.002;
const double kDeg = M_PI / 180.0;

struct Result
{
  double drift_mm = 0, settled_err_deg = 0, qd_max = 0, qdd_ratio = 0, sigma_min = 1e9, yaw_deg = 0;
};

// profile(t) is the command; errors count once `settled(t)` is true.
Result run(std::function<Eigen::Vector2d(double)> profile, double t_end, std::function<bool(double)> settled)
{
  const auto kin = test_support::sim_kinematics();
  const auto cp = test_support::controller_params();
  Joints q = test_support::home_joints();
  const Eigen::Isometry3d home = kin.fk_board(q);
  TiltController ctrl(kin, cp, home);
  ReferenceShaper ref(test_support::reference_params());
  ref.reset({0, 0});
  Result r;
  Joints last = Joints::Zero();
  for (int k = 0; k * kDt < t_end; ++k) {
    const double t = k * kDt;
    const Eigen::Vector2d cmd = profile(t);
    ref.set_command(cmd);
    ref.step(kDt);
    const Joints qd = ctrl.step(q, ref.value(), ref.rate(), kDt);
    r.qd_max = std::max(r.qd_max, (qd.cwiseAbs().array() / cp.max_joint_speed.array()).maxCoeff());
    r.qdd_ratio = std::max(r.qdd_ratio, (((qd - last) / kDt).cwiseAbs().array() / cp.max_joint_accel.array()).maxCoeff());
    r.sigma_min = std::min(r.sigma_min, ctrl.diag().sigma_min);
    last = qd;
    q += qd * kDt;
    const auto T = kin.fk_board(q);
    r.drift_mm = std::max(r.drift_mm, 1000.0 * (T.translation() - home.translation()).norm());
    // yaw against the desired orientation: Ry*Rx itself carries a second-order yaw (tx*ty/2)
    const Eigen::Matrix3d R_d = home.linear() * tilt_rotation(ref.value().x(), ref.value().y());
    r.yaw_deg = std::max(r.yaw_deg, std::abs(yaw_of(R_d, T.linear())) / kDeg);
    if (settled(t)) {
      const Eigen::Vector2d clamped = cmd.cwiseMax(-0.087).cwiseMin(0.087);
      r.settled_err_deg = std::max(
        r.settled_err_deg, (tilt_of(home.linear(), T.linear()) - clamped).cwiseAbs().maxCoeff() / kDeg);
    }
  }
  return r;
}

void expect_good(const Result & r, const std::string & name)
{
  EXPECT_LT(r.drift_mm, 1.0) << name;
  EXPECT_LT(r.settled_err_deg, 0.5) << name;
  EXPECT_LE(r.qd_max, 1.0 + 1e-9) << name;
  EXPECT_LE(r.qdd_ratio, 1.0 + 1e-6) << name;
  EXPECT_GT(r.sigma_min, 0.1) << name;
  EXPECT_LT(r.yaw_deg, 0.1) << name;
  std::printf("%-22s drift %.3f mm  err %.3f deg  qd %.2f  qdd %.2f  sigma %.3f\n", name.c_str(),
    r.drift_mm, r.settled_err_deg, r.qd_max, r.qdd_ratio, r.sigma_min);
}

}  // namespace

TEST(ControllerLoop, Steps)
{
  const double d = 5 * kDeg;
  for (const auto & [name, target] : std::initializer_list<std::pair<std::string, Eigen::Vector2d>>{
      {"+x", {d, 0}}, {"-x", {-d, 0}}, {"+y", {0, d}}, {"-y", {0, -d}}, {"+x+y", {d, d}}, {"-x+y", {-d, d}},
      {"+x-y", {d, -d}}, {"-x-y", {-d, -d}}, {"beyond limit", {0.2, -0.2}}})
  {
    const Eigen::Vector2d tgt = target;
    expect_good(run([tgt](double t) {return t >= 0.1 ? tgt : Eigen::Vector2d::Zero();}, 1.5,
      [](double t) {return t > 0.6;}), "step " + name);
  }
}

TEST(ControllerLoop, Sines)
{
  const double d = 5 * kDeg;
  const auto after_1s = [](double t) {return t > 1.0;};
  expect_good(run([d](double t) {return Eigen::Vector2d(d * std::sin(M_PI * t), 0);}, 4.0, after_1s), "sine x");
  expect_good(run([d](double t) {return Eigen::Vector2d(0, d * std::sin(M_PI * t));}, 4.0, after_1s), "sine y");
  expect_good(run([d](double t) {
      return Eigen::Vector2d(d * std::sin(M_PI * t), d * (std::cos(M_PI * t) - 1));}, 4.0, after_1s), "circle");
  expect_good(run([d](double t) {
      return Eigen::Vector2d(d * std::sin(M_PI * std::floor(t * 60) / 60), 0);}, 4.0, after_1s), "60 Hz hold sine");
}

TEST(ControllerLoop, FromRestLimitsAcceleration)
{
  // a reset() controller asked for a big tilt rate from rest ramps up, never jumps
  const auto kin = test_support::sim_kinematics();
  const auto cp = test_support::controller_params();
  const Joints q = test_support::home_joints();
  TiltController ctrl(kin, cp, kin.fk_board(q));
  const Joints qd = ctrl.step(q, {0.05, 0.0}, {0.5, 0.0}, kDt);
  EXPECT_LE((qd.cwiseAbs().array() / (cp.max_joint_accel.array() * kDt)).maxCoeff(), 1.0 + 1e-9);
  EXPECT_TRUE(ctrl.diag().accel_limited);
}

TEST(ControllerLoop, TwistIsCapped)
{
  // board 5 cm off its home: the linear correction is capped at the twist cap
  const auto kin = test_support::sim_kinematics();
  auto cp = test_support::controller_params();
  cp.max_joint_accel.setConstant(1e9);
  const Joints q = test_support::home_joints();
  Eigen::Isometry3d home = kin.fk_board(q);
  home.translation().z() += 0.05;
  TiltController ctrl(kin, cp, home);
  const Joints qd = ctrl.step(q, {0, 0}, {0, 0}, kDt);
  const Eigen::Vector3d v = (kin.jacobian_board(q) * qd).head<3>();
  EXPECT_TRUE(ctrl.diag().twist_capped);
  EXPECT_NEAR(v.norm(), cp.max_linear_speed, 1e-3);
  EXPECT_GT(v.z(), 0.0);
}

TEST(ControllerLoop, BoardRateCappedPerAxis)
{
  // a large tilt error (e.g. after a sim reset that moved the board 3 deg): the board turns no faster
  // than the 0.5 rad/s tilt-rate limit about either board axis
  const auto kin = test_support::sim_kinematics();
  auto cp = test_support::controller_params();
  cp.max_joint_accel.setConstant(1e9);
  const Joints q = test_support::home_joints();
  const Eigen::Isometry3d home = kin.fk_board(q);
  TiltController ctrl(kin, cp, home);
  for (const Eigen::Vector2d & tilt : {Eigen::Vector2d(0.06, 0.0), Eigen::Vector2d(0.0, -0.06),
      Eigen::Vector2d(0.06, 0.06)})
  {
    ctrl.reset();
    const Joints qd = ctrl.step(q, tilt, {0, 0}, kDt);
    const Eigen::Vector3d w_board = home.linear().transpose() * (kin.jacobian_board(q) * qd).tail<3>();
    EXPECT_LE(w_board.head<2>().cwiseAbs().maxCoeff(), cp.max_tilt_rate + 1e-6) << tilt.transpose();
    EXPECT_GT(w_board.head<2>().cwiseAbs().maxCoeff(), 0.9 * cp.max_tilt_rate) << tilt.transpose();
  }
}

TEST(HomeCheck, AcceptsAndRejects)
{
  Eigen::Isometry3d home = Eigen::Isometry3d::Identity();
  home.translation() << 0.77, 0.0, 0.41;
  Eigen::Isometry3d b = home;
  EXPECT_TRUE(home_check(home, b, 0.005, 1 * kDeg).ok);
  b.translation().x() += 0.006;
  auto c = home_check(home, b, 0.005, 1 * kDeg);
  EXPECT_FALSE(c.ok);
  EXPECT_NEAR(c.pos_err, 0.006, 1e-12);
  b = home;
  b.linear() = Eigen::AngleAxisd(1.2 * kDeg, Eigen::Vector3d::UnitY()).toRotationMatrix();
  c = home_check(home, b, 0.005, 1 * kDeg);
  EXPECT_FALSE(c.ok);
  EXPECT_NEAR(c.rot_err, 1.2 * kDeg, 1e-9);
}

TEST(JointJump, Threshold)
{
  Joints a = Joints::Zero(), b = Joints::Zero();
  b[3] = 0.09;
  EXPECT_FALSE(joint_jump(a, b, 0.1));
  b[3] = -0.11;
  EXPECT_TRUE(joint_jump(a, b, 0.1));
}
