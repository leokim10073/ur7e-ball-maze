// The C++ tilt controller: FK from the calibration file matches the UR DH model, it matches the
// Python reference (maze_arm), and the same kinematic closed-loop checks as test_tilt_control.py.

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <gtest/gtest.h>

#include "maze_arm_control/kinematics.hpp"
#include "maze_arm_control/tilt_controller.hpp"

using namespace maze_arm_control;

namespace
{
constexpr double DT = 0.01;  // controller step, s

// Test inputs: sim.yaml's start pose and grasp, and the L1 handle center, when this was written.
// The checks below hold for any reasonable values; the Python reference numbers use these.
const Vector6d Q0 = (Vector6d() << -2.7498, -1.6799, -2.2485, 0.7868, 1.1837, -1.5708).finished();
const Eigen::Vector3d GRASP_XYZ(0.0, 0.0, 0.15);
const Eigen::Vector3d GRASP_RPY(3.141593, -1.570796, 0.0);
const Eigen::Vector3d HANDLE_CENTER(-0.175, 0.0, 0.0);

UrKinematics ur7e()
{
  return load_ur_kinematics(ament_index_cpp::get_package_share_directory("ur_description") +
                            "/config/ur7e/default_kinematics.yaml");
}

// The UR DH model (base_link = DH base turned by pi about z), as in maze_arm/kinematics.py.
Eigen::Isometry3d dh_tool0(const Vector6d& q)
{
  const double d[6] = { 0.1625, 0, 0, 0.1333, 0.0997, 0.0996 };
  const double a[6] = { 0, -0.425, -0.3922, 0, 0, 0 };
  const double alpha[6] = { M_PI / 2, 0, 0, M_PI / 2, -M_PI / 2, 0 };
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.linear() = Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  for (int i = 0; i < 6; ++i)
  {
    const double ct = std::cos(q[i]), st = std::sin(q[i]), ca = std::cos(alpha[i]), sa = std::sin(alpha[i]);
    Eigen::Matrix4d A;
    A << ct, -st * ca, st * sa, a[i] * ct, st, ct * ca, -ct * sa, a[i] * st, 0, sa, ca, d[i], 0, 0, 0, 1;
    T = T * Eigen::Isometry3d(A);
  }
  return T;
}

struct LoopResult
{
  std::vector<Eigen::Vector3d> centers;
  std::vector<Eigen::Vector2d> tilts;
  double peak_speed = 0.0;
  Vector6d q;
};

// Kinematic loop: the arm follows the commanded joint velocities exactly.
template <class Cmd>
LoopResult simulate(TiltController& ctrl, Vector6d q, Cmd cmd, double seconds)
{
  LoopResult r;
  const int n = static_cast<int>(std::lround(seconds / DT));
  for (int k = 0; k < n; ++k)
  {
    const Vector6d qdot = ctrl.step(q, cmd(k * DT), DT);
    q += qdot * DT;
    r.centers.push_back(ctrl.board_pose(q).translation());
    r.tilts.push_back(ctrl.measured_tilt(q));
    r.peak_speed = std::max(r.peak_speed, qdot.cwiseAbs().maxCoeff());
  }
  r.q = q;
  return r;
}

auto constant(double tx, double ty)
{
  return [=](double) { return Eigen::Vector2d(tx, ty); };
}

class TiltControllerTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    kin = ur7e();
    cfg = load_arm_config(ARM_YAML);
    offset = board_offset(GRASP_XYZ, GRASP_RPY, HANDLE_CENTER);
    ctrl = std::make_unique<TiltController>(cfg, kin, offset, level_home(fk_tool0(kin, Q0) * offset));
    ctrl->reset(Q0);
  }

  double max_drift_mm(const LoopResult& r) const
  {
    double m = 0.0;
    for (const auto& c : r.centers)
    {
      m = std::max(m, (c - ctrl->home().translation()).norm() * 1000.0);
    }
    return m;
  }

  UrKinematics kin;
  ArmConfig cfg{};
  Eigen::Isometry3d offset;
  std::unique_ptr<TiltController> ctrl;
};
}  // namespace

// --- kinematics ---

TEST(Kinematics, CalibrationFileMatchesDh)
{
  const UrKinematics kin = ur7e();
  std::srand(0);
  for (int k = 0; k < 15; ++k)
  {
    const Vector6d q = Q0 + Vector6d::Random();
    const Eigen::Isometry3d T = fk_tool0(kin, q), T_dh = dh_tool0(q);
    EXPECT_LT((T.translation() - T_dh.translation()).norm(), 1e-9);
    EXPECT_LT((T.linear() - T_dh.linear()).cwiseAbs().maxCoeff(), 1e-9);
  }
}

TEST(Kinematics, MatchesPythonReference)
{
  // maze_arm.kinematics.fk_tool0(Q0)
  const Eigen::Isometry3d T = fk_tool0(ur7e(), Q0);
  EXPECT_NEAR(T.translation().x(), 0.44922179455904165, 1e-9);
  EXPECT_NEAR(T.translation().y(), 0.00068636663750656717, 1e-9);
  EXPECT_NEAR(T.translation().z(), 0.40695450420735790, 1e-9);
  EXPECT_NEAR(T.linear()(0, 2), 0.99998897225449168, 1e-9);
  EXPECT_NEAR(T.linear()(1, 1), -0.99998897225664629, 1e-9);
}

TEST(Kinematics, JointsFromStateAnyOrder)
{
  std::vector<std::string> names;
  std::vector<double> pos;
  for (int i : { 1, 2, 3, 4, 5, 0 })
  {
    names.push_back(UR_JOINTS[i]);
    pos.push_back(Q0[i]);
  }
  EXPECT_LT((joints_from_state(names, pos) - Q0).norm(), 1e-15);
  names.pop_back();
  pos.pop_back();
  EXPECT_THROW(joints_from_state(names, pos), std::runtime_error);
}

TEST_F(TiltControllerTest, JacobianMatchesFiniteDifferences)
{
  const Matrix6d J = board_jacobian(kin, Q0, offset);
  const Eigen::Isometry3d T0 = fk_tool0(kin, Q0) * offset;
  const double eps = 1e-6;
  for (int i = 0; i < 6; ++i)
  {
    Vector6d q = Q0;
    q[i] += eps;
    const Eigen::Isometry3d T1 = fk_tool0(kin, q) * offset;
    EXPECT_LT(((T1.translation() - T0.translation()) / eps - J.block<3, 1>(0, i)).norm(), 1e-5);
    EXPECT_LT((rotation_log(T1.linear() * T0.linear().transpose()) / eps - J.block<3, 1>(3, i)).norm(), 1e-5);
  }
}

// --- tilt convention (Contract 4) ---

TEST(TiltConvention, Signs)
{
  EXPECT_LT((tilt_rotation(0.05, 0.0) * Eigen::Vector3d(0.1, 0, 0)).z(), 0.0);  // tilt_x lowers +x edge
  EXPECT_LT((tilt_rotation(0.0, 0.05) * Eigen::Vector3d(0, 0.1, 0)).z(), 0.0);  // tilt_y lowers +y edge
  EXPECT_NEAR((tilt_rotation(0.05, 0.0) * Eigen::Vector3d(0, 0.1, 0)).z(), 0.0, 1e-12);
  EXPECT_NEAR((tilt_rotation(0.0, 0.05) * Eigen::Vector3d(0.1, 0, 0)).z(), 0.0, 1e-12);
}

TEST(TiltConvention, RoundTrip)
{
  for (const Eigen::Vector2d& t : { Eigen::Vector2d(0.05, 0.0), Eigen::Vector2d(0.0, -0.03),
                                   Eigen::Vector2d(0.06, 0.04), Eigen::Vector2d(-0.087, 0.087) })
  {
    EXPECT_LT((tilt_from_rotation(tilt_rotation(t[0], t[1])) - t).norm(), 1e-12);
  }
}

// --- closed loop (kinematic) ---

TEST_F(TiltControllerTest, MatchesPythonReference)
{
  // maze_arm.tilt_control: 20 steps of (0.05, -0.03) from Q0
  Vector6d q = Q0, qdot;
  for (int k = 0; k < 20; ++k)
  {
    qdot = ctrl->step(q, Eigen::Vector2d(0.05, -0.03), DT);
    q += qdot * DT;
  }
  const Vector6d qdot_py = (Vector6d() << -0.00046572068091278, 0.00651026914532287, 0.0031098895430829,
                            -0.01570002358757563, 0.00091742189312282, 0.00547292556866642)
                               .finished();
  const Vector6d q_py = (Vector6d() << -2.7503291829058854, -1.626736201696182, -2.2354702152332346,
                         0.6677985332925401, 1.1846510926425315, -1.5215022656757444)
                            .finished();
  EXPECT_LT((qdot - qdot_py).cwiseAbs().maxCoeff(), 1e-9);
  EXPECT_LT((q - q_py).cwiseAbs().maxCoeff(), 1e-9);
}

TEST_F(TiltControllerTest, StepReachesTiltAndCenterStays)
{
  for (const Eigen::Vector2d& cmd : { Eigen::Vector2d(0.05, 0.0), Eigen::Vector2d(0.0, 0.05),
                                     Eigen::Vector2d(0.0, -0.05), Eigen::Vector2d(0.04, -0.03) })
  {
    SetUp();
    const LoopResult r = simulate(*ctrl, Q0, constant(cmd[0], cmd[1]), 1.0);
    EXPECT_LT((r.tilts.back() - cmd).cwiseAbs().maxCoeff(), 1e-3) << cmd.transpose();
    EXPECT_LT(max_drift_mm(r), 1.0) << cmd.transpose();
    EXPECT_LE(r.peak_speed, cfg.max_joint_speed + 1e-9);
  }
}

TEST_F(TiltControllerTest, RateLimited)
{
  const LoopResult r = simulate(*ctrl, Q0, constant(0.087, 0.0), 0.5);
  double rate = 0.0;
  for (size_t k = 1; k < r.tilts.size(); ++k)
  {
    rate = std::max(rate, (r.tilts[k] - r.tilts[k - 1]).norm() / DT);
  }
  EXPECT_LE(rate, cfg.max_tilt_rate * 1.05);
}

TEST_F(TiltControllerTest, Clamped)
{
  const LoopResult r = simulate(*ctrl, Q0, constant(0.3, -0.3), 1.5);
  EXPECT_NEAR(r.tilts.back()[0], cfg.max_tilt, 1e-3);
  EXPECT_NEAR(r.tilts.back()[1], -cfg.max_tilt, 1e-3);
}

TEST_F(TiltControllerTest, SineTracking)
{
  const double amp = 0.04, f = 1.0;  // peak rate 0.25 rad/s, under the rate limit
  const LoopResult r = simulate(
      *ctrl, Q0, [=](double t) { return Eigen::Vector2d(amp * std::sin(2 * M_PI * f * t), 0.0); }, 3.0);
  double err = 0.0;
  for (size_t k = 0; k < r.tilts.size(); ++k)
  {
    const double t = (k + 1) * DT;
    if (t > 1.0)
    {
      err = std::max(err, std::abs(r.tilts[k][0] - amp * std::sin(2 * M_PI * f * t)));
    }
  }
  EXPECT_LT(err, 0.005);
  EXPECT_LT(max_drift_mm(r), 1.0);
}

TEST_F(TiltControllerTest, SagIsPulledBack)
{
  // /joint_states read from the sim after sagging under zero commands (wrist_3 at its start value)
  Vector6d q;
  q << -2.74984, -1.68144, -2.25004, 0.77801, 1.18374, Q0[5];
  EXPECT_GT((ctrl->board_pose(q).translation() - ctrl->home().translation()).norm() * 1000, 3.0);
  const LoopResult r = simulate(*ctrl, q, constant(0.0, 0.0), 1.0);
  EXPECT_LT((r.centers.back() - ctrl->home().translation()).norm() * 1000, 0.1);
  EXPECT_LT(r.tilts.back().cwiseAbs().maxCoeff(), 1e-3);
}

// --- config ---

TEST(Config, RejectsMissingAndNonPositive)
{
  const std::string path = ::testing::TempDir() + "arm_test.yaml";
  std::ofstream(path) << "max_tilt: 0.087\n";
  EXPECT_THROW(load_arm_config(path), std::runtime_error);
  std::ofstream(path) << "max_tilt: 0.087\nmax_tilt_rate: 0.5\nmax_joint_speed: 1.0\nk_pos: 5\nk_rot: 5\n"
                         "damping: 0\ncommand_timeout: 0.2\n";
  EXPECT_THROW(load_arm_config(path), std::runtime_error);
}
