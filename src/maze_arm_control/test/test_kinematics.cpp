// The C++ FK against maze_arm's Python FK (fixture), and the Jacobian against finite differences.
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

#include <vector>

#include "maze_arm_control/kinematics.hpp"
#include "maze_arm_control/so3.hpp"
#include "test_support.hpp"

using maze_arm_control::ArmKinematics;
using maze_arm_control::Joints;
using maze_arm_control::log_so3;

namespace
{

struct Pose
{
  Joints q;
  Eigen::Matrix4d T;
};

std::vector<Pose> fixture_poses()
{
  const YAML::Node root = YAML::LoadFile(std::string(MAZE_ARM_CONTROL_TEST_DATA) + "/fk_fixture.yaml");
  std::vector<Pose> out;
  for (const auto & p : root["poses"]) {
    Pose pose;
    for (int i = 0; i < 6; ++i) {pose.q[i] = p["q"][i].as<double>();}
    for (int i = 0; i < 16; ++i) {pose.T(i / 4, i % 4) = p["T"][i].as<double>();}
    out.push_back(pose);
  }
  return out;
}

}  // namespace

TEST(Kinematics, FixtureUsesTheMountUnderTest)
{
  const YAML::Node root = YAML::LoadFile(std::string(MAZE_ARM_CONTROL_TEST_DATA) + "/fk_fixture.yaml");
  const auto mount = test_support::sim_mount();
  for (int i = 0; i < 3; ++i) {
    EXPECT_DOUBLE_EQ(root["grasp_xyz"][i].as<double>(), mount.grasp_xyz[i]);
    EXPECT_DOUBLE_EQ(root["grasp_rpy"][i].as<double>(), mount.grasp_rpy[i]);
    EXPECT_DOUBLE_EQ(root["handle_center"][i].as<double>(), mount.handle_center[i]);
  }
}

TEST(Kinematics, FkMatchesPython)
{
  const ArmKinematics kin = test_support::sim_kinematics();
  const auto poses = fixture_poses();
  ASSERT_GE(poses.size(), 20u);
  for (const auto & p : poses) {
    const Eigen::Matrix4d T = kin.fk_board(p.q).matrix();
    EXPECT_LT((T - p.T).cwiseAbs().maxCoeff(), 1e-9) << "q = " << p.q.transpose();
  }
}

TEST(Kinematics, BaseLinkIsRotatedPiAboutZ)
{
  // At q = 0 the UR arm stretches along DH base -x (a2, a3 < 0); base_link's Rz(pi) turns that into +x.
  const ArmKinematics kin = test_support::sim_kinematics();
  const Eigen::Vector3d p = kin.fk_tool0(Joints::Zero()).translation();
  EXPECT_NEAR(p.x(), 0.425 + 0.3922, 1e-9);
  EXPECT_NEAR(p.y(), 0.1333 + 0.0996, 1e-9);  // d4 + d6 (tool0 z along DH base -y, turned to +y)
}

TEST(Kinematics, JacobianMatchesFiniteDifferences)
{
  const ArmKinematics kin = test_support::sim_kinematics();
  const double h = 1e-6;
  for (const auto & p : fixture_poses()) {
    const auto J = kin.jacobian_board(p.q);
    for (int i = 0; i < 6; ++i) {
      Joints qp = p.q, qm = p.q;
      qp[i] += h;
      qm[i] -= h;
      const auto Tp = kin.fk_board(qp), Tm = kin.fk_board(qm);
      const Eigen::Vector3d v = (Tp.translation() - Tm.translation()) / (2 * h);
      const Eigen::Vector3d w = log_so3(Tp.linear() * Tm.linear().transpose()) / (2 * h);
      EXPECT_LT((J.block<3, 1>(0, i) - v).norm(), 1e-6) << "joint " << i << " q = " << p.q.transpose();
      EXPECT_LT((J.block<3, 1>(3, i) - w).norm(), 1e-6) << "joint " << i << " q = " << p.q.transpose();
    }
  }
}
