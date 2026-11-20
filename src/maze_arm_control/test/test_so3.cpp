// Rotation helpers and the tilt sign convention (Contract 4), checked on the board's edges via FK.
#include <gtest/gtest.h>

#include <cmath>

#include "maze_arm_control/so3.hpp"
#include "test_support.hpp"

using maze_arm_control::log_so3;
using maze_arm_control::tilt_of;
using maze_arm_control::tilt_rotation;
using maze_arm_control::yaw_of;

TEST(So3, LogOfAngleAxis)
{
  for (double angle : {0.0, 1e-10, 1e-4, 0.3, 2.0, M_PI - 1e-3}) {
    const Eigen::Vector3d axis = Eigen::Vector3d(0.3, -0.5, 0.8).normalized();
    const Eigen::Matrix3d R = Eigen::AngleAxisd(angle, axis).toRotationMatrix();
    EXPECT_LT((log_so3(R) - angle * axis).norm(), 1e-7) << "angle " << angle;
  }
}

TEST(So3, TiltRoundTrip)
{
  const Eigen::Matrix3d R_home = Eigen::AngleAxisd(0.7, Eigen::Vector3d(1, 2, 3).normalized()).toRotationMatrix();
  // one axis at a time it is exact
  EXPECT_NEAR(tilt_of(R_home, R_home * tilt_rotation(0.06, 0))[0], 0.06, 1e-12);
  EXPECT_NEAR(tilt_of(R_home, R_home * tilt_rotation(0, -0.06))[1], -0.06, 1e-12);
  for (double tx : {-0.087, -0.03, 0.0, 0.05, 0.087}) {
    for (double ty : {-0.087, 0.0, 0.04, 0.087}) {
      const Eigen::Vector2d t = tilt_of(R_home, R_home * tilt_rotation(tx, ty));
      EXPECT_NEAR(t[0], tx, 1e-3) << tx << ", " << ty;  // second order in the angles
      EXPECT_NEAR(t[1], ty, 1e-3) << tx << ", " << ty;
      EXPECT_NEAR(yaw_of(R_home, R_home * tilt_rotation(tx, ty)), 0.0, 5e-3);
    }
  }
}

TEST(So3, EdgeSignsFromFk)
{
  // Contract 4: tilt_x > 0 lowers the board's +x edge, tilt_y > 0 lowers its +y edge.
  const auto kin = test_support::sim_kinematics();
  const Eigen::Isometry3d home = kin.fk_board(test_support::home_joints());
  const Eigen::Vector3d ex(0.15, 0, 0), ey(0, 0.15, 0);
  const auto edge_z = [&](double tx, double ty, const Eigen::Vector3d & edge) {
      return (home.linear() * tilt_rotation(tx, ty) * edge).z() - (home.linear() * edge).z();
    };
  EXPECT_LT(edge_z(+0.05, 0, ex), -0.005);
  EXPECT_GT(edge_z(-0.05, 0, ex), +0.005);
  EXPECT_LT(edge_z(0, +0.05, ey), -0.005);
  EXPECT_GT(edge_z(0, -0.05, ey), +0.005);
  // and the other axis' edges stay level
  EXPECT_NEAR(edge_z(+0.05, 0, ey), 0.0, 1e-4);
  EXPECT_NEAR(edge_z(0, +0.05, ex), 0.0, 1e-4);
}
