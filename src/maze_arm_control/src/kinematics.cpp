#include "maze_arm_control/kinematics.hpp"

#include <cmath>

namespace maze_arm_control
{

Eigen::Matrix3d rpy_matrix(double r, double p, double y)
{
  return (Eigen::AngleAxisd(y, Eigen::Vector3d::UnitZ()) * Eigen::AngleAxisd(p, Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(r, Eigen::Vector3d::UnitX())).toRotationMatrix();
}

ArmKinematics::ArmKinematics(const DhParams & dh, const BoardMount & mount)
: dh_(dh)
{
  Eigen::Isometry3d tool0_handle = Eigen::Isometry3d::Identity();
  tool0_handle.linear() = rpy_matrix(mount.grasp_rpy.x(), mount.grasp_rpy.y(), mount.grasp_rpy.z());
  tool0_handle.translation() = mount.grasp_xyz;
  Eigen::Isometry3d handle_board = Eigen::Isometry3d::Identity();
  handle_board.translation() = -mount.handle_center;  // board axes = handle axes
  tool0_to_board_ = tool0_handle * handle_board;
}

Eigen::Isometry3d ArmKinematics::dh(int i, double theta) const
{
  const double ct = std::cos(theta), st = std::sin(theta);
  const double ca = std::cos(dh_.alpha[i]), sa = std::sin(dh_.alpha[i]);
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.matrix().topRows<3>() << ct, -st * ca, st * sa, dh_.a[i] * ct,
    st, ct * ca, -ct * sa, dh_.a[i] * st,
    0.0, sa, ca, dh_.d[i];
  return T;
}

namespace
{
Eigen::Isometry3d base_link_to_dh_base()
{
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.linear() = Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  return T;
}
}  // namespace

Eigen::Isometry3d ArmKinematics::fk_tool0(const Joints & q) const
{
  Eigen::Isometry3d T = base_link_to_dh_base();
  for (int i = 0; i < 6; ++i) {
    T = T * dh(i, q[i]);
  }
  return T;
}

Eigen::Isometry3d ArmKinematics::fk_board(const Joints & q) const
{
  return fk_tool0(q) * tool0_to_board_;
}

Jacobian ArmKinematics::jacobian_board(const Joints & q) const
{
  // Joint i turns about z of frame i-1 (DH), through its origin; both in base_link.
  std::array<Eigen::Vector3d, 6> z, o;
  Eigen::Isometry3d T = base_link_to_dh_base();
  for (int i = 0; i < 6; ++i) {
    z[i] = T.linear().col(2);
    o[i] = T.translation();
    T = T * dh(i, q[i]);
  }
  const Eigen::Vector3d p = (T * tool0_to_board_).translation();
  Jacobian J;
  for (int i = 0; i < 6; ++i) {
    J.block<3, 1>(0, i) = z[i].cross(p - o[i]);
    J.block<3, 1>(3, i) = z[i];
  }
  return J;
}

}  // namespace maze_arm_control
