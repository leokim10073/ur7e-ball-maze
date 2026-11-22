// UR7e forward kinematics to the maze board center and its geometric Jacobian.
// Same model as maze_arm.kinematics (Python, the reference): nominal DH, base_link = DH base rotated
// by pi about z, tool0 -> handle (grasp offset) -> board center (minus the handle center).
#pragma once

#include <array>

#include <Eigen/Geometry>

namespace maze_arm_control
{

using Joints = Eigen::Matrix<double, 6, 1>;
using Jacobian = Eigen::Matrix<double, 6, 6>;

struct DhParams
{
  std::array<double, 6> d, a, alpha;
};

// The board relative to tool0, as maze_arm.kinematics.fk_board takes it.
struct BoardMount
{
  Eigen::Vector3d grasp_xyz;      // handle frame in tool0, m
  Eigen::Vector3d grasp_rpy;      // ROS fixed-axis roll, pitch, yaw, rad
  Eigen::Vector3d handle_center;  // handle center in the board frame, m (board axes = handle axes)
};

// ROS fixed-axis roll-pitch-yaw: Rz(y) * Ry(p) * Rx(r).
Eigen::Matrix3d rpy_matrix(double r, double p, double y);

class ArmKinematics
{
public:
  ArmKinematics(const DhParams & dh, const BoardMount & mount);

  Eigen::Isometry3d fk_tool0(const Joints & q) const;  // base_link -> tool0
  Eigen::Isometry3d fk_board(const Joints & q) const;  // base_link -> maze_board

  // Rows 0-2: base_link linear velocity of the board center; rows 3-5: base_link angular velocity.
  Jacobian jacobian_board(const Joints & q) const;

private:
  Eigen::Isometry3d dh(int i, double theta) const;

  DhParams dh_;
  Eigen::Isometry3d tool0_to_board_;
};

}  // namespace maze_arm_control
