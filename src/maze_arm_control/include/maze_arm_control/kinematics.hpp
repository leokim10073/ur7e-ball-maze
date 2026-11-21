// UR forward kinematics and the board Jacobian, from the robot's kinematics calibration file
// (ur_description/config/<ur_type>/default_kinematics.yaml, or the robot's own calibrated one).
// The C++ port of maze_arm/kinematics.py, which stays as the reference for tests and tools.
#pragma once

#include <array>
#include <string>
#include <vector>

#include <Eigen/Geometry>

namespace maze_arm_control
{

using Vector6d = Eigen::Matrix<double, 6, 1>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;

inline const std::array<std::string, 6> UR_JOINTS = { "shoulder_pan_joint", "shoulder_lift_joint",
                                                      "elbow_joint",        "wrist_1_joint",
                                                      "wrist_2_joint",      "wrist_3_joint" };

// R = Rz(yaw) Ry(pitch) Rx(roll), as in URDF.
Eigen::Matrix3d rpy_matrix(double roll, double pitch, double yaw);
Eigen::Isometry3d transform(const Eigen::Vector3d& xyz, const Eigen::Matrix3d& R = Eigen::Matrix3d::Identity());

// The six joint origins (parent link -> joint frame before it turns), in UR_JOINTS order: the
// shoulder, upper_arm, forearm, wrist_1, wrist_2, wrist_3 entries of the calibration file.
struct UrKinematics
{
  std::array<Eigen::Isometry3d, 6> joint_origin;
};

UrKinematics load_ur_kinematics(const std::string& path);  // throws std::runtime_error

// base_link -> the frame each joint turns about (its z axis), for joints 1..6, then tool0.
std::array<Eigen::Isometry3d, 7> joint_frames(const UrKinematics& kin, const Vector6d& q);
Eigen::Isometry3d fk_tool0(const UrKinematics& kin, const Vector6d& q);

// tool0 -> maze_board, from the handle pose in tool0 and the handle center in the board frame.
Eigen::Isometry3d board_offset(const Eigen::Vector3d& grasp_xyz, const Eigen::Vector3d& grasp_rpy,
                               const Eigen::Vector3d& handle_center);

// 6x6 Jacobian of the board center in base_link: [v; w] = J qdot.
Matrix6d board_jacobian(const UrKinematics& kin, const Vector6d& q, const Eigen::Isometry3d& T_tool0_board);

// Joint angles in UR_JOINTS order from a JointState's names and positions (any order).
// Throws std::runtime_error naming the missing joints.
Vector6d joints_from_state(const std::vector<std::string>& names, const std::vector<double>& positions);

}  // namespace maze_arm_control
