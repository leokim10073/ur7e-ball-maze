// Remote-pivot tilt control: tilt command -> UR7e joint velocities (no ROS).
//
// The board tilts about its own center: each step the controller asks for a board twist that
// holds the center where it was at home and turns the board toward the commanded tilt, then
// maps that twist to joint velocities with damped least squares (resolved-rate control).
//
// Tilt follows Contract 4 (docs/interfaces.md), relative to the level home pose:
//   tilt_x > 0 lowers the board's +x edge (rotation of +tilt_x about board +y),
//   tilt_y > 0 lowers the board's +y edge (rotation of -tilt_y about board +x).
//
// The C++ port of maze_arm/tilt_control.py.
#pragma once

#include <string>

#include <Eigen/Geometry>

#include "maze_arm_control/kinematics.hpp"

namespace maze_arm_control
{

struct ArmConfig
{
  double max_tilt;          // rad, hard limit on each tilt axis
  double max_tilt_rate;     // rad/s, limit on how fast the tilt reference moves
  double max_joint_speed;   // rad/s, every joint
  double k_pos;             // 1/s, board center position gain
  double k_rot;             // 1/s, board orientation gain
  double damping;           // damped least squares lambda
  double command_timeout;   // s, stop the arm if no tilt command for this long
};

// config/arm.yaml: every key required and > 0. Throws std::runtime_error naming the key.
ArmConfig load_arm_config(const std::string& path);

// Board orientation relative to home (in home board axes) for a Contract 4 tilt.
Eigen::Matrix3d tilt_rotation(double tilt_x, double tilt_y);
// Inverse of tilt_rotation, from the board normal (yaw about the normal is ignored).
Eigen::Vector2d tilt_from_rotation(const Eigen::Matrix3d& R_rel);
// Axis * angle of a rotation matrix.
Eigen::Vector3d rotation_log(const Eigen::Matrix3d& R);
// The board pose to hold: same center, level, same heading (board x projected flat).
Eigen::Isometry3d level_home(const Eigen::Isometry3d& T_board);

class TiltController
{
public:
  // T_tool0_board: board_offset(...). T_home: level_home(board pose at start).
  TiltController(const ArmConfig& cfg, const UrKinematics& kin, const Eigen::Isometry3d& T_tool0_board,
                 const Eigen::Isometry3d& T_home);

  Eigen::Isometry3d board_pose(const Vector6d& q) const;
  Eigen::Vector2d measured_tilt(const Vector6d& q) const;
  // Start the reference at the board's present tilt, so the arm does not jump.
  void reset(const Vector6d& q);
  // Joint velocities (rad/s, UR joint order) for one control step of length dt.
  Vector6d step(const Vector6d& q, const Eigen::Vector2d& tilt_cmd, double dt);

  const Eigen::Isometry3d& home() const { return T_home_; }
  const Eigen::Vector2d& reference() const { return ref_; }

private:
  ArmConfig cfg_;
  UrKinematics kin_;
  Eigen::Isometry3d T_tool0_board_;
  Eigen::Isometry3d T_home_;
  Eigen::Vector2d ref_ = Eigen::Vector2d::Zero();  // moves toward the command at max_tilt_rate
};

}  // namespace maze_arm_control
