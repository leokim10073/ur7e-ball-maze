// Rotation helpers and the Contract 4 tilt convention.
//   tilt_x > 0 lowers the board's +x edge: a rotation of +tilt_x about board +y.
//   tilt_y > 0 lowers the board's +y edge: a rotation of -tilt_y about board +x.
// Composed in the board frame as R_tilt = Ry(tilt_x) * Rx(-tilt_y); the desired board orientation is
// R_home * R_tilt. tilt_of() reads it back the same way, from the rotation vector of R_home^T * R.
#pragma once

#include <Eigen/Dense>

namespace maze_arm_control
{

// Rotation vector (axis * angle, angle in [0, pi]) of a rotation matrix.
Eigen::Vector3d log_so3(const Eigen::Matrix3d & R);

Eigen::Matrix3d tilt_rotation(double tilt_x, double tilt_y);

// (tilt_x, tilt_y) of orientation R relative to the level home orientation R_home (both base frame).
Eigen::Vector2d tilt_of(const Eigen::Matrix3d & R_home, const Eigen::Matrix3d & R);

// Rotation about the board's own z relative to home, rad (should stay ~0).
double yaw_of(const Eigen::Matrix3d & R_home, const Eigen::Matrix3d & R);

}  // namespace maze_arm_control
