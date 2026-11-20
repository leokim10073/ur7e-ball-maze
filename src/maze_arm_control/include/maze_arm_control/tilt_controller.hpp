// Resolved-rate control of the board about its own center (remote pivot), damped least squares.
//   e_p = p_home - p,  e_R = log(R_d R^T),  R_d = R_home * tilt_rotation(ref)        (base frame)
//   twist = [k_pos e_p ; k_rot e_R + R_home (-rate_y, rate_x, 0)], each part capped in norm
//   qdot = J^T (J J^T + lambda^2 I)^-1 twist, lambda^2 = (1 - (s/sigma_min)^2) lambda_max^2 below sigma_min
//   then the whole qdot is scaled to the joint speed limits, and the whole change from the previous
//   qdot is scaled to the joint acceleration limits.
#pragma once

#include <Eigen/Dense>

#include "maze_arm_control/kinematics.hpp"

namespace maze_arm_control
{

struct ControllerParams
{
  double k_pos, k_rot;                  // 1/s
  double sigma_min, lambda_max;         // DLS
  double max_linear_speed;              // m/s
  double max_angular_speed;             // rad/s
  Joints max_joint_speed;               // rad/s
  Joints max_joint_accel;               // rad/s^2
};

class TiltController
{
public:
  struct Diag
  {
    double pos_err = 0.0;   // m
    double rot_err = 0.0;   // rad
    double sigma_min = 0.0;
    bool twist_capped = false, vel_scaled = false, accel_limited = false;
  };

  TiltController(const ArmKinematics & kin, const ControllerParams & params, const Eigen::Isometry3d & home);

  // Joint velocities for the measured joints q, reference tilt and its rate; dt since the last step.
  Joints step(const Joints & q, const Eigen::Vector2d & tilt, const Eigen::Vector2d & tilt_rate, double dt);
  // Forget the previous command (the acceleration limit starts from rest).
  void reset() {last_qdot_.setZero();}
  // Tell the acceleration limit what was actually sent (zeros from the watchdog, say).
  void set_last_command(const Joints & qdot) {last_qdot_ = qdot;}

  const Eigen::Isometry3d & home() const {return home_;}
  const Diag & diag() const {return diag_;}

private:
  ArmKinematics kin_;
  ControllerParams p_;
  Eigen::Isometry3d home_;
  Joints last_qdot_ = Joints::Zero();
  Diag diag_;
};

struct HomeCheck
{
  bool ok;
  double pos_err;  // m
  double rot_err;  // rad
};

HomeCheck home_check(
  const Eigen::Isometry3d & home, const Eigen::Isometry3d & board, double max_pos, double max_rot);

// True if any joint moved more than threshold between two joint states (a reset or a teleport).
bool joint_jump(const Joints & q_prev, const Joints & q, double threshold);

}  // namespace maze_arm_control
