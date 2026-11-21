#include "maze_arm_control/tilt_controller.hpp"

#include <algorithm>
#include <cmath>

namespace maze_arm_control
{

Eigen::Matrix3d tilt_rotation(double tilt_x, double tilt_y)
{
  return (Eigen::AngleAxisd(tilt_x, Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(-tilt_y, Eigen::Vector3d::UnitX()))
      .toRotationMatrix();
}

Eigen::Vector2d tilt_from_rotation(const Eigen::Matrix3d& R_rel)
{
  const Eigen::Vector3d n = R_rel.col(2);
  return { std::atan2(n[0], n[2]), std::asin(std::clamp(n[1], -1.0, 1.0)) };
}

Eigen::Vector3d rotation_log(const Eigen::Matrix3d& R)
{
  const Eigen::Vector3d w = 0.5 * Eigen::Vector3d(R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1));
  const double s = w.norm();
  const double c = (R.trace() - 1.0) / 2.0;
  if (s < 1e-12)
  {
    return w;
  }
  return w / s * std::atan2(s, c);
}

Eigen::Isometry3d level_home(const Eigen::Isometry3d& T_board)
{
  Eigen::Vector3d x = T_board.linear().col(0);
  x[2] = 0.0;
  x.normalize();
  const Eigen::Vector3d z = Eigen::Vector3d::UnitZ();
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.linear().col(0) = x;
  T.linear().col(1) = z.cross(x);
  T.linear().col(2) = z;
  T.translation() = T_board.translation();
  return T;
}

TiltController::TiltController(const ArmConfig& cfg, const UrKinematics& kin, const Eigen::Isometry3d& T_tool0_board,
                               const Eigen::Isometry3d& T_home)
  : cfg_(cfg), kin_(kin), T_tool0_board_(T_tool0_board), T_home_(T_home)
{
}

Eigen::Isometry3d TiltController::board_pose(const Vector6d& q) const
{
  return fk_tool0(kin_, q) * T_tool0_board_;
}

Eigen::Vector2d TiltController::measured_tilt(const Vector6d& q) const
{
  return tilt_from_rotation(T_home_.linear().transpose() * board_pose(q).linear());
}

void TiltController::reset(const Vector6d& q)
{
  ref_ = measured_tilt(q);
}

Vector6d TiltController::step(const Vector6d& q, const Eigen::Vector2d& tilt_cmd, double dt)
{
  // Move the reference toward the clamped command, no faster than max_tilt_rate.
  const Eigen::Vector2d target = tilt_cmd.cwiseMax(-cfg_.max_tilt).cwiseMin(cfg_.max_tilt);
  Eigen::Vector2d delta = target - ref_;
  const double max_step = cfg_.max_tilt_rate * dt;
  if (delta.norm() > max_step)
  {
    delta *= max_step / delta.norm();
  }
  const Eigen::Vector2d old = ref_;
  ref_ += delta;

  const Eigen::Matrix3d R_home = T_home_.linear();
  const Eigen::Matrix3d R_old = R_home * tilt_rotation(old[0], old[1]);
  const Eigen::Matrix3d R_ref = R_home * tilt_rotation(ref_[0], ref_[1]);
  const Eigen::Vector3d w_ff = rotation_log(R_ref * R_old.transpose()) / dt;  // reference angular velocity

  const Eigen::Isometry3d T = board_pose(q);
  const Eigen::Vector3d e_pos = T_home_.translation() - T.translation();
  const Eigen::Vector3d e_rot = rotation_log(R_ref * T.linear().transpose());
  Vector6d twist;
  twist << cfg_.k_pos * e_pos, w_ff + cfg_.k_rot * e_rot;

  const Matrix6d J = board_jacobian(kin_, q, T_tool0_board_);
  const Matrix6d A = J * J.transpose() + cfg_.damping * cfg_.damping * Matrix6d::Identity();
  Vector6d qdot = J.transpose() * A.ldlt().solve(twist);

  const double peak = qdot.cwiseAbs().maxCoeff() / cfg_.max_joint_speed;
  if (peak > 1.0)
  {
    qdot /= peak;  // scale all joints together so the board still moves the right way
  }
  return qdot;
}

}  // namespace maze_arm_control
