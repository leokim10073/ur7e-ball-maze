#include "maze_arm_control/tilt_controller.hpp"

#include <algorithm>

#include "maze_arm_control/so3.hpp"

namespace maze_arm_control
{

TiltController::TiltController(
  const ArmKinematics & kin, const ControllerParams & params, const Eigen::Isometry3d & home)
: kin_(kin), p_(params), home_(home) {}

namespace
{
// Scale v so its norm is at most cap; true if it was scaled.
bool cap_norm(Eigen::Ref<Eigen::Vector3d> v, double cap)
{
  const double n = v.norm();
  if (n > cap) {
    v *= cap / n;
    return true;
  }
  return false;
}
}  // namespace

Joints TiltController::step(
  const Joints & q, const Eigen::Vector2d & tilt, const Eigen::Vector2d & tilt_rate, double dt)
{
  diag_ = Diag{};
  const Eigen::Isometry3d T = kin_.fk_board(q);
  const Eigen::Matrix3d R_d = home_.linear() * tilt_rotation(tilt.x(), tilt.y());
  const Eigen::Vector3d e_p = home_.translation() - T.translation();
  const Eigen::Vector3d e_R = log_so3(R_d * T.linear().transpose());
  diag_.pos_err = e_p.norm();
  diag_.rot_err = e_R.norm();

  // Reference angular velocity in base_link: the rates are about board +y (tilt_x) and -x (tilt_y).
  const Eigen::Vector3d w_ff = R_d * Eigen::Vector3d(-tilt_rate.y(), tilt_rate.x(), 0.0);
  Eigen::Matrix<double, 6, 1> twist;
  twist.head<3>() = p_.k_pos * e_p;
  twist.tail<3>() = p_.k_rot * e_R + w_ff;
  diag_.twist_capped = cap_norm(twist.head<3>(), p_.max_linear_speed);
  diag_.twist_capped |= cap_norm(twist.tail<3>(), p_.max_angular_speed);
  // the tilt-rate limit holds on the board itself, not only on the reference: a large tilt error
  // (after a reset, say) is corrected no faster than max_tilt_rate about either board axis
  const Eigen::Vector3d w_board = R_d.transpose() * twist.tail<3>();
  const double over = w_board.head<2>().cwiseAbs().maxCoeff() / p_.max_tilt_rate;
  if (over > 1.0) {
    twist.tail<3>() /= over;
    diag_.twist_capped = true;
  }

  const Jacobian J = kin_.jacobian_board(q);
  const Eigen::JacobiSVD<Jacobian> svd(J);
  const double s = svd.singularValues().minCoeff();
  diag_.sigma_min = s;
  double lambda2 = 0.0;
  if (s < p_.sigma_min) {
    const double r = s / p_.sigma_min;
    lambda2 = (1.0 - r * r) * p_.lambda_max * p_.lambda_max;
  }
  const Jacobian JJt = J * J.transpose() + lambda2 * Jacobian::Identity();
  Joints qdot = J.transpose() * JJt.ldlt().solve(twist);

  const double speed = (qdot.cwiseAbs().array() / p_.max_joint_speed.array()).maxCoeff();
  if (speed > 1.0) {
    qdot /= speed;
    diag_.vel_scaled = true;
  }
  if (dt > 0.0) {
    const Joints delta = qdot - last_qdot_;
    const double accel = (delta.cwiseAbs().array() / (p_.max_joint_accel.array() * dt)).maxCoeff();
    if (accel > 1.0) {
      qdot = last_qdot_ + delta / accel;
      diag_.accel_limited = true;
    }
  }
  last_qdot_ = qdot;
  return qdot;
}

HomeCheck home_check(
  const Eigen::Isometry3d & home, const Eigen::Isometry3d & board, double max_pos, double max_rot)
{
  HomeCheck c;
  c.pos_err = (board.translation() - home.translation()).norm();
  c.rot_err = log_so3(home.linear().transpose() * board.linear()).norm();
  c.ok = c.pos_err <= max_pos && c.rot_err <= max_rot;
  return c;
}

bool joint_jump(const Joints & q_prev, const Joints & q, double threshold)
{
  return (q - q_prev).cwiseAbs().maxCoeff() > threshold;
}

}  // namespace maze_arm_control
