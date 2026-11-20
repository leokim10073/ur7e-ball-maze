#include "maze_arm_control/reference.hpp"

#include <algorithm>
#include <cmath>

namespace maze_arm_control
{

ReferenceShaper::ReferenceShaper(const ReferenceParams & params)
: p_(params) {}

void ReferenceShaper::reset(const Eigen::Vector2d & measured)
{
  command_ = filtered_ = value_ = measured;
  rate_.setZero();
  filtered_rate_.setZero();
}

bool ReferenceShaper::set_command(const Eigen::Vector2d & tilt)
{
  if (!tilt.allFinite()) {
    return false;
  }
  command_ = tilt.cwiseMax(-p_.max_tilt).cwiseMin(p_.max_tilt);
  return true;
}

void ReferenceShaper::step(double dt)
{
  if (!(dt > 0.0)) {
    return;
  }
  const double alpha = p_.smoothing_tau > 0.0 ? 1.0 - std::exp(-dt / p_.smoothing_tau) : 1.0;
  const Eigen::Vector2d previous = filtered_;
  filtered_ += alpha * (command_ - filtered_);
  filtered_rate_ = ((filtered_ - previous) / dt).cwiseMax(-p_.max_rate).cwiseMin(p_.max_rate);

  const double a = p_.max_accel, dv = a * dt;
  // the fastest speed that still stops within distance d: this step at v, then braking at a
  // (v dt + v^2 / 2a <= d)
  const auto stop_speed = [a, dt](double d) {return a * (std::sqrt(dt * dt + 2.0 * d / a) - dt);};
  for (int i = 0; i < 2; ++i) {
    // follow the filtered command's rate, plus the fastest approach that can still brake to a stop on it
    const double e = filtered_[i] - value_[i];
    double v_des = filtered_rate_[i] + std::copysign(stop_speed(std::abs(e)), e);
    // never faster toward the command itself than allows stopping on it: the filtered command
    // brakes harder (tau) than max_accel can follow
    const double e_cmd = command_[i] - value_[i];
    const double v_stop = stop_speed(std::abs(e_cmd));
    v_des = e_cmd >= 0.0 ? std::min(v_des, v_stop) : std::max(v_des, -v_stop);
    double v = std::clamp(v_des, rate_[i] - dv, rate_[i] + dv);
    v = std::clamp(v, -p_.max_rate, p_.max_rate);
    // last step onto the target: land on it rather than step over (when the acceleration allows)
    if (std::abs(e) < std::abs(v) * dt && std::abs(e / dt - rate_[i]) <= dv) {
      v = e / dt;
    }
    double next = std::clamp(value_[i] + v * dt, -p_.max_tilt, p_.max_tilt);
    rate_[i] = (next - value_[i]) / dt;
    value_[i] = next;
  }
}

}  // namespace maze_arm_control
