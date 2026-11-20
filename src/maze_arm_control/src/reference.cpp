#include "maze_arm_control/reference.hpp"

#include <cmath>

namespace maze_arm_control
{

ReferenceShaper::ReferenceShaper(const ReferenceParams & params)
: p_(params) {}

void ReferenceShaper::reset(const Eigen::Vector2d & measured)
{
  command_ = filtered_ = value_ = measured;
  rate_.setZero();
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
    rate_.setZero();
    return;
  }
  const double alpha = p_.smoothing_tau > 0.0 ? 1.0 - std::exp(-dt / p_.smoothing_tau) : 1.0;
  filtered_ += alpha * (command_ - filtered_);
  const double max_step = p_.max_rate * dt;
  const Eigen::Vector2d step = (filtered_ - value_).cwiseMax(-max_step).cwiseMin(max_step);
  value_ += step;
  rate_ = step / dt;
}

}  // namespace maze_arm_control
