// The tilt reference the arm follows, shaped from the ~60 Hz /board/tilt_cmd:
// clamp to max_tilt -> first-order low-pass (smoothing_tau) -> second-order follower with a rate limit
// (max_rate) and an acceleration limit (max_accel) that brakes in time to stop on its target.
// The limits come last so the hard limits hold on the reference itself; the low-pass and the
// acceleration limit make the rate (used as feed-forward) continuous instead of a spike at every 60 Hz
// command, and without an abrupt stop at the end of a ramp, which a lagging arm would overshoot.
#pragma once

#include <Eigen/Dense>

namespace maze_arm_control
{

struct ReferenceParams
{
  double max_tilt;       // per axis, rad
  double max_rate;       // per axis, rad/s
  double max_accel;      // per axis, rad/s^2
  double smoothing_tau;  // s
};

class ReferenceShaper
{
public:
  explicit ReferenceShaper(const ReferenceParams & params);

  // Start (or restart) from the measured tilt, at rest. The command is set to it too.
  void reset(const Eigen::Vector2d & measured);
  // A new command; non-finite values are ignored (returns false), others are clamped.
  bool set_command(const Eigen::Vector2d & tilt);
  void step(double dt);

  Eigen::Vector2d value() const {return value_;}
  Eigen::Vector2d rate() const {return rate_;}
  Eigen::Vector2d command() const {return command_;}

private:
  ReferenceParams p_;
  Eigen::Vector2d command_ = Eigen::Vector2d::Zero();
  Eigen::Vector2d filtered_ = Eigen::Vector2d::Zero();
  Eigen::Vector2d filtered_rate_ = Eigen::Vector2d::Zero();
  Eigen::Vector2d value_ = Eigen::Vector2d::Zero();
  Eigen::Vector2d rate_ = Eigen::Vector2d::Zero();
};

}  // namespace maze_arm_control
