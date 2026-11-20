// Command and joint-state freshness. Times are steady (wall) clock seconds from the caller, so a
// stalled sim or a stopped publisher cannot hide behind a frozen ROS clock.
#pragma once

#include <optional>
#include <string>

namespace maze_arm_control
{

class Watchdog
{
public:
  Watchdog(double command_timeout, double state_timeout);

  void on_command(double t) {last_command_ = t;}
  void on_state(double t) {last_state_ = t;}
  void clear() {last_command_.reset(); last_state_.reset();}

  bool ok(double t) const {return reason(t).empty();}
  // Empty when ok, else why not.
  std::string reason(double t) const;

private:
  double command_timeout_, state_timeout_;
  std::optional<double> last_command_, last_state_;
};

}  // namespace maze_arm_control
