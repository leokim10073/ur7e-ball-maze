#include "maze_arm_control/watchdog.hpp"

#include <cstdio>

namespace maze_arm_control
{

Watchdog::Watchdog(double command_timeout, double state_timeout)
: command_timeout_(command_timeout), state_timeout_(state_timeout) {}

std::string Watchdog::reason(double t) const
{
  char buf[96];
  if (!last_command_) {
    return "no tilt_cmd yet";
  }
  if (t - *last_command_ > command_timeout_) {
    std::snprintf(buf, sizeof(buf), "no tilt_cmd for %.3f s", t - *last_command_);
    return buf;
  }
  if (!last_state_) {
    return "no joint states yet";
  }
  if (t - *last_state_ > state_timeout_) {
    std::snprintf(buf, sizeof(buf), "joint states %.3f s old", t - *last_state_);
    return buf;
  }
  return {};
}

}  // namespace maze_arm_control
