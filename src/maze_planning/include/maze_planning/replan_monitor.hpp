// When the planner node replans (PL4), free of ROS: the ball has left the route, or the last plan
// found no path and it is time to retry.
#pragma once

#include "maze_planning/route.hpp"

namespace maze_planning
{

struct ReplanParams
{
  double outside_margin = 0.003;      // the ball must be this far outside every corridor, m ...
  int outside_frames = 3;             // ... for this many consecutive usable ball states
  double no_path_retry_period = 0.5;  // s between retries after STATUS_NO_PATH
};

class ReplanMonitor
{
public:
  // throws std::invalid_argument unless outside_margin is finite and >= 0, outside_frames >= 1 and
  // no_path_retry_period is finite and > 0
  explicit ReplanMonitor(const ReplanParams & params);

  // a plan was made at `now` (node clock, s); an empty route means STATUS_NO_PATH
  void route_planned(const Route & route, double now);

  // one call per usable ball state (tracking_ok); false until the first route_planned
  bool should_replan(double x, double y, double now);

private:
  ReplanParams params_;
  std::vector<Corridor> corridors_;
  bool planned_ = false;
  bool no_path_ = false;
  double last_attempt_ = 0.0;
  int outside_count_ = 0;
};

}  // namespace maze_planning
