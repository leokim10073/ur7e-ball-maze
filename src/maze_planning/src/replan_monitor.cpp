#include "maze_planning/replan_monitor.hpp"

#include <cmath>
#include <stdexcept>

namespace maze_planning
{

ReplanMonitor::ReplanMonitor(const ReplanParams & params)
: params_(params)
{
  if (!(std::isfinite(params.outside_margin) && params.outside_margin >= 0.0)) {
    throw std::invalid_argument("replan.outside_margin must be finite and >= 0");
  }
  if (params.outside_frames < 1) {
    throw std::invalid_argument("replan.outside_frames must be >= 1");
  }
  if (!(std::isfinite(params.no_path_retry_period) && params.no_path_retry_period > 0.0)) {
    throw std::invalid_argument("replan.no_path_retry_period must be finite and > 0");
  }
}

void ReplanMonitor::route_planned(const Route & route, double now)
{
  corridors_ = route.corridors;
  planned_ = true;
  no_path_ = route.points.empty();
  last_attempt_ = now;
  outside_count_ = 0;
}

bool ReplanMonitor::should_replan(double x, double y, double now)
{
  if (!planned_) {
    return false;
  }
  if (no_path_) {
    return now - last_attempt_ >= params_.no_path_retry_period;
  }
  // Known risk: only "outside every corridor" triggers a replan. A ball that rolls backwards into an
  // earlier corridor of the same route is still "inside" and does not replan; controllers own
  // progress. If this causes bugs, switch to checking only corridors near the ball's progress.
  const double m = params_.outside_margin;
  for (const auto & k : corridors_) {
    if (x >= k.x_min - m && x <= k.x_max + m && y >= k.y_min - m && y <= k.y_max + m) {
      outside_count_ = 0;
      return false;
    }
  }
  return ++outside_count_ >= params_.outside_frames;
}

}  // namespace maze_planning
