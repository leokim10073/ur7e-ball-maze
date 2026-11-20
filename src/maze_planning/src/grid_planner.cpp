#include "maze_planning/grid_planner.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <limits>
#include <optional>
#include <queue>
#include <utility>

#include "maze_planning/blocked.hpp"
#include "maze_planning/clearance.hpp"
#include "geometry.hpp"

namespace maze_planning
{

using detail::Cell;
using detail::cell_of;
using detail::line_cells;

GridPath plan_grid_path(
  const Grid & grid, const std::vector<Circle> & holes, std::array<double, 2> start,
  const Circle & goal, const PlannerParams & params)
{
  GridPath out;
  validate(params);
  const auto clearance = clearance_map(grid);  // validates the grid
  const auto blocked = blocked_cells(grid, clearance, holes, params);
  const int w = grid.width, h = grid.height;
  auto idx = [w](Cell c) {return static_cast<size_t>(c[0] * w + c[1]);};

  const auto start_cell = cell_of(grid, start[0], start[1]);
  const auto goal_cell = cell_of(grid, goal.x, goal.y);
  if (!start_cell || !goal_cell || is_obstacle(grid.data[idx(*start_cell)]) || blocked[idx(*goal_cell)]) {
    return out;
  }

  // a ball already too close to a wall first goes straight to the nearest allowed cell
  std::vector<Cell> lead_in;
  Cell from = *start_cell;
  if (blocked[idx(from)]) {
    std::optional<Cell> nearest;
    double best = std::numeric_limits<double>::infinity();
    for (int r = 0; r < h; ++r) {
      for (int c = 0; c < w; ++c) {
        const double d = std::hypot(r - from[0], c - from[1]);
        if (!blocked[idx({r, c})] && d < best) {
          best = d;
          nearest = Cell{r, c};
        }
      }
    }
    if (!nearest) {
      return out;
    }
    lead_in = line_cells(from, *nearest);
    for (const auto & c : lead_in) {
      if (is_obstacle(grid.data[idx(c)])) {
        return out;
      }
    }
    lead_in.pop_back();  // the A* below starts from it
    from = *nearest;
  }

  // A*: g in cells, a step's length scaled up when its cell is closer to a wall than preferred
  const Cell to = *goal_cell;
  auto step_cost = [&](Cell c, double length) {
      const double shortfall = std::max(0.0, params.preferred_clearance - clearance[idx(c)]);
      return length * (1.0 + params.clearance_weight * shortfall / params.preferred_clearance);
    };
  auto heuristic = [&](Cell c) {return std::hypot(c[0] - to[0], c[1] - to[1]);};
  std::vector<double> g(grid.data.size(), std::numeric_limits<double>::infinity());
  std::vector<int> parent(grid.data.size(), -1);
  using Entry = std::pair<double, int>;
  std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> open;
  g[idx(from)] = 0.0;
  open.push({heuristic(from), static_cast<int>(idx(from))});
  bool reached = false;
  while (!open.empty()) {
    const auto [f, i] = open.top();
    open.pop();
    const Cell u{i / w, i % w};
    if (u == to) {
      reached = true;
      break;
    }
    if (f > g[i] + heuristic(u)) {
      continue;  // stale entry
    }
    for (int dr = -1; dr <= 1; ++dr) {
      for (int dc = -1; dc <= 1; ++dc) {
        const Cell v{u[0] + dr, u[1] + dc};
        if ((dr == 0 && dc == 0) || v[0] < 0 || v[0] >= h || v[1] < 0 || v[1] >= w || blocked[idx(v)]) {
          continue;
        }
        if (dr != 0 && dc != 0 && (blocked[idx({u[0] + dr, u[1]})] || blocked[idx({u[0], u[1] + dc})])) {
          continue;  // no cutting corners
        }
        const double cand = g[i] + step_cost(v, (dr != 0 && dc != 0) ? std::sqrt(2.0) : 1.0);
        if (cand < g[idx(v)]) {
          g[idx(v)] = cand;
          parent[idx(v)] = i;
          open.push({cand + heuristic(v), static_cast<int>(idx(v))});
        }
      }
    }
  }
  if (!reached) {
    return out;
  }

  std::vector<Cell> route;
  for (int i = static_cast<int>(idx(to)); i != -1; i = parent[i]) {
    route.push_back({i / w, i % w});
  }
  std::reverse(route.begin(), route.end());
  out.cells = lead_in;
  out.cells.insert(out.cells.end(), route.begin(), route.end());
  for (const auto & c : out.cells) {
    out.points.push_back(cell_center(grid, c[0], c[1]));
  }
  // exact ends: the ball where it is, the goal pocket's center
  out.points.front() = start;
  if (out.cells.size() == 1) {
    out.points.push_back({goal.x, goal.y});
  } else {
    out.points.back() = {goal.x, goal.y};
  }
  out.found = true;
  return out;
}

}  // namespace maze_planning
