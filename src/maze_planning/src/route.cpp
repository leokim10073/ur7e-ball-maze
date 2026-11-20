#include <algorithm>
#include <cmath>

#include "geometry.hpp"
#include "maze_planning/blocked.hpp"
#include "maze_planning/clearance.hpp"
#include "maze_planning/route.hpp"

namespace maze_planning
{

namespace
{

using Point = std::array<double, 2>;

struct Piece
{
  Point from, to;
  Corridor corridor;
};

void include(Corridor & k, const Point & p)
{
  k.x_min = std::min(k.x_min, p[0]);
  k.x_max = std::max(k.x_max, p[0]);
  k.y_min = std::min(k.y_min, p[1]);
  k.y_max = std::max(k.y_max, p[1]);
}

Corridor cell_box(const Grid & g, const std::vector<detail::Cell> & cells)
{
  Corridor k;
  const auto first = cell_center(g, cells[0][0], cells[0][1]);
  k = {first[0], first[0], first[1], first[1]};
  for (const auto & c : cells) {
    include(k, cell_center(g, c[0], c[1]));
  }
  return k;
}

}  // namespace

Route build_route(
  const Grid & grid, const std::vector<Circle> & holes, const GridPath & path,
  const PlannerParams & params)
{
  validate(params);
  Route route;
  const size_t n = path.cells.size();
  // plan_grid_path gives one point per cell, plus the goal center when start and goal share a cell
  const bool shape_ok = n > 0 && (path.points.size() == n || (n == 1 && path.points.size() == 2));
  if (!path.found || !shape_ok) {
    return route;
  }
  const auto clearance = clearance_map(grid);  // validates the grid
  const auto blocked = blocked_cells(grid, clearance, holes, params);
  for (const auto & c : path.cells) {
    if (c[0] < 0 || c[0] >= grid.height || c[1] < 0 || c[1] >= grid.width) {
      return route;
    }
  }

  // the lead-in of a pushed ball: leading blocked cells, one straight segment to the first allowed
  size_t lead = 0;
  while (lead < n && blocked[detail::index(grid, path.cells[lead])]) {
    ++lead;
  }
  if (lead == n) {
    return route;  // the goal cell is allowed in any path plan_grid_path returns
  }
  const Point start = path.points.front();
  const Point goal = path.points.back();
  auto center = [&](int r, int c) {return cell_center(grid, r, c);};

  std::vector<Piece> pieces;
  if (lead > 0) {
    std::vector<detail::Cell> box(path.cells.begin(), path.cells.begin() + static_cast<long>(lead + 1));
    Corridor k = cell_box(grid, box);
    include(k, start);
    pieces.push_back({start, center(path.cells[lead][0], path.cells[lead][1]), k});
  }
  const std::vector<std::array<int, 2>> rest(path.cells.begin() + static_cast<long>(lead), path.cells.end());
  const auto vertices = shortcut_cells(grid, clearance, blocked, rest, params);
  auto segments = split_segments(grid, blocked, vertices);
  if (segments.empty()) {  // start and goal in one cell
    segments.push_back({rest[0][0], rest[0][1], rest[0][0], rest[0][1]});
  }
  for (size_t s = 0; s < segments.size(); ++s) {
    const auto & q = segments[s];
    Piece piece{center(q[0], q[1]), center(q[2], q[3]),
      grow_corridor(grid, blocked, q[0], q[1], q[2], q[3])};
    if (s == 0 && lead == 0) {
      piece.from = start;
    }
    if (s + 1 == segments.size()) {
      piece.to = goal;
    }
    include(piece.corridor, piece.from);
    include(piece.corridor, piece.to);
    pieces.push_back(piece);
  }

  // sample every point_spacing; a shared vertex belongs to the earlier piece
  auto add = [&](const Point & p, uint32_t k) {
      if (!route.points.empty()) {
        const double d = std::hypot(p[0] - route.points.back()[0], p[1] - route.points.back()[1]);
        if (d < 1e-12) {
          return;
        }
        route.arc_length.push_back(route.arc_length.back() + d);
      } else {
        route.arc_length.push_back(0.0);
      }
      route.points.push_back(p);
      route.pose_corridor.push_back(k);
    };
  for (size_t k = 0; k < pieces.size(); ++k) {
    const Piece & piece = pieces[k];
    route.corridors.push_back(piece.corridor);
    const double length = std::hypot(piece.to[0] - piece.from[0], piece.to[1] - piece.from[1]);
    const int steps = std::max(1, static_cast<int>(std::ceil(length / params.point_spacing - 1e-9)));
    if (k == 0) {
      add(piece.from, 0);
    }
    for (int j = 1; j <= steps; ++j) {
      const double t = static_cast<double>(j) / steps;
      add(j == steps ? piece.to :
        Point{piece.from[0] + t * (piece.to[0] - piece.from[0]), piece.from[1] + t * (piece.to[1] - piece.from[1])},
        static_cast<uint32_t>(k));
    }
  }
  return route;
}

}  // namespace maze_planning
