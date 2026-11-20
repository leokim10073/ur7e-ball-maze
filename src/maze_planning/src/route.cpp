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
  uint32_t corridor;  // index into Route::corridors
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

  // the lead-in of a pushed ball: its leading blocked cells
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

  // the lead-in (leading blocked cells and the first allowed one) in boxes of passable cells, each at
  // whole-cell extent: a pushed ball jitters sideways. Consecutive boxes share a cell, so they overlap.
  std::vector<Piece> pieces;
  if (lead > 0) {
    const auto passable = detail::passable_cells(grid, holes);
    const std::vector<detail::Cell> cells(path.cells.begin(), path.cells.begin() + static_cast<long>(lead + 1));
    auto box_passable = [&](size_t i, size_t j) {
        int r0 = cells[i][0], r1 = r0, c0 = cells[i][1], c1 = c0;
        for (size_t k = i; k <= j; ++k) {
          r0 = std::min(r0, cells[k][0]);
          r1 = std::max(r1, cells[k][0]);
          c0 = std::min(c0, cells[k][1]);
          c1 = std::max(c1, cells[k][1]);
        }
        for (int r = r0; r <= r1; ++r) {
          for (int c = c0; c <= c1; ++c) {
            if (!passable[detail::index(grid, {r, c})]) {
              return false;
            }
          }
        }
        return true;
      };
    const double half = 0.5 * grid.resolution;
    for (size_t i = 0; i + 1 < cells.size(); ) {
      size_t j = i + 1;  // one step without corner cutting through passable cells: always a passable box
      while (j + 1 < cells.size() && box_passable(i, j + 1)) {
        ++j;
      }
      Corridor k = cell_box(grid, {cells.begin() + static_cast<long>(i), cells.begin() + static_cast<long>(j + 1)});
      k = {k.x_min - half, k.x_max + half, k.y_min - half, k.y_max + half};
      if (i == 0) {
        include(k, start);
      }
      const auto index = static_cast<uint32_t>(route.corridors.size());
      route.corridors.push_back(k);
      for (size_t s = i; s < j; ++s) {
        pieces.push_back({s == 0 ? start : center(cells[s][0], cells[s][1]),
            center(cells[s + 1][0], cells[s + 1][1]), index});
      }
      i = j;
    }
    route.lead_in_corridors = static_cast<uint32_t>(route.corridors.size());
  }
  const std::vector<std::array<int, 2>> rest(path.cells.begin() + static_cast<long>(lead), path.cells.end());
  const auto vertices = shortcut_cells(grid, clearance, blocked, rest, params);
  auto segments = split_segments(grid, blocked, vertices);
  if (segments.empty()) {  // start and goal in one cell
    segments.push_back({rest[0][0], rest[0][1], rest[0][0], rest[0][1]});
  }
  for (size_t s = 0; s < segments.size(); ++s) {
    const auto & q = segments[s];
    Piece piece{center(q[0], q[1]), center(q[2], q[3]), static_cast<uint32_t>(route.corridors.size())};
    Corridor k = grow_corridor(grid, blocked, q[0], q[1], q[2], q[3]);
    if (s == 0 && lead == 0) {
      piece.from = start;
    }
    if (s + 1 == segments.size()) {
      piece.to = goal;
    }
    include(k, piece.from);
    include(k, piece.to);
    route.corridors.push_back(k);
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
    const double length = std::hypot(piece.to[0] - piece.from[0], piece.to[1] - piece.from[1]);
    const int steps = std::max(1, static_cast<int>(std::ceil(length / params.point_spacing - 1e-9)));
    if (k == 0) {
      add(piece.from, piece.corridor);
    }
    for (int j = 1; j <= steps; ++j) {
      const double t = static_cast<double>(j) / steps;
      add(j == steps ? piece.to :
        Point{piece.from[0] + t * (piece.to[0] - piece.from[0]), piece.from[1] + t * (piece.to[1] - piece.from[1])},
        piece.corridor);
    }
  }
  return route;
}

}  // namespace maze_planning
