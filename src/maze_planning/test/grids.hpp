// Small synthetic grids shared by the tests: 5 mm cells, origin at (0, 0).
#pragma once

#include "maze_planning/grid.hpp"

namespace maze_planning
{

inline Grid free_grid(int width, int height)
{
  Grid g;
  g.width = width;
  g.height = height;
  g.resolution = 0.005;
  g.data.assign(static_cast<size_t>(width * height), 0);
  return g;
}

inline void wall_row(Grid & g, int row)
{
  for (int c = 0; c < g.width; ++c) {
    g.data[static_cast<size_t>(row * g.width + c)] = 100;
  }
}

inline void wall_col(Grid & g, int col, int row_from, int row_to)
{
  for (int r = row_from; r <= row_to; ++r) {
    g.data[static_cast<size_t>(r * g.width + col)] = 100;
  }
}

inline void wall_block(Grid & g, int row_from, int row_to, int col_from, int col_to)
{
  for (int c = col_from; c <= col_to; ++c) {
    wall_col(g, c, row_from, row_to);
  }
}

inline void wall_border(Grid & g)
{
  wall_row(g, 0);
  wall_row(g, g.height - 1);
  wall_col(g, 0, 0, g.height - 1);
  wall_col(g, g.width - 1, 0, g.height - 1);
}

// 40 x 11, walls along rows 0 and 10: rows 4-6 are open to a 25 mm ball
inline Grid channel()
{
  Grid g = free_grid(40, 11);
  wall_row(g, 0);
  wall_row(g, 10);
  return g;
}

// 34 x 30, walled border, wall cells in columns 7 and 14 rows 0-16: columns 8-13 form a 30 mm dead
// end open at the top, too narrow for the 25 mm ball (docs/figures/planning-pl4/bfs_lead_in.png)
inline Grid dead_end()
{
  Grid g = free_grid(34, 30);
  wall_border(g);
  wall_col(g, 7, 0, 16);
  wall_col(g, 14, 0, 16);
  return g;
}

inline Circle goal_at(const Grid & g, int row, int col)
{
  const auto c = cell_center(g, row, col);
  return {c[0], c[1], 0.01};
}

}  // namespace maze_planning
