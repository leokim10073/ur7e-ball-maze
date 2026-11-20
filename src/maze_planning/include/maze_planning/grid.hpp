// Plain types for the planner, free of ROS. The grid follows Contract 1 of docs/interfaces.md
// (nav_msgs/OccupancyGrid conventions): 0 free, 100 wall, -1 unknown; data[row * width + col];
// row 0 at the bottom (-y); origin = lower-left corner of the grid in maze_board, m.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace maze_planning
{

struct Grid
{
  int width = 0;
  int height = 0;
  double resolution = 0.0;  // m per cell
  double origin_x = 0.0;    // lower-left corner, m
  double origin_y = 0.0;
  std::vector<int8_t> data;
};

struct Circle
{
  double x = 0.0;  // center in maze_board, m
  double y = 0.0;
  double radius = 0.0;
};

struct PlannerParams
{
  double ball_radius = 0.0125;         // 25 mm marble (#19)
  double wall_margin = 0.003;          // extra wall clearance beyond the ball radius
  double hole_margin = 0.003;          // keep the ball center this far outside a hole's rim
  double preferred_clearance = 0.020;  // closer than this to a wall costs extra
  double clearance_weight = 2.0;       // how much extra
};

struct GridPath
{
  bool found = false;
  std::vector<std::array<int, 2>> cells;      // {row, col}, start to goal
  std::vector<std::array<double, 2>> points;  // {x, y} in maze_board, m
};

// walls and unknown cells: anything that is not free floor
bool is_obstacle(int8_t value);

std::array<double, 2> cell_center(const Grid & grid, int row, int col);

// throws std::invalid_argument on a grid with no cells, a non-positive resolution or a data size
// that does not match width * height
void validate(const Grid & grid);

}  // namespace maze_planning
