// The real L1 maze, exported from the CAD by scripts/export_fixture.py into test/data/L1.yaml.
#pragma once

#include <yaml-cpp/yaml.h>

#include <array>
#include <string>
#include <vector>

#include "maze_planning/grid.hpp"

namespace maze_planning
{

struct Fixture
{
  Grid grid;
  std::vector<Circle> holes;
  std::array<double, 2> start{};
  Circle goal;
  std::vector<std::array<double, 2>> route;  // the designed route's cell centers
};

inline Fixture load_fixture(const std::string & level)
{
  const YAML::Node y = YAML::LoadFile(std::string(MAZE_PLANNING_TEST_DATA) + "/" + level + ".yaml");
  Fixture f;
  f.grid.width = y["width"].as<int>();
  f.grid.height = y["height"].as<int>();
  f.grid.resolution = y["resolution"].as<double>();
  f.grid.origin_x = y["origin"][0].as<double>();
  f.grid.origin_y = y["origin"][1].as<double>();
  for (const auto & v : y["data"]) {
    f.grid.data.push_back(static_cast<int8_t>(v.as<int>()));
  }
  for (const auto & h : y["holes"]) {
    f.holes.push_back({h[0].as<double>(), h[1].as<double>(), h[2].as<double>()});
  }
  f.start = {y["start"][0].as<double>(), y["start"][1].as<double>()};
  f.goal = {y["goal"][0].as<double>(), y["goal"][1].as<double>(), y["goal"][2].as<double>()};
  for (const auto & p : y["route"]) {
    f.route.push_back({p[0].as<double>(), p[1].as<double>()});
  }
  return f;
}

}  // namespace maze_planning
