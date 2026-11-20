cmake_minimum_required(VERSION 3.8)
project(maze_planning)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)

# the planning algorithm, free of ROS so it can be tested on its own; the node (PL4) wraps it
add_library(maze_planning_core STATIC
  src/grid.cpp
  src/clearance.cpp
  src/blocked.cpp
  src/grid_planner.cpp
  src/shortcut.cpp
  src/corridors.cpp
)
target_compile_features(maze_planning_core PUBLIC cxx_std_17)
set_target_properties(maze_planning_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(maze_planning_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

install(DIRECTORY include/ DESTINATION include)
install(TARGETS maze_planning_core EXPORT export_maze_planning
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib)
ament_export_targets(export_maze_planning HAS_LIBRARY_TARGET)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_clearance test/test_clearance.cpp)
  target_link_libraries(test_clearance maze_planning_core)
  ament_add_gtest(test_blocked test/test_blocked.cpp)
  target_link_libraries(test_blocked maze_planning_core)
  ament_add_gtest(test_grid_planner test/test_grid_planner.cpp)
  target_link_libraries(test_grid_planner maze_planning_core)

  # the L1 maze, exported from the CAD by scripts/export_fixture.py
  find_package(yaml_cpp_vendor REQUIRED)
  find_package(yaml-cpp REQUIRED)
  ament_add_gtest(test_l1 test/test_l1.cpp)
  target_link_libraries(test_l1 maze_planning_core yaml-cpp)
  target_compile_definitions(test_l1 PRIVATE MAZE_PLANNING_TEST_DATA="${CMAKE_CURRENT_SOURCE_DIR}/test/data")

  # tests that also use the library's internal helpers (src/geometry.hpp) and the L1 fixture
  foreach(t test_shortcut test_corridors)
    ament_add_gtest(${t} test/${t}.cpp)
    target_link_libraries(${t} maze_planning_core yaml-cpp)
    target_include_directories(${t} PRIVATE src)
    target_compile_definitions(${t} PRIVATE MAZE_PLANNING_TEST_DATA="${CMAKE_CURRENT_SOURCE_DIR}/test/data")
  endforeach()
endif()

ament_package()
