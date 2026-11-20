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
endif()

ament_package()
