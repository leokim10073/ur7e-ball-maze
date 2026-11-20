cmake_minimum_required(VERSION 3.8)
project(maze_arm_control)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)

# kinematics and the control law, free of ROS so they can be tested on their own
add_library(maze_arm_control_core STATIC
  src/kinematics.cpp
  src/so3.cpp
  src/reference.cpp
  src/tilt_controller.cpp
  src/watchdog.cpp
)
target_compile_features(maze_arm_control_core PUBLIC cxx_std_17)
set_target_properties(maze_arm_control_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(maze_arm_control_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIR})

install(DIRECTORY include/ DESTINATION include)
install(TARGETS maze_arm_control_core EXPORT export_maze_arm_control
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib)
ament_export_targets(export_maze_arm_control HAS_LIBRARY_TARGET)
ament_export_dependencies(eigen3_cmake_module Eigen3)

# the node: start-up, controller switch, watchdog timer and topics around the core
find_package(rclcpp REQUIRED)
find_package(maze_interfaces REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(controller_manager_msgs REQUIRED)
add_library(maze_arm_control_node STATIC src/tilt_control_node.cpp)
target_include_directories(maze_arm_control_node PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)
target_link_libraries(maze_arm_control_node maze_arm_control_core)
ament_target_dependencies(maze_arm_control_node rclcpp maze_interfaces sensor_msgs std_msgs controller_manager_msgs)

add_executable(tilt_control src/tilt_control_main.cpp)
target_link_libraries(tilt_control maze_arm_control_node)
install(TARGETS tilt_control DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})
install(PROGRAMS scripts/plot_tilt_run.py DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(yaml_cpp_vendor REQUIRED)
  find_package(yaml-cpp REQUIRED)
  set(defs
    MAZE_ARM_CONTROL_TEST_DATA="${CMAKE_CURRENT_SOURCE_DIR}/test/data"
    MAZE_ARM_CONTROL_CONFIG="${CMAKE_CURRENT_SOURCE_DIR}/config/tilt_control.yaml")
  foreach(t test_kinematics test_so3 test_reference test_controller_loop test_watchdog)
    ament_add_gtest(${t} test/${t}.cpp)
    target_link_libraries(${t} maze_arm_control_core yaml-cpp)
    target_compile_definitions(${t} PRIVATE ${defs})
  endforeach()

  # the node in-process; its own DDS domain so a running sim does not interfere
  ament_add_gtest(test_node test/test_node.cpp ENV ROS_DOMAIN_ID=78 ROS_LOCALHOST_ONLY=1 TIMEOUT 120)
  target_link_libraries(test_node maze_arm_control_node yaml-cpp)
  target_compile_definitions(test_node PRIVATE ${defs})

  # the config against the sim's own files
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_config test/test_config.py)

  # the node driving the MuJoCo sim (Contract 4 "How to check", tracking, watchdog, reset, pid_node);
  # its own DDS domain, so a sim or the PL5 stand-in running elsewhere cannot answer /board/tilt_cmd
  find_package(launch_testing_ament_cmake REQUIRED)
  add_launch_test(test/test_sim_tilt.py TIMEOUT 900 ENV ROS_DOMAIN_ID=79 ROS_LOCALHOST_ONLY=1)
endif()

ament_package()
