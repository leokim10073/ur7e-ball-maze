cmake_minimum_required(VERSION 3.8)
project(maze_arm_control)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(controller_manager_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(maze_interfaces REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)

# Kinematics and the tilt controller, no ROS: used by the node and the unit tests.
add_library(tilt_controller STATIC src/kinematics.cpp src/tilt_controller.cpp src/config_io.cpp)
target_compile_features(tilt_controller PUBLIC cxx_std_17)
target_include_directories(tilt_controller PUBLIC include)
target_link_libraries(tilt_controller PUBLIC Eigen3::Eigen yaml-cpp)
set_target_properties(tilt_controller PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(tilt_node src/tilt_node.cpp)
target_link_libraries(tilt_node tilt_controller)
ament_target_dependencies(tilt_node rclcpp ament_index_cpp controller_manager_msgs maze_interfaces sensor_msgs
                          std_msgs)

install(TARGETS tilt_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_tilt_controller test/test_tilt_controller.cpp)
  target_link_libraries(test_tilt_controller tilt_controller)
  ament_target_dependencies(test_tilt_controller ament_index_cpp)
  target_compile_definitions(test_tilt_controller PRIVATE
                             ARM_YAML="${CMAKE_CURRENT_SOURCE_DIR}/config/arm.yaml")
endif()

ament_package()
