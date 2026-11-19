cmake_minimum_required(VERSION 3.8)
project(maze_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Circle.msg"
  "msg/MazeMap.msg"
  "msg/BallState.msg"
  "msg/Corridor.msg"
  "msg/PlannedRoute.msg"
  "msg/BoardTilt.msg"
  DEPENDENCIES std_msgs geometry_msgs nav_msgs
)

ament_export_dependencies(rosidl_default_runtime)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_messages test/test_messages.py TIMEOUT 60)
endif()
ament_package()
