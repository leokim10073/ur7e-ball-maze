"""The planner on /maze/route and /maze/path (Contract 3). Needs /maze/map and /ball/state:
    ros2 launch maze_planning planner.launch.py                        # next to the sim
    ros2 launch maze_planning planner.launch.py use_sim_time:=false params_file:=/path/to/planner.yaml
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    default_params = os.path.join(get_package_share_directory("maze_planning"), "config", "planner.yaml")
    args = [
        DeclareLaunchArgument("use_sim_time", default_value="true", description="true with the sim, false on the robot"),
        DeclareLaunchArgument("params_file", default_value=default_params, description="planner parameters (all required)"),
    ]
    planner = Node(
        package="maze_planning",
        executable="planner",
        name="planner",
        output="screen",
        parameters=[
            LaunchConfiguration("params_file"),
            {"use_sim_time": ParameterValue(LaunchConfiguration("use_sim_time"), value_type=bool)},
        ],
    )
    return LaunchDescription([*args, planner])
