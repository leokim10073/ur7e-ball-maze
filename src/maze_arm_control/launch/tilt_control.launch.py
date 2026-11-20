"""Remote-pivot tilt control (Contract 4): /board/tilt_cmd -> /forward_velocity_controller/commands,
/board/tilt_state from FK. Start it once the arm holds the board at the home pose (in the sim: after
/sim/reset_done); it switches to forward_velocity_controller itself and refuses to start away from home.
    ros2 launch maze_arm_control tilt_control.launch.py                      # next to the sim
    ros2 launch maze_arm_control tilt_control.launch.py use_sim_time:=false params_file:=/path/to/tilt_control.yaml
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    default_params = os.path.join(get_package_share_directory("maze_arm_control"), "config", "tilt_control.yaml")
    args = [
        DeclareLaunchArgument("use_sim_time", default_value="true", description="true with the sim, false on the robot"),
        DeclareLaunchArgument("params_file", default_value=default_params, description="parameters (all required)"),
    ]
    node = Node(
        package="maze_arm_control",
        executable="tilt_control",
        name="tilt_control",
        output="screen",
        parameters=[
            LaunchConfiguration("params_file"),
            {"use_sim_time": ParameterValue(LaunchConfiguration("use_sim_time"), value_type=bool)},
        ],
    )
    return LaunchDescription([*args, node])
