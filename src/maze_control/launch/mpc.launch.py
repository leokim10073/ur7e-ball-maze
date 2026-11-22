import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    default_cfg = os.path.join(get_package_share_directory('maze_control'),
                               'config', 'mpc.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=default_cfg),
        # true in the sim: the 200 ms watchdog then runs on /clock, so a
        # slower-than-real-time sim cannot trip it.
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        Node(package='maze_control', executable='mpc_node',
             name='maze_mpc_controller', output='screen',
             parameters=[LaunchConfiguration('config'),
                         {'use_sim_time': ParameterValue(
                             LaunchConfiguration('use_sim_time'), value_type=bool)}]),
    ])
