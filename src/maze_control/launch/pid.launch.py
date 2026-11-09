import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_cfg = os.path.join(get_package_share_directory('maze_control'),
                               'config', 'pid.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=default_cfg),
        Node(package='maze_control', executable='pid_node',
             name='maze_pid_controller', output='screen',
             parameters=[LaunchConfiguration('config')]),
    ])
