"""Perception on the real camera, in one command (P6, docs/perception_lab.md): the C922 (camera.launch.py),
the CAD maze map and ball_tracker (perception.launch.py) on the wall clock. Run from the repo root.
    ros2 launch maze_perception real.launch.py video_device:=/dev/video2 ball_radius:=0.0125 \\
        camera_info_url:=file:///home/$USER/c922_calibration.yaml
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    share = FindPackageShare("maze_perception")
    args = [
        DeclareLaunchArgument("video_device", default_value="/dev/video0"),
        DeclareLaunchArgument("camera_info_url",
                              default_value=["package://maze_perception/config/c922_nominal.yaml"]),
        DeclareLaunchArgument("ball_radius", default_value="0.0125", description="m; the 25 mm marble (#19)"),
        DeclareLaunchArgument("level", default_value="L1"),
        DeclareLaunchArgument("rectified_rate", default_value="5.0"),
    ]
    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([share, "launch", "camera.launch.py"])),
        launch_arguments={"video_device": LaunchConfiguration("video_device"),
                          "camera_info_url": LaunchConfiguration("camera_info_url")}.items(),
    )
    perception = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([share, "launch", "perception.launch.py"])),
        launch_arguments={"use_sim_time": "false", "ball_radius": LaunchConfiguration("ball_radius"),
                          "level": LaunchConfiguration("level"),
                          "rectified_rate": LaunchConfiguration("rectified_rate")}.items(),
    )
    return LaunchDescription([*args, camera, perception])
