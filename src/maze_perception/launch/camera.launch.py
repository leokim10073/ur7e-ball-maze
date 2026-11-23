"""The C922 on /camera/image_raw and /camera/camera_info (usb_cam), as the sim camera publishes them.
    ros2 launch maze_perception camera.launch.py video_device:=/dev/video2 \\
        camera_info_url:=file:///home/$USER/c922_calibration.yaml
Then lock focus and exposure: ros2 run maze_perception c922_setup.sh (docs/perception_lab.md).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    share = FindPackageShare("maze_perception")
    args = [
        DeclareLaunchArgument("video_device", default_value="/dev/video0",
                              description="the C922's /dev/video* (v4l2-ctl --list-devices)"),
        DeclareLaunchArgument("camera_info_url",
                              default_value=["package://maze_perception/config/c922_nominal.yaml"],
                              description="calibration file (file:///...); default: nominal intrinsics"),
    ]
    camera = Node(
        package="usb_cam",
        executable="usb_cam_node_exe",
        namespace="camera",
        name="c922",
        output="screen",
        parameters=[
            PathJoinSubstitution([share, "config", "c922.yaml"]),
            {"video_device": LaunchConfiguration("video_device"),
             "camera_info_url": LaunchConfiguration("camera_info_url")},
        ],
    )
    return LaunchDescription([*args, camera])
