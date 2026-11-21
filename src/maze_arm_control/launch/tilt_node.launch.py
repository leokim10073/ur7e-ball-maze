"""Start the arm control node (tilt command -> joint velocities) next to the sim.

    ros2 launch maze_arm_control tilt_node.launch.py [config:=<sim.yaml>] [use_sim_time:=true]

The grasp (handle pose in tool0) comes from sim.yaml board.grasp_offset and the handle center
from the board meshes, the same values the sim builds its scene from. The UR kinematics come from
ur_description for robot.ur_type.
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from maze_sim_description.board_assets import load_board_assets
from maze_sim_description.config import load_sim_config, resolve_path


def _node(context):
    config_path = Path(LaunchConfiguration("config").perform(context))
    cfg = load_sim_config(config_path)
    board = load_board_assets(resolve_path(cfg.board.mesh_dir, config_path.parent), units=cfg.board.units)
    kinematics = Path(get_package_share_directory("ur_description")) / "config" / cfg.robot.ur_type
    return [
        Node(
            package="maze_arm_control",
            executable="tilt_node",
            output="screen",
            parameters=[{
                "use_sim_time": LaunchConfiguration("use_sim_time").perform(context) == "true",
                "kinematics_file": str(kinematics / "default_kinematics.yaml"),
                "grasp_xyz": [float(v) for v in cfg.board.grasp_offset.xyz],
                "grasp_rpy": [float(v) for v in cfg.board.grasp_offset.rpy],
                "handle_center": [float(v) for v in board.handle_center_m],
            }],
        )
    ]


def generate_launch_description():
    default_config = Path(get_package_share_directory("maze_sim_description")) / "config" / "sim.yaml"
    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=str(default_config)),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        OpaqueFunction(function=_node),
    ])
