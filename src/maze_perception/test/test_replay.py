"""P5 offline replay: the pipeline over recorded frames, truth alignment, CSV, and (with ROS 2) a
rosbag written and read back."""

import csv

import numpy as np
import pytest

from conftest import board_image_with_ball, board_in_camera
from maze_perception.ball import WallGrid
from maze_perception.replay import Frame, Truth, run, summarize, write_csv

DT = 1 / 60


def outer_walls(cfg):
    """Contract 1's grid with only the outer walls (the synthetic board has no inner ones)."""
    centers = -0.15 + (np.arange(60) + 0.5) * 0.005
    cx, cy = np.meshgrid(centers, centers)
    x0, x1, y0, y1 = cfg.ball.floor
    outside = (cx < x0) | (cx > x1) | (cy < y0) | (cy > y1)
    return WallGrid(cells=outside, origin=(-0.15, -0.15), resolution=0.005)


def rolling(cfg, camera, n=30, v=(0.05, -0.03), start=(-0.02, 0.03)):
    """(frames, truth): the ball rolling at v (m/s) on a still, slightly tilted board."""
    R, t = board_in_camera(tilt_deg=(2, -1))
    frames, ball, board = [], [], []
    for k in range(n):
        xy = (start[0] + v[0] * k * DT, start[1] + v[1] * k * DT)
        image = board_image_with_ball(cfg, camera, R, t, xy)
        frames.append(Frame(stamp=10 + k * DT, image=image, camera=camera))
        ball.append((10 + k * DT, (*xy, cfg.ball.radius), (0.0, 0.0, 0.0, 1.0)))
        board.append((10 + k * DT, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)))  # board = world: truth is the ball
    return frames, Truth(ball, board)


def test_truth_in_the_board_frame():
    # board turned 90 deg about z and moved; ball 1 cm along world +y from the board's center
    q = (0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4))
    truth = Truth(ball=[(1.0, (0.5, 0.21, 0.4), (0, 0, 0, 1))], board=[(1.0, (0.5, 0.2, 0.4), q)])
    assert truth.at(1.0) == pytest.approx((0.01, 0.0))  # world +y is board +x
    assert truth.at(1.0 + 0.019) is not None and truth.at(1.0 + 0.03) is None  # nothing near


def test_replay_tracks_the_ball(cfg, camera):
    frames, truth = rolling(cfg, camera)
    results = list(run(frames, cfg, outer_walls(cfg), truth))
    assert len(results) == len(frames)
    s = summarize(results)
    assert s["board"] == 1.0 and s["ball"] == 1.0 and s["tracking_ok"] == 1.0
    assert s["fps"] == pytest.approx(60)
    assert s["measured_error_rms"] < 0.001 and s["state_error_max"] < 0.002
    last = results[-1].estimate
    assert np.hypot(last.vx - 0.05, last.vy + 0.03) < 0.02


def test_no_walls_no_ball(cfg, camera):
    # like ball_tracker before /maze/map arrives: the board, but no ball, every frame reported
    frames, _ = rolling(cfg, camera, n=5)
    results = list(run(frames, cfg, None))
    assert len(results) == 5 and all(r.board_found and r.measured is None for r in results)
    assert not any(r.estimate.tracking_ok for r in results)


def test_frames_out_of_order_are_skipped(cfg, camera):
    frames, _ = rolling(cfg, camera, n=6)
    frames = frames[:3] + [frames[1]] + frames[3:]
    assert [r.stamp for r in run(frames, cfg, outer_walls(cfg))] == [f.stamp for f in frames[:3] + frames[4:]]


def test_csv(cfg, camera, tmp_path):
    frames, truth = rolling(cfg, camera, n=4)
    path = tmp_path / "replay.csv"
    write_csv(run(frames, cfg, outer_walls(cfg), truth), path)
    rows = list(csv.DictReader(open(path)))
    assert len(rows) == 4 and rows[-1]["tracking_ok"] == "1"
    assert abs(float(rows[-1]["x"]) - float(rows[-1]["truth_x"])) < 0.002


def test_rosbag_round_trip(cfg, camera, tmp_path):
    # a bag as the sim records it (raw images, camera_info, the map, the sim's poses), read back
    rosbag2_py = pytest.importorskip("rosbag2_py")
    from builtin_interfaces.msg import Time
    from geometry_msgs.msg import PoseStamped
    from maze_interfaces.msg import MazeMap
    from rclpy.serialization import serialize_message
    from sensor_msgs.msg import CameraInfo
    from std_msgs.msg import Header

    from maze_perception.replay import read_bag
    from maze_perception.ros_image import rgb_to_image

    frames, _ = rolling(cfg, camera, n=12)
    bag = tmp_path / "bag"
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id="sqlite3"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    for name, typ in [("/camera/image_raw", "sensor_msgs/msg/Image"),
                      ("/camera/camera_info", "sensor_msgs/msg/CameraInfo"),
                      ("/maze/map", "maze_interfaces/msg/MazeMap"),
                      ("/sim/ball/pose", "geometry_msgs/msg/PoseStamped"),
                      ("/sim/board/pose", "geometry_msgs/msg/PoseStamped")]:
        writer.create_topic(rosbag2_py.TopicMetadata(name=name, type=typ, serialization_format="cdr"))

    walls = outer_walls(cfg)
    maze = MazeMap(source="test")
    maze.walls.info.resolution, maze.walls.info.width, maze.walls.info.height = 0.005, 60, 60
    maze.walls.info.origin.position.x = maze.walls.info.origin.position.y = -0.15
    maze.walls.data = np.where(walls.cells, 100, 0).astype(np.int8).ravel().tolist()
    writer.write("/maze/map", serialize_message(maze), int(9e9))
    for k, f in enumerate(frames):
        ns = int(round(f.stamp * 1e9))
        header = Header(frame_id="camera_optical_frame", stamp=Time(sec=ns // 10 ** 9, nanosec=ns % 10 ** 9))
        info = CameraInfo(header=header, width=camera.width, height=camera.height, distortion_model="plumb_bob",
                          k=camera.K.ravel().tolist(), d=[0.0] * 5)
        writer.write("/camera/camera_info", serialize_message(info), ns)
        writer.write("/camera/image_raw", serialize_message(rgb_to_image(f.image, header)), ns)
        x, y = -0.02 + 0.05 * k * DT, 0.03 - 0.03 * k * DT
        for topic, (px, py, pz) in (("/sim/ball/pose", (x, y, cfg.ball.radius)),
                                    ("/sim/board/pose", (0.0, 0.0, 0.0))):
            pose = PoseStamped(header=Header(frame_id="world", stamp=header.stamp))
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = px, py, pz
            pose.pose.orientation.w = 1.0
            writer.write(topic, serialize_message(pose), ns)
    del writer  # closes the bag

    read, bag_walls, truth = read_bag(str(bag), "/camera/image_raw", "/camera/camera_info")
    assert bag_walls is not None and bag_walls.cells.sum() == walls.cells.sum()
    results = list(run(read, cfg, bag_walls, truth))
    s = summarize(results)
    assert s["frames"] == 12 and s["tracking_ok"] == 1.0
    assert s["measured_error_rms"] < 0.001
