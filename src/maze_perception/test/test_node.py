"""board_tracker and ball_tracker end to end over ROS: synthetic camera_info and image in, board
pose and ball state out. Needs ROS 2 (run inside maze-sim after colcon build); skipped without rclpy."""

import time

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")

from geometry_msgs.msg import PoseStamped  # noqa: E402
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from rclpy.parameter import Parameter  # noqa: E402
from sensor_msgs.msg import CameraInfo, Image  # noqa: E402
from std_msgs.msg import Header  # noqa: E402
from builtin_interfaces.msg import Time  # noqa: E402

from conftest import PKG, board_in_camera, render_board  # noqa: E402
from maze_perception.board_node import BoardTracker  # noqa: E402
from maze_perception.board_pose import BoardPose  # noqa: E402
from maze_perception.ros_image import rgb_to_image  # noqa: E402


@pytest.fixture
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


def test_board_pose_published(ros, cfg, camera):
    tracker = BoardTracker(parameter_overrides=[
        Parameter("config", value=str(PKG / "config" / "perception.yaml")),
        Parameter("image_topic", value="/test/image"),
        Parameter("camera_info_topic", value="/test/camera_info"),
    ])
    peer = rclpy.create_node("board_tracker_test")
    executor = SingleThreadedExecutor()
    executor.add_node(tracker)
    executor.add_node(peer)

    R, t = board_in_camera(tilt_deg=(3, -2), offset=(0.01, 0.0))
    gray = render_board(cfg, camera, R, t)
    header = Header(frame_id="camera_optical_frame")
    image = rgb_to_image(np.dstack([gray] * 3), header)
    info = CameraInfo(header=header, width=camera.width, height=camera.height, distortion_model="plumb_bob",
                      k=camera.K.ravel().tolist(), d=[0.0] * 5)

    poses, boards = [], []
    peer.create_subscription(PoseStamped, "/perception/board_pose", poses.append, 10)
    peer.create_subscription(Image, "/perception/board_image", boards.append, 10)
    info_pub = peer.create_publisher(CameraInfo, "/test/camera_info", 10)
    image_pub = peer.create_publisher(Image, "/test/image", 10)
    deadline = time.monotonic() + 15.0
    while (not poses or not boards) and time.monotonic() < deadline:
        info_pub.publish(info)
        image_pub.publish(image)
        executor.spin_once(timeout_sec=0.1)
    tracker.destroy_node()
    peer.destroy_node()

    assert poses, "no /perception/board_pose"
    p = poses[-1]
    assert p.header.frame_id == "camera_optical_frame"
    np.testing.assert_allclose([p.pose.position.x, p.pose.position.y, p.pose.position.z], t, atol=1e-3)
    assert boards and (boards[-1].width, boards[-1].height) == cfg.rectified.size


def _with_ball(cfg, camera, R, t, gray, ball_xy):
    """RGB camera image: the board's markers on a tan floor, and a grey ball (r = cfg.ball.radius)
    resting at board (x, y)."""
    import cv2

    rgb = np.clip(np.dstack([gray] * 3) * (np.array([209, 184, 140]) / 128.0), 0, 255).astype(np.uint8)
    pose = BoardPose(R, t, (0, 1, 2, 3), 0.0)
    center = np.array([[*ball_xy, cfg.ball.radius]])
    c = pose.board_to_pixel(center, camera)[0]
    r_px = camera.K[0, 0] * cfg.ball.radius / (R @ center[0] + t)[2]
    cv2.circle(rgb, (int(round(c[0] * 16)), int(round(c[1] * 16))), int(round(r_px * 16)), (199, 199, 211), -1,
               lineType=cv2.LINE_AA, shift=4)
    return rgb


def test_ball_state_published(ros, cfg, camera):
    from maze_interfaces.msg import BallState

    from maze_perception.ball_node import BallTracker
    from maze_perception.board_node import SENSOR_QOS

    tracker = BallTracker(parameter_overrides=[
        Parameter("config", value=str(PKG / "config" / "perception.yaml")),
        Parameter("image_topic", value="/test/image"),
        Parameter("camera_info_topic", value="/test/camera_info"),
        Parameter("rectified_rate", value=0.0),
    ])
    peer = rclpy.create_node("ball_tracker_test")
    executor = SingleThreadedExecutor()
    executor.add_node(tracker)
    executor.add_node(peer)

    R, t = board_in_camera(tilt_deg=(2, -1), offset=(0.0, 0.01))
    ball_xy = (0.03, -0.05)
    with_ball = rgb_to_image(_with_ball(cfg, camera, R, t, render_board(cfg, camera, R, t), ball_xy), Header())
    no_board = rgb_to_image(np.full((camera.height, camera.width, 3), 128, np.uint8), Header())
    info = CameraInfo(header=Header(frame_id="camera_optical_frame"), width=camera.width, height=camera.height,
                      distortion_model="plumb_bob", k=camera.K.ravel().tolist(), d=[0.0] * 5)

    states = []
    # /ball/state is best effort (contract 2); a reliable subscription would not match it
    peer.create_subscription(BallState, "/ball/state", states.append, SENSOR_QOS)
    info_pub = peer.create_publisher(CameraInfo, "/test/camera_info", 10)
    image_pub = peer.create_publisher(Image, "/test/image", 10)

    def send(image, k):  # frame k, captured at 100 s + k / 60
        image.header = Header(frame_id="camera_optical_frame", stamp=Time(sec=100, nanosec=int(k * 1e9 / 60)))
        info_pub.publish(info)
        image_pub.publish(image)
        deadline = time.monotonic() + 5.0
        n = len(states)
        while len(states) == n and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.05)

    for k in range(20):
        send(with_ball, k)
    seen = list(states)
    for k in range(20, 25):  # the board leaves the view: still one message per frame, a prediction
        send(no_board, k)
    tracker.destroy_node()
    peer.destroy_node()

    assert seen, "no /ball/state"
    s = seen[-1]
    assert s.header.frame_id == "maze_board" and s.tracking_ok
    assert np.hypot(s.x - ball_xy[0], s.y - ball_xy[1]) < 0.002
    assert np.hypot(s.vx, s.vy) < 0.05
    assert s.position_covariance[0] > 0 and s.position_covariance[0] == s.position_covariance[3]
    lost = states[len(seen):]
    assert len(lost) == 5 and not any(m.tracking_ok for m in lost)
    assert [m.header.stamp.nanosec for m in lost] == [int(k * 1e9 / 60) for k in range(20, 25)]
    assert np.hypot(lost[-1].x - ball_xy[0], lost[-1].y - ball_xy[1]) < 0.005

