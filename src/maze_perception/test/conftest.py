from pathlib import Path

import cv2
import numpy as np
import pytest

from maze_perception.board_pose import Camera
from maze_perception.config import load_perception_config

PKG = Path(__file__).resolve().parents[1]
REPO = PKG.parents[1]


@pytest.fixture(scope="session")
def cfg():
    return load_perception_config(PKG / "config" / "perception.yaml")


@pytest.fixture(scope="session")
def camera():
    """The sim's C922 model: 1280 x 720, 70.42 deg horizontal field of view, no distortion."""
    w, h = 1280, 720
    f = (w / 2) / np.tan(np.radians(70.42) / 2)
    return Camera(K=np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1]]), D=np.zeros(5), width=w, height=h)


def marker_image(dictionary, marker_id, side_px):
    d = getattr(cv2.aruco, dictionary)
    if hasattr(cv2.aruco, "generateImageMarker"):
        return cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(d), marker_id, side_px)
    return cv2.aruco.drawMarker(cv2.aruco.Dictionary_get(d), marker_id, side_px)


def board_in_camera(height=1.0, tilt_deg=(0.0, 0.0), yaw_deg=90.0, offset=(0.0, 0.0)):
    """(R, t): camera `height` m above the board, optical axis straight down onto it; board
    rotated by yaw_deg about the view axis, then tilted by (about x, about y) degrees."""
    def rot(axis, deg):
        return cv2.Rodrigues(np.asarray(axis, float) * np.radians(deg))[0]

    # board z toward the camera: camera z (forward) = -board z
    R_down = np.array([[1.0, 0, 0], [0, -1, 0], [0, 0, -1]])
    R = R_down @ rot([0, 0, 1], yaw_deg) @ rot([1, 0, 0], tilt_deg[0]) @ rot([0, 1, 0], tilt_deg[1])
    t = np.array([offset[0], offset[1], height])
    return R, t


def render_board(cfg, camera, R, t, background=128, quiet_zone=0.005):
    """Synthetic camera image (gray) of the board's markers, each on a white sticker with a
    `quiet_zone` m border, on a uniform background."""
    img = np.full((camera.height, camera.width), background, np.uint8)
    rvec = cv2.Rodrigues(R)[0]
    for mid in cfg.board.markers:
        side = 400
        sticker = np.full((side + 2 * 50, side + 2 * 50), 255, np.uint8)
        sticker[50:50 + side, 50:50 + side] = marker_image(cfg.board.dictionary, mid, side)
        corners = cfg.board.marker_corners(mid)
        center = corners.mean(axis=0)
        scale = (cfg.board.marker_size / 2 + quiet_zone) / (cfg.board.marker_size / 2)
        outer = center + (corners - center) * scale
        dst, _ = cv2.projectPoints(outer, rvec, t, camera.K, camera.D)
        e = side + 100 - 0.5  # outer edges of the sticker's pixels (pixel centers are integers)
        src = np.float32([[-0.5, -0.5], [e, -0.5], [e, e], [-0.5, e]])
        H = cv2.getPerspectiveTransform(src, dst.reshape(4, 2).astype(np.float32))
        cv2.warpPerspective(sticker, H, (camera.width, camera.height), dst=img, flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_TRANSPARENT)
    return cv2.GaussianBlur(img, (0, 0), 0.7)  # a little lens blur, as a real camera has


def board_image_with_ball(cfg, camera, R, t, ball_xy, floor_rgb=(209, 184, 140), ball_rgb=(199, 199, 211)):
    """RGB camera image: render_board's markers on a tan floor, and a grey ball (cfg.ball.radius)
    resting at board (x, y) = ball_xy."""
    from maze_perception.board_pose import BoardPose

    gray = render_board(cfg, camera, R, t)
    rgb = np.clip(np.dstack([gray] * 3) * (np.asarray(floor_rgb) / 128.0), 0, 255).astype(np.uint8)
    pose = BoardPose(R, t, (0, 1, 2, 3), 0.0)
    center = np.array([[*ball_xy, cfg.ball.radius]])
    c = pose.board_to_pixel(center, camera)[0]
    r_px = camera.K[0, 0] * cfg.ball.radius / (R @ center[0] + t)[2]
    cv2.circle(rgb, (int(round(c[0] * 16)), int(round(c[1] * 16))), int(round(r_px * 16)), ball_rgb, -1,
               lineType=cv2.LINE_AA, shift=4)
    return rgb

