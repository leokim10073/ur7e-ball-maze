"""perception.yaml: the ArUco board model, the pose and rectification settings, and the ball."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import yaml


@dataclass(frozen=True)
class BoardModel:
    """Where the ArUco markers sit on the board, in the board frame (m)."""

    dictionary: str
    marker_size: float
    markers: Dict[int, Tuple[float, float, float]]  # id -> marker center

    def marker_corners(self, marker_id: int) -> np.ndarray:
        """(4, 3) corners of one marker in OpenCV's order (top-left, top-right, bottom-right,
        bottom-left of the marker image), with the image's top edge toward board +y."""
        cx, cy, cz = self.markers[marker_id]
        h = self.marker_size / 2.0
        return np.array([[cx - h, cy + h, cz], [cx + h, cy + h, cz], [cx + h, cy - h, cz], [cx - h, cy - h, cz]])


@dataclass(frozen=True)
class PoseSettings:
    min_markers: int
    max_reprojection_px: float


@dataclass(frozen=True)
class RectifiedSettings:
    """Board area drawn in the rectified (top-down) image and its scale."""

    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    px_per_m: float

    @property
    def size(self) -> Tuple[int, int]:
        """(width, height) in pixels."""
        return (int(round((self.x_range[1] - self.x_range[0]) * self.px_per_m)),
                int(round((self.y_range[1] - self.y_range[0]) * self.px_per_m)))


@dataclass(frozen=True)
class BallSettings:
    radius: float  # m
    floor: Tuple[float, float, float, float]  # x_min, x_max, y_min, y_max inside the outer walls, m
    not_floor: float  # CIELAB distance from the floor color at which a pixel is fully not floor
    min_score: float  # 0..1


@dataclass(frozen=True)
class PerceptionConfig:
    board: BoardModel
    pose: PoseSettings
    rectified: RectifiedSettings
    ball: BallSettings


def _pair(raw, key):
    v = tuple(float(x) for x in raw)
    if len(v) != 2 or not v[0] < v[1]:
        raise ValueError(f"{key} must be [min, max] with min < max, got {list(raw)}")
    return v


def load_perception_config(path) -> PerceptionConfig:
    """Read and check perception.yaml; a bad value raises ValueError naming its key."""
    raw = yaml.safe_load(Path(path).read_text())
    try:
        a, p, r = raw["aruco"], raw["pose"], raw["rectified"]
        markers = {int(k): tuple(float(x) for x in v) for k, v in a["markers"].items()}
        board = BoardModel(dictionary=str(a["dictionary"]), marker_size=float(a["marker_size"]), markers=markers)
        pose = PoseSettings(min_markers=int(p["min_markers"]), max_reprojection_px=float(p["max_reprojection_px"]))
        rect = RectifiedSettings(x_range=_pair(r["x_range"], "rectified.x_range"),
                                 y_range=_pair(r["y_range"], "rectified.y_range"), px_per_m=float(r["px_per_m"]))
        b = raw["ball"]
        if len(b["floor"]) != 4:
            raise ValueError("ball.floor must be [x_min, x_max, y_min, y_max]")
        fx, fy = _pair(b["floor"][:2], "ball.floor x"), _pair(b["floor"][2:], "ball.floor y")
        ball = BallSettings(radius=float(b["radius"]), floor=(*fx, *fy), not_floor=float(b["not_floor"]),
                            min_score=float(b["min_score"]))
    except KeyError as e:
        raise ValueError(f"{path}: missing key {e}") from None
    if any(len(c) != 3 for c in markers.values()):
        raise ValueError("aruco.markers: each center must be [x, y, z]")
    if board.marker_size <= 0:
        raise ValueError("aruco.marker_size must be positive")
    if not 1 <= pose.min_markers <= len(markers):
        raise ValueError(f"pose.min_markers must be between 1 and {len(markers)}")
    if rect.px_per_m <= 0:
        raise ValueError("rectified.px_per_m must be positive")
    if ball.radius <= 0 or ball.not_floor <= 0:
        raise ValueError("ball.radius and ball.not_floor must be positive")
    if not 0 < ball.min_score < 1:
        raise ValueError("ball.min_score must be between 0 and 1")
    if ball.radius * rect.px_per_m < 3:
        raise ValueError(f"ball.radius is {ball.radius * rect.px_per_m:.1f} px in the rectified image; "
                         "raise rectified.px_per_m so that it is at least 3 px")
    x0, x1, y0, y1 = ball.floor
    if x0 < rect.x_range[0] or x1 > rect.x_range[1] or y0 < rect.y_range[0] or y1 > rect.y_range[1]:
        raise ValueError("ball.floor must lie inside the rectified area")
    if min(x1 - x0, y1 - y0) <= 2 * ball.radius:
        raise ValueError("ball.floor is too small for the ball")
    return PerceptionConfig(board=board, pose=pose, rectified=rect, ball=ball)
