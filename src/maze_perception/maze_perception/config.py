"""perception.yaml: the ArUco board model, the pose and rectification settings, and the maze
geometry used for mapping."""

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
class LevelLayout:
    """One maze level's grid and its given (not detected) features; rows count from the top."""

    n: int
    cell: float
    start: Tuple[int, int]
    goal: Tuple[int, int]
    goal_radius: float
    holes: Tuple[Tuple[int, int, float], ...]


@dataclass(frozen=True)
class MazeSettings:
    plate: float
    margin: float
    wall_thickness: float
    resolution: float
    search_margin: float
    wall_contrast: float  # gray levels
    level: str
    layout: LevelLayout  # the active level's


@dataclass(frozen=True)
class PerceptionConfig:
    board: BoardModel
    pose: PoseSettings
    rectified: RectifiedSettings
    maze: MazeSettings


def _pair(raw, key):
    v = tuple(float(x) for x in raw)
    if len(v) != 2 or not v[0] < v[1]:
        raise ValueError(f"{key} must be [min, max] with min < max, got {list(raw)}")
    return v


def _cell(raw, key, n):
    rc = tuple(int(x) for x in raw)
    if len(rc) != 2 or not all(0 <= v < n for v in rc):
        raise ValueError(f"{key} must be [row, col] inside the {n} x {n} grid, got {list(raw)}")
    return rc


def _load_maze(m) -> MazeSettings:
    level = str(m["level"])
    if level not in m["levels"]:
        raise ValueError(f"maze.level {level} is not in maze.levels")
    lv = m["levels"][level]
    key = f"maze.levels.{level}"
    n = int(lv["n"])
    if n < 1:
        raise ValueError(f"{key}.n must be at least 1")
    holes = []
    for h in lv["holes"]:
        if len(h) != 3:
            raise ValueError(f"{key}.holes: each hole must be [row, col, diameter_m]")
        holes.append((*_cell(h[:2], f"{key}.holes", n), float(h[2])))
    layout = LevelLayout(n=n, cell=float(lv["cell"]), start=_cell(lv["start"], f"{key}.start", n),
                         goal=_cell(lv["goal"], f"{key}.goal", n), goal_radius=float(lv["goal_radius"]),
                         holes=tuple(holes))
    s = MazeSettings(plate=float(m["plate"]), margin=float(m["margin"]), wall_thickness=float(m["wall_thickness"]),
                     resolution=float(m["resolution"]), search_margin=float(m["search_margin"]),
                     wall_contrast=float(m["wall_contrast"]), level=level, layout=layout)
    for k in ("plate", "wall_thickness", "resolution", "wall_contrast"):
        if getattr(s, k) <= 0:
            raise ValueError(f"maze.{k} must be positive")
    if layout.cell <= 0 or layout.goal_radius <= 0:
        raise ValueError(f"{key}: cell and goal_radius must be positive")
    span = 2 * s.margin + n * layout.cell + (n + 1) * s.wall_thickness
    if span > s.plate + 1e-9:
        raise ValueError(f"{key}: the grid ({span * 1000:.1f} mm with margins) does not fit on the plate")
    return s


def load_perception_config(path) -> PerceptionConfig:
    """Read and check perception.yaml; a bad value raises ValueError naming its key."""
    raw = yaml.safe_load(Path(path).read_text())
    try:
        maze = _load_maze(raw["maze"])
        a, p, r = raw["aruco"], raw["pose"], raw["rectified"]
        markers = {int(k): tuple(float(x) for x in v) for k, v in a["markers"].items()}
        board = BoardModel(dictionary=str(a["dictionary"]), marker_size=float(a["marker_size"]), markers=markers)
        pose = PoseSettings(min_markers=int(p["min_markers"]), max_reprojection_px=float(p["max_reprojection_px"]))
        rect = RectifiedSettings(x_range=_pair(r["x_range"], "rectified.x_range"),
                                 y_range=_pair(r["y_range"], "rectified.y_range"), px_per_m=float(r["px_per_m"]))
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
    return PerceptionConfig(board=board, pose=pose, rectified=rect, maze=maze)
