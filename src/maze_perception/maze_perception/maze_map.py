"""The maze map from the rectified top-down board image (P2): which walls are there, as the
occupancy grid of docs/interfaces.md contract 1.

Every level is an n x n grid of cells separated by straight walls (maze_CAD/scripts/levels.py),
so a wall can only lie on a grid line between two grid nodes. The mapper scores each of these
2 n (n + 1) wall segments in the rectified image and keeps the ones that look like a wall. In the
sim the wall tops are the floor's color; a wall shows as the dark line of its edge and side face,
so the score is how much darker than the floor the darkest line across the segment is (median
along it, so the ball, a ring or a short glare does not decide it). A segment scoring at least
wall_contrast gray levels is a wall. The outer walls always exist; if they do not reach it either,
there is no board in the image and no map. Parts of the board outside the camera image are
UNKNOWN cells, not walls (rectify() leaves them black).

Start, goal and holes are given by the level's config, not detected (the ArUco pose places them).

Frames: board frame (m), origin at the plate center, x right, y up. The level layout counts rows
from the top (+y edge) as the CAD does; the occupancy grid counts rows from the bottom (-y edge)
as ROS does.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from maze_perception.board_pose import board_to_rectified
from maze_perception.config import MazeSettings, RectifiedSettings

FREE, WALL, UNKNOWN = 0, 100, -1

Segment = Tuple[str, int, int]  # ("h", grid line from the top, column) or ("v", grid line from the left, row)


@dataclass(frozen=True)
class Circle:
    x: float  # center in the board frame, m
    y: float
    radius: float


@dataclass(frozen=True)
class MazeGeometry:
    """Where the grid's walls and cells are on the board, for one level."""

    settings: MazeSettings

    @property
    def n(self) -> int:
        return self.settings.layout.n

    @property
    def pitch(self) -> float:
        return self.settings.layout.cell + self.settings.wall_thickness

    def line_pos(self, k: int) -> float:
        """Near edge of grid line k from the plate's left (or top) edge, m."""
        return self.settings.margin + k * self.pitch

    def _to_board(self, from_left: float, from_top: float) -> Tuple[float, float]:
        half = self.settings.plate / 2
        return from_left - half, half - from_top

    def segments(self) -> List[Segment]:
        n = self.n
        return [("h", j, c) for j in range(n + 1) for c in range(n)] + \
               [("v", i, r) for i in range(n + 1) for r in range(n)]

    def is_outer(self, seg: Segment) -> bool:
        return seg[1] in (0, self.n)

    def segment_rect(self, seg: Segment) -> Tuple[float, float, float, float]:
        """(x_min, x_max, y_min, y_max) of the segment's footprint in the board frame, including the
        wall-thickness squares at both grid nodes (as the CAD draws a wall)."""
        kind, line, idx = seg
        t = self.settings.wall_thickness
        a0, a1 = self.line_pos(idx), self.line_pos(idx + 1) + t  # along the wall
        b0, b1 = self.line_pos(line), self.line_pos(line) + t  # across it
        if kind == "h":
            (x0, y1), (x1, y0) = self._to_board(a0, b0), self._to_board(a1, b1)
        else:
            (x0, y1), (x1, y0) = self._to_board(b0, a0), self._to_board(b1, a1)
        return x0, x1, y0, y1

    def cell_center(self, row: int, col: int) -> Tuple[float, float]:
        """Board (x, y) of cell (row from the top, column from the left)."""
        t, half_cell = self.settings.wall_thickness, self.settings.layout.cell / 2
        return self._to_board(self.line_pos(col) + t + half_cell, self.line_pos(row) + t + half_cell)

    def inner_rect(self) -> Tuple[float, float, float, float]:
        """(x_min, x_max, y_min, y_max) inside the outer walls."""
        t = self.settings.wall_thickness
        (x0, y1), (x1, y0) = self._to_board(self.line_pos(0) + t, self.line_pos(0) + t), \
            self._to_board(self.line_pos(self.n), self.line_pos(self.n))
        return x0, x1, y0, y1

    def outer_rect(self) -> Tuple[float, float, float, float]:
        """(x_min, x_max, y_min, y_max) of the outer walls' outside edges."""
        t = self.settings.wall_thickness
        (x0, y1), (x1, y0) = self._to_board(self.line_pos(0), self.line_pos(0)), \
            self._to_board(self.line_pos(self.n) + t, self.line_pos(self.n) + t)
        return x0, x1, y0, y1


@dataclass
class BoardMap:
    """Contract 1 without ROS types."""

    grid: np.ndarray  # (height, width) int8, FREE, WALL or UNKNOWN; row 0 = the -y edge
    resolution: float  # m per cell
    origin: Tuple[float, float]  # board (x, y) of the grid's lower-left corner
    walls: Dict[Segment, Optional[bool]]  # every grid segment: is there a wall (None: not seen)
    scores: Dict[Segment, Optional[float]]  # darkness score per segment, gray levels (None: not seen)
    contrast: float  # the outer walls' median score, gray levels (how well the walls stand out)
    start: Tuple[float, float]
    goal: Circle
    holes: List[Circle]

    def occupancy_data(self) -> List[int]:
        """nav_msgs/OccupancyGrid.data order: row-major from row 0 (the -y edge)."""
        return self.grid.astype(np.int8).ravel().tolist()


def _pixel_box(rect, rectified: RectifiedSettings, shape) -> Tuple[slice, slice]:
    """Rows and columns of the rectified-image pixels whose centers lie in the board rect."""
    x0, x1, y0, y1 = rect
    (c0, r0), (c1, r1) = board_to_rectified(np.array([[x0, y1], [x1, y0]]), rectified)
    h, w = shape[:2]
    rows = slice(max(0, int(np.ceil(r0))), min(h, int(np.floor(r1)) + 1))
    cols = slice(max(0, int(np.ceil(c0))), min(w, int(np.floor(c1)) + 1))
    return rows, cols


def score_segments(gray: np.ndarray, geom: MazeGeometry, rectified: RectifiedSettings,
                   valid: Optional[np.ndarray] = None) -> Dict[Segment, Optional[float]]:
    """Darkness score of every grid segment: floor level minus the median, along the segment, of
    the darkest pixel across it. Only the stretch between the two grid nodes counts (the nodes
    are shared with the crossing walls); across, the footprint is widened by search_margin.

    valid: rectified_coverage() mask. Only valid pixels set the floor level, and a segment is
    scored on the stretch along it that is fully seen; with less than half of it seen it scores
    None (unseen). All None if no floor pixel is valid."""
    s = geom.settings
    gray = np.asarray(gray, float)
    if valid is None:
        valid = np.ones(gray.shape[:2], bool)
    rows, cols = _pixel_box(geom.inner_rect(), rectified, gray.shape)
    floor_px = gray[rows, cols][valid[rows, cols]]
    if floor_px.size == 0:
        return {seg: None for seg in geom.segments()}
    floor = float(np.median(floor_px))
    t, m = s.wall_thickness, s.search_margin
    out = {}
    for seg in geom.segments():
        x0, x1, y0, y1 = geom.segment_rect(seg)
        rect = (x0 + t, x1 - t, y0 - m, y1 + m) if seg[0] == "h" else (x0 - m, x1 + m, y0 + t, y1 - t)
        rows, cols = _pixel_box(rect, rectified, gray.shape)
        patch, ok = gray[rows, cols], valid[rows, cols]
        across = 0 if seg[0] == "h" else 1
        seen = ok.all(axis=across)  # per position along the segment
        if patch.size == 0 or 2 * seen.sum() < seen.size:
            out[seg] = None
            continue
        out[seg] = floor - float(np.median(patch.min(axis=across)[seen]))
    return out


def rasterize(walls: Dict[Segment, Optional[bool]], geom: MazeGeometry,
              cell_seen: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Occupancy grid over the plate. Row 0 = the -y edge. A cell is
    WALL     when its center lies outside the outer walls (the lip) or in a present wall's footprint;
    UNKNOWN  otherwise, when it was not seen (cell_seen False) or lies in an unseen segment (None);
    FREE     otherwise."""
    s = geom.settings
    size = int(round(s.plate / s.resolution))
    origin = (-s.plate / 2, -s.plate / 2)
    centers = origin[0] + (np.arange(size) + 0.5) * s.resolution
    X, Y = np.meshgrid(centers, origin[1] + (np.arange(size) + 0.5) * s.resolution)  # Y[row] grows with row
    ox0, ox1, oy0, oy1 = geom.outer_rect()
    wall = (X < ox0) | (X > ox1) | (Y < oy0) | (Y > oy1)
    unknown = np.zeros_like(wall) if cell_seen is None else ~np.asarray(cell_seen, bool)
    for seg, present in walls.items():
        if present is False:
            continue
        x0, x1, y0, y1 = geom.segment_rect(seg)
        inside = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
        if present:
            wall |= inside
        else:
            unknown |= inside
    grid = np.where(wall, WALL, np.where(unknown, UNKNOWN, FREE))
    return grid.astype(np.int8), origin


def cells_seen(valid: np.ndarray, geom: MazeGeometry, rectified: RectifiedSettings) -> np.ndarray:
    """(size, size) bool over the occupancy grid (row 0 = -y edge): every rectified pixel of the
    cell is valid."""
    s = geom.settings
    size = int(round(s.plate / s.resolution))
    lo = -s.plate / 2
    out = np.zeros((size, size), bool)
    for r in range(size):
        for c in range(size):
            x0, y0 = lo + c * s.resolution, lo + r * s.resolution
            rows, cols = _pixel_box((x0, x0 + s.resolution, y0, y0 + s.resolution), rectified, valid.shape)
            block = valid[rows, cols]
            out[r, c] = block.size > 0 and bool(block.all())
    return out


def build_board_map(gray: np.ndarray, settings: MazeSettings, rectified: RectifiedSettings,
                    valid: Optional[np.ndarray] = None) -> Optional[BoardMap]:
    """The map from a rectified grayscale image of the board (as rectify() makes it, z = 0).

    valid: rectified_coverage() of the same pose. Parts of the board outside the camera image
    become UNKNOWN cells (contract 1: the planner treats them as walls), not walls or floor.
    Outer walls are always walls, seen or not.

    None when the seen outer walls do not stand out from the floor by settings.wall_contrast, or
    none is seen (no board in the image, or the wrong area rectified): never a partial map.
    """
    geom = MazeGeometry(settings)
    scores = score_segments(gray, geom, rectified, valid)
    outer = [v for k, v in scores.items() if geom.is_outer(k) and v is not None]
    if not outer:
        return None
    contrast = float(np.median(outer))
    if contrast < settings.wall_contrast:
        return None
    walls = {k: True if geom.is_outer(k) else (None if v is None else v >= settings.wall_contrast)
             for k, v in scores.items()}
    seen = None if valid is None else cells_seen(valid, geom, rectified)
    grid, origin = rasterize(walls, geom, seen)
    lv = settings.layout
    return BoardMap(
        grid=grid, resolution=settings.resolution, origin=origin, walls=walls, scores=scores, contrast=contrast,
        start=geom.cell_center(*lv.start),
        goal=Circle(*geom.cell_center(*lv.goal), lv.goal_radius),
        holes=[Circle(*geom.cell_center(r, c), d / 2) for r, c, d in lv.holes],
    )
