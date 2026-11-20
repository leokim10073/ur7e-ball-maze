"""The ball's center in the board frame from one camera image (P3): a measurement, no filtering.

The image is rectified onto the plane z = ball radius, where the ball's center lies (contract 2),
so a rectified pixel is directly the board (x, y) of a ball centered there. It is smoothed a little
(0.5 mm, against camera noise; the ball is 14 mm across).

Only the floor counts: inside the outer walls, inside the camera image, and, with the maze map
(/maze/map, contract 1), not where a wall shows (its footprint up to its top, as the camera sees
it). The ball is never in a wall; wall ends and blurred wall lines would look like it.

Each floor pixel is scored by how unlike the floor it is: its CIELAB distance from the floor color
(the median over the floor; the ball covers a tiny part of it), less twice the floor's own spread
(camera noise, JPEG, uneven light), over `not_floor`, capped at 1. The cap makes it about shape,
not darkness: a black wall counts no more than the grey ball.

The ball is where a disc of its radius is most unlike the floor compared with the ring around it
(found at half resolution with square filters, then at full resolution with the disc, to a
fraction of a pixel). Whether it is the ball is decided at that spot on the ball's core (0.8
radii) against the ring 1.2 to 1.7 radii out, leaving out the edge band that blur, JPEG's half-
resolution color and motion smear: a ball scores about 0.8, at least `min_score` counts. Most of
the disc and some of the ring must be floor (not wall, not off the image) for a spot to count.

Color distance, not a color threshold: the sim ball is grey on a tan floor, the real ball is steel
on whatever the C922 makes of the real board; neither color is assumed.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from maze_perception.board_pose import (BoardPose, Camera, _rectified_to_board_matrix, board_to_rectified,
                                         rectified_coverage, rectified_to_board, rectify)
from maze_perception.config import BallSettings, RectifiedSettings

RING = 1.5  # the ring around the disc reaches this many ball radii
SMOOTH = 0.5  # mm, Gaussian sigma applied to the rectified image first: averages out camera noise
NOISE_MARGIN = 2.0  # floor spreads (median distance) taken off every pixel's distance
# The score that decides "ball or not" leaves out a band at the ball's edge, inside and out: blur,
# JPEG's half-resolution color and motion smear the edge, and should not cost the ball its score.
SCORE_CORE = 0.8  # radii: the solid core of the ball
SCORE_RING = (1.2, 1.7)  # radii: the floor around it
# Least share of the disc (and of the ring) that must be floor (not a wall, not off the image) for a
# position to count: the ball is never inside a wall, and in a corner a quarter of its ring is left.
MIN_DISC_SUPPORT, MIN_RING_SUPPORT = 0.8, 0.2


@dataclass(frozen=True)
class WallGrid:
    """The walls of contract 1's map (/maze/map walls): True = wall, outside the outer walls or
    unknown. Row 0 = the -y edge; cell (row, col) spans origin + (col, row) * resolution."""

    cells: np.ndarray  # (rows, cols) bool
    origin: Tuple[float, float]  # board (x, y) of the grid's lower-left corner, m
    resolution: float  # m per cell


@dataclass(frozen=True)
class BallDetection:
    x: float  # ball center in the board frame, m
    y: float
    score: float  # disc minus ring share of not-floor pixels, 0..1
    pixel: Tuple[float, float]  # (col, row) in the rectified image


def _disc(r: float) -> np.ndarray:
    n = int(np.ceil(r))
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
    return (xx ** 2 + yy ** 2 <= r ** 2).astype(np.float32)


def _sum_filter(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return cv2.filter2D(img, cv2.CV_32F, kernel, borderType=cv2.BORDER_CONSTANT)


def _ring(r_in: float, r_out: float) -> np.ndarray:
    outer = _disc(r_out)
    pad = (outer.shape[0] - _disc(r_in).shape[0]) // 2
    inner = np.pad(_disc(r_in), pad)
    return outer - inner


def _subpixel(resp: np.ndarray, r: int, c: int) -> Tuple[float, float]:
    """Peak of a parabola through the response at the peak and its neighbors, per axis."""
    def offset(a, b, m):
        d = a - 2 * m + b
        return 0.0 if d >= 0 else float(np.clip(0.5 * (a - b) / d, -0.5, 0.5))

    h, w = resp.shape
    dr = offset(resp[r - 1, c], resp[r + 1, c], resp[r, c]) if 0 < r < h - 1 else 0.0
    dc = offset(resp[r, c - 1], resp[r, c + 1], resp[r, c]) if 0 < c < w - 1 else 0.0
    return c + dc, r + dr


def _lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(rgb[..., :3], np.float32) * np.float32(1 / 255), cv2.COLOR_RGB2Lab)


def _floor_model(rgb: np.ndarray, area: np.ndarray) -> Tuple[np.ndarray, float]:
    """(floor color in CIELAB, noise): the median color over `area` (the ball covers far below
    half of it) and the median CIELAB distance of the floor pixels from it, i.e. how much this
    camera's noise, JPEG blocks and uneven light spread the floor's color. From a sample."""
    idx = np.flatnonzero(area)
    idx = idx[::max(1, len(idx) // 5000)]
    lab = _lab(rgb.reshape(-1, 3)[idx][None])[0]
    floor = np.median(lab, axis=0)
    return floor, float(np.median(np.linalg.norm(lab - floor, axis=1)))


def _not_floor(rgb: np.ndarray, area: np.ndarray, cap: float, floor: np.ndarray, noise: float) -> np.ndarray:
    """Per pixel, how unlike the floor it is, 0..1: its CIELAB distance from the floor color, less
    NOISE_MARGIN times the floor's own spread (so noise alone scores about 0), over `cap`, at
    most 1 (a dark wall counts no more than the ball); 0 outside `area`."""
    diff = _lab(rgb) - floor.astype(np.float32)
    dist = cv2.sqrt(cv2.transform(diff * diff, np.ones((1, 3), np.float32)))
    dist = np.clip((dist - NOISE_MARGIN * noise) / cap, 0, 1)
    dist[~area] = 0
    return dist


def _masked_means(sums, counts):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(counts > 0.5, sums / np.maximum(counts, 1e-6), 0.0)


def _response(dist: np.ndarray, area: np.ndarray, r_px: float) -> np.ndarray:
    """Mean over the ball's disc minus mean over the ring around it, at every pixel; both over
    the floor area only (beyond the walls is neither floor nor ball). -1 where too little of the
    disc or ring is floor."""
    a = area.astype(np.float32)
    disc, ring = _disc(r_px), _ring(r_px, RING * r_px)
    n_in, n_out = _sum_filter(a, disc), _sum_filter(a, ring)
    resp = _masked_means(_sum_filter(dist, disc), n_in) - _masked_means(_sum_filter(dist, ring), n_out)
    resp[(n_in < MIN_DISC_SUPPORT * disc.sum()) | (n_out < MIN_RING_SUPPORT * ring.sum())] = -1.0
    return resp


def _box_response(dist: np.ndarray, area: np.ndarray, r_px: float) -> np.ndarray:
    """_response with squares of the same areas: much faster, for the coarse search only."""
    a = max(1, int(round(r_px * np.sqrt(np.pi))) | 1)  # odd sides, centered
    b = max(a + 2, int(round(RING * r_px * np.sqrt(np.pi))) | 1)

    def box(img, k):
        return cv2.boxFilter(img, cv2.CV_32F, (k, k), normalize=False, borderType=cv2.BORDER_CONSTANT)

    m = area.astype(np.float32)
    d_in, d_out, n_in, n_out = box(dist, a), box(dist, b), box(m, a), box(m, b)
    resp = _masked_means(d_in, n_in) - _masked_means(d_out - d_in, n_out - n_in)
    resp[(n_in < MIN_DISC_SUPPORT * a * a) | (n_out - n_in < MIN_RING_SUPPORT * (b * b - a * a))] = -1.0
    return resp


def _score_at(dist: np.ndarray, area: np.ndarray, col: float, row: float, r_px: float) -> float:
    """Mean not-floor over the ball's core minus mean over the floor ring around it (both over
    `area`), centered at (col, row) of these arrays."""
    yy, xx = np.mgrid[:dist.shape[0], :dist.shape[1]]
    rr = np.hypot(xx - col, yy - row) / r_px
    core_all, ring_all = rr <= SCORE_CORE, (rr >= SCORE_RING[0]) & (rr <= SCORE_RING[1])
    core, ring = core_all & area, ring_all & area
    if core.sum() < MIN_DISC_SUPPORT * core_all.sum() or ring.sum() < MIN_RING_SUPPORT * ring_all.sum():
        return -1.0
    return float(dist[core].mean() - dist[ring].mean())


def _box_mask(rect, rectified: RectifiedSettings, shape) -> np.ndarray:
    """Rectified-image pixels whose centers lie in the board rect (x_min, x_max, y_min, y_max)."""
    x0, x1, y0, y1 = rect
    (c0, r0), (c1, r1) = board_to_rectified(np.array([[x0, y1], [x1, y0]]), rectified)
    h, w = shape[:2]
    out = np.zeros((h, w), bool)
    rows = slice(max(0, int(np.ceil(r0))), min(h, int(np.floor(r1)) + 1))
    cols = slice(max(0, int(np.ceil(c0))), min(w, int(np.floor(c1)) + 1))
    out[rows, cols] = True
    return out


def floor_outline(pose: BoardPose, camera: Camera, settings: BallSettings, z: float) -> np.ndarray:
    """(4, 2) board (x, y): where the foot of the outer walls' inner faces shows in the image
    rectified at height z. Off the camera's axis it lies a little inside the walls (z tan of the
    view angle, about 1 mm at the board's edge), the dark faces just outside it."""
    x0, x1, y0, y1 = settings.floor
    foot = np.array([[x0, y1, 0.0], [x1, y1, 0.0], [x1, y0, 0.0], [x0, y0, 0.0]])
    return pose.pixel_to_board(pose.board_to_pixel(foot, camera), camera, z)[:, :2]


WALL_LEVELS = 5  # heights from the foot to the top of a wall at which it is projected


def wall_mask(pose: BoardPose, camera: Camera, walls: WallGrid, settings: BallSettings,
              rectified: RectifiedSettings, z: float) -> np.ndarray:
    """(height, width) bool over the image rectified at height z: where a wall shows, from its foot
    to its top (settings.wall_height), as the camera sees it. Neither floor nor ball there, so
    the ball finder leaves it out: a wall end or a blurred wall line is never taken for the ball."""
    w, h = rectified.size
    res, (ox, oy) = walls.resolution, walls.origin
    grid_to_board = np.array([[res, 0, ox + res / 2], [0, res, oy + res / 2], [0, 0, 1]])  # cell centers
    board_to_rect = np.linalg.inv(_rectified_to_board_matrix(rectified))
    to_plane_z = np.linalg.inv(pose.plane_homography(camera, z))
    src = walls.cells.astype(np.uint8) * 255
    out = np.zeros((h, w), bool)
    for hz in np.linspace(0.0, settings.wall_height, WALL_LEVELS):
        M = board_to_rect @ to_plane_z @ pose.plane_homography(camera, hz) @ grid_to_board
        out |= cv2.warpPerspective(src, M, (w, h), flags=cv2.INTER_NEAREST) > 0
    return out


def find_ball(rgb: np.ndarray, settings: BallSettings, rectified: RectifiedSettings,
              valid: Optional[np.ndarray] = None, floor: Optional[np.ndarray] = None,
              walls: Optional[np.ndarray] = None) -> Optional[BallDetection]:
    """The ball in an RGB image rectified at z = settings.radius (rectify(..., z=radius)).

    valid: rectified_coverage() at the same z; the ball is found only with its whole disc seen.
    floor: floor_outline() at the same z (default: settings.floor, the walls' inner faces; exact
    for a camera straight above them). Only the floor inside it counts: the outer walls' dark faces next to a
    ball would push it away from them. None when nothing scores settings.min_score."""
    h, w = rgb.shape[:2]
    if valid is None:
        valid = np.ones((h, w), bool)
    r_px = settings.radius * rectified.px_per_m
    x0, x1, y0, y1 = settings.floor
    if floor is None:
        area = _box_mask((x0, x1, y0, y1), rectified, (h, w))
    else:
        poly = board_to_rectified(np.asarray(floor, float), rectified)
        area = cv2.fillConvexPoly(np.zeros((h, w), np.uint8), np.round(poly * 16).astype(np.int32), 1,
                                  lineType=cv2.LINE_8, shift=4).astype(bool)
    area &= valid
    if walls is not None:
        area &= ~walls
    if not area.any():
        return None
    m = settings.radius  # the center is a radius inside the walls, its disc fully seen
    search = _box_mask((x0 + m, x1 - m, y0 + m, y1 - m), rectified, (h, w))

    rgb = cv2.GaussianBlur(np.ascontiguousarray(rgb[..., :3]), (0, 0), SMOOTH * rectified.px_per_m / 1000)
    floor_lab, noise = _floor_model(rgb, area)

    # coarse: half resolution
    def half(mask):
        return cv2.resize(mask.astype(np.uint8), (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)

    small = cv2.resize(rgb, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    small_area = half(area).astype(bool)
    small_search = half(search).astype(bool) & cv2.erode(half(valid), _disc(r_px / 2 + 1).astype(np.uint8)).astype(bool)
    if not small_search.any():
        return None
    dist = _not_floor(small, small_area, settings.not_floor, floor_lab, noise)
    resp = np.where(small_search, _box_response(dist, small_area, r_px / 2), -np.inf)
    r, c = np.unravel_index(int(np.argmax(resp)), resp.shape)

    # fine: full resolution in a window around it
    r, c = 2 * r + 1, 2 * c + 1
    pad = int(np.ceil(RING * r_px)) + 4
    rows = slice(max(0, r - pad - 2), min(h, r + pad + 3))
    cols = slice(max(0, c - pad - 2), min(w, c + pad + 3))
    dist = _not_floor(rgb[rows, cols], area[rows, cols], settings.not_floor, floor_lab, noise)
    resp = _response(dist, area[rows, cols], r_px)
    near = np.zeros_like(search[rows, cols])
    near[max(0, r - rows.start - 2):r - rows.start + 3, max(0, c - cols.start - 2):c - cols.start + 3] = True
    seen = cv2.erode(valid[rows, cols].astype(np.uint8), _disc(r_px).astype(np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    resp_s = np.where(near & search[rows, cols] & seen, resp, -np.inf)
    if not np.isfinite(resp_s).any():
        return None
    rr, cc = np.unravel_index(int(np.argmax(resp_s)), resp_s.shape)
    col, row = _subpixel(resp, rr, cc)
    score = _score_at(dist, area[rows, cols], col, row, r_px)
    if score < settings.min_score:
        return None
    col, row = col + cols.start, row + rows.start
    x, y = rectified_to_board(np.array([[col, row]]), rectified)[0]
    return BallDetection(x=float(x), y=float(y), score=score, pixel=(col, row))


def locate_ball(image: np.ndarray, pose: BoardPose, camera: Camera, settings: BallSettings,
                rectified: RectifiedSettings, walls: Optional[WallGrid] = None) -> Optional[BallDetection]:
    """The ball from a camera image (RGB) and the board pose found in it; walls: the maze map's
    (/maze/map), if known."""
    z = settings.radius
    rgb = rectify(image, pose, camera, rectified, z)
    mask = None if walls is None else wall_mask(pose, camera, walls, settings, rectified, z)
    return find_ball(rgb, settings, rectified, rectified_coverage(pose, camera, rectified, z),
                     floor_outline(pose, camera, settings, z), mask)
