"""The ball's center in the board frame from one camera image (P3): a measurement, no filtering.

The image is rectified onto the plane z = ball radius, where the ball's center lies (contract 2),
so a rectified pixel is directly the board (x, y) of a ball centered there.

Each pixel is scored by how far its color is from the floor's: the CIELAB distance from the
floor color (the median over the maze floor; the ball covers a tiny part of it) over `not_floor`,
capped at 1. The cap makes the score about shape, not darkness: a black wall counts no more than
the grey ball. A position scores the mean over a disc of the ball's radius minus the mean over the
ring around it (out to 1.5 radii), both over the floor inside the outer walls only. A ball scores
near 1; a wall line, a wall end or a robot finger reaching over the wall fills only part of the
disc and runs on into the ring. The best position is the ball if it scores at least min_score.

Color distance, not a color threshold: the sim ball is grey on a tan floor, the real ball is steel
on whatever the C922 makes of the real board; neither color is assumed.

Searched at half resolution with square filters (fast), then refined at full resolution with the
disc around the best position, to a fraction of a pixel.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from maze_perception.board_pose import (BoardPose, Camera, board_to_rectified, rectified_coverage,
                                         rectified_to_board, rectify)
from maze_perception.config import BallSettings, RectifiedSettings

RING = 1.5  # the ring around the disc reaches this many ball radii


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


def _not_floor(rgb: np.ndarray, area: np.ndarray, cap: float,
               floor: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """(not_floor, floor): per pixel, CIELAB distance from the floor color over `cap`, at most 1
    (a dark wall counts no more than the ball); 0 outside `area`. The floor color is the median
    over `area` unless given."""
    lab = cv2.cvtColor(np.ascontiguousarray(rgb[..., :3], np.float32) * np.float32(1 / 255), cv2.COLOR_RGB2Lab)
    if floor is None:
        px = lab[area]
        floor = np.median(px[::max(1, len(px) // 5000)], axis=0)  # the ball is far below half of it
    diff = lab - floor.astype(np.float32)
    dist = cv2.sqrt(cv2.transform(diff * diff, np.full((1, 3), 1 / cap ** 2, np.float32)))
    np.minimum(dist, 1, out=dist)
    dist[~area] = 0
    return dist, floor


def _masked_means(sums, counts):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(counts > 0.5, sums / np.maximum(counts, 1e-6), 0.0)


def _response(dist: np.ndarray, area: np.ndarray, r_px: float) -> np.ndarray:
    """Mean over the ball's disc minus mean over the ring around it, at every pixel; both over
    the floor area only (beyond the walls is neither floor nor ball)."""
    a = area.astype(np.float32)
    disc, ring = _disc(r_px), _ring(r_px, RING * r_px)
    inside = _masked_means(_sum_filter(dist, disc), _sum_filter(a, disc))
    around = _masked_means(_sum_filter(dist, ring), _sum_filter(a, ring))
    return inside - around


def _box_response(dist: np.ndarray, area: np.ndarray, r_px: float) -> np.ndarray:
    """_response with squares of the same areas: much faster, for the coarse search only."""
    a = max(1, int(round(r_px * np.sqrt(np.pi))) | 1)  # odd sides, centered
    b = max(a + 2, int(round(RING * r_px * np.sqrt(np.pi))) | 1)

    def box(img, k):
        return cv2.boxFilter(img, cv2.CV_32F, (k, k), normalize=False, borderType=cv2.BORDER_CONSTANT)

    m = area.astype(np.float32)
    d_in, d_out, n_in, n_out = box(dist, a), box(dist, b), box(m, a), box(m, b)
    return _masked_means(d_in, n_in) - _masked_means(d_out - d_in, n_out - n_in)


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


def find_ball(rgb: np.ndarray, settings: BallSettings, rectified: RectifiedSettings,
              valid: Optional[np.ndarray] = None, floor: Optional[np.ndarray] = None) -> Optional[BallDetection]:
    """The ball in an RGB image rectified at z = settings.radius (rectify(..., z=radius)).

    valid: rectified_coverage() at the same z; the ball is found only with its whole disc seen.
    floor: floor_outline() at the same z (default: settings.floor, the walls' inner faces, exact for a camera
    straight above them). Only the floor inside it counts: the outer walls' dark faces next to a
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
    if not area.any():
        return None
    m = settings.radius  # the center is a radius inside the walls, its disc fully seen
    search = _box_mask((x0 + m, x1 - m, y0 + m, y1 - m), rectified, (h, w))

    # coarse: half resolution
    def half(mask):
        return cv2.resize(mask.astype(np.uint8), (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)

    small = cv2.resize(rgb[..., :3], (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    small_area = half(area).astype(bool)
    small_search = half(search).astype(bool) & cv2.erode(half(valid), _disc(r_px / 2 + 1).astype(np.uint8)).astype(bool)
    if not small_search.any():
        return None
    dist, floor = _not_floor(small, small_area, settings.not_floor)
    resp = np.where(small_search, _box_response(dist, small_area, r_px / 2), -np.inf)
    r, c = np.unravel_index(int(np.argmax(resp)), resp.shape)

    # fine: full resolution in a window around it
    r, c = 2 * r + 1, 2 * c + 1
    pad = int(np.ceil(RING * r_px)) + 4
    rows = slice(max(0, r - pad - 2), min(h, r + pad + 3))
    cols = slice(max(0, c - pad - 2), min(w, c + pad + 3))
    dist, _ = _not_floor(rgb[rows, cols], area[rows, cols], settings.not_floor, floor)
    resp = _response(dist, area[rows, cols], r_px)
    near = np.zeros_like(search[rows, cols])
    near[max(0, r - rows.start - 2):r - rows.start + 3, max(0, c - cols.start - 2):c - cols.start + 3] = True
    seen = cv2.erode(valid[rows, cols].astype(np.uint8), _disc(r_px).astype(np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    resp_s = np.where(near & search[rows, cols] & seen, resp, -np.inf)
    if not np.isfinite(resp_s).any():
        return None
    rr, cc = np.unravel_index(int(np.argmax(resp_s)), resp_s.shape)
    score = float(resp[rr, cc])
    if score < settings.min_score:
        return None
    col, row = _subpixel(resp, rr, cc)
    col, row = col + cols.start, row + rows.start
    x, y = rectified_to_board(np.array([[col, row]]), rectified)[0]
    return BallDetection(x=float(x), y=float(y), score=score, pixel=(col, row))


def locate_ball(image: np.ndarray, pose: BoardPose, camera: Camera, settings: BallSettings,
                rectified: RectifiedSettings) -> Optional[BallDetection]:
    """The ball from a camera image (RGB) and the board pose found in it."""
    z = settings.radius
    rgb = rectify(image, pose, camera, rectified, z)
    return find_ball(rgb, settings, rectified, rectified_coverage(pose, camera, rectified, z),
                     floor_outline(pose, camera, settings, z))
