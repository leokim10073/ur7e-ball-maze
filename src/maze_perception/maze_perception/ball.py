"""The ball's center in the board frame from one camera image (P3): a measurement, no filtering.

The image is rectified onto the plane z = ball radius, where the ball's center lies (contract 2),
so a rectified pixel is directly the board (x, y) of a ball centered there. It is smoothed a little
(0.5 mm, against camera noise). Anything at floor level (the ball's shadow, a wall's foot) shows
shifted toward the camera's nadir in this image, by radius / camera height times its distance
from the nadir: 3-4 mm at the board's edge for the 25 mm ball.

Only the floor counts: inside the outer walls, inside the camera image, and, with the maze map
(/maze/map, contract 1), not where a wall shows (from the ball's center height to the wall's top,
as the camera sees it; lower down, a ball against the wall hides it). The ball is never in a
wall; wall ends and blurred wall lines would look like it.

Each floor pixel is scored by how unlike the floor it is: its CIELAB distance from the floor
color (the median over the floor; the ball covers a tiny part of it), or from the floor in
shadow when it is darker (down to 40% of the floor's light), less twice the floor's own spread
(camera noise, JPEG, uneven light), over `not_floor`, capped at 1. The cap makes it about shape,
not darkness: a black wall counts no more than the grey ball.

The ball is where a disc of its radius is most unlike the floor compared with the ring around it
(found at half resolution with square filters, then at full resolution with the disc, to a
fraction of a pixel). Whether it is the ball is decided at that spot on the ball's core (0.8
radii) against the ring 1.2 to 1.7 radii out, leaving out the edge band that blur, JPEG's half-
resolution color and motion smear: a ball scores about 0.8, at least `min_score` counts. Most of
the disc and some of the ring must be floor (not wall, not off the image) for a spot to count.
The ring leaves out where the ball's own shadow shows (shadow_offset()); counted, the part of
it beside the ball pulled the ball 1-3 mm toward the nadir.

Color distance, not a color threshold: the sim ball is grey on a tan floor, the real ball is steel
on whatever the C922 makes of the real board; neither color is assumed.
"""

from dataclasses import dataclass, replace
from typing import Optional, Tuple

import cv2
import numpy as np

from maze_perception.board_pose import (BoardPose, Camera, _rectified_to_board_matrix, board_to_rectified,
                                         rectified_coverage, rectified_to_board, rectify)
from maze_perception.config import BallSettings, RectifiedSettings

RING = 1.5  # the ring around the disc reaches this many ball radii
BALL_PX = 14.0  # largest ball radius in pixels the finder works at (the 14 mm sim ball at 2 px/mm)
SMOOTH = 0.5  # mm, Gaussian sigma applied to the rectified image first: averages out camera noise
NOISE_MARGIN = 2.0  # floor spreads (median distance) taken off every pixel's distance
# The score that decides "ball or not" leaves out a band at the ball's edge, inside and out: blur,
# JPEG's half-resolution color and motion smear the edge, and should not cost the ball its score.
SCORE_CORE = 0.8  # radii: the solid core of the ball
SCORE_RING = (1.2, 1.7)  # radii: the floor around it
# Least share of the disc (and of the ring) that must be floor (not a wall, not off the image) for a
# position to count: the ball is never inside a wall, and in a corner a quarter of its ring is left.
MIN_DISC_SUPPORT, MIN_RING_SUPPORT = 0.8, 0.2
SHADOW_MIN_K = 0.4  # floor darkened down to this share of its light still counts as floor (shadow)
NEAR_PX = 4  # fine search, pixels around the coarse best (the coarse one can sit a shadow's width off)
LIGHTNESS_MIN = 0.6  # the ball's score in lightness alone above which lightness places it


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


@dataclass(frozen=True)
class _Floor:
    rgb_lin: np.ndarray  # median floor color, linear RGB 0..1 (shadows scale it)
    noise: float  # median not-floor distance of the floor itself, CIELAB units
    noise_l: float  # the same in lightness alone


def _to_lin(srgb01: np.ndarray) -> np.ndarray:
    return np.power(np.clip(srgb01, 0, 1), 2.2, dtype=np.float32)


def _floor_distance(rgb: np.ndarray, floor_lin: np.ndarray, lightness_only: bool = False) -> np.ndarray:
    """Per pixel, CIELAB distance from the floor color, or from the floor in shadow (the floor
    color darkened, down to SHADOW_MIN_K of its light) when the pixel is darker: the ball's own
    shadow is floor, not ball. Its shadow lies at floor level, and in the image rectified at the
    ball's center height a floor-level point shows shifted toward the camera's nadir (by
    radius x offset / height, ~3 mm at the board's edge for the 25 mm ball): counted as ball, the
    crescent of it beside the ball pulled the ball's center that way. Black (walls) and dark balls
    stay unlike the floor."""
    p = np.ascontiguousarray(rgb[..., :3], np.float32) * np.float32(1 / 255)
    f = floor_lin.astype(np.float32)
    k = np.clip(_to_lin(p) @ f / float(f @ f), SHADOW_MIN_K, 1.0)[..., None]
    shade = np.power(k * f, 1 / 2.2).astype(np.float32)  # the floor in that much shadow, sRGB
    diff = cv2.cvtColor(p, cv2.COLOR_RGB2Lab) - cv2.cvtColor(shade, cv2.COLOR_RGB2Lab)
    weights = np.array([[1, 0, 0]] if lightness_only else [[1, 1, 1]], np.float32)
    return cv2.sqrt(cv2.transform(diff * diff, weights))


def _floor_model(rgb: np.ndarray, area: np.ndarray) -> _Floor:
    """The floor's color (median over `area`; the ball covers far below half of it) and how much
    the floor itself spreads from it (camera noise, JPEG blocks, uneven light). From a sample."""
    idx = np.flatnonzero(area)
    idx = idx[::max(1, len(idx) // 5000)]
    px = rgb.reshape(-1, 3)[idx][None]
    f = _to_lin(np.median(px[0].astype(np.float32) / 255, axis=0))
    return _Floor(rgb_lin=f, noise=float(np.median(_floor_distance(px, f))),
                  noise_l=float(np.median(_floor_distance(px, f, lightness_only=True))))


def _not_floor(rgb: np.ndarray, area: np.ndarray, cap: float, floor: _Floor,
               lightness_only: bool = False) -> np.ndarray:
    """Per pixel, how unlike the floor (lit or in shadow) it is, 0..1: _floor_distance less
    NOISE_MARGIN times the floor's own spread (so noise alone scores about 0), over `cap`, at most
    1 (a dark wall counts no more than the ball); 0 outside `area`."""
    dist = _floor_distance(rgb, floor.rgb_lin, lightness_only)
    dist = np.clip((dist - NOISE_MARGIN * (floor.noise_l if lightness_only else floor.noise)) / cap, 0, 1)
    dist[~area] = 0
    return dist


def _masked_means(sums, counts):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(counts > 0.5, sums / np.maximum(counts, 1e-6), 0.0)


def _response(dist: np.ndarray, area: np.ndarray, r_px: float, shadow: Tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """Mean over the ball's disc minus mean over the ring around it, at every pixel; both over
    the floor area only (beyond the walls is neither floor nor ball). -1 where too little of the
    disc or ring is floor.

    shadow: (col, row) pixels from the ball's center to where its floor-level shadow shows
    (shadow_offset()). The ring leaves that disc out: whatever shows there, the shadow pulls
    nothing toward it."""
    a = area.astype(np.float32)
    disc, ring = _disc(r_px), _ring(r_px, RING * r_px)
    if shadow != (0.0, 0.0):
        n = ring.shape[0] // 2
        yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
        ring = ring * ((xx - shadow[0]) ** 2 + (yy - shadow[1]) ** 2 > (r_px + 0.5) ** 2)
    n_in, n_out = _sum_filter(a, disc), _sum_filter(a, ring)
    resp = _masked_means(_sum_filter(dist, disc), n_in) - _masked_means(_sum_filter(dist, ring), n_out)
    resp[(n_in < MIN_DISC_SUPPORT * disc.sum()) | (n_out < MIN_RING_SUPPORT * ring.sum())] = -1.0
    return resp


def shadow_offset(x: float, y: float, nadir: Tuple[float, float, float], radius: float,
                  rectified: RectifiedSettings) -> Tuple[float, float]:
    """(col, row) pixels from a ball centered at board (x, y) to its shadow's center as the image
    rectified at the ball's center height shows it. A point on the floor shows shifted toward the
    camera's nadir (nadir = (x, y, height) of the camera over the floor) by radius / height times
    its distance from it: 3-4 mm at the board's edge for the 25 mm ball, 1 m below the camera."""
    nx, ny, height = nadir
    s = radius / height * rectified.px_per_m
    return (nx - x) * s, -(ny - y) * s  # image rows run along board -y


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
    """(4, 2) board (x, y): the outer walls' inner faces as the image rectified at height z shows
    them, from z up. Below z a face shows a little inside it (toward the camera's nadir), but a
    ball against the wall hides that part: leaving it out, as the walls' foot once was, cut the
    ball's outer edge off and pushed it 1-3 mm off the wall."""
    x0, x1, y0, y1 = settings.floor
    face = np.array([[x0, y1, z], [x1, y1, z], [x1, y0, z], [x0, y0, z]])
    return pose.pixel_to_board(pose.board_to_pixel(face, camera), camera, z)[:, :2]


WALL_LEVELS = 5  # heights from the ball's center to the top of a wall at which it is projected


def wall_mask(pose: BoardPose, camera: Camera, walls: WallGrid, settings: BallSettings,
              rectified: RectifiedSettings, z: float) -> np.ndarray:
    """(height, width) bool over the image rectified at height z: where a wall shows, from height z
    (the ball's center) to its top (settings.wall_height), as the camera sees it. Neither floor nor
    ball there, so the ball finder leaves it out: a wall end or a blurred wall line is never taken
    for the ball. Below z, a ball against the wall hides it (see floor_outline)."""
    w, h = rectified.size
    res, (ox, oy) = walls.resolution, walls.origin
    grid_to_board = np.array([[res, 0, ox + res / 2], [0, res, oy + res / 2], [0, 0, 1]])  # cell centers
    board_to_rect = np.linalg.inv(_rectified_to_board_matrix(rectified))
    to_plane_z = np.linalg.inv(pose.plane_homography(camera, z))
    src = walls.cells.astype(np.uint8) * 255
    out = np.zeros((h, w), bool)
    for hz in np.linspace(z, settings.wall_height, WALL_LEVELS):
        M = board_to_rect @ to_plane_z @ pose.plane_homography(camera, hz) @ grid_to_board
        out |= cv2.warpPerspective(src, M, (w, h), flags=cv2.INTER_NEAREST) > 0
    return out


def find_ball(rgb: np.ndarray, settings: BallSettings, rectified: RectifiedSettings,
              valid: Optional[np.ndarray] = None, floor: Optional[np.ndarray] = None,
              walls: Optional[np.ndarray] = None,
              nadir: Optional[Tuple[float, float, float]] = None) -> Optional[BallDetection]:
    """The ball in an RGB image rectified at z = settings.radius (rectify(..., z=radius)).

    valid: rectified_coverage() at the same z; the ball is found only with its whole disc seen.
    floor: floor_outline() at the same z (default: settings.floor, the walls' inner faces; exact
    for a camera straight above them); only the floor inside it counts. walls: wall_mask() at the
    same z, left out too. nadir: the camera's (x, y, height) over the board, to leave the ball's
    shadow out of the ring (shadow_offset()); without it the ball is placed up to a third of
    radius x offset / height toward the nadir. None when nothing scores settings.min_score."""
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
    floor_model = _floor_model(rgb, area)

    # coarse: half resolution
    def half(mask):
        return cv2.resize(mask.astype(np.uint8), (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)

    small = cv2.resize(rgb, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    small_area = half(area).astype(bool)
    small_search = half(search).astype(bool) & cv2.erode(half(valid), _disc(r_px / 2 + 1).astype(np.uint8)).astype(bool)
    if not small_search.any():
        return None
    dist = _not_floor(small, small_area, settings.not_floor, floor_model)
    resp = np.where(small_search, _box_response(dist, small_area, r_px / 2), -np.inf)
    r, c = np.unravel_index(int(np.argmax(resp)), resp.shape)

    # fine: full resolution in a window around it
    r, c = 2 * r + 1, 2 * c + 1
    pad = int(np.ceil(RING * r_px)) + 4 + NEAR_PX
    rows = slice(max(0, r - pad), min(h, r + pad + 1))
    cols = slice(max(0, c - pad), min(w, c + pad + 1))
    dist = _not_floor(rgb[rows, cols], area[rows, cols], settings.not_floor, floor_model)
    shadow = (0.0, 0.0)
    if nadir is not None:
        shadow = shadow_offset(*rectified_to_board(np.array([[c, r]]), rectified)[0], nadir, settings.radius, rectified)
    resp = _response(dist, area[rows, cols], r_px, shadow)
    near = np.zeros_like(search[rows, cols])
    k = NEAR_PX
    near[max(0, r - rows.start - k):r - rows.start + k + 1, max(0, c - cols.start - k):c - cols.start + k + 1] = True
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
    # Placed by lightness when the ball differs from the floor in lightness too: JPEG (the C922's
    # MJPEG) keeps color at half resolution and a pixel or so off, which pulls a colored ball's
    # center by up to 2 mm; lightness keeps full resolution. A grey ball on a light floor (the
    # sim's) differs mostly in color and stays placed by color.
    light = _not_floor(rgb[rows, cols], area[rows, cols], settings.not_floor, floor_model, lightness_only=True)
    if _score_at(light, area[rows, cols], col, row, r_px) >= LIGHTNESS_MIN:
        resp_l = _response(light, area[rows, cols], r_px, shadow)
        rr, cc = np.unravel_index(int(np.argmax(np.where(near & search[rows, cols] & seen, resp_l, -np.inf))),
                                  resp_l.shape)
        col, row = _subpixel(resp_l, rr, cc)
    col, row = col + cols.start, row + rows.start
    x, y = rectified_to_board(np.array([[col, row]]), rectified)[0]
    return BallDetection(x=float(x), y=float(y), score=score, pixel=(col, row))


def ball_resolution(settings: BallSettings, rectified: RectifiedSettings) -> RectifiedSettings:
    """The rectified area at a scale where the ball's radius is at most BALL_PX pixels: enough for a
    fraction of a millimeter, and the filters cost the same for any ball (the 25 mm marble at the
    full 2 px/mm would take 25-35 ms a frame instead of ~10)."""
    px_per_m = min(rectified.px_per_m, BALL_PX / settings.radius)
    return rectified if px_per_m == rectified.px_per_m else replace(rectified, px_per_m=px_per_m)


def locate_ball(image: np.ndarray, pose: BoardPose, camera: Camera, settings: BallSettings,
                rectified: RectifiedSettings, walls: Optional[WallGrid] = None) -> Optional[BallDetection]:
    """The ball from a camera image (RGB) and the board pose found in it; walls: the maze map's
    (/maze/map), if known. Searched in an image rectified at ball_resolution(); the detection's
    pixel is in that image."""
    z = settings.radius
    rectified = ball_resolution(settings, rectified)
    rgb = rectify(image, pose, camera, rectified, z)
    mask = None if walls is None else wall_mask(pose, camera, walls, settings, rectified, z)
    cam = -pose.R.T @ pose.t  # the camera in the board frame
    return find_ball(rgb, settings, rectified, rectified_coverage(pose, camera, rectified, z),
                     floor_outline(pose, camera, settings, z), mask,
                     nadir=(float(cam[0]), float(cam[1]), float(cam[2])))
