"""P3 ball detection without the sim: synthetic rectified images (z = ball radius)."""

import importlib.util
from dataclasses import replace

import numpy as np
import pytest

from conftest import PKG, REPO
from maze_perception.ball import WallGrid, find_ball, wall_mask
from maze_perception.board_pose import board_to_rectified
from maze_perception.config import load_perception_config

TAN, GREY, BLACK = (209, 184, 140), (199, 199, 211), (20, 15, 10)


def _cad_module(name):
    path = REPO / "maze_CAD" / "scripts" / f"{name}.py"
    if not path.exists():
        pytest.skip(f"{path} not in this checkout")
    spec = importlib.util.spec_from_file_location(f"cad_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cad_walls(level="L1"):
    """{name: (x_min, x_max, y_min, y_max)} of every wall of the level in the board frame, m, from
    maze_CAD/scripts (the files the sim and the print are made from)."""
    params, levels = _cad_module("maze_params"), _cad_module("levels")
    half = params.P["base"] / 2
    return {name: ((x - half) / 1000, (x + w - half) / 1000, (y - half) / 1000, (y + h - half) / 1000)
            for name, (x, y, w, h) in params.wall_rects_named(levels.LEVELS[level])}


def cad_wall_grid(level="L1"):
    """The walls grid maze_cad_map publishes on /maze/map, as the ball finder takes it."""
    cad_map = pytest.importorskip("maze_cad_map.cad_map")
    m = cad_map.build_map(REPO / "maze_CAD" / "scripts", level)
    return WallGrid(cells=m.grid != 0, origin=m.origin, resolution=m.resolution)


def test_floor_matches_cad(cfg):
    # ball.floor is the area inside the outer walls of every level
    for level in _cad_module("levels").LEVELS:
        walls = cad_walls(level)
        inside = (walls["outer_left"][1], walls["outer_right"][0], walls["outer_bottom"][3], walls["outer_top"][2])
        assert inside == pytest.approx(cfg.ball.floor, abs=1e-9), level


def scene(cfg, ball=None, floor=TAN, ball_rgb=GREY, wall_rgb=BLACK, noise=3.0, seed=0, walls=True):
    """Rectified RGB image: floor color, the L1 inner walls as 3 px dark lines along one edge of
    their footprint, dark outside the floor (the outer walls' faces), and a shaded ball at `ball`."""
    rng = np.random.default_rng(seed)
    w, h = cfg.rectified.size
    img = np.empty((h, w, 3))
    img[:] = floor

    def px(rect):
        x0, x1, y0, y1 = rect
        (a, b), (c, d) = board_to_rectified(np.array([[x0, y1], [x1, y0]]), cfg.rectified)
        return tuple(int(round(v)) for v in (a, b, c, d))  # col0, row0, col1, row1

    c0, r0, c1, r1 = px(cfg.ball.floor)
    img[:r0, :], img[r1 + 1:, :], img[:, :c0], img[:, c1 + 1:] = wall_rgb, wall_rgb, wall_rgb, wall_rgb
    if walls:
        for name, rect in cad_walls().items():
            if name.startswith("outer"):
                continue
            a, b, c, d = px(rect)
            if c - a > d - b:  # horizontal: its lower edge
                img[d - 2:d + 1, a:c] = wall_rgb
            else:  # vertical: its left edge
                img[b:d, a:a + 3] = wall_rgb
    if ball is not None:
        (bc, br), = board_to_rectified(np.array([ball]), cfg.rectified)
        rad = cfg.ball.radius * cfg.rectified.px_per_m
        yy, xx = np.mgrid[:h, :w]
        rr = np.hypot(xx - bc, yy - br)
        shade = np.clip(1.1 - 0.4 * (rr / rad) ** 2, 0.6, 1.1)[..., None]  # bright top, darker rim
        cover = np.clip(rad + 0.5 - rr, 0, 1)[..., None]  # antialiased edge
        img = img * (1 - cover) + cover * np.asarray(ball_rgb) * shade
    return np.clip(img + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)


def _err_mm(found, true):
    return 1000 * np.hypot(found.x - true[0], found.y - true[1])


@pytest.mark.parametrize("ball", [(-0.12, 0.12), (0.0312, -0.0473), (0.1017, 0.0021), (-0.0655, -0.1099)])
def test_finds_the_ball(cfg, ball):
    b = find_ball(scene(cfg, ball), cfg.ball, cfg.rectified)
    assert b is not None and _err_mm(b, ball) < 0.3
    assert b.score > 0.7  # 1 - the walls in the ring


def test_ball_against_walls(cfg):
    # touching the left outer wall, and an inner horizontal wall from below
    x0 = cfg.ball.floor[0]
    r = cfg.ball.radius
    for ball in [(x0 + r, 0.08), (x0 + r, -0.13)]:
        b = find_ball(scene(cfg, ball), cfg.ball, cfg.rectified)
        assert b is not None and _err_mm(b, ball) < 0.5, ball
    name, (sx0, sx1, sy0, sy1) = next((k, v) for k, v in cad_walls().items()
                                      if k.startswith("h_") and v[1] - v[0] > 0.05)
    ball = ((sx0 + sx1) / 2, sy0 - r)  # just below it
    b = find_ball(scene(cfg, ball), cfg.ball, cfg.rectified)
    assert b is not None and _err_mm(b, ball) < 1.0, name


def test_no_ball(cfg):
    assert find_ball(scene(cfg), cfg.ball, cfg.rectified) is None


def test_dark_blob_reaching_over_the_wall_is_not_a_ball(cfg):
    # a gripper finger over the left wall: black, 20 mm long, 7 mm into the floor
    img = scene(cfg)
    (c, r), = board_to_rectified(np.array([[cfg.ball.floor[0], 0.02]]), cfg.rectified)
    img[int(r) - 20:int(r) + 20, :int(c) + 14] = BLACK
    assert find_ball(img, cfg.ball, cfg.rectified) is None


@pytest.mark.parametrize("floor,ball_rgb", [((230, 230, 225), (70, 70, 75)),  # dark steel on a light floor
                                            ((60, 50, 45), (200, 200, 205)),  # bright ball on a dark floor
                                            ((40, 120, 60), (180, 180, 180))])  # grey ball on a green print
def test_no_color_assumed(cfg, floor, ball_rgb):
    ball = (0.05, 0.05)
    b = find_ball(scene(cfg, ball, floor=floor, ball_rgb=ball_rgb), cfg.ball, cfg.rectified)
    assert b is not None and _err_mm(b, ball) < 0.3


def test_ball_partly_outside_the_camera_image(cfg):
    ball = (-0.12, 0.12)
    img = scene(cfg, ball)
    (c, _), = board_to_rectified(np.array([ball]), cfg.rectified)
    valid = np.ones(img.shape[:2], bool)
    valid[:, :int(c)] = False  # the left half of the ball is not seen
    img[~valid] = 0
    assert find_ball(img, cfg.ball, cfg.rectified, valid) is None
    # the same cut away from the ball changes nothing
    ball = (0.05, 0.05)
    img = scene(cfg, ball)
    img[~valid] = 0
    b = find_ball(img, cfg.ball, cfg.rectified, valid)
    assert b is not None and _err_mm(b, ball) < 0.3


def test_camera_noise(cfg):
    ball = (0.0312, -0.0473)
    for seed in range(3):
        b = find_ball(scene(cfg, ball, noise=8.0, seed=seed), cfg.ball, cfg.rectified)
        assert b is not None and _err_mm(b, ball) < 0.5


def test_ball_settings_are_checked(tmp_path):
    src = (PKG / "config" / "perception.yaml").read_text()
    for old, new, msg in [("min_score: 0.5", "min_score: 1.5", "ball.min_score"),
                          ("not_floor: 20.0", "not_floor: 0", "ball.radius, ball.wall_height and ball.not_floor"),
                          ("radius: 0.007 ", "radius: 0.001 ", "ball.radius is 2.0 px"),
                          ("floor: [-0.141, 0.141,", "floor: [-0.2, 0.141,", "inside the rectified area"),
                          ("floor: [-0.141, 0.141,", "floor: [0.141, -0.141,", "ball.floor x")]:
        p = tmp_path / "p.yaml"
        assert old in src
        p.write_text(src.replace(old, new))
        with pytest.raises(ValueError, match=msg):
            load_perception_config(p)


def test_settings_match_sim(cfg):
    import yaml

    path = REPO / "src" / "maze_sim_description" / "config" / "sim.yaml"
    if not path.exists():
        pytest.skip(f"{path} not in this checkout")
    assert cfg.ball.radius == pytest.approx(yaml.safe_load(path.read_text())["ball"]["radius"])


def test_lower_threshold_still_one_answer(cfg, camera):
    # with no threshold the best guess on an empty board is still reported with its low score
    walls = wall_mask(_top_down(), camera, cad_wall_grid(), cfg.ball, cfg.rectified, cfg.ball.radius)
    b = find_ball(scene(cfg), replace(cfg.ball, min_score=-1.0), cfg.rectified, walls=walls)
    assert b is not None and b.score < 0.3


def test_wall_height_matches_cad(cfg):
    assert cfg.ball.wall_height == pytest.approx(_cad_module("maze_params").P["wall_h"] / 1000)


def _top_down(R_t=None, height=1.0):
    from conftest import board_in_camera
    from maze_perception.board_pose import BoardPose

    R, t = board_in_camera(height=height) if R_t is None else R_t
    return BoardPose(R, t, (0, 1, 2, 3), 0.0)


def test_wall_mask_covers_the_walls_as_seen(cfg, camera):
    # camera straight above the board center: a wall's footprint, and its top spreading outward
    # from the center by height x tan(view angle); nothing on the open floor
    walls = cad_wall_grid()
    pose = _top_down()
    mask = wall_mask(pose, camera, walls, cfg.ball, cfg.rectified, cfg.ball.radius)
    for name, (x0, x1, y0, y1) in cad_walls().items():
        (c, r), = board_to_rectified(np.array([[(x0 + x1) / 2, (y0 + y1) / 2]]), cfg.rectified)
        assert mask[int(r), int(c)], name
    (c, r), = board_to_rectified(np.array([[-0.12, 0.12]]), cfg.rectified)  # the start cell's center
    assert not mask[int(r), int(c)]
    assert 0.15 < mask.mean() < 0.4


def test_wall_end_is_not_the_ball(cfg, camera):
    # a dark blob filling the end of a wall (its end face and top, as the sim shows it): left out
    # with the walls
    walls = cad_wall_grid()
    pose = _top_down()
    name, (x0, x1, y0, y1) = next((k, v) for k, v in cad_walls().items() if k.startswith("v_"))
    img = scene(cfg)
    (c, r), = board_to_rectified(np.array([[(x0 + x1) / 2, y0 + 0.003]]), cfg.rectified)
    yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
    img[np.hypot(xx - c, yy - r) <= 6] = BLACK  # 3 mm: the wall's half width
    best = find_ball(img, replace(cfg.ball, min_score=-1.0), cfg.rectified,
                     walls=wall_mask(pose, camera, walls, cfg.ball, cfg.rectified, cfg.ball.radius))
    assert best is None or best.score < 0.3, name


@pytest.mark.parametrize("noise", [10.0, 20.0])
def test_heavy_camera_noise(cfg, noise):
    ball = (0.0312, -0.0273)
    for seed in range(3):
        b = find_ball(scene(cfg, ball, noise=noise, seed=seed), cfg.ball, cfg.rectified)
        assert b is not None and _err_mm(b, ball) < 0.5


@pytest.mark.parametrize("quality", [50, 80])
def test_jpeg(cfg, quality):
    # the C922 sends MJPEG: color at half resolution blurs the ball's edge
    import cv2

    ball = (0.0312, -0.0273)
    img = scene(cfg, ball)
    img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1], cv2.IMREAD_UNCHANGED)
    b = find_ball(img, cfg.ball, cfg.rectified)
    assert b is not None and _err_mm(b, ball) < 0.5 and b.score > 0.6



MARBLE = 0.0125  # the 25 mm glass marble (#19)


@pytest.mark.parametrize("color,tol", [((40, 70, 170), 0.3),  # blue, as #19 suggests: placed by lightness
                                       ((199, 199, 211), 0.6)])  # grey on tan: placed by color
def test_marble_under_jpeg(cfg, color, tol):
    # MJPEG keeps color at half resolution and a pixel or so off; that pulled a blue ball 1-2 mm
    import cv2

    big = replace(cfg, ball=replace(cfg.ball, radius=MARBLE))
    ball = (0.05, 0.10)
    for quality in (50, 80):
        img = scene(big, ball, ball_rgb=color)
        img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1], cv2.IMREAD_UNCHANGED)
        b = find_ball(img, big.ball, big.rectified)
        assert b is not None and _err_mm(b, ball) < tol, quality


def test_ball_resolution(cfg):
    from maze_perception.ball import BALL_PX, ball_resolution

    assert ball_resolution(cfg.ball, cfg.rectified) == cfg.rectified  # the 14 mm sim ball: full 2 px/mm
    marble = ball_resolution(replace(cfg.ball, radius=MARBLE), cfg.rectified)
    assert marble.px_per_m * MARBLE == pytest.approx(BALL_PX) and marble.x_range == cfg.rectified.x_range
