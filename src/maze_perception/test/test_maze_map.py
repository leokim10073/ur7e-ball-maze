"""P2 mapping without the sim: geometry against the CAD and sim config, and wall detection on
synthetic rectified images."""

import importlib.util
from dataclasses import replace

import numpy as np
import pytest
import yaml

from conftest import PKG, REPO
from maze_perception.board_pose import board_to_rectified
from maze_perception.config import load_perception_config
from maze_perception.maze_map import FREE, WALL, MazeGeometry, build_board_map, rasterize


def _cad_module(name):
    path = REPO / "maze_CAD" / "scripts" / f"{name}.py"
    if not path.exists():
        pytest.skip(f"{path} not in this checkout")
    spec = importlib.util.spec_from_file_location(f"cad_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cad_walls(level_key, geom):
    """The level's walls from maze_CAD/scripts/levels.py, per grid segment (the test's answer key)."""
    L = _cad_module("levels").LEVELS[level_key]
    out = {}
    for seg in geom.segments():
        kind, line, idx = seg
        lines = L["h_walls"] if kind == "h" else L["v_walls"]
        out[seg] = geom.is_outer(seg) or any(k == line and a <= idx <= b for k, a, b in lines)
    return out


def test_layout_matches_cad(cfg):
    P = _cad_module("maze_params").P
    m = cfg.maze
    assert m.plate == pytest.approx(P["base"] / 1000)
    assert m.margin == pytest.approx(P["margin"] / 1000)
    assert m.wall_thickness == pytest.approx(P["wall_t"] / 1000)
    assert m.layout.goal_radius == pytest.approx(P["goal_d"] / 2000)
    L = _cad_module("levels").LEVELS[m.level]
    assert m.layout.n == L["n"]
    assert m.layout.cell == pytest.approx(L["cell"] / 1000)
    assert m.layout.start == tuple(L["start"])
    assert m.layout.goal == tuple(L["goal"])
    assert [tuple(h) for h in m.layout.holes] == [(r, c, d / 1000) for r, c, d in L["holes"]]


def test_start_and_goal_match_sim(cfg):
    path = REPO / "src" / "maze_sim_description" / "config" / "sim.yaml"
    if not path.exists():
        pytest.skip(f"{path} not in this checkout")
    sim = yaml.safe_load(path.read_text())
    geom = MazeGeometry(cfg.maze)
    np.testing.assert_allclose(geom.cell_center(*cfg.maze.layout.start), sim["ball"]["start"], atol=1e-9)
    np.testing.assert_allclose(geom.cell_center(*cfg.maze.layout.goal), sim["board"]["goal_hole"]["center"], atol=1e-9)
    assert cfg.maze.layout.goal_radius == pytest.approx(sim["board"]["goal_hole"]["radius"])


def test_grid_shape_and_conventions(cfg):
    geom = MazeGeometry(cfg.maze)
    walls = {seg: geom.is_outer(seg) for seg in geom.segments()}  # outer walls only
    grid, origin = rasterize(walls, geom)
    assert grid.shape == (60, 60) and origin == pytest.approx((-0.15, -0.15))
    # lip and outer walls all round, open floor inside
    assert (grid[0] == WALL).all() and (grid[-1] == WALL).all()
    assert (grid[:, 0] == WALL).all() and (grid[:, -1] == WALL).all()
    assert (grid[3:-3, 3:-3] == FREE).all()


def test_rows_count_from_the_bottom(cfg):
    geom = MazeGeometry(cfg.maze)
    top_line = ("h", 1, 2)  # grid line 1 from the TOP, i.e. near +y
    walls = {seg: geom.is_outer(seg) or seg == top_line for seg in geom.segments()}
    grid, origin = rasterize(walls, geom)
    x0, x1, y0, y1 = geom.segment_rect(top_line)
    row = int((((y0 + y1) / 2) - origin[1]) / cfg.maze.resolution)
    col = int((((x0 + x1) / 2) - origin[0]) / cfg.maze.resolution)
    assert row > 30 and grid[row, col] == WALL  # upper half of the board = high row numbers
    assert grid[60 - 1 - row, col] == FREE


def test_l1_walls_where_the_cad_has_them(cfg):
    geom = MazeGeometry(cfg.maze)
    grid, origin = rasterize(cad_walls(cfg.maze.level, geom), geom)
    res = cfg.maze.resolution

    def at(x, y):
        return grid[int((y - origin[1]) / res), int((x - origin[0]) / res)]

    for r, c in [cfg.maze.layout.start, cfg.maze.layout.goal]:
        assert at(*geom.cell_center(r, c)) == FREE
    for seg, present in cad_walls(cfg.maze.level, geom).items():
        x0, x1, y0, y1 = geom.segment_rect(seg)
        assert at((x0 + x1) / 2, (y0 + y1) / 2) == (WALL if present else FREE), seg


def synthetic(cfg, walls, floor=180, line=60, noise=4.0, seed=0):
    """Rectified gray image: uniform floor, a 2 px dark line along one edge of every present wall."""
    rng = np.random.default_rng(seed)
    w, h = cfg.rectified.size
    img = np.full((h, w), float(floor))
    geom = MazeGeometry(cfg.maze)
    for seg, present in walls.items():
        if not present:
            continue
        x0, x1, y0, y1 = geom.segment_rect(seg)
        (c0, r0), (c1, r1) = board_to_rectified(np.array([[x0, y1], [x1, y0]]), cfg.rectified)
        r0, r1, c0, c1 = (int(round(v)) for v in (r0, r1, c0, c1))
        if seg[0] == "h":
            img[r1 - 1:r1 + 1, c0:c1] = line
        else:
            img[r0:r1, c0:c0 + 2] = line
    return np.clip(img + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)


@pytest.mark.parametrize("seed", range(3))
def test_detects_cad_walls_on_synthetic_image(cfg, seed):
    geom = MazeGeometry(cfg.maze)
    truth = cad_walls(cfg.maze.level, geom)
    m = build_board_map(synthetic(cfg, truth, seed=seed), cfg.maze, cfg.rectified)
    assert m is not None
    assert m.walls == truth
    assert m.occupancy_data() == rasterize(truth, geom)[0].ravel().tolist()


def test_ball_on_the_floor_is_not_a_wall(cfg):
    geom = MazeGeometry(cfg.maze)
    truth = cad_walls(cfg.maze.level, geom)
    img = synthetic(cfg, truth)
    # a dark ball sitting right on an empty grid line (between cells (0,0) and (1,0))
    empty = next(seg for seg, p in truth.items() if not p and seg[0] == "h")
    x0, x1, y0, y1 = geom.segment_rect(empty)
    (c, r), = board_to_rectified(np.array([[(x0 + x1) / 2, (y0 + y1) / 2]]), cfg.rectified)
    yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
    img[(xx - c) ** 2 + (yy - r) ** 2 <= 14 ** 2] = 90  # 7 mm radius at 2 px/mm
    assert build_board_map(img, cfg.maze, cfg.rectified).walls == truth


def test_dim_lighting(cfg):
    geom = MazeGeometry(cfg.maze)
    truth = cad_walls(cfg.maze.level, geom)
    dim = synthetic(cfg, truth, floor=70, line=40, noise=2.0)  # dark room: walls only 30 gray levels below the floor
    assert build_board_map(dim, cfg.maze, cfg.rectified).walls == truth


def test_no_board_no_map(cfg):
    w, h = cfg.rectified.size
    blank = np.full((h, w), 128, np.uint8)
    assert build_board_map(blank, cfg.maze, cfg.rectified) is None


def test_given_features(cfg):
    m = build_board_map(synthetic(cfg, cad_walls(cfg.maze.level, MazeGeometry(cfg.maze))), cfg.maze, cfg.rectified)
    assert m.start == pytest.approx((-0.12, 0.12))
    assert (m.goal.x, m.goal.y, m.goal.radius) == pytest.approx((0.12, -0.072, 0.015))
    assert m.holes == []


def test_holes_from_config(cfg):
    layout = replace(cfg.maze.layout, holes=((2, 3, 0.024),))
    settings = replace(cfg.maze, layout=layout)
    geom = MazeGeometry(settings)
    m = build_board_map(synthetic(cfg, cad_walls(cfg.maze.level, geom)), settings, cfg.rectified)
    assert len(m.holes) == 1
    assert (m.holes[0].x, m.holes[0].y) == pytest.approx(geom.cell_center(2, 3))
    assert m.holes[0].radius == pytest.approx(0.012)


def _write(tmp_path, text):
    p = tmp_path / "p.yaml"
    p.write_text(text)
    return p


def test_goal_outside_grid_is_named(tmp_path):
    src = (PKG / "config" / "perception.yaml").read_text().replace("goal: [4, 5]", "goal: [4, 6]")
    with pytest.raises(ValueError, match="maze.levels.L1.goal"):
        load_perception_config(_write(tmp_path, src))


def test_grid_too_big_for_plate(tmp_path):
    src = (PKG / "config" / "perception.yaml").read_text().replace("cell: 0.042", "cell: 0.050")
    with pytest.raises(ValueError, match="does not fit"):
        load_perception_config(_write(tmp_path, src))


def test_unknown_level_is_named(tmp_path):
    src = (PKG / "config" / "perception.yaml").read_text().replace("level: L1", "level: L9")
    with pytest.raises(ValueError, match="maze.level L9"):
        load_perception_config(_write(tmp_path, src))
