"""P6 lab kit without the lab: p6_check's arithmetic, the camera config files, the scripts' syntax."""

import shutil
import subprocess

import numpy as np
import pytest
import yaml

from conftest import PKG, REPO
from maze_perception.p6_check import (GAP_MAX, cad_wall_rects, check_spots, corner_spots, draw_spots, frame_stats,
                                      latency_stats)

R = 0.0125  # the 25 mm marble (#19)


def test_frame_stats():
    t = list(np.arange(120) / 60.0)
    t[60:] = [x + 0.25 for x in t[60:]]  # one 0.27 s stall
    s = frame_stats(t)
    assert s["frames"] == 120 and s["gaps_over"] == 1
    assert s["gap_median"] == pytest.approx(1 / 60) and s["gap_max"] == pytest.approx(1 / 60 + 0.25)
    assert frame_stats([1.0]) == {"frames": 1.0}
    assert GAP_MAX == 0.2  # #45: count gaps over 0.2 s


def test_latency_stats():
    s = latency_stats([0.010, 0.012, 0.011, 0.060])
    assert s["latency_median"] == pytest.approx(0.0115) and s["latency_max"] == 0.060
    assert latency_stats([]) == {}


@pytest.fixture(scope="module")
def walls():
    cad = REPO / "maze_CAD" / "scripts"
    if not cad.exists():
        pytest.skip("maze_CAD/scripts not in this checkout")
    pytest.importorskip("maze_cad_map")
    return cad_wall_rects(cad, "L1")


def test_corner_spots_touch_two_walls_and_no_more(walls):
    def gap(x, y, r):
        x0, x1, y0, y1 = r
        return np.hypot(max(x0 - x, 0, x - x1), max(y0 - y, 0, y - y1))

    spots = corner_spots(walls, R)
    assert len(spots) >= 5
    for x, y in spots:
        gaps = sorted(gap(x, y, w) for w in walls)
        assert gaps[0] == pytest.approx(R, abs=1e-6) and gaps[1] == pytest.approx(R, abs=1e-6)  # two walls
        assert all(g >= R - 1e-6 for g in gaps)  # clear of every wall


def test_five_test_spots(walls, cfg):
    named = check_spots(corner_spots(walls, R))
    assert [n for n, _ in named][:4] == ["top-left (-x, +y)", "top-right (+x, +y)", "bottom-right (+x, -y)",
                                         "bottom-left (-x, -y)"]
    # the maze's corners: a ball radius inside the outer walls' inner faces (ball.floor)
    c = cfg.ball.floor[1] - R
    np.testing.assert_allclose([p for _, p in named[:4]], [(-c, c), (c, c), (c, -c), (-c, -c)], atol=1e-6)
    assert np.hypot(*named[4][1]) < 0.06  # the middle one
    assert len({p for _, p in named}) == 5


def test_spot_map(walls, tmp_path):
    named = check_spots(corner_spots(walls, R))
    draw_spots(walls, named, tmp_path / "spots.png")
    assert (tmp_path / "spots.png").stat().st_size > 1000


def test_c922_config_matches_the_sim_camera():
    params = yaml.safe_load((PKG / "config" / "c922.yaml").read_text())["/**"]["ros__parameters"]
    assert (params["image_width"], params["image_height"], params["framerate"]) == (1280, 720, 60.0)
    assert params["frame_id"] == "camera_optical_frame" and params["pixel_format"] == "mjpeg2rgb"
    info = yaml.safe_load((PKG / "config" / "c922_nominal.yaml").read_text())
    assert (info["image_width"], info["image_height"]) == (1280, 720)
    K = np.array(info["camera_matrix"]["data"]).reshape(3, 3)
    sim = REPO / "src" / "maze_sim_description" / "config" / "sim.yaml"
    if sim.exists():
        hfov = yaml.safe_load(sim.read_text())["camera"]["hfov_deg"]
        assert K[0, 0] == pytest.approx(640 / np.tan(np.radians(hfov) / 2), rel=1e-4)
    P = np.array(info["projection_matrix"]["data"]).reshape(3, 4)
    np.testing.assert_allclose(P[:, :3], K)


def test_launch_files_compile():
    for f in (PKG / "launch").glob("*.launch.py"):
        compile(f.read_text(), str(f), "exec")


@pytest.mark.skipif(shutil.which("bash") is None, reason="no bash")
@pytest.mark.parametrize("script", ["c922_setup.sh", "preflight.sh", "record.sh"])
def test_script_syntax(script):
    path = PKG / "scripts" / script
    assert path.read_text().startswith("#!/usr/bin/env bash")
    subprocess.run(["bash", "-n", str(path)], check=True)
