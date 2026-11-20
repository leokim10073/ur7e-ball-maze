"""P4 ball filter without ROS or the sim: synthetic measurement sequences at 60 Hz."""

from dataclasses import replace

import numpy as np
import pytest

from conftest import PKG
from maze_perception.ball_filter import BallFilter
from maze_perception.config import load_perception_config

DT = 1 / 60


def track(f, positions, t0=0.0, noise=0.0005, seed=0, drop=()):
    """Feed (x, y) per frame (None = not found); the estimates."""
    rng = np.random.default_rng(seed)
    out = []
    for k, p in enumerate(positions):
        m = None if p is None or k in drop else tuple(np.asarray(p) + rng.normal(0, noise, 2))
        out.append(f.step(t0 + k * DT, m))
    return out


def test_first_frame_starts_the_track(cfg):
    f = BallFilter(cfg.tracking)
    e = f.step(1.0, (0.01, -0.02))
    assert e.tracking_ok and (e.x, e.y, e.vx, e.vy) == (0.01, -0.02, 0.0, 0.0)
    assert e.stamp == 1.0 and e.position_covariance[0] > 0


def test_nothing_seen_yet_is_unknown(cfg):
    f = BallFilter(cfg.tracking)
    e = f.step(1.0, None)
    assert not e.tracking_ok and e.position_covariance == (0.0, 0.0, 0.0, 0.0)


def test_constant_velocity(cfg):
    v = np.array([0.15, -0.08])
    est = track(BallFilter(cfg.tracking), [v * k * DT for k in range(60)])
    e = est[-1]
    assert e.tracking_ok
    assert np.hypot(e.x - v[0] * 59 * DT, e.y - v[1] * 59 * DT) < 0.001
    assert np.hypot(e.vx - v[0], e.vy - v[1]) < 0.02  # m/s, after 1 s of 0.5 mm noise


def test_ball_at_rest_stays_put(cfg):
    # 1 mm of white noise per frame (worse than the sim's): the speed jitters, a few cm/s at most
    est = track(BallFilter(cfg.tracking), [(0.05, 0.05)] * 600, noise=0.001)
    assert all(e.tracking_ok for e in est)
    speed = np.array([np.hypot(e.vx, e.vy) for e in est[60:]])
    assert np.sqrt(np.mean(speed ** 2)) < 0.04 and speed.max() < 0.12
    assert np.hypot(est[-1].x - 0.05, est[-1].y - 0.05) < 0.002


def test_dropout_predicts_and_flags(cfg):
    v = np.array([0.2, 0.0])
    path = [v * k * DT for k in range(60)]
    est = track(BallFilter(cfg.tracking), path, drop=range(40, 46), noise=0.0)
    for k in range(40, 46):  # ball not found: the prediction, not ok, less sure every frame
        assert not est[k].tracking_ok
        assert est[k].x == pytest.approx(path[k][0], abs=0.002)
        assert est[k].position_covariance[0] > est[k - 1].position_covariance[0]
    assert est[46].tracking_ok


def test_forgets_after_lost_reset(cfg):
    f = BallFilter(cfg.tracking)
    track(f, [(0.0, 0.0)] * 10)
    n = int(cfg.tracking.lost_reset / DT) + 2
    est = track(f, [None] * n, t0=10 * DT)
    assert not est[-1].tracking_ok and est[-1].position_covariance == (0.0, 0.0, 0.0, 0.0)
    e = f.step(10 * DT + n * DT, (0.1, 0.1))  # seen again: a new track there
    assert e.tracking_ok and (e.x, e.y) == (0.1, 0.1)


def test_single_misdetection_is_not_used(cfg):
    path = [(0.0, 0.0)] * 30
    path[20] = (0.08, 0.0)  # e.g. a wall end once mistaken for the ball
    est = track(BallFilter(cfg.tracking), path)
    assert not est[20].tracking_ok and abs(est[20].x) < 0.002
    assert est[21].tracking_ok and abs(est[29].x) < 0.002


def test_ball_really_moved_restarts_the_track(cfg):
    # picked up and put down elsewhere: after max_rejects frames the filter believes it
    path = [(0.0, 0.0)] * 20 + [(0.1, -0.05)] * 10
    est = track(BallFilter(cfg.tracking), path, noise=0.0)
    k = 20 + cfg.tracking.max_rejects - 1
    assert not est[k - 1].tracking_ok
    assert est[k].tracking_ok and (est[k].x, est[k].y) == pytest.approx((0.1, -0.05))


def test_wall_stop_is_followed(cfg):
    # rolling at 0.3 m/s into a wall: the stop lands ~5 mm off the prediction; it must be used
    v = 0.3
    xs = [min(v * k * DT, 0.05) for k in range(60)]
    est = track(BallFilter(cfg.tracking), [(x, 0.0) for x in xs], noise=0.0003)
    assert all(e.tracking_ok for e in est)
    stop = next(k for k, x in enumerate(xs) if x == 0.05)
    assert all(abs(e.vx) < 0.05 for e in est[stop + 5:])  # settled within 5 frames (1/12 s)
    assert max(e.x for e in est) < 0.05 + 0.004  # a few mm through the wall for a frame or two at most


def test_old_frames_are_skipped(cfg):
    f = BallFilter(cfg.tracking)
    f.step(1.0, (0.0, 0.0))
    before = f.estimate(True)
    assert f.step(1.0, (0.05, 0.0)) is None and f.step(0.9, (0.05, 0.0)) is None
    assert f.estimate(True) == before


def test_covariance_stays_symmetric_positive(cfg):
    f = BallFilter(cfg.tracking)
    track(f, [(0.001 * k, 0.0) if k % 7 else None for k in range(300)], noise=0.001)
    P = f.P
    np.testing.assert_allclose(P, P.T, atol=1e-15)
    assert np.all(np.linalg.eigvalsh(P) > 0)


def test_tracking_settings_are_checked(tmp_path):
    src = (PKG / "config" / "perception.yaml").read_text()
    for old, new, msg in [("max_rejects: 3", "max_rejects: 0", "tracking.max_rejects"),
                          ("lost_reset: 1.0", "lost_reset: -1", "tracking.lost_reset"),
                          ("  max_jump: 0.02", "", "max_jump")]:
        assert old in src
        p = tmp_path / "p.yaml"
        p.write_text(src.replace(old, new))
        with pytest.raises(ValueError, match=msg):
            load_perception_config(p)


def test_noise_is_configurable(cfg):
    # a filter with more measurement noise trusts the measurements less: smoother, slower
    calm = replace(cfg.tracking, measurement_noise=0.005)
    a = track(BallFilter(cfg.tracking), [(0.0, 0.0)] * 5 + [(0.003, 0.0)] * 5, noise=0.0)
    b = track(BallFilter(calm), [(0.0, 0.0)] * 5 + [(0.003, 0.0)] * 5, noise=0.0)
    assert a[6].x > b[6].x
