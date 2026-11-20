"""Board pose (P1), ball (P3) and ball state (P4) from the sim's own overhead camera image, against
the sim's ground truth.

Renders the MuJoCo scene offline (no ROS), with the board tilted and shifted in its grasp.
Needs mujoco and maze_sim_description (both in the maze-sim container); skipped without them.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("maze_sim_description")

from maze_sim_description.camera_math import look_at_quaternion, pinhole_intrinsics, quat_to_matrix  # noqa: E402
from maze_sim_description.config import load_sim_config  # noqa: E402
from maze_sim_description.scene_builder import build_scene  # noqa: E402

from maze_perception.ball import locate_ball  # noqa: E402
from maze_perception.ball_filter import BallFilter  # noqa: E402
from maze_perception.board_pose import Camera, estimate_board_pose  # noqa: E402
from maze_perception.markers import MarkerDetector  # noqa: E402
from test_ball import cad_walls  # noqa: E402


def _sim_pkg():
    import maze_sim_description

    here = Path(maze_sim_description.__file__).resolve().parents[1]
    if (here / "config" / "sim.yaml").exists():  # source tree
        return here
    from ament_index_python.packages import get_package_share_directory  # installed

    return Path(get_package_share_directory("maze_sim_description"))


@pytest.fixture(scope="module")
def sim():
    pkg = _sim_pkg()
    cfg = load_sim_config(pkg / "config" / "sim.yaml")
    model = mujoco.MjModel.from_xml_string(build_scene(cfg, pkg / "meshes" / "board"))
    w, h = cfg.camera.resolution
    try:
        renderer = mujoco.Renderer(model, h, w)
    except Exception as e:  # no OpenGL context on this machine
        pytest.skip(f"cannot render offscreen: {e}")
    yield cfg, model, renderer
    renderer.close()


def _angle_deg(Ra, Rb):
    return np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))


def _render(sim, seed, ball="start"):
    """(image, data, camera): the sim's overhead image with the board at its start pose (seed 0)
    or tilted up to 6 deg and shifted up to 2 cm in the grasp; the ball at the start cell, resting
    at board (x, y) = `ball`, or (None) off the board."""
    sim_cfg, model, renderer = sim
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("start").id)
    qadr = model.jnt_qposadr[model.body("board").jntadr[0]]
    if seed:  # tilt up to 6 deg about a random horizontal axis, shift up to 2 cm
        rng = np.random.default_rng(seed)
        axis = np.append(rng.normal(size=2), 0.0)
        q = np.zeros(4)
        mujoco.mju_axisAngle2Quat(q, axis / np.linalg.norm(axis), np.radians(rng.uniform(-6, 6)))
        q_new = np.zeros(4)
        mujoco.mju_mulQuat(q_new, q, data.qpos[qadr + 3:qadr + 7].copy())
        data.qpos[qadr + 3:qadr + 7] = q_new
        data.qpos[qadr:qadr + 3] += rng.uniform(-0.02, 0.02, 3)
    if ball != "start":
        mujoco.mj_forward(model, data)
        badr = model.joint("ball_free").qposadr[0]
        if ball is None:
            data.qpos[badr:badr + 3] = (0.0, 0.0, -1.0)
        else:
            R_b = data.body("board").xmat.reshape(3, 3)
            data.qpos[badr:badr + 3] = data.body("board").xpos + R_b @ np.array([*ball, sim_cfg.ball.radius])
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="overhead_camera")
    image = renderer.render()

    w, h = sim_cfg.camera.resolution
    fx, fy, cx, cy = pinhole_intrinsics(w, h, sim_cfg.camera.hfov_deg)
    camera = Camera(K=np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]), D=np.zeros(5), width=w, height=h)
    return image, data, camera


@pytest.mark.parametrize("seed", range(6))
def test_board_pose_matches_ground_truth(sim, cfg, seed):
    sim_cfg = sim[0]
    image, data, camera = _render(sim, seed)
    corners = MarkerDetector(cfg.board.dictionary, ids=cfg.board.markers).detect(image)
    assert sorted(corners) == [0, 1, 2, 3]
    pose = estimate_board_pose(corners, cfg.board, camera, cfg.pose)
    assert pose is not None

    # board in world = camera optical frame in world * board in camera
    R_wc = quat_to_matrix(look_at_quaternion(sim_cfg.camera.position, sim_cfg.camera.look_at, sim_cfg.camera.up))
    p_wb = R_wc @ pose.t + np.asarray(sim_cfg.camera.position)
    R_wb = R_wc @ pose.R
    assert np.linalg.norm(p_wb - data.body("board").xpos) < 1e-3
    assert _angle_deg(R_wb, data.body("board").xmat.reshape(3, 3)) < 0.5

    # the ball's center pixel back onto the board: on the ball's plane (z = ball radius when it
    # rests on the floor; here the board was moved under it, so its own height in the board frame)
    ball_w = data.body("ball").xpos
    truth = data.body("board").xmat.reshape(3, 3).T @ (ball_w - data.body("board").xpos)
    px = camera.K @ (R_wc.T @ (ball_w - np.asarray(sim_cfg.camera.position)))
    found = pose.pixel_to_board(px[:2] / px[2], camera, z=truth[2])[0]
    assert np.linalg.norm(found[:2] - truth[:2]) < 1e-3


def _ball_spots(cfg, rng, n):
    """n random ball positions on the L1 floor, clear of every wall; every other one against a wall."""
    walls = list(cad_walls().values())
    x0, x1, y0, y1 = cfg.ball.floor
    r = cfg.ball.radius

    def gap(x, y):  # distance from the ball's edge to the nearest wall
        return min(np.hypot(max(a - x, 0, x - b), max(c - y, 0, y - d)) for a, b, c, d in walls) - r

    out = []
    while len(out) < n:
        x, y = rng.uniform(x0 + r, x1 - r), rng.uniform(y0 + r, y1 - r)
        g = gap(x, y)
        if g >= 0.0005 and (len(out) % 2 == 0 or g < 0.004):
            out.append((x, y))
    return out


@pytest.mark.parametrize("seed", range(6))
def test_ball_matches_ground_truth(sim, cfg, seed):
    # P3: the ball's center within 2 mm of the sim's (#13 done-when: a few mm), anywhere on the floor
    rng = np.random.default_rng(100 + seed)
    for spot in _ball_spots(cfg, rng, 5):
        image, data, camera = _render(sim, seed, ball=spot)
        corners = MarkerDetector(cfg.board.dictionary, ids=cfg.board.markers).detect(image)
        pose = estimate_board_pose(corners, cfg.board, camera, cfg.pose)
        b = locate_ball(image, pose, camera, cfg.ball, cfg.rectified)
        assert b is not None, spot
        truth = data.body("board").xmat.reshape(3, 3).T @ (data.body("ball").xpos - data.body("board").xpos)
        assert truth[2] == pytest.approx(cfg.ball.radius, abs=1e-6)  # resting on the floor
        assert np.hypot(b.x - truth[0], b.y - truth[1]) < 0.002, spot
        assert b.score > 1.1 * cfg.ball.min_score, spot  # margin over the cut


@pytest.mark.parametrize("seed", range(6))
def test_no_ball_on_the_board(sim, cfg, seed):
    # nothing else on the board looks like the ball: wall ends, the gripper's fingers, the goal
    image, _, camera = _render(sim, seed, ball=None)
    corners = MarkerDetector(cfg.board.dictionary, ids=cfg.board.markers).detect(image)
    pose = estimate_board_pose(corners, cfg.board, camera, cfg.pose)
    best = locate_ball(image, pose, camera, replace(cfg.ball, min_score=-1.0), cfg.rectified)
    assert best.score < 0.9 * cfg.ball.min_score, (best.x, best.y)


ARM = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]


def _ball_in_board(model, data):
    """(position, velocity) of the ball relative to the board, in the board frame."""
    board, ball = model.body("board").id, model.body("ball").id
    R = data.body("board").xmat.reshape(3, 3)
    vb, vball = np.zeros(6), np.zeros(6)  # [angular, linear], world frame
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, board, vb, 0)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, ball, vball, 0)
    r = data.body("ball").xpos - data.xipos[board]
    return R.T @ (data.body("ball").xpos - data.body("board").xpos), R.T @ (vball[3:] - vb[3:] - np.cross(vb[:3], r))


@pytest.mark.parametrize("seed", range(2))
def test_ball_state_while_rolling(sim, cfg, seed):
    # P4: the ball rolls on the board as the arm tilts it (up to 4 deg about two wrist joints, a
    # new tilt every 0.7 s), into walls and corners; a camera frame every 16 ms (62.5 Hz) through
    # markers, board pose, ball and filter, as ball_tracker does. The arm is held by a PD loop
    # with gravity compensation here (ros2_control does it in the full sim).
    sim_cfg, model, renderer = sim
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("start").id)
    mujoco.mj_forward(model, data)
    qa = [model.joint(j).qposadr[0] for j in ARM]
    da = [model.joint(j).dofadr[0] for j in ARM]
    ua = [model.actuator(j).id for j in ARM]
    home = data.qpos[qa].copy()
    w, h = sim_cfg.camera.resolution
    fx, fy, cx, cy = pinhole_intrinsics(w, h, sim_cfg.camera.hfov_deg)
    camera = Camera(K=np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]), D=np.zeros(5), width=w, height=h)
    detector = MarkerDetector(cfg.board.dictionary, ids=cfg.board.markers)
    filt = BallFilter(cfg.tracking)
    rng = np.random.default_rng(seed)
    steps = int(round(1 / 60 / model.opt.timestep))  # 8 x 2 ms
    target, next_tilt, pos_err, vel_err, speeds = home, 0.0, [], [], []
    while data.time < 3.0:
        if data.time >= next_tilt:
            target = home.copy()
            target[3] += np.radians(rng.uniform(-4, 4))
            target[5] += np.radians(rng.uniform(-4, 4))
            next_tilt += 0.7
        for _ in range(steps):
            data.ctrl[ua] = data.qfrc_bias[da] + 2000 * (target - data.qpos[qa]) - 100 * data.qvel[da]
            mujoco.mj_step(model, data)
        renderer.update_scene(data, camera="overhead_camera")
        image = renderer.render()
        pose = estimate_board_pose(detector.detect(image), cfg.board, camera, cfg.pose)
        b = None if pose is None else locate_ball(image, pose, camera, cfg.ball, cfg.rectified)
        est = filt.step(data.time, None if b is None else (b.x, b.y))
        p, v = _ball_in_board(model, data)
        assert est.tracking_ok, data.time
        pos_err.append(np.hypot(est.x - p[0], est.y - p[1]))
        vel_err.append(np.hypot(est.vx - v[0], est.vy - v[1]))
        speeds.append(np.hypot(*v[:2]))
    pos_err, vel_err = np.array(pos_err[10:]), np.array(vel_err[10:])  # after the first 1/6 s
    assert max(speeds) > 0.05  # it did roll
    assert np.sqrt(np.mean(pos_err ** 2)) < 0.002 and pos_err.max() < 0.004
    assert np.sqrt(np.mean(vel_err ** 2)) < 0.04  # m/s

