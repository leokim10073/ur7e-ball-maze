"""tilt_control driving the MuJoCo sim (Contract 4, #14): the ball goes the commanded way, the board
tilts about its own center (ground truth from /sim/board/pose), steps and sines are tracked, the
watchdog stops the arm, a sim reset is survived, and pid_node drives the ball through this node.

Runs headless in its own ROS domain (CMake sets ROS_DOMAIN_ID). Each run's samples go to
$TILT_RUN_DIR (default: a temporary directory) as CSV; scripts/plot_tilt_run.py plots them.
"""

import csv
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import pytest
import rclpy
from ament_index_python.packages import get_package_prefix, get_package_share_directory
from geometry_msgs.msg import PoseStamped
from maze_interfaces.msg import BallState, BoardTilt
from nav_msgs.msg import Path as PathMsg
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty
from std_srvs.srv import Trigger

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "maze_sim_bringup" / "test"))
from sim_helpers import (  # noqa: E402
    _call, controller_states, pose_matrix, sim_launch_description, spin_until, stamp, wait_for_startup,
)

DEG = np.pi / 180.0
RUN_DIR = Path(os.environ.get("TILT_RUN_DIR") or tempfile.mkdtemp(prefix="tilt_runs_"))
CMD_PERIOD = 1.0 / 60.0  # the ball controller's rate


@pytest.mark.launch_test
def generate_test_description():
    return sim_launch_description()


def log_so3(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = np.arccos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2
    return v if th < 1e-9 else th / np.sin(th) * v


def tilt_of(R0, R):
    """Contract 4 tilt of R relative to the level R0 (same convention as so3.hpp)."""
    rv = log_so3(R0.T @ R)
    return np.array([rv[1], -rv[0]])


def _start_node(executable_pkg, executable, params, extra=()):
    exe = os.path.join(get_package_prefix(executable_pkg), "lib", executable_pkg, executable)
    return subprocess.Popen([exe, "--ros-args", "--params-file", params, "-p", "use_sim_time:=true", *extra])


def _stop(proc):
    if proc and proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


class TestSimTilt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = rclpy.create_node("tilt_sim_test")
        n = cls.node
        cls.joints, cls.board, cls.ball, cls.state = [], [], [], []
        n.create_subscription(JointState, "/joint_states", cls.joints.append, 1000)
        n.create_subscription(PoseStamped, "/sim/board/pose", cls.board.append, 1000)
        n.create_subscription(PoseStamped, "/sim/ball/pose", cls.ball.append, 1000)
        best_effort = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        n.create_subscription(BoardTilt, "/board/tilt_state", cls.state.append, best_effort)
        cls.cmd_pub = n.create_publisher(BoardTilt, "/board/tilt_cmd", QoSProfile(depth=1))
        cls.reset_done = []
        n.create_subscription(
            Empty, "/sim/reset_done", cls.reset_done.append,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        cls.startup_states = wait_for_startup(n)
        # the level home: the board as the sim's start-up reset leaves it, before tilt_control runs
        spin_until(n, lambda: cls.board, 5.0)
        cls.R_home = pose_matrix(cls.board[-1].pose)[:3, :3]
        share = get_package_share_directory("maze_arm_control")
        cls.proc = _start_node("maze_arm_control", "tilt_control", os.path.join(share, "config", "tilt_control.yaml"))
        print(f"tilt runs -> {RUN_DIR}", file=sys.stderr)

    @classmethod
    def tearDownClass(cls):
        _stop(cls.proc)
        cls.node.destroy_node()
        rclpy.shutdown()

    # helpers -------------------------------------------------------------------------------

    def sim_now(self):
        spin_until(self.node, lambda: self.joints, 5.0)
        return stamp(self.joints[-1])

    def wait_sim(self, seconds):
        t0 = self.sim_now()
        spin_until(self.node, lambda: stamp(self.joints[-1]) - t0 >= seconds, 10.0 * seconds + 10.0)

    def running(self, timeout=20.0):
        """Wait until the node has the velocity controller and publishes tilt_state."""
        return spin_until(
            self.node, lambda: controller_states(self.node).get("forward_velocity_controller") == "active"
            and controller_states(self.node).get("scaled_joint_trajectory_controller") == "inactive", timeout)

    def reset(self):
        self.reset_done.clear()
        result = _call(self.node, Trigger, "/sim/reset", Trigger.Request())
        # /sim/reset blocks for ~1.5 s of sim time; the helper's 5 s may run out, so wait on reset_done
        spin_until(self.node, lambda: self.reset_done, 60.0)
        self.assertTrue(self.reset_done, f"no /sim/reset_done (reset answered {result})")
        self.assertTrue(self.running(), controller_states(self.node))
        self.wait_sim(0.5)  # the node re-runs start-up after reset_done

    def send(self, tilt_x, tilt_y):
        msg = BoardTilt(tilt_x=float(tilt_x), tilt_y=float(tilt_y))
        msg.header.frame_id = "maze_board"
        self.cmd_pub.publish(msg)

    def drive(self, profile, seconds, name=None):
        """Publish profile(t) (t = sim seconds from now) at 60 Hz for `seconds`; return one sample per
        ground-truth board pose (100 Hz): sim time, the command in effect at that pose's stamp, the
        board's tilt from the level home (ground truth), FK tilt_state, board center offset from the
        start of this run, ball in maze_board."""
        for buf in (self.board, self.ball, self.state):
            buf.clear()
        spin_until(self.node, lambda: self.board and self.ball and self.state, 5.0)
        T0 = pose_matrix(self.board[-1].pose)
        t0 = self.sim_now()
        sent, next_cmd, done = [], t0, 0  # (sim time sent, command)
        while True:
            rclpy.spin_once(self.node, timeout_sec=0.01)
            t = stamp(self.joints[-1])
            if t - t0 > seconds:
                break
            if t >= next_cmd:
                cmd = np.asarray(profile(t - t0), float)
                self.send(*cmd)
                sent.append((t, cmd))
                next_cmd += CMD_PERIOD
        rows = []
        for b in self.board:
            tb = stamp(b)
            past = [c for ts, c in sent if ts <= tb]
            if not past or tb - t0 > seconds:
                continue
            T = pose_matrix(b.pose)
            ball = [m for m in self.ball if stamp(m) <= tb] or self.ball[:1]
            state = [m for m in self.state if stamp(m) <= tb] or self.state[:1]
            bb = np.linalg.inv(T) @ np.r_[pose_matrix(ball[-1].pose)[:3, 3], 1.0]
            rows.append([tb - t0, *past[-1], *tilt_of(self.R_home, T[:3, :3]), state[-1].tilt_x, state[-1].tilt_y,
                         *(T[:3, 3] - T0[:3, 3]), bb[0], bb[1]])
        self.assertTrue(rows, "no samples (is tilt_control still running?)")
        data = np.array(rows)
        if name:
            path = RUN_DIR / f"{name}.csv"
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t", "cmd_x", "cmd_y", "tilt_x", "tilt_y", "fk_x", "fk_y", "dx", "dy", "dz",
                            "ball_x", "ball_y"])
                w.writerows(data.tolist())
        return data

    @staticmethod
    def center_mm(data):
        return 1000.0 * np.max(np.linalg.norm(data[:, 7:10], axis=1))

    def level(self):
        self.drive(lambda t: (0.0, 0.0), 1.0)

    # tests (name order) ---------------------------------------------------------------------

    def test_1_starts_and_switches(self):
        self.assertTrue(self.running(), controller_states(self.node))
        self.assertIsNone(self.proc.poll(), "tilt_control exited")
        spin_until(self.node, lambda: self.state, 5.0)
        self.assertTrue(self.state, "no /board/tilt_state")
        self.assertEqual(self.state[-1].header.frame_id, "maze_board")
        self.assertLess(np.max(np.abs([self.state[-1].tilt_x, self.state[-1].tilt_y])), 1.0 * DEG)

    def _contract4(self, axis):
        self.reset()
        self.level()
        cmd = np.zeros(2)
        cmd[axis] = 0.035
        data = self.drive(lambda t: cmd, 0.5, f"contract4_{'xy'[axis]}")
        moved = data[-1, 10:12] - data[0, 10:12]
        print(f"CONTRACT4 {'xy'[axis]}: ball moved {moved * 1000} mm, center {self.center_mm(data):.2f} mm",
              file=sys.stderr)
        self.assertGreater(moved[axis], 0.003, f"ball moved {moved} m")
        self.assertLess(abs(moved[1 - axis]), 0.5 * moved[axis], f"ball moved {moved} m")
        self.assertLess(self.center_mm(data), 5.0)
        self.level()

    def test_2_contract4_tilt_x(self):
        self._contract4(0)

    def test_3_contract4_tilt_y(self):
        self._contract4(1)

    def test_4_step_tracking(self):
        for axis, sign in ((0, 1), (1, 1), (0, -1), (1, -1)):
            self.level()
            cmd = np.zeros(2)
            cmd[axis] = sign * 5 * DEG
            name = f"step_{'+' if sign > 0 else '-'}{'xy'[axis]}"
            data = self.drive(lambda t: cmd if t > 0.1 else np.zeros(2), 1.5, name)
            settled = data[data[:, 0] > 0.8]
            err = np.max(np.abs(settled[:, 3:5] - settled[:, 1:3])) / DEG
            fk_err = np.max(np.abs(settled[:, 5:7] - settled[:, 1:3])) / DEG
            peak = np.max(np.abs(data[:, 3 + axis])) / DEG
            print(f"STEP {name}: settled error {err:.3f} deg (FK {fk_err:.3f}), peak {peak:.3f} deg, "
                  f"center {self.center_mm(data):.2f} mm", file=sys.stderr)
            self.assertLessEqual(err, 0.5, name)
            self.assertLessEqual(peak, 5.0 + 0.1, f"{name}: overshoot past the 5 deg hard limit")
            self.assertLess(self.center_mm(data), 5.0, name)
        self.level()

    def test_5_sine_tracking(self):
        for axis in (0, 1):
            self.level()
            name = f"sine_{'xy'[axis]}"

            def profile(t, axis=axis):
                cmd = np.zeros(2)
                cmd[axis] = 3 * DEG * np.sin(2 * np.pi * 0.5 * t)
                return cmd

            data = self.drive(profile, 4.0, name)
            tail = data[data[:, 0] > 1.0]
            err = np.max(np.abs(tail[:, 3:5] - tail[:, 1:3])) / DEG
            print(f"SINE {name}: error {err:.3f} deg, center {self.center_mm(data):.2f} mm", file=sys.stderr)
            self.assertLessEqual(err, 0.5, name)
            self.assertLess(self.center_mm(data), 5.0, name)
        self.level()

    def test_6_watchdog_stops(self):
        self.drive(lambda t: (3 * DEG * np.sin(2 * np.pi * 0.5 * t), 0.0), 0.6)  # moving when it stops
        t_stop = self.sim_now()
        self.wait_sim(0.4)
        late = [m for m in self.joints[-400:] if stamp(m) > t_stop + 0.3]
        self.assertTrue(late)
        speed = max(max(abs(v) for v in m.velocity[:6]) for m in late)
        print(f"WATCHDOG: max joint speed 0.3-0.4 s after the last command: {speed:.4f} rad/s", file=sys.stderr)
        self.assertLess(speed, 0.01)
        self.assertIsNone(self.proc.poll())

    def test_7_reset_resync(self):
        self.drive(lambda t: (2 * DEG, 0.0), 0.5)
        self.reset()
        data = self.drive(lambda t: (0.0, 2 * DEG), 1.5, "after_reset")
        err = np.max(np.abs(data[data[:, 0] > 0.8][:, 3:5] - data[data[:, 0] > 0.8][:, 1:3])) / DEG
        print(f"RESET: error after reset {err:.3f} deg", file=sys.stderr)
        self.assertLessEqual(err, 0.5)
        self.assertIsNone(self.proc.poll())
        self.level()

    def test_8_pid_drives_ball(self):
        """pid_node (maze_control) -> /board/tilt_cmd -> this node -> the arm -> the ball. The test plays
        perception: /ball/state from ground truth, and a straight /maze/path along the start corridor."""
        self.reset()
        self.level()  # back from the velocity-mode droop the reset leaves, before the center is measured
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        path_pub = self.node.create_publisher(PathMsg, "/maze/path", latched)
        state_pub = self.node.create_publisher(
            BallState, "/ball/state", QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        spin_until(self.node, lambda: self.board and self.ball, 5.0)
        T = pose_matrix(self.board[-1].pose)
        start = (np.linalg.inv(T) @ np.r_[pose_matrix(self.ball[-1].pose)[:3, 3], 1.0])[:2]
        goal = start + np.array([0.08, 0.0])  # +x along the top corridor (the first wall is at x = 0)
        path = PathMsg()
        path.header.frame_id = "maze_board"
        for s in np.linspace(0, 1, 17):
            ps = PoseStamped()
            ps.header.frame_id = "maze_board"
            ps.pose.position.x, ps.pose.position.y = (float(v) for v in start + s * (goal - start))
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        path_pub.publish(path)
        pid = _start_node("maze_control", "pid_node",
                          os.path.join(get_package_share_directory("maze_control"), "config", "pid.yaml"))
        try:
            T0 = pose_matrix(self.board[-1].pose)
            t0, next_state, prev, center, rows = self.sim_now(), 0.0, None, 0.0, []
            while self.sim_now() - t0 < 6.0:
                rclpy.spin_once(self.node, timeout_sec=0.01)
                t = stamp(self.joints[-1])
                if t < next_state or not (self.board and self.ball):
                    continue
                next_state = t + CMD_PERIOD
                T = pose_matrix(self.board[-1].pose)
                b = (np.linalg.inv(T) @ np.r_[pose_matrix(self.ball[-1].pose)[:3, 3], 1.0])[:2]
                v = np.zeros(2) if prev is None else (b - prev[1]) / max(t - prev[0], 1e-3)
                prev = (t, b)
                msg = BallState(x=float(b[0]), y=float(b[1]), vx=float(v[0]), vy=float(v[1]), tracking_ok=True)
                msg.header.stamp = self.joints[-1].header.stamp
                msg.header.frame_id = "maze_board"
                state_pub.publish(msg)
                center = max(center, 1000 * np.linalg.norm(T[:3, 3] - T0[:3, 3]))
                rows.append([t - t0, *b])
            with open(RUN_DIR / "pid_ball.csv", "w", newline="") as f:
                csv.writer(f).writerows([["t", "ball_x", "ball_y"], *rows])
            progress = rows[-1][1] - start[0]
            print(f"PID: ball progressed {progress * 1000:.1f} mm along the path, center {center:.2f} mm",
                  file=sys.stderr)
            self.assertGreater(progress, 0.03)
            self.assertLess(abs(rows[-1][2] - start[1]), 0.02)
            self.assertLess(center, 5.0)
            self.assertIsNone(self.proc.poll())
        finally:
            _stop(pid)
            self.node.destroy_publisher(path_pub)
            self.node.destroy_publisher(state_pub)
