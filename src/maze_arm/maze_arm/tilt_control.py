"""Remote-pivot tilt control: tilt command -> UR7e joint velocities (no ROS).

The board tilts about its own center: each step the controller asks for a board twist that
holds the center where it was at home and turns the board toward the commanded tilt, then
maps that twist to joint velocities with damped least squares (resolved-rate control).

Tilt follows Contract 4 (docs/interfaces.md), relative to the level home pose:
  tilt_x > 0 lowers the board's +x edge (rotation of +tilt_x about board +y),
  tilt_y > 0 lowers the board's +y edge (rotation of -tilt_y about board +x).
"""
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import yaml

from maze_arm.kinematics import board_jacobian, fk_tool0


@dataclass(frozen=True)
class ArmConfig:
    max_tilt: float          # rad, hard limit on each tilt axis
    max_tilt_rate: float     # rad/s, limit on how fast the tilt reference moves
    max_joint_speed: float   # rad/s, every joint
    k_pos: float             # 1/s, board center position gain
    k_rot: float             # 1/s, board orientation gain
    damping: float           # damped least squares lambda
    command_timeout: float   # s, stop the arm if no tilt command for this long


def load_arm_config(path):
    raw = yaml.safe_load(Path(path).read_text())
    values = {}
    for f in fields(ArmConfig):
        if f.name not in raw:
            raise ValueError(f"{path}: missing key '{f.name}'")
        v = float(raw[f.name])
        if not v > 0:
            raise ValueError(f"{path}: '{f.name}' must be > 0, got {v}")
        values[f.name] = v
    return ArmConfig(**values)


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def tilt_rotation(tilt_x, tilt_y):
    """Board orientation relative to home (in home board axes) for a Contract 4 tilt."""
    return _ry(tilt_x) @ _rx(-tilt_y)


def tilt_from_rotation(R_rel):
    """Inverse of tilt_rotation, from the board normal (yaw about the normal is ignored)."""
    n = R_rel[:, 2]
    return np.array([np.arctan2(n[0], n[2]), np.arcsin(np.clip(n[1], -1.0, 1.0))])


def rotation_log(R):
    """Axis * angle of a rotation matrix."""
    w = 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = np.linalg.norm(w)
    c = (np.trace(R) - 1.0) / 2.0
    if s < 1e-12:
        return w
    return w / s * np.arctan2(s, c)


def level_home(T_board):
    """The board pose to hold: same center, level, same heading (board x projected flat)."""
    x = T_board[:3, 0].copy()
    x[2] = 0.0
    x /= np.linalg.norm(x)
    z = np.array([0.0, 0.0, 1.0])
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, np.cross(z, x), z])
    T[:3, 3] = T_board[:3, 3]
    return T


class TiltController:
    def __init__(self, cfg, T_tool0_board, T_home):
        """T_tool0_board: kinematics.board_offset(...). T_home: level_home(board pose at start)."""
        self.cfg = cfg
        self.T_tool0_board = T_tool0_board
        self.T_home = T_home
        self.ref = np.zeros(2)   # tilt reference, moves toward the command at max_tilt_rate

    def board_pose(self, q):
        return fk_tool0(q) @ self.T_tool0_board

    def measured_tilt(self, q):
        R = self.board_pose(q)[:3, :3]
        return tilt_from_rotation(self.T_home[:3, :3].T @ R)

    def reset(self, q):
        """Start the reference at the board's present tilt, so the arm does not jump."""
        self.ref = self.measured_tilt(q)

    def _step_reference(self, tilt_cmd, dt):
        target = np.clip(np.asarray(tilt_cmd, float), -self.cfg.max_tilt, self.cfg.max_tilt)
        delta = target - self.ref
        step = self.cfg.max_tilt_rate * dt
        n = np.linalg.norm(delta)
        if n > step:
            delta *= step / n
        old = self.ref
        self.ref = self.ref + delta
        return old, self.ref

    def step(self, q, tilt_cmd, dt):
        """Joint velocities (rad/s, UR joint order) for one control step of length dt."""
        R_home = self.T_home[:3, :3]
        old, new = self._step_reference(tilt_cmd, dt)
        R_old = R_home @ tilt_rotation(*old)
        R_ref = R_home @ tilt_rotation(*new)
        w_ff = rotation_log(R_ref @ R_old.T) / dt   # reference angular velocity, base_link

        T = self.board_pose(q)
        e_pos = self.T_home[:3, 3] - T[:3, 3]
        e_rot = rotation_log(R_ref @ T[:3, :3].T)
        twist = np.concatenate([self.cfg.k_pos * e_pos, w_ff + self.cfg.k_rot * e_rot])

        J = board_jacobian(q, self.T_tool0_board)
        lam2 = self.cfg.damping ** 2
        qdot = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(6), twist)

        peak = np.max(np.abs(qdot)) / self.cfg.max_joint_speed
        if peak > 1.0:
            qdot /= peak   # scale all joints together so the board still moves the right way
        return qdot
