import numpy as np

# UR7e nominal DH (same as UR5e): ur_description/config/ur7e/default_kinematics.yaml
D = [0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996]
A = [0.0, -0.425, -0.3922, 0.0, 0.0, 0.0]
ALPHA = [np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0]

UR_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
             "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")

def joints_from_state(names, positions):
    """Joint angles in UR_JOINTS order from a JointState's names and positions (any order)."""
    pos = dict(zip(names, positions))
    missing = [j for j in UR_JOINTS if j not in pos]
    if missing:
        raise ValueError(f"joint state is missing {missing}")
    return [pos[j] for j in UR_JOINTS]

def dh(a, alpha, d, theta):
    ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
    return np.array([[ct, -st * ca,  st * sa, a * ct],
                     [st,  ct * ca, -ct * sa, a * st],
                     [0.0,      sa,       ca,      d],
                     [0.0,     0.0,      0.0,    1.0]])


def rpy_matrix(r, p, y):
    """ROS fixed-axis roll-pitch-yaw, same as the sim's scene_builder."""
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def transform(xyz=(0, 0, 0), R=np.eye(3)):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = xyz
    return T


def joint_frames(q):
    """base_link -> the frame each joint turns about (its z axis), for joints 1..6, then tool0."""
    T = transform(R=rpy_matrix(0, 0, np.pi))
    frames = [T]
    for i in range(6):
        T = T @ dh(A[i], ALPHA[i], D[i], q[i])
        frames.append(T)
    return frames


def fk_tool0(q):
    """base_link -> tool0. base_link is the UR DH base rotated by pi about z."""
    return joint_frames(q)[-1]


def board_offset(grasp_xyz, grasp_rpy, handle_center):
    """tool0 -> maze_board."""
    T_tool0_handle = transform(grasp_xyz, rpy_matrix(*grasp_rpy))
    T_handle_board = transform(-np.asarray(handle_center))  # board axes = handle axes
    return T_tool0_handle @ T_handle_board


def fk_board(q, grasp_xyz, grasp_rpy, handle_center):
    """base_link -> maze_board (board center, z = 0 on the maze floor)."""
    return fk_tool0(q) @ board_offset(grasp_xyz, grasp_rpy, handle_center)


def board_jacobian(q, T_tool0_board):
    """6x6 Jacobian of the board center in base_link: [v; w] = J @ qdot,
    v = linear velocity of the board center, w = angular velocity of the board."""
    frames = joint_frames(q)
    p = (frames[-1] @ T_tool0_board)[:3, 3]
    J = np.zeros((6, 6))
    for i in range(6):
        z, o = frames[i][:3, 2], frames[i][:3, 3]
        J[:3, i] = np.cross(z, p - o)
        J[3:, i] = z
    return J
