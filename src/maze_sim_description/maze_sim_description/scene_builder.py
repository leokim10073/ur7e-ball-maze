"""Compose the MuJoCo scene: Menagerie UR5e, with the UR7e link lengths from ur_description, and a
Robotiq Hand-E with V-groove fingertips on the flange, the maze board held in the gripper, the ball,
the overhead camera, and keyframe "start".

The arm's position servos are replaced by torque (motor) actuators named after the joints, so
mujoco_ros2_control's PIDs can serve both position and velocity command interfaces. The Hand-E's
two fingers slide on prismatic joints: the left one carries the gripper actuator (a force-limited
position servo named after its joint, in metres of travel from fully open), and the right one
follows it through a joint equality, as the real gripper's single motor drives both.
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np
import yaml

from .board_assets import board_mjcf_fragment, load_board_assets
from .camera_math import matrix_to_quat, mujoco_camera_quaternion
from .config import load_sim_config

UR_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
UR_TORQUE_LIMITS = (150.0, 150.0, 150.0, 28.0, 28.0, 28.0)  # N m, UR7e/UR5e datasheet
GRIPPER_PREFIX = "gripper_"
GRIPPER_ACTUATOR = GRIPPER_PREFIX + "left_finger_joint"
GRIPPER_FOLLOWER = GRIPPER_PREFIX + "right_finger_joint"
BOARD_PREFIX = "maze_"
# Robotiq Hand-E, from its STEP files (maze_CAD/gripper/robotiq_hande) and Robotiq's spec. In the
# tool0 frame: z along the approach, x along the fingers' closing axis.
HANDE_STROKE = 0.025  # m of travel per finger: grip faces 50 mm apart fully open, touching fully closed
HANDE_MOUNT_Z = 0.110  # m from the flange to the fingers' mounting face: 11 mm I/O coupling + 99 mm gripper
HANDE_STEP_MOUNT_Y = 0.0146  # m: that face in the base STEP's frame, whose +y is the approach axis
HANDE_BODY_TOP_Z = HANDE_MOUNT_Z - HANDE_STEP_MOUNT_Y + 0.0072  # top of the housing (STEP y = 7.2 mm)
HANDE_HALF_WIDTH = 0.0375  # 75 x 75 mm housing
HANDE_MASS = 0.98  # kg: 0.86 gripper + 0.12 coupling (Robotiq); the fingertips add 2 x 0.038
HANDE_FINGER_MASS = 0.038
HANDE_FINGER_ARMATURE = 1.0  # kg: the motor and lead screw seen at the finger; keeps the stiff servo stable
GRIPPER_KP = 1e5  # N/m: commanded fully closed, a finger stopped by the 12 mm handle is 6 mm short,
GRIPPER_KV = 600.0  # so the servo saturates at its force limit
# MuJoCo's soft friction lets a board held only by contacts creep in the grasp (~0.01°/s under
# arm motion, whatever the friction or finger force). While the gripper is closed on the board,
# GraspWeldPlugin (maze_sim_nodes) switches this weld on; opening switches it off, so a release
# is still contacts and gravity. The rule is grasp_weld_engaged, mirrored in the plugin, which
# reads the tolerances and the gripper actuator from the model (<numeric>/<text> "grasp_weld").
GRASP_WELD = "grasp_weld"
GRASP_WELD_POS_TOL = 0.005  # m: engage only for a board this close to its grasp pose...
GRASP_WELD_ANG_TOL_DEG = 3.0  # ...and this close in angle, so a dropped board is never pulled back
SETTLE_S = 0.5  # pre-simulation that closes the fingers on the handle (board pinned)
LOAD_SETTLE_S = 1.0  # then the board hangs in the grasp under its own weight


def _menagerie_dir():
    here = Path(__file__).resolve().parents[1] / "mjcf" / "menagerie"
    if here.is_dir():
        return here
    from ament_index_python.packages import get_package_share_directory

    return Path(get_package_share_directory("maze_sim_description")) / "mjcf" / "menagerie"


def ur_dh_parameters(ur_type):
    """The UR's DH lengths (d1, a2, a3, d4, d5, d6), m, from ur_description's
    default_kinematics.yaml for ur_type: the same file format a robot's own calibration produces."""
    from ament_index_python.packages import get_package_share_directory

    path = Path(get_package_share_directory("ur_description")) / "config" / ur_type / "default_kinematics.yaml"
    k = yaml.safe_load(path.read_text())["kinematics"]
    return dict(d1=k["shoulder"]["z"], a2=-k["forearm"]["x"], a3=-k["wrist_1"]["x"],
                d4=k["wrist_1"]["z"], d5=-k["wrist_2"]["y"], d6=k["wrist_3"]["y"])


def _set_ur_link_lengths(spec, dh):
    """Menagerie's UR5e rounds the link lengths (0.163 vs 0.1625 m, ...); set the real ones. The
    shoulder, elbow and wrist_2 offsets lie along parallel joint axes, so only their sum (d4) matters:
    keep Menagerie's first two (they place the meshes) and put the rest in wrist_2."""
    spec.body("shoulder_link").pos = [0.0, 0.0, dh["d1"]]
    forearm = spec.body("forearm_link")
    forearm.pos = [0.0, forearm.pos[1], dh["a2"]]
    spec.body("wrist_1_link").pos = [0.0, 0.0, dh["a3"]]
    elbow_offset = spec.body("upper_arm_link").pos[1] + forearm.pos[1]
    spec.body("wrist_2_link").pos = [0.0, dh["d4"] - elbow_offset, 0.0]
    spec.body("wrist_3_link").pos = [0.0, 0.0, dh["d5"]]
    spec.site("attachment_site").pos = [0.0, dh["d6"], 0.0]


def _absolute_meshes(spec, mesh_dir):
    """Mesh and texture files as absolute paths, so the scene can be written anywhere."""
    for asset in [*spec.meshes, *spec.textures]:
        if asset.file:
            asset.file = str(Path(mesh_dir).resolve() / asset.file)
    spec.meshdir = spec.texturedir = ""


def _rpy_matrix(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx  # ROS fixed-axis roll, pitch, yaw


def gripper_command(model, closed):
    """ctrl value that fully closes (force-limited) or fully opens the fingers."""
    lo, hi = model.actuator(GRIPPER_ACTUATOR).ctrlrange
    return float(hi if closed else lo)


def grasp_weld_engaged(model, data, engaged):
    """Whether the grasp weld should be on for the next step: the gripper is commanded closed,
    and the weld is already on or the board sits within tolerance of its grasp pose."""
    act = model.actuator(GRIPPER_ACTUATOR)
    lo, hi = act.ctrlrange
    if data.ctrl[act.id] <= (lo + hi) / 2:
        return False
    if engaged:
        return True
    tol = model.numeric(GRASP_WELD)
    pos_tol, ang_tol_deg = model.numeric_data[tol.adr[0] : tol.adr[0] + 2]
    weld = model.equality(GRASP_WELD)
    rel = model.eq_data[weld.id]
    R_w = data.xmat[weld.obj1id[0]].reshape(3, 3)
    R_b = data.xmat[weld.obj2id[0]].reshape(3, 3)
    p_err = R_w.T @ (data.xpos[weld.obj2id[0]] - data.xpos[weld.obj1id[0]]) - rel[3:6]
    R_rel = np.zeros(9)
    mujoco.mju_quat2Mat(R_rel, rel[6:10])
    cos = (np.trace(R_rel.reshape(3, 3).T @ R_w.T @ R_b) - 1) / 2
    return bool(np.linalg.norm(p_err) < pos_tol and cos > np.cos(np.radians(ang_tol_deg)))


def _hande_mesh_dir():
    here = Path(__file__).resolve().parents[1] / "meshes" / "hande"
    if here.is_dir():
        return here
    from ament_index_python.packages import get_package_share_directory

    return Path(get_package_share_directory("maze_sim_description")) / "meshes" / "hande"


def _quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).ravel())
    return q


def _hande_spec(cfg):
    """Robotiq Hand-E with V-groove fingertips, body "base_mount" at the flange (tool0 frame).

    Visuals are the STEP meshes (maze_CAD/scripts/export_hande.py); collisions are boxes: the
    housing, and per finger its 6 mm blade (the flat 40 mm grip face; the V-groove is only visual)
    and its mounting block. The finger carriers stay where the STEP has them, as part of the housing
    visual: in the sim the fingers only ever sit near closed on the handle.
    """
    mesh_dir = _hande_mesh_dir()
    s = mujoco.MjSpec()
    s.compiler.degree = False
    s.add_mesh(name="hande_base", file=str(mesh_dir / "hande_base.stl"))
    s.add_mesh(name="hande_finger", file=str(mesh_dir / "hande_finger_vgroove.stl"))
    base = s.worldbody.add_body(name="base_mount", gravcomp=1.0)
    visual = dict(contype=0, conaffinity=0, group=2, mass=0.0)
    # coupling: a plain cylinder between the flange and the housing
    base.add_geom(name="coupling", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                  size=[HANDE_HALF_WIDTH, 0.5 * (HANDE_MOUNT_Z - HANDE_STEP_MOUNT_Y - 0.0861), 0],
                  pos=[0, 0, 0.5 * (HANDE_MOUNT_Z - HANDE_STEP_MOUNT_Y - 0.0861)], rgba=[0.2, 0.2, 0.2, 1], **visual)
    # base STEP frame -> tool0: STEP x -> x (closing axis), STEP y -> z (approach), STEP z -> -y
    base.add_geom(name="housing_visual", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="hande_base",
                  pos=[0, 0, HANDE_MOUNT_Z - HANDE_STEP_MOUNT_Y],
                  quat=_quat([[1, 0, 0], [0, 0, -1], [0, 1, 0]]), rgba=[0.15, 0.15, 0.15, 1], **visual)
    base.add_geom(name="housing", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=[HANDE_HALF_WIDTH, HANDE_HALF_WIDTH, 0.5 * HANDE_BODY_TOP_Z],
                  pos=[0, 0, 0.5 * HANDE_BODY_TOP_Z], mass=HANDE_MASS, group=3)
    # fingertip STEP frame -> finger body: STEP x (width) -> y, STEP y (length) -> z, STEP z (out of
    # the grip face) -> +x for the left finger; the right one is that turned by pi about z
    R_left = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    R_right = np.diag([-1, -1, 1]) @ R_left
    pad = dict(type=mujoco.mjtGeom.mjGEOM_BOX, friction=cfg.board.surface_friction, priority=1,
               solimp=[0.95, 0.99, 0.001, 0.5, 2], solref=[0.004, 1], group=3)  # Menagerie 2F-85 pads' contact
    for side, R, x in (("left", R_left, HANDE_STROKE), ("right", R_right, -HANDE_STROKE)):
        finger = base.add_body(name=f"{side}_finger", pos=[x, 0, HANDE_MOUNT_Z], quat=_quat(R), gravcomp=1.0)
        finger.add_joint(name=f"{side}_finger_joint", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, -1],
                         range=[0, HANDE_STROKE], armature=HANDE_FINGER_ARMATURE)
        finger.add_geom(name=f"{side}_finger_visual", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="hande_finger",
                        rgba=[0.7, 0.7, 0.72, 1], **visual)
        # blade: the whole 21 x 40 mm grip face at STEP z = 0, 6 mm thick
        finger.add_geom(name=f"{side}_pad", size=[0.0105, 0.020, 0.003], pos=[0, 0.020, 0.003],
                        mass=HANDE_FINGER_MASS, **pad)
        finger.add_geom(name=f"{side}_block", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.0105, 0.0065, 0.0046],
                        pos=[0, 0.0065, 0.0106], mass=0.0, group=3)
    act = s.add_actuator(name="left_finger_joint", target="left_finger_joint", trntype=mujoco.mjtTrn.mjTRN_JOINT)
    act.set_to_position(kp=GRIPPER_KP, kv=GRIPPER_KV)
    act.ctrllimited = act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
    act.ctrlrange = [0.0, HANDE_STROKE]
    # one motor drives both fingers: through the coupling, the actuator's force on the left finger's
    # joint splits evenly between the two jaws, so twice the grip force gives grip force per jaw
    act.forcerange = [-2 * cfg.gripper.force_limit, 2 * cfg.gripper.force_limit]
    s.add_equality(name="finger_coupling", type=mujoco.mjtEq.mjEQ_JOINT, objtype=mujoco.mjtObj.mjOBJ_JOINT,
                   name1="right_finger_joint", name2="left_finger_joint", data=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   solref=[2 * cfg.physics.timestep, 1.0])  # stiffest stable: the default lets the fingers differ ~1 mm
    return s


def _compose(cfg, board):
    menagerie = _menagerie_dir()
    spec = mujoco.MjSpec.from_file(str(menagerie / "universal_robots_ur5e" / "ur5e.xml"))
    _absolute_meshes(spec, menagerie / "universal_robots_ur5e" / "assets")
    spec.modelname = "maze_sim"
    spec.option.timestep = cfg.physics.timestep
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.noslip_iterations = cfg.physics.noslip_iterations
    spec.option.impratio = 10.0  # as Menagerie's 2F-85, which the grasp was tuned with
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    for key in list(spec.keys):
        spec.delete(key)
    for act in list(spec.actuators):
        spec.delete(act)
    for joint, limit in zip(UR_JOINTS, UR_TORQUE_LIMITS):
        act = spec.add_actuator(name=joint, target=joint, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.set_to_motor()
        act.ctrllimited = act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.ctrlrange = act.forcerange = [-limit, limit]

    # Menagerie's world is the UR "base" frame; ROS world = base_link is that rotated by pi about z.
    spec.body("base").quat = [1.0, 0.0, 0.0, 0.0]
    for body in spec.bodies:
        body.gravcomp = 1.0  # arm, then the gripper below: the UR controller cancels arm and tool weight
    for joint in UR_JOINTS:
        spec.joint(joint).actgravcomp = True

    _set_ur_link_lengths(spec, ur_dh_parameters(cfg.robot.ur_type))
    flange = spec.site("attachment_site")
    spec.body("wrist_3_link").add_site(name="tool0", pos=flange.pos, quat=flange.quat)

    flange.attach_body(_hande_spec(cfg).body("base_mount"), GRIPPER_PREFIX, "")

    board_spec = mujoco.MjSpec.from_string(
        "<mujoco>" + board_mjcf_fragment(board, cfg.board.mass, cfg.board.surface_friction) + "</mujoco>"
    )
    _absolute_meshes(board_spec, board.visual_meshes[0].parent)
    # An empty prefix would copy the board's main default as an unnamed class, which to_xml
    # cannot write back; attach with a prefix and strip it from the named elements.
    spec.worldbody.add_frame().attach_body(board_spec.body("board"), BOARD_PREFIX, "")
    for element in [*spec.meshes, *spec.bodies, *spec.joints, *spec.geoms]:
        if element.name.startswith(BOARD_PREFIX):
            element.name = element.name[len(BOARD_PREFIX):]
    for geom in spec.geoms:
        if geom.meshname.startswith(BOARD_PREFIX):
            geom.meshname = geom.meshname[len(BOARD_PREFIX):]

    spec.add_equality(
        name=GRASP_WELD, type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
        name1="wrist_3_link", name2="board", active=False,
        solref=[2 * cfg.physics.timestep, 1.0],  # stiffest stable: the default gives way ~5 mm to the fingers' squeeze
    )  # its target pose is set in _scene_spec, once the arm's start pose is known
    spec.add_numeric(name=GRASP_WELD, data=[GRASP_WELD_POS_TOL, GRASP_WELD_ANG_TOL_DEG])
    spec.add_text(name=GRASP_WELD + "_actuator", data=GRIPPER_ACTUATOR)

    ball_cls = spec.add_default("ball", spec.default)
    ball_cls.geom.friction = cfg.ball.friction
    ball_cls.geom.condim = 6
    ball_cls.geom.priority = 1  # ball friction wins against the board's
    ball_cls.geom.solref = [2 * cfg.physics.timestep, 1.0]  # stiffest stable contact: walls stop the ball
    ball = spec.worldbody.add_body(name="ball")
    ball.add_freejoint(name="ball_free")
    ball.add_geom(
        default=ball_cls, name="ball", type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[cfg.ball.radius, 0, 0], mass=cfg.ball.mass, rgba=[0.75, 0.75, 0.8, 1],
    )

    spec.worldbody.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.05], rgba=[0.5, 0.5, 0.5, 1])
    # Room light from above, specular kept so the steel ball shows a highlight; the camera-side
    # headlight dimmed and without specular, or the lit board clips (test_render)
    spec.worldbody.add_light(pos=[0.5, 0, 3], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    spec.visual.headlight.ambient = [0.1, 0.1, 0.1]
    spec.visual.headlight.diffuse = [0.2, 0.2, 0.2]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]

    w, h = cfg.camera.resolution
    hfov = np.radians(cfg.camera.hfov_deg)
    spec.worldbody.add_camera(
        name="overhead_camera",
        pos=cfg.camera.position,
        quat=mujoco_camera_quaternion(cfg.camera.position, cfg.camera.look_at, cfg.camera.up),
        fovy=float(np.degrees(2 * np.arctan(np.tan(hfov / 2) * h / w))),
        resolution=[w, h],
    )
    spec.visual.global_.offwidth = max(w, spec.visual.global_.offwidth)
    spec.visual.global_.offheight = max(h, spec.visual.global_.offheight)

    return spec


def _handle_target(data, cfg):
    """Handle frame in world: tool0 * grasp_offset. Its axes are the board's axes."""
    R_t = data.site("tool0").xmat.reshape(3, 3)
    return data.site("tool0").xpos + R_t @ np.asarray(cfg.board.grasp_offset.xyz), R_t @ _rpy_matrix(cfg.board.grasp_offset.rpy)


def _grasp_relpose(model, cfg, handle):
    """The board's pose in the wrist_3_link frame with its handle exactly at grasp_offset."""
    data = mujoco.MjData(model)
    data.qpos[[model.joint(j).qposadr[0] for j in UR_JOINTS]] = cfg.robot.initial_joints
    mujoco.mj_kinematics(model, data)
    p_h, R_h = _handle_target(data, cfg)
    R_w = data.body("wrist_3_link").xmat.reshape(3, 3)
    p_b = p_h - R_h @ handle
    return np.concatenate([R_w.T @ (p_b - data.body("wrist_3_link").xpos), matrix_to_quat(R_w.T @ R_h)])


def _grasp_and_settle(model, cfg, handle, p_h, R_h):
    """Pin the board with its handle at (p_h, R_h), close the fingers on it, then let the board
    hang in the grasp under its weight. Arm held at initial_joints, ball parked away."""
    data = mujoco.MjData(model)
    arm_q = [model.joint(j).qposadr[0] for j in UR_JOINTS]
    arm_v = [model.joint(j).dofadr[0] for j in UR_JOINTS]
    b_adr, b_dof = model.joint("board_free").qposadr[0], model.joint("board_free").dofadr[0]
    ball_adr, ball_dof = model.joint("ball_free").qposadr[0], model.joint("ball_free").dofadr[0]
    lo, hi = model.actuator(GRIPPER_ACTUATOR).ctrlrange
    for joint in (GRIPPER_ACTUATOR, GRIPPER_FOLLOWER):
        data.joint(joint).qpos = lo + cfg.gripper.closed_position * (hi - lo)
    board_q = np.concatenate([p_h - R_h @ handle, matrix_to_quat(R_h)])
    data.ctrl[model.actuator(GRIPPER_ACTUATOR).id] = gripper_command(model, closed=True)

    def hold(pin_board):
        data.qpos[arm_q] = cfg.robot.initial_joints
        data.qvel[arm_v] = 0
        data.qpos[ball_adr : ball_adr + 7] = (0, 0, -1.0, 1, 0, 0, 0)
        data.qvel[ball_dof : ball_dof + 6] = 0
        if pin_board:
            data.qpos[b_adr : b_adr + 7] = board_q
            data.qvel[b_dof : b_dof + 6] = 0

    weld = model.equality(GRASP_WELD).id
    for pin_board, seconds in ((True, SETTLE_S), (False, LOAD_SETTLE_S)):
        for _ in range(int(round(seconds / model.opt.timestep))):
            hold(pin_board)
            data.eq_active[weld] = grasp_weld_engaged(model, data, bool(data.eq_active[weld]))
            mujoco.mj_step(model, data)
    hold(False)
    data.qvel[:] = 0
    mujoco.mj_kinematics(model, data)
    return data


def _start_keyframe(model, cfg, board):
    """Arm at initial_joints, fingers closed on the handle with the board pinned at grasp_offset,
    then the board settled in the grasp under its weight with the grasp weld on (so it stays at
    grasp_offset), ball resting at ball.start on the board surface."""
    handle = np.asarray(board.handle_center_m)
    data = mujoco.MjData(model)
    data.qpos[[model.joint(j).qposadr[0] for j in UR_JOINTS]] = cfg.robot.initial_joints
    mujoco.mj_kinematics(model, data)
    data = _grasp_and_settle(model, cfg, handle, *_handle_target(data, cfg))
    p_b, R_b = data.body("board").xpos.copy(), data.body("board").xmat.reshape(3, 3).copy()
    ball = model.joint("ball_free").qposadr[0]
    data.qpos[ball : ball + 7] = np.concatenate([p_b + R_b @ np.array([*cfg.ball.start, cfg.ball.radius]), [1, 0, 0, 0]])
    return data.qpos.copy(), data.ctrl.copy()


def _scene_spec(cfg, board_dir):
    board = load_board_assets(board_dir, units=cfg.board.units)
    spec = _compose(cfg, board)
    model = spec.compile()
    weld_data = [0.0, 0.0, 0.0, *_grasp_relpose(model, cfg, np.asarray(board.handle_center_m)), 1.0]
    next(e for e in spec.equalities if e.name == GRASP_WELD).data = weld_data
    model.eq_data[model.equality(GRASP_WELD).id] = weld_data
    qpos, ctrl = _start_keyframe(model, cfg, board)
    spec.add_key(name="start", qpos=qpos, ctrl=ctrl)
    return spec


def build_scene(cfg, board_dir):
    """MJCF text of the whole scene, with absolute mesh paths."""
    return _scene_spec(cfg, board_dir).to_xml()


def write_scene(cfg, board_dir, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_scene(cfg, board_dir))
    return out_path


def main():
    p = argparse.ArgumentParser(description="Write the MuJoCo scene for a sim.yaml.")
    p.add_argument("config")
    p.add_argument("board_dir")
    p.add_argument("out")
    a = p.parse_args()
    print(write_scene(load_sim_config(a.config), a.board_dir, a.out))


if __name__ == "__main__":
    main()
