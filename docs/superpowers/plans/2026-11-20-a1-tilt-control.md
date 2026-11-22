# A1 Remote-Pivot Tilt Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `/board/tilt_cmd` (BoardTilt, Contract 4a) into UR7e joint velocities on `/forward_velocity_controller/commands` so the board tilts about its own center, and publish `/board/tilt_state` (Contract 4b).

**Architecture:** New ament_cmake package `src/maze_arm_control`. A ROS-free core library (UR DH FK + board mount, analytic geometric Jacobian at the board center, SO(3) helpers, reference shaper, resolved-rate DLS controller with velocity/acceleration/twist limits, watchdog) is unit- and kinematically-tested with gtest; a thin C++ node wraps it (startup state machine, controller switch, resync, tilt_state). Sim launch tests check Contract 4 end to end.

**Tech Stack:** C++17, Eigen 3, rclcpp (Humble), controller_manager_msgs, yaml-cpp (tests), gtest, launch_testing (Python).

**Spec:** issue #14 last comment "Plan for A1 (validated, not built yet)" and the working copy `a1_plan.md` (validation numbers + 9 review fixes); reference law `a1_sweep.py`; `docs/interfaces.md` Contract 4.

## Global Constraints

- Hard limits: 0.087 rad tilt per axis, 0.5 rad/s tilt rate per axis (Contract 4).
- No `tilt_cmd` for 200 ms, or joint states older than 20 ms -> zero joint velocities + warning; checked on its own steady (wall) timer.
- Startup: switch controllers with STRICT and check `ok`; wait for fresh joint states; home pose from config; refuse to start if the board is > 1 deg or > 5 mm from it; rate limiter starts from the measured tilt.
- Resync on `/sim/reset_done` and on a joint jump.
- Rotation convention: `R_tilt = Ry(tilt_x) * Rx(-tilt_y)` in the board frame; `R_d = R_home * R_tilt`; tilt read back from `log(R_home^T R)` as `(rv_y, -rv_x)`.
- base_link = UR DH base rotated by pi about z (as `maze_arm.kinematics.fk_tool0`).
- All gains, limits and topics are required parameters in `config/tilt_control.yaml` (no defaults; fail at startup naming the key), like `maze_planning`.
- `src/maze_arm`, `src/maze_control`, `src/maze_planning`, sim packages: unchanged.
- Code/docs/commits in English, no attribution lines.

## Review Focus

1. A sim reset (`/sim/reset`) while the node runs: controllers are briefly deactivated and joints jump; the node must not fight the reset nor wind up, and must resume tracking afterwards (sim test `test_reset_resync`).
2. `tilt_cmd` with NaN/inf: ignored, never reaches the arm (unit test in reference shaper).
3. Joint states arriving with names in another order or with extra joints: mapped by name (node unit test via `joints_from_state`).
4. Command stream stops mid-motion: velocities go to zero within 200 ms + one timer period, arm holds still (sim test `test_watchdog_stops`).
5. Startup away from home (e.g. arm left tilted): node refuses with an error naming the distance (unit test of `home_check`).

---

### Task 1: Package skeleton, kinematics, Jacobian

**Files:** create `src/maze_arm_control/{package.xml,CMakeLists.txt}`, `include/maze_arm_control/kinematics.hpp`, `src/kinematics.cpp`, `scripts/export_fk_fixture.py`, `test/data/fk_fixture.yaml`, `test/test_kinematics.cpp`.

**Produces:**
- `struct DhParams { std::array<double,6> d, a, alpha; };`
- `struct BoardMount { Eigen::Vector3d grasp_xyz, grasp_rpy, handle_center; };` (same arguments as `fk_board`)
- `using Joints = Eigen::Matrix<double,6,1>; using Jacobian = Eigen::Matrix<double,6,6>;`
- `class ArmKinematics { ArmKinematics(DhParams, BoardMount); Eigen::Isometry3d fk_tool0(const Joints&) const; Eigen::Isometry3d fk_board(const Joints&) const; Jacobian jacobian_board(const Joints&) const; };` Jacobian rows 0-2 = base-frame linear velocity of the board center, rows 3-5 = base-frame angular velocity.
- `Eigen::Matrix3d rpy_matrix(double r, double p, double y)` (Rz*Ry*Rx).

Steps:
- [ ] Fixture script: loads sim.yaml + parts_report.json, evaluates `maze_arm.kinematics.fk_board` on home + 20 seeded random poses (home +/- 1 rad) and writes `{grasp_xyz, grasp_rpy, handle_center, poses: [{q, T (16 row-major)}]}`.
- [ ] Tests: `FkMatchesPython` (all poses, 1e-9 m / 1e-9 elementwise), `JacobianMatchesFiniteDifference` (central differences of `fk_board`, h=1e-6, tol 1e-6, on all fixture poses, angular part via `log(Td R^T)/h`), `BaseLinkIsRotatedPi` (fk_tool0 at q=0 has x of tool0 position > 0, i.e. -a2-a3 mapped by Rz(pi)).
- [ ] Implement, build, run, commit.

### Task 2: SO(3) helpers and tilt convention

**Files:** `include/maze_arm_control/so3.hpp`, `src/so3.cpp`, `test/test_so3.cpp`.

**Produces:** `Eigen::Vector3d log_so3(const Eigen::Matrix3d&)`; `Eigen::Matrix3d tilt_rotation(double tilt_x, double tilt_y)`; `Eigen::Vector2d tilt_of(const Eigen::Matrix3d& R_home, const Eigen::Matrix3d& R)`; `double yaw_of(R_home, R)` (rv_z).

Steps:
- [ ] Tests: `LogOfExp` round trip incl. angle near 0 and near pi; `TiltRoundTrip` (tilt_of(R_home, R_home*tilt_rotation(tx,ty)) == (tx,ty) within 1e-4 rad for |t| <= 0.087; exact for single axis); `EdgeSignsFromFk` using ArmKinematics at home: board pose `R_home*tilt_rotation(+0.05,0)` lowers the +x edge (base z of R*(0.15,0,0) < 0) and `(0,+0.05)` lowers +y edge, and opposite for negatives.
- [ ] Implement, run, commit.

### Task 3: Reference shaper, controller law, watchdog (ROS-free)

**Files:** `include/maze_arm_control/{reference.hpp,tilt_controller.hpp,watchdog.hpp}`, `src/{reference.cpp,tilt_controller.cpp,watchdog.cpp}`, `test/{test_reference.cpp,test_controller_loop.cpp,test_watchdog.cpp}`.

**Produces:**
- `struct ReferenceParams { double max_tilt, max_rate, smoothing_tau; };` `class ReferenceShaper { void reset(Eigen::Vector2d measured); void set_command(Eigen::Vector2d) /* ignores non-finite, clamps */; void step(double dt); Eigen::Vector2d value() const; Eigen::Vector2d rate() const; };` order: clamp -> first-order low-pass (tau) -> rate limit; `rate()` = (value - previous)/dt.
- `struct ControllerParams { double k_pos, k_rot, sigma_min, lambda_max, max_linear_speed, max_angular_speed; Joints max_joint_speed, max_joint_accel; };`
- `class TiltController { TiltController(const ArmKinematics&, ControllerParams, Eigen::Isometry3d home); Joints step(const Joints& q, Eigen::Vector2d tilt, Eigen::Vector2d tilt_rate, double dt); void reset(); /* last qdot := 0 */ struct Diag { double pos_err, rot_err, sigma_min; bool vel_scaled, accel_limited, twist_capped; } diag() const; };` twist = [k_pos e_p; k_rot e_R + R_home*(-rate_y, rate_x, 0)]; cap |v|,|w|; DLS with lambda^2 = (1-(s/eps)^2)*lambda_max^2 below eps; scale whole qdot to joint speed limits; then scale whole delta-qdot so |dqdot_i| <= a_i*dt.
- `struct HomeCheck { bool ok; double pos_err, rot_err; }; HomeCheck home_check(const Eigen::Isometry3d& home, const Eigen::Isometry3d& board, double max_pos, double max_rot);`
- `class Watchdog { Watchdog(double cmd_timeout, double state_timeout); void on_command(double t); void on_state(double t); bool ok(double t) const; std::string reason(double t) const; };` no command yet = not ok.
- `bool joint_jump(const Joints& q_prev, const Joints& q, double threshold);`

Steps:
- [ ] Tests reference: clamp to 0.087; NaN ignored; step response rate never > 0.5 rad/s; reset starts from measured; smoothing makes a 60 Hz zero-order-hold staircase's rate continuous (max rate jump between consecutive 2 ms steps <= max_rate*dt/tau*2).
- [ ] Tests controller loop (integrate q += qdot*dt at 500 Hz from home, from the sweep): 5 deg steps on x, y, both; 0.5 Hz 5 deg sine on each axis; 60 Hz ZOH sine. Assert: center drift < 1 mm, tilt error < 0.5 deg 0.5 s after the last step change (sine: tracking error < 0.5 deg after 1 s), |qdot| <= limits, |dqdot|/dt <= accel limits, sigma_min > 0.1. Plus `HomeCheckRejects` (6 mm or 1.2 deg away -> not ok) and `JointJump`.
- [ ] Tests watchdog: 200 ms command timeout, 20 ms state timeout, no-command-yet.
- [ ] Implement, run, commit.

### Task 4: ROS node, config, launch

**Files:** `include/maze_arm_control/tilt_control_node.hpp`, `src/tilt_control_node.cpp`, `src/tilt_control_main.cpp`, `config/tilt_control.yaml`, `launch/tilt_control.launch.py`, `test/test_config.py` (yaml vs sim.yaml/parts_report), `test/test_node_params.cpp` (missing parameter throws naming it; `joints_from_state` by name).

Node `tilt_control`: required params `topics.{tilt_cmd,tilt_state,joint_states,velocity_command,reset_done}`, `controllers.{velocity,others}` (list to deactivate if active), `kinematics.{d,a,alpha}`, `mount.{grasp_xyz,grasp_rpy,handle_center}`, `home_joints`, `home_tolerance.{position,rotation}`, `limits.{max_tilt,max_rate}`, `reference.smoothing_tau`, `gains.{k_pos,k_rot}`, `dls.{sigma_min,lambda_max}`, `twist_cap.{linear,angular}`, `joint_limits.{max_speed,max_accel}`, `watchdog.{command_timeout,state_timeout,period}`, `joint_jump_threshold`, `speed_scaling_topic` ("" = none). State machine on a steady timer: WAIT_STATE -> CHECK_HOME -> SWITCH (list + STRICT switch, check ok) -> RUNNING; failure -> log error, `rclcpp::shutdown()`. RUNNING: control in the joint-state callback, watchdog on the steady timer publishes zeros while not ok, every 1 s check the velocity controller is still active (else HALTED, zeros); HALTED/RUNNING on `/sim/reset_done` -> re-enter WAIT_STATE (no re-switch if already active). tilt_state from every joint state (best effort, keep last 1, stamp = joint_state stamp, frame maze_board). Speed scaling < 1.0 -> zeros + warning. Yaw drift > 0.5 deg -> throttled warning. Destructor publishes zeros.

Steps: tests -> implement -> build -> commit.

### Task 5: Sim tests and figure

**Files:** `test/test_sim_tilt.py` (launch_testing, sim + node, registered with `add_launch_test`, TIMEOUT 600), `scripts/plot_tilt_run.py` (writes a PNG of commanded vs measured tilt and center motion from a recorded CSV).

Tests (one launch, name-ordered): `test_1_starts_and_switches` (velocity controller active, trajectory inactive); `test_2_contract4_tilt_x` / `test_3_contract4_tilt_y` (from `/sim/reset`, tilt +0.035 for 0.5 s: ball moves toward +x/+y in maze_board, board center < 5 mm); `test_4_step_tracking` (5 deg step each axis: settled error <= 0.5 deg, center < 5 mm); `test_5_sine_tracking` (0.5 Hz 3 deg sine on x: error <= 0.5 deg after 1 s, center < 5 mm, logs CSV); `test_6_watchdog_stops` (stop publishing: joint speeds ~0 within 0.3 s); `test_7_reset_resync`; `test_8_pid_drives_ball` (launch pid_node from maze_control; the test publishes /ball/state from ground truth and a straight /maze/path from ball start toward +x; ball progresses > 3 cm along the path within 5 s with center < 5 mm).

Steps: write tests -> run in distrobox -> plot -> commit.

### Task 6: Finish

- [ ] README section for `maze_arm_control` (run, parameters, real-robot notes: speed scaling, calibrated DH).
- [ ] Full `colcon test` in the distrobox (rerun `maze_sim_bringup` alone if `test_camera_rate` flakes).
- [ ] Whole-branch review by a fresh Opus reviewer; fix findings.
- [ ] `graphify update .`; commit; ask before any push.

## Changes made while building

- Reference shaper: an acceleration limit (`reference.max_accel`, 8 rad/s^2) that brakes in time to stop on the
  command, after the sim showed 5 deg steps overshooting to 6 deg (the arm lags the abrupt end of the 0.5 rad/s ramp).
  Low-pass tau is 15 ms (25 ms lagged a 0.5 Hz 5 deg sine by more than 0.5 deg).
- `/sim/reset_done`: re-check the controllers and resync, no new home check. Switching to the velocity controller
  makes the sim arm droop ~5 mm / 0.6 deg, and a reset with the velocity controller active leaves the arm where it
  was; a home check there refused to run. The home check stays at first start.
- After the whole-branch review: the reference starts inside max_tilt on a resync; the board's angular velocity
  is capped at max_rate about each board axis (the twist cap alone allowed 1 rad/s while correcting a large
  error); a controller_manager reply later than 5 s fails start-up (or is retried while running) instead of
  hanging; a stale speed scaling value (> 1 s) holds the arm; zeros are flushed before exit.
