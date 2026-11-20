# Interfaces v0

This file is the contract between our own modules: what each one publishes, in which frame and units, how often, and what happens when something goes wrong. The message definitions live in `src/maze_interfaces/msg/`; this file explains them.

Robot-side topics (`/joint_states`, `/forward_velocity_controller/commands`, `/camera/image_raw`, ...) are fixed by the lab driver and copied by the sim; see the README's [Simulation → Interfaces](../README.md#interfaces) table. Perception's board pose topics are in the README's [Perception](../README.md#perception) section.

- Last updated: 2026-11-19
- Status: v0, reviewed
- Owner: Leo Chen (from Jing Xu)
- Reviewers: Zhuoren (perception), Steven (ball control), Chelsea (arm control), Leo (sim)

## Shared rules

- **Board frame** `maze_board`**.** Origin at the center of the maze area, z up, z = 0 on the maze floor, handle on the −x edge, metres. Same as the sim, the CAD (`maze_CAD/README.md`) and perception's TF frame. Every message between our modules uses it.
- **Units.** SI only: m, m/s, rad, rad/s, s. The CAD is in mm; convert once, where it is read.
- **Time.** Every message has a `std_msgs/Header`. In the sim every node runs with `use_sim_time:=true`.
- **No magic numbers.** Ball radius, board size and tilt limits come from config or from a message, never from constants in code.
- **Messages** live in the `maze_interfaces` package. Standard ROS types are used inside them (`nav_msgs/OccupancyGrid`, `nav_msgs/Path`, `geometry_msgs/Point`) so RViz and ROS tools work.
- **Topic names** follow the ones ball control (#6) and arm control (#14) already use: `/maze/`* for the maze and route, `/ball/*` for the ball, `/board/*` for the board tilt.
- **One publisher per topic.** For example, the sim stand-in and the ball tracker both publish `/ball/state`; never run both.

## Overview

| # | Contract | Topic | Type | Publisher → Subscribers | Rate | QoS |
|---|---|---|---|---|---|---|
| 1 | Maze map | `/maze/map` | `maze_interfaces/MazeMap` | CAD map tool → planner | Once per maze | reliable, transient local |
| 2 | Ball state | `/ball/state` | `maze_interfaces/BallState` | perception P3/P4 → planner, ball control | Every camera frame, ~60 Hz | best effort, last 1 |
| 3 | Planned route | `/maze/route` (+ `/maze/path`) | `maze_interfaces/PlannedRoute` (+ `nav_msgs/Path`) | planner → ball control | On each plan or replan | reliable, transient local |
| 4a | Tilt command | `/board/tilt_cmd` | `maze_interfaces/BoardTilt` | ball control → arm control | Control rate, ~60 Hz | reliable, last 1 |
| 4b | Measured tilt | `/board/tilt_state` | `maze_interfaces/BoardTilt` | arm control → ball control, system ID | ≥ 60 Hz | best effort, last 1 |

```
 CAD map ────────────────(1) /maze/map──► planner ──(3) /maze/route──► ball control ──(4a) /board/tilt_cmd──► arm control ──► UR7e / sim
 camera ──► perception                      ▲                            ▲    ▲                                   │
               └──────────(2) /ball/state───┴────────────────────────────┘    └──────(4b) /board/tilt_state───────┘
```

"Latched" below means reliable + transient local durability: a node that starts later (or restarts) still receives the last message, like a notice pinned on a board.

---

## Contract 1: Maze map

**Publisher:** the CAD map tool (`maze_cad_map`, interfaces I3), in the sim and on the real board.

**The map always comes from the CAD; perception does not map the maze.** The sim board and the printed board are both built from the same CAD files, so the CAD map is exact for both. A map read from the camera could only add error. Perception's part is the board pose (ArUco), which places the CAD map in the camera image, and the ball state (contract 2). What is left is how well the printed board matches the CAD: check it in the fabrication trial (#4) and cover it with the planner's wall margin.

**Subscribers:** planner.

**Topic:** `/maze/map` (the CAD tool also publishes the walls alone on `/maze/map/walls` as a `nav_msgs/OccupancyGrid`, for RViz).

**Type:** `maze_interfaces/MazeMap`

```
# MazeMap.msg
std_msgs/Header header        # frame_id = "maze_board"; stamp = the time it was published
string source                 # "cad:L1", "cad:L2", ...
nav_msgs/OccupancyGrid walls  # 0 = free floor, 100 = wall or outside the outer walls, -1 = unknown
                              # (the planner treats -1 as wall). Holes are NOT in the grid.
                              # resolution 0.005 m; 60 x 60 cells cover the 300 x 300 mm maze area
                              # info.origin = lower-left corner (-0.15, -0.15, 0) in maze_board
                              # row 0 = bottom (-y) edge; cell (row, col) = data[row * width + col]
                              # cell center: x = -0.15 + (col + 0.5) * 0.005, same for y with row
Circle[] holes                # trap holes the ball must avoid (L1: none). Not the goal.
geometry_msgs/Point start     # ball start, z = 0, m
Circle goal                   # goal pocket, from the level's CAD (not detected)

# Circle.msg
float64 x        # center in maze_board, m
float64 y
float64 radius   # m
```

**Why a grid for walls and circles for holes:** A* runs directly on a grid, and RViz draws it. Holes and the goal are round, so a circle is exact, and the planner chooses its own clearance from them. The grid's rows count from the bottom, as in every ROS tool; the CAD counts rows from the top, so the CAD tool flips them.

**The grid covers the maze area, not the whole board.** The board is larger: the ArUco markers sit at y = ±0.179 m (`maze_perception/config/perception.yaml`), outside the grid. Size the grid from the maze area. It matches perception's rectified image `/perception/board_image` (the same ±0.15 m square at 2 px/mm), so each 5 mm cell is a 10 × 10 px block of that image.

**The goal is given, not detected.** Start and goal come from the level's CAD (`maze_CAD/scripts/levels.py`); the ArUco board pose places them in the image. The green ring around the goal pocket (#4) is only a visual cue: perception may check that it is where the CAD says, but the published `goal` is the CAD value.

**Rate:** once per maze, after the map is built. Published again only if the map changes (the planner then replans).

**QoS:** latched (reliable, keep last 1, transient local). There is only one message, so it must not be lost.

**Edge cases:**

- Level not in `levels.py`, or the CAD files not found: publish no map and log an error. Never publish a partial map.
- No map received yet: the planner does not plan; ball control keeps the board level.

**How to check:**

- `maze_cad_map`'s tests: the CAD map's walls, start and goal match the sim export (`parts_report.json`) and `sim.yaml`.
- In RViz, `/maze/map/walls` drawn over the sim board: walls line up within one cell (5 mm).

**Example message (L1 from the CAD tool):**

```yaml
header: {stamp: {sec: 3, nanosec: 0}, frame_id: maze_board}
source: cad:L1
walls:
  header: {frame_id: maze_board}
  info: {resolution: 0.005, width: 60, height: 60,
         origin: {position: {x: -0.15, y: -0.15, z: 0.0}, orientation: {w: 1.0}}}
  data: [100, 100, 100, ..., 100, 0, 0, ...]   # 3600 values, row 0 (bottom edge) first
holes: []
start: {x: -0.12, y: 0.12, z: 0.0}
goal: {x: 0.12, y: -0.072, radius: 0.015}
```

---

## Contract 2: Ball state

**Publisher:** ball tracker in `maze_perception` (Zhuoren, #13 P3/P4). Until it exists, ball control's stand-in turns `/sim/ball/pose` + `/sim/board/pose` into this message (#6).

**Subscribers:** planner (to check the ball is still in its corridor), ball controllers (PID, LQR, MPC).

**Topic:** `/ball/state`

**Type:** `maze_interfaces/BallState`

```
# BallState.msg
std_msgs/Header header   # stamp = capture time of the camera image this came from (not publish time)
                         # frame_id = "maze_board"
float64 x                # ball center in maze_board, m (its z = ball radius is dropped)
float64 y                # m
float64 vx               # m/s, from the Kalman filter
float64 vy               # m/s
float64[4] position_covariance   # 2x2 [xx, xy, yx, yy], m^2; all zeros if unknown
bool tracking_ok         # false = ball or board not found in this frame; x, y, vx, vy are the filter's prediction
```

**Rate:** one message per camera frame. About 60 Hz on the robot, 56–62.5 Hz in the sim.

**Timing:** the message is the state at `header.stamp`, the image capture time. A controller that wants the present state predicts forward itself: `x + vx * (now - stamp)` (proposal: "Kalman filter estimates velocity and compensates for delay"). Perception logs the capture-to-publish latency, so we know how big the gap is.

**Ball center, not floor point:** the center is one ball radius above the floor. The tracker solves for the center at z = r in `maze_board`; intersecting the camera ray with z = 0 instead would be off by a few mm on a tilted board.

**QoS:** best effort, keep last 1. Old ball states are useless; only the newest matters. (`board_node.py` uses the same `SENSOR_QOS`.)

**Edge cases:**

- Ball not detected, or board not found (fewer than 3 ArUco markers): still publish, with the filter's prediction and `tracking_ok = false`.
- One rule for subscribers: `tracking_ok` false for more than `tracking_timeout` (0.2 s, about 12 frames at 60 Hz; a config value, not a constant) → command a level board and stop the run (proposal: "predict briefly, then stop the trial if tracking remains lost").
- No message for `tracking_timeout`: the tracker has died; same reaction.

**How to check:** in the sim, convert `/sim/ball/pose` (world frame) into `maze_board` using `/sim/board/pose`; `x, y` must be within a few mm (#13's done-when).

**Example message (L1 start cell, ball at rest):**

```yaml
header: {stamp: {sec: 12, nanosec: 400000000}, frame_id: maze_board}
x: -0.12
y: 0.12
vx: 0.0
vy: 0.0
position_covariance: [1.0e-6, 0.0, 0.0, 1.0e-6]
tracking_ok: true
```

---

## Contract 3: Planned route

**Publisher:** planner.

**Subscribers:** ball controllers: PID (#6) follows the path; constrained MPC (#10) also uses the corridors as constraints; MPCC (#11) uses the arc length as its progress variable.

**Topic:** `/maze/route`. The planner also publishes the same path alone on `/maze/path` as a `nav_msgs/Path`, the topic `pid_node` already reads (#6) and RViz can draw.

**Type:** `maze_interfaces/PlannedRoute`

```
# PlannedRoute.msg
uint8 STATUS_OK = 0
uint8 STATUS_NO_PATH = 1           # no route exists: path and corridors are empty

std_msgs/Header header             # frame_id = "maze_board"; stamp = when it was planned
uint32 plan_id                     # +1 on every new plan; a controller resets its progress when it changes
uint8 status
nav_msgs/Path path                 # ball-center waypoints from the ball's position to the goal, z = 0;
                                   # path.header.frame_id = "maze_board"; pose orientations unused
float64[] arc_length               # distance along the path from its first pose, m; one per path pose
Corridor[] corridors               # in path order; consecutive corridors overlap
uint32[] pose_corridor             # index into corridors for each path pose

# Corridor.msg: a convex safe region for the BALL CENTER, axis-aligned in maze_board, m.
# Wall clearance (ball radius + margin) is already taken off, so a controller can use the bounds
# directly as constraints on the ball state's x and y.
float64 x_min
float64 x_max
float64 y_min
float64 y_max
```

**Why corridors bound the ball center:** the MPC constrains the ball state `(x, y)`, which is the center. If the planner already subtracts the ball radius and a margin from the walls, the controller needs no geometry of its own, and the clearance lives in one place (planner config).

**Rate:** on every plan or replan: once when the map and a ball state are available, then each time the ball leaves its corridor.

**QoS:** latched (reliable, keep last 1, transient local), so a controller that restarts gets the current route.

**Edge cases:**

- No route exists: publish `status = STATUS_NO_PATH` with empty `path` and `corridors`; ball control keeps the board level.
- Ball leaves its corridor: the planner replans from the ball's position and publishes a new route with `plan_id + 1`. A controller seeing a new `plan_id` starts tracking from the new path's start.
- No route received yet: ball control keeps the board level.

**How to check:** on the CAD map of L1, the route passes through the cells of `levels.py`'s `path`; every path pose lies inside its `pose_corridor` corridor; consecutive corridors overlap; in RViz the path on `/maze/path` stays inside the maze.

**Example message (two corridors, three poses):**

```yaml
header: {stamp: {sec: 4, nanosec: 0}, frame_id: maze_board}
plan_id: 1
status: 0
path:
  header: {frame_id: maze_board}
  poses:
    - pose: {position: {x: -0.12, y: 0.12, z: 0.0}}
    - pose: {position: {x: -0.072, y: 0.12, z: 0.0}}
    - pose: {position: {x: -0.072, y: 0.072, z: 0.0}}
arc_length: [0.0, 0.048, 0.096]
corridors:
  - {x_min: -0.13, x_max: -0.062, y_min: 0.11, y_max: 0.13}
  - {x_min: -0.082, x_max: -0.062, y_min: 0.062, y_max: 0.13}
pose_corridor: [0, 0, 1]
```

**TODO:** set the ball radius for the corridor clearance once the real ball is bought (sim: 7 mm).

---

## Contract 4: Board tilt (command and measurement)

**4a, command.** Publisher: ball controllers (Steven, #5). Subscriber: arm control (Chelsea, #14). Topic `/board/tilt_cmd`.

**4b, measurement.** Publisher: arm control, from forward kinematics of `/joint_states` (#14). Subscribers: ball controllers, system ID logging (#7). Topic `/board/tilt_state`.

**Type (both):** `maze_interfaces/BoardTilt`

```
# BoardTilt.msg
# Defined by the effect on the ball, relative to the board's level home pose:
#   tilt_x > 0 lowers the board's +x edge, so the ball accelerates toward +x:  a_x = 5/7 g sin(tilt_x)
#   tilt_y > 0 lowers the board's +y edge, so the ball accelerates toward +y:  a_y = 5/7 g sin(tilt_y)
# As rotations of maze_board (right-handed): tilt_x = +angle about board +y,
#                                            tilt_y = -angle about board +x.
std_msgs/Header header   # stamp = when the command was computed (cmd) or measured (state);
                         # frame_id = "maze_board"
float64 tilt_x           # rad
float64 tilt_y           # rad
```

**Why the sign is defined by its effect:** with right-hand rotations alone, a positive rotation about +y sends the ball to +x but a positive rotation about +x sends it to −y. Writing it as "the ball goes toward +x when `tilt_x > 0`" makes both axes match the ball model of #7, so a controller can write `tilt_x = k * error_x` with no sign flips. The rotation line is there for arm control.

**Angles, not rates:** ball control thinks in tilt angles; arm control turns the angle error into joint velocities (resolved-rate control, #14). The board tilts about its own center (the remote pivot); the board center should stay put.

**Rate:** one command per control step, about 60 Hz (the camera rate). Measured tilt at least as fast.

**QoS:** command reliable, keep last 1 (only the newest command matters, but it should arrive). Measured tilt best effort, keep last 1.

**Limits and edge cases:**

- Both sides clamp: the controller to its own `max_tilt` and tilt-rate limits, arm control again to its hard limits (safety). Hard limits in arm control: 0.087 rad (5°) tilt, revisited after the grip test (#4), and 0.5 rad/s tilt rate. The controller stays below them so it never hits the clamp, e.g. 0.3 rad/s. All of these are config values and will be updated after system ID (#7).
- No command for 200 ms: arm control stops the arm (zero joint velocities) and logs a warning, the same timeout as the PID watchdog (#6, #14).
- `tilt_state` comes from forward kinematics, not from `/perception/board_pose`, so it does not wait on the camera.

**How to check:** in the sim, from `/sim/reset`, publish `tilt_x = +0.035` (2°) for 0.5 s: the ball (`/sim/ball/pose` converted to `maze_board`) moves toward +x, and `/sim/board/pose` shows the board center moving less than 5 mm (#14's done-when). Repeat for `tilt_y`.

**Example message:**

```yaml
header: {stamp: {sec: 15, nanosec: 16000000}, frame_id: maze_board}
tilt_x: 0.035
tilt_y: -0.010
```

---

## Changelog

- v0 (2026-11-12): first draft. Contracts 1–4, `maze_interfaces` messages, CAD map tool.
- v0 (2026-11-18): review. Contract 1: goal from CAD, grid = maze area. Contract 2: one `tracking_timeout` (0.2 s), publish a prediction when the board is lost, ball center at z = r, controller predicts forward from `header.stamp`. Contract 4: hard limits 5° and 0.5 rad/s. Confirmed by Steven: `/maze/path` and the tilt sign.
- v0 (2026-11-19): Contract 1: the map always comes from the CAD, in the sim and on the real board; perception no longer maps the maze.
