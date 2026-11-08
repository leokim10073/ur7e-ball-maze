# maze_control

Ball controllers for the UR7e ball-in-maze. This version is the PID baseline.

## Layout

| File | What it does |
|---|---|
| `maze_control/path_tracker.py` | Projects the ball onto the path (windowed, so U-turns don't jump legs) and returns the point `lookahead` ahead; holds the goal once within `goal_tol`. |
| `maze_control/pid_controller.py` | Two-axis PID. D on the KF velocity, I with integral zone + clamp + conditional integration; tilt clip, then rate limit. |
| `maze_control/controller.py` | Tracker + PID + state watchdog (no `/ball/state` for `state_timeout` → ramp tilt to 0). |
| `maze_control/ball_sim.py` | Offline sim: `a = (5/7) g sin θ − c v`, command delay `d`, first-order tilt lag `τ`. |
| `maze_control/pid_node.py` | Thin ROS 2 wrapper. |
| `config/pid.yaml` | All gains, limits, signs, topics. |
| `config/sim.yaml` | Simulator plant parameters. |

The core modules only need numpy (and pyyaml for the configs); none of them import ROS.

## Interface

| Topic | Type | Notes |
|---|---|---|
| `/ball/state` (in) | `nav_msgs/Odometry` | `pose.pose.position.x/y` [m], `twist.twist.linear.x/y` [m/s], board frame |
| `/maze/path` (in) | `nav_msgs/Path` | board frame [m]; a new path restarts tracking |
| `/board/tilt_cmd` (out) | `geometry_msgs/Vector3Stamped` | `x = θx`, `y = θy` [rad], 60 Hz |

Signs: `θy > 0` pushes the ball to +x and `θx > 0` pushes it to −y. Both are set by `sign.*` in `config/pid.yaml`.

## Tests and plots (no ROS)

```bash
cd ros_ws/src/maze_control
python3 -m pytest test -s
python3 scripts/plot_trajectories.py   # -> figures/pid_trajectories.png
```

## Run on ROS 2

```bash
cd ros_ws && colcon build --packages-select maze_control && source install/setup.bash
ros2 launch maze_control pid_controller.launch.py   # config:=/path/to/other.yaml
```
