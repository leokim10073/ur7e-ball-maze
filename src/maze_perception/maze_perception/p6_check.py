"""P6 checks on the real camera (#45, docs/perception_lab.md), against ball_tracker as it runs.

    ros2 run maze_perception p6_check stream [--seconds 120]
        Frame rate and drop-outs (camera_info stamps), board found, ball tracked, capture-to-receive
        latency of /ball/state. Pass: >= 55 fps, no gap > 0.2 s, latency <= 50 ms, and with the ball
        rolled by hand, tracking_ok >= 98%.
    ros2 run maze_perception p6_check spots [--radius 0.0125] [--level L1]
        The ball pushed into five corners of the maze, where two walls fix its center to within a
        fraction of a mm: the four corners of the maze and the wall corner nearest the middle. Each
        is averaged over 2 s of /ball/state. Pass: within 3 mm. Writes p6_spots.png, a map of where
        the corners are.

The arithmetic (frame statistics, the corner spots) is plain Python, tested without ROS.
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

Rect = Tuple[float, float, float, float]  # x_min, x_max, y_min, y_max, board frame, m

FPS_MIN, GAP_MAX, LATENCY_MAX, TRACKED_MIN, SPOT_MAX = 55.0, 0.2, 0.050, 0.98, 0.003


# --- arithmetic ---

def frame_stats(stamps: Sequence[float]) -> Dict[str, float]:
    """Frame rate and gaps from capture stamps (s)."""
    t = np.sort(np.asarray(stamps, float))
    if len(t) < 2:
        return {"frames": float(len(t))}
    gaps = np.diff(t)
    return {"frames": float(len(t)), "fps": (len(t) - 1) / (t[-1] - t[0]), "gap_median": float(np.median(gaps)),
            "gap_max": float(gaps.max()), "gaps_over": float(np.sum(gaps > GAP_MAX))}


def latency_stats(latencies: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(latencies, float)
    if not len(a):
        return {}
    return {"latency_median": float(np.median(a)), "latency_p95": float(np.percentile(a, 95)),
            "latency_max": float(a.max())}


def cad_wall_rects(cad_scripts, level: str = "L1") -> List[Rect]:
    """Every wall of the level as a board-frame rectangle (maze_CAD/scripts, as maze_cad_map reads it)."""
    from maze_cad_map.cad_map import load_cad

    params, levels = load_cad(cad_scripts)
    half = params.P["base"] / 2
    return [((x - half) / 1000, (x + w - half) / 1000, (y - half) / 1000, (y + h - half) / 1000)
            for x, y, w, h in params.wall_rects(levels.LEVELS[level])]


def _gap(x: float, y: float, rect: Rect) -> float:
    x0, x1, y0, y1 = rect
    return float(np.hypot(max(x0 - x, 0.0, x - x1), max(y0 - y, 0.0, y - y1)))


def corner_spots(walls: Sequence[Rect], radius: float, tol: float = 1e-6) -> List[Tuple[float, float]]:
    """Where a ball of `radius` sits pushed into a corner between a horizontal and a vertical wall,
    touching both and no other: every such center, board frame, m."""
    horiz = [r for r in walls if r[1] - r[0] > r[3] - r[2]]
    vert = [r for r in walls if r[3] - r[2] >= r[1] - r[0]]
    out = []
    for h in horiz:
        for v in vert:
            for x in (v[0] - radius, v[1] + radius):
                for y in (h[2] - radius, h[3] + radius):
                    if abs(_gap(x, y, h) - radius) > tol or abs(_gap(x, y, v) - radius) > tol:
                        continue  # the wall does not reach this side of the other
                    if any(_gap(x, y, r) < radius - tol for r in walls):
                        continue  # inside a wall, or another wall in the way
                    if not any(np.hypot(x - a, y - b) < 1e-4 for a, b in out):
                        out.append((x, y))
    return out


def check_spots(spots: Sequence[Tuple[float, float]]) -> List[Tuple[str, Tuple[float, float]]]:
    """The five spots of the corner test: the maze's four corners and the corner nearest the middle."""
    pts = np.asarray(spots)
    named = []
    for name, sx, sy in (("top-left (-x, +y)", -1, 1), ("top-right (+x, +y)", 1, 1),
                         ("bottom-right (+x, -y)", 1, -1), ("bottom-left (-x, -y)", -1, -1)):
        i = int(np.argmax(sx * pts[:, 0] + sy * pts[:, 1]))
        named.append((name, tuple(pts[i])))
    used = {n[1] for n in named}
    rest = [p for p in map(tuple, pts) if p not in used]
    middle = min(rest, key=lambda p: np.hypot(*p))
    named.append(("middle", middle))
    return named


def draw_spots(walls: Sequence[Rect], spots, path, px_per_m: float = 2000.0) -> None:
    """A top-down map (board +y up, handle on the left) of the walls and the numbered spots."""
    import cv2

    size = int(0.3 * px_per_m)
    img = np.full((size, size, 3), 235, np.uint8)

    def px(x, y):
        return int(round((x + 0.15) * px_per_m)), int(round((0.15 - y) * px_per_m))

    for x0, x1, y0, y1 in walls:
        cv2.rectangle(img, px(x0, y1), px(x1, y0), (60, 60, 60), -1)
    for k, (name, (x, y)) in enumerate(spots, 1):
        cv2.circle(img, px(x, y), int(0.0125 * px_per_m), (200, 90, 30), 2)
        cv2.putText(img, str(k), (px(x, y)[0] - 8, px(x, y)[1] + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 90, 30), 2)
    cv2.putText(img, "handle", (4, size // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
    cv2.imwrite(str(path), img)


# --- ROS ---

@dataclass
class _Collected:
    info_stamps: list
    states: list  # (stamp, receive time, x, y, tracking_ok)
    poses: int


def _collect(seconds: float) -> _Collected:
    import rclpy
    from maze_interfaces.msg import BallState
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import CameraInfo

    from maze_perception.board_node import SENSOR_QOS

    node = rclpy.create_node("p6_check")
    got = _Collected([], [], 0)

    def stamp(m):
        return m.header.stamp.sec + m.header.stamp.nanosec * 1e-9

    def on_state(m):
        got.states.append((stamp(m), time.time(), m.x, m.y, m.tracking_ok))

    def on_pose(_):
        got.poses += 1

    node.create_subscription(CameraInfo, "/camera/camera_info", lambda m: got.info_stamps.append(stamp(m)), SENSOR_QOS)
    node.create_subscription(BallState, "/ball/state", on_state, SENSOR_QOS)
    node.create_subscription(PoseStamped, "/perception/board_pose", on_pose, 10)
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.05)
    node.destroy_node()
    return got


def _verdict(ok: bool) -> str:
    return "pass" if ok else "FAIL"


def stream(seconds: float) -> int:
    print(f"listening {seconds:.0f} s (roll the ball by hand for the tracking number) ...")
    got = _collect(seconds)
    fs = frame_stats(got.info_stamps)
    if fs["frames"] < 2:
        print("no /camera/camera_info: is the camera running (real.launch.py)?")
        return 1
    print(f"camera      {fs['fps']:.1f} fps   [{_verdict(fs['fps'] >= FPS_MIN)}: >= {FPS_MIN:.0f}]")
    print(f"gaps        median {1000 * fs['gap_median']:.1f} ms, max {1000 * fs['gap_max']:.0f} ms, "
          f"{int(fs['gaps_over'])} over {1000 * GAP_MAX:.0f} ms   [{_verdict(fs['gaps_over'] == 0)}]")
    print(f"board       found in {100 * got.poses / fs['frames']:.0f}% of frames")
    if not got.states:
        print("no /ball/state: is ball_tracker running? (it needs /maze/map)")
        return 1
    tracked = sum(s[4] for s in got.states) / len(got.states)
    # capture time on this machine's clock (usb_cam stamps with it) to when the message got here
    ls = latency_stats([recv - st for st, recv, *_ in got.states])
    print(f"ball        tracking_ok in {100 * tracked:.1f}% of {len(got.states)} states   "
          f"[{_verdict(tracked >= TRACKED_MIN)}: >= {100 * TRACKED_MIN:.0f}%]")
    print(f"latency     capture to here: median {1000 * ls['latency_median']:.1f} ms, "
          f"p95 {1000 * ls['latency_p95']:.1f}, max {1000 * ls['latency_max']:.1f}   "
          f"[{_verdict(ls['latency_p95'] <= LATENCY_MAX)}: <= 50 ms]")
    print("(the camera's own exposure and USB delay come on top; measure total latency against the arm, #32 stage 2)")
    return 0


def spots(radius: float, cad_scripts: str, level: str, seconds: float = 2.0) -> int:
    walls = cad_wall_rects(cad_scripts, level)
    named = check_spots(corner_spots(walls, radius))
    draw_spots(walls, named, "p6_spots.png")
    print(f"{len(named)} spots for a {2000 * radius:.0f} mm ball (map: p6_spots.png; board +y up, handle on -x):")
    for k, (name, (x, y)) in enumerate(named, 1):
        print(f"  {k}. {name:22} x {1000 * x:+7.1f} mm, y {1000 * y:+7.1f} mm")
    results = []
    for k, (name, (x, y)) in enumerate(named, 1):
        input(f"\n{k}. Push the ball into the {name} corner (touching both walls, board level), then Enter ")
        got = _collect(seconds)
        pts = np.array([(s[2], s[3]) for s in got.states if s[4]])
        if not len(pts):
            print("   no tracked /ball/state: ball not found")
            results.append(float("inf"))
            continue
        m, sd = pts.mean(axis=0), pts.std(axis=0)
        err = float(np.hypot(m[0] - x, m[1] - y))
        results.append(err)
        print(f"   measured x {1000 * m[0]:+7.1f}, y {1000 * m[1]:+7.1f} mm (jitter {1000 * sd.max():.2f} mm), "
              f"off by {1000 * err:.1f} mm   [{_verdict(err <= SPOT_MAX)}: <= 3 mm]")
    print(f"\n{sum(e <= SPOT_MAX for e in results)}/{len(results)} spots within 3 mm")
    return 0 if all(e <= SPOT_MAX for e in results) else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description="P6 checks on the real camera (#45)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stream", help="frame rate, gaps, board, tracking, latency")
    s.add_argument("--seconds", type=float, default=120.0)
    p = sub.add_parser("spots", help="ball pushed into five maze corners against the CAD")
    p.add_argument("--radius", type=float, default=0.0125, help="ball radius, m (the 25 mm marble)")
    p.add_argument("--cad-scripts", default="maze_CAD/scripts")
    p.add_argument("--level", default="L1")
    p.add_argument("--seconds", type=float, default=2.0, help="averaging time per spot")
    args = ap.parse_args(argv)

    import rclpy

    rclpy.init()
    try:
        if args.cmd == "stream":
            return stream(args.seconds)
        if not Path(args.cad_scripts).is_dir():
            print(f"{args.cad_scripts} not found: run from the repo root or pass --cad-scripts")
            return 1
        return spots(args.radius, args.cad_scripts, args.level, args.seconds)
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
