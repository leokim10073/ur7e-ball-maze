"""Offline replay (P5): the perception pipeline over recorded camera frames, without a running ROS
graph. Same steps as ball_tracker (markers, board pose, ball, filter), frame by frame, so a
rosbag from the sim or the robot can be rerun after every change and compared.

    ros2 run maze_perception replay_bag <bag> [--csv out.csv] [--image-topic ...] [--info-topic ...]

The bag needs the camera images (sensor_msgs/Image or CompressedImage) and camera_info. The walls
come from the bag's /maze/map if it was recorded, else from the CAD (--cad-scripts, --level). If
the bag has the sim's /sim/ball/pose and /sim/board/pose, the ball's true position in the board
frame is put next to each frame and the errors are summed up.

The bag is read twice: first everything but the images, then the images one at a time, so a long
bag never has to fit in memory.
"""

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from maze_perception.ball import WallGrid, locate_ball
from maze_perception.ball_filter import BallEstimate, BallFilter
from maze_perception.board_pose import Camera, estimate_board_pose
from maze_perception.config import PerceptionConfig, load_perception_config
from maze_perception.markers import MarkerDetector

TRUTH_MAX_GAP = 0.02  # s: a truth sample farther than this from a frame is not used for it


@dataclass(frozen=True)
class Frame:
    stamp: float  # s, capture time
    image: np.ndarray  # RGB
    camera: Camera


@dataclass(frozen=True)
class FrameResult:
    stamp: float
    board_found: bool
    measured: Optional[Tuple[float, float]]  # the ball this frame (P3), board frame, m
    score: float  # its score, nan if none
    estimate: BallEstimate  # the filter's state (P4), what /ball/state carries
    truth: Optional[Tuple[float, float]]  # the sim's ball position in the board frame, if recorded


class Truth:
    """The sim's ball position in the board frame over time, from world-frame ball and board poses
    ((stamp, position (3,), quaternion (x, y, z, w)) each)."""

    def __init__(self, ball: Sequence, board: Sequence):
        self.ball = sorted(ball, key=lambda s: s[0])
        self.board = sorted(board, key=lambda s: s[0])
        self._tb = np.array([s[0] for s in self.board])
        self._tball = np.array([s[0] for s in self.ball])

    @staticmethod
    def _nearest(times, samples, t):
        if len(times) == 0:
            return None
        i = int(np.clip(np.searchsorted(times, t), 1, len(times) - 1)) if len(times) > 1 else 0
        if len(times) > 1 and abs(times[i - 1] - t) < abs(times[i] - t):
            i -= 1
        return samples[i] if abs(times[i] - t) <= TRUTH_MAX_GAP else None

    def at(self, t: float) -> Optional[Tuple[float, float]]:
        ball = self._nearest(self._tball, self.ball, t)
        board = self._nearest(self._tb, self.board, t)
        if ball is None or board is None:
            return None
        x, y, z, w = board[2]
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        p = R.T @ (np.asarray(ball[1], float) - np.asarray(board[1], float))
        return float(p[0]), float(p[1])


def run(frames: Iterable[Frame], cfg: PerceptionConfig, walls: Optional[WallGrid],
        truth: Optional[Truth] = None) -> Iterator[FrameResult]:
    """ball_tracker's per-frame work over `frames` (in capture order), one result per frame
    (frames not newer than the previous one are skipped, as the node does)."""
    detector = MarkerDetector(cfg.board.dictionary, ids=cfg.board.markers)
    filt = BallFilter(cfg.tracking)
    for f in frames:
        pose = estimate_board_pose(detector.detect(f.image), cfg.board, f.camera, cfg.pose)
        found = None
        if pose is not None and walls is not None:
            found = locate_ball(f.image, pose, f.camera, cfg.ball, cfg.rectified, walls)
        est = filt.step(f.stamp, None if found is None else (found.x, found.y))
        if est is None:
            continue
        yield FrameResult(stamp=f.stamp, board_found=pose is not None,
                          measured=None if found is None else (found.x, found.y),
                          score=float("nan") if found is None else found.score, estimate=est,
                          truth=None if truth is None else truth.at(f.stamp))


def summarize(results: List[FrameResult]) -> dict:
    """Shares of frames with the board, the ball and tracking ok; with truth, the position errors
    of the measurement and of the filtered state (m)."""
    n = len(results)
    out = {"frames": n}
    if not n:
        return out
    out["board"] = sum(r.board_found for r in results) / n
    out["ball"] = sum(r.measured is not None for r in results) / n
    out["tracking_ok"] = sum(r.estimate.tracking_ok for r in results) / n
    if len(results) > 1:
        out["fps"] = (n - 1) / (results[-1].stamp - results[0].stamp)
    for key, pick in (("measured", lambda r: r.measured),
                      ("state", lambda r: (r.estimate.x, r.estimate.y) if r.estimate.tracking_ok else None)):
        errs = [np.hypot(p[0] - r.truth[0], p[1] - r.truth[1]) for r in results
                if r.truth is not None and (p := pick(r)) is not None]
        if errs:
            out[f"{key}_error_rms"] = float(np.sqrt(np.mean(np.square(errs))))
            out[f"{key}_error_max"] = float(np.max(errs))
    return out


def write_csv(results: Iterable[FrameResult], path) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stamp", "board_found", "measured_x", "measured_y", "score", "x", "y", "vx", "vy",
                    "var_x", "var_y", "tracking_ok", "truth_x", "truth_y"])
        for r in results:
            e = r.estimate
            m = r.measured or ("", "")
            t = r.truth or ("", "")
            score = "" if np.isnan(r.score) else f"{r.score:.3f}"
            w.writerow([f"{r.stamp:.6f}", int(r.board_found), *m, score, e.x, e.y, e.vx, e.vy,
                        e.position_covariance[0], e.position_covariance[3], int(e.tracking_ok), *t])


# --- rosbag2 (needs ROS 2; everything above does not) ---

def _stamp(msg) -> float:
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def _open(bag: str, topics: Optional[set] = None):
    import rosbag2_py

    last = None
    for storage in ("sqlite3", "mcap"):
        try:
            reader = rosbag2_py.SequentialReader()
            reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id=storage),
                        rosbag2_py.ConverterOptions("cdr", "cdr"))
            break
        except RuntimeError as e:  # wrong storage plugin for this bag
            last = e
    else:
        raise RuntimeError(f"cannot open {bag}: {last}")
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if topics is not None:
        reader.set_filter(rosbag2_py.StorageFilter(topics=[t for t in topics if t in types]))
    return reader, types


def _messages(bag: str, topics: set):
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader, types = _open(bag, topics)
    classes = {t: get_message(types[t]) for t in topics if t in types}
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic in classes:
            yield topic, deserialize_message(data, classes[topic])


def _rgb(msg) -> np.ndarray:
    from maze_perception.ros_image import image_to_rgb

    if hasattr(msg, "format"):  # CompressedImage
        bgr = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"cannot decode a {msg.format!r} image")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return image_to_rgb(msg)


def read_bag(bag: str, image_topic: str, info_topic: str, map_topic: str = "/maze/map",
             ball_topic: str = "/sim/ball/pose", board_topic: str = "/sim/board/pose"):
    """(frames, walls or None, truth or None): frames is an iterator that reads the images on demand."""
    from maze_perception.ball_node import walls_from_map

    _, types = _open(bag)
    if image_topic not in types:
        raise ValueError(f"{image_topic} is not in {bag} (topics: {', '.join(sorted(types))})")
    infos, walls, ball, board = [], None, [], []
    for topic, msg in _messages(bag, {info_topic, map_topic, ball_topic, board_topic}):
        if topic == info_topic:
            infos.append((_stamp(msg), Camera.from_camera_info(msg.k, msg.d, msg.width, msg.height)))
        elif topic == map_topic:
            walls = walls_from_map(msg)
        else:
            p, q = msg.pose.position, msg.pose.orientation
            (ball if topic == ball_topic else board).append((_stamp(msg), (p.x, p.y, p.z), (q.x, q.y, q.z, q.w)))
    if not infos:
        raise ValueError(f"no {info_topic} in {bag}")
    info_t = np.array([t for t, _ in infos])
    truth = Truth(ball, board) if ball and board else None

    def frames():
        for _, msg in _messages(bag, {image_topic}):
            t = _stamp(msg)
            i = max(0, int(np.searchsorted(info_t, t, side="right")) - 1)  # the latest camera_info
            yield Frame(stamp=t, image=_rgb(msg), camera=infos[i][1])

    return frames(), walls, truth


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run the ball tracker over a rosbag, offline.")
    ap.add_argument("bag")
    ap.add_argument("--image-topic", default="/camera/image_raw")
    ap.add_argument("--info-topic", default="/camera/camera_info")
    ap.add_argument("--config", default=None, help="perception.yaml (default: the installed one)")
    ap.add_argument("--cad-scripts", default="maze_CAD/scripts",
                    help="maze_CAD/scripts, for the walls if the bag has no /maze/map")
    ap.add_argument("--level", default="L1")
    ap.add_argument("--csv", default=None, help="write one row per frame here")
    args = ap.parse_args(argv)

    if args.config is None:
        from ament_index_python.packages import get_package_share_directory

        args.config = Path(get_package_share_directory("maze_perception")) / "config" / "perception.yaml"
    cfg = load_perception_config(args.config)
    frames, walls, truth = read_bag(args.bag, args.image_topic, args.info_topic)
    if walls is None:
        from maze_cad_map.cad_map import build_map

        m = build_map(args.cad_scripts, args.level)
        walls = WallGrid(cells=m.grid != 0, origin=m.origin, resolution=m.resolution)
        print(f"no /maze/map in the bag: walls from the CAD ({args.cad_scripts}, {args.level})")
    results = list(run(frames, cfg, walls, truth))
    if args.csv:
        write_csv(results, args.csv)
    s = summarize(results)
    print(f"{s['frames']} frames" + (f", {s['fps']:.1f} per second" if "fps" in s else ""))
    if s["frames"]:
        print(f"board found {100 * s['board']:.0f}%, ball {100 * s['ball']:.0f}%, "
              f"tracking ok {100 * s['tracking_ok']:.0f}%")
    for key in ("measured", "state"):
        if f"{key}_error_rms" in s:
            print(f"{key} vs /sim/ball/pose: {1000 * s[f'{key}_error_rms']:.2f} mm rms, "
                  f"{1000 * s[f'{key}_error_max']:.2f} mm max")
    return 0


if __name__ == "__main__":
    sys.exit(main())
