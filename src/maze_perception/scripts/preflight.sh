#!/usr/bin/env bash
# Check the lab machine before running perception on the real camera (P6, docs/perception_lab.md).
# Read-only: installs nothing, moves nothing. Run from the repo root after colcon build and
# `source install/setup.bash`:
#   ros2 run maze_perception preflight.sh
# Prints one line per check; exits 1 if any required check fails.
set -uo pipefail

fail=0
ok()   { echo "  ok    $*"; }
bad()  { echo "  FAIL  $*"; fail=1; }
warn() { echo "  warn  $*"; }

echo "ROS"
[[ "${ROS_DISTRO:-}" == "humble" ]] && ok "ROS 2 humble sourced" || bad "ROS_DISTRO='${ROS_DISTRO:-}', want humble (source /opt/ros/humble/setup.bash)"
for pkg in maze_perception maze_interfaces maze_cad_map; do
  ros2 pkg prefix "$pkg" >/dev/null 2>&1 && ok "$pkg built" || bad "$pkg not found: colcon build && source install/setup.bash"
done
ros2 pkg prefix usb_cam >/dev/null 2>&1 && ok "usb_cam" || bad "usb_cam missing: sudo apt install ros-humble-usb-cam"
ros2 pkg prefix camera_calibration >/dev/null 2>&1 && ok "camera_calibration" \
  || warn "camera_calibration missing (only for step 3): sudo apt install ros-humble-camera-calibration"
ros2 pkg prefix compressed_image_transport >/dev/null 2>&1 && ok "compressed images (for recording)" \
  || warn "no compressed_image_transport: bags hold raw images, 160 MB/s (sudo apt install ros-humble-image-transport-plugins)"

echo "Python"
python3 - <<'EOF' && ok "OpenCV with the new ArUco detector" || bad "OpenCV >= 4.7 needed (P1): pip install --user 'opencv-contrib-python-headless==4.10.0.84' 'numpy<2'"
import cv2, sys
major, minor = (int(v) for v in cv2.__version__.split(".")[:2])
print(f"        cv2 {cv2.__version__}")
sys.exit(0 if (major, minor) >= (4, 7) and hasattr(cv2.aruco, "ArucoDetector") else 1)
EOF
python3 -c "import numpy; assert int(numpy.__version__.split('.')[0]) < 2" 2>/dev/null \
  && ok "numpy < 2 (ROS humble's)" || warn "numpy >= 2: ROS humble packages may break (pip install --user 'numpy<2')"

echo "Camera"
if command -v v4l2-ctl >/dev/null; then
  ok "v4l2-ctl"
  dev="$(v4l2-ctl --list-devices 2>/dev/null | awk '/C922/ {getline; print $1; exit}')"
  if [[ -n "$dev" ]]; then
    ok "C922 at $dev (pass video_device:=$dev)"
    v4l2-ctl -d "$dev" --list-formats-ext 2>/dev/null \
      | awk '/\[[0-9]+\]:/ {m = /MJPG/} /Size:/ {s = m && /1280x720/} s && /60\.000 fps/ {f = 1} END {exit !f}' \
      && ok "1280x720 at 60 fps in MJPEG" || warn "could not confirm 1280x720 @ 60 fps MJPEG (v4l2-ctl -d $dev --list-formats-ext)"
  else
    bad "no C922 plugged in (v4l2-ctl --list-devices)"
  fi
else
  bad "v4l2-ctl missing: sudo apt install v4l-utils"
fi

echo "Repo"
[[ -d maze_CAD/scripts ]] && ok "maze_CAD/scripts here (maze_cad_map needs it)" || bad "run from the repo root: maze_CAD/scripts not found in $(pwd)"
free_gb="$(df -BG --output=avail . | tail -1 | tr -dc 0-9)"
(( free_gb >= 10 )) && ok "${free_gb} GB free for bags" || warn "only ${free_gb} GB free: a minute of compressed camera is ~0.5 GB"

echo
(( fail )) && echo "Fix the FAIL lines first." || echo "Ready: ros2 launch maze_perception real.launch.py video_device:=<dev>"
exit $fail
