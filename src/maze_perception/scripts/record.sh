#!/usr/bin/env bash
# Record what perception sees and says, for replay_bag at home (P5/P6, docs/perception_lab.md).
#   ros2 run maze_perception record.sh [name]      # Ctrl-C to stop; bag in ./bags/<name>_<time>
# Compressed images when compressed_image_transport is installed (~8 MB/s), else raw (~160 MB/s).
# Arm topics are added when they exist, so the same bag serves the grip check and system ID (#32).
set -euo pipefail

name="${1:-p6}"
out="bags/${name}_$(date +%Y%m%d_%H%M%S)"
mkdir -p bags

topics=(/camera/camera_info /ball/state /perception/board_pose /maze/map /tf_static)
present="$(ros2 topic list)"
if grep -qx /camera/image_raw/compressed <<<"$present"; then
  topics+=(/camera/image_raw/compressed)
else
  echo "no /camera/image_raw/compressed: recording raw images (big)" >&2
  topics+=(/camera/image_raw)
fi
for t in /joint_states /board/tilt_cmd /board/tilt_state /maze/path; do
  grep -qx "$t" <<<"$present" && topics+=("$t")
done

echo "recording to $out: ${topics[*]}"
exec ros2 bag record -o "$out" "${topics[@]}"
