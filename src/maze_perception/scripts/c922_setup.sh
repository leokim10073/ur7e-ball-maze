#!/usr/bin/env bash
# Lock the C922's focus, exposure and white balance for perception (P6, docs/perception_lab.md).
#
#   ros2 run maze_perception c922_setup.sh [/dev/videoN]        # default: the first C922 found
#   FOCUS=10 EXPOSURE=80 ros2 run maze_perception c922_setup.sh  # other values
#
# Why: autofocus hunting blurs the ArUco markers past reading (P5: from 2.5 px of blur the board is
# lost), auto exposure drops 60 fps to 30 in dim light and lets the board clip to white, and auto
# white balance changes the floor color under the ball finder. Run it after the camera has started
# (usb_cam sets controls of its own at start); the values stay until the camera is unplugged.
#
# FOCUS     focus_absolute, 0 (far) .. 250 (near); the board is about 1 m away. Default 0.
# EXPOSURE  exposure time in 100 us units; at most 166 for 60 fps. Default 100 (10 ms: a ball at
#           0.3 m/s smears 3 mm). Raise it if the image is too dark, keep the floor below white.
# WB        white balance temperature, K. Default 4000.
# Kernel versions name the controls differently (focus_auto / focus_automatic_continuous, ...);
# each setting tries every name it is known by.
set -euo pipefail

FOCUS="${FOCUS:-0}"
EXPOSURE="${EXPOSURE:-100}"
WB="${WB:-4000}"

command -v v4l2-ctl >/dev/null || { echo "v4l2-ctl missing: sudo apt install v4l-utils" >&2; exit 1; }

DEV="${1:-}"
if [[ -z "$DEV" ]]; then
  DEV="$(v4l2-ctl --list-devices 2>/dev/null | awk '/C922/ {getline; print $1; exit}')"
  [[ -n "$DEV" ]] || { echo "no C922 found (v4l2-ctl --list-devices); pass the device" >&2; exit 1; }
fi
echo "C922 at $DEV"
CTRLS="$(v4l2-ctl -d "$DEV" --list-ctrls)"

set_ctrl() {  # set_ctrl VALUE NAME [NAME ...]: the first name this camera has
  local value="$1"; shift
  for name in "$@"; do
    if grep -qE "^\s*${name}\b" <<<"$CTRLS"; then
      v4l2-ctl -d "$DEV" -c "${name}=${value}" && echo "  ${name}=${value}"
      return 0
    fi
  done
  echo "  (none of $* on this camera)" >&2
}

# automatic modes off first, or the manual values are ignored
set_ctrl 0 focus_automatic_continuous focus_auto
set_ctrl "$FOCUS" focus_absolute
set_ctrl 1 auto_exposure exposure_auto                    # 1 = manual
set_ctrl 0 exposure_dynamic_framerate exposure_auto_priority  # keep 60 fps
set_ctrl "$EXPOSURE" exposure_time_absolute exposure_absolute
set_ctrl 0 white_balance_automatic white_balance_temperature_auto
set_ctrl "$WB" white_balance_temperature
set_ctrl 2 power_line_frequency                           # 60 Hz mains (US): no flicker bands

echo
v4l2-ctl -d "$DEV" --list-ctrls | grep -E "focus|exposure|white_balance|power_line"
