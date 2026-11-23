# Perception on the real camera: lab runbook (P6)

Step-by-step for #45 (P6, part of #13) and stage 2 of the real-robot plan (#32): the C922 over the
maze, `ball_tracker` on its images, and the numbers #45 asks for. Everything here only reads the camera
and runs perception; nothing moves the arm. Stage 2 comes after stage 1 (arm and board) has passed.

The kit (all in `maze_perception`):

| What | For |
|---|---|
| `ros2 run maze_perception preflight.sh` | is the lab machine ready (ROS, usb_cam, OpenCV, the camera, the repo) |
| `ros2 launch maze_perception camera.launch.py` | the C922 alone, on `/camera/image_raw` and `/camera/camera_info` like the sim camera |
| `ros2 run maze_perception c922_setup.sh` | lock focus, exposure and white balance |
| `ros2 launch maze_perception real.launch.py` | camera + CAD maze map + `ball_tracker`, wall clock, the 25 mm marble |
| `ros2 run maze_perception p6_check stream` | frame rate, drop-outs, board found, ball tracked, latency |
| `ros2 run maze_perception p6_check spots` | ball in five maze corners against the CAD |
| `ros2 run maze_perception record.sh` | a bag for `replay_bag` at home |

## Before the lab

- [ ] The lab machine has Ubuntu 22.04 and ROS 2 Humble, plus `ros-humble-usb-cam`, `v4l-utils`,
  `ros-humble-camera-calibration` and `ros-humble-image-transport-plugins` (ask the lab staff if
  there is no sudo). OpenCV ≥ 4.7 for the ArUco detector:
  `pip install --user "opencv-contrib-python-headless==4.10.0.84" "numpy<2"`.
- [ ] Clone the repo on the lab machine and build: `colcon build --packages-up-to maze_perception`,
  then `source install/setup.bash`. Run everything below **from the repo root** (the CAD map reads
  `maze_CAD/scripts`).
- [ ] The board with its four ArUco stickers (#4), the marble (#19), and a printed **checkerboard**
  for calibration (e.g. 9 × 6 inner corners of 25 mm, flat on cardboard; measure a square after
  printing).
- [ ] A way to hold the C922 about 1 m above the board, looking straight down (the sim camera sits
  1.0 m above the board center). All four markers must stay in view at ±5° of tilt.

## 1. Preflight (2 min)

Plug the C922 into a USB port of its own (720p at 60 fps needs a good share of USB 2).

```bash
ros2 run maze_perception preflight.sh
```

Fix every `FAIL`. It prints the camera's device, e.g. `/dev/video2`; use it below as `<dev>`.

## 2. Camera, focus and exposure (10 min)

```bash
ros2 launch maze_perception camera.launch.py video_device:=<dev>
ros2 run rqt_image_view rqt_image_view /camera/image_raw        # second terminal
ros2 run maze_perception c922_setup.sh <dev>                    # third terminal
```

`c922_setup.sh` turns autofocus, auto exposure and auto white balance off. Why it matters (P5): a
hunting autofocus blurs the markers until the board is lost, auto exposure drops 60 fps to 30 in dim
light and lets the board clip to white, and a drifting white balance changes the floor color.

Then tune by eye, rerunning with other values:
- `FOCUS=<n>` (0 far … 250 near; the board is ~1 m away, try 0–30): the marker edges sharp.
- `EXPOSURE=<n>` (100 µs units, **≤ 166** for 60 fps; default 100): the floor bright but **never
  white**; the ball must not clip either. Turn the room light up rather than the exposure.

## 3. Calibrate (15 min, once per camera and focus)

With focus and resolution as they will be used (calibrate after step 2):

```bash
ros2 run camera_calibration cameracalibrator --size 9x6 --square 0.025 --no-service-check \
    --ros-args -r image:=/camera/image_raw
```

Move the checkerboard over the whole image, tilted and at several distances, until the X, Y, Size
and Skew bars are green; **CALIBRATE**, then **SAVE**. The result is in
`/tmp/calibrationdata.tar.gz`:

```bash
tar -xzf /tmp/calibrationdata.tar.gz -C /tmp ost.yaml && cp /tmp/ost.yaml ~/c922_calibration.yaml
```

A good calibration reprojects within about 0.3 px (printed in the terminal). Without one, the
launch files fall back to `config/c922_nominal.yaml` (the sim's pinhole), which puts the ball a few
mm off near the image edges.

## 4. Perception (5 min)

Stop `camera.launch.py` (Ctrl-C), then everything in one go:

```bash
ros2 launch maze_perception real.launch.py video_device:=<dev> \
    camera_info_url:=file://$HOME/c922_calibration.yaml
ros2 run maze_perception c922_setup.sh <dev>                    # again: the camera restarted
ros2 run rqt_image_view rqt_image_view /perception/board_image   # the rectified board
```

`ball_radius` defaults to 0.0125 (the 25 mm marble); pass the measured radius once #19 has it. In the
log, look for `maze map 'cad:L1'` (needed before the ball is looked for) and, every 10 s, the share
of frames with the board and the ball, and the capture-to-publish latency.

## 5. The #45 checks (20 min)

Keep a bag of all of it (third terminal): `ros2 run maze_perception record.sh p6`.

1. **Stream, ball rolled by hand** for two minutes, under the room's normal light:
   ```bash
   ros2 run maze_perception p6_check stream --seconds 120
   ```
   Pass: ≥ 55 fps, no gap > 0.2 s, latency ≤ 50 ms, `tracking_ok` ≥ 98%.
2. **No ball** on the board for 30 s (`p6_check stream --seconds 30`): `tracking_ok` must be 0%.
   Anything else is a false detection: reflections, the gripper, a sticker. Note where
   (`ros2 topic echo /ball/state`) and keep the bag.
3. **Five corners**, board level:
   ```bash
   ros2 run maze_perception p6_check spots
   ```
   It writes `p6_spots.png`, a map of the five spots (board +y up, handle on the left), then asks for
   the ball pushed into each: the maze's four corners and the wall corner nearest the middle. Two
   walls fix the ball's center to a fraction of a mm there. Pass: within 3 mm. Expect about 1.5 mm:
   against a wall the finder leaves the wall's faces out and with them the ball's outermost
   millimeter; on open floor it is within a fraction of a mm.

Write the numbers into #45, with the bag's name.

## 6. At home

```bash
ros2 run maze_perception replay_bag bags/p6_<time> --csv p6.csv
```

reruns `ball_tracker` on the recording after any change (thresholds, `accel_noise`, the ball's
size), with no camera or robot.

## When something is off

| Symptom | Likely cause |
|---|---|
| ~30 fps | auto exposure still on, or the exposure over 166: rerun `c922_setup.sh`; a USB hub shared with other devices |
| board not found | markers blurred (focus), clipped white (exposure), or out of view at a tilt |
| board found, never the ball | no `/maze/map` yet (log), wrong `ball_radius`, the ball clipped white |
| ball found where there is none | a reflection or the gripper: keep the bag, note the position, raise `ball.min_score` only if it is far below the ball's own score |
| ball jumps between frames | the exposure too long (smear) or too short (noise): see the scores in `replay_bag`'s CSV |
| `numpy` / `cv2` import errors | numpy 2 from pip: `pip install --user "numpy<2"` |
