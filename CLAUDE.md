# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a ROS2 Humble workspace for the RoArm-M2-S robotic arm mounted on a 2-axis XY gantry, integrating MoveIt2 for motion planning and calibration. The gantry extends the arm's workspace by positioning the arm base, then the arm reaches the final target. Designed for Ubuntu 22.04 with ROS2 Humble, running on a Jetson (aarch64/tegra).

## Build and Compilation

### Initial Setup (First Time Only)
```bash
./build_first.sh
```
Runs `colcon build` and adds workspace source to `~/.bashrc`.

**Note:** The build scripts reference `~/roarm_calibration` as the workspace path. If your checkout is elsewhere, update the `cd` path in `build_first.sh` and `build_common.sh`.

### Regular Development Build
```bash
./build_common.sh
```
Runs `colcon build` then `source install/setup.bash`.

### Build Single Package
```bash
colcon build --packages-select <package_name>
source install/setup.bash
```

## Hardware Connection

### RoArm-M2-S (USB Serial)
Connects via USB serial (typically `/dev/ttyUSB0`). Before running the driver:
```bash
sudo chmod 666 /dev/ttyUSB0
```

If using a different serial port, modify `src/roarm_main/roarm_driver/roarm_driver/roarm_driver.py` line 15.

### Arduino Gantry
The gantry Arduino typically appears on `/dev/ttyUSB1`. For network-bridged serial (e.g., Arduino on another machine):
```bash
sudo socat PTY,link=/dev/ttyArduino,rawer TCP:192.168.64.1:54321 &
sudo chmod 666 /dev/ttyArduino
```

## Common Launch Commands

### Core Control

```bash
# Driver node (required for physical robot)
ros2 run roarm_driver roarm_driver

# Joint control with Rviz2
ros2 launch roarm_description display.launch.py

# Interactive MoveIt2 control (drag end-effector)
ros2 launch roarm_moveit interact.launch.py

# Command-based control (services/actions)
ros2 launch roarm_moveit_cmd command_control.launch.py

# Calibrated command control (applies offsets)
ros2 launch roarm_moveit_cmd command_control_calibrated.launch.py
```

### Gantry + Arm Coordinated Control

Requires the ROS2 stack to be running (driver + command_control.launch.py).

```bash
# Interactive mode
python3 scripts/gantry_arm_controller.py --gantry-port /dev/ttyUSB1

# Single move then exit
python3 scripts/gantry_arm_controller.py --gantry-port /dev/ttyUSB1 --goto 200 200
```

### Camera-Guided Pick & Place

Requires RealSense D435I, ROS2 stack running, and gantry Arduino connected.

```bash
# 1. Calibrate camera-to-robot mapping (click 4 points, enter world coords)
python3 scripts/camera_calibration.py

# 2. Verify calibration (live overlay of world coords)
python3 scripts/camera_calibration.py --verify

# 3. Run pick controller (click mode — click on camera feed to pick)
python3 scripts/camera_pick_controller.py

# 4. Run with YOLO detection (auto-detect objects)
python3 scripts/camera_pick_controller.py --yolo yolov8n.pt
```

Pick controller settings (ports, heights, calibration path) are in `config/pick_controller.yaml`.

### Service Commands

Require `command_control.launch.py` or `command_control_calibrated.launch.py` to be running.

```bash
# Get current pose (run node in terminal 1, call service in terminal 2)
ros2 run roarm_moveit_cmd getposecmd
ros2 service call /get_pose_cmd roarm_moveit/srv/GetPoseCmd

# Move to position (metres)
ros2 run roarm_moveit_cmd movepointcmd
ros2 service call /move_point_cmd roarm_moveit/srv/MovePointCmd "{x: 0.2, y: 0, z: 0}"

# Gripper control (radians)
ros2 run roarm_moveit_cmd setgrippercmd
ros2 topic pub /gripper_cmd std_msgs/msg/Float32 "{data: 0.0}" -1

# Draw circle
ros2 run roarm_moveit_cmd movecirclecmd
ros2 service call /move_circle_cmd roarm_moveit/srv/MoveCircleCmd "{x: 0.2, y: 0, z: 0, radius: 0.1}"
```

## Calibration

### Single-Target (Quick Accuracy Test)
```bash
ros2 run roarm_moveit_cmd calibrate_roarm.py \
  --target 0.2 -0.1 -0.1 --iterations 20 --home 0.2 0.0 0.1 \
  --settling-time 1.0 --output-dir .
```

### Multi-Target (Comprehensive Workspace Mapping)
```bash
ros2 run roarm_moveit_cmd calibrate_roarm.py \
  --targets-config src/roarm_main/roarm_moveit_cmd/config/calibration_targets.yaml \
  --loops 2 --output-dir .
```

Movement pattern per loop: `home → target1 → home → target2 → home → ...`

Default config has 9 targets in a 3×3 grid (3 heights × 3 radial distances × 3 lateral positions). Custom targets go in `config/calibration_targets.yaml`.

Calibration offsets stored in `config/calibration_offsets.yaml`, applied by `command_control_calibrated.launch.py`.

## Architecture

### Package Structure

- **`scripts/`** — Top-level Python scripts
  - `gantry_arm_controller.py` — Coordinated gantry+arm controller. Arm via ROS2 `/move_point_cmd` (MoveIt2). Gantry Arduino receives `"dx_mm, dy_mm\n"` over serial.
  - `camera_calibration.py` — 4-point homography calibration (pixel→world mm). Saves to `config/camera_calibration.yaml`.
  - `camera_pick_controller.py` — Camera-guided pick & place (click mode + YOLO mode). Reads `config/pick_controller.yaml`.
  - `test_realsense_depth.py` — RealSense depth diagnostic.
- **`config/`** — Configuration files
  - `camera_calibration.yaml` — Homography matrix + calibration points (generated by `camera_calibration.py`)
  - `pick_controller.yaml` — Pick controller settings (gantry port, pick heights, camera params, YOLO model)
- **`roarm_driver/`** — Hardware driver, serial communication with ESP32
- **`roarm_description/`** — URDF robot model
- **`roarm_moveit/`** — MoveIt2 configuration (kinematics, planning, controllers)
- **`roarm_moveit_ikfast_plugins/`** — IKFast inverse kinematics solver
- **`roarm_moveit_cmd/`** — C++ service nodes + calibration scripts
  - `src/movepointcmd.cpp` — Move to XYZ position service
  - `src/getposecmd_moveit2.cpp` — Get current pose service
  - `src/setgrippercmd.cpp` — Gripper control action
  - `src/movecirclecmd.cpp` — Circular motion planner
  - `src/keyboardcontrol.cpp` — Keyboard input handler
  - `scripts/calibrate_roarm.py` — Calibration automation (single/multi-target)
- **`moveit_servo/`** — Real-time servo control for keyboard/gamepad

### Coordinate Frames

The gantry XY frame and RoArm frame are rotated 90° clockwise relative to each other:
- Gantry `(local_x, local_y)` → Arm `(local_y, -local_x)`
- See `gantry_to_arm_coords()` in `scripts/gantry_arm_controller.py`

### RoArm Serial Protocol (Direct Control)

The RoArm uses Waveshare JSON protocol over serial, coordinates in mm:
- `T:100` — Move to init position
- `T:104` — Cartesian move, blocking: `{"T":104,"x":235,"y":0,"z":234,"t":3.14,"spd":0.25}`
- `T:1041` — Cartesian move, non-blocking: `{"T":1041,"x":235,"y":0,"z":234,"t":3.14}`
- `T:105` — Get feedback (returns T:1051 with position/angles/torque)

### Arm Geometry

From `ik.h`: L2=238.7mm, L3=280.2mm, theoretical max reach ~519mm. Default comfortable reach radius: 200mm. Default Z height: -15mm.

### Control Flow

1. **Hardware Layer**: `roarm_driver` ↔ ESP32 via serial
2. **ROS2 Control**: `ros2_control_node` manages joint controllers
3. **MoveIt2 Layer**: Motion planning, IK solving, collision checking
4. **Command Layer**: `roarm_moveit_cmd` services expose high-level XYZ commands
5. **Gantry Layer**: `scripts/gantry_arm_controller.py` coordinates gantry positioning + arm reach
6. **Vision Layer**: `scripts/camera_pick_controller.py` — RealSense camera → homography → world coords → gantry+arm commands

### Key Topics and Services

- `/joint_states` — Current joint positions
- `/move_point_cmd` — Move end-effector to XYZ (metres, via MoveIt2)
- `/get_pose_cmd` — Get current end-effector pose
- `/gripper_cmd` — Gripper control topic (radians)
- `/led_ctrl` — Gripper LED control (0-255)

## Python Dependencies

```bash
python3 -m pip install -r requirements.txt
```

Key packages: `pyserial` (hardware communication), `opencv-python`, `numpy`, `pyrealsense2`.

## Known Issues / Workarounds

### RealSense on Jetson (aarch64)
On this Jetson build, `rs.align()`, color intrinsics, and extrinsics all return NaN. The camera scripts work around this by:
- Skipping `rs.align()` entirely
- Using proportional scaling (`color_pixel * depth_res / color_res`) to map color pixels to depth pixels
- Only relying on depth stream intrinsics (which work correctly)

See `RealSenseCamera` class in `scripts/camera_calibration.py`.
