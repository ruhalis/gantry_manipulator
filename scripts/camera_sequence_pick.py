#!/usr/bin/env python3
"""
Sequence Pick-and-Place with Camera Recording.

Runs a fixed 4-pose pick-and-place routine:
    home -> pick(430,338) -> drop(640,270)
         -> pick(452,632) -> drop(680,534) -> home
At each pick: approach at travel Z, descend to pick Z, close gripper,
lift to travel Z. At each drop: approach, descend, open gripper, lift.

The OpenCV window shows the live RealSense feed and records it to an
MP4 file so YOLO (best.pt) can be run offline in post-processing.
Gantry + arm motion runs in a worker thread so the camera window never
freezes during the routine.

Usage:
  python3 scripts/camera_sequence_pick.py
  python3 scripts/camera_sequence_pick.py --autostart
  python3 scripts/camera_sequence_pick.py --no-record
"""

import argparse
import math
import os
import sys
import threading
import time
from datetime import datetime

import cv2
import serial
import yaml

try:
    import pyrealsense2 as rs  # noqa: F401
except ImportError:
    print("ERROR: pyrealsense2 not installed")
    sys.exit(1)

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Float32
from roarm_moveit.srv import MovePointCmd

from camera_calibration import RealSenseCamera, load_calibration


GRIPPER_OPEN = 3.14
GRIPPER_CLOSED = 0.25
GRIPPER_SETTLE_S = 1.5

SEQUENCE = [
    ("pick", 430.0, 338.0),
    ("drop", 640.0, 270.0),
    ("pick", 452.0, 632.0),
    ("drop", 680.0, 534.0),
]


class GantryController:
    """Talks to the Arduino running motor_control.cpp over serial."""

    def __init__(self, port: str, baud: int = 115200, timeout: float = 2.0):
        self.ser = serial.Serial(port, baud, timeout=timeout)
        time.sleep(2)
        while self.ser.in_waiting:
            self.ser.readline()
        self.pos_x = 0.0
        self.pos_y = 0.0
        self._lock = threading.Lock()

    def move_relative(self, dx_mm: float, dy_mm: float, wait: bool = True):
        with self._lock:
            cmd = f"{dx_mm:.2f}, {dy_mm:.2f}\n"
            self.ser.write(cmd.encode('utf-8'))
            self.pos_x += dx_mm
            self.pos_y += dy_mm
        if wait:
            self._wait_for_move(dx_mm, dy_mm)
        return self.pos_x, self.pos_y

    def move_absolute(self, target_x_mm: float, target_y_mm: float, wait: bool = True):
        dx = target_x_mm - self.pos_x
        dy = target_y_mm - self.pos_y
        return self.move_relative(dx, dy, wait=wait)

    def _wait_for_move(self, dx_mm: float, dy_mm: float):
        steps_per_mm = 3200.0 / 125.0
        max_speed_mm_s = 4000.0 / steps_per_mm
        accel_mm_s2 = 2000.0 / steps_per_mm
        dist = max(abs(dx_mm), abs(dy_mm))
        if dist < 0.01:
            return
        t_accel = max_speed_mm_s / accel_mm_s2
        d_accel = 0.5 * accel_mm_s2 * t_accel ** 2
        if dist < 2 * d_accel:
            t_total = 2.0 * math.sqrt(dist / accel_mm_s2)
        else:
            t_cruise = (dist - 2 * d_accel) / max_speed_mm_s
            t_total = 2 * t_accel + t_cruise
        time.sleep(t_total + 0.3)

    def get_position(self):
        return self.pos_x, self.pos_y

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()


def compute_gantry_target(target_x_mm: float, target_y_mm: float,
                          cur_gx: float, cur_gy: float,
                          home_x_mm: float, home_y_mm: float,
                          reach_radius_mm: float) -> tuple:
    cur_wx = cur_gx + home_x_mm
    cur_wy = cur_gy + home_y_mm
    dx = target_x_mm - cur_wx
    dy = target_y_mm - cur_wy
    dist = math.hypot(dx, dy)
    if dist <= reach_radius_mm or dist < 1e-6:
        return cur_gx, cur_gy, dx, dy
    scale = (dist - reach_radius_mm) / dist
    new_wx = cur_wx + dx * scale
    new_wy = cur_wy + dy * scale
    return new_wx - home_x_mm, new_wy - home_y_mm, target_x_mm - new_wx, target_y_mm - new_wy


class ArmMover:
    """Controls RoArm via ROS2 /move_point_cmd service + /gripper_cmd topic."""

    def __init__(self, node: Node):
        self._node = node
        self._client = node.create_client(MovePointCmd, '/move_point_cmd')
        self._gripper_pub = node.create_publisher(Float32, '/gripper_cmd', 10)

    def wait_for_service(self, timeout_sec=10.0):
        print("Waiting for /move_point_cmd service...")
        if not self._client.wait_for_service(timeout_sec=timeout_sec):
            print("ERROR: /move_point_cmd not available!")
            return False
        print("Service ready.")
        return True

    def wait_for_gripper(self, timeout_sec: float = 5.0) -> bool:
        print("Waiting for /gripper_cmd subscriber (setgrippercmd node)...")
        t0 = time.time()
        while time.time() - t0 < timeout_sec:
            if self._gripper_pub.get_subscription_count() > 0:
                print("  Gripper bridge connected.")
                return True
            time.sleep(0.1)
        print("  WARNING: no subscriber on /gripper_cmd — gripper will NOT move.")
        print("  Start it with:  ros2 run roarm_moveit_cmd setgrippercmd")
        return False

    def move_to(self, x_mm, y_mm, z_mm):
        req = MovePointCmd.Request()
        req.x = x_mm / 1000.0
        req.y = y_mm / 1000.0
        req.z = z_mm / 1000.0
        print(f"  Arm -> ({x_mm:.1f}, {y_mm:.1f}, {z_mm:.1f}) mm")
        future = self._client.call_async(req)
        t0 = time.time()
        while not future.done():
            if time.time() - t0 > 30.0:
                print("  Service call timed out")
                return False
            time.sleep(0.01)
        result = future.result()
        ok = bool(result and result.success)
        print(f"  {'OK' if ok else 'FAILED'}")
        return ok

    def gripper(self, value: float):
        if self._gripper_pub.get_subscription_count() == 0:
            print(f"  Gripper cmd {value:.2f} DROPPED: no subscriber on /gripper_cmd "
                  "(run: ros2 run roarm_moveit_cmd setgrippercmd)")
            return
        msg = Float32()
        msg.data = float(value)
        self._gripper_pub.publish(msg)
        print(f"  Gripper cmd -> {value:.2f}")


def goto_world_position(gantry, arm, wx_t, wy_t, home_x, home_y, arm_z, reach_radius_mm):
    cur_gx, cur_gy = gantry.get_position() if gantry else (0.0, 0.0)
    gantry_x, gantry_y, rx, ry = compute_gantry_target(
        wx_t, wy_t, cur_gx, cur_gy, home_x, home_y, reach_radius_mm)

    print(f"  Target world: ({wx_t:.1f}, {wy_t:.1f}) mm")
    print(f"  Gantry target: ({gantry_x:.1f}, {gantry_y:.1f})  residual ({rx:.1f}, {ry:.1f})")

    if gantry:
        dgx = gantry_x - cur_gx
        dgy = gantry_y - cur_gy
        if abs(dgx) > 0.1 or abs(dgy) > 0.1:
            gantry.move_relative(dgx, dgy, wait=True)

    arm_x = home_x - ry
    arm_y = home_y + rx
    arm.move_to(arm_x, arm_y, arm_z)
    return arm_x, arm_y


def run_sequence(gantry, arm, home_x, home_y, pick_z, travel_z, reach_radius, busy):
    try:
        busy[0] = True
        print("\n=== SEQUENCE START ===")
        arm.gripper(GRIPPER_OPEN)
        time.sleep(GRIPPER_SETTLE_S)

        for i in range(0, len(SEQUENCE), 2):
            _, px, py = SEQUENCE[i]
            _, dx, dy = SEQUENCE[i + 1]
            print(f"\n-- Pair {i//2 + 1}: pick ({px},{py}) -> drop ({dx},{dy}) --")

            print("[1] open gripper before gantry move")
            arm.gripper(GRIPPER_OPEN)
            time.sleep(GRIPPER_SETTLE_S)
            print("[2] approach pick @ travel z")
            ax, ay = goto_world_position(gantry, arm, px, py,
                                         home_x, home_y, travel_z, reach_radius)
            print("[3] descend to pick z")
            arm.move_to(ax, ay, pick_z)
            print("[4] close gripper on cube")
            arm.gripper(GRIPPER_CLOSED)
            time.sleep(GRIPPER_SETTLE_S)
            print("[5] lift to travel z")
            arm.move_to(ax, ay, travel_z)

            print("[6] approach drop @ travel z (holding cube)")
            ax, ay = goto_world_position(gantry, arm, dx, dy,
                                         home_x, home_y, travel_z, reach_radius)
            print("[7] descend to pick z")
            arm.move_to(ax, ay, pick_z)
            print("[8] open gripper to release")
            arm.gripper(GRIPPER_OPEN)
            time.sleep(GRIPPER_SETTLE_S)
            print("[9] lift to travel z")
            arm.move_to(ax, ay, travel_z)

        print("\n-- Returning home --")
        if gantry:
            gx, gy = gantry.get_position()
            if abs(gx) > 0.1 or abs(gy) > 0.1:
                print(f"  Gantry ({gx:.1f}, {gy:.1f}) -> (0, 0)")
                gantry.move_absolute(0.0, 0.0, wait=True)
        arm.move_to(home_x, home_y, travel_z)
        print("=== SEQUENCE DONE ===\n")
    except Exception as e:
        print(f"Sequence error: {e}")
    finally:
        busy[0] = False


def main():
    parser = argparse.ArgumentParser(description="4-pose pick/place with camera recording")
    parser.add_argument("--config", default="config/pick_controller.yaml")
    parser.add_argument("--autostart", action="store_true",
                        help="Run the sequence immediately on launch")
    parser.add_argument("--no-record", action="store_true",
                        help="Disable MP4 recording")
    parser.add_argument("--record-dir", default="recordings",
                        help="Directory to write recordings into (default: recordings)")
    parser.add_argument("--record-fps", type=float, default=15.0,
                        help="FPS to stamp into the recorded MP4 (default: 15)")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    print(f"Config loaded from {args.config}")

    arm_cfg = cfg['arm']
    cam_cfg = cfg['camera']
    gantry_cfg = cfg.get('gantry', {})

    home_x = arm_cfg.get('home_x_mm', 200.0)
    home_y = arm_cfg.get('home_y_mm', 0.0)
    pick_z = arm_cfg.get('default_z_mm', -15.0)
    travel_z = 0.0
    reach_radius = arm_cfg.get('reach_radius_mm', 200.0)
    calib_file = cam_cfg.get('calibration_file', 'config/camera_calibration.yaml')

    gantry_port = gantry_cfg.get('port', '/dev/ttyUSB1')
    gantry_baud = gantry_cfg.get('baud', 115200)

    _, _ = load_calibration(calib_file)
    print(f"Calibration loaded from {calib_file}")

    cam_w = cam_cfg.get('color_width', 1280)
    cam_h = cam_cfg.get('color_height', 720)
    cam = RealSenseCamera(cam_w, cam_h, cam_cfg.get('fps', 30))
    cam.start()

    writer = None
    record_path = None
    if not args.no_record:
        os.makedirs(args.record_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        record_path = os.path.join(args.record_dir, f"sequence_{ts}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(record_path, fourcc, args.record_fps, (cam_w, cam_h))
        if not writer.isOpened():
            print(f"WARNING: could not open video writer for {record_path}")
            writer = None
        else:
            print(f"Recording to {record_path} @ {args.record_fps} fps")

    print(f"Connecting to gantry on {gantry_port} ...")
    try:
        gantry = GantryController(gantry_port, gantry_baud)
        print("  Gantry connected.")
    except serial.SerialException as e:
        print(f"  WARNING: gantry not available: {e}")
        gantry = None

    rclpy.init()
    node = Node('camera_sequence_pick')
    arm = ArmMover(node)

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    arm.wait_for_gripper()

    if not arm.wait_for_service():
        cam.stop()
        if gantry:
            gantry.close()
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    cv2.namedWindow("Sequence Pick", cv2.WINDOW_NORMAL)

    busy = [False]
    worker = [None]

    def launch_sequence():
        if busy[0]:
            print("Sequence already running")
            return
        worker[0] = threading.Thread(
            target=run_sequence,
            args=(gantry, arm, home_x, home_y, pick_z, travel_z, reach_radius, busy),
            daemon=True,
        )
        worker[0].start()

    print()
    print("=" * 60)
    print("CAMERA SEQUENCE PICK  (4 predefined poses)")
    print("=" * 60)
    print(f"  Home offset: arm ({home_x:.0f}, {home_y:.0f}) mm = world (0, 0)")
    print(f"  Pick Z: {pick_z:.0f} mm   Travel Z: {travel_z:.0f} mm")
    print(f"  Reach radius: {reach_radius:.0f} mm")
    print(f"  Sequence: {SEQUENCE}")
    print()
    print("  's': start sequence   'q': quit")
    print()

    if args.autostart:
        launch_sequence()

    try:
        while True:
            color_image, _, _ = cam.get_frames()
            if color_image is None:
                continue

            if writer is not None:
                writer.write(color_image)

            display = color_image.copy()
            status = "RUNNING" if busy[0] else "IDLE"
            gx, gy = gantry.get_position() if gantry else (0.0, 0.0)
            rec = "REC" if writer is not None else "no-rec"
            info = f"{status}  Gantry:({gx:.0f},{gy:.0f})  {rec}  s=start  q=quit"
            cv2.putText(display, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if writer is not None:
                cv2.circle(display, (cam_w - 30, 30), 10, (0, 0, 255), -1)

            cv2.imshow("Sequence Pick", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                launch_sequence()

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        print("Cleaning up...")
        if writer is not None:
            writer.release()
            print(f"Saved recording: {record_path}")
        cv2.destroyAllWindows()
        cam.stop()
        if gantry:
            gantry.close()
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
        print("Done.")


if __name__ == "__main__":
    main()
