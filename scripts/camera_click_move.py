#!/usr/bin/env python3
"""
Camera Click-to-Move Controller for RoArm + Gantry.

Shows live camera feed with world coordinate overlay (like --verify mode).
Click on a point to move the gantry+arm there. The gantry positions the arm
base within reach_radius_mm of the target, then the arm reaches the target.

Uses the existing camera calibration (homography) to convert pixel → world mm.

Prerequisites:
  1. Run camera_calibration.py first to generate camera_calibration.yaml
  2. ROS2 stack must be running (roarm_driver + command_control.launch.py)
  3. Gantry Arduino connected on the port in pick_controller.yaml

Usage:
  python3 scripts/camera_click_move.py
  python3 scripts/camera_click_move.py --config config/pick_controller.yaml
"""

import argparse
import math
import sys
import time
import serial
import cv2
import yaml

try:
    import pyrealsense2 as rs  # noqa: F401
except ImportError:
    print("ERROR: pyrealsense2 not installed")
    sys.exit(1)

import rclpy
from rclpy.node import Node
from roarm_moveit.srv import MovePointCmd

from camera_calibration import RealSenseCamera, load_calibration, pixel_to_world


# ---------------------------------------------------------------------------
# Gantry controller (mirrors GantryController in gantry_arm_controller.py)
# ---------------------------------------------------------------------------

class GantryController:
    """Talks to the Arduino running motor_control.cpp over serial."""

    def __init__(self, port: str, baud: int = 115200, timeout: float = 2.0):
        self.ser = serial.Serial(port, baud, timeout=timeout)
        time.sleep(2)
        self._drain_startup()
        self.pos_x = 0.0
        self.pos_y = 0.0

    def _drain_startup(self):
        while self.ser.in_waiting:
            self.ser.readline()

    def move_relative(self, dx_mm: float, dy_mm: float, wait: bool = True):
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
    """Split a world target between gantry and arm.

    The gantry odometry (cur_gx, cur_gy) is offset from the world frame by
    (home_x_mm, home_y_mm): when gantry odom is (0,0) and the arm is at its
    home pose, the end-effector sits at world (home_x_mm, home_y_mm).

    If the target is within reach_radius_mm of the current arm-home point,
    the gantry does not move and the arm covers the whole residual.
    Otherwise the gantry slides along the line to the target and stops
    exactly reach_radius_mm short, leaving the arm at full extension.

    Returns (gantry_x, gantry_y, residual_x, residual_y) where the gantry
    values are in gantry-odom frame and the residual is in world mm.
    """
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
    gx_new = new_wx - home_x_mm
    gy_new = new_wy - home_y_mm
    rx = target_x_mm - new_wx
    ry = target_y_mm - new_wy
    return gx_new, gy_new, rx, ry


# ---------------------------------------------------------------------------
# Arm controller
# ---------------------------------------------------------------------------

class ArmMover:
    """Controls RoArm via ROS2 /move_point_cmd service."""

    def __init__(self, node: Node):
        self._node = node
        self._client = node.create_client(MovePointCmd, '/move_point_cmd')

    def wait_for_service(self, timeout_sec=10.0):
        print("Waiting for /move_point_cmd service...")
        if not self._client.wait_for_service(timeout_sec=timeout_sec):
            print("ERROR: /move_point_cmd not available!")
            return False
        print("Service ready.")
        return True

    def move_to(self, x_mm, y_mm, z_mm):
        """Move end-effector to (x, y, z) in mm (arm frame)."""
        req = MovePointCmd.Request()
        req.x = x_mm / 1000.0
        req.y = y_mm / 1000.0
        req.z = z_mm / 1000.0
        print(f"  Moving arm to ({x_mm:.1f}, {y_mm:.1f}, {z_mm:.1f}) mm")
        future = self._client.call_async(req)
        rclpy.spin_until_future_complete(self._node, future, timeout_sec=30.0)
        if future.result() is not None:
            ok = future.result().success
            print(f"  {'OK' if ok else 'FAILED'}")
            return ok
        print("  Service call timed out")
        return False


# ---------------------------------------------------------------------------
# Coordinated move
# ---------------------------------------------------------------------------

def goto_world_position(gantry: GantryController, arm: ArmMover,
                        wx_t: float, wy_t: float,
                        home_x: float, home_y: float,
                        arm_z: float, reach_radius_mm: float):
    """Drive gantry + arm to a world target, splitting the move at the reach radius."""
    cur_gx, cur_gy = gantry.get_position()
    gantry_x, gantry_y, rx, ry = compute_gantry_target(
        wx_t, wy_t, cur_gx, cur_gy, home_x, home_y, reach_radius_mm)

    print(f"  Target world: ({wx_t:.1f}, {wy_t:.1f}) mm")
    print(f"  Gantry target: ({gantry_x:.1f}, {gantry_y:.1f}) mm  residual ({rx:.1f}, {ry:.1f}) mm")

    dgx = gantry_x - cur_gx
    dgy = gantry_y - cur_gy
    if abs(dgx) > 0.1 or abs(dgy) > 0.1:
        print(f"  Moving gantry by ({dgx:.1f}, {dgy:.1f}) mm ...")
        gantry.move_relative(dgx, dgy, wait=True)
        print(f"  Gantry now at: ({gantry.pos_x:.1f}, {gantry.pos_y:.1f}) mm")
    else:
        print(f"  Gantry already in position ({cur_gx:.1f}, {cur_gy:.1f}) mm")

    arm_x = home_x - ry
    arm_y = home_y + rx
    print(f"  Arm cmd (arm frame): ({arm_x:.1f}, {arm_y:.1f}, {arm_z:.1f}) mm")
    arm.move_to(arm_x, arm_y, arm_z)


def main():
    parser = argparse.ArgumentParser(description="Click on camera to move gantry + arm")
    parser.add_argument("--config", default="config/pick_controller.yaml",
                        help="Config YAML file (default: config/pick_controller.yaml)")
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    print(f"Config loaded from {args.config}")

    arm_cfg = cfg['arm']
    cam_cfg = cfg['camera']
    gantry_cfg = cfg.get('gantry', {})

    home_x = arm_cfg.get('home_x_mm', 200.0)
    home_y = arm_cfg.get('home_y_mm', 0.0)
    arm_z = arm_cfg.get('default_z_mm', -15.0)
    reach_radius_mm = arm_cfg.get('reach_radius_mm', 200.0)
    calib_file = cam_cfg.get('calibration_file', 'config/camera_calibration.yaml')

    gantry_port = gantry_cfg.get('port', '/dev/ttyUSB1')
    gantry_baud = gantry_cfg.get('baud', 115200)

    # Load calibration
    H, calib_data = load_calibration(calib_file)
    print(f"Calibration loaded from {calib_file}")

    # Start camera
    cam = RealSenseCamera(cam_cfg.get('color_width', 1280),
                          cam_cfg.get('color_height', 720),
                          cam_cfg.get('fps', 30))
    cam.start()

    # Connect gantry
    print(f"Connecting to gantry on {gantry_port} ...")
    try:
        gantry = GantryController(gantry_port, gantry_baud)
        print(f"  Gantry connected.")
    except serial.SerialException as e:
        print(f"  WARNING: Could not connect to gantry: {e}")
        print(f"  Gantry moves will be skipped — arm only.")
        gantry = None

    # Init ROS2
    rclpy.init()
    node = Node('camera_click_move')
    arm = ArmMover(node)
    if not arm.wait_for_service():
        cam.stop()
        if gantry:
            gantry.close()
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    mouse_pos = [0, 0]
    click_target = [None]  # (wx, wy) when clicked

    def on_mouse(event, x, y, flags, param):
        mouse_pos[0] = x
        mouse_pos[1] = y
        if event == cv2.EVENT_LBUTTONDOWN:
            wx, wy = pixel_to_world(H, x, y)
            click_target[0] = (wx, wy)
            print(f"\nClicked: pixel ({x}, {y}) → world ({wx:.1f}, {wy:.1f}) mm")

    cv2.namedWindow("Click to Move", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Click to Move", on_mouse)

    gantry_status = f"port={gantry_port}" if gantry else "NOT CONNECTED"
    print()
    print("=" * 60)
    print("CAMERA CLICK-TO-MOVE  (gantry + arm)")
    print("=" * 60)
    print(f"  Gantry:           {gantry_status}")
    print(f"  Reach radius:     {reach_radius_mm:.0f} mm")
    print(f"  Home offset: arm ({home_x:.0f}, {home_y:.0f}) mm = world (0, 0)")
    print(f"  Arm Z: {arm_z:.0f} mm")
    print()
    print("  Left-click: move gantry+arm to that world position")
    print("  'h': return everything to home (0, 0)")
    print("  '+'/'-': adjust Z height by 10mm")
    print("  'q': quit")
    print()

    try:
        while True:
            color_image, depth_image, depth_frame = cam.get_frames()
            if color_image is None:
                continue

            display = color_image.copy()

            # Cursor world coords
            u, v = mouse_pos
            wx, wy = pixel_to_world(H, u, v)
            depth_mm = cam.get_depth_at_pixel(depth_frame, u, v)

            cv2.drawMarker(display, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

            text = f"Pixel: ({u}, {v})  World: ({wx:.1f}, {wy:.1f}) mm  Depth: {depth_mm:.0f} mm"
            cv2.putText(display, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            gx, gy = gantry.get_position() if gantry else (0.0, 0.0)
            info = (f"Gantry:({gx:.0f},{gy:.0f})  Radius:{reach_radius_mm:.0f}  "
                    f"Z:{arm_z:.0f}mm  |  Click=move  h=home  q=quit")
            cv2.putText(display, info, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

            # Draw calibration reference points
            for i, (px, py) in enumerate(calib_data['pixel_points']):
                cv2.circle(display, (int(px), int(py)), 6, (255, 0, 0), 2)
                wx_cal, wy_cal = calib_data['world_points'][i]
                cv2.putText(display, f"({wx_cal:.0f},{wy_cal:.0f})",
                            (int(px) + 10, int(py) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            cv2.imshow("Click to Move", display)
            key = cv2.waitKey(1) & 0xFF

            # Process click
            if click_target[0] is not None:
                wx_t, wy_t = click_target[0]
                click_target[0] = None
                if gantry:
                    goto_world_position(gantry, arm, wx_t, wy_t,
                                        home_x, home_y, arm_z, reach_radius_mm)
                else:
                    # Arm-only fallback (no gantry)
                    arm_x = home_x - wy_t
                    arm_y = home_y + wx_t
                    print(f"  World ({wx_t:.1f}, {wy_t:.1f}) → Arm ({arm_x:.1f}, {arm_y:.1f}, {arm_z:.1f})")
                    arm.move_to(arm_x, arm_y, arm_z)

            if key == ord('q'):
                break
            elif key == ord('h'):
                print("Homing...")
                if gantry:
                    print("  Arm to home position first...")
                    arm.move_to(home_x, home_y, arm_z)
                    gx, gy = gantry.get_position()
                    if abs(gx) > 0.1 or abs(gy) > 0.1:
                        print(f"  Moving gantry from ({gx:.1f}, {gy:.1f}) to (0, 0) ...")
                        gantry.move_absolute(0.0, 0.0, wait=True)
                        print(f"  Gantry at home")
                else:
                    arm.move_to(home_x, home_y, arm_z)
            elif key == ord('+') or key == ord('='):
                arm_z += 10.0
                print(f"  Z = {arm_z:.0f} mm")
            elif key == ord('-'):
                arm_z -= 10.0
                print(f"  Z = {arm_z:.0f} mm")

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        print("Cleaning up...")
        cv2.destroyAllWindows()
        cam.stop()
        if gantry:
            gantry.close()
        node.destroy_node()
        rclpy.shutdown()
        print("Done.")


if __name__ == "__main__":
    main()
