#!/usr/bin/env python3
"""
Camera Click-to-Move Controller for RoArm.

Shows live camera feed with world coordinate overlay (like --verify mode).
Click on a point to move the arm there. The arm starts at its home position
which represents the world origin (0, 0). An offset defines where the arm's
end-effector actually is in arm coordinates when at "home".

Uses the existing camera calibration (homography) to convert pixel → world mm.

Prerequisites:
  1. Run camera_calibration.py first to generate camera_calibration.yaml
  2. ROS2 stack must be running (roarm_driver + command_control.launch.py)

Usage:
  python3 scripts/camera_click_move.py
  python3 scripts/camera_click_move.py --home-x 200 --home-y 0 --arm-z -15
  python3 scripts/camera_click_move.py --calib config/camera_calibration.yaml
"""

import argparse
import sys
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


def main():
    parser = argparse.ArgumentParser(description="Click on camera to move arm")
    parser.add_argument("--config", default="config/pick_controller.yaml",
                        help="Config YAML file (default: config/pick_controller.yaml)")
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    print(f"Config loaded from {args.config}")

    arm_cfg = cfg['arm']
    cam_cfg = cfg['camera']
    home_x = arm_cfg.get('home_x_mm', 200.0)
    home_y = arm_cfg.get('home_y_mm', 0.0)
    arm_z = arm_cfg.get('default_z_mm', -15.0)
    calib_file = cam_cfg.get('calibration_file', 'config/camera_calibration.yaml')

    # Load calibration
    H, calib_data = load_calibration(calib_file)
    print(f"Calibration loaded from {calib_file}")

    # Start camera
    cam = RealSenseCamera(cam_cfg.get('color_width', 1280),
                          cam_cfg.get('color_height', 720),
                          cam_cfg.get('fps', 30))
    cam.start()

    # Init ROS2
    rclpy.init()
    node = Node('camera_click_move')
    arm = ArmMover(node)
    if not arm.wait_for_service():
        cam.stop()
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

    print()
    print("=" * 60)
    print("CAMERA CLICK-TO-MOVE")
    print("=" * 60)
    print(f"  Home offset: arm ({home_x:.0f}, {home_y:.0f}) mm = world (0, 0)")
    print(f"  Arm Z: {arm_z:.0f} mm")
    print()
    print("  Left-click: move arm to that world position")
    print("  'h': return arm to home")
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

            info = f"Home: ({home_x:.0f}, {home_y:.0f})  Z: {arm_z:.0f} mm  |  Click=move  h=home  q=quit"
            cv2.putText(display, info, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

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
                # Rotate world (camera) coords 90° CCW into the arm frame
                # before offsetting from home: (x, y) -> (-y, x)
                arm_x = home_x - wy_t
                arm_y = home_y + wx_t
                print(f"  World ({wx_t:.1f}, {wy_t:.1f}) → Arm ({arm_x:.1f}, {arm_y:.1f}, {arm_z:.1f})")
                arm.move_to(arm_x, arm_y, arm_z)

            if key == ord('q'):
                break
            elif key == ord('h'):
                print("Homing arm...")
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
        node.destroy_node()
        rclpy.shutdown()
        print("Done.")


if __name__ == "__main__":
    main()
