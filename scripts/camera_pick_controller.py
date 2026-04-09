#!/usr/bin/env python3
"""
Camera-guided Pick & Place Controller for Gantry + RoArm.

Uses RealSense D435I with a pre-computed homography (from camera_calibration.py)
to convert clicked pixel positions into real-world coordinates, then commands
the gantry+arm system to pick at that location.

Modes:
  - Click mode (default): click on camera feed to pick at that position
  - YOLO mode (--yolo): auto-detect objects and pick them (requires ultralytics)

Prerequisites:
  1. Run camera_calibration.py first to generate camera_calibration.yaml
  2. ROS2 stack must be running (roarm_driver + command_control.launch.py)
  3. Gantry Arduino connected

Usage:
  python3 scripts/camera_pick_controller.py --config config/pick_controller.yaml
  python3 scripts/camera_pick_controller.py --config config/pick_controller.yaml --yolo yolov8n.pt
"""

import argparse
import os
import sys
import time
import numpy as np
import cv2
import yaml

try:
    import pyrealsense2 as rs
except ImportError:
    print("ERROR: pyrealsense2 not installed")
    sys.exit(1)

import rclpy
from rclpy.node import Node
from roarm_moveit.srv import MovePointCmd

from camera_calibration import RealSenseCamera, load_calibration, pixel_to_world
from gantry_arm_controller import (
    GantryController, RoArmController,
    goto_position, go_home,
    ARM_REACH_RADIUS_MM, ARM_DEFAULT_Z_MM,
)


def load_config(config_path: str) -> dict:
    """Load pick controller config from YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def pick_at_world(gantry, arm, wx_mm, wy_mm, cfg: dict):
    """
    Pick sequence:
      1. Move gantry+arm to (wx, wy) at approach height
      2. Lower to pick height
      3. (gripper close would go here)
    """
    pick_cfg = cfg['pick']
    reach = cfg['arm']['reach_radius_mm']
    approach_z = pick_cfg['approach_z_mm']
    pick_z = pick_cfg['pick_z_mm']

    print(f"\n--- PICK at world ({wx_mm:.1f}, {wy_mm:.1f}) mm ---")

    print("  Approaching...")
    goto_position(gantry, arm, wx_mm, wy_mm,
                  arm_z_mm=approach_z, reach_radius_mm=reach)
    time.sleep(0.5)

    print("  Lowering to pick height...")
    goto_position(gantry, arm, wx_mm, wy_mm,
                  arm_z_mm=pick_z, reach_radius_mm=reach)
    time.sleep(0.5)

    # TODO: close gripper here via /gripper_cmd topic
    print("  (gripper close not yet implemented)")

    print("  Lifting...")
    goto_position(gantry, arm, wx_mm, wy_mm,
                  arm_z_mm=approach_z, reach_radius_mm=reach)


def click_mode(cam, H, calib_data, gantry, arm, cfg):
    """Click on the camera feed to pick at that position."""
    pick_targets = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            wx, wy = pixel_to_world(H, x, y)
            pick_targets.append((x, y, wx, wy))
            print(f"  Queued pick: pixel ({x}, {y}) → world ({wx:.1f}, {wy:.1f}) mm")

    cv2.namedWindow("Pick Controller", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Pick Controller", on_mouse)

    print("=" * 60)
    print("CAMERA PICK CONTROLLER — Click Mode")
    print("=" * 60)
    print("  Left-click: queue a pick at that position")
    print("  'p': execute all queued picks")
    print("  'c': clear queue")
    print("  'h': home gantry + arm")
    print("  'q': quit")
    print()

    while True:
        color_image, depth_image, depth_frame = cam.get_frames()
        if color_image is None:
            continue

        display = color_image.copy()

        # Draw queued picks
        for i, (px, py, wx, wy) in enumerate(pick_targets):
            cv2.circle(display, (px, py), 10, (0, 0, 255), 2)
            cv2.putText(display, f"{i+1}: ({wx:.0f},{wy:.0f})",
                        (px + 12, py - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Draw calibration reference points
        for i, (px, py) in enumerate(calib_data['pixel_points']):
            cv2.circle(display, (int(px), int(py)), 4, (255, 0, 0), -1)

        status = f"Queue: {len(pick_targets)} picks | 'p'=execute 'c'=clear 'h'=home 'q'=quit"
        cv2.putText(display, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Pick Controller", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('c'):
            pick_targets.clear()
            print("  Queue cleared.")

        elif key == ord('h'):
            go_home(gantry, arm)

        elif key == ord('p') and pick_targets:
            print(f"\nExecuting {len(pick_targets)} pick(s)...")
            for i, (px, py, wx, wy) in enumerate(pick_targets):
                depth_mm = cam.get_depth_at_pixel(depth_frame, px, py)
                print(f"\n  Pick {i+1}/{len(pick_targets)}: "
                      f"world ({wx:.1f}, {wy:.1f}) mm, depth {depth_mm:.0f} mm")
                pick_at_world(gantry, arm, wx, wy, cfg)
                time.sleep(1.0)

            print("\nAll picks done. Homing...")
            go_home(gantry, arm)
            pick_targets.clear()


def yolo_mode(cam, H, calib_data, gantry, arm, model_path, cfg):
    """YOLO object detection + pick."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Install with: pip install ultralytics")
        return

    model = YOLO(model_path)
    print(f"YOLO model loaded: {model_path}")

    print("=" * 60)
    print("CAMERA PICK CONTROLLER — YOLO Mode")
    print("=" * 60)
    print("  'd': detect objects")
    print("  'p': pick all detected objects")
    print("  'h': home")
    print("  'q': quit")
    print()

    detections = []

    while True:
        color_image, depth_image, depth_frame = cam.get_frames()
        if color_image is None:
            continue

        display = color_image.copy()

        # Draw detections
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            cx, cy = det['center_px']
            wx, wy = det['world_xy']
            label = det['label']
            conf = det['confidence']

            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(display, (cx, cy), 5, (0, 0, 255), -1)
            text = f"{label} {conf:.2f} ({wx:.0f},{wy:.0f})mm"
            cv2.putText(display, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        status = f"Detections: {len(detections)} | 'd'=detect 'p'=pick 'h'=home 'q'=quit"
        cv2.putText(display, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Pick Controller", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('h'):
            go_home(gantry, arm)

        elif key == ord('d'):
            print("  Running YOLO detection...")
            results = model(color_image, verbose=False)
            detections.clear()

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    wx, wy = pixel_to_world(H, cx, cy)
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]
                    conf = float(box.conf[0])

                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'center_px': (cx, cy),
                        'world_xy': (wx, wy),
                        'label': label,
                        'confidence': conf,
                    })

            print(f"  Found {len(detections)} objects")
            for d in detections:
                print(f"    {d['label']} ({d['confidence']:.2f}): "
                      f"world ({d['world_xy'][0]:.1f}, {d['world_xy'][1]:.1f}) mm")

        elif key == ord('p') and detections:
            print(f"\nPicking {len(detections)} objects...")
            for i, det in enumerate(detections):
                wx, wy = det['world_xy']
                print(f"\n  Pick {i+1}: {det['label']} at ({wx:.1f}, {wy:.1f}) mm")
                pick_at_world(gantry, arm, wx, wy, cfg)
                time.sleep(1.0)

            print("\nAll picks done. Homing...")
            go_home(gantry, arm)
            detections.clear()


def main():
    parser = argparse.ArgumentParser(description="Camera-guided pick & place")
    parser.add_argument("--config", default="config/pick_controller.yaml",
                        help="Config YAML file (default: config/pick_controller.yaml)")
    parser.add_argument("--yolo", type=str, default=None,
                        help="Override YOLO model path (overrides config)")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    print(f"Config loaded from {args.config}")

    # Determine YOLO model: CLI flag overrides config
    yolo_model = args.yolo or cfg.get('yolo', {}).get('model_path', '') or None

    # Load calibration
    calib_file = cfg['camera']['calibration_file']
    print("Loading calibration...")
    H, calib_data = load_calibration(calib_file)
    print(f"  Homography loaded from {calib_file}")

    # Start camera
    print("Starting RealSense...")
    cam_cfg = cfg['camera']
    cam = RealSenseCamera(cam_cfg['color_width'], cam_cfg['color_height'], cam_cfg['fps'])
    cam.start()

    # Connect gantry
    gantry_cfg = cfg['gantry']
    print(f"Connecting gantry on {gantry_cfg['port']}...")
    gantry = GantryController(gantry_cfg['port'], gantry_cfg['baud'])

    # Init ROS2
    rclpy.init()
    ros2_node = Node('camera_pick_controller')
    arm = RoArmController(node=ros2_node)

    if not arm.wait_for_service(timeout_sec=10.0):
        print("ERROR: /move_point_cmd not available")
        cam.stop()
        gantry.close()
        ros2_node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    try:
        if yolo_model:
            yolo_mode(cam, H, calib_data, gantry, arm, yolo_model, cfg)
        else:
            click_mode(cam, H, calib_data, gantry, arm, cfg)
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        print("Cleaning up...")
        cv2.destroyAllWindows()
        cam.stop()
        gantry.close()
        arm.close()
        ros2_node.destroy_node()
        rclpy.shutdown()
        print("Done.")


if __name__ == "__main__":
    main()
