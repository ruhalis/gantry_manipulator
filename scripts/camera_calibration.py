#!/usr/bin/env python3
"""
Camera-to-Robot Calibration for RealSense D435I + Gantry + RoArm.

Workflow:
  1. Place markers at 4 known positions in the workspace (measure in mm).
  2. Run this script — click each marker in the camera feed.
  3. Enter the real-world coordinates (mm) for each point.
  4. Saves a calibration file (camera_calibration.yaml) with the homography matrix.

The calibration maps pixel (u, v) → world (x_mm, y_mm) using a perspective
transform. Depth from the RealSense provides Z, but the homography handles
the 2D pixel-to-world mapping on the workspace plane.

Usage:
  python3 camera_calibration.py [--output camera_calibration.yaml]
  python3 camera_calibration.py --verify  # verify existing calibration
"""

import argparse
import sys
import numpy as np
import cv2
import yaml

try:
    import pyrealsense2 as rs
except ImportError:
    print("ERROR: pyrealsense2 not installed. Install with: pip install pyrealsense2")
    sys.exit(1)


class RealSenseCamera:
    """Manages RealSense D435I pipeline for RGB + depth.

    NOTE: On this Jetson/aarch64 build, rs.align() and color intrinsics/
    extrinsics return NaN. We work around this by using the depth stream's
    valid intrinsics and scaling color pixel coordinates proportionally
    to map into the depth frame.
    """

    DEPTH_W = 848
    DEPTH_H = 480

    def __init__(self, color_width=1280, color_height=720, fps=30):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.color_width = color_width
        self.color_height = color_height
        self.config.enable_stream(rs.stream.color, color_width, color_height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, self.DEPTH_W, self.DEPTH_H, rs.format.z16, fps)
        self.depth_scale = None
        self.depth_intrinsics = None

    def start(self):
        profile = self.pipeline.start(self.config)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        depth_profile = profile.get_stream(rs.stream.depth)
        self.depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()

        print(f"  Depth intrinsics: {self.depth_intrinsics.width}x{self.depth_intrinsics.height}, "
              f"fx={self.depth_intrinsics.fx:.1f}, fy={self.depth_intrinsics.fy:.1f}, "
              f"ppx={self.depth_intrinsics.ppx:.1f}, ppy={self.depth_intrinsics.ppy:.1f}")
        print(f"  Depth scale: {self.depth_scale}")

        # Let auto-exposure settle
        for _ in range(30):
            self.pipeline.wait_for_frames()

    def get_frames(self):
        """Returns (color_image, depth_image, depth_frame) — unaligned."""
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return None, None, None
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        return color_image, depth_image, depth_frame

    def color_pixel_to_depth_pixel(self, u_color, v_color):
        """Map a color pixel (u,v) to the corresponding depth pixel.

        Uses proportional scaling — valid because the D435I color and
        depth sensors share nearly the same optical axis and FOV overlap
        is good in the central region.
        """
        du = int(round(u_color * self.DEPTH_W / self.color_width))
        dv = int(round(v_color * self.DEPTH_H / self.color_height))
        du = max(0, min(du, self.DEPTH_W - 1))
        dv = max(0, min(dv, self.DEPTH_H - 1))
        return du, dv

    def get_depth_at_pixel(self, depth_frame, u_color, v_color, kernel=5):
        """Get depth in mm at a color image pixel (u, v).
        Maps to depth frame via proportional scaling, reads with median kernel."""
        du, dv = self.color_pixel_to_depth_pixel(u_color, v_color)

        depth_image = np.asanyarray(depth_frame.get_data())
        h, w = depth_image.shape
        half = kernel // 2
        u_min = max(0, du - half)
        u_max = min(w, du + half + 1)
        v_min = max(0, dv - half)
        v_max = min(h, dv + half + 1)
        region = depth_image[v_min:v_max, u_min:u_max].astype(float)
        valid = region[region > 0]
        if len(valid) > 0:
            return float(np.median(valid)) * self.depth_scale * 1000.0  # mm
        return 0.0

    def stop(self):
        self.pipeline.stop()


def calibrate(output_path: str):
    """Interactive 4-point calibration."""
    cam = RealSenseCamera()
    cam.start()

    clicked_pixels = []
    current_frame = [None]

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked_pixels) < 4:
            clicked_pixels.append((x, y))
            print(f"  Point {len(clicked_pixels)}: pixel ({x}, {y})")

    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Calibration", on_mouse)

    print("=" * 60)
    print("CAMERA-TO-ROBOT CALIBRATION")
    print("=" * 60)
    print()
    print("Place 4 markers at known positions in the workspace.")
    print("Click each marker in the camera feed (in order).")
    print("Suggested order: top-left, top-right, bottom-right, bottom-left")
    print("  (or any consistent order — you'll enter world coords next)")
    print()
    print("The depth value under the cursor is shown live.")
    print("Avoid clicking where depth = 0 (no data, typically at frame edges).")
    print()
    print("Press 'q' to quit, 'r' to reset points.")
    print()

    mouse_pos = [0, 0]

    def on_mouse_move(event, x, y, flags, param):
        mouse_pos[0] = x
        mouse_pos[1] = y
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked_pixels) < 4:
            clicked_pixels.append((x, y))
            print(f"  Point {len(clicked_pixels)}: pixel ({x}, {y})")

    cv2.setMouseCallback("Calibration", on_mouse_move)

    while True:
        color_image, depth_image, depth_frame = cam.get_frames()
        if color_image is None:
            continue

        current_frame[0] = color_image.copy()
        display = color_image.copy()

        # Show live depth under cursor
        mu, mv = mouse_pos
        cursor_depth = cam.get_depth_at_pixel(depth_frame, mu, mv)
        depth_text = f"Cursor: ({mu}, {mv}) depth={cursor_depth:.0f}mm"
        cv2.putText(display, depth_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.drawMarker(display, (mu, mv), (255, 255, 0), cv2.MARKER_CROSS, 15, 1)

        # Draw clicked points with their depth
        for i, (px, py) in enumerate(clicked_pixels):
            d = cam.get_depth_at_pixel(depth_frame, px, py)
            cv2.circle(display, (px, py), 8, (0, 255, 0), 2)
            cv2.putText(display, f"P{i+1} d={d:.0f}", (px + 10, py - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw lines between points
        if len(clicked_pixels) >= 2:
            for i in range(len(clicked_pixels) - 1):
                cv2.line(display, clicked_pixels[i], clicked_pixels[i + 1],
                         (0, 255, 0), 1)
            if len(clicked_pixels) == 4:
                cv2.line(display, clicked_pixels[3], clicked_pixels[0],
                         (0, 255, 0), 1)

        status = f"Points: {len(clicked_pixels)}/4"
        if len(clicked_pixels) == 4:
            status += " -- Press ENTER to proceed"
        cv2.putText(display, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Calibration", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Calibration cancelled.")
            cam.stop()
            cv2.destroyAllWindows()
            return

        if key == ord('r'):
            clicked_pixels.clear()
            print("  Points reset.")

        if key == 13 and len(clicked_pixels) == 4:  # Enter
            break

    cv2.destroyAllWindows()

    # Grab fresh frames and read depth for each clicked point
    print("\nReading depth for calibration points...")
    pixel_depths = []
    # Average over multiple frames for stable depth
    depth_samples = {i: [] for i in range(4)}
    for _ in range(10):
        _, _, depth_frame = cam.get_frames()
        if depth_frame is None:
            continue
        for i, (px, py) in enumerate(clicked_pixels):
            d = cam.get_depth_at_pixel(depth_frame, px, py)
            if d > 0:
                depth_samples[i].append(d)

    for i, (px, py) in enumerate(clicked_pixels):
        samples = depth_samples[i]
        if samples:
            d = float(np.median(samples))
        else:
            d = 0.0
            print(f"  WARNING: No depth data for point {i+1} ({px}, {py}) — "
                  f"this pixel is outside depth sensor coverage")
        pixel_depths.append(d)
        print(f"  Point ({px}, {py}): depth = {d:.1f} mm ({len(samples)} samples)")

    cam.stop()

    # Get world coordinates from user
    print()
    print("Now enter the real-world coordinates (mm) for each point.")
    print("Use your gantry+arm coordinate system (origin = home position).")
    print()

    world_points = []
    for i, (px, py) in enumerate(clicked_pixels):
        while True:
            try:
                coords = input(f"  Point {i+1} pixel ({px}, {py}), depth {pixel_depths[i]:.0f}mm → world x_mm, y_mm: ")
                parts = coords.replace(",", " ").split()
                wx = float(parts[0])
                wy = float(parts[1])
                world_points.append((wx, wy))
                break
            except (ValueError, IndexError):
                print("    Enter two numbers: x_mm y_mm (e.g., 100 200)")

    # Compute homography: pixel → world
    src = np.array(clicked_pixels, dtype=np.float32)
    dst = np.array(world_points, dtype=np.float32)
    H, status = cv2.findHomography(src, dst)

    if H is None:
        print("ERROR: Could not compute homography. Points may be collinear.")
        return

    # Verify: transform pixel points back and check error
    print()
    print("Calibration verification:")
    errors = []
    for i, (px, py) in enumerate(clicked_pixels):
        pt = np.array([px, py, 1.0])
        result = H @ pt
        result /= result[2]
        wx_est, wy_est = result[0], result[1]
        wx_true, wy_true = world_points[i]
        err = np.sqrt((wx_est - wx_true)**2 + (wy_est - wy_true)**2)
        errors.append(err)
        print(f"  P{i+1}: estimated ({wx_est:.1f}, {wy_est:.1f}), "
              f"actual ({wx_true:.1f}, {wy_true:.1f}), error: {err:.2f} mm")

    print(f"  Mean error: {np.mean(errors):.2f} mm")
    print(f"  Max error:  {np.max(errors):.2f} mm")

    # Save calibration
    calib_data = {
        'homography': H.tolist(),
        'pixel_points': [list(p) for p in clicked_pixels],
        'world_points': [list(p) for p in world_points],
        'pixel_depths_mm': pixel_depths,
        'mean_workspace_depth_mm': float(np.mean(pixel_depths)),
    }

    with open(output_path, 'w') as f:
        yaml.dump(calib_data, f, default_flow_style=False)

    print(f"\nCalibration saved to {output_path}")


def load_calibration(path: str) -> np.ndarray:
    """Load homography matrix from calibration file."""
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    H = np.array(data['homography'], dtype=np.float64)
    return H, data


def pixel_to_world(H: np.ndarray, u: int, v: int) -> tuple:
    """Convert pixel (u, v) to world (x_mm, y_mm) using homography."""
    pt = np.array([u, v, 1.0])
    result = H @ pt
    result /= result[2]
    return float(result[0]), float(result[1])


def verify_calibration(calib_path: str):
    """Live verification — shows camera feed with world coords overlay."""
    H, data = load_calibration(calib_path)
    cam = RealSenseCamera()
    cam.start()

    print("Verification mode — hover mouse to see world coordinates.")
    print("Press 'q' to quit.")

    mouse_pos = [0, 0]

    def on_mouse(event, x, y, flags, param):
        mouse_pos[0] = x
        mouse_pos[1] = y

    cv2.namedWindow("Verify Calibration", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Verify Calibration", on_mouse)

    while True:
        color_image, depth_image, depth_frame = cam.get_frames()
        if color_image is None:
            continue

        display = color_image.copy()
        u, v = mouse_pos
        wx, wy = pixel_to_world(H, u, v)
        depth_mm = cam.get_depth_at_pixel(depth_frame, u, v)

        # Draw crosshair at mouse
        cv2.drawMarker(display, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

        # Show world coordinates
        text = f"Pixel: ({u}, {v})  World: ({wx:.1f}, {wy:.1f}) mm  Depth: {depth_mm:.0f} mm"
        cv2.putText(display, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Draw calibration points
        for i, (px, py) in enumerate(data['pixel_points']):
            cv2.circle(display, (int(px), int(py)), 6, (255, 0, 0), 2)
            wx_cal, wy_cal = data['world_points'][i]
            cv2.putText(display, f"({wx_cal:.0f},{wy_cal:.0f})",
                        (int(px) + 10, int(py) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        cv2.imshow("Verify Calibration", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.stop()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Camera-to-robot calibration")
    parser.add_argument("--output", default="config/camera_calibration.yaml",
                        help="Output calibration file (default: config/camera_calibration.yaml)")
    parser.add_argument("--verify", action="store_true",
                        help="Verify existing calibration (live overlay)")
    args = parser.parse_args()

    if args.verify:
        verify_calibration(args.output)
    else:
        calibrate(args.output)


if __name__ == "__main__":
    main()
