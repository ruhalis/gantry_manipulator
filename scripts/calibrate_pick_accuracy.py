#!/usr/bin/env python3
"""
End-to-end pick accuracy calibration.

For each point in a 3x3 world grid:
  1. Drive the full gantry+arm pipeline to (target_x, target_y) at pick_z.
  2. Prompt user to place an ArUco marker under the gripper tip.
  3. Detect the marker in the camera, map its center pixel through the
     existing homography to get the actual landed world coords.
  4. Record residual = target - measured.

After all 9 points, fit a 2D affine correction
    [x'; y'] = A * [x; y] + b
that takes a *desired* world point and returns the *commanded* world point
which (empirically) lands at the desired one. Save to
config/pick_correction.yaml.

Both camera_sequence_pick.py and camera_sort_pick.py can apply this
correction at the top of goto_world_position to mop up homography +
gantry zero + arm offset error in one pass.

Usage:
  python3 scripts/calibrate_pick_accuracy.py
  python3 scripts/calibrate_pick_accuracy.py --grid-x 450 565 680 --grid-y 285 460 635
  python3 scripts/calibrate_pick_accuracy.py --dry-run     # no robot, simulate
"""

import argparse
import math
import sys
import threading
import time

import cv2
import numpy as np
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

from camera_calibration import RealSenseCamera, load_calibration, pixel_to_world


GRIPPER_OPEN = 3.14
GRIPPER_SETTLE_S = 1.0


# ---------- Robot plumbing (kept self-contained, mirrors camera_sort_pick.py) ----------

class GantryController:
    def __init__(self, port: str, baud: int = 115200, timeout: float = 2.0):
        self.ser = serial.Serial(port, baud, timeout=timeout)
        time.sleep(2)
        while self.ser.in_waiting:
            self.ser.readline()
        self.pos_x = 0.0
        self.pos_y = 0.0
        self._lock = threading.Lock()

    def move_relative(self, dx_mm, dy_mm, wait=True):
        with self._lock:
            cmd = f"{dx_mm:.2f}, {dy_mm:.2f}\n"
            self.ser.write(cmd.encode('utf-8'))
            self.pos_x += dx_mm
            self.pos_y += dy_mm
        if wait:
            self._wait_for_move(dx_mm, dy_mm)
        return self.pos_x, self.pos_y

    def move_absolute(self, tx, ty, wait=True):
        return self.move_relative(tx - self.pos_x, ty - self.pos_y, wait=wait)

    def _wait_for_move(self, dx, dy):
        steps_per_mm = 3200.0 / 125.0
        max_speed = 4000.0 / steps_per_mm
        accel = 2000.0 / steps_per_mm
        dist = max(abs(dx), abs(dy))
        if dist < 0.01:
            return
        t_a = max_speed / accel
        d_a = 0.5 * accel * t_a ** 2
        if dist < 2 * d_a:
            t_total = 2.0 * math.sqrt(dist / accel)
        else:
            t_total = 2 * t_a + (dist - 2 * d_a) / max_speed
        time.sleep(t_total + 0.3)

    def get_position(self):
        return self.pos_x, self.pos_y

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()


def compute_gantry_target(tx, ty, gx, gy, hx, hy, R):
    cwx = gx + hx
    cwy = gy + hy
    dx = tx - cwx
    dy = ty - cwy
    d = math.hypot(dx, dy)
    if d <= R or d < 1e-6:
        return gx, gy, dx, dy
    s = (d - R) / d
    nwx = cwx + dx * s
    nwy = cwy + dy * s
    return nwx - hx, nwy - hy, tx - nwx, ty - nwy


class ArmMover:
    def __init__(self, node):
        self._client = node.create_client(MovePointCmd, '/move_point_cmd')
        self._gripper_pub = node.create_publisher(Float32, '/gripper_cmd', 10)

    def wait_for_service(self, t=10.0):
        return self._client.wait_for_service(timeout_sec=t)

    def move_to(self, x_mm, y_mm, z_mm):
        req = MovePointCmd.Request()
        req.x = x_mm / 1000.0
        req.y = y_mm / 1000.0
        req.z = z_mm / 1000.0
        future = self._client.call_async(req)
        t0 = time.time()
        while not future.done():
            if time.time() - t0 > 30.0:
                return False
            time.sleep(0.01)
        result = future.result()
        return bool(result and result.success)

    def gripper(self, value):
        if self._gripper_pub.get_subscription_count() == 0:
            return
        msg = Float32()
        msg.data = float(value)
        self._gripper_pub.publish(msg)


def goto_world_position(gantry, arm, wx_t, wy_t, hx, hy, z, R):
    gx, gy = gantry.get_position() if gantry else (0.0, 0.0)
    gtx, gty, rx, ry = compute_gantry_target(wx_t, wy_t, gx, gy, hx, hy, R)
    if gantry:
        dgx = gtx - gx
        dgy = gty - gy
        if abs(dgx) > 0.1 or abs(dgy) > 0.1:
            gantry.move_relative(dgx, dgy, wait=True)
    ax = hx - ry
    ay = hy + rx
    arm.move_to(ax, ay, z)
    return ax, ay


# ---------- ArUco detection ----------

ARUCO_DICTS = {
    'DICT_4X4_50': cv2.aruco.DICT_4X4_50,
    'DICT_4X4_100': cv2.aruco.DICT_4X4_100,
    'DICT_5X5_50': cv2.aruco.DICT_5X5_50,
    'DICT_6X6_50': cv2.aruco.DICT_6X6_50,
}


def make_aruco_detector(dict_name):
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dict_name])
    # Support both old and new opencv aruco APIs
    if hasattr(cv2.aruco, 'ArucoDetector'):
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        def detect(img):
            return detector.detectMarkers(img)
    else:
        params = cv2.aruco.DetectorParameters_create()
        def detect(img):
            return cv2.aruco.detectMarkers(img, aruco_dict, parameters=params)
    return detect


def detect_marker_center(detect_fn, frame, marker_id):
    corners, ids, _ = detect_fn(frame)
    if ids is None:
        return None
    ids = ids.flatten()
    for i, mid in enumerate(ids):
        if int(mid) == marker_id:
            c = corners[i].reshape(-1, 2)
            return float(c[:, 0].mean()), float(c[:, 1].mean())
    return None


def measure_world_xy(cam, detect_fn, marker_id, H, n_frames):
    """Average the marker's center pixel over n_frames, return world (x, y) mm
    or None if the marker was never seen."""
    pix_us, pix_vs = [], []
    last_frame = None
    for _ in range(n_frames):
        color, _, _ = cam.get_frames()
        if color is None:
            continue
        last_frame = color
        c = detect_marker_center(detect_fn, color, marker_id)
        if c is not None:
            pix_us.append(c[0])
            pix_vs.append(c[1])
    if not pix_us:
        return None, last_frame
    u = float(np.median(pix_us))
    v = float(np.median(pix_vs))
    wx, wy = pixel_to_world(H, u, v)
    return (wx, wy, u, v, len(pix_us)), last_frame


# ---------- Affine fit ----------

def fit_affine(targets, measured):
    """
    Find A (2x2) and b (2,) such that:
        commanded = A @ desired + b
    given samples where we COMMANDED `targets[i]` and OBSERVED `measured[i]`.

    Logic: at each grid point we asked the robot to go to T_i and it actually
    landed at M_i. We want a function f such that if we want the robot to land
    at D, we command f(D). Empirically the robot maps commanded C -> landed M
    with M ~ A_fwd C + b_fwd. So f(D) = A_fwd^-1 (D - b_fwd).

    Fit forward map from targets (commanded) -> measured (landed), then invert.
    """
    T = np.asarray(targets, dtype=np.float64)        # commanded
    M = np.asarray(measured, dtype=np.float64)       # landed
    n = T.shape[0]
    # Solve M = T_aug @ P  where T_aug = [T | 1], P is 3x2
    T_aug = np.hstack([T, np.ones((n, 1))])
    P, *_ = np.linalg.lstsq(T_aug, M, rcond=None)    # P shape (3, 2)
    A_fwd = P[:2, :].T                               # M = A_fwd @ T + b_fwd
    b_fwd = P[2, :]
    A_inv = np.linalg.inv(A_fwd)
    b_inv = -A_inv @ b_fwd                           # commanded = A_inv @ desired + b_inv
    return A_fwd, b_fwd, A_inv, b_inv


def apply_affine(A, b, x, y):
    v = A @ np.array([x, y]) + b
    return float(v[0]), float(v[1])


# ---------- Main ----------

def main():
    p = argparse.ArgumentParser(description="Pick accuracy calibration via ArUco")
    p.add_argument("--config", default="config/pick_controller.yaml")
    p.add_argument("--grid-x", type=float, nargs='+', default=None,
                   help="Override grid x coords (mm)")
    p.add_argument("--grid-y", type=float, nargs='+', default=None,
                   help="Override grid y coords (mm)")
    p.add_argument("--marker-id", type=int, default=None)
    p.add_argument("--dict", default=None, help="ArUco dictionary name")
    p.add_argument("--dry-run", action="store_true",
                   help="Skip robot motion (for testing detection/IO)")
    p.add_argument("--auto", action="store_true",
                   help="No ENTER prompts — just settle and measure")
    args = p.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    arm_cfg = cfg['arm']
    cam_cfg = cfg['camera']
    gantry_cfg = cfg.get('gantry', {})
    cal_cfg = cfg.get('calibration', {})

    home_x = arm_cfg.get('home_x_mm', 200.0)
    home_y = arm_cfg.get('home_y_mm', 0.0)
    pick_z = arm_cfg.get('default_z_mm', -15.0)
    travel_z = 0.0
    reach = arm_cfg.get('reach_radius_mm', 200.0)
    calib_file = cam_cfg.get('calibration_file', 'config/camera_calibration.yaml')

    grid_x = args.grid_x or cal_cfg.get('grid_x_mm', [450.0, 565.0, 680.0])
    grid_y = args.grid_y or cal_cfg.get('grid_y_mm', [285.0, 460.0, 635.0])
    dict_name = args.dict or cal_cfg.get('aruco_dict', 'DICT_4X4_50')
    marker_id = args.marker_id if args.marker_id is not None \
        else int(cal_cfg.get('aruco_id', 0))
    measure_frames = int(cal_cfg.get('measure_frames', 15))
    settle_s = float(cal_cfg.get('settle_seconds', 1.5))
    correction_file = cal_cfg.get('correction_file', 'config/pick_correction.yaml')

    if dict_name not in ARUCO_DICTS:
        print(f"ERROR: unsupported aruco dict {dict_name}. Options: {list(ARUCO_DICTS)}")
        sys.exit(1)

    H, _ = load_calibration(calib_file)
    print(f"Homography loaded from {calib_file}")
    print(f"ArUco: dict={dict_name} id={marker_id}")
    print(f"Grid X: {grid_x}")
    print(f"Grid Y: {grid_y}")
    print(f"Pick Z: {pick_z} mm   Reach: {reach} mm")
    print()

    cam = RealSenseCamera(
        cam_cfg.get('color_width', 1280),
        cam_cfg.get('color_height', 720),
        cam_cfg.get('fps', 30),
    )
    cam.start()
    detect_fn = make_aruco_detector(dict_name)

    gantry = None
    arm = None
    node = None
    executor = None
    if not args.dry_run:
        try:
            gantry = GantryController(gantry_cfg.get('port', '/dev/ttyUSB1'),
                                      gantry_cfg.get('baud', 115200))
            print("Gantry connected.")
        except serial.SerialException as e:
            print(f"WARNING: gantry not available: {e}")
            gantry = None

        rclpy.init()
        node = Node('calibrate_pick_accuracy')
        arm = ArmMover(node)
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        threading.Thread(target=executor.spin, daemon=True).start()

        print("Waiting for /move_point_cmd ...")
        if not arm.wait_for_service(10.0):
            print("ERROR: /move_point_cmd not available")
            cam.stop()
            sys.exit(1)
        arm.gripper(GRIPPER_OPEN)
        time.sleep(GRIPPER_SETTLE_S)

    cv2.namedWindow("Calibrate", cv2.WINDOW_NORMAL)

    # Build the grid (snake order to minimise gantry travel)
    points = []
    for j, gy in enumerate(grid_y):
        xs = grid_x if (j % 2 == 0) else list(reversed(grid_x))
        for gx in xs:
            points.append((float(gx), float(gy)))

    targets, measured = [], []
    skipped = []

    print(f"\n=== {len(points)} grid points ===")
    for i, (tx, ty) in enumerate(points):
        print(f"\n[{i+1}/{len(points)}] target = ({tx:.1f}, {ty:.1f}) mm")
        if not args.dry_run:
            goto_world_position(gantry, arm, tx, ty, home_x, home_y, travel_z, reach)
            goto_world_position(gantry, arm, tx, ty, home_x, home_y, pick_z, reach)
        time.sleep(settle_s)

        if not args.auto:
            input("  Place marker so its center is exactly under the gripper tip, then press ENTER...")

        result, frame = measure_world_xy(cam, detect_fn, marker_id, H, measure_frames)
        if result is None:
            print("  MARKER NOT DETECTED — skipping this point")
            skipped.append((tx, ty))
            if not args.dry_run:
                goto_world_position(gantry, arm, tx, ty, home_x, home_y, travel_z, reach)
            continue
        wx, wy, u, v, n_seen = result
        ex = wx - tx
        ey = wy - ty
        print(f"  measured = ({wx:.2f}, {wy:.2f}) mm  pixel=({u:.0f},{v:.0f}) "
              f"frames={n_seen}/{measure_frames}")
        print(f"  residual = ({ex:+.2f}, {ey:+.2f}) mm   |e|={math.hypot(ex, ey):.2f}")

        targets.append((tx, ty))
        measured.append((wx, wy))

        if frame is not None:
            disp = frame.copy()
            cv2.circle(disp, (int(u), int(v)), 8, (0, 255, 0), 2)
            cv2.putText(disp, f"P{i+1} t=({tx:.0f},{ty:.0f}) m=({wx:.1f},{wy:.1f}) e=({ex:+.1f},{ey:+.1f})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Calibrate", disp)
            cv2.waitKey(300)

        if not args.dry_run:
            goto_world_position(gantry, arm, tx, ty, home_x, home_y, travel_z, reach)

    # Return home
    if not args.dry_run:
        if gantry:
            gantry.move_absolute(0.0, 0.0, wait=True)
        arm.move_to(home_x, home_y, travel_z)

    cv2.destroyAllWindows()
    cam.stop()
    if gantry:
        gantry.close()
    if executor is not None:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()

    if len(targets) < 4:
        print(f"\nNot enough points ({len(targets)}) for a robust affine fit. Aborting save.")
        sys.exit(1)

    # ---- Fit & report ----
    A_fwd, b_fwd, A_inv, b_inv = fit_affine(targets, measured)

    print("\n=== RESIDUAL STATS ===")
    raw_err = [math.hypot(m[0] - t[0], m[1] - t[1]) for t, m in zip(targets, measured)]
    print(f"  raw          mean={np.mean(raw_err):.2f} max={np.max(raw_err):.2f} mm")

    # If we'd commanded the corrected target, the forward map predicts where
    # we'd land. Use that to estimate post-correction residual.
    post_err = []
    for t, m in zip(targets, measured):
        c = apply_affine(A_inv, b_inv, t[0], t[1])     # what we'd command
        landed_pred = apply_affine(A_fwd, b_fwd, c[0], c[1])  # forward model
        post_err.append(math.hypot(landed_pred[0] - t[0], landed_pred[1] - t[1]))
    print(f"  post-affine  mean={np.mean(post_err):.2f} max={np.max(post_err):.2f} mm  (model fit)")

    # ---- Save ----
    out = {
        'description': 'Affine correction: commanded = A @ desired + b. '
                       'Apply at the start of goto_world_position before splitting gantry/arm.',
        'A': A_inv.tolist(),
        'b': b_inv.tolist(),
        'forward_A': A_fwd.tolist(),
        'forward_b': b_fwd.tolist(),
        'samples': [
            {'target': list(t), 'measured': list(m)}
            for t, m in zip(targets, measured)
        ],
        'skipped': [list(s) for s in skipped],
        'aruco_dict': dict_name,
        'aruco_id': marker_id,
        'pick_z_mm': pick_z,
        'home_x_mm': home_x,
        'home_y_mm': home_y,
    }
    with open(correction_file, 'w') as f:
        yaml.dump(out, f, default_flow_style=False, sort_keys=False)
    print(f"\nSaved {correction_file}")
    print("\nTo apply: load A, b at startup of camera_sequence_pick.py / camera_sort_pick.py")
    print("and inside goto_world_position do: wx_t, wy_t = A @ [wx_t, wy_t] + b BEFORE the split.")


if __name__ == "__main__":
    main()
