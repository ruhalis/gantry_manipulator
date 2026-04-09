#!/usr/bin/env python3
"""Quick diagnostic — tests proportional color-to-depth mapping."""
import numpy as np
import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)

profile = pipeline.start(config)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
di = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
print(f"Depth: {di.width}x{di.height}, fx={di.fx:.1f} fy={di.fy:.1f} ppx={di.ppx:.1f} ppy={di.ppy:.1f}")
print(f"Depth scale: {depth_scale}")

for _ in range(30):
    pipeline.wait_for_frames()

frames = pipeline.wait_for_frames()
depth_frame = frames.get_depth_frame()
depth_img = np.asanyarray(depth_frame.get_data())
nonzero = np.count_nonzero(depth_img)
print(f"\nRaw depth: nonzero={nonzero}/{depth_img.size} ({100*nonzero/depth_img.size:.1f}%)")

print("\n--- Proportional color->depth mapping ---")
COLOR_W, COLOR_H = 1280, 720
DEPTH_W, DEPTH_H = 848, 480

test_points = [
    (640, 360, "center"),
    (320, 180, "top-left area"),
    (960, 540, "bottom-right area"),
    (547, 144, "calibration P1"),
    (889, 134, "calibration P2"),
    (934, 531, "calibration P3"),
    (400, 540, "calibration P4"),
]

for u_color, v_color, label in test_points:
    du = int(round(u_color * DEPTH_W / COLOR_W))
    dv = int(round(v_color * DEPTH_H / COLOR_H))
    du = max(0, min(du, DEPTH_W - 1))
    dv = max(0, min(dv, DEPTH_H - 1))

    # Median over 5x5 kernel
    half = 2
    region = depth_img[max(0,dv-half):min(DEPTH_H,dv+half+1),
                       max(0,du-half):min(DEPTH_W,du+half+1)].astype(float)
    valid = region[region > 0]
    depth_mm = float(np.median(valid)) * depth_scale * 1000.0 if len(valid) > 0 else 0.0

    print(f"  {label}: color({u_color},{v_color}) -> depth({du},{dv}) = {depth_mm:.0f} mm")

pipeline.stop()
print("\nDone.")
