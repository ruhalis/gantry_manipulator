#!/usr/bin/env python3
"""
Play a video with YOLO detection and overlay two user-defined colored squares (blue and red).

Phase 1: Show first frame, user clicks 4 corners for the BLUE square, then 4 corners for the RED square.
Phase 2: Play video with YOLO tracking (persistent IDs, no confidence scores) and alpha-blended squares.

Usage:
  python3 scripts/run_video_yolo.py sequence_20260414_123221.mp4 best.pt
"""

import sys
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except ImportError:
    print("ERROR: deep_sort_realtime not installed. Run: pip install deep_sort_realtime")
    sys.exit(1)

ALPHA = 0.35  # opacity of colored squares (0=invisible, 1=solid)
BLUE_COLOR  = (255, 0, 0)    # BGR blue
RED_COLOR   = (0, 0, 255)    # BGR red


# ── click-collection helper ──────────────────────────────────────────────────

class PointCollector:
    def __init__(self):
        self.points = []

    def callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))


def collect_quad(window_name, base_frame, prompt, color, existing_quads=None):
    """
    Show base_frame in window_name, let user click 4 points.
    Returns list of 4 (x, y) tuples in click order.
    """
    collector = PointCollector()
    cv2.setMouseCallback(window_name, collector.callback)

    print(f"\n{prompt}")
    print("  Click 4 corners (any order). They will be connected as a filled quad.")

    while len(collector.points) < 4:
        display = base_frame.copy()

        # Draw already-defined quads
        if existing_quads:
            for pts, col in existing_quads:
                _draw_quad_alpha(display, pts, col)

        # Draw points collected so far for this quad
        for i, pt in enumerate(collector.points):
            cv2.circle(display, pt, 6, color, -1)
            if i > 0:
                cv2.line(display, collector.points[i - 1], pt, color, 2)

        # Instruction overlay
        remaining = 4 - len(collector.points)
        label = f"{prompt}  —  {remaining} click(s) left"
        cv2.putText(display, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow(window_name, display)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            print("Aborted.")
            sys.exit(0)

    return collector.points[:4]


# ── drawing helper ───────────────────────────────────────────────────────────

def _draw_quad_alpha(frame, pts, color):
    """Fill an arbitrary quadrilateral with color at ALPHA transparency."""
    overlay = frame.copy()
    hull = cv2.convexHull(np.array(pts, dtype=np.int32))
    cv2.fillConvexPoly(overlay, hull, color)
    # Must NOT use frame as dst — in-place with src2=frame causes corrupt reads
    blended = cv2.addWeighted(overlay, ALPHA, frame, 1 - ALPHA, 0)
    frame[:] = blended


def draw_all_squares(frame, blue_pts, red_pts):
    _draw_quad_alpha(frame, blue_pts, BLUE_COLOR)
    _draw_quad_alpha(frame, red_pts, RED_COLOR)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 run_video_yolo.py <video> <model.pt>")
        sys.exit(1)

    video_path = sys.argv[1]
    model_path = sys.argv[2]

    print(f"Loading YOLO model: {model_path}")
    model = YOLO(model_path)
    tracker = DeepSort(max_age=30, n_init=3, nn_budget=100)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        sys.exit(1)

    ret, first_frame = cap.read()
    if not ret:
        print("ERROR: Cannot read first frame.")
        sys.exit(1)

    # ── Phase 1: collect square corners ──
    WIN = "Setup — click squares"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    blue_pts = collect_quad(WIN, first_frame, "BLUE square: click 4 corners", BLUE_COLOR)
    red_pts  = collect_quad(WIN, first_frame, "RED square: click 4 corners",  RED_COLOR,
                            existing_quads=[(blue_pts, BLUE_COLOR)])

    cv2.destroyWindow(WIN)

    # ── Phase 2: play + save video ──
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    delay  = max(1, int(1000 / fps))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind

    out_path = video_path.rsplit(".", 1)[0] + "_annotated.mp4"
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    WIN2 = "Video + YOLO + Squares"
    cv2.namedWindow(WIN2, cv2.WINDOW_NORMAL)

    print(f"\nPlaying. Saving annotated video to: {out_path}")
    print("Press 'q' to quit, Space to pause.")
    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break  # end of video — stop and save

            # YOLO detection only — DeepSORT handles ID assignment
            results = model(frame, verbose=False)
            detections = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls  = int(box.cls[0])
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

            tracks = tracker.update_tracks(detections, frame=frame)

            annotated = frame.copy()
            for track in tracks:
                if not track.is_confirmed():
                    continue
                tid = track.track_id
                l, t, r, b = (int(v) for v in track.to_ltrb())
                cls_id = track.get_det_class()
                label = f"ID {tid}"
                if cls_id is not None and results[0].names:
                    label = f"{results[0].names[cls_id]} {tid}"
                cv2.rectangle(annotated, (l, t), (r, b), (0, 255, 0), 2)
                cv2.putText(annotated, label, (l, t - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            draw_all_squares(annotated, blue_pts, red_pts)
            writer.write(annotated)
            cv2.imshow(WIN2, annotated)

        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
