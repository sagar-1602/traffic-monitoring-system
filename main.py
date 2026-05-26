"""
Traffic Monitoring System
=========================
Real-time vehicle detection, Kalman Filter path tracking,
congestion analysis, and optional night vision mode.
 
Usage:
    python main.py                             # webcam
    python main.py --source video.mp4          # video file
    python main.py --source video.mp4 --save   # save output
    python main.py --source video.mp4 --night  # night vision mode
 
Keyboard shortcuts while running:
    Q  — quit
    N  — toggle night vision on/off live
    S  — take a screenshot (saved to output/)
"""
 
import argparse
import cv2
import time
import os
import csv
from dotenv import load_dotenv
 
from src.detector   import VehicleDetector
from src.tracker    import MultiObjectTracker
from src.analyzer   import TrafficAnalyzer
from src.visualizer import (draw_trackers, draw_stats,
                             draw_legend, apply_night_vision)
 
load_dotenv()
 
 
def parse_args():
    parser = argparse.ArgumentParser(description="Traffic Monitoring System")
    parser.add_argument("--source",     default="0",
                        help="0=webcam or path to video file")
    parser.add_argument("--confidence", default=0.5,  type=float,
                        help="YOLO detection confidence (0-1)")
    parser.add_argument("--save",       action="store_true",
                        help="Save annotated video to output/")
    parser.add_argument("--night",      action="store_true",
                        help="Start with night vision mode enabled")
    parser.add_argument("--width",      default=1280, type=int,
                        help="Display window width (default 1280)")
    parser.add_argument("--height",     default=720,  type=int,
                        help="Display window height (default 720)")
    parser.add_argument("--models-dir", default="models")
    return parser.parse_args()
 
 
def main():
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
 
    detector = VehicleDetector(models_dir=args.models_dir,
                               confidence_threshold=args.confidence)
    tracker  = MultiObjectTracker(max_missed=10)
    analyzer = TrafficAnalyzer()
 
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        return
 
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
 
    writer = None
    if args.save:
        os.makedirs("output", exist_ok=True)
        out_path = "output/annotated_output.mp4"
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        writer   = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        print(f"[INFO] Saving output to {out_path}")
 
    night_mode   = args.night
    frame_count  = 0
    screenshot_n = 0
 
    print("[INFO] Starting. Shortcuts: Q=quit  N=toggle night vision  S=screenshot")
 
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of stream.")
            break
 
        frame_count += 1
 
        # ── Night vision enhancement (applied BEFORE detection) ──────
        process_frame = apply_night_vision(frame) if night_mode else frame
 
        # ── Detection every other frame for speed ────────────────────
        if frame_count % 2 == 0:
            detections = detector.detect(process_frame)
        else:
            detections = []
 
        active_trackers = tracker.update(detections)
        stats = analyzer.get_stats()
        analyzer.update(active_trackers)
 
        # ── Draw on the display frame ────────────────────────────────
        display = process_frame.copy()
        display = draw_trackers(display, active_trackers)
        display = draw_stats(display, stats, night_mode=night_mode)
        display = draw_legend(display)
 
        # Save full-res frame to video
        if writer:
            writer.write(display)
 
        # Resize for display window
        display = cv2.resize(display, (args.width, args.height))
        cv2.imshow("Traffic Monitoring System", display)
 
        # ── Keyboard controls ────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
 
        if key == ord("q"):
            print("[INFO] Quit.")
            break
 
        elif key == ord("n"):
            night_mode = not night_mode
            status = "ON" if night_mode else "OFF"
            print(f"[INFO] Night vision {status}")
 
        elif key == ord("s"):
            os.makedirs("output", exist_ok=True)
            screenshot_n += 1
            path = f"output/screenshot_{screenshot_n:03d}.jpg"
            cv2.imwrite(path, display)
            print(f"[INFO] Screenshot saved: {path}")
 
    # ── Save stats CSV ───────────────────────────────────────────────
    os.makedirs("output", exist_ok=True)
    if analyzer.time_series:
        csv_path = "output/traffic_stats.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=analyzer.time_series[0].keys())
            w.writeheader()
            w.writerows(analyzer.time_series)
        print(f"[INFO] Stats saved to {csv_path}")
 
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")
 
 
if __name__ == "__main__":
    main()