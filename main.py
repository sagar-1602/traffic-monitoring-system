"""
Traffic Monitoring System
=========================
Real-time vehicle detection, tracking, and congestion analysis
using YOLOv3 + Kalman Filter + OpenCV.

Usage:
    python main.py                        # uses webcam
    python main.py --source video.mp4     # uses a video file
    python main.py --source video.mp4 --save
"""

import argparse
import cv2
import time
import os
import csv
from dotenv import load_dotenv

from src.detector  import VehicleDetector
from src.tracker   import MultiObjectTracker
from src.analyzer  import TrafficAnalyzer
from src.visualizer import draw_trackers, draw_stats

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="Traffic Monitoring System")
    parser.add_argument("--source",     default="0",    help="Video source: 0=webcam, or path to video file")
    parser.add_argument("--confidence", default=0.5,    type=float, help="YOLO detection confidence (0–1)")
    parser.add_argument("--save",       action="store_true",        help="Save annotated video to output/")
    parser.add_argument("--models-dir", default="models",           help="Path to YOLO model files")
    return parser.parse_args()


def main():
    args = parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    detector = VehicleDetector(
        models_dir=args.models_dir,
        confidence_threshold=args.confidence,
    )
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
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        print(f"[INFO] Saving output to {out_path}")

    print("[INFO] Starting traffic monitoring. Press Q to quit.")
    frame_count = 0
    start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video stream.")
            break

        frame_count += 1

        # Run detection every other frame for speed
        if frame_count % 2 == 0:
            detections = detector.detect(frame)
        else:
            detections = []

        active_trackers = tracker.update(detections)
        stats = analyzer.get_stats()
        analyzer.update(active_trackers)

        frame = draw_trackers(frame, active_trackers)
        frame = draw_stats(frame, stats)

        if writer:
            writer.write(frame)

        cv2.imshow("Traffic Monitoring System", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Quit signal received.")
            break

    # Save stats CSV
    os.makedirs("output", exist_ok=True)
    csv_path = "output/traffic_stats.csv"
    with open(csv_path, "w", newline="") as f:
        writer_csv = csv.DictWriter(f, fieldnames=analyzer.time_series[0].keys() if analyzer.time_series else [])
        if analyzer.time_series:
            writer_csv.writeheader()
            writer_csv.writerows(analyzer.time_series)
    print(f"[INFO] Stats saved to {csv_path}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()