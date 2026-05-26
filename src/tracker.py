import numpy as np
from filterpy.kalman import KalmanFilter
import math
 
 
class VehicleTracker:
    """
    Tracks a single vehicle using a Kalman Filter.
    Records full trajectory history and estimates speed.
    """
 
    _id_counter = 0
 
    def __init__(self, bbox, class_name):
        VehicleTracker._id_counter += 1
        self.id = VehicleTracker._id_counter
        self.class_name = class_name
        self.missed_frames = 0
 
        # Full trajectory: list of (cx, cy) centre points
        self.trajectory = []
 
        # Kalman predicted future path points
        self.predicted_path = []
 
        # Speed in pixels/frame (estimated from last N frames)
        self.speed_px = 0.0
 
        # Kalman Filter state = [cx, cy, vx, vy, w, h]
        self.kf = KalmanFilter(dim_x=6, dim_z=4)
        x, y, w, h = bbox
 
        self.kf.x = np.array([x + w / 2, y + h / 2, 0, 0, w, h], dtype=float)
 
        self.kf.F = np.array([
            [1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=float)
 
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=float)
 
        self.kf.R *= 10
        self.kf.P *= 100
        self.kf.Q[2:, 2:] *= 0.01
 
    def predict(self):
        self.kf.predict()
        cx, cy, _, _, w, h = self.kf.x
        return (int(cx - w / 2), int(cy - h / 2), int(w), int(h))
 
    def update(self, bbox):
        x, y, w, h = bbox
        measurement = np.array([x + w / 2, y + h / 2, w, h], dtype=float)
        self.kf.update(measurement)
 
        cx = int(x + w / 2)
        cy = int(y + h / 2)
        self.trajectory.append((cx, cy))
 
        # Keep last 60 points for the trail (longer = more visible path)
        if len(self.trajectory) > 60:
            self.trajectory.pop(0)
 
        # Estimate speed from last 5 positions
        if len(self.trajectory) >= 5:
            p1 = self.trajectory[-5]
            p2 = self.trajectory[-1]
            self.speed_px = round(
                math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) / 5, 1
            )
 
        # Build predicted future path (next 20 frames using current velocity)
        self._update_predicted_path()
 
        self.missed_frames = 0
 
    def _update_predicted_path(self):
        """
        Uses Kalman velocity (vx, vy) to project where the vehicle
        will be over the next 20 frames — this is the core Kalman output.
        """
        cx, cy, vx, vy, w, h = self.kf.x
        self.predicted_path = []
        px, py = float(cx), float(cy)
        for _ in range(20):
            px += vx
            py += vy
            self.predicted_path.append((int(px), int(py)))
 
    def get_bbox(self):
        cx, cy, _, _, w, h = self.kf.x
        return (int(cx - w / 2), int(cy - h / 2), int(w), int(h))
 
    def get_direction(self):
        """Returns compass direction string based on velocity vector."""
        _, _, vx, vy, _, _ = self.kf.x
        if abs(vx) < 0.5 and abs(vy) < 0.5:
            return "Stationary"
        angle = math.degrees(math.atan2(-vy, vx))  # negative vy because y is flipped
        dirs = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
        idx = int((angle + 202.5) / 45) % 8
        return dirs[idx]
 
 
class MultiObjectTracker:
    """
    Manages multiple VehicleTracker instances.
    Matches detections to tracks using IoU.
    """
 
    def __init__(self, max_missed=10, iou_threshold=0.2):
        self.trackers = []
        self.max_missed = max_missed
        self.iou_threshold = iou_threshold
 
    @staticmethod
    def _iou(boxA, boxB):
        ax, ay, aw, ah = boxA
        bx, by, bw, bh = boxB
        ix1 = max(ax, bx)
        iy1 = max(ay, by)
        ix2 = min(ax + aw, bx + bw)
        iy2 = min(ay + ah, by + bh)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0
 
    def update(self, detections):
        for t in self.trackers:
            t.predict()
            t.missed_frames += 1
 
        matched_ids = set()
 
        for det in detections:
            bbox = det["bbox"]
            best_iou, best_tracker = 0, None
 
            for t in self.trackers:
                iou = self._iou(bbox, t.get_bbox())
                if iou > best_iou:
                    best_iou, best_tracker = iou, t
 
            if best_iou >= self.iou_threshold and best_tracker:
                best_tracker.update(bbox)
                matched_ids.add(best_tracker.id)
            else:
                new_t = VehicleTracker(bbox, det["class_name"])
                new_t.update(bbox)
                self.trackers.append(new_t)
 
        self.trackers = [
            t for t in self.trackers
            if t.id in matched_ids or t.missed_frames <= self.max_missed
        ]
 
        return self.trackers