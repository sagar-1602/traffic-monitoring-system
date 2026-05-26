import numpy as np
from filterpy.kalman import KalmanFilter


class VehicleTracker:
    """
    Tracks a single vehicle using a Kalman Filter.
    Predicts position between detections for smooth, persistent tracking.
    """

    _id_counter = 0

    def __init__(self, bbox, class_name):
        VehicleTracker._id_counter += 1
        self.id = VehicleTracker._id_counter
        self.class_name = class_name
        self.missed_frames = 0
        self.trajectory = []

        # Kalman Filter: state = [x, y, vx, vy, w, h]
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
        self.trajectory.append((int(x + w / 2), int(y + h / 2)))
        if len(self.trajectory) > 30:
            self.trajectory.pop(0)
        self.missed_frames = 0

    def get_bbox(self):
        cx, cy, _, _, w, h = self.kf.x
        return (int(cx - w / 2), int(cy - h / 2), int(w), int(h))


class MultiObjectTracker:
    """
    Manages multiple VehicleTracker instances.
    Matches new detections to existing tracks using IoU.
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
        # Predict next position for all existing tracks
        for t in self.trackers:
            t.predict()
            t.missed_frames += 1

        matched_tracker_ids = set()

        for det in detections:
            bbox = det["bbox"]
            best_iou, best_tracker = 0, None

            for t in self.trackers:
                iou = self._iou(bbox, t.get_bbox())
                if iou > best_iou:
                    best_iou, best_tracker = iou, t

            if best_iou >= self.iou_threshold and best_tracker:
                best_tracker.update(bbox)
                matched_tracker_ids.add(best_tracker.id)
            else:
                new_tracker = VehicleTracker(bbox, det["class_name"])
                new_tracker.update(bbox)
                self.trackers.append(new_tracker)

        # Remove tracks that have been missing too long
        self.trackers = [
            t for t in self.trackers
            if t.id in matched_tracker_ids or t.missed_frames <= self.max_missed
        ]

        return self.trackers