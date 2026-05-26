from collections import defaultdict, deque
import time


class TrafficAnalyzer:
    """
    Analyzes traffic patterns: vehicle counts, congestion level,
    per-class statistics, and time-series data.
    """

    CONGESTION_THRESHOLDS = {
        "Low":      (0, 5),
        "Moderate": (5, 12),
        "High":     (12, 20),
        "Severe":   (20, float("inf")),
    }

    def __init__(self, history_seconds=30, fps=25):
        self.history_len = history_seconds * fps
        self.vehicle_counts   = deque(maxlen=self.history_len)
        self.class_counts     = defaultdict(int)
        self.total_vehicles   = 0
        self.frame_count      = 0
        self.start_time       = time.time()
        self.time_series      = []

    def update(self, trackers):
        self.frame_count += 1
        current_count = len(trackers)
        self.vehicle_counts.append(current_count)

        for t in trackers:
            self.class_counts[t.class_name] += 1

        if self.frame_count % 25 == 0:
            self.time_series.append({
                "elapsed_s": round(time.time() - self.start_time, 1),
                "count": current_count,
                "congestion": self.get_congestion_level(current_count),
            })

        return current_count

    @staticmethod
    def get_congestion_level(count):
        for level, (low, high) in TrafficAnalyzer.CONGESTION_THRESHOLDS.items():
            if low <= count < high:
                return level
        return "Unknown"

    def get_stats(self):
        counts = list(self.vehicle_counts)
        avg = sum(counts) / len(counts) if counts else 0
        current = counts[-1] if counts else 0
        elapsed = round(time.time() - self.start_time, 1)

        return {
            "current_vehicles": current,
            "average_vehicles": round(avg, 1),
            "peak_vehicles": max(counts) if counts else 0,
            "congestion_level": self.get_congestion_level(current),
            "class_breakdown": dict(self.class_counts),
            "total_frames_processed": self.frame_count,
            "elapsed_seconds": elapsed,
        }