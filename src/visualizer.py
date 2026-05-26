import cv2
import numpy as np

CLASS_COLORS = {
    "car":        (0, 255, 100),
    "bus":        (0, 100, 255),
    "truck":      (255, 100, 0),
    "motorbike":  (255, 200, 0),
    "bicycle":    (200, 0, 255),
}
DEFAULT_COLOR = (200, 200, 200)

CONGESTION_COLORS = {
    "Low":      (0, 200, 0),
    "Moderate": (0, 200, 255),
    "High":     (0, 100, 255),
    "Severe":   (0, 0, 255),
}


def draw_trackers(frame, trackers):
    for t in trackers:
        x, y, w, h = t.get_bbox()
        color = CLASS_COLORS.get(t.class_name, DEFAULT_COLOR)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        label = f"ID:{t.id} {t.class_name}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x, y - lh - 8), (x + lw + 4, y), color, -1)
        cv2.putText(frame, label, (x + 2, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        if len(t.trajectory) > 1:
            pts = np.array(t.trajectory, dtype=np.int32)
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                c = tuple(int(v * alpha) for v in color)
                cv2.line(frame, tuple(pts[i - 1]), tuple(pts[i]), c, 2)

    return frame


def draw_stats(frame, stats):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    panel_h = 130
    cv2.rectangle(overlay, (0, 0), (280, panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    congestion = stats["congestion_level"]
    c_color = CONGESTION_COLORS.get(congestion, (200, 200, 200))

    lines = [
        ("Traffic Monitoring System",        (255, 255, 255), 0.55, True),
        (f"Vehicles now: {stats['current_vehicles']}",  (200, 255, 200), 0.5,  False),
        (f"Average:      {stats['average_vehicles']}",  (180, 220, 180), 0.45, False),
        (f"Peak:         {stats['peak_vehicles']}",     (180, 220, 180), 0.45, False),
        (f"Congestion:   {congestion}",                 c_color,         0.5,  False),
        (f"Elapsed:      {stats['elapsed_seconds']}s",  (180, 180, 180), 0.4,  False),
    ]

    y = 18
    for text, color, scale, bold in lines:
        thickness = 2 if bold else 1
        cv2.putText(frame, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
        y += 20

    return frame