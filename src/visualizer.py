import cv2
import numpy as np
import math
 
CLASS_COLORS = {
    "car":        (0, 255, 100),
    "bus":        (0, 100, 255),
    "truck":      (255, 100, 0),
    "motorbike":  (255, 200, 0),
    "bicycle":    (200, 0, 255),
}
DEFAULT_COLOR = (200, 200, 200)
 
CONGESTION_COLORS = {
    "Low":      (0, 220, 0),
    "Moderate": (0, 200, 255),
    "High":     (0, 100, 255),
    "Severe":   (0, 0, 255),
}
 
 
# ─────────────────────────────────────────────
#  NIGHT VISION ENHANCEMENT
# ─────────────────────────────────────────────
def apply_night_vision(frame):
    """
    Simulates night-vision / low-light enhancement:
    1. Converts to grayscale
    2. Applies CLAHE (adaptive histogram equalisation) to boost contrast
    3. Applies a green tint (classic night-vision look)
    4. Sharpens edges so vehicle details are clearer
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    # CLAHE — boosts contrast locally so dark areas become visible
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
 
    # Sharpen
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
 
    # Green tint: put enhanced luma into green channel only
    night = np.zeros_like(frame)
    night[:, :, 1] = sharpened   # green channel
    night[:, :, 0] = (sharpened * 0.15).astype(np.uint8)  # tiny blue
    return night
 
 
# ─────────────────────────────────────────────
#  VEHICLE BOXES + TRAILS + PREDICTED PATH
# ─────────────────────────────────────────────
def draw_trackers(frame, trackers):
    h_frame, w_frame = frame.shape[:2]
 
    for t in trackers:
        x, y, w, h = t.get_bbox()
        color = CLASS_COLORS.get(t.class_name, DEFAULT_COLOR)
 
        # Bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
 
        # Label background + text
        direction = t.get_direction()
        label = f"ID:{t.id} {t.class_name} {t.speed_px}px/f {direction}"
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.52
        thickness  = 1
        (lw, lh), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        pad = 4
        # Dark background rect for label
        cv2.rectangle(frame,
                      (x, y - lh - baseline - pad * 2),
                      (x + lw + pad * 2, y),
                      (0, 0, 0), -1)
        cv2.rectangle(frame,
                      (x, y - lh - baseline - pad * 2),
                      (x + lw + pad * 2, y),
                      color, 1)
        cv2.putText(frame, label,
                    (x + pad, y - baseline - pad),
                    font, font_scale, color, thickness, cv2.LINE_AA)
 
        # Historical trail (solid, fading)
        pts = t.trajectory
        if len(pts) > 1:
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                trail_color = tuple(int(c * alpha) for c in color)
                cv2.line(frame, pts[i - 1], pts[i], trail_color,
                         max(1, int(alpha * 3)))
            cv2.circle(frame, pts[0], 3, color, -1)
 
        # Kalman predicted path (dashed + arrow)
        pred = t.predicted_path
        if len(pred) > 1:
            for i in range(0, len(pred) - 1, 2):
                p1, p2 = pred[i], pred[i + 1]
                if (0 <= p1[0] < w_frame and 0 <= p1[1] < h_frame and
                        0 <= p2[0] < w_frame and 0 <= p2[1] < h_frame):
                    cv2.line(frame, p1, p2, color, 1)
            if len(pred) >= 3:
                ep = pred[-1]
                if 0 <= ep[0] < w_frame and 0 <= ep[1] < h_frame:
                    _draw_arrow(frame, pred[-3], ep, color)
 
        # Centre dot
        cx, cy = x + w // 2, y + h // 2
        cv2.circle(frame, (cx, cy), 4, color, -1)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 0), 1)
 
    return frame
 
 
def _draw_arrow(frame, pt1, pt2, color, tip=12):
    dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1:
        return
    dx /= length; dy /= length
    left  = (int(pt2[0] - tip * (dx - dy * 0.5)),
             int(pt2[1] - tip * (dy + dx * 0.5)))
    right = (int(pt2[0] - tip * (dx + dy * 0.5)),
             int(pt2[1] - tip * (dy - dx * 0.5)))
    cv2.line(frame, pt2, left,  color, 2)
    cv2.line(frame, pt2, right, color, 2)
 
 
# ─────────────────────────────────────────────
#  STATS PANEL  (large, clear, readable)
# ─────────────────────────────────────────────
def draw_stats(frame, stats, night_mode=False):
    h_frame, w_frame = frame.shape[:2]
 
    # Panel dimensions — scale with frame size
    panel_w = max(320, w_frame // 4)
    panel_h = 175
    margin  = 12
 
    # Semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
 
    # Coloured top bar
    congestion = stats["congestion_level"]
    bar_color  = CONGESTION_COLORS.get(congestion, (200, 200, 200))
    cv2.rectangle(frame, (0, 0), (panel_w, 6), bar_color, -1)
 
    # Border
    cv2.rectangle(frame, (0, 0), (panel_w, panel_h), (80, 80, 80), 1)
 
    font  = cv2.FONT_HERSHEY_DUPLEX
    fontS = cv2.FONT_HERSHEY_SIMPLEX
 
    # Title
    cv2.putText(frame, "Traffic Monitoring System",
                (margin, 30), font, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
 
    # Divider
    cv2.line(frame, (margin, 38), (panel_w - margin, 38), (80, 80, 80), 1)
 
    # Stats rows
    rows = [
        ("Vehicles now", str(stats["current_vehicles"]),  (120, 255, 150)),
        ("Average",      str(stats["average_vehicles"]),  (160, 220, 160)),
        ("Peak",         str(stats["peak_vehicles"]),     (160, 220, 160)),
        ("Congestion",   congestion,                       bar_color),
        ("Elapsed",      f"{stats['elapsed_seconds']}s",  (160, 160, 160)),
    ]
 
    y = 60
    for label, value, val_color in rows:
        # Label (left)
        cv2.putText(frame, label + ":",
                    (margin, y), fontS, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
        # Value (right-aligned inside panel)
        (vw, _), _ = cv2.getTextSize(value, fontS, 0.58, 2)
        cv2.putText(frame, value,
                    (panel_w - vw - margin, y),
                    fontS, 0.58, val_color, 2, cv2.LINE_AA)
        y += 24
 
    # Night mode indicator
    if night_mode:
        cv2.putText(frame, "[ NIGHT VISION ON ]",
                    (margin, panel_h - 10),
                    fontS, 0.42, (0, 255, 120), 1, cv2.LINE_AA)
 
    return frame
 
 
# ─────────────────────────────────────────────
#  LEGEND
# ─────────────────────────────────────────────
def draw_legend(frame):
    h, w = frame.shape[:2]
    lx, ly = 10, h - 80
 
    overlay = frame.copy()
    cv2.rectangle(overlay, (lx - 4, ly - 18),
                  (lx + 260, ly + 58), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
 
    font = cv2.FONT_HERSHEY_SIMPLEX
 
    # Solid line = trail
    cv2.line(frame, (lx, ly), (lx + 38, ly), (0, 255, 100), 3)
    cv2.putText(frame, "Past trajectory trail",
                (lx + 46, ly + 4), font, 0.4, (200, 255, 200), 1, cv2.LINE_AA)
 
    # Dashed = predicted
    for i in range(0, 34, 8):
        cv2.line(frame, (lx + i, ly + 22), (lx + i + 5, ly + 22),
                 (0, 255, 100), 1)
    cv2.putText(frame, "Kalman predicted path",
                (lx + 46, ly + 26), font, 0.4, (200, 255, 200), 1, cv2.LINE_AA)
 
    cv2.putText(frame, "Arrow = predicted direction",
                (lx + 46, ly + 48), font, 0.4, (160, 160, 160), 1, cv2.LINE_AA)
 
    return frame