# 🚦 Intelligent Traffic Monitoring System

A real-time traffic monitoring and congestion analysis system built with **YOLOv3**, **Kalman Filter tracking**, and **OpenCV**. Detects and tracks vehicles across video frames, predicts their future paths, analyzes congestion patterns, supports night vision enhancement, and generates exportable traffic reports.

> Built as part of Computer Science Engineering portfolio — Chandigarh University

---

## 📸 Demo

|                                     Normal Mode                                      |                   Night Vision Mode                    |
| :----------------------------------------------------------------------------------: | :----------------------------------------------------: |
| Vehicles detected with bounding boxes, trajectory trails, and Kalman predicted paths | CLAHE-enhanced green-tinted view for low-light footage |

---

## ✨ Features

- **Real-time vehicle detection** using YOLOv3 — supports cars, buses, trucks, motorcycles, and bicycles
- **Kalman Filter tracking** — each vehicle gets a unique ID and is tracked persistently across frames, even through brief occlusions
- **Trajectory trail** — solid fading line showing exactly where each vehicle has been (last 60 frames)
- **Kalman predicted path** — dashed line with directional arrow showing where each vehicle is predicted to go over the next 20 frames
- **Speed estimation** — pixels/frame speed calculated from last 5 positions
- **Direction detection** — compass direction (N, NE, E, SE, S, SW, W, NW, Stationary) per vehicle
- **Traffic congestion analysis** — severity levels: Low / Moderate / High / Severe with colour-coded indicator
- **Night vision mode** — CLAHE adaptive contrast enhancement + sharpening + green tint for low-light footage
- **Live stats panel** — large, readable HUD showing vehicle count, average, peak, congestion level, elapsed time
- **Live keyboard controls** — toggle night vision, take screenshots, quit — all without restarting
- **CSV export** — time-series traffic data saved automatically on exit
- **Webcam and video file** support
- **Save annotated video** output option

---

## 🛠️ Tech Stack

| Component        | Technology                                               |
| ---------------- | -------------------------------------------------------- |
| Object Detection | YOLOv3 (Darknet) via OpenCV DNN module                   |
| Object Tracking  | Kalman Filter (FilterPy) + IoU-based matching            |
| Path Prediction  | Kalman velocity state (vx, vy) projected 20 frames ahead |
| Night Vision     | CLAHE + Sharpening kernel + Green channel mapping        |
| Video Processing | OpenCV                                                   |
| Data Export      | Python CSV module                                        |
| Configuration    | python-dotenv                                            |

---

## 📁 Project Structure

```
traffic-monitoring-system/
├── src/
│   ├── __init__.py
│   ├── detector.py       # YOLOv3 vehicle detection
│   ├── tracker.py        # Kalman Filter multi-object tracking + path prediction
│   ├── analyzer.py       # Congestion analysis and time-series statistics
│   └── visualizer.py     # Bounding boxes, trails, predicted paths, HUD, night vision
├── models/
│   ├── yolov3.weights    # Download separately — see instructions below
│   ├── yolov3.cfg        # YOLO network configuration
│   └── coco.names        # 80 class names (cars, buses, trucks etc.)
├── output/               # Annotated video, screenshots, and CSV stats saved here
├── main.py               # Entry point
├── requirements.txt      # Python dependencies
├── .env                  # Environment configuration
├── .gitignore
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/traffic-monitoring-system.git
cd traffic-monitoring-system
```

### 2. Create and activate virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download YOLO model files

Download all three files and place them inside the `models/` folder:

| File             | Size    | Link                                                                      |
| ---------------- | ------- | ------------------------------------------------------------------------- |
| `yolov3.weights` | ~236 MB | https://pjreddie.com/media/files/yolov3.weights                           |
| `yolov3.cfg`     | ~8 KB   | https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg  |
| `coco.names`     | ~1 KB   | https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names |

> These files are excluded from Git via `.gitignore` due to their size.

**Windows users** — if the files download as `yolov3.cfg.txt` or `coco.names.txt`, rename them:

```bash
rename models\coco.names.txt as models\coco.names
rename models\yolov3.cfg.txt as models\yolov3.cfg
```

---

## 🚀 Usage

### Run with webcam

```bash
python main.py
```

### Run with a video file

```bash
python main.py --source path\to\traffic_video.mp4
```

### Run and save annotated output video

```bash
python main.py --source traffic_video.mp4 --save
```

### Start directly in night vision mode

```bash
python main.py --source traffic_video.mp4 --night
```

### Adjust display window size

```bash
python main.py --source traffic_video.mp4 --width 1024 --height 576
```

### Adjust detection confidence threshold

```bash
python main.py --source traffic_video.mp4 --confidence 0.6
```

---

## ⌨️ Keyboard Shortcuts

| Key | Action                            |
| --- | --------------------------------- |
| `N` | Toggle night vision on / off live |
| `S` | Save a screenshot to `output/`    |
| `Q` | Quit the application              |

---

## 📊 Output Files

All outputs are saved to the `output/` folder automatically:

| File                   | Description                                                     |
| ---------------------- | --------------------------------------------------------------- |
| `annotated_output.mp4` | Full annotated video (when `--save` is used)                    |
| `traffic_stats.csv`    | Time-series data: elapsed time, vehicle count, congestion level |
| `screenshot_001.jpg`   | Screenshots taken with the `S` key                              |

---

## 🔧 Configuration

Edit `.env` to adjust system parameters without touching the code:

```env
DEFAULT_CONFIDENCE=0.5        # Detection confidence threshold (0.0 – 1.0)
DEFAULT_NMS_THRESHOLD=0.4     # Non-max suppression threshold
MAX_MISSED_FRAMES=10          # Frames before a lost track is dropped
```

---

## 🧠 How It Works

### Detection

Each frame is passed through YOLOv3, which outputs bounding boxes for detected vehicles above the confidence threshold. Only vehicle classes are kept: car, bus, truck, motorbike, bicycle.

### Tracking

New detections are matched to existing tracks using **IoU (Intersection over Union)**. A matched track is updated; an unmatched detection spawns a new track. Tracks missing for more than `MAX_MISSED_FRAMES` frames are dropped.

### Kalman Filter — Path Prediction

Each track runs its own **6-state Kalman Filter**: `[cx, cy, vx, vy, w, h]`. The velocity components `(vx, vy)` are estimated automatically from position updates. On every frame, the filter:

1. **Predicts** — extrapolates position using current velocity
2. **Updates** — corrects the prediction with the new detection measurement

The predicted path (dashed line) is generated by projecting `(cx, cy)` forward by `(vx, vy)` for 20 steps — this is the core output of Kalman filtering beyond just tracking.

### Congestion Analysis

A rolling window of vehicle counts determines congestion severity:

| Level    | Vehicle Count |
| -------- | ------------- |
| Low      | 0 – 4         |
| Moderate | 5 – 11        |
| High     | 12 – 19       |
| Severe   | 20+           |

### Night Vision

Three OpenCV operations applied in sequence:

1. **Grayscale conversion** — removes colour noise, works on luminance only
2. **CLAHE** (Adaptive Histogram Equalisation) — brightens dark zones without overexposing bright areas
3. **Sharpening kernel** — enhances edges so vehicle outlines are crisp in low light
4. **Green channel mapping** — classic night-vision green tint for visual clarity

---

## 🖥️ Visualization Guide

| Visual Element              | Meaning                                               |
| --------------------------- | ----------------------------------------------------- |
| Coloured bounding box       | Detected vehicle (colour = class)                     |
| `ID:3 car 4.2px/f NE` label | Track ID, vehicle class, speed, direction             |
| **Solid fading trail**      | Historical path — where the vehicle has been          |
| **Dashed line ahead**       | Kalman predicted future path (next 20 frames)         |
| **Arrow at end of dash**    | Predicted direction of travel                         |
| Top-left HUD panel          | Live stats: count, average, peak, congestion, elapsed |
| Coloured top bar on HUD     | Congestion severity at a glance                       |
| Bottom-left legend          | Explains trail vs predicted path                      |

---

## 🎨 Vehicle Class Colours

| Class     | Colour |
| --------- | ------ |
| Car       | Green  |
| Bus       | Blue   |
| Truck     | Orange |
| Motorbike | Yellow |
| Bicycle   | Purple |

---

## 👤 Author

**Sagar Kumar Singh**  
B.E. Computer Science Engineering — Chandigarh University (CGPA: 8.03/10)  
📧 sagar1622005@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/sagar-kumar-singh-)  
💻 [LeetCode](https://leetcode.com/u/sagar1622005)

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.
