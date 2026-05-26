# 🚦 Intelligent Traffic Monitoring System

A real-time traffic monitoring and congestion analysis system using **YOLOv3**, **Kalman Filter tracking**, and **OpenCV**. Detects and tracks vehicles across video frames, analyzes congestion patterns, and generates exportable traffic reports.

---

## 🎯 Features

- **Real-time vehicle detection** using YOLOv3 with multi-class support (cars, buses, trucks, motorcycles, bicycles)
- **Kalman Filter-based tracking** for persistent object identity across frames — handles occlusion and brief disappearances
- **Traffic congestion analysis** with severity levels: Low / Moderate / High / Severe
- **Trajectory visualization** — color-coded movement paths per vehicle
- **Live statistics overlay** — current count, average, peak, elapsed time
- **CSV export** of time-series traffic data for further analysis
- **Webcam and video file** support
- **Optional video output** saving with full annotations

---

## 🛠️ Tech Stack

| Component        | Technology                              |
| ---------------- | --------------------------------------- |
| Object Detection | YOLOv3 (Darknet) via OpenCV DNN         |
| Object Tracking  | Kalman Filter (FilterPy) + IoU matching |
| Video Processing | OpenCV                                  |
| Data Analysis    | NumPy, Pandas                           |
| Configuration    | python-dotenv                           |

---

## 📁 Project Structure

traffic-monitoring-system/
├── src/
│ ├── detector.py # YOLOv3 vehicle detection
│ ├── tracker.py # Kalman Filter multi-object tracking
│ ├── analyzer.py # Congestion analysis and statistics
│ └── visualizer.py # Bounding boxes, trajectories, HUD overlay
├── models/
│ ├── yolov3.weights # Download separately (see below)
│ ├── yolov3.cfg
│ └── coco.names
├── output/ # Annotated video and CSV stats saved here
├── main.py # Entry point
├── requirements.txt
├── .env # Environment configuration
└── README.md

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

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download YOLO model files

Place these files inside the `models/` folder (see `models/README.md`):

- [yolov3.weights](https://pjreddie.com/media/files/yolov3.weights) (~236 MB)
- [yolov3.cfg](https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg)
- [coco.names](https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names)

---

## 🚀 Usage

### Run with webcam

```bash
python main.py
```

### Run with a video file

```bash
python main.py --source path/to/traffic_video.mp4
```

### Run and save annotated output

```bash
python main.py --source path/to/video.mp4 --save
```

### Adjust detection confidence

```bash
python main.py --source video.mp4 --confidence 0.6
```

Press **Q** to quit. Stats are automatically saved to `output/traffic_stats.csv`.

---

## 📊 Output

- **Annotated video** (if `--save` used): `output/annotated_output.mp4`
- **Traffic stats CSV**: `output/traffic_stats.csv` — time-series of vehicle count and congestion level

---

## 🔧 Configuration

Edit `.env` to adjust system parameters:

```env
DEFAULT_CONFIDENCE=0.5        # Detection confidence threshold
DEFAULT_NMS_THRESHOLD=0.4     # Non-max suppression threshold
MAX_MISSED_FRAMES=10          # Frames before a track is dropped
```

---

## 🧠 How It Works

1. **Detection**: Each frame is passed through YOLOv3, which outputs bounding boxes for vehicles above the confidence threshold
2. **Tracking**: IoU-based matching assigns each detection to an existing track or creates a new one. Kalman Filter predicts position between frames
3. **Analysis**: Rolling window statistics compute average density, detect congestion severity, and build time-series data
4. **Visualization**: Annotated frames show bounding boxes, vehicle IDs, trajectory trails, and a live HUD

---

## 👤 Author

**Sagar Kumar Singh**  
Computer Science Engineering, Chandigarh University  
[LinkedIn](https://linkedin.com/in/sagar-kumar-singh-) | [LeetCode](https://leetcode.com/u/sagar1622005)

---

## 📄 License

MIT License — free to use and modify.
