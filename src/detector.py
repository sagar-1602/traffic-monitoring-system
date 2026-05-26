import cv2
import numpy as np
import os


class VehicleDetector:
    """
    Detects vehicles in video frames using YOLOv3.
    Supports cars, buses, trucks, motorcycles, and bicycles.
    """

    VEHICLE_CLASSES = {"car", "bus", "truck", "motorbike", "bicycle"}

    def __init__(self, models_dir="models", confidence_threshold=0.5, nms_threshold=0.4):
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        weights_path = os.path.join(models_dir, "yolov3.weights")
        config_path  = os.path.join(models_dir, "yolov3.cfg")
        names_path   = os.path.join(models_dir, "coco.names")

        if not all(os.path.exists(p) for p in [weights_path, config_path, names_path]):
            raise FileNotFoundError(
                "YOLO model files not found in 'models/' folder.\n"
                "Download yolov3.weights, yolov3.cfg, and coco.names and place them there."
            )

        print("[INFO] Loading YOLO model...")
        self.net = cv2.dnn.readNet(weights_path, config_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        with open(names_path, "r") as f:
            self.class_names = [line.strip() for line in f.readlines()]

        layer_names = self.net.getLayerNames()
        self.output_layers = [
            layer_names[i - 1]
            for i in self.net.getUnconnectedOutLayers().flatten()
        ]
        print("[INFO] YOLO model loaded successfully.")

    def detect(self, frame):
        """
        Run YOLO detection on a single frame.
        Returns list of dicts: {bbox, confidence, class_name}
        """
        height, width = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (416, 416), swapRB=True, crop=False
        )
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        boxes, confidences, class_ids = [], [], []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])

                if confidence < self.confidence_threshold:
                    continue

                class_name = self.class_names[class_id]
                if class_name not in self.VEHICLE_CLASSES:
                    continue

                cx, cy, w, h = (detection[:4] * [width, height, width, height]).astype(int)
                x = cx - w // 2
                y = cy - h // 2

                boxes.append([x, y, w, h])
                confidences.append(confidence)
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, self.confidence_threshold, self.nms_threshold
        )

        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                detections.append({
                    "bbox": (max(0, x), max(0, y), w, h),
                    "confidence": round(confidences[i], 3),
                    "class_name": self.class_names[class_ids[i]],
                })

        return detections