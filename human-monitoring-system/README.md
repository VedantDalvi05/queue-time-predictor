# Human Monitoring System: Detailed Technical Report

This repository contains a highly optimized, real-time human detection, tracking, and monitoring software architecture. Using deep learning techniques combined with object tracking algorithms, the system counts people, monitors queue lengths, and predicts average wait times, all while maintaining high performance across video sources.

## 📌 Executive Summary
The system operates by employing a **MobileNet SSD** (Single Shot MultiBox Detector) model for human detection. To optimize the processing pipeline, object detection is not executed on every single frame; rather, the system interweaves object *detection* with object *tracking* computationally efficiently using `dlib` correlation trackers. Finally, `scipy` driven **Centroid Tracking** handles unique ID assignment and logging.

### Primary Capabilities:
1. **Real-Time Human Tracking:** Detects, tracks, and uniquely identifies people via ID.
2. **Directional Flow Analysis:** Evaluates people passing an "entry" and "exit" threshold.
3. **Queue Wait-Time Analytics:** Calculates average service times and predicts wait times based on physical queue parameters (distance, spacing).
4. **Automated Features:** Built-in scheduler, execution timer, multi-threaded camera ingestion, daily CSV logging, and real-time email alerts.

---

## 📂 Project Architecture and Directory Structure
The architecture is modular, separating execution logic from libraries and configuration.

```
human-monitoring-system/
│
├── main.py                          # Primary execution script tying all features together
├── README.md                        # This detailed technical report
├── requirements.txt                 # Python dependencies
├── Log.csv                          # Auto-generated analytics log (when Config.Log = True)
│
├── lib/                             # Core Python Library Modules
│   ├── config.py                    # Global configuration variables (Alerts, threshold, flags)
│   ├── centroidtracker.py           # Core logic for centroid assignment via Euclidean distance
│   ├── trackableobject.py           # Data structure mapping objects to centroid histories
│   ├── thread.py                    # Multi-threading queue class for async frame IO
│   ├── mailer.py                    # SMTP wrapper for email notifications
│   └── creds.py                     # Secure storage for SMTP credentials (excluded by .gitignore)
│
├── mobilenet_ssd/                   # Neural Network files for object detection
│   ├── MobileNetSSD_deploy.prototxt # Caffe architecture definition file
│   └── MobileNetSSD_deploy.caffemodel# Caffe pre-trained weights
│
└── videos/                          # Directory for testing video clips
```

---

## ⚙️ Core Technical Components & Workflows

### 1. Object Detection (MobileNet SSD)
* **Script:** `main.py`
* **Details:** We use a Caffe implementation of MobileNet SSD. It processes a frame transformed into a blob natively using OpenCV's `dnn` module. The confidence threshold bounds predictions, filtering out weak detections (default: `0.25`). While SSD can detect 20 classes (PASCAL VOC), this software explicitly filters bounding boxes to strictly target the `person` class.

### 2. Framerate Optimization via Correlation Trackers
* **Library:** `dlib.correlation_tracker()`
* **Details:** Neural network forward passes are computationally expensive. The system runs detection every `N` frames (configurable via `--skip-frames`, default: 30). For the intermediate 29 frames, the system leverages `dlib` correlation trackers, which are vastly faster, allowing the software to maintain high FPS throughput even on low-end hardware.

### 3. Centroid Tracking Algorithm
* **Script:** `lib/centroidtracker.py`
* **Details:** This script implements an object association algorithm using the Euclidean distance (`scipy.spatial.distance.cdist`).
  - As bounding boxes shift, the algorithm calculates the centroid `(x, y)` of the box.
  - By computing the distance between *new* centroids and *existing* centroids from the previous frame, it matches the newly detected object to an existing Unique ID.
  - If a centroid disappears for `maxDisappeared` consecutive frames (default: 40), the ID is deregistered.

### 4. Wait-Time Prediction Mathematics
* **Script:** `main.py`
* **Details:** The system incorporates a queue predictive feature based on physical dimension arguments:
  - `Queue Length` = `Distance` (`--distance`) / `Spacing` (`--spacing`)
  - A person receives a timestamp `entry_times[ID]` upon passing the **Entry Line (top)** and an `exit_times[ID]` upon crossing the **Counter Exit Line (bottom)**.
  - `Service Time` = `Exit Time` - `Entry Time`
  - `Predicted Wait Time` = `Queue Length` * `Average Service Time`

### 5. Multi-Threaded Video Streaming
* **Script:** `lib/thread.py`
* **Details:** Internal OpenCV video buffers (`VideoCapture`) can fill up or block, causing lag on physical web/IP cameras. The `ThreadingClass` creates a daemon `threading.Thread` to constantly poll the camera and place the freshest frame into a Python `queue.Queue`. The main loop only grabs the top of this queue, aggressively eliminating I/O bottlenecks.

### 6. Sub-Components
* **Automated Scheduling & Lifespan:** Handled utilizing the `schedule` package and delta time evaluation (`time.time()`).
* **Email Mailer:** Uses Python's `smtplib` via an SSL secure port (`465`) to notify building administrators if thresholds are breached.
* **CSV Logging:** Generates `Log.csv` daily using `itertools.zip_longest` to format "In", "Out", and "Total" columns properly against timestamps.

---

## 🛠️ Usage & Configuration

### Dependencies
Install the strictly version-locked dependencies:
```bash
pip install -r requirements.txt
```
*(Key deps: `opencv-python`, `dlib`, `imutils`, `scipy`, `schedule`, `numpy`)*

### Configuration (`lib/config.py`)
Edit `lib/config.py` to enable sub-features dynamically without changing the execution parameters:
- `ALERT`: Sets email notifications. Requires editing `lib/creds.py` with valid sender credentials.
- `Thread`: Enables async framing. Set to `True` for live webcams/IP cams.
- `Log`: Stores daily statistical logs.
- `Scheduler` & `Timer`: For zero-interaction, headless server deployments. Set `Timer = True` (e.g. 28800s / 8 hours limit) and `Scheduler = True` (run at `09:00` daily).

### Execution Commands

**1. Run Against a Pre-Recorded Video**
```bash
python main.py \
  --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
  --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel \
  --input videos/example.mp4
```

**2. Run Against a Live IP / Web Camera**
*(Note: Change `url = 0` in `config.py` for standard local webcams, or inject an RTSP/HTTP feed url)*
```bash
python main.py \
  --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
  --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel
```

**Optional Tuning Arguments:**
- `-c` or `--confidence`: Filter weak detections (Default: `0.25`)
- `-s` or `--skip-frames`: Frames to track between MobileNet detections (Default: `30`)
- `-d` or `--distance`: Total queue length in meters (Default: `12.0`)
- `-sp` or `--spacing`: Space between persons in queue (Default: `0.7` m) 

---
### Performance Notes
If running without a dedicated GPU, ensure `-s` (skip-frames) remains high (20-30). Deep neural network inferences are heavily CPU-bound when native OpenCV modules (`cv2.dnn`) execute without CUDA backends. Object tracking handles the bulk of the workflow perfectly to compensate.
