# CivicGuard: Automated Road Damage & Pothole Segmentation System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8 Segmentation](https://img.shields.io/badge/YOLOv8-Instance%20Segmentation-00FFFF.svg?logo=ultralytics&logoColor=white)](https://ultralytics.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

CivicGuard is an AI-powered computer vision infrastructure surveillance platform designed for municipal bodies, road safety authorities, and smart city engineers. It automates the detection, instance segmentation, and quantification of road surface distress—including potholes, alligator cracks, longitudinal cracks, and rutting—from video feeds and road imagery in real time.

---

## Key Features

- **Multi-Class Distress Segmentation**: Employs fine-tuned YOLOv8 instance segmentation (`yolov8s-seg`) to delineate precise pothole and crack boundaries rather than simple bounding boxes.
- **Real-Time Video Analytics**: High-throughput video stream processor with real-time HUD overlays, incident counting, and low-latency inference on CUDA-enabled GPUs.
- **Robust Data Augmentation Pipeline**: Trained with mosaic, mixup, perspective transformations, scaling, and HSV color space shifts tailored specifically to heterogeneous Indian and global road conditions.
- **Comprehensive Evaluation Toolkit**: Native mAP calculator supporting both standard COCO metrics (`mAP@50`, `mAP@50:95`) and class-level Precision, Recall, and F1 breakdowns.
- **Urban Condition Analytics**: Integration support for municipal telemetry, road age, traffic density, and monsoon weather metrics for predictive road maintenance.

---

## Model Benchmark & Evaluation Summary

Evaluated on test splits against standard COCO evaluation criteria:

| Metric | Score | Description |
|---|---|---|
| **mAP@50** | **0.6458** | Mean Average Precision at IoU 0.50 |
| **mAP@50:95** | **0.2466** | COCO standard metric averaged across IoU 0.50:0.95 |
| **Average Precision** | **0.7518** | Mean detection confidence across detected classes |
| **Average Recall** | **0.7321** | Mean damage detection coverage |
| **Average F1-Score** | **0.7397** | Harmonic balance between precision and recall |

### Class-Level Breakdown

| Distress Class | AP@50 | AP@50:95 | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| **Pothole** | 0.5952 | 0.1686 | 0.7692 | 0.7143 | 0.7407 |
| **Alligator Cracking** | 0.6667 | 0.3452 | 0.8333 | 0.7143 | 0.7692 |
| **Longitudinal Crack** | 0.5333 | 0.1489 | 0.5714 | 0.6667 | 0.6154 |
| **Rutting** | 0.7879 | 0.3237 | 0.8333 | 0.8333 | 0.8333 |

---

## Repository Architecture

```text
CivicGuard/
├── Pothole_Segmentation_YOLOv8/
│   ├── map.py                      # Multi-threshold mAP & CV metric evaluation suite
│   ├── data.yaml                   # YOLO dataset definition (classes & directory paths)
│   └── sample_video.mp4            # Sample dashcam footage for inference demo
├── detect.py                       # Real-time video detection & HUD visualizer
├── result.py                       # Video inference pipeline with real-time damage counter
├── main.py                         # Production model training pipeline (YOLOv8s-seg)
├── newmain.py                      # Optimized training script with cosine LR and regularization
├── detectecal.py                   # Model training and calibration runner
├── kaggle.py                       # Cloud training bootstrap script for Kaggle environments
├── indian_roads_mock_data.csv      # Municipal telemetry dataset (road age, traffic, weather)
├── map_evaluation_report.json      # Structured benchmark evaluation logs
├── map_evaluation_report_summary.txt# Summary of metric evaluation results
└── .gitignore                      # Excludes venvs, caches, and weight binaries
```

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- NVIDIA GPU with CUDA support (recommended for real-time video processing and training)

### 1. Clone Repository & Setup Environment

```bash
git clone https://github.com/Ash7810/Civic-Guard.git
cd Civic-Guard

# Create and activate virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Linux / macOS:
# source .venv/bin/activate

# Upgrade pip & install core dependencies
pip install --upgrade pip
pip install ultralytics opencv-python numpy matplotlib
```

### 2. Run Real-Time Video Inference

To execute detection on dashcam footage with live on-screen distress visualization:

```bash
python detect.py
```

To run inference with live incident counting overlay:

```bash
python result.py
```

> **Note**: Update the model weight path in `detect.py` / `result.py` (`runs/segment/.../weights/best.pt`) to point to your trained weights.

### 3. Training the Model

To initiate training on custom annotated datasets:

```bash
python main.py
```

Key training hyperparameters configured in `main.py`:
- **Base Architecture**: `yolov8s-seg.pt` (Speed / Accuracy balanced)
- **Input Resolution**: `640x640`
- **Batch Size**: `6` (tuned for standard GPU VRAM headroom)
- **Optimizations**: Mixed Precision (`amp=True`), Cosine LR scheduling (`cos_lr=True`), Disk caching (`cache='disk'`)
- **Augmentation**: Mosaic, MixUp (0.2), Copy-Paste (0.3), Scale & Shear

### 4. Running Benchmarks & Metrics Evaluation

To compute mAP, precision, recall, and class breakdowns:

```bash
python Pothole_Segmentation_YOLOv8/map.py
```

The evaluator produces formatted summaries and structured JSON outputs (`map_evaluation_report.json`).

---

## Road Condition & Telemetry Analytics

CivicGuard links computer vision detections with physical environmental factors (`indian_roads_mock_data.csv`):
- **Monsoon & Coastal Weather Impact**: Correlating waterlogging with pothole frequency.
- **Traffic Load & Road Age**: Severity ranking based on traffic tier (Peak vs Medium) and pavement lifecycle.
- **Smart City Dashboards**: Enables priority-ranked repair scheduling for urban planning authorities.

---

## Roadmap

- [x] YOLOv8 instance segmentation fine-tuning for pothole and road distress.
- [x] Multi-metric evaluation framework (`mAP@50`, `mAP@50:95`, F1-score).
- [x] Real-time live HUD video counter.
- [ ] Automated REST API service with FastAPI.
- [ ] GPS tag extraction and GIS map heatmapping.
- [ ] Edge device deployment (NVIDIA Jetson / Raspberry Pi with ONNX runtime).

---

## Contributing

Contributions are welcome! Please open an issue first to discuss intended changes or submit a pull request:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/InnovativeFeature`)
3. Commit your Changes (`git commit -m "Add InnovativeFeature"`)
4. Push to the Branch (`git push origin feature/InnovativeFeature`)
5. Open a Pull Request

---

## License

Distributed under the MIT License. See `LICENSE` for more information.
