# Attention / Behaviour Detection Model

Real-time student attention detection using **MediaPipe**, **YOLOv8**, and machine-learning classifiers (XGBoost + Random Forest). Classifies webcam frames into three states: **Focused**, **Drowsy**, and **Looking Away**, with additional phone-usage detection.

## Features

- Face-mesh, hand, and pose landmark extraction via MediaPipe
- Head-pose estimation using solvePnP
- Object detection (phone, book, laptop, etc.) via YOLOv8
- Two feature pipelines (V3 / V4) with trained XGBoost and Random Forest models
- Live webcam inference with rule-based phone-usage override

## Project Structure

```
main.py                   # CLI entry point
requirements.txt
yolov8n.pt                # YOLOv8-nano weights
data/                     # Dataset images (Drowsy / Focused / Looking_away)
features/                 # Extracted feature CSVs
models/                   # Trained model pickles + evaluation plots
scripts/
  extract_features_v3.py  # V3 feature extraction (81 features)
  extract_features_v4.py  # V4 feature extraction (~78 generalised features)
  train_m_dataset_v3.py   # Train on V3 features
  train_v4.py             # Train on V4 features
  live_test_v3.py         # Live webcam — V3 ML models
  live_test_v4.py         # Live webcam — rule-based engine
  live_test_v5.py         # Live webcam — V4 ML + phone cascade
```

## Quick Start

```bash
# 1. Create virtual environment & install dependencies
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

# 2. Run the full pipeline (extract → train → live)
python main.py all

# Or run steps individually:
python main.py extract        # Extract V4 features
python main.py train          # Train models
python main.py live           # Live webcam detection
```

## All Commands

| Command | Description |
|---------|-------------|
| `python main.py extract` | Extract V4 features from dataset |
| `python main.py train` | Train XGBoost + RF on V4 features |
| `python main.py live` | Live webcam detection (V5 — recommended) |
| `python main.py extract-v3` | Extract V3 features (legacy) |
| `python main.py train-v3` | Train on V3 features (legacy) |
| `python main.py live-v3` | Live webcam V3 (legacy) |
| `python main.py live-rules` | Live webcam rule-based engine |
| `python main.py all` | Full V4/V5 pipeline |
| `python main.py all-v3` | Full V3 legacy pipeline |

## Requirements

- Python 3.10+
- Webcam (for live detection)
- See [requirements.txt](requirements.txt) for packages
