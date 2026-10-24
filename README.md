<h1 align="center">⚽ Football Match Analysis</h1>

<p align="center">
  <b>AI-powered football video analysis — player tracking, team detection, speed & distance estimation</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-purple?logo=yolo&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.40+-red?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenCV-4.8+-green?logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/Railway-deployable-0B0D0E?logo=railway&logoColor=white" />
  <img src="https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey" />
</p>

---

## 📌 Overview

**Football Match Analysis** takes any football match video and produces a fully annotated output with:

- 🔍 **Player & ball detection** using a fine-tuned YOLOv8 model
- 🏃 **Multi-object tracking** across every frame with ByteTrack
- 👕 **Automatic team assignment** based on jersey color clustering
- ⚡ **Speed & distance estimation** mapped onto calibrated pitch coordinates
- 🎥 **Camera movement compensation** for accurate position tracking
- 🏆 **Ball possession tracking** per team throughout the match

> **Note:** Speed and distance are estimates, not ground-truth measurements. Accuracy depends on video quality, detection confidence, and pitch calibration in `view_transformer/`.

---

## 🏗️ How It Works

```
┌─────────────────────────────────┐
│   Input Video + YOLO Weights    │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Detect & Track Players / Ball  │  ← YOLOv8 + ByteTrack
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Assign Teams & Ball Possession │  ← K-Means jersey color clustering
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Compensate Camera Movement     │  ← Optical flow estimation
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Map Positions to Pitch Coords  │  ← Perspective transformation
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Estimate Speed & Distance      │  ← Frame-rate-aware calculation
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Annotate & Export MP4          │
└─────────────────────────────────┘
```

The core pipeline lives in [`football_analysis/pipeline.py`](football_analysis/pipeline.py).  
Both `app.py` (Streamlit web UI) and `main.py` (CLI) run the same pipeline.

---

## ✨ Features at a Glance

| Feature | Details |
|---|---|
| 🎯 Object Detection | Fine-tuned YOLOv8 — players, referees, ball |
| 🔗 Multi-object Tracking | ByteTrack for consistent IDs across frames |
| 👕 Team Detection | K-Means clustering on jersey colors |
| 📍 Pitch Mapping | Perspective transform to real-world coordinates |
| 📷 Camera Compensation | Optical flow to cancel camera pan/tilt |
| ⚡ Speed & Distance | Per-player estimates in km/h and metres |
| 🏆 Possession Stats | Ball possession % per team |
| 🖥️ Web Interface | Streamlit UI — upload, analyse, download |
| ⌨️ CLI | Scriptable command-line interface |

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11**
- A compatible YOLO weights file (see [Model Weights](#-model-weights))

### 1 — Clone & Set Up Environment

```bash
git clone https://github.com/Arsen1302/yolo-football-tracker.git
cd yolo-football-tracker

python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2 — Add Your Model Weights

Place your YOLOv8 weights at the default path:

```
models/yolo-football-player-detection.pt
```

Or point to a custom location via environment variable:

```bash
export FOOTBALL_ANALYSIS_MODEL_PATH="/path/to/best.pt"
```

### 3 — Launch the Web App

```bash
streamlit run app.py
```

Open the URL printed by Streamlit, upload a video (MP4, MOV, AVI, MKV — up to 200 MB), and click **Analyze video**.  
The annotated result is saved to `output_videos/` and played back in the browser.

---

## ⌨️ Command-Line Interface

For scripting or batch processing, skip the web UI entirely:

```bash
python main.py path/to/match.mp4 \
  --output output_videos/result.mp4 \
  --model  models/best.pt
```

| Argument | Required | Description |
|---|---|---|
| `video` | No | Path to the input video file |
| `--output` | No | Output MP4 path (auto-generated if omitted) |
| `--model` | No | YOLO weights path (falls back to env var or default) |

---

## 🌍 Deployment

### ☁️ Railway (Recommended — pre-configured)

Your repo already includes a `Dockerfile` and `.railwayignore`.

1. Push to GitHub
2. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub**
3. Select your repo — Railway auto-detects the Dockerfile
4. Set the `FOOTBALL_ANALYSIS_MODEL_PATH` environment variable in Railway's dashboard
5. Your app gets a public HTTPS URL automatically ✅

### 🐳 Docker

```bash
docker build -t football-analysis .
docker run -p 8501:8501 \
  -e FOOTBALL_ANALYSIS_MODEL_PATH=/app/models/best.pt \
  -v $(pwd)/models:/app/models \
  football-analysis
```

### 🌐 Streamlit Community Cloud (Free)

1. Push to a **public** GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Point to `app.py` as the entry point
4. Add `FOOTBALL_ANALYSIS_MODEL_PATH` as a secret

### 🔗 Quick Public Tunnel (ngrok)

```bash
# Terminal 1
streamlit run app.py

# Terminal 2
ngrok http 8501
```

ngrok prints a public URL like `https://abc123.ngrok.io`.

---

## 🧠 Model Weights

Model weights are **not** stored in this repository (they are excluded by `.gitignore`).

The project was trained on Roboflow's  
[football-players-detection v1](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/1) dataset  
— 663 images, 15,798 valid bounding boxes, licensed under **CC BY 4.0**.

Pre-trained YOLOv8 base weights (`yolov8n.pt`, `yolov8s.pt`, `yolov8x.pt`) are included locally for convenience but are not committed to Git.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `FOOTBALL_ANALYSIS_MODEL_PATH` | `models/yolo-football-player-detection.pt` | Path to YOLO weights |
| `FOOTBALL_ANALYSIS_IMAGE_SIZE` | `640` | Input image size for the detector |
| `FOOTBALL_DATASET_ROOT` | — | Path to local YOLO dataset (training only) |

---

## 🗂️ Project Structure

```
football_analysis/
├── app.py                          # Streamlit web interface
├── main.py                         # CLI entry point
│
├── football_analysis/
│   ├── __init__.py                 # Public API: process_video()
│   ├── config.py                   # Model & output path resolution
│   └── pipeline.py                 # Main analysis pipeline (orchestrator)
│
├── trackers/
│   └── tracker.py                  # YOLOv8 detection + ByteTrack tracking
├── team_assigner/
│   └── team_assigner.py            # Jersey-color team assignment
├── player_ball_assigner/
│   └── player_ball_assigner.py     # Ball possession estimation
├── camera_movement_estimator/
│   └── camera_movement_estimator.py # Optical flow camera compensation
├── view_transformer/
│   └── view_transformer.py         # Perspective transform to pitch coords
├── speed_and_distance_estimator/
│   └── speed_and_distance_estimator.py # Speed & distance calculations
├── utils/
│   ├── video_utils.py              # Video I/O helpers
│   └── bbox_utils.py               # Bounding-box geometry helpers
│
├── src/
│   ├── football_dataset.py         # Dataset validation & YAML generation
│   └── train_models.py             # Multi-run training & experiment tracking
│
├── notebooks/
│   └── 01_football_dataset_eda.ipynb # Dataset exploration & audit
├── development_and_analysis/
│   └── color_assignement.ipynb     # Team color assignment exploration
│
├── tests/
│   └── test_application_architecture.py # Architecture unit tests
│
├── models/                         # YOLO weights (not in Git)
├── input_videos/                   # Source videos (not in Git)
├── output_videos/                  # Processed output videos (not in Git)
├── requirements.txt
└── Dockerfile
```

---

## 🏋️ Training Your Own Model

Dataset exploration and training are independent of the video analysis app.

### 1 — Set Dataset Path

```bash
export FOOTBALL_DATASET_ROOT="$HOME/Downloads/football-players-detection.v1i.yolov5pytorch"
```

Expected layout:

```
<dataset-root>/
├── data.yaml
├── train/{images,labels}/
├── valid/{images,labels}/
└── test/{images,labels}/
```

### 2 — Explore the Dataset

Open [`notebooks/01_football_dataset_eda.ipynb`](notebooks/01_football_dataset_eda.ipynb) in VS Code (Jupyter extension) or JupyterLab.  
It audits splits, missing labels, invalid boxes, class distribution, and box sizes.

> ⚠️ The test split contains only **13 images** — test metrics from this split are highly uncertain.

### 3 — Run Training Experiments

```bash
python -m src.train_models \
  --dataset-root "$FOOTBALL_DATASET_ROOT" \
  --models       yolov8n.pt,yolov8s.pt \
  --learning-rates 0.001,0.01 \
  --epochs       20 \
  --imgsz        640 \
  --batch        4
```

Results are written to `outputs/experiments/`:

| File | Contents |
|---|---|
| `experiment_metrics.csv` | Parameters, status, errors, metrics per run |
| `selected_model.json` | Best checkpoint + test-split evaluation |
| `runs/` | Training logs and model checkpoints |

The script auto-selects the device: **CUDA → Apple MPS → CPU**.  
Use `--device` to override.

---

## 🧪 Tests

```bash
python -m unittest discover -s tests -v
```

Covers model/output path configuration and pipeline selection behavior.

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `ultralytics` | latest | YOLOv8 detection & training |
| `supervision` | ≥0.25.1, <0.28 | Annotation & tracking utilities |
| `opencv-python` | ≥4.8, <4.12 | Video I/O & optical flow |
| `streamlit` | ≥1.40, <2 | Web interface |
| `numpy` | 2.2.6 | Numerical operations |
| `pandas` | 2.2.3 | Experiment metrics |
| `scikit-learn` | latest | K-Means color clustering |
| `matplotlib` | latest | Visualizations |
| `PyYAML` | ≥6.0, <7 | Dataset YAML handling |

---

## 📄 License

Dataset: **CC BY 4.0** — credit [Roboflow](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/1) when using or redistributing.

---

<p align="center">Made with ⚽ and Python</p>
