# Football Match Analysis

This project analyzes football match videos with computer vision. A YOLO model
detects players, referees, and the ball. ByteTrack follows detections between
frames, and later stages estimate teams, possession, and player movement. The
Streamlit application displays the processed video and saves it as an MP4.

## Features

- Detect players, referees, and the ball; track player IDs across frames.
- Select a ball candidate using its size, shape, confidence, and previous position.
- Estimate two teams by clustering jersey colors with K-Means.
- Estimate camera movement using Lucas–Kanade optical flow.
- Project image coordinates onto a calibrated pitch and estimate player speed and distance.
- Upload and review processed videos in a Streamlit interface.
- Audit and explore a YOLO dataset, and run tracked model-training experiments.

Speed and distance are estimates, not ground-truth measurements. Their accuracy
depends on the video, detections, and pitch calibration in `view_transformer/`.

## Requirements

- Python 3.11
- Packages listed in `requirements.txt`
- Application weights at `models/yolo-football-player-detection.pt`, or a
  different model path supplied with `FOOTBALL_ANALYSIS_MODEL_PATH`

Model weights, videos, datasets, and training outputs are local assets and are
not committed to Git. Do not commit private videos, credentials, or other
sensitive files.

## Run locally

From the project root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py --server.port 7860
```

Open [http://localhost:7860](http://localhost:7860), upload an MP4, MOV, AVI, or
MKV video, and select **Analyze video**. The result is saved in `output_videos/`.
Processing time depends on video length and available hardware. Streamlit's
default per-file upload limit is 200 MB.

To use a different model:

```bash
FOOTBALL_ANALYSIS_MODEL_PATH=/path/to/best.pt streamlit run app.py --server.port 7860
```

The command-line interface uses the same processing pipeline:

```bash
python main.py path/to/match.mp4 --output output_videos/result.mp4 --model /path/to/best.pt
```

The video, `--output`, and `--model` arguments are optional. Without a video
argument, the CLI looks for
`input_videos/Screen Recording 2026-09-20 at 16.27.37.mov`. Without `--model`,
it uses `FOOTBALL_ANALYSIS_MODEL_PATH` or the default model path.

## Dataset

EDA and training use Roboflow's
[football-players-detection v1](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/1)
YOLOv5 export. The dataset is licensed under **CC BY 4.0**. Credit Roboflow and
follow the license when using or redistributing the dataset.

Expected local dataset layout:

```text
<dataset-root>/
├── data.yaml
├── train/{images,labels}/
├── valid/{images,labels}/
└── test/{images,labels}/
```

For example:

```bash
export FOOTBALL_DATASET_ROOT="$HOME/Downloads/football-players-detection.v1i.yolov5pytorch"
```

The dataset tools create a corrected Ultralytics YAML in the experiment output
directory. They do not modify the original dataset configuration or labels.

## Exploratory data analysis

Open `notebooks/01_football_dataset_eda.ipynb` in VS Code's Jupyter extension or
another notebook editor. The notebook audits splits, missing, empty, and orphan
label files, invalid bounding boxes, class distributions, bounding-box sizes,
engineered descriptive features, correlations, and labeled image examples.

The dataset audit found:

| Split | Images | Bounding boxes |
|---|---:|---:|
| Train | 612 | 14,583 |
| Validation | 38 | 905 |
| Test | 13 | 310 |
| **Total** | **663** | **15,798** |

No missing label files, empty label files, or invalid bounding-box rows were
found in the audit. The splits are the ones supplied with the dataset. The test
split contains only 13 images, so its metrics are highly uncertain and should
not be treated as evidence of broad generalization.

## Training, hyperparameter tuning, and experiment tracking

The training command runs a grid over model sizes and learning rates, compares
completed runs using validation mAP50-95, selects the best validation run, and
evaluates the held-out test split once for that selected model:

```bash
python -m src.train_models \
  --dataset-root "$FOOTBALL_DATASET_ROOT" \
  --models yolov8n.pt,yolov8s.pt \
  --learning-rates 0.001,0.01 \
  --epochs 20 \
  --imgsz 640 \
  --batch 4
```

By default, the script uses the available device: CUDA, Apple MPS, or CPU. Set a
specific device with `--device`. Results are written to `outputs/experiments/`,
which is excluded from Git:

- `experiment_metrics.csv` records each run's parameters, status, errors, and
  validation metrics.
- `selected_model.json` records the validation-selected checkpoint and its test metrics.
- `runs/` contains Ultralytics run files and checkpoints.

Training can take a long time and use substantial memory and disk space. Review
the CSV and JSON outputs after a run.

### Recorded experiment results (Apple M1, MPS)

Two YOLOv8n learning-rate runs completed with 20 epochs. The run with
`lr0=0.01` was selected based on validation mAP50-95. A YOLOv8s run stopped
during its first epoch because the disk ran out of space while saving a
checkpoint. The full model-size comparison therefore remains incomplete.

| Model | `lr0` | Validation mAP50 | Validation mAP50-95 | Status |
|---|---:|---:|---:|---|
| YOLOv8n | 0.001 | 0.6637 | 0.4272 | Completed |
| YOLOv8n | 0.01 | 0.6868 | 0.4378 | Selected |
| YOLOv8s | 0.001 | — | — | Incomplete: disk-space error |
| YOLOv8s | 0.01 | — | — | Not run |

The selected YOLOv8n model was evaluated once on the test split:

| Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|
| 0.8570 | 0.6303 | 0.7186 | 0.4421 |

These test results are based on only 13 images and are highly uncertain. They
should not be presented as proof of general performance. PyTorch also warned
that some MPS operations are not fully deterministic, so repeating a run with
the same seed may produce slightly different results. Check that enough disk
space is available before completing the model-size comparison.

## Docker

Make the application model weights available at
`models/yolo-football-player-detection.pt` during the build, then run:

```bash
docker build -t football-analysis .
docker run --rm -p 7860:7860 \
  -v "$(pwd)/output_videos:/app/output_videos" \
  football-analysis
```

The container listens on the `PORT` environment variable (default `7860`).
Input videos and training datasets are excluded from the Docker build context.
Provide the model weights during the build or deployment.

## Deployment

Connect the GitHub repository to Railway or deploy with the Railway CLI; the
project's `Dockerfile` is used. Ensure the model weights are available to the
service and that the app listens on Railway's `PORT`. For larger models, use a
model storage option supported by your hosting platform and set
`FOOTBALL_ANALYSIS_MODEL_PATH` to the downloaded weights. Test with a short
clip first and monitor memory usage.

## Project structure

```text
.
├── app.py
├── main.py
├── football_analysis/
│   ├── config.py
│   └── pipeline.py
├── src/
│   ├── football_dataset.py
│   └── train_models.py
├── notebooks/
│   └── 01_football_dataset_eda.ipynb
├── data/                         # Dataset setup guide
├── trackers/
├── camera_movement_estimator/
├── view_transformer/
├── speed_and_distance_estimator/
├── team_assigner/
├── player_ball_assigner/
├── utils/
└── models/                       # Local YOLO weights; excluded from Git
```

Application layers:

- `app.py` — Streamlit interface for uploading videos and viewing results.
- `main.py` — command-line entry point.
- `football_analysis/config.py` — model and output path configuration.
- `football_analysis/pipeline.py` — video-analysis workflow.
- `trackers/`, `team_assigner/`, `camera_movement_estimator/`,
  `view_transformer/`, `speed_and_distance_estimator/`, and
  `player_ball_assigner/` — analysis components.
- `utils/` — shared video and bounding-box helpers.
- `src/` and `notebooks/` — dataset, EDA, and training tools, separate from the
  production inference pipeline.
- `presentation/football_analysis_presentation_hy_v2.pptx` — Armenian
  presentation deck with speaker notes on each slide.
- `presentation/build_presentation.py` — presentation build script.

The Streamlit UI and CLI use the same `football_analysis.process_video`
service. Run the focused application tests with:

```bash
python -m unittest discover -s tests -v
```
