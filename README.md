# Football Match Analysis

Football Match Analysis processes a football video and produces an annotated
video with detected and tracked players, referees, and the ball. It also
estimates teams, ball possession, camera movement, player speed, and distance.
The project provides a Streamlit interface for video uploads and a command-line
interface (CLI); both use the same processing pipeline.

Speed and distance are estimates, not ground-truth measurements. Their
accuracy depends on detections, video quality, and the pitch calibration in
`view_transformer/`.

## How the application works

```text
Input video + YOLO weights
          │
          ▼
Detect objects and track them between frames
          │
          ▼
Estimate player positions, teams, and ball possession
          │
          ▼
Estimate camera movement and map positions onto a pitch
          │
          ▼
Draw annotations and save a processed MP4
```

The main analysis code is in `football_analysis/pipeline.py`. It coordinates
the analysis components in the project; `app.py` and `main.py` are two ways to
run that same service.

## Requirements and local setup

- Python 3.11
- A YOLO model weight file. By default, the application looks for
  `models/yolo-football-player-detection.pt`. Model weights are not included
  in Git, so provide your own compatible weights or set
  `FOOTBALL_ANALYSIS_MODEL_PATH`.
- The packages listed in `requirements.txt`

Create an environment, install the packages, and start the web application
from the project root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, upload an MP4, MOV, AVI, or MKV file,
and choose **Analyze video**. The processed video is saved under
`output_videos/` and displayed in the page. Processing time depends on video
length and available hardware. Streamlit's default per-file upload limit is
200 MB.

If the model is stored somewhere else, set its path before starting the app:

```bash
export FOOTBALL_ANALYSIS_MODEL_PATH="/path/to/best.pt"
streamlit run app.py
```

The detector uses an image size of 640 by default. To change it, set
`FOOTBALL_ANALYSIS_IMAGE_SIZE` to a positive integer before starting the app.

### Run a video from the command line

The CLI is useful for scripts or when you do not need the upload interface:

```bash
python main.py path/to/match.mp4 \
  --output output_videos/result.mp4 \
  --model /path/to/best.pt
```

The input video, output path, and model path can be provided independently:

- `video` is the optional positional input-video path.
- `--output` is an optional destination for the processed MP4.
- `--model` is an optional YOLO weights path.

If `--model` is omitted, the program uses `FOOTBALL_ANALYSIS_MODEL_PATH` or the
default model path. If `--output` is omitted, a uniquely named MP4 is created
in `output_videos/`. Pass the input video explicitly; the CLI's fallback
sample-video path is only useful if that sample exists on your computer.

## Dataset exploration and model training

Dataset exploration and training are separate from the video-analysis
application. They require a local YOLO dataset; the dataset itself is not
stored in this repository.

The EDA and training tools were prepared for Roboflow's
[football-players-detection v1 dataset](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/1),
provided as a YOLOv5 export and licensed under **CC BY 4.0**. Follow the
license and credit Roboflow when using or redistributing the dataset.

Expected dataset layout:

```text
<dataset-root>/
├── data.yaml
├── train/{images,labels}/
├── valid/{images,labels}/
└── test/{images,labels}/
```

Set the path to your local dataset:

```bash
export FOOTBALL_DATASET_ROOT="$HOME/Downloads/football-players-detection.v1i.yolov5pytorch"
```

### Explore and audit the dataset

Open `notebooks/01_football_dataset_eda.ipynb` in VS Code with the Jupyter
extension or in another notebook editor. It reviews dataset splits, missing or
orphan labels, invalid bounding boxes, class distribution, box sizes, and
example images. The notebook uses `src/football_dataset.py` for the dataset
audit. The tools generate a corrected Ultralytics YAML in the experiment
output folder; they do not edit the original dataset or its labels.

The recorded audit covered 663 images and 15,798 valid boxes. The test split
contains only 13 images, so any test metrics from this dataset are highly
uncertain.

### Train and compare YOLO models

Run the experiment script from the project root:

```bash
python -m src.train_models \
  --dataset-root "$FOOTBALL_DATASET_ROOT" \
  --models yolov8n.pt,yolov8s.pt \
  --learning-rates 0.001,0.01 \
  --epochs 20 \
  --imgsz 640 \
  --batch 4
```

The script compares completed runs using validation mAP50-95, selects the
best validation run, and evaluates the selected model on the held-out test
split. It chooses CUDA, Apple MPS, or CPU when available; use `--device` to
specify a device. Results are written to `outputs/experiments/`:

- `experiment_metrics.csv` records parameters, status, errors, and metrics
  for each run.
- `selected_model.json` records the selected checkpoint and its test results.
- `runs/` contains training output and model checkpoints.

Training uses significant time, disk space, and memory. The tested YOLOv8n
runs completed, but a YOLOv8s run stopped because the disk ran out of space.
The selected model's reported test metrics were measured on only 13 images and
should not be treated as evidence of broad generalization.

## Project files

### Application entry points and shared service

- `app.py` — Streamlit user interface. Accepts a video upload, calls the
  shared processing service, and displays the resulting MP4.
- `main.py` — CLI entry point. Parses video, model, and output paths, then
  calls the same service as the web interface.
- `football_analysis/__init__.py` — exposes `process_video` as the package's
  public application function.
- `football_analysis/config.py` — resolves the model and output paths. It
  reads `FOOTBALL_ANALYSIS_MODEL_PATH` and creates output directories as
  needed.
- `football_analysis/pipeline.py` — coordinates detection, tracking, team
  assignment, possession estimation, camera-motion correction, pitch
  transformation, speed/distance estimates, and annotated-video writing.

### Video analysis components

- `trackers/tracker.py` — runs YOLO detections, associates objects across
  frames with ByteTrack, selects ball detections, and prepares track positions
  and annotations.
- `team_assigner/team_assigner.py` — assigns players to teams using jersey
  colors.
- `player_ball_assigner/player_ball_assigner.py` — estimates which player has
  possession based on player and ball positions.
- `camera_movement_estimator/camera_movement_estimator.py` — estimates camera
  movement between frames and adjusts tracked positions.
- `view_transformer/view_transformer.py` — maps image positions to the
  project's calibrated pitch coordinates.
- `speed_and_distance_estimator/speed_and_distance_estimator.py` — estimates
  player movement speed and distance from transformed positions and video
  frame rate.
- `utils/video_utils.py` — reads video metadata and frames and writes
  processed video.
- `utils/bbox_utils.py` — provides bounding-box geometry helpers used by the
  analysis components.

### Data, experiments, tests, and presentation

- `src/football_dataset.py` — validates the dataset layout, audits YOLO
  annotations, and creates a training-ready Ultralytics dataset YAML.
- `src/train_models.py` — runs and tracks model/learning-rate experiments,
  selects by validation metrics, and evaluates the selected model on the test
  split.
- `notebooks/01_football_dataset_eda.ipynb` — interactive dataset exploration
  and visual analysis.
- `development_and_analysis/color_assignement.ipynb` — exploratory notebook
  for player-color/team assignment.
- `data/README.md` — explains where to keep the dataset and its expected
  directory layout.
- `tests/test_application_architecture.py` — unit tests for model/output path
  configuration and pipeline selection behavior.
- `presentation/football_analysis_presentation_hy_v2.pptx` — Armenian
  presentation deck for explaining the project.
- `presentation/build_presentation.py` — source script used to build the
  presentation.

### Project configuration and local assets

- `requirements.txt` — Python dependencies for the application and project
  tools, including the pinned NumPy and pandas versions.
- `Dockerfile` — describes the Python runtime, system packages, dependencies,
  and application command for a container image.
- `.dockerignore` — excludes local environments, caches, datasets, videos,
  and development-only files from a Docker build context.
- `.railwayignore` — excludes local-only files from the files sent to Railway.
- `.gitignore` — keeps local model weights, datasets, videos, outputs,
  secrets, and generated files out of Git.
- `models/` — location for local YOLO weights. The weights are intentionally
  not committed to the repository.
- `input_videos/` — optional location for local source videos.
- `output_videos/` — default location for generated processed videos.

## Tests

Run the application architecture tests from the project root:

```bash
python -m unittest discover -s tests -v
```
