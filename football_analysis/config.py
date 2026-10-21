import os
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO_PATH = (
    PROJECT_ROOT
    / "input_videos"
    / "Screen Recording 2026-09-20 at 16.27.37.mov"
)
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT / "models" / "yolo-football-player-detection.pt"
)
OUTPUT_DIR = PROJECT_ROOT / "output_videos"


def resolve_model_path(model_path: str | Path | None = None) -> Path:
    configured_path = (
        model_path
        if model_path is not None
        else os.environ.get("FOOTBALL_ANALYSIS_MODEL_PATH")
    )
    if configured_path is None:
        configured_path = DEFAULT_MODEL_PATH
    return Path(configured_path).expanduser().resolve()


def resolve_output_path(output_path: str | Path | None = None) -> Path:
    if output_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"processed_{uuid4().hex}.mp4"

    resolved_path = Path(output_path).expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    return resolved_path
