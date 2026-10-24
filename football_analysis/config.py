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

HF_REPO_ID = "Arsen006/football-analysis-model"
HF_FILENAME = "yolo-football-player-detection.pt"


def _ensure_model(path: Path) -> Path:
    """Download the model from Hugging Face Hub if it is not present locally."""
    if path.is_file():
        return path
    try:
        from huggingface_hub import hf_hub_download

        print(
            f"Model not found at {path}. "
            f"Downloading from Hugging Face Hub ({HF_REPO_ID})…"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            local_dir=str(path.parent),
        )
        return Path(downloaded)
    except Exception as exc:
        raise FileNotFoundError(
            f"Model weights not found at {path} and could not be downloaded "
            f"from Hugging Face Hub ({HF_REPO_ID}/{HF_FILENAME}). "
            f"Error: {exc}"
        ) from exc


def resolve_model_path(model_path: str | Path | None = None) -> Path:
    configured_path = (
        model_path
        if model_path is not None
        else os.environ.get("FOOTBALL_ANALYSIS_MODEL_PATH")
    )
    if configured_path is None:
        configured_path = DEFAULT_MODEL_PATH
    resolved = Path(configured_path).expanduser().resolve()
    return _ensure_model(resolved)


def resolve_output_path(output_path: str | Path | None = None) -> Path:
    if output_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"processed_{uuid4().hex}.mp4"

    resolved_path = Path(output_path).expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    return resolved_path
