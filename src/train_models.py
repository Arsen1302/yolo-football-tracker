from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO

from .football_dataset import create_ultralytics_data_yaml

CSV_FIELDS = [
    "run_id",
    "status",
    "error",
    "model",
    "lr0",
    "epochs",
    "imgsz",
    "batch",
    "seed",
    "device",
    "train_seconds",
    "best_weights",
    "val_precision",
    "val_recall",
    "val_map50",
    "val_map50_95",
    "test_precision",
    "test_recall",
    "test_map50",
    "test_map50_95",
    "selected",
]


def parse_csv_values(value: str, cast: type) -> list[Any]:
    try:
        values = [cast(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"Invalid comma-separated value: {value}"
        ) from error
    if not values:
        raise argparse.ArgumentTypeError("At least one value is required.")
    return values


def resolve_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def append_or_update_csv(path: Path, row: dict[str, Any]) -> None:
    rows: list[dict[str, str]] = []
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))

    updated = False
    for index, existing in enumerate(rows):
        if existing["run_id"] == row["run_id"]:
            rows[index] = {field: str(row.get(field, "")) for field in CSV_FIELDS}
            updated = True
            break
    if not updated:
        rows.append({field: str(row.get(field, "")) for field in CSV_FIELDS})

    temporary_path = path.with_suffix(".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def get_box_metrics(model_path: Path, data_yaml: Path, split: str, output_dir: Path):
    results = YOLO(str(model_path)).val(
        data=str(data_yaml),
        split=split,
        project=str(output_dir),
        name=f"{split}_{model_path.parent.parent.name}",
        exist_ok=True,
        plots=False,
        verbose=False,
    )
    box = results.box
    return {
        f"{split}_precision": float(box.mp),
        f"{split}_recall": float(box.mr),
        f"{split}_map50": float(box.map50),
        f"{split}_map50_95": float(box.map),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train, compare, tune, and track YOLO football detectors."
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--models", default="yolov8n.pt,yolov8s.pt")
    parser.add_argument("--learning-rates", default="0.001,0.01")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiments"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.epochs <= 0 or args.imgsz <= 0 or args.batch <= 0 or args.workers < 0:
        raise SystemExit("epochs/imgsz/batch must be positive and workers non-negative.")

    models = parse_csv_values(args.models, str)
    learning_rates = parse_csv_values(args.learning_rates, float)
    if len(set(models)) != len(models):
        raise SystemExit("Model names must be unique.")
    if len(set(learning_rates)) != len(learning_rates):
        raise SystemExit("Learning rates must be unique.")
    if any(not math.isfinite(rate) or rate <= 0 for rate in learning_rates):
        raise SystemExit("All learning rates must be finite and greater than zero.")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = create_ultralytics_data_yaml(
        args.dataset_root,
        output_dir / "dataset.yaml",
    )
    run_root = output_dir / "runs"
    validation_root = output_dir / "validation"
    test_root = output_dir / "test"
    run_root.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    log_path = output_dir / "experiment_metrics.csv"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    completed_runs: list[dict[str, Any]] = []

    for model_name, learning_rate in product(models, learning_rates):
        model_slug = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(model_name).stem)
        run_id = f"{timestamp}_{model_slug}_lr{learning_rate:g}"
        run_dir = run_root / run_id
        if run_dir.exists():
            raise FileExistsError(f"Experiment directory already exists: {run_dir}")

        row: dict[str, Any] = {
            "run_id": run_id,
            "status": "running",
            "error": "",
            "model": model_name,
            "lr0": learning_rate,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "seed": args.seed,
            "device": device,
            "selected": False,
        }
        append_or_update_csv(log_path, row)
        start = time.perf_counter()
        best_weights = run_dir / "weights" / "best.pt"
        try:
            model = YOLO(model_name)
            model.train(
                data=str(data_yaml),
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                workers=args.workers,
                device=device,
                seed=args.seed,
                deterministic=True,
                optimizer="AdamW",
                lr0=learning_rate,
                patience=6,
                amp=False,
                cache=False,
                project=str(run_root),
                name=run_id,
                exist_ok=False,
            )
            train_seconds = round(time.perf_counter() - start, 2)
            if not best_weights.is_file():
                raise FileNotFoundError(
                    f"Training did not produce best weights: {best_weights}"
                )
            validation_metrics = get_box_metrics(
                best_weights,
                data_yaml,
                "val",
                validation_root,
            )
        except Exception as error:
            row.update(
                {
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "train_seconds": round(time.perf_counter() - start, 2),
                }
            )
            append_or_update_csv(log_path, row)
            raise

        row.update(validation_metrics)
        row.update(
            {
                "status": "completed",
                "train_seconds": train_seconds,
                "best_weights": str(best_weights),
            }
        )
        append_or_update_csv(log_path, row)
        completed_runs.append(row)
        print(
            f"{run_id}: val mAP50-95={row['val_map50_95']:.4f}, "
            f"mAP50={row['val_map50']:.4f}"
        )

    best_run = max(
        completed_runs,
        key=lambda run: (run["val_map50_95"], run["val_map50"]),
    )
    test_metrics = get_box_metrics(
        Path(best_run["best_weights"]),
        data_yaml,
        "test",
        test_root,
    )
    best_run.update(test_metrics)
    best_run["selected"] = True
    append_or_update_csv(log_path, best_run)

    selection = {
        "selection_metric": "validation mAP50-95",
        "selected_run": best_run["run_id"],
        "model": best_run["model"],
        "weights": best_run["best_weights"],
        "validation_metrics": {
            key: best_run[key]
            for key in ("val_precision", "val_recall", "val_map50", "val_map50_95")
        },
        "test_metrics": test_metrics,
        "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
        "dataset_yaml": str(data_yaml),
        "seed": args.seed,
    }
    selection_path = output_dir / "selected_model.json"
    selection_path.write_text(
        json.dumps(selection, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Selected model: {selection['weights']}")
    print(f"Experiment log: {log_path}")
    print(f"Selection summary: {selection_path}")


if __name__ == "__main__":
    main()
