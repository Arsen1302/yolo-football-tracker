from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_DIRECTORIES = {"train": "train", "val": "valid", "test": "test"}


def load_dataset_spec(dataset_root: str | Path) -> tuple[Path, list[str], dict[str, Path]]:
    root = Path(dataset_root).expanduser().resolve()
    yaml_path = root / "data.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(f"YOLO dataset config not found: {yaml_path}")

    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or "names" not in config:
        raise ValueError(f"Expected a YOLO data.yaml with a 'names' field: {yaml_path}")

    names = config["names"]
    if isinstance(names, dict):
        names = [
            name
            for _, name in sorted(names.items(), key=lambda item: int(item[0]))
        ]
    if not isinstance(names, list) or not names or not all(
        isinstance(name, str) and name.strip() for name in names
    ):
        raise ValueError(f"Invalid class names in {yaml_path}")
    if len(set(names)) != len(names):
        raise ValueError(f"Class names must be unique in {yaml_path}")

    split_paths = {
        split: root / directory / "images"
        for split, directory in SPLIT_DIRECTORIES.items()
    }
    for split, images_path in split_paths.items():
        if not images_path.is_dir():
            raise FileNotFoundError(
                f"Missing {split} images directory: {images_path}"
            )
        if not any(
            path.suffix.lower() in IMAGE_SUFFIXES for path in images_path.iterdir()
        ):
            raise ValueError(f"No images found for {split} split: {images_path}")

        labels_path = images_path.parent / "labels"
        if not labels_path.is_dir():
            raise FileNotFoundError(
                f"Missing {split} labels directory: {labels_path}"
            )

    return root, names, split_paths


def create_ultralytics_data_yaml(
    dataset_root: str | Path,
    output_path: str | Path,
) -> Path:
    root, names, _ = load_dataset_spec(dataset_root)
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "path": str(root),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(names),
        "names": names,
    }
    target.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return target


def scan_yolo_dataset(
    dataset_root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    root, names, split_paths = load_dataset_spec(dataset_root)
    annotation_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []

    for split, images_path in split_paths.items():
        labels_path = images_path.parent / "labels"
        images = sorted(
            path
            for path in images_path.iterdir()
            if path.suffix.lower() in IMAGE_SUFFIXES
        )
        image_stems = {path.stem for path in images}
        label_files = list(labels_path.glob("*.txt"))
        missing_label_files = 0
        empty_label_files = 0
        invalid_rows = 0
        valid_boxes = 0

        for image_path in images:
            label_path = labels_path / f"{image_path.stem}.txt"
            if not label_path.is_file():
                missing_label_files += 1
                continue

            lines = [
                line.split()
                for line in label_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if not lines:
                empty_label_files += 1
                continue

            for line_number, parts in enumerate(lines, start=1):
                if len(parts) != 5:
                    invalid_rows += 1
                    continue
                try:
                    class_id = int(parts[0])
                    center_x, center_y, width, height = map(float, parts[1:])
                except ValueError:
                    invalid_rows += 1
                    continue

                normalized = (center_x, center_y, width, height)
                if (
                    class_id < 0
                    or class_id >= len(names)
                    or not all(math.isfinite(value) for value in normalized)
                    or not all(0 <= value <= 1 for value in normalized)
                    or width <= 0
                    or height <= 0
                ):
                    invalid_rows += 1
                    continue

                area = width * height
                annotation_rows.append(
                    {
                        "split": split,
                        "image": image_path.name,
                        "label_line": line_number,
                        "class_id": class_id,
                        "class_name": names[class_id],
                        "center_x": center_x,
                        "center_y": center_y,
                        "width": width,
                        "height": height,
                        "area": area,
                        "aspect_ratio": width / height,
                        "vertical_zone": (
                            "upper" if center_y < 1 / 3
                            else "middle" if center_y < 2 / 3
                            else "lower"
                        ),
                        "relative_scale": (
                            "small" if area < 0.0005
                            else "medium" if area < 0.0015
                            else "large"
                        ),
                    }
                )
                valid_boxes += 1

        orphan_label_files = sum(
            label_path.stem not in image_stems for label_path in label_files
        )
        audit_rows.append(
            {
                "split": split,
                "images": len(images),
                "label_files": len(label_files),
                "missing_label_files": missing_label_files,
                "empty_label_files": empty_label_files,
                "orphan_label_files": orphan_label_files,
                "invalid_rows": invalid_rows,
                "valid_boxes": valid_boxes,
            }
        )

    annotations = pd.DataFrame(annotation_rows)
    audit = pd.DataFrame(audit_rows)
    if annotations.empty:
        raise ValueError(f"No valid YOLO boxes found under {root}")
    return annotations, audit, names
