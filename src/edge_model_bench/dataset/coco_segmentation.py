"""COCO instance-segmentation dataset using YOLO polygon labels."""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np


class CocoInstanceSegmentation:
    """Load image/polygon pairs from an ``image_path,label_path`` CSV."""

    def __init__(
        self,
        csv_path: str | Path,
        image_root: str | Path | None = None,
        label_root: str | Path | None = None,
        pipeline: list[Callable] | None = None,
    ) -> None:
        self.path = Path(csv_path)
        self.image_root = Path(image_root) if image_root else None
        self.label_root = Path(label_root) if label_root else None
        self.pipeline = pipeline or []
        with self.path.open(encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None or not {
                "image_path",
                "label_path",
            }.issubset(reader.fieldnames):
                raise ValueError(
                    "Instance segmentation CSV must contain image_path and "
                    f"label_path columns, got {reader.fieldnames}"
                )
            self.samples = [
                self._sample(row["image_path"], row["label_path"]) for row in reader
            ]

    def _resolve(self, path: str, root: Path | None) -> Path:
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = (root if root is not None else self.path.parent) / resolved
        return resolved

    def _parse_label(self, path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            raise FileNotFoundError(f"Segmentation label file not found: {path}")
        instances: list[dict[str, Any]] = []
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) < 7 or (len(parts) - 1) % 2:
                raise ValueError(
                    f"Invalid YOLO segmentation label at {path}:{line_number}: "
                    "expected class_id followed by at least 3 xy pairs"
                )
            class_id = int(parts[0])
            polygon = np.asarray(parts[1:], dtype=np.float32).reshape(-1, 2)
            if not np.isfinite(polygon).all() or np.any((polygon < 0) | (polygon > 1)):
                raise ValueError(
                    f"Invalid normalized polygon at {path}:{line_number}"
                )
            instances.append({"label": class_id, "polygon": polygon})
        return instances

    def _sample(self, image_path: str, label_path: str) -> dict[str, Any]:
        image = self._resolve(image_path, self.image_root)
        label = self._resolve(label_path, self.label_root)
        if not image.is_file():
            raise FileNotFoundError(f"Image file not found: {image}")
        return {
            "image_path": str(image),
            "label_path": str(label),
            "label": self._parse_label(label),
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __call__(self) -> list[dict[str, Any]]:
        return self.samples

    def __getitem__(self, index: int) -> tuple[np.ndarray, dict[str, Any]]:
        sample = self.samples[index]
        image: Any = sample["image_path"]
        for transform in self.pipeline:
            image = transform(image)
        return image, sample


CocoSegmentation = CocoInstanceSegmentation
