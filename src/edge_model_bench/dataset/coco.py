"""COCO detection dataset loaded from an image/label CSV."""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np


class Coco:
    """Load image/label pairs from CSV and parse YOLO labels.

    CSV columns:
        image_path,label_path

    Each YOLO label line:
        class_id x_center y_center width height

    is converted to:
        {"label": class_id, "box": [x_center, y_center, width, height]}
    """

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
            self.samples = [
                self._sample(row["image_path"], row["label_path"])
                for row in csv.DictReader(file)
            ]

    def _resolve_path(self, path: str | Path, root: Path | None) -> Path:
        resolved = Path(path)
        if root and not resolved.is_absolute():
            resolved = root / resolved
        return resolved

    def _parse_label(self, label_path: Path) -> list[dict[str, Any]]:
        if not label_path.is_file():
            raise FileNotFoundError(f"Label file not found: {label_path}")

        annotations: list[dict[str, Any]] = []
        text = label_path.read_text(encoding="utf-8").strip()
        if not text:
            return annotations

        for line_number, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(
                    f"Invalid YOLO label at {label_path}:{line_number}: expected "
                    f"5 fields, got {len(parts)} ({line!r})"
                )
            class_id, x_center, y_center, width, height = parts
            annotations.append(
                {
                    "label": int(class_id),
                    "box": [
                        float(x_center),
                        float(y_center),
                        float(width),
                        float(height),
                    ],
                }
            )
        return annotations

    def _sample(self, image_path: str, label_path: str) -> dict[str, Any]:
        image = self._resolve_path(image_path, self.image_root)
        label = self._resolve_path(label_path, self.label_root)
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
        image = sample["image_path"]
        for transform in self.pipeline:
            image = transform(image)
        return image, sample
