"""Pascal VOC semantic-segmentation dataset."""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


VOC_CLASSES: tuple[str, ...] = (
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)


class VocSegmentation:
    """Load Pascal VOC images and palette-index semantic masks.

    For full splits, ``root`` may point either to ``VOCdevkit`` or directly to
    ``VOC2012``. For benchmark subsets, ``csv_path`` accepts the same
    ``image_path,label_path`` manifest convention as the COCO detector dataset.
    Palette indices in ``SegmentationClass/*.png`` are preserved, including
    VOC's void value 255.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        split: str = "val",
        year: str = "2012",
        pipeline: list[Callable] | None = None,
        csv_path: str | Path | None = None,
        image_root: str | Path | None = None,
        label_root: str | Path | None = None,
    ) -> None:
        self.pipeline = pipeline or []
        self.path = Path(csv_path) if csv_path is not None else None

        if self.path is not None:
            self.root = None
            resolved_image_root = Path(image_root) if image_root is not None else None
            resolved_label_root = Path(label_root) if label_root is not None else None
            with self.path.open(encoding="utf-8-sig") as file:
                reader = csv.DictReader(file)
                if reader.fieldnames is None or not {
                    "image_path",
                    "label_path",
                }.issubset(reader.fieldnames):
                    raise ValueError(
                        "VOC segmentation CSV must contain image_path and label_path "
                        f"columns, got {reader.fieldnames}"
                    )
                self.samples = [
                    self._make_sample_from_paths(
                        self._resolve_path(row["image_path"], resolved_image_root),
                        self._resolve_path(row["label_path"], resolved_label_root),
                    )
                    for row in reader
                ]
            return

        if root is None:
            raise ValueError("Either root or csv_path must be provided")
        root_path = Path(root)
        expected_name = f"VOC{year}"
        self.root = (
            root_path if root_path.name == expected_name else root_path / expected_name
        )

        split_path = self.root / "ImageSets" / "Segmentation" / f"{split}.txt"
        if not split_path.is_file():
            raise FileNotFoundError(f"VOC segmentation split not found: {split_path}")

        image_ids = [
            line.strip()
            for line in split_path.read_text().splitlines()
            if line.strip()
        ]
        self.samples = [self._make_sample(image_id) for image_id in image_ids]

    def _resolve_path(self, path: str | Path, root: Path | None) -> Path:
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = (root if root is not None else self.path.parent) / resolved
        return resolved

    def _make_sample(self, image_id: str) -> dict[str, Any]:
        image_path = self.root / "JPEGImages" / f"{image_id}.jpg"
        label_path = self.root / "SegmentationClass" / f"{image_id}.png"
        return self._make_sample_from_paths(image_path, label_path, image_id=image_id)

    def _make_sample_from_paths(
        self,
        image_path: Path,
        label_path: Path,
        image_id: str | None = None,
    ) -> dict[str, Any]:
        if not image_path.is_file():
            raise FileNotFoundError(f"VOC image not found: {image_path}")
        if not label_path.is_file():
            raise FileNotFoundError(f"VOC segmentation mask not found: {label_path}")
        return {
            "id": image_id or image_path.stem,
            "image_path": str(image_path),
            "label_path": str(label_path),
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[np.ndarray, dict[str, Any]]:
        metadata = self.samples[index]
        image: Any = metadata["image_path"]
        for transform in self.pipeline:
            image = transform(image)

        # PIL exposes the actual indices of VOC's paletted PNG. OpenCV expands
        # the palette to BGR values and therefore cannot be used here.
        with Image.open(metadata["label_path"]) as label_image:
            label = np.asarray(label_image, dtype=np.uint8).copy()
        if label.ndim != 2:
            raise ValueError(
                f"VOC mask must be a 2-D palette-index image, got {label.shape}: "
                f"{metadata['label_path']}"
            )

        sample = dict(metadata)
        sample["label"] = label
        sample["original_size"] = label.shape
        return image, sample


# Conventional spelling used by torchvision and other dataset libraries.
VOCSegmentation = VocSegmentation
