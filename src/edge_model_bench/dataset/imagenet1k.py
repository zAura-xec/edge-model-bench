"""ImageNet-1K dataset."""

import csv
from collections.abc import Callable
from pathlib import Path

import numpy as np


class ImageNet1K:
    """Load labels from CSV, or infer them from a TXT list and synset mapping."""

    def __init__(
        self,
        csv_path: str | Path,
        image_root: str | Path | None = None,
        pipeline: list[Callable] | None = None,
        class_mapping_path: str | Path | None = None,
    ) -> None:
        self.path = Path(csv_path)
        self.image_root = Path(image_root) if image_root else None
        self.pipeline = pipeline or []

        if class_mapping_path:
            with Path(class_mapping_path).open(encoding="utf-8-sig") as file:
                labels = {
                    row["synset"]: int(row["index"])
                    for row in csv.DictReader(file, delimiter="\t")
                }
            paths = [line.strip() for line in self.path.read_text().splitlines() if line.strip()]
            self.samples = [self._sample(path, labels[Path(path).parent.name]) for path in paths]
        else:
            with self.path.open(encoding="utf-8-sig") as file:
                self.samples = [
                    self._sample(row["image_path"], int(row["label"]))
                    for row in csv.DictReader(file)
                ]

    def _sample(self, image_path: str, label: int) -> dict[str, str | int]:
        path = Path(image_path)
        if self.image_root and not path.is_absolute():
            path = self.image_root / path
        return {"image_path": str(path), "label": label}

    def __len__(self) -> int:
        return len(self.samples)

    def __call__(self) -> list[dict[str, str | int]]:
        return self.samples

    def __getitem__(self, index: int) -> tuple[np.ndarray, dict[str, str | int]]:
        sample = self.samples[index]
        image = sample["image_path"]
        for transform in self.pipeline:
            image = transform(image)
        return image, sample
