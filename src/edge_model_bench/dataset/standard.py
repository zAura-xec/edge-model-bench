"""Dataset preprocessing pipeline."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm


class StandardDataset:
    def __init__(
        self,
        images: str | Path,
        pipeline: list[Callable],
        max_samples: int | None = None,
    ) -> None:
        self.image_list = Path(images)
        self.pipeline = pipeline
        self.images = [
            line.strip()
            for line in self.image_list.read_text().splitlines()
            if line.strip()
        ]
        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError("max_samples must be positive")
            self.images = self.images[:max_samples]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> np.ndarray:
        image_path = Path(self.images[index])
        if not image_path.is_absolute():
            image_path = self.image_list.parent / image_path
        image = image_path
        for transform in self.pipeline:
            image = transform(image)
        return image

    def __call__(self) -> list[np.ndarray]:
        return [
            self[index]
            for index in tqdm(range(len(self)), desc="Processing dataset", unit="image")
        ]
