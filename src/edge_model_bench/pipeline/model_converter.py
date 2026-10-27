"""Reusable calibration preparation and optional model conversion."""

from collections.abc import Callable
from typing import Any

from tqdm.auto import tqdm


def prepare_calibration(dataset: Any) -> int:
    count = len(dataset)
    if count == 0:
        raise ValueError("Calibration dataset is empty")
    for index in tqdm(range(count), desc="Preparing calibration data", unit="sample"):
        dataset[index]
    return count


class ModelConverter:
    def __init__(self, converter: Callable, calibration_dataset: Any | None = None) -> None:
        self.converter = converter
        self.calibration_dataset = calibration_dataset

    def __call__(self) -> Any:
        if self.calibration_dataset is not None:
            prepare_calibration(self.calibration_dataset)
        return self.converter()
