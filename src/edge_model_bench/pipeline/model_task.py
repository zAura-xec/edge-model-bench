"""Compose reusable calibration, conversion, and evaluation stages."""

from collections.abc import Callable
from typing import Any

from .model_converter import ModelConverter
from .model_evaluator import ModelEvaluator


class ModelTask:
    def __init__(
        self,
        dataset: Any,
        runner: Callable,
        postprocessor: Callable,
        evaluator: Callable,
        model: Any = None,
        converter: Callable | None = None,
        calibration_dataset: Any | None = None,
        warmup: int = 0,
    ) -> None:
        self.dataset = dataset
        self.runner = runner
        self.postprocessor = postprocessor
        self.evaluator = evaluator
        self.model = model
        self.converter = converter
        self.calibration_dataset = calibration_dataset
        self.warmup = warmup

    def __call__(self) -> dict[str, Any]:
        model = self.model
        if self.converter is not None:
            model = ModelConverter(self.converter, self.calibration_dataset)()
        elif self.calibration_dataset is not None:
            raise ValueError("calibration_dataset requires a converter")
        return ModelEvaluator(
            model, self.dataset, self.runner, self.postprocessor, self.evaluator, warmup=self.warmup
        )()
