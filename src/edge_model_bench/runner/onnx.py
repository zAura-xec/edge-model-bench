"""ONNX Runtime model runner."""

from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np


class OnnxRunner:
    def __init__(
        self,
        model_path: str | Path,
        providers: list[str] | None = None,
        output_mapping: bool = False,
        required_provider: str | None = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ModuleNotFoundError as exc:
            raise RuntimeError("Install ONNX support with: pip install 'edge-model-bench[onnx]'") from exc
        if providers and "CUDAExecutionProvider" in providers and hasattr(
            ort, "preload_dlls"
        ):
            ort.preload_dlls(directory="")
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        if required_provider and required_provider not in self.session.get_providers():
            raise RuntimeError(
                f"Required ONNX Runtime provider {required_provider!r} is unavailable; "
                f"active providers: {self.session.get_providers()}"
            )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.output_mapping = output_mapping
        self.latencies: list[float] = []

    def __call__(self, image: np.ndarray) -> Any:
        start = perf_counter()
        inputs = image if isinstance(image, Mapping) else {self.input_name: image}
        result = self.session.run(None, inputs)
        self.latencies.append(perf_counter() - start)
        return dict(zip(self.output_names, result)) if self.output_mapping else result

    def close(self) -> None:
        self.session = None
