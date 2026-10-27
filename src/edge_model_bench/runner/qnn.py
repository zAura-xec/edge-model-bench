"""QAIRT BIN model runner."""

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import qairt


class QnnRunner:
    def __init__(
        self,
        bin_path: str | Path,
        backend: str = "HTP",
        initialize: bool = False,
        log_level: str = "error",
    ) -> None:
        self.backend = backend
        self.log_level = log_level
        self.latencies: list[float] = []
        self.model = qairt.load(str(bin_path))
        if initialize:
            self.model.initialize(backend=backend, log_level=log_level)

    def __call__(self, image: np.ndarray) -> Any:
        start = perf_counter()
        result = self.model(inputs=image, backend=self.backend, log_level=self.log_level)
        self.latencies.append(perf_counter() - start)
        return result

    def close(self) -> None:
        self.model.destroy()
