"""Run any compatible backend through the same evaluation loop."""

from collections.abc import Callable
from copy import deepcopy
from math import isfinite
from typing import Any

import numpy as np
from tqdm.auto import tqdm


class ModelEvaluator:
    """Runners record inference durations in seconds and provide close()."""

    def __init__(
        self,
        model: Any,
        dataset: Any,
        runner: Callable,
        postprocessor: Callable,
        evaluator: Callable,
        warmup: int = 0,
    ) -> None:
        if warmup < 0:
            raise ValueError("warmup must be non-negative")
        self.model = model
        self.dataset = dataset
        self.runner = runner
        self.postprocessor = postprocessor
        self.evaluator = evaluator
        self.warmup = warmup

    def __call__(self) -> dict[str, Any]:
        if len(self.dataset) == 0:
            raise ValueError("Evaluation dataset is empty")
        runner = self.runner(self.model)
        samples = []
        latencies = []
        try:
            if self.warmup:
                image, _ = self.dataset[0]
                for _ in range(self.warmup):
                    runner(image)
            start = len(runner.latencies)
            for index in tqdm(range(len(self.dataset)), desc="Running inference", unit="sample"):
                image, sample = self.dataset[index]
                samples.append(self.postprocessor(deepcopy(sample), runner(image)))
            # Snapshot before closing: a backend can free its timing buffers.
            latencies = list(runner.latencies[start:])
        finally:
            runner.close()

        result = self.evaluator(samples)
        if latencies:
            if not all(isfinite(value) and value > 0 for value in latencies):
                raise ValueError("Runner latencies must be finite, positive seconds")
            milliseconds = np.asarray(latencies) * 1000
            result.update(
                average_latency_ms=float(milliseconds.mean()),
                p50_latency_ms=float(np.percentile(milliseconds, 50)),
                p90_latency_ms=float(np.percentile(milliseconds, 90)),
                fps=float(1000 / milliseconds.mean()),
                num_inferences=len(latencies),
            )
        return result
