"""Pipeline wrapper for model latency profilers."""

from __future__ import annotations

from typing import Any


class ModelProfiler:
    """Run a profiler and return a JSON-friendly benchmark result."""

    def __init__(self, profiler: Any, warmup: int = 5, runs: int = 100) -> None:
        self.profiler = profiler
        self.warmup = warmup
        self.runs = runs

    def __call__(self) -> dict[str, Any]:
        model_names = list(self.profiler.model_names)
        try:
            statistics = self.profiler.profile(warmup=self.warmup, runs=self.runs)
        finally:
            self.profiler.close()
        return {
            "models": model_names,
            "warmup": self.warmup,
            "runs": self.runs,
            "latency_unit": "ms",
            **statistics,
        }
