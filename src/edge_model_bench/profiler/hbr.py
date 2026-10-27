"""Concurrent latency profiling for HBR HBM models on RDK S boards."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np


LatencyStats = dict[str, float]


class HbrMultiModelProfiler:
    """Profile multiple HBM models submitted together through ``hbm_runtime``.

    A call to :meth:`profile` submits all models in one ``runtime.run`` call.
    Consequently, each recorded latency is the end-to-end makespan of the
    concurrent model batch, rather than the isolated latency of one model.

    Args:
        model_paths: Already converted HBM model files. Each HBM must contain
            exactly one model, and model names must be unique.
        priority: Scheduling priority applied to every model.
        bpu_cores: BPU cores available to every model.
        seed: Seed used to generate reproducible random inputs.
        runtime_factory: Runtime constructor override, primarily for tests.
    """

    def __init__(
        self,
        model_paths: Sequence[str | Path],
        priority: int = 0,
        bpu_cores: Sequence[int] | None = None,
        seed: int | None = None,
        runtime_factory: Callable[[Sequence[str]], Any] | None = None,
    ) -> None:
        paths = [str(Path(path)) for path in model_paths]
        if not paths:
            raise ValueError("model_paths 不能为空")

        if runtime_factory is None:
            try:
                import hbm_runtime
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "HbrMultiModelProfiler 只能在安装 hbm_runtime 的 RDK S 板端运行"
                ) from exc
            runtime_factory = hbm_runtime.HB_HBMRuntime

        self.runtime = runtime_factory(paths)
        self.model_paths = tuple(paths)
        self.model_names = tuple(self.runtime.model_names)
        if len(self.model_names) != len(paths):
            raise ValueError(
                "每个 HBM 必须且只能包含一个模型: "
                f"传入 {len(paths)} 个 HBM，加载到 {len(self.model_names)} 个模型 "
                f"{list(self.model_names)}"
            )
        if len(set(self.model_names)) != len(self.model_names):
            raise ValueError(f"模型名必须唯一: {list(self.model_names)}")

        cores = list(bpu_cores) if bpu_cores is not None else [0]
        if not cores:
            raise ValueError("bpu_cores 不能为空")
        self.runtime.set_scheduling_params(
            priority={name: priority for name in self.model_names},
            bpu_cores={name: cores for name in self.model_names},
        )
        self._rng = np.random.default_rng(seed)
        self.inputs = self._generate_inputs()
        self.latencies_ms: list[float] = []

    def _generate_inputs(self) -> dict[str, dict[str, np.ndarray]]:
        inputs: dict[str, dict[str, np.ndarray]] = {}
        for model_name in self.model_names:
            model_inputs = {}
            for input_name in self.runtime.input_names[model_name]:
                shape = tuple(self.runtime.input_shapes[model_name][input_name])
                if not shape or any(int(dimension) <= 0 for dimension in shape):
                    raise ValueError(
                        f"模型 {model_name!r} 输入 {input_name!r} 包含动态或非法 shape: {shape}"
                    )
                model_inputs[input_name] = np.ascontiguousarray(
                    self._rng.random(shape, dtype=np.float32)
                )
            inputs[model_name] = model_inputs
        return inputs

    @staticmethod
    def summarize(latencies_ms: Sequence[float]) -> LatencyStats:
        """Calculate the requested latency statistics in milliseconds."""
        if not latencies_ms:
            raise ValueError("没有可统计的延迟数据")
        values = np.asarray(latencies_ms, dtype=np.float64)
        return {
            "mean": float(np.mean(values)),
            "P50": float(np.percentile(values, 50)),
            "P90": float(np.percentile(values, 90)),
            "P95": float(np.percentile(values, 95)),
            "P99": float(np.percentile(values, 99)),
            "max": float(np.max(values)),
        }

    def profile(self, warmup: int = 5, runs: int = 100) -> LatencyStats:
        """Warm up, concurrently submit all models, and summarize latency."""
        if warmup < 0:
            raise ValueError("warmup 必须大于等于 0")
        if runs < 1:
            raise ValueError("runs 必须大于 0")

        for _ in range(warmup):
            self.runtime.run(self.inputs)

        self.latencies_ms = []
        for _ in range(runs):
            start = perf_counter()
            self.runtime.run(self.inputs)
            self.latencies_ms.append((perf_counter() - start) * 1000.0)
        return self.summarize(self.latencies_ms)

    def __call__(self, warmup: int = 5, runs: int = 100) -> LatencyStats:
        return self.profile(warmup=warmup, runs=runs)

    def close(self) -> None:
        self.runtime = None

    def __enter__(self) -> "HbrMultiModelProfiler":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
