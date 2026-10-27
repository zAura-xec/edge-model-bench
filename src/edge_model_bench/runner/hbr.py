"""Horizon OE HBM model runner for RDK S-series boards."""

from collections.abc import Mapping
from pathlib import Path
from time import perf_counter

import numpy as np


class HbrRunner:
    """Execute a single-model HBM through ``hbm_runtime`` on an RDK S board."""

    def __init__(
        self,
        hbm_path: str | Path,
        priority: int = 0,
        bpu_cores: list[int] | None = None,
    ) -> None:
        try:
            import hbm_runtime
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "HbrRunner 只能在安装 hbm_runtime 的 RDK S 系列板端运行"
            ) from exc

        self.runtime = hbm_runtime.HB_HBMRuntime(str(hbm_path))
        if len(self.runtime.model_names) != 1:
            raise ValueError(f"仅支持单模型 HBM: {self.runtime.model_names}")

        self.model_name = self.runtime.model_names[0]
        self.input_names = tuple(self.runtime.input_names[self.model_name])
        self.input_shapes = {
            name: tuple(self.runtime.input_shapes[self.model_name][name])
            for name in self.input_names
        }
        self.latencies: list[float] = []
        self.runtime.set_scheduling_params(
            priority={self.model_name: priority},
            bpu_cores={self.model_name: bpu_cores or [0]},
        )

    def _prepare_tensor(self, name: str, value: np.ndarray) -> np.ndarray:
        tensor = np.asarray(value, dtype=np.float32)
        expected_shape = self.input_shapes[name]
        if tensor.shape == expected_shape[1:] and expected_shape[0] == 1:
            tensor = tensor[None, ...]
        if tensor.shape != expected_shape:
            raise ValueError(
                f"输入 {name!r} 的 shape 不匹配: 得到 {tensor.shape}，"
                f"模型要求 {expected_shape}"
            )
        return np.ascontiguousarray(tensor)

    def __call__(self, image: np.ndarray | Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        if isinstance(image, Mapping):
            provided_names = set(image)
            expected_names = set(self.input_names)
            if provided_names != expected_names:
                missing = sorted(expected_names - provided_names)
                extra = sorted(provided_names - expected_names)
                raise ValueError(f"HBM 输入名不匹配，缺少={missing}，额外={extra}")
            inputs = {
                name: self._prepare_tensor(name, image[name]) for name in self.input_names
            }
        else:
            if len(self.input_names) != 1:
                raise ValueError(
                    f"多输入 HBM 需要以 dict 传入全部输入: {list(self.input_names)}"
                )
            name = self.input_names[0]
            inputs = {name: self._prepare_tensor(name, image)}

        start = perf_counter()
        result = self.runtime.run({self.model_name: inputs})
        self.latencies.append(perf_counter() - start)
        return result[self.model_name]

    def close(self) -> None:
        self.runtime = None
