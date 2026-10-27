"""Standalone ONNX Runtime static quantization.

Supports chunked ``collect_data()`` for Entropy (and other histogram-based)
calibration so intermediate activation tensors never need to be held for the
full calibration set at once.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class OnnxStaticQuantConfig:
    """Configuration for ONNX Runtime QDQ static quantization."""

    input_model: str | Path
    output_model: str | Path
    calibration_list: str | Path
    max_calibration_samples: int | None = None
    input_name: str | None = None
    calibrate_method: str = "Entropy"
    activation_type: str = "QInt8"
    weight_type: str = "QInt8"
    quant_format: str = "QDQ"
    per_channel: bool = False
    op_types_to_quantize: list[str] | None = None
    extra_options: dict[str, Any] = field(default_factory=dict)
    # When set, calibration runs ``collect_data()`` on successive slices of the
    # sample list.  Each call merges activations into compact histograms (for
    # Entropy / Percentile) or running ranges (for MinMax) and then frees the
    # intermediate tensors, which greatly reduces peak memory.
    calib_chunk_size: int | None = None
    # If True, expand CHW ``.npy`` tensors to NCHW when the model input rank is 4.
    auto_batch_dim: bool = True
    use_external_data_format: bool = True

    def __post_init__(self) -> None:
        self.input_model = Path(self.input_model)
        self.output_model = Path(self.output_model)
        self.calibration_list = Path(self.calibration_list)
        if self.calib_chunk_size is not None and self.calib_chunk_size <= 0:
            raise ValueError("calib_chunk_size must be a positive integer or None")


class NpyCalibrationDataReader:
    """ONNX Runtime calibration reader backed by a text file of ``.npy`` tensors.

    Implements ``__len__`` / ``set_range`` so callers can drive chunked
    ``collect_data()`` for memory-efficient Entropy calibration.
    """

    def __init__(
        self,
        calibration_list: Path,
        input_name: str,
        max_samples: int | None = None,
        auto_batch_dim: bool = True,
        expected_rank: int | None = None,
    ) -> None:
        self.paths = [
            Path(line.strip())
            for line in calibration_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not self.paths:
            raise ValueError(f"ONNX 校准清单为空: {calibration_list}")
        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError("max_calibration_samples 必须为正数")
            self.paths = self.paths[:max_samples]
        missing = [path for path in self.paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"ONNX 校准输入不存在: {missing[:3]}")
        self.input_name = input_name
        self.auto_batch_dim = auto_batch_dim
        self.expected_rank = expected_rank
        self._start = 0
        self._end = len(self.paths)
        self._iterator = iter(self.paths[self._start : self._end])

    def __len__(self) -> int:
        return len(self.paths)

    def set_range(self, start_index: int, end_index: int) -> None:
        if start_index < 0 or end_index > len(self.paths) or start_index >= end_index:
            raise ValueError(
                f"非法校准区间 [{start_index}, {end_index})，样本数={len(self.paths)}"
            )
        self._start = start_index
        self._end = end_index
        self._iterator = iter(self.paths[self._start : self._end])

    def rewind(self) -> None:
        self._iterator = iter(self.paths[self._start : self._end])

    def _prepare_array(self, path: Path) -> np.ndarray:
        array = np.ascontiguousarray(np.load(path), dtype=np.float32)
        if not self.auto_batch_dim:
            return array
        # Horizon OE dumps CHW without a batch axis; ORT models usually expect NCHW.
        if self.expected_rank is not None:
            while array.ndim < self.expected_rank:
                array = array[None]
            return np.ascontiguousarray(array, dtype=np.float32)
        if array.ndim == 3:
            return np.ascontiguousarray(array[None], dtype=np.float32)
        return array

    def get_next(self) -> dict[str, np.ndarray] | None:
        path = next(self._iterator, None)
        if path is None:
            return None
        return {self.input_name: self._prepare_array(path)}


class OnnxStaticQuantizer:
    """Generate a QDQ ONNX model with ONNX Runtime static calibration."""

    def __init__(self, config: OnnxStaticQuantConfig) -> None:
        self.config = config

    @staticmethod
    def _enum_member(enum: Any, name: str) -> Any:
        normalized = name.replace("_", "").lower()
        for member in enum:
            if member.name.replace("_", "").lower() == normalized:
                return member
        raise ValueError(f"不支持的 {enum.__name__} 值: {name!r}")

    def _input_meta(self) -> tuple[str, int | None]:
        """Return ``(input_name, rank)`` for the single non-initializer graph input."""
        if self.config.input_name:
            input_name = self.config.input_name
            rank: int | None = None
            try:
                import onnx

                model = onnx.load(self.config.input_model, load_external_data=False)
                for item in model.graph.input:
                    if item.name == input_name:
                        rank = len(item.type.tensor_type.shape.dim) or None
                        break
            except Exception:  # pragma: no cover - best-effort rank probe
                rank = None
            return input_name, rank

        import onnx

        model = onnx.load(self.config.input_model, load_external_data=False)
        initializers = {initializer.name for initializer in model.graph.initializer}
        inputs = [item for item in model.graph.input if item.name not in initializers]
        if len(inputs) != 1:
            raise ValueError(
                "静态 ONNX 量化需要显式 input_name（模型输入数为 "
                f"{len(inputs)}）"
            )
        rank = len(inputs[0].type.tensor_type.shape.dim) or None
        return inputs[0].name, rank

    def _build_reader(self) -> NpyCalibrationDataReader:
        input_name, rank = self._input_meta()
        return NpyCalibrationDataReader(
            self.config.calibration_list,
            input_name,
            self.config.max_calibration_samples,
            auto_batch_dim=self.config.auto_batch_dim,
            expected_rank=rank,
        )

    def _quantize_with_chunks(self, reader: NpyCalibrationDataReader) -> None:
        """Run static quantization with chunked ``collect_data()`` calls.

        Entropy / Percentile calibrators keep a compact histogram across
        chunks and free per-sample intermediate activations after every
        ``collect_data()`` call, so peak memory scales with chunk size rather
        than the full calibration set.
        """
        from onnxruntime.quantization import (
            CalibrationMethod,
            QuantFormat,
            QuantType,
            create_calibrator,
        )
        from onnxruntime.quantization.onnx_quantizer import ONNXQuantizer
        from onnxruntime.quantization.qdq_quantizer import QDQQuantizer
        from onnxruntime.quantization.quantize import check_static_quant_arguments
        from onnxruntime.quantization.quant_utils import (
            QuantizationMode,
            load_model_with_shape_infer,
            model_has_pre_process_metadata,
        )
        from onnxruntime.quantization.registry import QDQRegistry, QLinearOpsRegistry

        calibrate_method = self._enum_member(
            CalibrationMethod, self.config.calibrate_method
        )
        quant_format = self._enum_member(QuantFormat, self.config.quant_format)
        activation_type = self._enum_member(QuantType, self.config.activation_type)
        weight_type = self._enum_member(QuantType, self.config.weight_type)
        extra_options = dict(self.config.extra_options)
        op_types_to_quantize = list(self.config.op_types_to_quantize or [])
        if not op_types_to_quantize:
            op_types_to_quantize = list(
                set(list(QLinearOpsRegistry.keys()) + list(QDQRegistry.keys()))
            )

        model = load_model_with_shape_infer(self.config.input_model)
        if not model_has_pre_process_metadata(model):
            logger.warning(
                "模型缺少 quant pre-process 元数据；如量化失败可先运行 "
                "onnxruntime.quantization.quant_pre_process"
            )

        calib_extra_options_keys = [
            ("CalibTensorRangeSymmetric", "symmetric"),
            ("CalibMovingAverage", "moving_average"),
            ("CalibMovingAverageConstant", "averaging_constant"),
            ("CalibMaxIntermediateOutputs", "max_intermediate_outputs"),
            ("num_bins", "num_bins"),
            ("num_quantized_bins", "num_quantized_bins"),
            ("percentile", "percentile"),
        ]
        calib_extra_options = {
            key: extra_options[name]
            for name, key in calib_extra_options_keys
            if name in extra_options
        }

        chunk_size = self.config.calib_chunk_size
        total = len(reader)
        if chunk_size is None or chunk_size >= total:
            chunks = [(0, total)]
        else:
            chunks = [
                (start, min(start + chunk_size, total))
                for start in range(0, total, chunk_size)
            ]

        with tempfile.TemporaryDirectory(prefix="ort.quant.chunk.") as quant_tmp_dir:
            augmented_model_path = str(Path(quant_tmp_dir) / "augmented_model.onnx")
            calibrator = create_calibrator(
                self.config.input_model,
                op_types_to_quantize,
                augmented_model_path=augmented_model_path,
                calibrate_method=calibrate_method,
                use_external_data_format=self.config.use_external_data_format,
                extra_options=calib_extra_options,
            )
            logger.info(
                "分块校准: method=%s samples=%d chunks=%d chunk_size=%s",
                self.config.calibrate_method,
                total,
                len(chunks),
                chunk_size,
            )
            for index, (start, end) in enumerate(chunks, start=1):
                reader.set_range(start_index=start, end_index=end)
                logger.info(
                    "collect_data chunk %d/%d  samples[%d, %d)",
                    index,
                    len(chunks),
                    start,
                    end,
                )
                calibrator.collect_data(reader)
            tensors_range = calibrator.compute_data()
            del calibrator

        check_static_quant_arguments(quant_format, activation_type, weight_type)
        if quant_format is QuantFormat.QOperator:
            quantizer = ONNXQuantizer(
                model,
                self.config.per_channel,
                False,  # reduce_range
                QuantizationMode.QLinearOps,
                True,  # static
                weight_type,
                activation_type,
                tensors_range,
                None,  # nodes_to_quantize
                [],  # nodes_to_exclude
                op_types_to_quantize,
                extra_options,
            )
        else:
            quantizer = QDQQuantizer(
                model,
                self.config.per_channel,
                False,  # reduce_range
                weight_type,
                activation_type,
                tensors_range,
                None,  # nodes_to_quantize
                [],  # nodes_to_exclude
                op_types_to_quantize,
                extra_options,
            )
        quantizer.quantize_model()
        quantizer.model.save_model_to_file(
            str(self.config.output_model),
            self.config.use_external_data_format,
        )

    def __call__(self) -> Path:
        if not self.config.input_model.is_file():
            raise FileNotFoundError(f"ONNX 不存在: {self.config.input_model}")
        if not self.config.calibration_list.is_file():
            raise FileNotFoundError(f"ONNX 校准清单不存在: {self.config.calibration_list}")

        try:
            from onnxruntime.quantization import (
                CalibrationMethod,
                QuantFormat,
                QuantType,
                quantize_static,
            )
        except ModuleNotFoundError as exc:  # pragma: no cover - board dependency
            raise RuntimeError("ONNX 静态量化需要安装 onnxruntime") from exc

        self.config.output_model.parent.mkdir(parents=True, exist_ok=True)
        reader = self._build_reader()
        logger.info(
            "ONNX Runtime 静态量化: %s -> %s (%s, %d samples, chunk_size=%s)",
            self.config.input_model,
            self.config.output_model,
            self.config.calibrate_method,
            len(reader),
            self.config.calib_chunk_size,
        )

        # Always use the chunked path when chunk size is set, or for Entropy /
        # Percentile so intermediate activations are merged incrementally even
        # with a single logical chunk (same API, same code path).
        method = self.config.calibrate_method.replace("_", "").lower()
        use_chunked = self.config.calib_chunk_size is not None or method in {
            "entropy",
            "percentile",
            "distribution",
        }
        if use_chunked:
            self._quantize_with_chunks(reader)
        else:
            quantize_static(
                model_input=str(self.config.input_model),
                model_output=str(self.config.output_model),
                calibration_data_reader=reader,
                quant_format=self._enum_member(QuantFormat, self.config.quant_format),
                activation_type=self._enum_member(QuantType, self.config.activation_type),
                weight_type=self._enum_member(QuantType, self.config.weight_type),
                per_channel=self.config.per_channel,
                op_types_to_quantize=self.config.op_types_to_quantize,
                calibrate_method=self._enum_member(
                    CalibrationMethod, self.config.calibrate_method
                ),
                use_external_data_format=self.config.use_external_data_format,
                extra_options=dict(self.config.extra_options),
            )

        if not self.config.output_model.is_file():
            raise FileNotFoundError(f"ONNX 静态量化未生成模型: {self.config.output_model}")
        return self.config.output_model
