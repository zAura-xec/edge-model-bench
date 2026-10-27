"""Compose portable ONNX quantization with the optional MTC adapter."""

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from .mtnn import MtnnConvertConfig, MtnnPipeline, _plain_mapping
from .static_onnx import OnnxStaticQuantConfig, OnnxStaticQuantizer


class StaticOnnxMtnnPipeline:
    """Run ONNX Runtime static quantization, then import that QDQ ONNX in MTC."""

    def __init__(
        self,
        input_model: str | Path,
        static_model_path: str | Path,
        calibration_list: str | Path,
        mtnn_config: MtnnConvertConfig | Mapping[str, Any],
        # ``config`` may be inherited from a regular MtnnPipeline Hydra node.
        # Keep it accepted so a static pipeline can override that node cleanly.
        config: Any | None = None,
        input_name: str | None = None,
        max_calibration_samples: int | None = None,
        calibrate_method: str = "Entropy",
        activation_type: str = "QInt8",
        weight_type: str = "QInt8",
        quant_format: str = "QDQ",
        per_channel: bool = False,
        op_types_to_quantize: Sequence[str] | None = None,
        extra_options: Mapping[str, Any] | None = None,
        calib_chunk_size: int | None = None,
        auto_batch_dim: bool = True,
        use_external_data_format: bool = True,
    ) -> None:
        self.quantizer = OnnxStaticQuantizer(
            OnnxStaticQuantConfig(
                input_model=input_model,
                output_model=static_model_path,
                calibration_list=calibration_list,
                max_calibration_samples=max_calibration_samples,
                input_name=input_name,
                calibrate_method=calibrate_method,
                activation_type=activation_type,
                weight_type=weight_type,
                quant_format=quant_format,
                per_channel=per_channel,
                op_types_to_quantize=list(op_types_to_quantize)
                if op_types_to_quantize is not None
                else None,
                extra_options=dict(extra_options or {}),
                calib_chunk_size=calib_chunk_size,
                auto_batch_dim=auto_batch_dim,
                use_external_data_format=use_external_data_format,
            )
        )
        self._inherited_config = config
        if isinstance(mtnn_config, MtnnConvertConfig):
            self.mtnn_config = mtnn_config
            self.mtnn_options: dict[str, Any] | None = None
        else:
            self.mtnn_config = None
            self.mtnn_options = _plain_mapping(mtnn_config)
            # A plain Hydra node can retain its construction marker.
            self.mtnn_options.pop("_target_", None)

    def __call__(self) -> Path:
        static_model = self.quantizer()
        return self.convert_static(static_model)

    def convert_static(self, static_model: str | Path | None = None) -> Path:
        """Import an already-created QDQ ONNX model with MTC."""
        static_model = Path(static_model or self.quantizer.config.output_model)
        if not static_model.is_file():
            raise FileNotFoundError(f"静态量化 ONNX 不存在: {static_model}")
        if self.mtnn_config is not None:
            config = replace(
                self.mtnn_config,
                onnx_path=static_model,
                static_quantization=True,
            )
        else:
            assert self.mtnn_options is not None
            options = dict(self.mtnn_options)
            options.update(onnx_path=static_model, static_quantization=True)
            config = MtnnConvertConfig(**options)
        return MtnnPipeline(config)()
