"""ONNX to QAIRT HTP BIN conversion."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import qairt
from qairt.api.common.backends.htp import HtpGraphConfig
from qairt.api.converter.converter_config import CalibrationConfig
from qti.aisw.tools.core.utilities.devices.api.device_factory import DeviceFactory


@dataclass
class QnnConfig:
    onnx_path: str | Path
    output_dir: str | Path
    chipset: str
    model_name: str | None = None
    backend: str = "HTP"
    fp16_relaxed_precision: bool = False
    conversion_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.onnx_path = Path(self.onnx_path)
        self.output_dir = Path(self.output_dir)
        self.model_name = self.model_name or self.onnx_path.stem

    @property
    def dlc_path(self) -> Path:
        return self.output_dir / f"{self.model_name}_quantized.dlc"

    @property
    def bin_path(self) -> Path:
        return self.output_dir / f"{self.model_name}_compiled.bin"


class OnnxToDlc:
    def __init__(
        self,
        config: QnnConfig,
        calibration_config: CalibrationConfig | None = None,
    ) -> None:
        self.config = config
        self.calibration_config = calibration_config

    def __call__(self) -> Path:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        model = qairt.convert(
            str(self.config.onnx_path),
            calibration_config=self.calibration_config,
            **self.config.conversion_options,
        )
        model.save(str(self.config.dlc_path))
        return self.config.dlc_path


class DlcToBin:
    def __init__(self, config: QnnConfig) -> None:
        self.config = config

    def __call__(self, dlc_path: str | Path | None = None) -> Path:
        model = qairt.Model.load(str(dlc_path or self.config.dlc_path))
        details = DeviceFactory.get_device_soc_details(self.config.backend, self.config.chipset)
        graph_config = HtpGraphConfig(
            name=model.module.info.graphs[0].name,
            fp16_relaxed_precision=self.config.fp16_relaxed_precision,
            vtcm_size_in_mb=details.vtcm_size_in_mb,
            hvx_threads=details.num_of_hvx_threads,
        )
        compiled_model = qairt.compile(
            model,
            config=qairt.CompileConfig(
                backend=self.config.backend,
                soc_details=details,
                graph_custom_configs=[graph_config],
            ),
        )
        compiled_model.save(str(self.config.bin_path))
        return self.config.bin_path


class QnnPipeline:
    def __init__(self, onnx_to_dlc: OnnxToDlc, dlc_to_bin: DlcToBin) -> None:
        self.onnx_to_dlc = onnx_to_dlc
        self.dlc_to_bin = dlc_to_bin

    def __call__(self) -> Path:
        return self.dlc_to_bin(self.onnx_to_dlc())
