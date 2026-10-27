"""ONNX to Moore Threads NPU ``.mtnn`` conversion via MTC.

MTC writes its intermediate and final artefacts below the toolkit's
``data/net_data/onnx/<Name>`` directory.  This module keeps the generated
YAML and log in the benchmark work directory, then copies the final ``.mtnn``
model back there so conversion and evaluation do not depend on a hard-coded
toolkit output path.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def _plain_mapping(value: Any) -> dict[str, Any]:
    """Convert an optional Hydra mapping to a regular dictionary."""
    if value is None:
        return {}
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, dict):
        raise TypeError(f"期望字典配置，实际为 {type(value).__name__}")
    return dict(value)


@dataclass
class MtnnConvertConfig:
    """Configuration for one ``mtc --config`` conversion.

    ``input_file_list`` is the MTC model-zoo style list of calibration
    ``.npy`` files.  ``input_dataset_list`` is also supported for toolkit
    versions/configurations that consume text files containing these paths.
    Only one of them is normally needed.
    """

    onnx_path: str | Path
    output_dir: str | Path
    model_name: str | None = None
    name: str | None = None

    # MTC ModelConfig
    input_file_list: list[str | Path] | str | Path | None = None
    input_file_dir: str | Path | None = None
    input_dataset_list: list[str | Path] | str | Path | None = None
    force_fp32input_list: bool | str = True
    force_fp32output_list: bool | str = True
    model_config: dict[str, Any] | None = None

    # MTC QuantConfig.  The defaults match the model-zoo recommended UINT8
    # style; task configs override them for FP16/INT8 benchmark variants.
    quant_type: str = "uint8"
    quantizer: str = "asymmetric_affine"
    quant_config: dict[str, Any] | None = None
    static_quantization: bool = False
    require_calibration: bool | None = None

    # Remaining MTC sections
    func_config: dict[str, Any] | None = None
    perf_collect_enable: bool = True
    test_enable: bool = True
    use_single_core: bool = True
    extra_config: dict[str, Any] | None = None

    # Invocation and artifact discovery
    mtc: str = "mtc"
    mtc_output_root: str | Path = "/opt/m1000-mtc-toolkit/bin/baselib/data/net_data/onnx"
    artifact_path: str | Path | None = None
    extra_args: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.onnx_path = Path(self.onnx_path)
        self.output_dir = Path(self.output_dir)
        self.model_name = self.model_name or self.onnx_path.stem
        self.name = self.name or self.model_name
        self.mtc_output_root = Path(self.mtc_output_root)
        if self.input_file_dir is not None:
            self.input_file_dir = Path(self.input_file_dir)
        if self.artifact_path is not None:
            self.artifact_path = Path(self.artifact_path)

    @property
    def config_path(self) -> Path:
        return self.output_dir / "mtc_config.yaml"

    @property
    def log_path(self) -> Path:
        return self.output_dir / "mtc.log"

    @property
    def output_path(self) -> Path:
        return self.artifact_path or self.output_dir / f"{self.name}.mtnn"

    @staticmethod
    def _path_list(value: list[str | Path] | str | Path | None) -> list[str]:
        if value is None:
            return []
        if isinstance(value, (str, Path)):
            return [str(Path(value).expanduser().resolve())]
        return [str(Path(path).expanduser().resolve()) for path in value]

    def to_dict(self) -> dict[str, Any]:
        """Build the documented MTC YAML structure."""
        model_config = _plain_mapping(self.model_config)
        model_config.setdefault("force_fp32input_list", self.force_fp32input_list)
        model_config.setdefault("force_fp32output_list", self.force_fp32output_list)
        input_files = self.resolve_input_files()
        if input_files:
            model_config.setdefault("input_file_list", [str(path) for path in input_files])
        if self.input_dataset_list is not None:
            model_config.setdefault(
                "input_dataset_list", self._path_list(self.input_dataset_list)
            )

        func_config = {
            "layer_data_analyze": False,
            "layer_data_compare": False,
            "mtnn_test_enable": False,
            **_plain_mapping(self.func_config),
        }
        if self.static_quantization:
            # A QDQ ONNX has already been calibrated by ONNX Runtime. MTC must
            # import its quantization parameters instead of quantizing again.
            quant_config = {
                "static_quantization": True,
                **_plain_mapping(self.quant_config),
            }
        else:
            quant_config = {
                "quant_type": self.quant_type,
                "quantizer": self.quantizer,
                **_plain_mapping(self.quant_config),
            }
        config: dict[str, Any] = {
            "FuncConfig": func_config,
            "ModelConfig": model_config,
            "ModelPath": str(self.onnx_path.resolve()),
            "Name": self.name,
            "PerfConfig": {"perf_collect_enable": self.perf_collect_enable},
            "QuantConfig": quant_config,
            "TestEnable": self.test_enable,
            "UseSingleCore": self.use_single_core,
        }
        config.update(_plain_mapping(self.extra_config))
        return config

    def resolve_input_files(self) -> list[Path]:
        """Return explicit MTC inputs plus sorted ``.npy`` files from a directory."""
        paths = [Path(path) for path in self._path_list(self.input_file_list)]
        if self.input_file_dir is not None:
            paths.extend(sorted(self.input_file_dir.expanduser().glob("*.npy")))
        # Preserve the user-supplied order while removing accidental duplicates.
        return list(dict.fromkeys(path.resolve() for path in paths))

    def save(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        logger.info("写入 MTC 配置: %s", self.config_path)
        return self.config_path

    def needs_calibration(self) -> bool:
        if self.static_quantization:
            return False
        if self.require_calibration is not None:
            return self.require_calibration
        return self.quant_type.lower() in {"int8", "uint8"}

    def validate(self) -> None:
        if not self.onnx_path.is_file():
            raise FileNotFoundError(f"ONNX 不存在: {self.onnx_path}")
        if not self.needs_calibration():
            return
        inputs = [str(path) for path in self.resolve_input_files()] + self._path_list(
            self.input_dataset_list
        )
        if not inputs:
            raise ValueError(
                "INT8/UINT8 转换需要 input_file_list 或 input_dataset_list 校准数据"
            )
        missing = [path for path in inputs if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"MTC 校准数据不存在: {missing}")

    def find_generated_artifact(self) -> Path:
        """Locate the final artifact in MTC's documented output hierarchy."""
        generated_dir = self.mtc_output_root / str(self.name) / "mtc" / "dump_npu_files"
        matches = sorted(
            generated_dir.glob("**/*.mtnn"),
            key=lambda path: path.stat().st_mtime_ns,
        )
        if not matches:
            raise FileNotFoundError(
                "MTC 已完成但未找到 .mtnn: "
                f"{generated_dir}（Name={self.name!r}）"
            )
        return matches[-1]


class MtnnPipeline:
    """Generate MTC YAML, invoke MTC, and return a stable model artifact."""

    def __init__(self, config: MtnnConvertConfig) -> None:
        self.config = config

    def __call__(self) -> Path:
        self.config.validate()
        config_path = self.config.save()
        self._run(config_path)

        source = self.config.find_generated_artifact()
        destination = self.config.output_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        logger.info("MTNN 转换完成: %s", destination)
        return destination

    def _run(self, config_path: Path) -> None:
        cmd = [self.config.mtc, "--config", str(config_path.resolve()), *self.config.extra_args]
        logger.info("执行: %s", " ".join(cmd))
        with self.config.log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.run(
                cmd,
                cwd=self.config.output_dir,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
        if process.returncode == 0:
            return
        tail = self.config.log_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()[-40:]
        raise RuntimeError(
            f"mtc 转换失败 (exit={process.returncode})，详见 {self.config.log_path}\n"
            + "\n".join(tail)
        )
