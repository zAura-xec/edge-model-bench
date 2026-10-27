"""Horizon OE 模型转换（hb_compile）。

流程：Hydra 关键词 → 量化配置 / 转换 YAML → 调用 hb_compile。

官方配置结构见:
https://toolchain.d-robotics.cc/guide/ptq/ptq_tool/hb_compile/convert.html
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import onnx
import yaml
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


@dataclass
class HbrQuantConfig:
    """量化配置类：生成 hb_compile 的 quant_config。

    两种用法：
    1. 直接传完整字典 quant_config
    2. 只设 all_node_type（如 float16 / int8 / int16）自动包一层 model_config
    """

    quant_config: dict[str, Any] | None = None
    all_node_type: str | None = None

    def to_dict(self) -> dict[str, Any] | None:
        """生成 quant_config 字典；无配置时返回 None。"""
        if self.quant_config is not None:
            return dict(self.quant_config)
        if self.all_node_type is None:
            return None
        return {"model_config": {"all_node_type": self.all_node_type}}

    def save(self, path: str | Path) -> Path | None:
        """写入 quant_config.json；无内容则跳过。"""
        data = self.to_dict()
        if data is None:
            return None
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        logger.info("写入量化配置: %s", path)
        return path


@dataclass
class HbrOnnxOpsetDowngrader:
    """将 ONNX 标准域 opset 降至 HBR 可接受的版本。

    降级结果写入单独的文件，绝不修改源模型。该操作只适用于目标
    opset 的算子定义仍兼容源模型的场景；转换后会检查全部节点的
    schema 并运行 ONNX 结构校验。
    """

    onnx_path: str | Path
    output_path: str | Path
    target_opset: int

    def __post_init__(self) -> None:
        self.onnx_path = Path(self.onnx_path)
        self.output_path = Path(self.output_path)

    def __call__(self) -> Path:
        if self.target_opset < 1:
            raise ValueError(f"target_opset 必须大于 0: {self.target_opset}")
        if not self.onnx_path.is_file():
            raise FileNotFoundError(f"ONNX 不存在: {self.onnx_path}")
        if self.output_path.resolve() == self.onnx_path.resolve():
            raise ValueError("opset 降级输出路径不能覆盖源 ONNX")

        model = onnx.load(self.onnx_path)
        standard_opsets = [
            opset
            for opset in model.opset_import
            if opset.domain in {"", "ai.onnx"}
        ]
        if len(standard_opsets) != 1:
            raise ValueError("ONNX 必须且只能包含一个标准 ai.onnx opset 声明")

        source_opset = standard_opsets[0].version
        if source_opset <= self.target_opset:
            logger.info(
                "ONNX opset 已满足要求: %s <= %s，复用源模型: %s",
                source_opset,
                self.target_opset,
                self.onnx_path,
            )
            return self.onnx_path

        for node in model.graph.node:
            if node.domain not in {"", "ai.onnx"}:
                continue
            domain = node.domain or ""
            try:
                onnx.defs.get_schema(node.op_type, self.target_opset, domain)
            except onnx.defs.SchemaError as exc:
                raise ValueError(
                    f"算子 {node.op_type!r} (domain={domain or 'ai.onnx'}) "
                    f"不支持降至 opset {self.target_opset}"
                ) from exc

        standard_opsets[0].version = self.target_opset
        try:
            onnx.checker.check_model(model)
        except onnx.checker.ValidationError as exc:
            raise ValueError(
                f"ONNX 无法降至 opset {self.target_opset}: {exc}"
            ) from exc

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, self.output_path)
        logger.info(
            "ONNX opset 已降级: %s -> %s，输出: %s",
            source_opset,
            self.target_opset,
            self.output_path,
        )
        return self.output_path


@dataclass
class HbrConvertConfig:
    """转换配置类：把 Hydra 关键词组装成 hb_compile YAML。

    YAML 四段对应关系：
    - model_parameters   : onnx / march / 输出目录与前缀
    - input_parameters   : 输入类型、布局（预处理在 dataset.pipeline）
    - calibration_parameters : 校准数据与量化配置
    - compiler_parameters : 编译优化参数

    march 常用值：nash-e(S100) / nash-m(S100P, 默认) / nash-p(S600)
    """

    # model_parameters
    onnx_path: str | Path
    output_dir: str | Path
    model_name: str | None = None
    march: str = "nash-m"
    working_dir: str | Path | None = None
    output_model_file_prefix: str | None = None
    # HBR 4.7.5 最多支持 opset 20；None 表示不处理 ONNX opset。
    target_opset: int | None = None
    opset_output_path: str | Path | None = None

    # input_parameters
    input_name: str = ""
    # 默认按已预处理 tensor 输入；图像预处理放在 dataset.pipeline
    input_type_rt: str = "featuremap"
    input_type_train: str = "featuremap"
    input_layout_train: str = "NCHW"
    input_shape: str = ""             # 如 1x3x224x224，空则用模型自身 shape
    norm_type: str = ""               # 空=不做工具链内归一化
    mean_value: str = ""
    scale_value: str = ""
    std_value: str = ""

    # calibration_parameters
    cal_data_dir: str | Path | None = None  # 校准 .npy 目录（SaveNpy 产出）
    calibration_type: str = "default"       # default / mix / kl / max / skip / load
    cal_data_type: str = "float32"
    per_channel: bool = False
    # 量化配置：dict、HbrQuantConfig，或已有 json 路径
    quant_config: dict[str, Any] | HbrQuantConfig | str | Path | None = None

    # compiler_parameters
    optimize_level: str = "O2"
    compile_mode: str = "latency"
    core_num: int = 1
    debug: bool = False

    # 工具
    hb_compile: str = "hb_compile"
    extra_args: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # 统一成 Path，并补默认模型名 / 输出前缀 / 工作目录
        self.onnx_path = Path(self.onnx_path)
        self.output_dir = Path(self.output_dir)
        self.model_name = self.model_name or self.onnx_path.stem
        self.working_dir = Path(self.working_dir or self.output_dir / "model_output")
        if not self.output_model_file_prefix:
            self.output_model_file_prefix = f"{self.model_name}_224x224_nv12"
        if self.cal_data_dir is not None:
            self.cal_data_dir = Path(self.cal_data_dir)
        if self.opset_output_path is not None:
            self.opset_output_path = Path(self.opset_output_path)

    @property
    def config_yaml_path(self) -> Path:
        return self.output_dir / "hb_compile_config.yaml"

    @property
    def quant_config_path(self) -> Path:
        return self.output_dir / "quant_config.json"

    @property
    def resolved_opset_output_path(self) -> Path:
        if self.opset_output_path is not None:
            return Path(self.opset_output_path)
        return self.output_dir / f"{self.onnx_path.stem}_opset{self.target_opset}.onnx"

    def prepare_onnx(self) -> Path:
        """按 target_opset 生成 HBR 使用的 ONNX，并返回其路径。"""
        if self.target_opset is None:
            return self.onnx_path
        return HbrOnnxOpsetDowngrader(
            onnx_path=self.onnx_path,
            output_path=self.resolved_opset_output_path,
            target_opset=self.target_opset,
        )()

    def resolve_input_name(self) -> str:
        """返回 HBR 所需的、按 ONNX 图顺序排列的输入名。

        ``hb_compile`` 可以为单输入模型推导 ``input_name``，但面对多输入
        模型会直接拒绝编译。ONNX 初始值也可能同时出现在 ``graph.input`` 中，
        因此这里将它们排除，以匹配 HBR 识别的运行时输入集合。
        """
        if self.input_name:
            return self.input_name

        model = onnx.load(self.onnx_path, load_external_data=False)
        initializer_names = {initializer.name for initializer in model.graph.initializer}
        input_names = [
            value.name
            for value in model.graph.input
            if value.name not in initializer_names
        ]
        if not input_names:
            raise ValueError(f"ONNX 未包含运行时输入: {self.onnx_path}")
        return ";".join(input_names)

    def resolve_input_parameter(self, value: str) -> str:
        """Expand a scalar HBR input setting to every model input.

        HBR accepts one scalar setting for a single input, but requires a
        semicolon-separated value for each input of a multi-input graph.
        Explicit semicolon-separated settings are preserved unchanged.
        """
        input_count = len(self.resolve_input_name().split(";"))
        if input_count <= 1 or ";" in value:
            return value
        return ";".join([value] * input_count)

    def resolve_calibration_dirs(self) -> list[Path]:
        """Return one calibration directory per HBR model input."""
        if self.cal_data_dir is None:
            return []
        return [Path(value) for value in str(self.cal_data_dir).split(";") if value]

    def build_quant(self) -> HbrQuantConfig:
        """从 quant_config 字段得到量化配置对象。"""
        value = self.quant_config
        if isinstance(value, HbrQuantConfig):
            return value
        # Hydra instantiates nested mappings as DictConfig rather than plain
        # dict.  Convert it before type dispatch so ``quant_config`` is not
        # silently dropped from the hb_compile configuration.
        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=True)
        if isinstance(value, dict):
            return HbrQuantConfig(quant_config=value)
        if isinstance(value, (str, Path)):
            path = Path(value)
            data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
            return HbrQuantConfig(quant_config=data)
        return HbrQuantConfig()

    def to_dict(self, quant_config_path: str | Path | None = None) -> dict[str, Any]:
        """组装 hb_compile 完整配置字典。"""
        model_parameters = {
            "onnx_model": str(self.onnx_path.resolve()),
            "march": self.march,
            "working_dir": str(Path(self.working_dir).resolve()),
            "output_model_file_prefix": self.output_model_file_prefix,
        }
        input_parameters: dict[str, Any] = {
            "input_name": self.resolve_input_name(),
            "input_type_rt": self.resolve_input_parameter(self.input_type_rt),
            "input_type_train": self.resolve_input_parameter(self.input_type_train),
            "input_layout_train": self.resolve_input_parameter(self.input_layout_train),
            "input_shape": self.input_shape,
        }
        # 归一化参数可选；默认预处理在 dataset.pipeline，这里不写入
        if self.norm_type:
            input_parameters["norm_type"] = self.norm_type
        if self.mean_value:
            input_parameters["mean_value"] = self.mean_value
        if self.scale_value:
            input_parameters["scale_value"] = self.scale_value
        if self.std_value:
            input_parameters["std_value"] = self.std_value

        calibration_parameters: dict[str, Any] = {
            "calibration_type": self.calibration_type,
            "cal_data_type": self.resolve_input_parameter(self.cal_data_type),
            "per_channel": self.per_channel,
        }
        if self.cal_data_dir is not None:
            calibration_parameters["cal_data_dir"] = ";".join(
                str(path.resolve()) for path in self.resolve_calibration_dirs()
            )
        if quant_config_path is not None:
            calibration_parameters["quant_config"] = str(Path(quant_config_path).resolve())

        compiler_parameters = {
            "optimize_level": self.optimize_level,
            "compile_mode": self.compile_mode,
            "core_num": self.core_num,
            "debug": self.debug,
        }
        return {
            "model_parameters": model_parameters,
            "input_parameters": input_parameters,
            "calibration_parameters": calibration_parameters,
            "compiler_parameters": compiler_parameters,
        }

    def save(self, quant_config_path: str | Path | None = None) -> Path:
        """写出 hb_compile_config.yaml。"""
        path = self.config_yaml_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(
                self.to_dict(quant_config_path),
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        logger.info("写入转换配置: %s", path)
        return path

    def resolve_output_model(self) -> Path:
        """查找转换产物：优先 .hbm，其次 .bc。"""
        prefix = self.output_model_file_prefix
        working_dir = Path(self.working_dir)
        for path in (
            working_dir / f"{prefix}.hbm",
            working_dir / f"{prefix}_quantized_model.bc",
            working_dir / f"{prefix}.bc",
        ):
            if path.is_file():
                return path
        matches = sorted(working_dir.glob("*.hbm")) + sorted(working_dir.glob("*.bc"))
        if matches:
            return matches[-1]
        raise FileNotFoundError(f"未找到转换产物: {working_dir} (prefix={prefix})")


class HbrPipeline:
    """模型转换类：生成配置并调用 hb_compile。

    Hydra 示例::

        conversion:
          hbr_config:
            _target_: edge_model_bench.converter.hbr.HbrConvertConfig
            onnx_path: ${..input_model}
            output_dir: ${..output_dir}
          converter:
            _target_: edge_model_bench.converter.hbr.HbrPipeline
            config: ${..hbr_config}
    """

    def __init__(self, config: HbrConvertConfig) -> None:
        self.config = config

    def __call__(self) -> Path:
        cfg = self.config
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        Path(cfg.working_dir).mkdir(parents=True, exist_ok=True)

        if not cfg.onnx_path.is_file():
            raise FileNotFoundError(f"ONNX 不存在: {cfg.onnx_path}")
        cfg.onnx_path = cfg.prepare_onnx()

        # 非 skip/load 时要求校准 .npy 已准备好
        if cfg.calibration_type not in {"skip", "load"}:
            if cfg.cal_data_dir is None:
                raise ValueError("calibration_type 非 skip/load 时必须设置 cal_data_dir")
            cal_dirs = cfg.resolve_calibration_dirs()
            invalid_dirs = [
                path for path in cal_dirs if not path.is_dir() or not any(path.glob("*.npy"))
            ]
            if invalid_dirs:
                raise FileNotFoundError(
                    f"校准目录缺少 .npy: {invalid_dirs}，请先跑 SaveNpy 校准阶段"
                )

        # 写量化配置 → 写转换 YAML → 执行编译
        quant_path = cfg.build_quant().save(cfg.quant_config_path)
        yaml_path = cfg.save(quant_config_path=quant_path)
        self._run(yaml_path)

        output = cfg.resolve_output_model()
        logger.info("HBR 转换完成: %s", output)
        return output

    def _run(self, config_path: Path) -> None:
        """执行 hb_compile，日志落到 output_dir/hb_compile.log。"""
        cfg = self.config
        cmd = [cfg.hb_compile, "--config", str(config_path.resolve()), *cfg.extra_args]
        logger.info("执行: %s", " ".join(cmd))
        log_path = cfg.output_dir / "hb_compile.log"
        with log_path.open("w", encoding="utf-8") as log_file:
            proc = subprocess.run(
                cmd,
                cwd=str(cfg.output_dir.resolve()),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
        if proc.returncode != 0:
            tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
            raise RuntimeError(
                f"hb_compile 失败 (exit={proc.returncode})，详见 {log_path}\n"
                + "\n".join(tail)
            )
