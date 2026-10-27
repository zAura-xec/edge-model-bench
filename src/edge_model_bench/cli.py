"""Compose trusted Hydra configurations and run individual pipeline stages."""

import argparse
import logging
from functools import partial
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from .pipeline.model_converter import prepare_calibration
from .pipeline.model_evaluator import ModelEvaluator
from .results import save_result

STAGES = (
    "task",
    "calibrate",
    "convert",
    "quantize-onnx",
    "convert-static-onnx",
    "evaluate-onnx",
    "evaluate",
    "evaluate-qnn",
)


def default_config_root() -> Path:
    bundled = Path(__file__).parent / "config"
    return bundled if bundled.is_dir() else Path(__file__).resolve().parents[2] / "config"


def load_config(path: str | Path, overrides=(), config_root: Path | None = None):
    """Load a file, or a config name relative to an explicit or bundled root."""
    root = Path(config_root or default_config_root()).expanduser().resolve()
    candidate = Path(path).expanduser()
    if candidate.is_file():
        candidate = candidate.resolve()
        try:
            name = candidate.relative_to(root).with_suffix("").as_posix()
        except ValueError:
            if config_root is not None:
                raise ValueError(
                    f"Config {candidate} must be inside --config-root {root}"
                ) from None
            root, name = candidate.parent, candidate.stem
    else:
        if candidate.is_absolute() or ".." in candidate.parts:
            raise FileNotFoundError(candidate)
        name = candidate.with_suffix("").as_posix()
    with initialize_config_dir(version_base=None, config_dir=str(root)):
        config = compose(config_name=name, overrides=list(overrides))
    OmegaConf.resolve(config)
    return config


def evaluation_model(config: Any) -> Any:
    """Use an explicit artifact, or resolve an unambiguous legacy QNN artifact."""
    for node, key in (
        (config.pipeline, "model"),
        (config, "input_bin"),
        (config, "compiled_model"),
    ):
        if node.get(key) is not None:
            return node[key]
    conversion = config.get("conversion", {})
    if "qnn_config" in conversion and "output_dir" in conversion:
        directory = Path(conversion.output_dir)
        name = conversion.qnn_config.get("model_name") or Path(conversion.input_model).stem
        expected = directory / f"{name}_compiled.bin"
        if expected.is_file():
            return expected
        matches = sorted(directory.glob("*_compiled.bin"))
        if len(matches) == 1:
            return matches[0]
        raise ValueError(
            f"Expected one compiled QNN artifact in {directory}; found {len(matches)}. "
            "Set +compiled_model=/path/to/artifact explicitly."
        )
    if config.get("model_path") is not None:
        return config.model_path
    raise ValueError("Set pipeline.model or +compiled_model=/path/to/artifact for evaluation")


def run_stage(config: Any, stage: str = "task", provider: str = "CPUExecutionProvider") -> Any:
    if stage not in STAGES:
        raise ValueError(f"Unsupported stage: {stage}")
    if stage == "task":
        return instantiate(config.pipeline, _recursive_=True)()
    if stage == "calibrate":
        node = config.pipeline.get("calibration_dataset")
        if node is None:
            raise ValueError("This configuration has no pipeline.calibration_dataset")
        return {"samples": prepare_calibration(instantiate(node))}
    if stage in {"convert", "quantize-onnx", "convert-static-onnx"}:
        node = config.pipeline.get("converter")
        if node is None:
            raise ValueError("This configuration has no pipeline.converter")
        converter = instantiate(node)
        if stage == "convert":
            return converter()
        operation = "quantizer" if stage == "quantize-onnx" else "convert_static"
        callback = getattr(converter, operation, None)
        if callback is None:
            raise ValueError(f"The selected converter does not support {stage}")
        return callback()

    # Instantiate only evaluation components. Conversion SDKs are never loaded here.
    dataset = instantiate(config.pipeline.dataset)
    postprocessor = instantiate(config.pipeline.postprocessor)
    evaluator = instantiate(config.pipeline.evaluator)
    if stage == "evaluate-onnx":
        from .runner.onnx import OnnxRunner

        # Most device decoders consume named outputs. Preserve the list format
        # of configs explicitly designed for the ONNX list-output runner.
        runner_node = config.pipeline.get("runner", {})
        is_onnx = runner_node.get("_target_", "").endswith(".OnnxRunner")
        mapping = runner_node.get("output_mapping", False) if is_onnx else True
        runner = partial(
            OnnxRunner, providers=[provider], required_provider=provider, output_mapping=mapping
        )
        model = config.get("conversion", {}).get("input_model") or config.get("model_path")
        if model is None:
            raise ValueError("ONNX evaluation needs model_path or conversion.input_model")
    else:
        runner = instantiate(config.pipeline.runner)
        model = evaluation_model(config)
    return ModelEvaluator(
        model, dataset, runner, postprocessor, evaluator, warmup=config.pipeline.get("warmup", 0)
    )()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="YAML file or bundled config name")
    parser.add_argument("overrides", nargs="*", help="Hydra key=value overrides")
    parser.add_argument("--config-root", type=Path, help="Root for absolute Hydra defaults")
    parser.add_argument("--stage", choices=STAGES, default="task")
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--save-result", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_intermixed_args(argv)
    logging.basicConfig(level=logging.INFO)
    config = load_config(args.config, args.overrides, args.config_root)
    if args.print_config:
        print(OmegaConf.to_yaml(config), end="")
        return
    result = run_stage(config, args.stage, args.provider)
    save = args.save_result
    if save is None:
        save = args.stage in {"task", "evaluate", "evaluate-qnn", "evaluate-onnx"} and config.get(
            "save_result", True
        )
    if save:
        output_path = args.output or Path(config.get("work_dir", "./work")) / "result.json"
        save_result(result, output_path)
        logging.info("Result saved to: %s", output_path)
    logging.info("Pipeline finished: %s", result)
