import ast
import importlib
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from edge_model_bench.cli import default_config_root, load_config, run_stage, evaluation_model

ROOT = default_config_root()
CONFIGS = sorted(ROOT.rglob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: str(p.relative_to(ROOT)))
def test_existing_config_composes(path):
    # Fragments are intentionally unresolved until a model config supplies
    # image sizes, model names and other task-specific interpolation values.
    with initialize_config_dir(version_base=None, config_dir=str(ROOT)):
        config = compose(config_name=path.relative_to(ROOT).with_suffix("").as_posix())
    if "_base_" not in path.parts and not path.stem.endswith("task"):
        OmegaConf.resolve(config)
    assert config is not None

    # Validate target modules/symbols statically, without importing vendor SDKs.
    def walk(value):
        if isinstance(value, dict):
            target = value.get("_target_")
            if target:
                if not target.startswith("edge_model_bench."):
                    assert target == "qairt.api.converter.converter_config.CalibrationConfig"
                    return
                module, symbol = target.rsplit(".", 1)
                source = Path(__file__).resolve().parents[1] / "src" / Path(*module.split("."))
                source = (
                    source.with_suffix(".py")
                    if source.with_suffix(".py").is_file()
                    else source / "__init__.py"
                )
                assert source.is_file(), target
                tree = ast.parse(source.read_text())
                names = {
                    n.name for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                }
                for node in tree.body:
                    if isinstance(node, ast.ImportFrom):
                        names.update(alias.asname or alias.name for alias in node.names)
                # Package __getattr__ exports are checked via their explicit map.
                assert symbol in names or symbol in source.read_text(), target
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(OmegaConf.to_container(config, resolve=False))


def test_config_by_name_from_other_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_config("demo").pipeline.warmup == 2


def test_explicit_config_root_and_interleaved_overrides(tmp_path):
    (tmp_path / "custom.yaml").write_text("value: 1\n")
    assert load_config("custom", ["value=2"], tmp_path).value == 2


def test_unknown_stage_is_rejected():
    with pytest.raises(ValueError, match="Unsupported stage"):
        run_stage(OmegaConf.create({}), "invalid")


def test_ambiguous_artifact_is_not_selected(tmp_path):
    for name in ("a_compiled.bin", "b_compiled.bin"):
        (tmp_path / name).touch()
    config = OmegaConf.create(
        {
            "pipeline": {},
            "conversion": {
                "output_dir": str(tmp_path),
                "input_model": "missing.onnx",
                "qnn_config": {},
            },
        }
    )
    with pytest.raises(ValueError, match="found 2"):
        evaluation_model(config)


def test_public_modules_import_without_vendor_sdks():
    for module in (
        "classifier",
        "dataset",
        "detector",
        "segmenter",
        "pipeline",
        "runner",
        "converter",
        "profiler",
    ):
        imported = importlib.import_module(f"edge_model_bench.{module}")
        assert imported.__all__
