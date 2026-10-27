import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
ROOT = Path(__file__).resolve().parents[1]


def test_cpu_demo_and_staged_evaluation(tmp_path):
    create = runpy.run_path(str(ROOT / "examples/create_demo.py"))["create_demo"]
    data = create(tmp_path / "data")
    for stage in ("task", "evaluate-onnx"):
        output = tmp_path / stage / "result.json"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "edge_model_bench",
                "demo",
                f"demo_root={data}",
                "--stage",
                stage,
                "--output",
                str(output),
            ],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            env={**os.environ, "OMP_NUM_THREADS": "1"},
        )
        result = json.loads(output.read_text())
        assert result["top_1_accuracy"] == 1
        assert result["num_samples"] == result["num_inferences"] == 6
        assert result["average_latency_ms"] > 0


def test_missing_provider_does_not_silently_fallback(tmp_path):
    from edge_model_bench.runner import OnnxRunner

    create = runpy.run_path(str(ROOT / "examples/create_demo.py"))["create_demo"]
    data = create(tmp_path)
    with pytest.raises(RuntimeError, match="unavailable"):
        OnnxRunner(
            data / "model.onnx",
            providers=["CPUExecutionProvider"],
            required_provider="MissingProvider",
        )
