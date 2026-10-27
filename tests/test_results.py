import json
from pathlib import Path

import numpy as np
import pytest

from edge_model_bench.results import save_result


def test_numpy_and_paths(tmp_path):
    path = tmp_path / "nested/result.json"
    save_result({"array": np.array([1, 2]), "value": np.float32(0.5), "path": Path("model")}, path)
    assert json.loads(path.read_text()) == {"array": [1, 2], "value": 0.5, "path": "model"}
    assert not list(path.parent.glob("*.tmp"))


def test_failed_serialization_keeps_previous_result(tmp_path):
    path = tmp_path / "result.json"
    save_result({"valid": True}, path)
    with pytest.raises(ValueError):
        save_result({"invalid": float("nan")}, path)
    assert json.loads(path.read_text()) == {"valid": True}
