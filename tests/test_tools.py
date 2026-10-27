import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_mixed_task_report(tmp_path):
    for name, data in (("cls", {"accuracy": 1, "classes": [0, 1]}), ("seg", {"mean_iou": 0.5})):
        path = tmp_path / name / "result.json"
        path.parent.mkdir()
        path.write_text(json.dumps(data))
    export = module("export_benchmark_csv")
    assert export.export_results(tmp_path, tmp_path / "summary.csv") == 2
    assert "mean_iou" in (tmp_path / "summary.csv").read_text()
    assert "classes" not in (tmp_path / "summary.csv").read_text()


def test_audit_redacts_values_and_finds_internal_inputs():
    audit = module("audit_release")
    secret = "".join(["password", "=", chr(34), "demo-secret", chr(34)])
    host = "10." + "20.30.40"
    home = "/home" + "/employee/project"
    labels = {label for _, label in audit.findings(f"{secret}\n{host}\n{home}".encode())}
    assert labels == {"embedded password", "internal IP address", "personal absolute path"}
