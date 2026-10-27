"""Portable, atomic result serialization."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_result(result: Any, output_path: Path) -> None:
    """Replace a result only after a complete, valid JSON document is written."""
    payload = json.dumps(
        result, ensure_ascii=False, indent=2, default=json_default, allow_nan=False
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output_path.parent, suffix=".tmp", delete=False
        ) as file:
            temporary_path = Path(file.name)
            file.write(payload + "\n")
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
