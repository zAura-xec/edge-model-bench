"""PointNet output adapters for the ModelNet10 point-cloud task."""

from __future__ import annotations

from typing import Any

import numpy as np



def _softmax(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    values = values - values.max()
    scores = np.exp(values)
    return scores / scores.sum()


def _decode(sample: dict[str, Any], logits: np.ndarray, logits_name: str) -> dict[str, Any]:
    values = np.asarray(logits).reshape(-1)
    scores = _softmax(values)
    sample["pred"] = dict(enumerate(scores.tolist()))
    sample["pred_label"] = int(scores.argmax())
    sample["logits_name"] = logits_name
    return sample


class QnnPointNet:
    """Decode the QNN output dictionary (the classification output is ``x``)."""

    def __init__(self, class_logits: str = "x") -> None:
        self.class_logits = class_logits

    def __call__(self, sample: dict[str, Any], result: dict[str, np.ndarray]) -> dict[str, Any]:
        return _decode(sample, result[self.class_logits], self.class_logits)


class OnnxPointNet:
    """Decode ONNX Runtime's ordered output list."""

    def __init__(self, output_index: int = 0) -> None:
        self.output_index = output_index

    def __call__(self, sample: dict[str, Any], result: list[np.ndarray]) -> dict[str, Any]:
        return _decode(sample, result[self.output_index], str(self.output_index))


# User-facing aliases used by 3D task configurations.
PointNet3DSegmenter = QnnPointNet
QnnPointNetSegmenter = QnnPointNet
