from typing import Any

import numpy as np


def _classify(sample: dict[str, Any], logits: np.ndarray) -> dict[str, Any]:
    logits = logits.reshape(-1)
    confidence = np.exp(logits - logits.max())
    confidence /= confidence.sum()
    sample["pred"] = dict(enumerate(confidence.tolist()))
    return sample


class QnnClassifier:
    def __init__(self, class_logits: str) -> None:
        self.class_logits = class_logits

    def __call__(
        self,
        sample: dict[str, Any],
        result: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        return _classify(sample, result[self.class_logits])


class OnnxClassifier:
    def __init__(self, output_index: int = 0) -> None:
        self.output_index = output_index

    def __call__(
        self,
        sample: dict[str, Any],
        result: list[np.ndarray],
    ) -> dict[str, Any]:
        return _classify(sample, result[self.output_index])
