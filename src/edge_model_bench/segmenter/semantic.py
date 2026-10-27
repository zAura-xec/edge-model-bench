"""Postprocessors for models that directly output a semantic class mask."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import cv2
import numpy as np


class QnnSemanticSegmenter:
    """Normalize a QNN class-index tensor into ``sample['pred']``.

    Supported mask shapes are ``[H, W]``, ``[1, H, W]`` and
    ``[1, 1, H, W]``. DeepLabV3+ MobileNet and FCN-ResNet50 in this benchmark
    both expose a uint8 output named ``mask`` with shape ``[1, 520, 520]``.
    """

    def __init__(self, mask: str = "mask", num_classes: int | None = None) -> None:
        if num_classes is not None and num_classes <= 0:
            raise ValueError("num_classes must be positive when set")
        self.mask = mask
        self.num_classes = num_classes

    def _select_output(self, result: Any) -> np.ndarray:
        if isinstance(result, Mapping):
            if self.mask not in result:
                raise KeyError(
                    f"Segmentation output {self.mask!r} not found; "
                    f"available outputs: {sorted(result)}"
                )
            return np.asarray(result[self.mask])
        if isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
            if len(result) != 1:
                raise ValueError(
                    "A sequence result is only supported for a single-output model, "
                    f"got {len(result)} outputs"
                )
            return np.asarray(result[0])
        # QAIRT returns ExecutionResult, which supports output-name indexing
        # but intentionally does not implement collections.abc.Mapping.
        try:
            return np.asarray(result[self.mask])
        except (KeyError, IndexError, TypeError) as error:
            raise TypeError(
                f"Unsupported segmentation result type: {type(result).__name__}"
            ) from error

    def __call__(self, sample: dict[str, Any], result: Any) -> dict[str, Any]:
        mask = self._select_output(result)
        while mask.ndim > 2 and mask.shape[0] == 1:
            mask = mask[0]
        if mask.ndim != 2:
            raise ValueError(
                "Semantic class mask must have shape [H,W], [1,H,W], or "
                f"[1,1,H,W], got {mask.shape}"
            )
        if not np.issubdtype(mask.dtype, np.integer):
            if not np.isfinite(mask).all() or not np.equal(mask, np.floor(mask)).all():
                raise ValueError("Semantic mask must contain finite integer class indices")
        mask = mask.astype(np.int64, copy=False)
        if self.num_classes is not None:
            invalid = (mask < 0) | (mask >= self.num_classes)
            if invalid.any():
                values = np.unique(mask[invalid])
                raise ValueError(
                    f"Semantic prediction contains classes outside [0, {self.num_classes}): "
                    f"{values[:10].tolist()}"
                )
        sample["pred"] = mask
        return sample


class QnnDeepLabV3Plus(QnnSemanticSegmenter):
    """DeepLabV3+ MobileNet direct-mask postprocessor."""


class QnnFcnResNet50(QnnSemanticSegmenter):
    """FCN-ResNet50 direct-mask postprocessor."""


class QnnSegFormer(QnnSemanticSegmenter):
    """Convert SegFormer class logits to an original-resolution class mask."""

    def __init__(
        self,
        logits: str = "class_logits",
        num_classes: int = 150,
    ) -> None:
        super().__init__(mask=logits, num_classes=num_classes)
        self.logits = logits

    def __call__(self, sample: dict[str, Any], result: Any) -> dict[str, Any]:
        logits = self._select_output(result)
        while logits.ndim > 3 and logits.shape[0] == 1:
            logits = logits[0]
        if logits.ndim != 3:
            raise ValueError(
                "SegFormer logits must have shape [C,H,W], [H,W,C], or a "
                f"batch-one equivalent, got {logits.shape}"
            )
        class_axes = [
            axis for axis, size in enumerate(logits.shape) if size == self.num_classes
        ]
        if len(class_axes) != 1:
            raise ValueError(
                f"Cannot identify the {self.num_classes}-class axis in {logits.shape}"
            )
        logits = np.moveaxis(logits, class_axes[0], -1).astype(np.float32, copy=False)

        original_size = sample.get("original_size")
        if original_size is not None:
            height, width = map(int, original_size[:2])
            if logits.shape[:2] != (height, width):
                logits = cv2.resize(
                    logits, (width, height), interpolation=cv2.INTER_LINEAR
                )
                if logits.ndim == 2:
                    logits = logits[..., None]
        sample["pred"] = np.argmax(logits, axis=-1).astype(np.int64, copy=False)
        return sample
