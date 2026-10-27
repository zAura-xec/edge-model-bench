"""Semantic-segmentation metrics."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import cv2
import numpy as np
from tqdm.auto import tqdm


class SemanticSegmentationEvaluator:
    """Compute pixel accuracy, class accuracy, mIoU, and frequency-weighted IoU.

    Each sample must contain a two-dimensional class-index ``label`` and
    ``pred``. Predictions are resized to the ground-truth resolution with
    nearest-neighbour interpolation. Pixels whose target equals
    ``ignore_index`` (255 for Pascal VOC) are excluded from every metric.
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        class_names: Sequence[str] | None = None,
    ) -> None:
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if class_names is not None and len(class_names) != num_classes:
            raise ValueError(
                f"class_names must contain {num_classes} entries, got {len(class_names)}"
            )
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.class_names = tuple(class_names) if class_names is not None else None

    @staticmethod
    def _as_mask(value: Any, name: str) -> np.ndarray:
        mask = np.asarray(value)
        while mask.ndim > 2 and mask.shape[0] == 1:
            mask = mask[0]
        if mask.ndim != 2:
            raise ValueError(
                f"{name} must be a two-dimensional class mask, got {mask.shape}"
            )
        if not np.issubdtype(mask.dtype, np.integer):
            if not np.isfinite(mask).all() or not np.equal(mask, np.floor(mask)).all():
                raise ValueError(f"{name} must contain finite integer class indices")
        return mask.astype(np.int64, copy=False)

    def __call__(self, samples: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        num_samples = 0
        num_ignored = 0

        for sample in tqdm(samples, desc="Evaluating", unit="sample"):
            num_samples += 1
            target = self._as_mask(sample["label"], "label")
            prediction = self._as_mask(sample["pred"], "pred")
            if prediction.shape != target.shape:
                prediction = cv2.resize(
                    prediction,
                    (target.shape[1], target.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(np.int64, copy=False)

            valid = target != self.ignore_index
            num_ignored += int((~valid).sum())
            invalid_target = valid & ((target < 0) | (target >= self.num_classes))
            if invalid_target.any():
                values = np.unique(target[invalid_target])
                raise ValueError(
                    f"Ground truth contains classes outside [0, {self.num_classes}) "
                    f"and ignore_index={self.ignore_index}: {values[:10].tolist()}"
                )
            invalid_prediction = valid & (
                (prediction < 0) | (prediction >= self.num_classes)
            )
            if invalid_prediction.any():
                values = np.unique(prediction[invalid_prediction])
                raise ValueError(
                    f"Prediction contains classes outside [0, {self.num_classes}): "
                    f"{values[:10].tolist()}"
                )

            encoded = self.num_classes * target[valid] + prediction[valid]
            confusion += np.bincount(
                encoded, minlength=self.num_classes**2
            ).reshape(self.num_classes, self.num_classes)

        true_positive = np.diag(confusion).astype(np.float64)
        target_count = confusion.sum(axis=1).astype(np.float64)
        prediction_count = confusion.sum(axis=0).astype(np.float64)
        union = target_count + prediction_count - true_positive
        total = float(target_count.sum())

        class_accuracy = np.divide(
            true_positive,
            target_count,
            out=np.full(self.num_classes, np.nan),
            where=target_count > 0,
        )
        class_iou = np.divide(
            true_positive,
            union,
            out=np.full(self.num_classes, np.nan),
            where=union > 0,
        )
        frequency = target_count / total if total > 0 else np.zeros(self.num_classes)

        per_class: dict[str, dict[str, Any]] = {}
        for index in range(self.num_classes):
            name = (
                self.class_names[index]
                if self.class_names is not None
                else str(index)
            )
            per_class[str(index)] = {
                "name": name,
                "iou": None if np.isnan(class_iou[index]) else float(class_iou[index]),
                "accuracy": (
                    None if np.isnan(class_accuracy[index]) else float(class_accuracy[index])
                ),
                "num_pixels": int(target_count[index]),
            }

        pixel_accuracy = float(true_positive.sum() / total) if total > 0 else 0.0
        mean_accuracy = (
            float(np.nanmean(class_accuracy)) if np.any(target_count > 0) else 0.0
        )
        mean_iou = float(np.nanmean(class_iou)) if np.any(union > 0) else 0.0
        frequency_weighted_iou = float(np.nansum(frequency * class_iou))
        return {
            "num_samples": num_samples,
            "num_evaluated_pixels": int(total),
            "num_ignored_pixels": num_ignored,
            "num_classes": self.num_classes,
            "pixel_accuracy": pixel_accuracy,
            "mean_accuracy": mean_accuracy,
            "mean_iou": mean_iou,
            "frequency_weighted_iou": frequency_weighted_iou,
            "confusion_matrix": confusion.tolist(),
            "per_class": per_class,
        }


SegmentationEvaluator = SemanticSegmentationEvaluator


class Ade20kSegmentationEvaluator(SemanticSegmentationEvaluator):
    """Semantic metrics configured for zero-based ADE150 labels."""

    def __init__(self, ignore_index: int = 255) -> None:
        from edge_model_bench.dataset.ade20k import ADE20K_CLASSES

        super().__init__(
            num_classes=len(ADE20K_CLASSES),
            ignore_index=ignore_index,
            class_names=ADE20K_CLASSES,
        )
