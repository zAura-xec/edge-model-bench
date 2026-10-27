"""Packed-mask instance-segmentation metrics."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import cv2
import numpy as np
from tqdm.auto import tqdm

from edge_model_bench.evaluator.detection import _average_precision


_POPCOUNT = np.asarray([int(value).bit_count() for value in range(256)], dtype=np.uint8)


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(_POPCOUNT[np.bitwise_and(first, second)].sum())
    union = int(_POPCOUNT[np.bitwise_or(first, second)].sum())
    return intersection / union if union else 0.0


class InstanceSegmentationEvaluator:
    """Compute mask precision, recall, AP50 and AP50:95 for YOLO polygons."""

    def __init__(
        self,
        num_classes: int = 80,
        mask_size: Sequence[int] = (160, 160),
        iou_thresholds: Sequence[float] | None = None,
        class_names: Sequence[str] | None = None,
    ) -> None:
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if len(mask_size) != 2 or any(int(value) <= 0 for value in mask_size):
            raise ValueError("mask_size must be positive [height, width]")
        if class_names is not None and len(class_names) != num_classes:
            raise ValueError(
                f"class_names must contain {num_classes} entries, got {len(class_names)}"
            )
        self.num_classes = int(num_classes)
        self.mask_height, self.mask_width = (int(value) for value in mask_size)
        self.mask_pixels = self.mask_height * self.mask_width
        self.packed_width = (self.mask_pixels + 7) // 8
        self.iou_thresholds = tuple(
            float(value)
            for value in (
                iou_thresholds
                if iou_thresholds is not None
                else np.arange(0.5, 0.96, 0.05)
            )
        )
        self.class_names = tuple(class_names) if class_names is not None else None

    def _target_mask(self, polygon: Any) -> np.ndarray:
        points = np.asarray(polygon, dtype=np.float32)
        if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
            raise ValueError(f"Target polygon must have shape [N,2], got {points.shape}")
        scaled = points.copy()
        scaled[:, 0] *= self.mask_width - 1
        scaled[:, 1] *= self.mask_height - 1
        integer_points = np.rint(scaled).astype(np.int32)
        mask = np.zeros((self.mask_height, self.mask_width), dtype=np.uint8)
        cv2.fillPoly(mask, [integer_points], 1)
        return np.packbits(mask.reshape(-1), bitorder="little")

    def _predictions(
        self, pred: Mapping[str, Any]
    ) -> list[tuple[int, float, np.ndarray]]:
        scores = np.asarray(pred["scores"], dtype=np.float32).reshape(-1)
        labels = np.asarray(pred["labels"], dtype=np.int64).reshape(-1)
        masks = np.asarray(pred["masks"], dtype=np.uint8)
        shape = tuple(int(value) for value in np.asarray(pred["mask_shape"]).reshape(-1))
        if shape != (self.mask_height, self.mask_width):
            raise ValueError(f"Prediction mask shape {shape} != evaluator mask size")
        if masks.shape != (len(scores), self.packed_width) or len(labels) != len(scores):
            raise ValueError("Prediction scores, labels, and packed masks do not align")
        return [
            (int(label), float(score), mask)
            for label, score, mask in zip(labels, scores, masks, strict=True)
            if 0 <= int(label) < self.num_classes
        ]

    def _match(
        self,
        predictions: list[tuple[str, int, float, np.ndarray]],
        targets: list[tuple[str, int, np.ndarray]],
        class_index: int,
        threshold: float,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        class_predictions = sorted(
            (
                (image_id, score, mask)
                for image_id, label, score, mask in predictions
                if label == class_index
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        class_targets = [
            (image_id, mask)
            for image_id, label, mask in targets
            if label == class_index
        ]
        unmatched = set(range(len(class_targets)))
        true_positives = np.zeros(len(class_predictions), dtype=np.float64)
        false_positives = np.zeros(len(class_predictions), dtype=np.float64)
        for index, (image_id, _, mask) in enumerate(class_predictions):
            candidates = [
                target_index
                for target_index in unmatched
                if class_targets[target_index][0] == image_id
            ]
            matched = max(
                candidates,
                key=lambda item: _mask_iou(mask, class_targets[item][1]),
                default=None,
            )
            if matched is not None and _mask_iou(mask, class_targets[matched][1]) >= threshold:
                true_positives[index] = 1.0
                unmatched.remove(matched)
            else:
                false_positives[index] = 1.0
        return true_positives, false_positives, len(class_targets)

    def __call__(self, samples: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        predictions: list[tuple[str, int, float, np.ndarray]] = []
        targets: list[tuple[str, int, np.ndarray]] = []
        num_samples = 0
        for index, sample in enumerate(tqdm(samples, desc="Evaluating", unit="sample")):
            num_samples += 1
            image_id = str(sample.get("image_path", index))
            for label, score, mask in self._predictions(sample["pred"]):
                predictions.append((image_id, label, score, mask))
            for target in sample["label"]:
                label = int(target["label"])
                if 0 <= label < self.num_classes:
                    targets.append((image_id, label, self._target_mask(target["polygon"])))

        ap50_values: list[float] = []
        ap5095_values: list[float] = []
        per_class: dict[str, dict[str, Any]] = {}
        total_tp = total_fp = total_targets = 0.0
        for class_index in range(self.num_classes):
            tp, fp, target_count = self._match(
                predictions, targets, class_index, threshold=0.5
            )
            total_tp += float(tp.sum())
            total_fp += float(fp.sum())
            total_targets += target_count
            if target_count == 0:
                continue
            ap50 = _average_precision(tp, fp, target_count)
            aps = [
                _average_precision(
                    *self._match(predictions, targets, class_index, threshold)
                )
                for threshold in self.iou_thresholds
            ]
            ap5095 = float(np.mean(aps))
            ap50_values.append(ap50)
            ap5095_values.append(ap5095)
            per_class[str(class_index)] = {
                "name": (
                    self.class_names[class_index]
                    if self.class_names is not None
                    else str(class_index)
                ),
                "mask_ap50": ap50,
                "mask_ap50_95": ap5095,
                "num_targets": target_count,
            }

        precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
        recall = total_tp / total_targets if total_targets else 0.0
        mask_map50 = float(np.mean(ap50_values)) if ap50_values else 0.0
        mask_map50_95 = float(np.mean(ap5095_values)) if ap5095_values else 0.0
        return {
            "num_samples": num_samples,
            "num_predictions": len(predictions),
            "num_targets": len(targets),
            "num_classes": self.num_classes,
            "mask_precision": float(precision),
            "mask_recall": float(recall),
            "mask_map50": mask_map50,
            "mask_map50_95": mask_map50_95,
            "map50": mask_map50,
            "map50_95": mask_map50_95,
            "per_class": per_class,
        }


InstanceSegEvaluator = InstanceSegmentationEvaluator
