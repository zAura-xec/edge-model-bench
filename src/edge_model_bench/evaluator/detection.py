"""Shared detection metrics for YOLO, DETR, and similar detectors."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from tqdm.auto import tqdm


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def _average_precision(
    true_positives: np.ndarray,
    false_positives: np.ndarray,
    target_count: int,
) -> float:
    if target_count == 0:
        return 0.0
    recall = np.cumsum(true_positives) / target_count
    precision = np.cumsum(true_positives) / np.maximum(
        np.cumsum(true_positives + false_positives), 1.0
    )
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    change = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[change + 1] - recall[change]) * precision[change + 1]))


def _yolo_cxcywh_to_xyxy(
    box: Sequence[float],
    height: float,
    width: float,
) -> np.ndarray:
    """Convert normalized YOLO ``cxcywh`` to pixel ``xyxy``."""
    cx, cy, box_width, box_height = (float(value) for value in box)
    x1 = (cx - box_width / 2.0) * width
    y1 = (cy - box_height / 2.0) * height
    x2 = (cx + box_width / 2.0) * width
    y2 = (cy + box_height / 2.0) * height
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def _parse_predictions(
    pred: Mapping[str, Any],
    ignored_classes: set[int],
    num_classes: int | None,
) -> list[tuple[int, float, np.ndarray]]:
    """Parse detector ``pred`` into ``(class_id, score, xyxy)``.

    Canonical format produced by all detectors (``QnnYolo``, ``QnnDetr``, ...):
      ``{"boxes": (K, 4) xyxy, "scores": (K,), "labels": (K,)}``
    """
    if not all(key in pred for key in ("boxes", "scores", "labels")):
        raise ValueError(
            "pred must contain canonical keys: boxes, scores, labels "
            f"(got {sorted(pred.keys())})"
        )

    boxes = np.asarray(pred["boxes"], dtype=np.float32)
    scores = np.asarray(pred["scores"], dtype=np.float32).reshape(-1)
    labels = np.asarray(pred["labels"], dtype=np.int64).reshape(-1)

    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"pred boxes must have shape (K, 4), got {boxes.shape}")
    if scores.shape[0] != boxes.shape[0] or labels.shape[0] != boxes.shape[0]:
        raise ValueError(
            "pred boxes/scores/labels length mismatch: "
            f"{boxes.shape[0]}, {scores.shape[0]}, {labels.shape[0]}"
        )

    predictions: list[tuple[int, float, np.ndarray]] = []
    if boxes.shape[0] == 0:
        return predictions

    for class_id, score, box in zip(labels, scores, boxes, strict=True):
        class_index = int(class_id)
        if class_index in ignored_classes:
            continue
        if num_classes is not None and not 0 <= class_index < num_classes:
            continue
        predictions.append((class_index, float(score), np.asarray(box, dtype=np.float32)))
    return predictions


def _parse_targets(
    labels: Sequence[Mapping[str, Any]],
    height: float,
    width: float,
    ignored_classes: set[int],
) -> list[tuple[int, np.ndarray]]:
    """Parse dataset labels into ``(class_id, xyxy)`` in input pixel space."""
    targets: list[tuple[int, np.ndarray]] = []
    for item in labels:
        class_index = int(item["label"])
        if class_index in ignored_classes:
            continue
        box = _yolo_cxcywh_to_xyxy(item["box"], height=height, width=width)
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        targets.append((class_index, box))
    return targets


class DetectionEvaluator:
    """Compute detection precision, recall, mAP@0.5 and mAP@0.5:0.95.

    Shared evaluator for YOLO, DETR / RF-DETR, and other detectors. Detectors
    are responsible for normalizing model-specific outputs into one ``pred``
    format; this evaluator does not branch on detector type.

    Expected sample fields:
      - ``label``: ``[{"label": class_id, "box": [cx, cy, w, h]}, ...]``
        with normalized YOLO boxes
      - ``pred``: ``{"boxes": (K, 4) xyxy, "scores": (K,), "labels": (K,)}``
        in model input pixel space (from ``QnnYolo`` / ``QnnDetr``)

    Ground-truth boxes are scaled to ``input_size`` (``[height, width]``),
    matching a plain resize preprocess.
    """

    def __init__(
        self,
        input_size: Sequence[int],
        num_classes: int | None = None,
        ignored_classes: Sequence[int] | None = None,
        iou_thresholds: Sequence[float] | None = None,
    ) -> None:
        if len(input_size) != 2:
            raise ValueError("input_size must be [height, width]")
        height, width = (int(value) for value in input_size)
        if height <= 0 or width <= 0:
            raise ValueError("input_size values must be positive")
        if num_classes is not None and int(num_classes) <= 0:
            raise ValueError("num_classes must be positive when set")

        self.input_height = height
        self.input_width = width
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.ignored_classes = {int(value) for value in (ignored_classes or ())}
        self.iou_thresholds = tuple(
            float(value)
            for value in (
                iou_thresholds
                if iou_thresholds is not None
                else np.arange(0.5, 0.96, 0.05)
            )
        )
        if not self.iou_thresholds:
            raise ValueError("iou_thresholds must not be empty")
        if any(not 0.0 < threshold <= 1.0 for threshold in self.iou_thresholds):
            raise ValueError("iou_thresholds must be in (0, 1]")

    def _match(
        self,
        predictions: list[tuple[str, int, float, np.ndarray]],
        targets: list[tuple[str, int, np.ndarray]],
        class_index: int,
        threshold: float,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        class_predictions = sorted(
            (
                (image_id, score, box)
                for image_id, pred_class, score, box in predictions
                if pred_class == class_index
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        class_targets = [
            (image_id, box)
            for image_id, target_class, box in targets
            if target_class == class_index
        ]
        unmatched = set(range(len(class_targets)))
        true_positives = np.zeros(len(class_predictions), dtype=np.float64)
        false_positives = np.zeros(len(class_predictions), dtype=np.float64)

        for index, (image_id, _, box) in enumerate(class_predictions):
            candidates = [
                target_index
                for target_index in unmatched
                if class_targets[target_index][0] == image_id
            ]
            matched = max(
                candidates,
                key=lambda item: _iou(box, class_targets[item][1]),
                default=None,
            )
            if (
                matched is not None
                and _iou(box, class_targets[matched][1]) >= threshold
            ):
                true_positives[index] = 1.0
                unmatched.remove(matched)
            else:
                false_positives[index] = 1.0
        return true_positives, false_positives, len(class_targets)

    def __call__(self, samples: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        predictions: list[tuple[str, int, float, np.ndarray]] = []
        targets: list[tuple[str, int, np.ndarray]] = []
        class_ids: set[int] = set()
        num_samples = 0

        for index, sample in enumerate(tqdm(samples, desc="Evaluating", unit="sample")):
            num_samples += 1
            image_id = str(sample.get("image_path", index))
            sample_predictions = _parse_predictions(
                sample["pred"],
                ignored_classes=self.ignored_classes,
                num_classes=self.num_classes,
            )
            sample_targets = _parse_targets(
                sample["label"],
                height=self.input_height,
                width=self.input_width,
                ignored_classes=self.ignored_classes,
            )
            for class_id, score, box in sample_predictions:
                class_ids.add(class_id)
                predictions.append((image_id, class_id, score, box))
            for class_id, box in sample_targets:
                class_ids.add(class_id)
                targets.append((image_id, class_id, box))

        if self.num_classes is not None:
            class_count = self.num_classes
        else:
            class_count = (max(class_ids) + 1) if class_ids else 0

        ap50: list[float] = []
        ap5095: list[float] = []
        per_class: dict[str, dict[str, float]] = {}
        total_tp = total_fp = total_targets = 0.0

        for class_index in range(class_count):
            if class_index in self.ignored_classes:
                continue
            tp, fp, target_count = self._match(
                predictions, targets, class_index, threshold=0.5
            )
            total_tp += float(tp.sum())
            total_fp += float(fp.sum())
            total_targets += target_count
            if target_count == 0:
                continue

            class_ap50 = _average_precision(tp, fp, target_count)
            class_aps = [
                _average_precision(
                    *self._match(predictions, targets, class_index, threshold)
                )
                for threshold in self.iou_thresholds
            ]
            class_ap5095 = float(np.mean(class_aps))
            ap50.append(class_ap50)
            ap5095.append(class_ap5095)
            per_class[str(class_index)] = {
                "ap50": class_ap50,
                "ap50_95": class_ap5095,
                "num_targets": float(target_count),
            }

        precision = (
            total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        )
        recall = total_tp / total_targets if total_targets > 0 else 0.0
        return {
            "num_samples": num_samples,
            "num_predictions": len(predictions),
            "num_targets": int(total_targets),
            "num_classes": class_count,
            "precision": float(precision),
            "recall": float(recall),
            "map50": float(np.mean(ap50)) if ap50 else 0.0,
            "map50_95": float(np.mean(ap5095)) if ap5095 else 0.0,
            "per_class": per_class,
        }
