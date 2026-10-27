"""DETR / RF-DETR detectors for QNN inference outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

# COCO 80 contiguous indices (YOLO / Ultralytics style) from 91-category IDs.
COCO80_CATEGORY_IDS: tuple[int, ...] = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    27,
    28,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    67,
    70,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
)
COCO_CATEGORY_TO_CONTIGUOUS: dict[int, int] = {
    category_id: index for index, category_id in enumerate(COCO80_CATEGORY_IDS)
}


def _as_float32(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=np.float32)


def _as_int64(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=np.int64)


def _squeeze_batch(array: np.ndarray, name: str) -> np.ndarray:
    """Accept ``[1, N, ...]`` or ``[N, ...]`` tensors and drop the batch dim."""
    array = np.asarray(array)
    if array.ndim >= 1 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 0:
        raise ValueError(f"DETR {name} must not be a scalar, got shape {array.shape}")
    return array


def _normalize_detr_outputs(
    boxes: np.ndarray,
    logits: np.ndarray,
    classes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize raw DETR tensors to ``boxes [N,4]``, ``scores [N]``, ``labels [N]``."""
    boxes = _as_float32(_squeeze_batch(boxes, "boxes"))
    scores = _as_float32(_squeeze_batch(logits, "logits")).reshape(-1)
    labels = _as_int64(_squeeze_batch(classes, "classes")).reshape(-1)

    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"DETR boxes must have shape [N, 4] or [1, N, 4], got {boxes.shape}")
    if scores.shape[0] != boxes.shape[0]:
        raise ValueError(
            f"DETR boxes/logits candidate count mismatch: {boxes.shape[0]} vs {scores.shape[0]}"
        )
    if labels.shape[0] != boxes.shape[0]:
        raise ValueError(
            f"DETR boxes/classes candidate count mismatch: {boxes.shape[0]} vs {labels.shape[0]}"
        )
    if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
        raise ValueError("DETR boxes/logits must contain only finite values")
    return boxes, scores, labels


def _cxcywh_norm_to_xyxy(
    boxes: np.ndarray,
    height: float,
    width: float,
) -> np.ndarray:
    """Convert normalized ``cxcywh`` boxes to pixel ``xyxy``."""
    cx, cy, box_width, box_height = boxes.T
    x1 = (cx - box_width / 2.0) * width
    y1 = (cy - box_height / 2.0) * height
    x2 = (cx + box_width / 2.0) * width
    y2 = (cy + box_height / 2.0) * height
    return np.stack((x1, y1, x2, y2), axis=1).astype(np.float32, copy=False)


def _map_class_ids(
    labels: np.ndarray,
    class_id_map: Mapping[int, int] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Map model class ids. Returns ``(mapped_labels, valid_mask)``."""
    if class_id_map is None:
        return labels, np.ones(labels.shape[0], dtype=bool)

    mapped = np.full(labels.shape[0], -1, dtype=np.int64)
    valid = np.zeros(labels.shape[0], dtype=bool)
    for index, class_id in enumerate(labels.tolist()):
        contiguous = class_id_map.get(int(class_id))
        if contiguous is None:
            continue
        mapped[index] = int(contiguous)
        valid[index] = True
    return mapped, valid


def _empty_pred() -> dict[str, np.ndarray]:
    """Canonical empty detection prediction."""
    return {
        "boxes": np.zeros((0, 4), dtype=np.float32),
        "scores": np.zeros((0,), dtype=np.float32),
        "labels": np.zeros((0,), dtype=np.int64),
    }


def _decode_detr(
    boxes: np.ndarray,
    logits: np.ndarray,
    classes: np.ndarray,
    *,
    conf_threshold: float,
    max_detections: int,
    box_format: str,
    input_size: Sequence[int] | None,
    class_id_map: Mapping[int, int] | None,
    num_classes: int,
) -> dict[str, np.ndarray]:
    """Decode DETR outputs into filtered ``{boxes, scores, labels}``.

    ``box_format``:
      - ``xyxy``: pixel ``x1,y1,x2,y2`` (RF-DETR / simplified DETR export)
      - ``cxcywh_norm``: normalized ``cx,cy,w,h`` in ``[0, 1]`` (classic DETR)
    """
    boxes, scores, labels = _normalize_detr_outputs(boxes, logits, classes)

    if box_format == "xyxy":
        decoded_boxes = boxes
    elif box_format == "cxcywh_norm":
        if input_size is None or len(input_size) != 2:
            raise ValueError("input_size [height, width] is required for cxcywh_norm boxes")
        height, width = (float(value) for value in input_size)
        decoded_boxes = _cxcywh_norm_to_xyxy(boxes, height=height, width=width)
    else:
        raise ValueError(f"Unsupported box_format: {box_format!r}")

    keep = scores >= conf_threshold
    labels, map_valid = _map_class_ids(labels, class_id_map)
    keep &= map_valid
    keep &= (labels >= 0) & (labels < num_classes)
    keep &= (decoded_boxes[:, 2] > decoded_boxes[:, 0]) & (
        decoded_boxes[:, 3] > decoded_boxes[:, 1]
    )

    decoded_boxes = decoded_boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    if decoded_boxes.size == 0:
        return _empty_pred()

    order = np.argsort(scores)[::-1][:max_detections]
    return {
        "boxes": decoded_boxes[order].astype(np.float32, copy=False),
        "scores": scores[order].astype(np.float32, copy=False),
        "labels": labels[order].astype(np.int64, copy=False),
    }


class QnnDetr:
    """Decode QNN DETR / RF-DETR outputs into canonical detection predictions.

    Expected raw tensors (by name in the runner result dict):
      - boxes: ``[1, N, 4]`` proposals (``xyxy`` pixel or ``cxcywh_norm``)
      - logits: ``[1, N]`` confidence scores in ``[0, 1]``
      - classes: ``[1, N]`` class ids (COCO category ids by default)

    ``sample["pred"]`` is set to the shared detection format:
      - boxes: ``(K, 4)`` float32 ``xyxy`` in model input pixel space
      - scores: ``(K,)`` float32 confidences
      - labels: ``(K,)`` int64 contiguous class indices (0-based)
    """

    def __init__(
        self,
        boxes: str = "boxes",
        logits: str = "logits",
        classes: str = "classes",
        conf_threshold: float = 0.5,
        max_detections: int = 300,
        box_format: str = "xyxy",
        input_size: Sequence[int] | None = None,
        num_classes: int = 80,
        map_coco_categories: bool = True,
        class_id_map: Mapping[int, int] | None = None,
    ) -> None:
        if not 0.0 <= conf_threshold <= 1.0:
            raise ValueError("conf_threshold must be between 0 and 1")
        if max_detections <= 0:
            raise ValueError("max_detections must be positive")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if box_format not in {"xyxy", "cxcywh_norm"}:
            raise ValueError(f"Unsupported box_format: {box_format!r}")
        if box_format == "cxcywh_norm":
            if input_size is None or len(input_size) != 2:
                raise ValueError("input_size [height, width] is required for cxcywh_norm")
            height, width = (int(value) for value in input_size)
            if height <= 0 or width <= 0:
                raise ValueError("input_size values must be positive")

        if class_id_map is not None:
            resolved_map = {int(key): int(value) for key, value in class_id_map.items()}
        elif map_coco_categories:
            resolved_map = dict(COCO_CATEGORY_TO_CONTIGUOUS)
        else:
            resolved_map = None

        self.boxes = boxes
        self.logits = logits
        self.classes = classes
        self.conf_threshold = float(conf_threshold)
        self.max_detections = int(max_detections)
        self.box_format = box_format
        self.input_size = (
            None if input_size is None else [int(value) for value in input_size]
        )
        self.num_classes = int(num_classes)
        self.class_id_map = resolved_map

    def __call__(
        self,
        sample: dict[str, Any],
        result: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        sample["pred"] = _decode_detr(
            result[self.boxes],
            result[self.logits],
            result[self.classes],
            conf_threshold=self.conf_threshold,
            max_detections=self.max_detections,
            box_format=self.box_format,
            input_size=self.input_size,
            class_id_map=self.class_id_map,
            num_classes=self.num_classes,
        )
        return sample
