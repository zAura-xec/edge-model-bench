"""YOLO detectors for QNN inference outputs."""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_float32(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=np.float32)


def _normalize_boxes_scores(
    boxes: np.ndarray,
    class_conf: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize raw model outputs to boxes ``[N, 4]`` and scores ``[N, C]``.

    Only accepts:
      - boxes: ``[1, 4, N]``
      - class_conf: ``[1, C, N]``
    """
    boxes = _as_float32(boxes)
    class_conf = _as_float32(class_conf)

    if boxes.ndim != 3 or boxes.shape[0] != 1 or boxes.shape[1] != 4:
        raise ValueError(f"YOLO boxes must have shape [1, 4, N], got {boxes.shape}")
    if class_conf.ndim != 3 or class_conf.shape[0] != 1:
        raise ValueError(f"YOLO class_conf must have shape [1, C, N], got {class_conf.shape}")
    if boxes.shape[2] != class_conf.shape[2]:
        raise ValueError(
            f"YOLO boxes/class_conf candidate count mismatch: "
            f"{boxes.shape[2]} vs {class_conf.shape[2]}"
        )
    if not np.isfinite(boxes).all() or not np.isfinite(class_conf).all():
        raise ValueError("YOLO outputs must contain only finite values")

    # [1, 4, N] -> [N, 4], [1, C, N] -> [N, C]
    return boxes[0].T, class_conf[0].T



def _cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    cx, cy, width, height = boxes.T
    return np.stack(
        (cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0),
        axis=1,
    )


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy NMS. boxes are xyxy."""
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.int64)

    order = np.argsort(scores)[::-1]
    kept: list[int] = []
    while order.size:
        current = int(order[0])
        kept.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[current, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[current, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[current, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[current, 3], boxes[rest, 3])
        intersection = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_current = np.maximum(0.0, boxes[current, 2] - boxes[current, 0]) * np.maximum(
            0.0, boxes[current, 3] - boxes[current, 1]
        )
        area_rest = np.maximum(0.0, boxes[rest, 2] - boxes[rest, 0]) * np.maximum(
            0.0, boxes[rest, 3] - boxes[rest, 1]
        )
        iou = intersection / np.maximum(
            area_current + area_rest - intersection, np.finfo(np.float32).eps
        )
        order = rest[iou <= iou_threshold]
    return np.asarray(kept, dtype=np.int64)


def _empty_pred() -> dict[str, np.ndarray]:
    """Canonical empty detection prediction."""
    return {
        "boxes": np.zeros((0, 4), dtype=np.float32),
        "scores": np.zeros((0,), dtype=np.float32),
        "labels": np.zeros((0,), dtype=np.int64),
    }


def _decode_yolo(
    boxes: np.ndarray,
    class_conf: np.ndarray,
    *,
    conf_threshold: float,
    iou_threshold: float,
    max_detections: int,
    multi_label: bool,
    box_format: str = "xyxy",
) -> dict[str, np.ndarray]:
    """Decode YOLO boxes/scores into filtered ``{boxes, scores, labels}``.

    ``box_format``:
      - ``xyxy``: model outputs pixel ``x1,y1,x2,y2`` (e.g. YOLO26)
      - ``cxcywh``: model outputs pixel ``cx,cy,w,h`` (e.g. some YOLOv8 exports)
    """
    boxes, class_conf = _normalize_boxes_scores(boxes, class_conf)
    if box_format not in {"xyxy", "cxcywh"}:
        raise ValueError(f"Unsupported box_format: {box_format!r}")

    if multi_label:
        candidate_indices, class_indices = np.nonzero(class_conf >= conf_threshold)
        selected_scores = class_conf[candidate_indices, class_indices]
    else:
        class_indices = np.argmax(class_conf, axis=1)
        selected_scores = class_conf[np.arange(class_conf.shape[0]), class_indices]
        candidate_indices = np.flatnonzero(selected_scores >= conf_threshold)
        class_indices = class_indices[candidate_indices]
        selected_scores = selected_scores[candidate_indices]

    if candidate_indices.size == 0:
        return _empty_pred()

    selected_boxes = boxes[candidate_indices]
    if box_format == "cxcywh":
        selected_boxes = _cxcywh_to_xyxy(selected_boxes)
    valid = (selected_boxes[:, 2] > selected_boxes[:, 0]) & (
        selected_boxes[:, 3] > selected_boxes[:, 1]
    )
    selected_boxes = selected_boxes[valid]
    class_indices = class_indices[valid]
    selected_scores = selected_scores[valid]

    if selected_boxes.size == 0:
        return _empty_pred()

    kept: list[int] = []
    for class_index in np.unique(class_indices):
        positions = np.flatnonzero(class_indices == class_index)
        kept.extend(
            positions[
                _nms(
                    selected_boxes[positions],
                    selected_scores[positions],
                    iou_threshold,
                )
            ].tolist()
        )

    kept = sorted(kept, key=lambda index: float(selected_scores[index]), reverse=True)[
        :max_detections
    ]
    if not kept:
        return _empty_pred()

    kept_indices = np.asarray(kept, dtype=np.int64)
    return {
        "boxes": selected_boxes[kept_indices].astype(np.float32, copy=False),
        "scores": selected_scores[kept_indices].astype(np.float32, copy=False),
        "labels": class_indices[kept_indices].astype(np.int64, copy=False),
    }


class QnnYolo:
    """Decode QNN YOLO outputs into canonical detection predictions.

    Expected raw tensors (by name in the runner result dict):
      - boxes: ``[1, 4, N]`` proposals (``xyxy`` or ``cxcywh``, see ``box_format``)
      - class_conf: ``[1, C, N]`` per-class confidence scores

    ``sample["pred"]`` is set to the shared detection format:
      - boxes: ``(K, 4)`` float32 ``xyxy`` in model input pixel space
      - scores: ``(K,)`` float32 confidences
      - labels: ``(K,)`` int64 class indices
    """

    def __init__(
        self,
        boxes: str,
        class_conf: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        max_detections: int = 300,
        multi_label: bool = False,
        box_format: str = "xyxy",
    ) -> None:
        if not 0.0 <= conf_threshold <= 1.0:
            raise ValueError("conf_threshold must be between 0 and 1")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1")
        if max_detections <= 0:
            raise ValueError("max_detections must be positive")
        if box_format not in {"xyxy", "cxcywh"}:
            raise ValueError(f"Unsupported box_format: {box_format!r}")

        self.boxes = boxes
        self.class_conf = class_conf
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_detections = int(max_detections)
        self.multi_label = bool(multi_label)
        self.box_format = box_format

    def __call__(
        self,
        sample: dict[str, Any],
        result: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        sample["pred"] = _decode_yolo(
            result[self.boxes],
            result[self.class_conf],
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
            max_detections=self.max_detections,
            multi_label=self.multi_label,
            box_format=self.box_format,
        )
        return sample


class MtnnYolo:
    """Decode ordered MTNN YOLO outputs.

    MTNN models expose output tensors by index, whereas QNN exposes named
    tensors.  This adapter keeps the shared decoder and evaluator unchanged.
    """

    def __init__(
        self,
        boxes_index: int = 0,
        class_conf_index: int = 1,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        max_detections: int = 300,
        multi_label: bool = False,
        box_format: str = "xyxy",
    ) -> None:
        self.boxes_index = boxes_index
        self.class_conf_index = class_conf_index
        self.decoder = QnnYolo(
            boxes="boxes",
            class_conf="class_logits",
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            multi_label=multi_label,
            box_format=box_format,
        )

    def __call__(self, sample: dict[str, Any], result: list[np.ndarray]) -> dict[str, Any]:
        try:
            outputs = {
                "boxes": result[self.boxes_index],
                "class_logits": result[self.class_conf_index],
            }
        except IndexError as exc:
            raise ValueError(
                "MTNN YOLO 输出数量不足: "
                f"需要索引 {self.boxes_index}/{self.class_conf_index}，得到 {len(result)}"
            ) from exc
        return self.decoder(sample, outputs)
