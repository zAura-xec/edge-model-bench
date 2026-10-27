"""SSD detectors for QNN inference outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from edge_model_bench.detector.detr import _decode_detr


class QnnSsd:
    """Decode QNN SSD outputs into canonical detection predictions.

    Expected raw tensors (by name in the runner result dict):
      - boxes: ``[1, N, 4]`` proposals (pixel ``xyxy`` by default)
      - scores: ``[1, N]`` confidences in ``[0, 1]``
      - labels: ``[1, N]`` class indices (contiguous 0-based COCO80 by default)

    ``sample["pred"]`` is set to the shared detection format:
      - boxes: ``(K, 4)`` float32 ``xyxy`` in model input pixel space
      - scores: ``(K,)`` float32 confidences
      - labels: ``(K,)`` int64 class indices
    """

    def __init__(
        self,
        boxes: str = "boxes",
        scores: str = "scores",
        labels: str = "labels",
        conf_threshold: float = 0.05,
        max_detections: int = 100,
        box_format: str = "xyxy",
        input_size: Sequence[int] | None = None,
        num_classes: int = 80,
        map_coco_categories: bool = False,
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
            from edge_model_bench.detector.detr import COCO_CATEGORY_TO_CONTIGUOUS

            resolved_map = dict(COCO_CATEGORY_TO_CONTIGUOUS)
        else:
            resolved_map = None

        self.boxes = boxes
        self.scores = scores
        self.labels = labels
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
            result[self.scores],
            result[self.labels],
            conf_threshold=self.conf_threshold,
            max_detections=self.max_detections,
            box_format=self.box_format,
            input_size=self.input_size,
            class_id_map=self.class_id_map,
            num_classes=self.num_classes,
        )
        return sample
