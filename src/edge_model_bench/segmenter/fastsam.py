"""FastSAM output adapter for the common packed-mask evaluator."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from edge_model_bench.detector.yolo import _nms


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


class FastSamSegmenter:
    """Decode FastSAM's split boxes, scores, coefficients and prototypes."""

    def __init__(
        self,
        score_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        max_detections: int = 100,
        input_size: Sequence[int] = (640, 640),
        prediction: str | None = None,
        prototypes: str | None = None,
        num_classes: int | None = None,
        layout: str | None = None,
        conf_threshold: float | None = None,
        mask_threshold: float | None = None,
    ) -> None:
        # FastSAM reuses the YOLO-seg task configuration, whose output fields
        # do not apply to FastSAM's explicit boxes/scores/prototype tensors.
        del prediction, prototypes, num_classes, layout, conf_threshold, mask_threshold
        self.score_threshold = float(score_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_detections = int(max_detections)
        self.input_height, self.input_width = (int(value) for value in input_size)

    def __call__(self, sample: dict[str, Any], result: dict[str, np.ndarray]) -> dict[str, Any]:
        boxes = np.asarray(result["boxes"], dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(result["scores"], dtype=np.float32).reshape(-1)
        coefficients = np.asarray(result["mask_coeffs"], dtype=np.float32).reshape(len(boxes), -1)
        prototypes = np.asarray(result["mask_protos"], dtype=np.float32)
        if prototypes.ndim != 4 or prototypes.shape[0] != 1:
            raise ValueError(f"FastSAM prototypes must be [1,M,H,W], got {prototypes.shape}")
        prototypes = prototypes[0]
        if coefficients.shape[1] != prototypes.shape[0]:
            raise ValueError("FastSAM coefficient and prototype dimensions differ")

        valid = np.flatnonzero(scores >= self.score_threshold)
        valid = valid[(boxes[valid, 2] > boxes[valid, 0]) & (boxes[valid, 3] > boxes[valid, 1])]
        if len(valid):
            kept = _nms(boxes[valid], scores[valid], self.iou_threshold)
            indices = valid[kept]
            indices = indices[np.argsort(scores[indices])[::-1]][: self.max_detections]
        else:
            indices = np.empty(0, dtype=np.int64)

        height, width = prototypes.shape[1:]
        if len(indices):
            logits = coefficients[indices] @ prototypes.reshape(prototypes.shape[0], -1)
            masks = _sigmoid(logits).reshape(-1, height, width)
            scaled = boxes[indices].copy()
            scaled[:, [0, 2]] *= width / self.input_width
            scaled[:, [1, 3]] *= height / self.input_height
            columns = np.arange(width, dtype=np.float32)[None, None, :]
            rows = np.arange(height, dtype=np.float32)[None, :, None]
            crop = (
                (columns >= scaled[:, 0, None, None])
                & (columns < scaled[:, 2, None, None])
                & (rows >= scaled[:, 1, None, None])
                & (rows < scaled[:, 3, None, None])
            )
            packed = np.packbits(((masks >= 0.5) & crop).reshape(len(indices), -1), axis=1, bitorder="little")
        else:
            packed = np.zeros((0, (height * width + 7) // 8), dtype=np.uint8)
        sample["pred"] = {
            "boxes": boxes[indices],
            "scores": scores[indices],
            "labels": np.zeros(len(indices), dtype=np.int64),
            "masks": packed,
            "mask_shape": np.asarray([height, width], dtype=np.int32),
        }
        return sample
