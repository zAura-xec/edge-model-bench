"""YOLO instance-segmentation postprocessing for QNN outputs."""

from __future__ import annotations

from typing import Any

import numpy as np

from edge_model_bench.detector.yolo import _cxcywh_to_xyxy, _nms


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


class QnnYoloSegmentation:
    """Decode YOLO-seg predictions into boxes and packed prototype masks.

    ``layout='standard'`` accepts ``[1, 4+C+M, N]`` (YOLOv8/YOLO11), while
    ``layout='end2end'`` accepts ``[1, N, 6+M]`` rows containing
    ``xyxy, score, class_id, mask_coefficients`` (YOLO26).
    """

    def __init__(
        self,
        prediction: str = "output0",
        prototypes: str = "output1",
        boxes: str = "boxes",
        class_scores: str = "class_scores",
        scores: str = "scores",
        labels: str = "labels",
        mask_coefficients: str = "mask_coefficients",
        class_scores_are_logits: bool = False,
        num_classes: int = 80,
        layout: str = "standard",
        conf_threshold: float = 0.001,
        iou_threshold: float = 0.7,
        mask_threshold: float = 0.5,
        max_detections: int = 100,
        input_size: tuple[int, int] | list[int] = (640, 640),
    ) -> None:
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if layout not in {
            "standard",
            "end2end",
            "split_standard",
            "split_end2end",
        }:
            raise ValueError(f"Unsupported YOLO segmentation layout: {layout!r}")
        if not 0.0 <= conf_threshold <= 1.0:
            raise ValueError("conf_threshold must be between 0 and 1")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1")
        if not 0.0 <= mask_threshold <= 1.0:
            raise ValueError("mask_threshold must be between 0 and 1")
        if max_detections <= 0:
            raise ValueError("max_detections must be positive")
        if len(input_size) != 2 or any(int(value) <= 0 for value in input_size):
            raise ValueError("input_size must contain positive [height, width]")
        self.prediction = prediction
        self.prototypes = prototypes
        self.boxes = boxes
        self.class_scores = class_scores
        self.scores = scores
        self.labels = labels
        self.mask_coefficients = mask_coefficients
        self.class_scores_are_logits = bool(class_scores_are_logits)
        self.num_classes = int(num_classes)
        self.layout = layout
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.mask_threshold = float(mask_threshold)
        self.max_detections = int(max_detections)
        self.input_height, self.input_width = (int(value) for value in input_size)

    def _standard(
        self, prediction: np.ndarray, mask_dim: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        expected_channels = 4 + self.num_classes + mask_dim
        if prediction.ndim != 3 or prediction.shape[0] != 1:
            raise ValueError(
                f"Standard YOLO-seg prediction must be [1,C,N], got {prediction.shape}"
            )
        if prediction.shape[1] != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} prediction channels, got {prediction.shape[1]}"
            )
        values = prediction[0].T
        boxes = _cxcywh_to_xyxy(values[:, :4])
        class_scores = values[:, 4 : 4 + self.num_classes]
        labels = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(len(values)), labels]
        coefficients = values[:, 4 + self.num_classes :]
        candidates = np.flatnonzero(scores >= self.conf_threshold)
        boxes, scores, labels, coefficients = (
            boxes[candidates],
            scores[candidates],
            labels[candidates],
            coefficients[candidates],
        )
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes, scores, labels, coefficients = (
            boxes[valid],
            scores[valid],
            labels[valid],
            coefficients[valid],
        )

        kept: list[int] = []
        for class_index in np.unique(labels):
            positions = np.flatnonzero(labels == class_index)
            kept.extend(
                positions[
                    _nms(boxes[positions], scores[positions], self.iou_threshold)
                ].tolist()
            )
        kept = sorted(kept, key=lambda index: float(scores[index]), reverse=True)[
            : self.max_detections
        ]
        indices = np.asarray(kept, dtype=np.int64)
        return boxes[indices], scores[indices], labels[indices], coefficients[indices]

    def _end2end(
        self, prediction: np.ndarray, mask_dim: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        expected_columns = 6 + mask_dim
        if (
            prediction.ndim != 3
            or prediction.shape[0] != 1
            or prediction.shape[2] != expected_columns
        ):
            raise ValueError(
                f"End-to-end YOLO-seg prediction must be [1,N,{expected_columns}], "
                f"got {prediction.shape}"
            )
        values = prediction[0]
        scores = values[:, 4]
        labels = np.rint(values[:, 5]).astype(np.int64)
        keep = scores >= self.conf_threshold
        keep &= (labels >= 0) & (labels < self.num_classes)
        keep &= (values[:, 2] > values[:, 0]) & (values[:, 3] > values[:, 1])
        indices = np.flatnonzero(keep)
        indices = indices[np.argsort(scores[indices])[::-1]][: self.max_detections]
        return (
            values[indices, :4],
            scores[indices],
            labels[indices],
            values[indices, 6:],
        )

    def _masks(
        self,
        boxes: np.ndarray,
        coefficients: np.ndarray,
        prototypes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        _, mask_height, mask_width = prototypes.shape
        if len(boxes) == 0:
            packed_width = (mask_height * mask_width + 7) // 8
            return (
                np.zeros((0, packed_width), dtype=np.uint8),
                np.asarray([mask_height, mask_width], dtype=np.int32),
            )
        logits = coefficients @ prototypes.reshape(prototypes.shape[0], -1)
        masks = _sigmoid(logits).reshape(-1, mask_height, mask_width)

        scaled = boxes.astype(np.float32, copy=True)
        scaled[:, [0, 2]] *= mask_width / self.input_width
        scaled[:, [1, 3]] *= mask_height / self.input_height
        columns = np.arange(mask_width, dtype=np.float32)[None, None, :]
        rows = np.arange(mask_height, dtype=np.float32)[None, :, None]
        crop = (
            (columns >= scaled[:, 0, None, None])
            & (columns < scaled[:, 2, None, None])
            & (rows >= scaled[:, 1, None, None])
            & (rows < scaled[:, 3, None, None])
        )
        binary = (masks >= self.mask_threshold) & crop
        packed = np.packbits(binary.reshape(len(binary), -1), axis=1, bitorder="little")
        return packed, np.asarray([mask_height, mask_width], dtype=np.int32)

    def __call__(self, sample: dict[str, Any], result: Any) -> dict[str, Any]:
        prototypes = np.asarray(result[self.prototypes], dtype=np.float32)
        if prototypes.ndim != 4 or prototypes.shape[0] != 1:
            raise ValueError(
                f"YOLO mask prototypes must have shape [1,M,H,W], got {prototypes.shape}"
            )
        prototypes = prototypes[0]
        if self.layout == "standard":
            prediction = np.asarray(result[self.prediction], dtype=np.float32)
            boxes, scores, labels, coefficients = self._standard(
                prediction, prototypes.shape[0]
            )
        elif self.layout == "end2end":
            prediction = np.asarray(result[self.prediction], dtype=np.float32)
            boxes, scores, labels, coefficients = self._end2end(
                prediction, prototypes.shape[0]
            )
        elif self.layout == "split_standard":
            class_scores = np.asarray(result[self.class_scores], dtype=np.float32)
            if self.class_scores_are_logits:
                class_scores = _sigmoid(class_scores)
            prediction = np.concatenate(
                [
                    np.asarray(result[self.boxes], dtype=np.float32),
                    class_scores,
                    np.asarray(result[self.mask_coefficients], dtype=np.float32),
                ],
                axis=1,
            )
            boxes, scores, labels, coefficients = self._standard(
                prediction, prototypes.shape[0]
            )
        else:
            prediction = np.concatenate(
                [
                    np.asarray(result[self.boxes], dtype=np.float32),
                    np.asarray(result[self.scores], dtype=np.float32),
                    np.asarray(result[self.labels], dtype=np.float32),
                    np.asarray(result[self.mask_coefficients], dtype=np.float32),
                ],
                axis=2,
            )
            boxes, scores, labels, coefficients = self._end2end(
                prediction, prototypes.shape[0]
            )
        boxes = boxes.astype(np.float32, copy=False)
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, self.input_width)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, self.input_height)
        packed_masks, mask_shape = self._masks(boxes, coefficients, prototypes)
        sample["pred"] = {
            "boxes": boxes,
            "scores": scores.astype(np.float32, copy=False),
            "labels": labels.astype(np.int64, copy=False),
            "masks": packed_masks,
            "mask_shape": mask_shape,
        }
        return sample


QnnYoloSeg = QnnYoloSegmentation


class MtnnYoloSegmentation:
    """Adapt ordered MTNN outputs to the split YOLO segmentation decoder."""

    def __init__(
        self,
        boxes_index: int = 0,
        class_scores_index: int = 1,
        mask_coefficients_index: int = 2,
        prototypes_index: int = 3,
        **kwargs: Any,
    ) -> None:
        self.boxes_index = int(boxes_index)
        self.class_scores_index = int(class_scores_index)
        self.mask_coefficients_index = int(mask_coefficients_index)
        self.prototypes_index = int(prototypes_index)
        self.decoder = QnnYoloSegmentation(
            layout="split_standard",
            boxes="boxes",
            class_scores="class_scores",
            mask_coefficients="mask_coefficients",
            prototypes="prototypes",
            **kwargs,
        )

    def __call__(self, sample: dict[str, Any], result: Any) -> dict[str, Any]:
        if not isinstance(result, (list, tuple)):
            raise TypeError("MTNN YOLO-seg output must be an ordered list of tensors")
        outputs = {
            "boxes": result[self.boxes_index],
            "class_scores": result[self.class_scores_index],
            "mask_coefficients": result[self.mask_coefficients_index],
            "prototypes": result[self.prototypes_index],
        }
        return self.decoder(sample, outputs)
