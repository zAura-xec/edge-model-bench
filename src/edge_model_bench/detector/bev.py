"""Output adapters for BEVDet and CVT ONNX/QNN models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from edge_model_bench.dataset.nuscenes import NUSCENES_DETECTION_CLASSES


def _output(result: Any, name: str, index: int | None = None) -> np.ndarray:
    if isinstance(result, Mapping):
        return np.asarray(result[name])
    if index is not None and isinstance(result, Sequence):
        return np.asarray(result[index])
    try:
        return np.asarray(result[name])
    except (KeyError, IndexError, TypeError) as error:
        raise KeyError(f"Output {name!r} is unavailable") from error


class BevDetDecoder:
    """Decode CenterPoint-style BEVDet heads into ego-frame 3D boxes."""

    def __init__(
        self,
        score_threshold: float = 0.2,
        max_detections: int = 100,
        voxel_size: tuple[float, float] = (0.8, 0.8),
        pc_range: tuple[float, float] = (-51.2, -51.2),
    ) -> None:
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self.voxel_size = voxel_size
        self.pc_range = pc_range

    def __call__(self, sample: dict[str, Any], result: Any) -> dict[str, Any]:
        heatmap = _output(result, "heatmap", 5)
        reg = _output(result, "reg", 0)
        height = _output(result, "height", 1)
        dim = _output(result, "dim", 2)
        rot = _output(result, "rot", 3)
        vel = _output(result, "vel", 4)
        heatmap = heatmap[0]
        reg, height, dim, rot, vel = reg[0], height[0], dim[0], rot[0], vel[0]
        scores = heatmap.reshape(heatmap.shape[0], -1)
        candidates = []
        for cls, values in enumerate(scores):
            for flat in np.flatnonzero(values >= self.score_threshold):
                y, x = np.unravel_index(flat, values.shape[-2:] if values.ndim > 2 else heatmap.shape[1:])
                candidates.append((float(values[flat]), cls, int(x), int(y)))
        candidates.sort(reverse=True)
        boxes = []
        for score, cls, x, y in candidates[: self.max_detections]:
            dx, dy = reg[:, y, x]
            cx = (x + float(dx)) * self.voxel_size[0] + self.pc_range[0]
            cy = (y + float(dy)) * self.voxel_size[1] + self.pc_range[1]
            dimensions = np.exp(dim[:, y, x]).astype(float)
            yaw = float(np.arctan2(rot[0, y, x], rot[1, y, x]))
            boxes.append(
                {
                    "center": [cx, cy, float(height[0, y, x])],
                    "size": dimensions.tolist(),
                    "yaw": yaw,
                    "velocity": vel[:, y, x].astype(float).tolist(),
                    "score": score,
                    "label": cls,
                    "class_name": NUSCENES_DETECTION_CLASSES[cls],
                }
            )
        sample["pred_boxes"] = boxes
        return sample


class QnnBEVDet(BevDetDecoder):
    pass


class OnnxBEVDet(BevDetDecoder):
    pass


class CvtBevDecoder:
    """Decode CVT's single BEV foreground-logit output."""

    def __init__(self, threshold: float = 0.0, output: str = "bev") -> None:
        self.threshold = threshold
        self.output = output

    def __call__(self, sample: dict[str, Any], result: Any) -> dict[str, Any]:
        bev = _output(result, self.output, 0)
        bev = bev.reshape(bev.shape[-2:]).astype(np.float32)
        sample["pred_bev"] = bev >= self.threshold
        sample["pred_bev_logits"] = bev
        return sample


class QnnCVT(CvtBevDecoder):
    pass


class OnnxCVT(CvtBevDecoder):
    pass
