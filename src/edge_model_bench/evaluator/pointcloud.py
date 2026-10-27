"""ModelNet10 point-cloud evaluation."""

from __future__ import annotations

from typing import Any

from .cls_task import ClsTaskEvaluator
from edge_model_bench.dataset.pointcloud import MODELNET10_CLASSES


class ModelNet10Evaluator(ClsTaskEvaluator):
    def __init__(self, class_names: tuple[str, ...] = MODELNET10_CLASSES, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.class_names = tuple(class_names)

    def __call__(self, samples: Any) -> dict[str, Any]:
        result = super().__call__(samples)
        result["class_names"] = list(self.class_names)
        result["task"] = "ModelNet10 point-cloud classification"
        return result


# Explicit alias for callers that use the 3D segmentation terminology.
PointCloud3DEvaluator = ModelNet10Evaluator
PointNet3DEvaluator = ModelNet10Evaluator
