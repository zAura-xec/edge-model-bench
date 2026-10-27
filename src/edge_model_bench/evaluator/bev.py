"""Lightweight nuScenes BEV detection and occupancy evaluation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

import cv2
import numpy as np

from edge_model_bench.dataset.nuscenes import quaternion_to_rotation


CVT_VEHICLE_CLASSES = {
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "motorcycle",
    "bicycle",
}


def _center_distance(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    return float(np.linalg.norm(np.asarray(a["center"][:2]) - np.asarray(b["translation"][:2])))


class NuScenesBevDetectionEvaluator:
    """Class-wise center-distance AP proxy for nuScenes mini annotations."""

    def __init__(self, distance_threshold: float = 2.0) -> None:
        self.distance_threshold = distance_threshold

    def __call__(self, samples: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        samples = list(samples)
        matched = defaultdict(int)
        predicted = defaultdict(int)
        ground_truth = defaultdict(int)
        errors: list[float] = []
        for sample in samples:
            annotations = sample.get("annotations", [])
            predictions = sorted(sample.get("pred_boxes", []), key=lambda x: x.get("score", 0), reverse=True)
            used: set[int] = set()
            for ann in annotations:
                name = ann.get("category_name")
                if name:
                    ground_truth[name] += 1
            for prediction in predictions:
                name = prediction.get("class_name")
                predicted[name] += 1
                candidates = [
                    (distance, i)
                    for i, ann in enumerate(annotations)
                    if i not in used and ann.get("category_name") == name
                    for distance in [_center_distance(prediction, ann)]
                ]
                if candidates:
                    distance, index = min(candidates)
                    if distance <= self.distance_threshold:
                        used.add(index)
                        matched[name] += 1
                        errors.append(distance)
        classes = sorted(set(ground_truth) | set(predicted))
        ap = {
            name: (matched[name] / max(ground_truth[name], predicted[name], 1))
            for name in classes
        }
        return {
            "num_samples": len(samples),
            "classes": classes,
            "matched": dict(matched),
            "ground_truth": dict(ground_truth),
            "predicted": dict(predicted),
            "class_ap": ap,
            "mean_ap": float(np.mean(list(ap.values()))) if ap else 0.0,
            "mean_center_error": float(np.mean(errors)) if errors else None,
            "distance_threshold_m": self.distance_threshold,
        }


class NuScenesCvtBevEvaluator:
    """IoU evaluator for CVT's 200x200 BEV foreground map."""

    def __init__(self, grid_size: int = 200, pc_range: tuple[float, float, float, float] = (-50, -50, 50, 50)) -> None:
        self.grid_size = grid_size
        self.pc_range = pc_range

    def _target(self, sample: Mapping[str, Any]) -> np.ndarray:
        target = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        x0, y0, x1, y1 = self.pc_range
        scale_x = self.grid_size / (x1 - x0)
        scale_y = self.grid_size / (y1 - y0)
        view = np.asarray(
            [[0, -scale_y, -y0 * scale_y], [-scale_x, 0, x1 * scale_x], [0, 0, 1]],
            dtype=np.float32,
        )
        global_to_ego = np.asarray(sample.get("global_to_cvt_key", np.eye(4)), dtype=np.float32)
        for ann in sample.get("annotations", []):
            if ann.get("category_name") not in CVT_VEHICLE_CLASSES:
                continue
            width, length = map(float, ann.get("size", [0, 0])[:2])
            local = np.asarray(
                [
                    [length / 2, length / 2, -length / 2, -length / 2],
                    [width / 2, -width / 2, -width / 2, width / 2],
                    [0, 0, 0, 0],
                ],
                dtype=np.float32,
            )
            rotation = quaternion_to_rotation(ann.get("rotation", [1, 0, 0, 0]))
            corners = rotation @ local + np.asarray(ann.get("translation", [0, 0, 0]), dtype=np.float32)[:, None]
            corners = np.concatenate((corners, np.ones((1, 4), dtype=np.float32)))
            ego = global_to_ego @ corners
            pixels = (view @ ego[[0, 1, 3]])[:2].T.round().astype(np.int32)
            cv2.fillConvexPoly(target.view(np.uint8), pixels, 1)
        return target

    def __call__(self, samples: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        ious = []
        for sample in samples:
            prediction = np.asarray(sample.get("pred_bev", False), dtype=bool)
            target = self._target(sample)
            intersection = np.logical_and(prediction, target).sum()
            union = np.logical_or(prediction, target).sum()
            ious.append(float(intersection / union) if union else 1.0)
        return {"num_samples": len(ious), "bev_iou": float(np.mean(ious)) if ious else 0.0}


BevDetectionEvaluator = NuScenesBevDetectionEvaluator
CvtBevEvaluator = NuScenesCvtBevEvaluator
