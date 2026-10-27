"""Convert COCO instance annotations to YOLO segmentation label files.

Each output line has the standard form::

    class_id x1 y1 x2 y2 ... xn yn

Coordinates are normalized to ``[0, 1]``. COCO crowd annotations use RLE masks
and are skipped. Multiple disconnected polygons belonging to one COCO instance
are joined into one polygon line by their nearest vertices.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm


def _load_coco(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    for key in ("images", "annotations", "categories"):
        if key not in data:
            raise KeyError(f"COCO JSON missing required key: {key}")
    return data


def _category_map(categories: list[dict[str, Any]]) -> dict[int, tuple[int, str]]:
    ordered = sorted(
        ((int(category["id"]), str(category["name"])) for category in categories),
        key=lambda item: item[0],
    )
    return {
        category_id: (class_id, name)
        for class_id, (category_id, name) in enumerate(ordered)
    }


def _polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2.0)


def _valid_polygons(segmentation: Any, width: float, height: float) -> list[np.ndarray]:
    if not isinstance(segmentation, list):
        return []
    polygons: list[np.ndarray] = []
    for coordinates in segmentation:
        if not isinstance(coordinates, list) or len(coordinates) < 6 or len(coordinates) % 2:
            continue
        points = np.asarray(coordinates, dtype=np.float64).reshape(-1, 2)
        if not np.isfinite(points).all():
            continue
        points[:, 0] = np.clip(points[:, 0], 0.0, width)
        points[:, 1] = np.clip(points[:, 1], 0.0, height)
        if np.unique(points, axis=0).shape[0] < 3 or _polygon_area(points) <= 0.0:
            continue
        polygons.append(points)
    return polygons


def _merge_polygons(polygons: list[np.ndarray]) -> np.ndarray:
    """Join disconnected polygons into one deterministic vertex chain."""
    if len(polygons) == 1:
        return polygons[0]

    remaining = sorted(polygons, key=_polygon_area, reverse=True)
    merged = remaining.pop(0)
    while remaining:
        best: tuple[float, int, int, bool] | None = None
        for polygon_index, polygon in enumerate(remaining):
            distances = np.sum((polygon - merged[-1]) ** 2, axis=1)
            point_index = int(np.argmin(distances))
            candidate = (
                float(distances[point_index]),
                polygon_index,
                point_index,
                False,
            )
            reversed_polygon = polygon[::-1]
            reversed_distances = np.sum((reversed_polygon - merged[-1]) ** 2, axis=1)
            reversed_index = int(np.argmin(reversed_distances))
            reversed_candidate = (
                float(reversed_distances[reversed_index]),
                polygon_index,
                reversed_index,
                True,
            )
            for option in (candidate, reversed_candidate):
                if best is None or option < best:
                    best = option

        assert best is not None
        _, polygon_index, point_index, reverse = best
        selected = remaining.pop(polygon_index)
        if reverse:
            selected = selected[::-1]
        selected = np.roll(selected, -point_index, axis=0)
        merged = np.concatenate((merged, selected), axis=0)
    return merged


def convert(
    coco_json: str | Path,
    output_dir: str | Path,
    *,
    classes_file: str | Path | None = None,
    empty_labels: bool = True,
) -> dict[str, int]:
    data = _load_coco(Path(coco_json))
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    category_map = _category_map(data["categories"])

    if classes_file is not None:
        ordered_names = sorted(category_map.values(), key=lambda item: item[0])
        Path(classes_file).write_text(
            "".join(f"{name}\n" for _, name in ordered_names), encoding="utf-8"
        )

    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in data["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    stats = {
        "images": 0,
        "labels_written": 0,
        "instances_written": 0,
        "instances_skipped_crowd": 0,
        "instances_skipped_invalid": 0,
        "multi_polygon_instances": 0,
        "empty_labels": 0,
    }
    for image in tqdm(data["images"], desc="Converting", unit="image"):
        stats["images"] += 1
        image_id = int(image["id"])
        width = float(image["width"])
        height = float(image["height"])
        lines: list[str] = []

        for annotation in annotations_by_image.get(image_id, []):
            if int(annotation.get("iscrowd", 0)) == 1:
                stats["instances_skipped_crowd"] += 1
                continue
            polygons = _valid_polygons(annotation.get("segmentation"), width, height)
            if not polygons:
                stats["instances_skipped_invalid"] += 1
                continue
            if len(polygons) > 1:
                stats["multi_polygon_instances"] += 1
            points = _merge_polygons(polygons)
            points[:, 0] /= width
            points[:, 1] /= height
            category_id = int(annotation["category_id"])
            class_id = category_map[category_id][0]
            coordinates = " ".join(f"{value:.6f}" for value in points.reshape(-1))
            lines.append(f"{class_id} {coordinates}")
            stats["instances_written"] += 1

        if lines or empty_labels:
            file_name = Path(str(image["file_name"])).stem + ".txt"
            (output_path / file_name).write_text(
                ("\n".join(lines) + "\n") if lines else "", encoding="utf-8"
            )
            stats["labels_written"] += 1
            if not lines:
                stats["empty_labels"] += 1

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert COCO instances JSON to YOLO segmentation labels."
    )
    parser.add_argument("coco_json", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--classes-file", type=Path)
    parser.add_argument("--no-empty-labels", action="store_true")
    args = parser.parse_args()

    stats = convert(
        args.coco_json,
        args.output_dir,
        classes_file=args.classes_file,
        empty_labels=not args.no_empty_labels,
    )
    print("Conversion finished:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print(f"  labels_dir: {args.output_dir}")


if __name__ == "__main__":
    main()
