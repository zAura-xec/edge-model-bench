"""Convert COCO detection annotations to YOLO label files.

COCO bbox format: [x_min, y_min, width, height] in absolute pixels.
YOLO bbox format: class_id x_center y_center width height (normalized 0-1).

Example:
    python tools/dataset/coco_to_yolo.py \\
        /path/to/instances_val2017.json \\
        /path/to/labels \\
        --classes-file /path/to/classes.txt
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm


def coco_bbox_to_yolo(
    bbox: list[float],
    image_width: float,
    image_height: float,
) -> tuple[float, float, float, float]:
    """Convert a COCO xywh bbox to YOLO normalized cxcywh."""
    x_min, y_min, width, height = bbox
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"Invalid image size: {image_width}x{image_height}")
    if width < 0 or height < 0:
        raise ValueError(f"Invalid bbox size: width={width}, height={height}")

    x_center = (x_min + width / 2.0) / image_width
    y_center = (y_min + height / 2.0) / image_height
    norm_width = width / image_width
    norm_height = height / image_height
    return x_center, y_center, norm_width, norm_height


def clip_yolo_bbox(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
) -> tuple[float, float, float, float] | None:
    """Clip a YOLO bbox to [0, 1]. Return None if the box is empty after clipping."""
    x_min = max(0.0, x_center - width / 2.0)
    y_min = max(0.0, y_center - height / 2.0)
    x_max = min(1.0, x_center + width / 2.0)
    y_max = min(1.0, y_center + height / 2.0)
    if x_max <= x_min or y_max <= y_min:
        return None
    return (
        (x_min + x_max) / 2.0,
        (y_min + y_max) / 2.0,
        x_max - x_min,
        y_max - y_min,
    )


def build_category_map(
    categories: list[dict[str, Any]],
    keep_names: set[str] | None = None,
    keep_ids: set[int] | None = None,
) -> dict[int, tuple[int, str]]:
    """Map COCO category_id -> (yolo_class_id, category_name).

    YOLO class ids are reassigned contiguously starting from 0, ordered by
    COCO category id for stable output.
    """
    selected = []
    for category in categories:
        category_id = int(category["id"])
        name = str(category["name"])
        if keep_ids is not None and category_id not in keep_ids:
            continue
        if keep_names is not None and name not in keep_names:
            continue
        selected.append((category_id, name))

    selected.sort(key=lambda item: item[0])
    return {
        category_id: (yolo_id, name)
        for yolo_id, (category_id, name) in enumerate(selected)
    }


def load_coco(coco_json: Path) -> dict[str, Any]:
    with coco_json.open("r", encoding="utf-8") as file:
        data = json.load(file)
    for key in ("images", "annotations", "categories"):
        if key not in data:
            raise KeyError(f"COCO json missing required key: {key}")
    return data


def write_classes_file(path: Path, category_map: dict[int, tuple[int, str]]) -> None:
    ordered = sorted(category_map.values(), key=lambda item: item[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{name}\n" for _, name in ordered), encoding="utf-8")


def convert(
    coco_json: Path,
    output_dir: Path,
    *,
    classes_file: Path | None = None,
    skip_crowd: bool = True,
    keep_names: set[str] | None = None,
    keep_ids: set[int] | None = None,
    clip_bbox: bool = True,
    empty_labels: bool = True,
) -> dict[str, int]:
    """Convert COCO annotations to YOLO label files.

    Returns a small stats dict for logging.
    """
    data = load_coco(coco_json)
    category_map = build_category_map(
        data["categories"],
        keep_names=keep_names,
        keep_ids=keep_ids,
    )
    if not category_map:
        raise ValueError("No categories selected for conversion")

    images_by_id = {int(image["id"]): image for image in data["images"]}
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in data["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    output_dir.mkdir(parents=True, exist_ok=True)
    if classes_file is not None:
        write_classes_file(classes_file, category_map)

    stats = {
        "images": 0,
        "labels_written": 0,
        "boxes_written": 0,
        "boxes_skipped_crowd": 0,
        "boxes_skipped_category": 0,
        "boxes_skipped_invalid": 0,
        "empty_labels": 0,
    }

    for image_id, image in tqdm(images_by_id.items(), desc="Converting", unit="image"):
        stats["images"] += 1
        image_width = float(image["width"])
        image_height = float(image["height"])
        file_name = Path(image["file_name"]).name
        label_path = output_dir / f"{Path(file_name).stem}.txt"

        lines: list[str] = []
        for annotation in annotations_by_image.get(image_id, []):
            if skip_crowd and int(annotation.get("iscrowd", 0)) == 1:
                stats["boxes_skipped_crowd"] += 1
                continue

            category_id = int(annotation["category_id"])
            if category_id not in category_map:
                stats["boxes_skipped_category"] += 1
                continue

            try:
                yolo_bbox = coco_bbox_to_yolo(
                    annotation["bbox"],
                    image_width,
                    image_height,
                )
            except (KeyError, TypeError, ValueError):
                stats["boxes_skipped_invalid"] += 1
                continue

            if clip_bbox:
                clipped = clip_yolo_bbox(*yolo_bbox)
                if clipped is None:
                    stats["boxes_skipped_invalid"] += 1
                    continue
                yolo_bbox = clipped

            x_center, y_center, width, height = yolo_bbox
            if min(x_center, y_center, width, height) < 0 or max(
                x_center, y_center, width, height
            ) > 1:
                stats["boxes_skipped_invalid"] += 1
                continue

            yolo_class_id, _ = category_map[category_id]
            lines.append(
                f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )
            stats["boxes_written"] += 1

        if not lines and not empty_labels:
            continue

        label_path.write_text(
            ("\n".join(lines) + "\n") if lines else "",
            encoding="utf-8",
        )
        stats["labels_written"] += 1
        if not lines:
            stats["empty_labels"] += 1

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert COCO detection annotations to YOLO label files."
    )
    parser.add_argument("coco_json", type=Path, help="Path to COCO instances_*.json")
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory to write YOLO .txt label files",
    )
    parser.add_argument(
        "--classes-file",
        type=Path,
        help="Optional path to write class names (one per line, YOLO id order)",
    )
    parser.add_argument(
        "--category-names",
        nargs="+",
        help="Only keep these category names",
    )
    parser.add_argument(
        "--category-ids",
        nargs="+",
        type=int,
        help="Only keep these COCO category ids",
    )
    parser.add_argument(
        "--keep-crowd",
        action="store_true",
        help="Keep annotations with iscrowd=1 (skipped by default)",
    )
    parser.add_argument(
        "--no-clip-bbox",
        action="store_true",
        help="Do not clip normalized boxes to [0, 1]",
    )
    parser.add_argument(
        "--no-empty-labels",
        action="store_true",
        help="Do not write empty label files for images without boxes",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keep_names = set(args.category_names) if args.category_names else None
    keep_ids = set(args.category_ids) if args.category_ids else None

    stats = convert(
        args.coco_json,
        args.output_dir,
        classes_file=args.classes_file,
        skip_crowd=not args.keep_crowd,
        keep_names=keep_names,
        keep_ids=keep_ids,
        clip_bbox=not args.no_clip_bbox,
        empty_labels=not args.no_empty_labels,
    )

    print("Conversion finished:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    if args.classes_file is not None:
        print(f"  classes_file: {args.classes_file}")
    print(f"  labels_dir: {args.output_dir}")


if __name__ == "__main__":
    main()
