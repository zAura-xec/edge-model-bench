"""Create deterministic VOC segmentation calibration and validation subsets.

The output layout mirrors the COCO2017 benchmark manifests:

* ``calibration_512.txt`` contains image paths sampled from the train split.
* ``validation_1000.csv`` contains image/mask path pairs sampled from val.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def _resolve_voc_root(root: Path, year: str) -> Path:
    expected_name = f"VOC{year}"
    return root if root.name == expected_name else root / expected_name


def _read_split(voc_root: Path, split: str) -> list[str]:
    split_path = voc_root / "ImageSets" / "Segmentation" / f"{split}.txt"
    if not split_path.is_file():
        raise FileNotFoundError(f"VOC segmentation split not found: {split_path}")
    return [line.strip() for line in split_path.read_text().splitlines() if line.strip()]


def _sample(items: list[str], count: int, seed: int, split: str) -> list[str]:
    if count <= 0:
        raise ValueError("Subset count must be positive")
    if count > len(items):
        raise ValueError(
            f"Requested {count} samples from VOC {split}, which has only {len(items)}"
        )
    return random.Random(seed).sample(items, count)


def create_subsets(
    root: str | Path,
    output_dir: str | Path | None = None,
    *,
    year: str = "2012",
    calibration_count: int = 512,
    validation_count: int = 1000,
    seed: int = 0,
) -> tuple[Path, Path]:
    root_path = Path(root).resolve()
    voc_root = _resolve_voc_root(root_path, year)
    destination = (
        Path(output_dir).resolve()
        if output_dir is not None
        else (voc_root.parent if root_path.name == f"VOC{year}" else root_path)
    )
    destination.mkdir(parents=True, exist_ok=True)

    train_ids = _sample(
        _read_split(voc_root, "train"), calibration_count, seed, "train"
    )
    val_ids = _sample(
        _read_split(voc_root, "val"), validation_count, seed, "val"
    )

    calibration_path = destination / f"calibration_{calibration_count}.txt"
    calibration_images = [
        (voc_root / "JPEGImages" / f"{image_id}.jpg").resolve()
        for image_id in train_ids
    ]
    missing_images = [path for path in calibration_images if not path.is_file()]
    if missing_images:
        raise FileNotFoundError(f"VOC calibration image not found: {missing_images[0]}")
    calibration_path.write_text(
        "".join(f"{path}\n" for path in calibration_images), encoding="utf-8"
    )

    validation_path = destination / f"validation_{validation_count}.csv"
    validation_rows: list[tuple[Path, Path]] = []
    for image_id in val_ids:
        image_path = (voc_root / "JPEGImages" / f"{image_id}.jpg").resolve()
        label_path = (voc_root / "SegmentationClass" / f"{image_id}.png").resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"VOC validation image not found: {image_path}")
        if not label_path.is_file():
            raise FileNotFoundError(f"VOC validation mask not found: {label_path}")
        validation_rows.append((image_path, label_path))

    with validation_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "label_path"])
        writer.writerows(validation_rows)

    return calibration_path, validation_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create VOC segmentation calibration TXT and validation CSV."
    )
    parser.add_argument("root", type=Path, help="VOCdevkit or VOC<year> directory")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--year", default="2012")
    parser.add_argument("--calibration-count", type=int, default=512)
    parser.add_argument("--validation-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    calibration_path, validation_path = create_subsets(
        args.root,
        args.output_dir,
        year=args.year,
        calibration_count=args.calibration_count,
        validation_count=args.validation_count,
        seed=args.seed,
    )
    print(f"Calibration subset: {calibration_path}")
    print(f"Validation subset: {validation_path}")


if __name__ == "__main__":
    main()
