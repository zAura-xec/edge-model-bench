"""Create deterministic ADE20K calibration and validation manifests."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from edge_model_bench.dataset.ade20k import (
    ADE20K_CLASSES,
    list_ade20k_images,
    resolve_ade20k_release,
)


def _collect_images(release: Path, split: str, prefix: str) -> list[Path]:
    del split
    return list_ade20k_images(release, prefix)


def _sample(items: list[Path], count: int, seed: int, split: str) -> list[Path]:
    if count <= 0:
        raise ValueError("Subset count must be positive")
    if count > len(items):
        raise ValueError(
            f"Requested {count} samples from ADE20K {split}, which has only {len(items)}"
        )
    return random.Random(seed).sample(items, count)


def create_subsets(
    root: str | Path,
    output_dir: str | Path | None = None,
    *,
    calibration_count: int = 512,
    validation_count: int = 1000,
    seed: int = 0,
) -> tuple[Path, Path]:
    outer_root = Path(root).resolve()
    release = resolve_ade20k_release(outer_root).resolve()
    destination = Path(output_dir).resolve() if output_dir else outer_root
    destination.mkdir(parents=True, exist_ok=True)

    training = _sample(
        _collect_images(release, "training", "ADE_train_"),
        calibration_count,
        seed,
        "training",
    )
    validation = _sample(
        _collect_images(release, "validation", "ADE_val_"),
        validation_count,
        seed,
        "validation",
    )

    calibration_path = destination / f"calibration_{calibration_count}.txt"
    calibration_path.write_text(
        "".join(f"{path.resolve()}\n" for path in training), encoding="utf-8"
    )

    validation_path = destination / f"validation_{validation_count}.csv"
    with validation_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "label_path"])
        for image_path in validation:
            label_path = image_path.with_name(f"{image_path.stem}_seg.png")
            if not label_path.is_file():
                raise FileNotFoundError(f"ADE20K mask not found: {label_path}")
            writer.writerow([image_path.resolve(), label_path.resolve()])

    (destination / "classes.txt").write_text(
        "".join(f"{name}\n" for name in ADE20K_CLASSES), encoding="utf-8"
    )
    return calibration_path, validation_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create ADE20K segmentation calibration TXT and validation CSV."
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--calibration-count", type=int, default=512)
    parser.add_argument("--validation-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    calibration, validation = create_subsets(
        args.root,
        args.output_dir,
        calibration_count=args.calibration_count,
        validation_count=args.validation_count,
        seed=args.seed,
    )
    print(f"Calibration subset: {calibration}")
    print(f"Validation subset: {validation}")


if __name__ == "__main__":
    main()
