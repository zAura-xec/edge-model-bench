"""Create deterministic ModelNet10 manifests for PointNet."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from edge_model_bench.dataset.pointcloud import MODELNET10_CLASS_TO_INDEX, MODELNET10_CLASSES


def _files(root: Path, split: str) -> list[Path]:
    return sorted(
        path
        for class_name in MODELNET10_CLASSES
        for path in (root / class_name / split).glob("*.off")
        if path.is_file()
    )


def create_subsets(root: Path, calibration_size: int = 512, seed: int = 0) -> tuple[int, int]:
    train = _files(root, "train")
    test = _files(root, "test")
    if not train or not test:
        raise FileNotFoundError(f"Expected ModelNet10/<class>/(train|test) under {root}")
    rng = random.Random(seed)
    calibration = train.copy()
    rng.shuffle(calibration)
    calibration = calibration[: min(calibration_size, len(calibration))]
    (root / "calibration_512.txt").write_text(
        "".join(f"{path.resolve()}\n" for path in calibration), encoding="utf-8"
    )

    with (root / "validation.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("point_path", "label", "class_name", "split"))
        writer.writeheader()
        for path in test:
            class_name = path.parent.parent.name
            writer.writerow(
                {
                    "point_path": str(path.resolve()),
                    "label": MODELNET10_CLASS_TO_INDEX[class_name],
                    "class_name": class_name,
                    "split": "test",
                }
            )
    (root / "classes.txt").write_text("\n".join(MODELNET10_CLASSES) + "\n", encoding="utf-8")
    return len(calibration), len(test)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--calibration-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    calibration, validation = create_subsets(args.root, args.calibration_size, args.seed)
    print(f"wrote {calibration} calibration samples and {validation} validation samples")


if __name__ == "__main__":
    main()
