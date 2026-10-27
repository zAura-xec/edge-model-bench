"""Pair images with YOLO label files and write a sampled CSV.

CSV columns: image_path,label_path

Example:
    python tools/dataset/create_image_label_csv.py \\
        /path/to/images \\
        /path/to/labels \\
        /path/to/output.csv \\
        1000 \\
        --seed 0
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from tqdm.auto import tqdm


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_pairs(
    image_dir: Path,
    label_dir: Path,
    *,
    skip_empty: bool = False,
) -> list[tuple[Path, Path]]:
    """Collect (image_path, label_path) pairs matched by stem."""
    label_by_stem: dict[str, Path] = {}
    for path in label_dir.iterdir():
        if path.is_file() and path.suffix.lower() == ".txt":
            if skip_empty and path.stat().st_size == 0:
                continue
            label_by_stem[path.stem] = path

    pairs: list[tuple[Path, Path]] = []
    for path in image_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label_path = label_by_stem.get(path.stem)
        if label_path is None:
            continue
        pairs.append((path.resolve(), label_path.resolve()))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample image/label pairs into a CSV file."
    )
    parser.add_argument("image_dir", type=Path, help="Directory containing images")
    parser.add_argument("label_dir", type=Path, help="Directory containing YOLO .txt labels")
    parser.add_argument("output", type=Path, help="Output CSV path")
    parser.add_argument("count", type=int, help="Number of pairs to sample")
    parser.add_argument("--seed", type=int, help="Random seed for sampling")
    parser.add_argument(
        "--skip-empty",
        action="store_true",
        help="Skip images whose label file is empty",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Write all matched pairs and ignore count",
    )
    args = parser.parse_args()

    pairs = collect_pairs(args.image_dir, args.label_dir, skip_empty=args.skip_empty)
    if not pairs:
        raise SystemExit("No matched image/label pairs found")

    if args.all:
        selected = pairs
    else:
        if args.count <= 0:
            raise SystemExit("count must be positive unless --all is set")
        if args.count > len(pairs):
            raise SystemExit(
                f"Requested {args.count} pairs but only {len(pairs)} are available"
            )
        selected = random.Random(args.seed).sample(pairs, args.count)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "label_path"])
        for image_path, label_path in tqdm(selected, desc="Writing CSV", unit="row"):
            writer.writerow([str(image_path), str(label_path)])

    print(f"Matched pairs: {len(pairs)}")
    print(f"Written pairs: {len(selected)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
