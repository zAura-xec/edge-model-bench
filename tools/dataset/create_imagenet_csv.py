"""Randomly sample an ImageNet validation directory and write a labeled CSV."""

import argparse
import csv
import os
import random
from pathlib import Path

from tqdm.auto import tqdm


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("imagenet_dir", type=Path)
    parser.add_argument("class_txt", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("count", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    classes = {
        synset.strip(): label
        for label, synset in enumerate(args.class_txt.read_text().splitlines())
        if synset.strip()
    }

    images = []
    for directory, _, filenames in tqdm(
        os.walk(args.imagenet_dir), desc="Scanning directories", unit="dir"
    ):
        if Path(directory).name in classes:
            images.extend(
                Path(directory, filename)
                for filename in filenames
                if Path(filename).suffix.lower() in IMAGE_SUFFIXES
            )
    selected = random.Random(args.seed).sample(images, args.count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "label"])
        writer.writerows((path.absolute(), classes[path.parent.name]) for path in selected)


if __name__ == "__main__":
    main()
