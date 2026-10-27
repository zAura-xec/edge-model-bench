"""Randomly sample images from a directory and write a TXT list."""

import argparse
import os
import random
from pathlib import Path

from tqdm.auto import tqdm


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("count", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    images = []
    for directory, _, filenames in tqdm(
        os.walk(args.image_dir), desc="Scanning directories", unit="dir"
    ):
        images.extend(
            Path(directory, filename)
            for filename in filenames
            if Path(filename).suffix.lower() in IMAGE_SUFFIXES
        )
    selected = random.Random(args.seed).sample(images, args.count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(str(path.absolute()) for path in selected) + "\n")


if __name__ == "__main__":
    main()
