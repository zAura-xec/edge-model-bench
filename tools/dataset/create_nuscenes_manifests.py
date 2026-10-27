"""Create nuScenes mini calibration and validation manifests."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def create_manifests(root: Path, version: str = "v1.0-mini", calibration_size: int = 512, seed: int = 0) -> tuple[int, int]:
    version_root = root / version
    samples = json.loads((version_root / "sample.json").read_text())
    tokens = [row["token"] for row in samples]
    rng = random.Random(seed)
    if calibration_size <= len(tokens):
        calibration = rng.sample(tokens, calibration_size)
    else:
        calibration = [tokens[index % len(tokens)] for index in range(calibration_size)]
        rng.shuffle(calibration)
    (root / f"calibration_{calibration_size}.txt").write_text(
        "".join(f"{token}\n" for token in calibration),
        encoding="utf-8",
    )
    with (root / "validation.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("sample_token", "split"))
        writer.writeheader()
        writer.writerows({"sample_token": token, "split": "mini"} for token in tokens)
    return len(calibration), len(tokens)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--calibration-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    calibration, validation = create_manifests(args.root, args.version, args.calibration_size, args.seed)
    print(f"wrote {calibration} calibration tokens and {validation} validation tokens")


if __name__ == "__main__":
    main()
