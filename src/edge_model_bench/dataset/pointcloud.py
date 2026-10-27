"""ModelNet10 point-cloud loading and preprocessing.

The PointNet artifact used by this benchmark consumes a ``[1, 3, 1024]``
float32 tensor.  ModelNet10 stores triangle meshes in OFF files, so this
module deliberately keeps the mesh reader dependency-free and samples the
mesh vertices into a deterministic point cloud.
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np


MODELNET10_CLASSES = (
    "bathtub",
    "bed",
    "chair",
    "desk",
    "dresser",
    "monitor",
    "night_stand",
    "sofa",
    "table",
    "toilet",
)
MODELNET10_CLASS_TO_INDEX = {name: index for index, name in enumerate(MODELNET10_CLASSES)}


def read_off(path: str | Path) -> np.ndarray:
    """Read the vertex coordinates from an OFF mesh."""

    path = Path(path)
    with path.open(encoding="utf-8", errors="replace") as file:
        lines = (line.strip() for line in file)
        header = next(lines, "")
        if header != "OFF":
            # A few exporters put the counts on the same line as the header.
            if not header.startswith("OFF"):
                raise ValueError(f"{path} is not an OFF mesh")
            counts = header[3:].strip()
        else:
            counts = ""
        while not counts:
            counts = next(lines, "")
        n_vertices = int(counts.split()[0])
        vertices: list[list[float]] = []
        while len(vertices) < n_vertices:
            values = next(lines).split()
            if len(values) >= 3:
                vertices.append([float(values[0]), float(values[1]), float(values[2])])
    if not vertices:
        raise ValueError(f"{path} contains no vertices")
    return np.asarray(vertices, dtype=np.float32)


def _stable_seed(path: Path, seed: int) -> int:
    digest = hashlib.blake2b(str(path).encode(), digest_size=8).digest()
    return (int.from_bytes(digest, "little") + seed) & 0xFFFFFFFF


def sample_points(
    vertices: np.ndarray,
    num_points: int = 1024,
    seed: int = 0,
    path: str | Path | None = None,
) -> np.ndarray:
    """Select exactly ``num_points`` vertices, with replacement if necessary."""

    vertices = np.asarray(vertices, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] < 3:
        raise ValueError(f"Expected [N,3] vertices, got {vertices.shape}")
    rng = np.random.default_rng(_stable_seed(Path(path), seed) if path else seed)
    replace = len(vertices) < num_points
    indices = rng.choice(len(vertices), size=num_points, replace=replace)
    return np.ascontiguousarray(vertices[indices, :3], dtype=np.float32)


def normalize_points(points: np.ndarray) -> np.ndarray:
    """Center a point cloud and scale it into the unit ball."""

    points = np.asarray(points, dtype=np.float32)
    centered = points - points.mean(axis=0, keepdims=True)
    scale = float(np.linalg.norm(centered, axis=1).max())
    if scale > 1e-8:
        centered = centered / scale
    return np.ascontiguousarray(centered, dtype=np.float32)


class LoadingPointCloud:
    def __call__(self, path: str | Path) -> np.ndarray:
        return read_off(path)


class SamplePointCloud:
    def __init__(self, num_points: int = 1024, seed: int = 0) -> None:
        self.num_points = num_points
        self.seed = seed
        self.path: Path | None = None

    def __call__(self, points: np.ndarray) -> np.ndarray:
        # The loader records the path when used by the dataset.  A fixed seed
        # still gives reproducible standalone transform use.
        return sample_points(points, self.num_points, self.seed, self.path)


class NormalizePointCloud:
    def __call__(self, points: np.ndarray) -> np.ndarray:
        return normalize_points(points)


class PointCloudToNCHW:
    """Convert ``[N,3]`` points into PointNet's ``[1,3,N]`` layout."""

    def __call__(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Expected [N,3] points, got {points.shape}")
        return np.ascontiguousarray(points.T[None], dtype=np.float32)


class ModelNet10:
    """ModelNet10 samples described by a CSV with ``point_path,label``."""

    def __init__(
        self,
        csv_path: str | Path,
        pipeline: list[Callable] | None = None,
        class_mapping: dict[str, int] | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.pipeline = pipeline or []
        mapping = class_mapping or MODELNET10_CLASS_TO_INDEX
        with self.csv_path.open(encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            self.samples: list[dict[str, Any]] = []
            for row in reader:
                path_value = row.get("point_path") or row.get("image_path") or row.get("path")
                if not path_value:
                    raise ValueError(f"Missing point_path in {self.csv_path}")
                path = Path(path_value)
                if not path.is_absolute():
                    path = self.csv_path.parent / path
                class_name = row.get("class_name") or path.parent.name
                label = int(row["label"]) if row.get("label") not in (None, "") else mapping[class_name]
                self.samples.append(
                    {"point_path": str(path), "label": label, "class_name": class_name}
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[np.ndarray, dict[str, Any]]:
        sample = self.samples[index]
        value: Any = sample["point_path"]
        for transform in self.pipeline:
            # Sampling can use the source path to keep calibration deterministic.
            if isinstance(transform, SamplePointCloud):
                transform.path = Path(sample["point_path"])
            value = transform(value)
        return value, sample

    def __call__(self) -> list[dict[str, Any]]:
        return self.samples


ModelNet10PointCloud = ModelNet10
