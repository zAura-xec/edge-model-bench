"""ADE20K semantic-segmentation dataset and ADE150 label mapping."""

from __future__ import annotations

import csv
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ADE20K_CLASSES: tuple[str, ...] = (
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed",
    "windowpane", "grass", "cabinet", "sidewalk", "person", "earth", "door",
    "table", "mountain", "plant", "curtain", "chair", "car", "water",
    "painting", "sofa", "shelf", "house", "sea", "mirror", "rug", "field",
    "armchair", "seat", "fence", "desk", "rock", "wardrobe", "lamp",
    "bathtub", "railing", "cushion", "base", "box", "column", "signboard",
    "chest of drawers", "counter", "sand", "sink", "skyscraper", "fireplace",
    "refrigerator", "grandstand", "path", "stairs", "runway", "case",
    "pool table", "pillow", "screen door", "stairway", "river", "bridge",
    "bookcase", "blind", "coffee table", "toilet", "flower", "book", "hill",
    "bench", "countertop", "stove", "palm", "kitchen island", "computer",
    "swivel chair", "boat", "bar", "arcade machine", "hovel", "bus", "towel",
    "light", "truck", "tower", "chandelier", "awning", "streetlight", "booth",
    "television", "airplane", "dirt track", "apparel", "pole", "land",
    "bannister", "escalator", "ottoman", "bottle", "buffet", "poster", "stage",
    "van", "ship", "fountain", "conveyer belt", "canopy", "washer",
    "plaything", "swimming pool", "stool", "barrel", "basket", "waterfall",
    "tent", "bag", "minibike", "cradle", "oven", "ball", "food", "step",
    "tank", "trade name", "microwave", "pot", "animal", "bicycle", "lake",
    "dishwasher", "screen", "blanket", "sculpture", "hood", "sconce", "vase",
    "traffic light", "tray", "ashcan", "fan", "pier", "crt screen", "plate",
    "monitor", "bulletin board", "shower", "radiator", "glass", "clock", "flag",
)

# Object indices encoded in ADE20K_2021 ``*_seg.png`` files, in ADE150 order.
ADE20K_RAW_CLASS_IDS: tuple[int, ...] = (
    2978, 312, 2420, 976, 2855, 447, 2131, 165, 3055, 1125, 350, 2377,
    1831, 838, 774, 2684, 1610, 1910, 687, 471, 401, 2994, 1735, 2473,
    2329, 1276, 2264, 1564, 2178, 913, 57, 2272, 907, 724, 2138, 2985,
    1395, 155, 2053, 689, 137, 266, 581, 2380, 491, 627, 2212, 2388,
    2423, 943, 2096, 1121, 1788, 2530, 2185, 420, 1948, 1869, 2251,
    2531, 2128, 294, 239, 212, 571, 2793, 978, 236, 1240, 181, 629,
    2598, 1744, 1374, 591, 2679, 223, 123, 47, 1282, 327, 2821, 1451,
    2880, 2828, 480, 77, 2616, 247, 2733, 14, 738, 38, 1936, 1401, 120,
    868, 1702, 249, 308, 1969, 2526, 2928, 2337, 1023, 609, 389, 2989,
    1930, 2668, 2586, 131, 146, 3016, 2739, 95, 1563, 642, 1708, 103,
    1002, 2569, 2704, 2833, 1551, 1981, 29, 187, 1393, 747, 2254, 206,
    2262, 1260, 2243, 2932, 2836, 2850, 64, 894, 1858, 3109, 1919,
    1583, 318, 2356, 2046, 1098, 530, 954,
)

_RAW_TO_ADE150 = np.full(65536, 255, dtype=np.uint8)
_RAW_TO_ADE150[np.asarray(ADE20K_RAW_CLASS_IDS)] = np.arange(150, dtype=np.uint8)


def resolve_ade20k_release(root: str | Path) -> Path:
    """Resolve either the outer ADE20k directory or a versioned release root."""
    path = Path(root)
    if (path / "images" / "ADE").is_dir():
        return path
    candidates = sorted(
        child for child in path.glob("ADE20K_*") if (child / "images" / "ADE").is_dir()
    )
    if not candidates:
        raise FileNotFoundError(f"ADE20K release not found below: {path}")
    return candidates[-1]


def list_ade20k_images(release: str | Path, prefix: str) -> list[Path]:
    """Read image paths from ADE20K's bundled index without scanning instances."""
    release_path = resolve_ade20k_release(release).resolve()
    index_path = release_path / "index_ade20k.pkl"
    if not index_path.is_file():
        raise FileNotFoundError(f"ADE20K index not found: {index_path}")
    with index_path.open("rb") as file:
        index = pickle.load(file)  # noqa: S301 - trusted file bundled with ADE20K
    filenames = index.get("filename")
    folders = index.get("folder")
    if not isinstance(filenames, list) or not isinstance(folders, list):
        raise ValueError(f"Invalid ADE20K index structure: {index_path}")
    if len(filenames) != len(folders):
        raise ValueError("ADE20K index filename/folder lengths do not match")
    images = []
    for filename, folder in zip(filenames, folders, strict=True):
        if str(filename).startswith(prefix):
            folder_path = Path(folder)
            if not folder_path.is_absolute():
                folder_path = release_path.parent / folder_path
            images.append(folder_path / filename)
    if not images:
        raise FileNotFoundError(f"No ADE20K images with prefix {prefix!r} in {index_path}")
    return images


def decode_ade20k_mask(mask: np.ndarray) -> np.ndarray:
    """Convert raw ADE20K or ADEChallenge labels to zero-based ADE150 labels."""
    array = np.asarray(mask)
    if array.ndim == 3 and array.shape[2] >= 2:
        # Raw ADE20K encoding: object_id = floor(R / 10) * 256 + G.
        object_ids = (array[..., 0].astype(np.int32) // 10) * 256
        object_ids += array[..., 1].astype(np.int32)
        if object_ids.max(initial=0) >= len(_RAW_TO_ADE150):
            raise ValueError("ADE20K raw mask contains an unsupported object index")
        return _RAW_TO_ADE150[object_ids]
    if array.ndim == 2:
        # ADEChallengeData2016 stores background as 0 and categories as 1..150.
        values = array.astype(np.int32, copy=False)
        invalid = (values < 0) | (values > 150)
        if invalid.any():
            bad = np.unique(values[invalid])
            raise ValueError(f"ADEChallenge mask contains invalid labels: {bad[:10].tolist()}")
        result = np.full(values.shape, 255, dtype=np.uint8)
        foreground = values > 0
        result[foreground] = values[foreground] - 1
        return result
    raise ValueError(f"ADE20K mask must be HxW or HxWx3, got {array.shape}")


class Ade20kSegmentation:
    """Load raw ADE20K 2021 images and map their masks to ADE150."""

    def __init__(
        self,
        root: str | Path | None = None,
        split: str = "validation",
        pipeline: list[Callable] | None = None,
        csv_path: str | Path | None = None,
        image_root: str | Path | None = None,
        label_root: str | Path | None = None,
    ) -> None:
        self.pipeline = pipeline or []
        self.path = Path(csv_path) if csv_path is not None else None
        if self.path is not None:
            image_base = Path(image_root) if image_root is not None else None
            label_base = Path(label_root) if label_root is not None else None
            with self.path.open(encoding="utf-8-sig") as file:
                reader = csv.DictReader(file)
                required = {"image_path", "label_path"}
                if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                    raise ValueError(
                        "ADE20K CSV must contain image_path and label_path columns, "
                        f"got {reader.fieldnames}"
                    )
                self.samples = [
                    self._make_sample(
                        self._resolve_path(row["image_path"], image_base),
                        self._resolve_path(row["label_path"], label_base),
                    )
                    for row in reader
                ]
            return

        if root is None:
            raise ValueError("Either root or csv_path must be provided")
        release = resolve_ade20k_release(root)
        prefix = "ADE_train_" if split == "training" else "ADE_val_"
        image_paths = list_ade20k_images(release, prefix)
        self.samples = [
            self._make_sample(image, image.with_name(f"{image.stem}_seg.png"))
            for image in image_paths
        ]

    def _resolve_path(self, value: str | Path, root: Path | None) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = (root if root is not None else self.path.parent) / path
        return path

    @staticmethod
    def _make_sample(image_path: Path, label_path: Path) -> dict[str, Any]:
        if not image_path.is_file():
            raise FileNotFoundError(f"ADE20K image not found: {image_path}")
        if not label_path.is_file():
            raise FileNotFoundError(f"ADE20K segmentation mask not found: {label_path}")
        return {
            "id": image_path.stem,
            "image_path": str(image_path),
            "label_path": str(label_path),
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[np.ndarray, dict[str, Any]]:
        metadata = self.samples[index]
        image: Any = metadata["image_path"]
        for transform in self.pipeline:
            image = transform(image)
        with Image.open(metadata["label_path"]) as label_image:
            label = decode_ade20k_mask(np.asarray(label_image).copy())
        sample = dict(metadata)
        sample["label"] = label
        sample["original_size"] = label.shape
        return image, sample


ADE20KSegmentation = Ade20kSegmentation
