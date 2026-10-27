"""Image transforms."""

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np


class LoadingImage:
    def __init__(self, imread_flag: int = cv2.IMREAD_COLOR) -> None:
        self.imread_flag = imread_flag

    def __call__(self, image_path: str | Path) -> np.ndarray:
        image = cv2.imread(str(image_path), self.imread_flag)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        return image


class Resize:
    def __init__(self, size: Sequence[int], interpolation: int = cv2.INTER_LINEAR) -> None:
        self.size = tuple(size)
        self.interpolation = interpolation

    def __call__(self, image: np.ndarray) -> np.ndarray:
        height, width = self.size
        return cv2.resize(image, (width, height), interpolation=self.interpolation)


class BGR2RGB:
    def __call__(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class Normalize:
    def __init__(self, mean: Sequence[float] | float, std: Sequence[float] | float) -> None:
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return (image.astype(np.float32) - self.mean) / self.std


class HWC2NCHW:
    def __call__(self, image: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(image.transpose(2, 0, 1)[None])


class SaveRaw:
    def __init__(
        self,
        output_dir: str | Path,
        input_list_path: str | Path | None = None,
        dtype: str = "float32",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.input_list_path = Path(
            input_list_path or self.output_dir.parent / "input_list.txt"
        )
        self.input_list_path.parent.mkdir(parents=True, exist_ok=True)
        self.dtype = dtype
        self.index = 0

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if self.index == 0:
            self.input_list_path.write_text("")
        raw_path = self.output_dir / f"{self.index:06d}.raw"
        np.asarray(image, dtype=self.dtype).tofile(raw_path)
        with self.input_list_path.open("a") as file:
            file.write(f"{raw_path.resolve()}\n")
        self.index += 1
        return image


class SaveNpy:
    """Save calibration tensors as ``.npy`` files for Horizon OE ``hb_compile``.

    OE calibration loaders expect per-sample arrays **without** the batch
    dimension (e.g. CHW for NCHW models). A leading batch size of 1 is stripped.
    """

    def __init__(
        self,
        output_dir: str | Path,
        dtype: str = "float32",
        drop_batch: bool = True,
        input_list_path: str | Path | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dtype = dtype
        self.drop_batch = drop_batch
        self.input_list_path = (
            Path(input_list_path) if input_list_path is not None else None
        )
        if self.input_list_path is not None:
            self.input_list_path.parent.mkdir(parents=True, exist_ok=True)
        self.index = 0

    def __call__(self, image: np.ndarray) -> np.ndarray:
        array = np.asarray(image, dtype=self.dtype)
        if self.drop_batch and array.ndim >= 3 and array.shape[0] == 1:
            array = array[0]
        npy_path = self.output_dir / f"{self.index:06d}.npy"
        np.save(npy_path, array)
        if self.input_list_path is not None:
            mode = "w" if self.index == 0 else "a"
            with self.input_list_path.open(mode, encoding="utf-8") as input_list:
                input_list.write(f"{npy_path.resolve()}\n")
        self.index += 1
        return image
