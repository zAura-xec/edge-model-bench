"""Dependency-light nuScenes multi-camera dataset and BEV preprocessing."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np


NUSCENES_CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
)
NUSCENES_DETECTION_CLASSES = (
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
)


def quaternion_to_rotation(quaternion: list[float]) -> np.ndarray:
    """Convert a nuScenes ``[w,x,y,z]`` quaternion to a 3x3 matrix."""

    w, x, y, z = map(float, quaternion)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def pose_matrix(
    translation: list[float],
    rotation: list[float],
    *,
    flat: bool = False,
) -> np.ndarray:
    result = np.eye(4, dtype=np.float32)
    rotation_matrix = quaternion_to_rotation(rotation)
    if flat:
        yaw = float(np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))
        cosine, sine = np.cos(yaw), np.sin(yaw)
        rotation_matrix = np.asarray(
            [[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]],
            dtype=np.float32,
        )
    result[:3, :3] = rotation_matrix
    result[:3, 3] = np.asarray(translation, dtype=np.float32)
    return result


def _category_name(name: str) -> str | None:
    for target in NUSCENES_DETECTION_CLASSES:
        if name == f"vehicle.{target}" or name == f"human.{target}" or name.endswith(f".{target}"):
            return target
    return None


class NuScenesMultiView:
    """Read nuScenes keyframes and produce model input dictionaries."""

    def __init__(
        self,
        root: str | Path,
        manifest_path: str | Path | None = None,
        version: str = "v1.0-mini",
        pipeline: list[Callable] | None = None,
    ) -> None:
        self.root = Path(root)
        self.version = version
        self.version_root = self.root / version
        self.pipeline = pipeline or []
        self._load_tables()
        tokens = self._read_manifest(manifest_path) if manifest_path else list(self.samples)
        self.tokens = [token for token in tokens if token in self.samples]
        if not self.tokens:
            raise ValueError("No valid nuScenes sample tokens found")

    def _load_tables(self) -> None:
        def load(name: str) -> list[dict[str, Any]]:
            with (self.version_root / f"{name}.json").open(encoding="utf-8") as file:
                return json.load(file)

        self.samples = {row["token"]: row for row in load("sample")}
        self.sample_data = {row["token"]: row for row in load("sample_data")}
        sensors = {row["token"]: row for row in load("sensor")}
        self.calibrated = {row["token"]: row for row in load("calibrated_sensor")}
        self.poses = {row["token"]: row for row in load("ego_pose")}
        categories = {row["token"]: row["name"] for row in load("category")}
        instances = {row["token"]: row for row in load("instance")}
        self.annotations_by_sample: dict[str, list[dict[str, Any]]] = {}
        for annotation in load("sample_annotation"):
            instance = instances.get(annotation.get("instance_token"), {})
            name = _category_name(annotation.get("category_name", ""))
            if name is None:
                name = _category_name(categories.get(instance.get("category_token", ""), ""))
            if name is not None:
                annotation = dict(annotation)
                annotation["category_name"] = name
                self.annotations_by_sample.setdefault(annotation["sample_token"], []).append(annotation)
        self.channel_by_sensor = {token: row["channel"] for token, row in sensors.items()}

    @staticmethod
    def _read_manifest(path: str | Path) -> list[str]:
        path = Path(path)
        if path.suffix.lower() == ".csv":
            with path.open(encoding="utf-8-sig", newline="") as file:
                return [row.get("sample_token", "") for row in csv.DictReader(file)]
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]

    def _record(self, token: str) -> dict[str, Any]:
        sample_data = [row for row in self.sample_data.values() if row["sample_token"] == token and row["is_key_frame"]]
        by_channel: dict[str, dict[str, Any]] = {}
        all_by_channel: dict[str, dict[str, Any]] = {}
        for row in sample_data:
            calib = self.calibrated[row["calibrated_sensor_token"]]
            channel = self.channel_by_sensor[calib["sensor_token"]]
            all_by_channel[channel] = row
            if channel in NUSCENES_CAMERAS and row["fileformat"] == "jpg":
                by_channel[channel] = row
        if set(by_channel) != set(NUSCENES_CAMERAS):
            raise ValueError(f"Sample {token} does not contain all six camera keyframes")

        key_pose = self.poses[by_channel["CAM_FRONT"]["ego_pose_token"]]
        key_ego_inv = np.linalg.inv(pose_matrix(key_pose["translation"], key_pose["rotation"]))
        cvt_key_data = all_by_channel.get("LIDAR_TOP", by_channel["CAM_FRONT"])
        cvt_key_pose = self.poses[cvt_key_data["ego_pose_token"]]
        global_to_cvt_key = np.linalg.inv(
            pose_matrix(cvt_key_pose["translation"], cvt_key_pose["rotation"], flat=True)
        ).astype(np.float32)
        camera_paths: list[str] = []
        intrinsics: list[np.ndarray] = []
        extrinsics: list[np.ndarray] = []
        sensor2keyegos: list[np.ndarray] = []
        cvt_extrinsics: list[np.ndarray] = []
        for channel in NUSCENES_CAMERAS:
            data = by_channel[channel]
            calib = self.calibrated[data["calibrated_sensor_token"]]
            pose = self.poses[data["ego_pose_token"]]
            sensor_to_ego = pose_matrix(calib["translation"], calib["rotation"])
            ego_to_key = key_ego_inv @ pose_matrix(pose["translation"], pose["rotation"])
            camera_paths.append(str(self.root / data["filename"]))
            intrinsics.append(np.asarray(calib["camera_intrinsic"], dtype=np.float32))
            extrinsics.append(sensor_to_ego)
            sensor2keyegos.append(ego_to_key @ sensor_to_ego)
            cvt_extrinsics.append(
                global_to_cvt_key
                @ pose_matrix(pose["translation"], pose["rotation"])
                @ sensor_to_ego
            )
        return {
            "sample_token": token,
            "camera_paths": camera_paths,
            "intrinsics": np.stack(intrinsics),
            "extrinsics": np.stack(extrinsics),
            "sensor2keyegos": np.stack(sensor2keyegos),
            "cvt_extrinsics": np.stack(cvt_extrinsics).astype(np.float32),
            "global_to_cvt_key": global_to_cvt_key,
            "annotations": self.annotations_by_sample.get(token, []),
        }

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, index: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        record = self._record(self.tokens[index])
        value: Any = record
        for transform in self.pipeline:
            value = transform(value)
        if not isinstance(value, Mapping) or "inputs" not in value:
            raise ValueError("nuScenes pipeline must produce a mapping containing 'inputs'")
        return value["inputs"], record

    def __call__(self) -> list[dict[str, Any]]:
        return [self._record(token) for token in self.tokens]


class PrepareNuScenesInputs:
    """Load six camera images and build BEVDet or CVT input tensors."""

    def __init__(
        self,
        model: str = "bevdet",
        image_size: tuple[int, int] | list[int] = (256, 704),
        top_crop: int = 46,
        mean: tuple[float, float, float] | list[float] = (123.675, 116.28, 103.53),
        std: tuple[float, float, float] | list[float] = (58.395, 57.12, 57.375),
    ) -> None:
        self.model = model.lower()
        self.image_size = tuple(map(int, image_size))
        self.top_crop = int(top_crop)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    def __call__(self, record: dict[str, Any]) -> dict[str, Any]:
        images = []
        cvt_intrinsics = []
        for path, intrinsic in zip(record["camera_paths"], record["intrinsics"]):
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            original_height, original_width = image.shape[:2]
            if self.model == "cvt":
                resize_height = self.image_size[0] + self.top_crop
                image = cv2.resize(
                    image,
                    (self.image_size[1], resize_height),
                    interpolation=cv2.INTER_LINEAR,
                )
                image = image[self.top_crop : self.top_crop + self.image_size[0]]
                images.append(image.astype(np.float32) / 255.0)
                adjusted = np.asarray(intrinsic, dtype=np.float32).copy()
                adjusted[0, 0] *= self.image_size[1] / original_width
                adjusted[0, 2] *= self.image_size[1] / original_width
                adjusted[1, 1] *= resize_height / original_height
                adjusted[1, 2] *= resize_height / original_height
                adjusted[1, 2] -= self.top_crop
                # The exported ONNX graph consumes the inverses computed in the
                # original PyTorch encoder before its camera-ray MatMuls.
                cvt_intrinsics.append(np.linalg.inv(adjusted).astype(np.float32))
            else:
                image = cv2.resize(
                    image,
                    (self.image_size[1], self.image_size[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                images.append((image.astype(np.float32) - self.mean) / self.std)
        stacked = np.stack(images).astype(np.float32)
        if self.model == "bevdet":
            image = np.ascontiguousarray(stacked.transpose(0, 3, 1, 2).reshape(1, 18, *self.image_size))
            inputs = {
                "image": image,
                "sensor2keyegos": record["sensor2keyegos"][None],
                "inv_intrins": np.linalg.inv(record["intrinsics"])[None].astype(np.float32),
                "inv_post_rots": np.tile(np.eye(3, dtype=np.float32), (1, 6, 1, 1)),
                "post_trans": np.zeros((1, 6, 1, 3), dtype=np.float32),
            }
        elif self.model == "cvt":
            inputs = {
                "image": np.ascontiguousarray(stacked.transpose(0, 3, 1, 2)[None]),
                "intrinsics": np.stack(cvt_intrinsics)[None],
                "extrinsics": record["cvt_extrinsics"][None],
            }
        else:
            raise ValueError(f"Unsupported nuScenes model: {self.model}")
        record["inputs"] = inputs
        return record


class SaveMultiInputRaw:
    """Dump named multi-input tensors and QAIRT native input-list entries."""

    def __init__(self, output_dir: str | Path, input_list_path: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.input_list_path = Path(input_list_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.input_list_path.parent.mkdir(parents=True, exist_ok=True)
        self.index = 0

    def __call__(self, record: dict[str, Any]) -> dict[str, Any]:
        if self.index == 0:
            self.input_list_path.write_text("")
        paths = []
        for name, value in record["inputs"].items():
            directory = self.output_dir / name
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{self.index:06d}.raw"
            np.asarray(value, dtype=np.float32).tofile(path)
            paths.append(f"{name}:={path.resolve()}")
        with self.input_list_path.open("a") as file:
            file.write(" ".join(paths) + "\n")
        self.index += 1
        return record


NuScenes = NuScenesMultiView
