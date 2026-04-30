from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


DEFAULT_ROOT = Path("/home2020/home/icube/camma_files/camma_data/bsafe_dataset/bsafe")
SUPPORTED_DATASETS = ("Bsafe",)
SPLIT_TO_ANNOTATION_FILE = {
    "train": "train_annotations.json",
    "val": "val_annotations.json",
    "test": "test_annotations.json",
}


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def load_image(path: str | Path) -> np.ndarray:
    if plt is not None:
        return plt.imread(path)
    with Image.open(path) as image:
        return np.array(image)


def normalize_split(split: str) -> str:
    split_normalized = split.strip().lower()
    if split_normalized in {"val", "valid", "validation"}:
        return "val"
    if split_normalized in {"train", "test"}:
        return split_normalized
    raise ValueError(f"Unsupported split: {split!r}. Expected one of: train, val, test.")


def default_annotation_file_for_split(split: str) -> str:
    return SPLIT_TO_ANNOTATION_FILE[normalize_split(split)]


def normalize_dataset_name(dataset_name: str) -> str:
    if dataset_name in SUPPORTED_DATASETS:
        return dataset_name
    if dataset_name.strip().lower() == "bsafe":
        return "Bsafe"
    raise ValueError(
        f"Unsupported dataset_name={dataset_name!r}. Expected one of: {SUPPORTED_DATASETS}."
    )


def parse_frame_id(file_name: str) -> int | None:
    stem = Path(file_name).stem
    _, _, frame_token = stem.partition("_")
    if not frame_token:
        return None
    try:
        return int(frame_token)
    except ValueError:
        return None


def normalize_image_metadata(image: dict[str, Any]) -> BsafeImageMetadata:
    metadata = dict(image)
    metadata.setdefault("frame_id", parse_frame_id(str(metadata.get("file_name", ""))))
    return metadata


def normalize_bbox_annotation(annotation: dict[str, Any]) -> BsafeAnnotation:
    normalized = dict(annotation)
    area = normalized.get("area")
    if isinstance(area, (list, tuple)) and len(area) == 2:
        width = float(area[0])
        height = float(area[1])
        normalized["area_wh"] = [width, height]
        normalized["area"] = width * height
    elif area is not None:
        normalized["area"] = float(area)
    return normalized


class BsafeImageMetadata(TypedDict, total=False):
    file_name: str
    height: int
    width: int
    id: int
    video_id: int
    frame_id: int | None
    ROI: str


class BsafeAnnotation(TypedDict, total=False):
    category_id: int
    bbox: list[float]
    area: float
    area_wh: list[float]
    id: int
    image_id: int
    iscrowd: int
    point: list[float] | None
    type: str


class BsafeLineAnnotation(TypedDict, total=False):
    category_id: int
    line: list[list[float]]
    id: int
    image_id: int


@dataclass
class BsafeFrame:
    dataset_name: str
    split: str
    metadata: BsafeImageMetadata
    annotations: list[BsafeAnnotation]
    line_annotations: list[BsafeLineAnnotation]
    pixel_array: np.ndarray

    def __lt__(self, other: BsafeFrame) -> bool:
        if not isinstance(other, BsafeFrame):
            return NotImplemented
        return self.metadata["id"] < other.metadata["id"]

    def __gt__(self, other: BsafeFrame) -> bool:
        if not isinstance(other, BsafeFrame):
            return NotImplemented
        return self.metadata["id"] > other.metadata["id"]

    @property
    def annotations_lines(self) -> list[BsafeLineAnnotation]:
        return self.line_annotations


@dataclass
class BsafeContext:
    root_dir: Path
    split: str
    annotation_path: Path
    coco: dict[str, Any]
    annotations_by_image_id: dict[int, list[BsafeAnnotation]]
    line_annotations_by_image_id: dict[int, list[BsafeLineAnnotation]]
    image_id_to_index: dict[int, int]
    file_name_to_index: dict[str, int]
    category_id_to_name: dict[int, str]
    line_category_id_to_name: dict[int, str]
    video_id_to_image_ids: dict[int, list[int]]

    @classmethod
    def build(
        cls,
        root_dir: str | Path,
        split: str,
        annotation_file: str | Path | None = None,
    ) -> BsafeContext:
        root_dir_path = Path(root_dir)
        split_normalized = normalize_split(split)

        if annotation_file is None:
            annotation_path = root_dir_path / default_annotation_file_for_split(split_normalized)
        else:
            annotation_path = Path(annotation_file)
            if not annotation_path.is_absolute():
                annotation_path = root_dir_path / annotation_path

        if not annotation_path.exists():
            raise FileNotFoundError(f"Missing annotation json: {annotation_path}")

        raw_coco = load_json(annotation_path)

        normalized_images: list[BsafeImageMetadata] = []
        image_id_to_index: dict[int, int] = {}
        file_name_to_index: dict[str, int] = {}
        video_id_to_image_ids: dict[int, list[int]] = {}
        for idx, image in enumerate(raw_coco.get("images", [])):
            normalized_image = normalize_image_metadata(image)
            normalized_images.append(normalized_image)
            image_id = int(normalized_image["id"])
            image_id_to_index[image_id] = idx
            file_name_to_index[str(normalized_image["file_name"])] = idx
            video_id = normalized_image.get("video_id")
            if video_id is not None:
                video_id_to_image_ids.setdefault(int(video_id), []).append(image_id)

        normalized_annotations: list[BsafeAnnotation] = []
        annotations_by_image_id: dict[int, list[BsafeAnnotation]] = {}
        for annotation in raw_coco.get("annotations", []):
            normalized_annotation = normalize_bbox_annotation(annotation)
            normalized_annotations.append(normalized_annotation)
            image_id = int(normalized_annotation["image_id"])
            annotations_by_image_id.setdefault(image_id, []).append(normalized_annotation)

        normalized_line_annotations: list[BsafeLineAnnotation] = []
        line_annotations_by_image_id: dict[int, list[BsafeLineAnnotation]] = {}
        for annotation in raw_coco.get("annotations_lines", []):
            normalized_annotation = dict(annotation)
            normalized_line_annotations.append(normalized_annotation)
            image_id = int(normalized_annotation["image_id"])
            line_annotations_by_image_id.setdefault(image_id, []).append(normalized_annotation)

        category_id_to_name: dict[int, str] = {}
        for category in raw_coco.get("categories", []):
            category_id_to_name[int(category["id"])] = str(category["name"])

        line_category_id_to_name: dict[int, str] = {}
        for category in raw_coco.get("categories_lines", []):
            line_category_id_to_name[int(category["id"])] = str(category["name"])

        coco = dict(raw_coco)
        coco["images"] = normalized_images
        coco["annotations"] = normalized_annotations
        coco["annotations_lines"] = normalized_line_annotations

        return cls(
            root_dir=root_dir_path,
            split=split_normalized,
            annotation_path=annotation_path,
            coco=coco,
            annotations_by_image_id=annotations_by_image_id,
            line_annotations_by_image_id=line_annotations_by_image_id,
            image_id_to_index=image_id_to_index,
            file_name_to_index=file_name_to_index,
            category_id_to_name=category_id_to_name,
            line_category_id_to_name=line_category_id_to_name,
            video_id_to_image_ids=video_id_to_image_ids,
        )

    def resolve_image_path(self, file_name: str, video_id: int | None = None) -> Path:
        file_path = Path(file_name)
        candidates = [self.root_dir / file_path]

        if file_path.parent == Path(".") and video_id is not None:
            candidates.append(self.root_dir / str(video_id) / file_path.name)

        candidates.append(self.root_dir / file_path.name)

        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            f"Could not find image '{file_name}' under dataset root {self.root_dir}"
        )

    def get_frame(self, idx: int, dataset_name: str, split: str) -> BsafeFrame:
        frame_metadata = self.coco["images"][idx]
        image_id = int(frame_metadata["id"])
        pixel_array = load_image(
            self.resolve_image_path(
                frame_metadata["file_name"],
                video_id=frame_metadata.get("video_id"),
            )
        )

        return BsafeFrame(
            dataset_name=dataset_name,
            split=split,
            metadata=frame_metadata,
            annotations=self.annotations_by_image_id.get(image_id, []),
            line_annotations=self.line_annotations_by_image_id.get(image_id, []),
            pixel_array=pixel_array,
        )


class BsafeDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path = DEFAULT_ROOT,
        dataset_name: str = "Bsafe",
        split: str = "train",
        annotation_file: str | Path | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.dataset_name = normalize_dataset_name(dataset_name)
        self.split = normalize_split(split)
        self.annotation_file = (
            Path(annotation_file)
            if annotation_file is not None
            else Path(default_annotation_file_for_split(self.split))
        )
        self.context = BsafeContext.build(
            root_dir=self.root_dir,
            split=self.split,
            annotation_file=self.annotation_file,
        )

    def __len__(self) -> int:
        return len(self.context.coco["images"])

    def __getitem__(self, idx: int) -> BsafeFrame:
        return self.context.get_frame(idx, dataset_name=self.dataset_name, split=self.split)

    def get_frame_by_file_name(self, file_name: str) -> BsafeFrame:
        idx = self.context.file_name_to_index.get(file_name)
        if idx is None:
            raise ValueError(f"Frame with file name {file_name} not found")
        return self.context.get_frame(idx, dataset_name=self.dataset_name, split=self.split)

    def get_frame_by_id(self, frame_id: int) -> BsafeFrame:
        idx = self.context.image_id_to_index.get(frame_id)
        if idx is None:
            raise ValueError(f"Frame with id {frame_id} not found")
        return self.context.get_frame(idx, dataset_name=self.dataset_name, split=self.split)

    def get_category_name(self, category_id: int) -> str | None:
        return self.context.category_id_to_name.get(category_id)

    def get_line_category_name(self, category_id: int) -> str | None:
        return self.context.line_category_id_to_name.get(category_id)

    def get_frames_by_video_id(self, video_id: int) -> list[BsafeFrame]:
        return [
            self.get_frame_by_id(image_id)
            for image_id in self.context.video_id_to_image_ids.get(video_id, [])
        ]
