from __future__ import annotations

from typing import Any, TypedDict, Required
from pathlib import Path
import json
from dataclasses import dataclass
import re
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


DEFAULT_ROOT = Path('/home2020/home/miv/vedrenne/data/camma')
SUPPORTED_DATASETS = (
    "Endoscapes-Seg201-CBD",
    "ICG-LC-EAES",
)
SUPPORTED_BBOX_ANCHORS = ('topleft', 'center')
BBOX_ANCHOR_BY_DATASET = {
    'Endoscapes-Seg201-CBD': 'topleft',
    'ICG-LC-EAES': 'center',
}


def load_json(path: str | Path) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def load_image(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image)


def parse_trailing_numeric_token(value: str) -> int | None:
    digits = "".join(character for character in value if character.isdigit())
    if not digits:
        return None
    return int(digits)


def infer_camma_file_family(stem: str) -> str:
    if re.match(r"^V\d+(?:_\d+)?_photo_\d+$", stem, flags=re.IGNORECASE):
        return "photo"
    if re.match(
        r"^\d{4}-\d{2}-\d{2}_\d{6}_VID\d+-mp4_\d{8}_\d{6}-\d+$",
        stem,
        flags=re.IGNORECASE,
    ):
        return "vid_mp4"
    if re.match(r"^\d{4}[-_]\d{2}[-_]\d{2}_Video\d+_\d+$", stem, flags=re.IGNORECASE):
        return "video_dated"
    if re.match(r"^\d{4}-\d{2}-\d{2}_DL_CMRP_\d+(?:_[A-Za-z0-9-]+)*_\d+$", stem, flags=re.IGNORECASE):
        return "dl_cmrp"
    if re.match(r"^\d{4}[-_]\d{2}[-_]\d{2}_CMRP\d+(?:_[A-Za-z0-9-]+)*_\d+$", stem, flags=re.IGNORECASE):
        return "cmrp_dated"
    return "generic"


def parse_camma_file_name(file_name: str) -> tuple[str, str, int | None, int | None, int | None]:
    stem = Path(file_name).stem
    family = infer_camma_file_family(stem)
    video_key, separator, trailing_token = stem.rpartition("_")
    if not separator:
        video_key = stem
        trailing_token = stem

    video_id = None
    for pattern in (
        r"CMRP[_-]?(\d+)",
        r"Video0*(\d+)",
        r"VID0*(\d+)",
        r"(?:^|_)V(\d+)(?:_|$)",
    ):
        match = re.search(pattern, stem, flags=re.IGNORECASE)
        if match is not None:
            video_id = int(match.group(1))
            break

    frame_token = parse_trailing_numeric_token(trailing_token)
    frame_id = frame_token if family in {"photo", "generic"} else None
    return video_key, family, video_id, frame_id, frame_token


def normalize_image_metadata(image: dict[str, Any]) -> CammaImageMetadata:
    metadata = dict(image)
    video_key, file_family, video_id, frame_id, frame_token = parse_camma_file_name(
        str(metadata.get("file_name", ""))
    )
    metadata.setdefault("video_key", video_key)
    metadata.setdefault("file_family", file_family)
    metadata.setdefault("video_id", video_id)
    metadata.setdefault("frame_id", frame_id)
    metadata.setdefault("frame_token", frame_token)
    return metadata


class CammaImageMetadata(TypedDict, total=False):
    file_name: str
    height: int
    width: int
    id: Required[int]
    is_det_keyframe: bool
    ds: list[float]
    video_id: int
    video_key: str
    file_family: str
    frame_id: int | None
    frame_token: int | None
    is_ds_keyframe: bool


class CammaAnnotation(TypedDict, total=False):
    category_id: Required[int]
    bbox: list[float]  # [x, y, width, height], anchor depends on dataset
    area: float
    id: int
    image_id: int
    iscrowd: int
    segmentation: Any
    obb: list[float]


@dataclass
class CammaFrame:
    dataset_name: str
    split: str
    metadata: CammaImageMetadata
    annotations: list[CammaAnnotation]
    pixel_array: np.ndarray | None = None

    def __lt__(self, other: CammaFrame) -> bool:
        if not isinstance(other, CammaFrame):
            return NotImplemented
        return self.metadata['id'] < other.metadata['id']

    def __gt__(self, other: CammaFrame) -> bool:
        if not isinstance(other, CammaFrame):
            return NotImplemented
        return self.metadata['id'] > other.metadata['id']


@dataclass
class CammaContext:
    root_dir: Path
    dataset_dir: Path
    split_dir: Path
    images_dir: Path
    coco: dict
    annotations_by_image_id: dict[int, list[CammaAnnotation]]
    image_id_to_index: dict[int, int]
    file_name_to_index: dict[str, int]
    category_id_to_name: dict[int, str]
    video_id_to_image_ids: dict[int, list[int]]
    video_key_to_image_ids: dict[str, list[int]]

    @classmethod
    def build(
        cls,
        root_dir: str | Path,
        dataset_name: str,
        split: str,
        annotation_file: str = "annotation_coco.json",
    ) -> CammaContext:
        root_dir_path = Path(root_dir)
        dataset_dir = root_dir_path / dataset_name
        split_dir = dataset_dir / split
        images_dir = split_dir / "images"
        annotation_path = split_dir / annotation_file

        if not annotation_path.exists():
            raise FileNotFoundError(f"Missing COCO json: {annotation_path}")

        coco = load_json(annotation_path)

        normalized_images: list[CammaImageMetadata] = []
        annotations_by_image_id: dict[int, list[CammaAnnotation]] = {}
        for ann in coco.get("annotations", []):
            image_id = int(ann["image_id"])
            annotations_by_image_id.setdefault(image_id, []).append(ann)

        image_id_to_index: dict[int, int] = {}
        file_name_to_index: dict[str, int] = {}
        video_id_to_image_ids: dict[int, list[int]] = {}
        video_key_to_image_ids: dict[str, list[int]] = {}
        for idx, image in enumerate(coco.get("images", [])):
            normalized_image = normalize_image_metadata(image)
            normalized_images.append(normalized_image)
            image_id = int(normalized_image["id"])
            image_id_to_index[image_id] = idx
            file_name_to_index[str(normalized_image["file_name"])] = idx

            video_key = str(normalized_image.get("video_key", "")).strip()
            if video_key:
                video_key_to_image_ids.setdefault(video_key, []).append(image_id)

            video_id = normalized_image.get("video_id")
            if video_id is not None:
                video_id_to_image_ids.setdefault(int(video_id), []).append(image_id)

        def sort_image_ids(image_ids: list[int]) -> None:
            image_ids.sort(
                key=lambda image_id: (
                    normalized_images[image_id_to_index[image_id]].get("frame_id") is None,
                    int(normalized_images[image_id_to_index[image_id]].get("frame_id") or -1),
                    image_id,
                )
            )

        for image_ids in video_key_to_image_ids.values():
            sort_image_ids(image_ids)
        for image_ids in video_id_to_image_ids.values():
            sort_image_ids(image_ids)

        category_id_to_name: dict[int, str] = {}
        for category in coco.get("categories", []):
            category_id_to_name[int(category["id"])] = str(category["name"])

        normalized_coco = dict(coco)
        normalized_coco["images"] = normalized_images

        return cls(
            root_dir_path,
            dataset_dir,
            split_dir,
            images_dir,
            normalized_coco,
            annotations_by_image_id,
            image_id_to_index,
            file_name_to_index,
            category_id_to_name,
            video_id_to_image_ids,
            video_key_to_image_ids,
        )

    def resolve_image_path(self, file_name: str) -> Path:
        image_path = self.images_dir / file_name
        if image_path.exists():
            return image_path

        fallback = self.split_dir / file_name
        if fallback.exists():
            return fallback

        raise FileNotFoundError(
            f"Could not find image '{file_name}' in {self.images_dir} or {self.split_dir}"
        )

    def get_frame(self, idx: int, dataset_name: str, split: str, load_pixel_array: bool = True) -> CammaFrame:
        frame_metadata = self.coco["images"][idx]
        image_annotations = self.annotations_by_image_id.get(int(frame_metadata["id"]), [])
        pixel_array = load_image(self.resolve_image_path(frame_metadata["file_name"])) if load_pixel_array else None

        return CammaFrame(
            dataset_name=dataset_name,
            split=split,
            metadata=frame_metadata,
            annotations=image_annotations,
            pixel_array=pixel_array,
        )


def normalize_split(split: str) -> str:
    split_normalized = split.strip().lower()
    if split_normalized in {"val", "valid", "validation"}:
        return "val"
    if split_normalized in {"train", "test"}:
        return split_normalized
    raise ValueError(f"Unsupported split: {split!r}. Expected one of: train, val, test.")


def normalize_bbox_anchor(anchor: str) -> str:
    normalized = anchor.strip().lower()
    if normalized not in SUPPORTED_BBOX_ANCHORS:
        raise ValueError(
            f'Unsupported bbox_anchor={anchor!r}. '
            f'Expected one of: {SUPPORTED_BBOX_ANCHORS}.'
        )
    return normalized


def default_bbox_anchor_for_dataset(dataset_name: str) -> str:
    if dataset_name not in BBOX_ANCHOR_BY_DATASET:
        raise ValueError(
            f'No default bbox anchor configured for dataset_name={dataset_name!r}. '
            f'Expected one of: {tuple(BBOX_ANCHOR_BY_DATASET)}.'
        )
    return BBOX_ANCHOR_BY_DATASET[dataset_name]

class CammaDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path = DEFAULT_ROOT,
        dataset_name: str = "Endoscapes-Seg201-CBD",
        split: str = "train",
        annotation_file: str = "annotation_coco.json",
    ) -> None:
        if dataset_name not in SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset_name={dataset_name!r}. "
                f"Expected one of: {SUPPORTED_DATASETS}."
            )

        self.root_dir = Path(root_dir)
        self.dataset_name = dataset_name
        self.split = normalize_split(split)
        self.annotation_file = annotation_file
        self.context = CammaContext.build(
            root_dir=self.root_dir,
            dataset_name=self.dataset_name,
            split=self.split,
            annotation_file=self.annotation_file,
        )

    def __len__(self) -> int:
        return len(self.context.coco["images"])

    def __getitem__(self, idx: int) -> CammaFrame:
        return self.context.get_frame(idx, dataset_name=self.dataset_name, split=self.split)

    def get_frame_by_file_name(self, file_name: str) -> CammaFrame:
        idx = self.context.file_name_to_index.get(file_name)
        if idx is None:
            raise ValueError(f"Frame with file name {file_name} not found")
        return self.context.get_frame(idx, dataset_name=self.dataset_name, split=self.split)

    def get_frame_by_id(self, frame_id: int) -> CammaFrame:
        idx = self.context.image_id_to_index.get(frame_id)
        if idx is None:
            raise ValueError(f"Frame with id {frame_id} not found")
        return self.context.get_frame(idx, dataset_name=self.dataset_name, split=self.split)

    def get_category_name(self, category_id: int) -> str | None:
        return self.context.category_id_to_name.get(category_id)
    
    def get_frames_by_video_id(self, video_id: int) -> list[CammaFrame]:
        return [self.get_frame_by_id(idx)
                for idx in self.context.video_id_to_image_ids.get(video_id, [])]


class EndoscapesSeg201CBDDataset(CammaDataset):
    def __init__(
        self,
        root_dir: str | Path = DEFAULT_ROOT,
        split: str = "train",
        annotation_file: str = "annotation_coco.json",
    ) -> None:
        super().__init__(
            root_dir=root_dir,
            dataset_name="Endoscapes-Seg201-CBD",
            split=split,
            annotation_file=annotation_file,
        )


class ICGLCEAESDataset(CammaDataset):
    def __init__(
        self,
        root_dir: str | Path = DEFAULT_ROOT,
        split: str = "train",
        annotation_file: str = "annotation_coco.json",
    ) -> None:
        super().__init__(
            root_dir=root_dir,
            dataset_name="ICG-LC-EAES",
            split=split,
            annotation_file=annotation_file,
        )
