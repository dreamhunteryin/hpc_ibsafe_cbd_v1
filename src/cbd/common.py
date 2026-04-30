from __future__ import annotations

from dataclasses import dataclass
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage

from sam3.image_utils import resize_image_to_square
from sam3.model.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh

from .sources import CBDSourceConfig, CBDSourceDataset, metadata_frame_index
from .video import load_frames_from_video, probe_video_fps, resolve_video_frame_location


RGB_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
RGB_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
DEFAULT_CLIP_LEN = 25
DEFAULT_IMAGE_SIZE = 384
TARGET_TYPE_ORDER = ("soft", "hard")
TARGET_TYPE_TO_LABEL = {name: index for index, name in enumerate(TARGET_TYPE_ORDER)}
UNLABELED_TARGET_TYPE = -1


@dataclass(frozen=True)
class ClipFrames:
    images: list[PILImage.Image]
    frame_names: list[str]
    original_size: tuple[int, int]


@dataclass(frozen=True)
class AugmentationParams:
    flip: bool
    scale: float
    translate_x: float
    translate_y: float
    angle_deg: float
    shear_x_deg: float
    shear_y_deg: float


def resolve_input_size(model_config: dict | None = None) -> int:
    model_config = model_config or {}
    return int(model_config.get("input_size", model_config.get("image_size", DEFAULT_IMAGE_SIZE)))


def normalize_target_types(target_type: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if target_type is None:
        return TARGET_TYPE_ORDER
    if isinstance(target_type, str):
        normalized = target_type.strip().lower()
        if normalized in {"all", "*", ""}:
            return TARGET_TYPE_ORDER
        if "," in normalized:
            return tuple(part.strip() for part in normalized.split(",") if part.strip())
        return (normalized,)
    return tuple(str(part).strip().lower() for part in target_type if str(part).strip())


def target_type_to_label(target_type: str) -> int:
    return TARGET_TYPE_TO_LABEL[str(target_type).strip().lower()]


def label_to_target_type(label: int) -> str:
    return TARGET_TYPE_ORDER[int(label)]


def frame_id_from_metadata(source_config: CBDSourceConfig, metadata: dict) -> int | None:
    return metadata_frame_index(source_config, metadata)


def parse_frame_number_from_path(path: str | Path) -> int:
    stem = Path(path).stem
    _, separator, trailing = stem.rpartition("_")
    if not separator:
        trailing = stem
    digits = "".join(character for character in trailing if character.isdigit())
    if digits:
        return int(digits)
    digits = "".join(character for character in stem if character.isdigit())
    if digits:
        return int(digits)
    raise ValueError(f"Could not parse a frame number from path={path!r}.")


def list_clip_frame_paths(clip_dir: str | Path) -> list[Path]:
    clip_dir = Path(clip_dir)
    frame_paths = [
        path
        for path in clip_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]
    frame_paths.sort(key=parse_frame_number_from_path)
    return frame_paths


def sample_clip_frame_indices(
    target_frame_id: int,
    fps: int,
    clip_len: int = DEFAULT_CLIP_LEN,
    target_fps: int = 5,
) -> list[int]:
    step = max(1, int(round(int(fps) / int(target_fps))))
    end_idx = int(target_frame_id)
    start_idx = end_idx - step * (int(clip_len) - 1)
    return list(range(start_idx, end_idx + 1, step))


def _finalize_clip_frames(images: list[PILImage.Image], frame_names: list[str], clip_len: int) -> ClipFrames:
    while images and len(images) < clip_len:
        images.insert(0, images[0].copy())
        frame_names.insert(0, frame_names[0])
    images = images[-clip_len:]
    frame_names = frame_names[-clip_len:]
    width, height = images[-1].size
    return ClipFrames(images=images, frame_names=frame_names, original_size=(height, width))


def load_clip_frames_from_video(
    dataset: CBDSourceDataset,
    source_config: CBDSourceConfig,
    metadata: dict,
    *,
    clip_len: int = DEFAULT_CLIP_LEN,
    target_fps: int = 5,
    target_box_cxcywh: torch.Tensor | np.ndarray | tuple[float, float, float, float] | None = None,
) -> ClipFrames:
    resolved = resolve_video_frame_location(
        dataset,
        source_config,
        metadata,
        target_box_cxcywh=target_box_cxcywh,
    )
    fps = max(1, int(round(probe_video_fps(resolved.video_path))))
    frame_indices = sample_clip_frame_indices(
        target_frame_id=resolved.frame_index,
        fps=fps,
        clip_len=clip_len,
        target_fps=target_fps,
    )
    valid_indices = [frame_idx for frame_idx in frame_indices if frame_idx >= 0]
    images = load_frames_from_video(resolved.video_path, valid_indices)
    frame_names = [f"frame_{frame_idx}.png" for frame_idx in valid_indices]
    return _finalize_clip_frames(images, frame_names, clip_len)

def load_effective_clip_frames(
    dataset: CBDSourceDataset,
    source_config: CBDSourceConfig,
    metadata: dict,
    clip_dir: str | Path,
    clip_len: int = DEFAULT_CLIP_LEN,
    target_fps: int = 5,
    prefer_existing_clip: bool = True,
    target_box_cxcywh: torch.Tensor | np.ndarray | tuple[float, float, float, float] | None = None,
) -> ClipFrames:
    clip_dir = Path(clip_dir)
    frame_paths = list_clip_frame_paths(clip_dir) if prefer_existing_clip and clip_dir.exists() else []
    if frame_paths:
        images = [PILImage.open(path).convert("RGB") for path in frame_paths]
        frame_names = [path.name for path in frame_paths]

        if source_config.kind == "bsafe":
            target_frame_id = frame_id_from_metadata(source_config, metadata)
            last_frame_id = parse_frame_number_from_path(frame_paths[-1]) if frame_paths else -1
            if target_frame_id is not None and last_frame_id != target_frame_id:
                target_path = dataset.context.resolve_image_path(
                    str(metadata["file_name"]),
                    video_id=int(metadata["video_id"]),
                )
                images.append(PILImage.open(target_path).convert("RGB"))
                frame_names.append(f"frame_{target_frame_id}.png")

        return _finalize_clip_frames(images, frame_names, clip_len)

    return load_clip_frames_from_video(
        dataset,
        source_config,
        metadata,
        clip_len=clip_len,
        target_fps=target_fps,
        target_box_cxcywh=target_box_cxcywh,
    )


def pil_to_normalized_tensor(image: PILImage.Image, image_size: int) -> torch.Tensor:
    resized = resize_image_to_square(image, image_size)
    image_np = np.asarray(resized, dtype=np.float32) / 255.0
    image_t = torch.from_numpy(image_np).permute(2, 0, 1)
    return (image_t - RGB_MEAN) / RGB_STD


def load_rgb_clip_tensor(frames: ClipFrames, image_size: int = DEFAULT_IMAGE_SIZE) -> torch.Tensor:
    return torch.stack([pil_to_normalized_tensor(image, image_size) for image in frames.images], dim=0)


def resize_mask_sequence(mask_array: np.ndarray, image_size: int = DEFAULT_IMAGE_SIZE) -> torch.Tensor:
    masks = torch.from_numpy(mask_array).float()
    masks = F.interpolate(masks, size=(image_size, image_size), mode="nearest")
    return masks


def load_mask_cache_tensor(
    mask_cache_path: str | Path,
    image_size: int = DEFAULT_IMAGE_SIZE,
    clip_len: int = DEFAULT_CLIP_LEN,
) -> tuple[torch.Tensor, dict[str, np.ndarray]]:
    payload = np.load(mask_cache_path, allow_pickle=True)
    sliced = {key: payload[key] for key in payload.files}
    payload.close()
    masks_np = sliced["masks"]
    if masks_np.shape[0] < clip_len:
        pad = np.repeat(masks_np[:1], clip_len - masks_np.shape[0], axis=0)
        masks_np = np.concatenate([pad, masks_np], axis=0)
    sliced["masks"] = masks_np[-clip_len:]
    if "frame_names" in sliced:
        frame_names = sliced["frame_names"]
        if frame_names.shape[0] < clip_len:
            pad = np.repeat(frame_names[:1], clip_len - frame_names.shape[0], axis=0)
            frame_names = np.concatenate([pad, frame_names], axis=0)
        sliced["frame_names"] = frame_names[-clip_len:]
    masks = resize_mask_sequence(sliced["masks"], image_size=image_size)
    return masks, sliced


def sample_augmentation_params(image_size: int, augmentation_level: int) -> AugmentationParams | None:
    if augmentation_level <= 0:
        return None
    if augmentation_level == 1:
        scale_min, scale_max = 0.90, 1.10
        translate = 0.06 * image_size
        angle = 0.0
        shear = 0.0
    elif augmentation_level == 2:
        scale_min, scale_max = 0.85, 1.15
        translate = 0.10 * image_size
        angle = 10.0
        shear = 4.0
    else:
        scale_min, scale_max = 0.75, 1.25
        translate = 0.15 * image_size
        angle = 20.0
        shear = 8.0
    return AugmentationParams(
        flip=random.random() < 0.5,
        scale=random.uniform(scale_min, scale_max),
        translate_x=random.uniform(-translate, translate),
        translate_y=random.uniform(-translate, translate),
        angle_deg=random.uniform(-angle, angle),
        shear_x_deg=random.uniform(-shear, shear),
        shear_y_deg=random.uniform(-shear, shear),
    )


def translation_matrix(tx: float, ty: float, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    )


def scale_matrix(scale: float, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    )


def rotation_shear_matrix(
    angle_deg: float,
    shear_x_deg: float,
    shear_y_deg: float,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    angle = math.radians(angle_deg)
    shear_x = math.tan(math.radians(shear_x_deg))
    shear_y = math.tan(math.radians(shear_y_deg))
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    rotation = torch.tensor(
        [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    )
    shear = torch.tensor(
        [[1.0, shear_x, 0.0], [shear_y, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    )
    return rotation @ shear


def flip_matrix(*, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    )


def build_forward_augmentation_matrix(
    params: AugmentationParams,
    image_size: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    center = 0.5 * float(image_size - 1)
    linear = rotation_shear_matrix(
        params.angle_deg,
        params.shear_x_deg,
        params.shear_y_deg,
        dtype=dtype,
        device=device,
    ) @ scale_matrix(params.scale, dtype=dtype, device=device)
    if params.flip:
        linear = linear @ flip_matrix(dtype=dtype, device=device)
    return (
        translation_matrix(params.translate_x, params.translate_y, dtype=dtype, device=device)
        @ translation_matrix(center, center, dtype=dtype, device=device)
        @ linear
        @ translation_matrix(-center, -center, dtype=dtype, device=device)
    )


def normalized_to_pixel_matrix(image_size: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    size = float(image_size)
    return torch.tensor(
        [[size / 2.0, 0.0, size / 2.0 - 0.5], [0.0, size / 2.0, size / 2.0 - 0.5], [0.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    )


def pixel_to_normalized_matrix(image_size: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    size = float(image_size)
    return torch.tensor(
        [[2.0 / size, 0.0, 1.0 / size - 1.0], [0.0, 2.0 / size, 1.0 / size - 1.0], [0.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    )


def forward_matrix_to_theta(
    forward_matrix: torch.Tensor,
    image_size: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    normalized_to_pixel = normalized_to_pixel_matrix(image_size, dtype=dtype, device=device)
    pixel_to_normalized = pixel_to_normalized_matrix(image_size, dtype=dtype, device=device)
    inverse_matrix = torch.linalg.inv(forward_matrix)
    theta = pixel_to_normalized @ inverse_matrix @ normalized_to_pixel
    return theta[:2]


def transform_normalized_box(
    target_box: torch.Tensor,
    forward_matrix: torch.Tensor,
    image_size: int,
) -> torch.Tensor:
    box_xyxy = box_cxcywh_to_xyxy(target_box.unsqueeze(0)).squeeze(0)
    x0, y0, x1, y1 = box_xyxy.tolist()
    scale = float(image_size - 1)
    corners = torch.tensor(
        [
            [x0 * scale, y0 * scale, 1.0],
            [x1 * scale, y0 * scale, 1.0],
            [x1 * scale, y1 * scale, 1.0],
            [x0 * scale, y1 * scale, 1.0],
        ],
        dtype=forward_matrix.dtype,
        device=forward_matrix.device,
    )
    transformed = (forward_matrix @ corners.t()).t()
    transformed_xy = transformed[:, :2]
    transformed_xy[:, 0] = transformed_xy[:, 0].clamp(0.0, scale)
    transformed_xy[:, 1] = transformed_xy[:, 1].clamp(0.0, scale)
    min_xy = transformed_xy.min(dim=0).values
    max_xy = transformed_xy.max(dim=0).values
    transformed_box = torch.tensor(
        [min_xy[0] / scale, min_xy[1] / scale, max_xy[0] / scale, max_xy[1] / scale],
        dtype=target_box.dtype,
        device=target_box.device,
    )
    return box_xyxy_to_cxcywh(transformed_box.unsqueeze(0)).squeeze(0).clamp(0.0, 1.0)


def apply_geometric_augmentation(
    rgb: torch.Tensor,
    masks: torch.Tensor,
    target_box: torch.Tensor,
    *,
    image_size: int,
    augmentation_level: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    params = sample_augmentation_params(image_size=image_size, augmentation_level=augmentation_level)
    if params is None:
        return rgb, masks, target_box

    device = rgb.device
    dtype = rgb.dtype
    forward_matrix = build_forward_augmentation_matrix(params, image_size, dtype=dtype, device=device)
    theta = forward_matrix_to_theta(forward_matrix, image_size, dtype=dtype, device=device)

    rgb_theta = theta.unsqueeze(0).expand(rgb.shape[0], -1, -1)
    rgb_grid = F.affine_grid(rgb_theta, rgb.shape, align_corners=False)
    rgb = F.grid_sample(rgb, rgb_grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    mask_theta = theta.unsqueeze(0).expand(masks.shape[0], -1, -1)
    mask_grid = F.affine_grid(mask_theta, masks.shape, align_corners=False)
    masks = F.grid_sample(masks, mask_grid, mode="nearest", padding_mode="zeros", align_corners=False)
    masks = (masks > 0.5).float()

    target_box = transform_normalized_box(target_box, forward_matrix, image_size=image_size)
    return rgb, masks, target_box


def build_center_targets(
    target_boxes: torch.Tensor,
    grid_h: int,
    grid_w: int,
    *,
    sigma: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_x = torch.clamp((target_boxes[:, 0] * grid_w).long(), 0, grid_w - 1)
    target_y = torch.clamp((target_boxes[:, 1] * grid_h).long(), 0, grid_h - 1)
    target_indices = target_y * grid_w + target_x

    yy, xx = torch.meshgrid(
        torch.arange(grid_h, device=target_boxes.device, dtype=target_boxes.dtype),
        torch.arange(grid_w, device=target_boxes.device, dtype=target_boxes.dtype),
        indexing="ij",
    )
    heatmaps = []
    sigma_sq = max(float(sigma), 1e-6) ** 2
    for cx, cy in zip(target_x.tolist(), target_y.tolist()):
        dist_sq = (xx - float(cx)).pow(2) + (yy - float(cy)).pow(2)
        heatmaps.append(torch.exp(-0.5 * dist_sq / sigma_sq))
    return target_indices, torch.stack(heatmaps, dim=0)
