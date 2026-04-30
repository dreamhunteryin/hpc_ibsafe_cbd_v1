from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from PIL import Image as PILImage


@dataclass
class InferenceMetadata:
    coco_image_id: int
    original_image_id: int
    original_category_id: int
    original_size: Tuple[int, int]
    object_id: int
    frame_index: int
    is_conditioning_only: Optional[bool] = False


@dataclass
class FindQuery:
    query_text: str
    image_id: int
    object_ids_output: List[int]
    is_exhaustive: bool
    query_processing_order: int = 0
    input_bbox: Optional[torch.Tensor] = None
    input_bbox_label: Optional[torch.Tensor] = None
    input_points: Optional[torch.Tensor] = None
    semantic_target: Optional[torch.Tensor] = None
    is_pixel_exhaustive: Optional[bool] = None


@dataclass
class FindQueryLoaded(FindQuery):
    inference_metadata: Optional[InferenceMetadata] = None


@dataclass
class Object:
    bbox: torch.Tensor
    area: float
    object_id: Optional[int] = -1
    frame_index: Optional[int] = -1
    segment: Optional[Union[torch.Tensor, dict]] = None
    is_crowd: bool = False
    source: Optional[str] = None


@dataclass
class Image:
    data: Union[torch.Tensor, PILImage.Image]
    objects: List[Object]
    size: Tuple[int, int]
    blurring_mask: Optional[dict] = None


@dataclass
class Datapoint:
    find_queries: List[FindQueryLoaded]
    images: List[Image]
    raw_images: Optional[List[PILImage.Image]] = None
