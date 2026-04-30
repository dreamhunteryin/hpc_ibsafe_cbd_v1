from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage


def resize_image_to_square(
    pil_image: PILImage.Image,
    target_size: int,
    resample: int = PILImage.BILINEAR,
) -> PILImage.Image:
    return pil_image.resize((target_size, target_size), resample)


def resize_mask_to_square(mask_np: np.ndarray, target_size: int) -> torch.Tensor:
    mask_t = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0)
    resized_mask = F.interpolate(
        mask_t,
        size=(target_size, target_size),
        mode="nearest",
    ).squeeze(0).squeeze(0)
    return resized_mask > 0.5


def resize_mask_to_original(
    mask_tensor: torch.Tensor,
    orig_h: int,
    orig_w: int,
) -> torch.Tensor:
    if mask_tensor.numel() == 0:
        return torch.zeros((orig_h, orig_w), dtype=mask_tensor.dtype, device=mask_tensor.device)
    return F.interpolate(
        mask_tensor.unsqueeze(0).unsqueeze(0).float(),
        size=(orig_h, orig_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)


def anchored_bbox_to_original_xywh(
    bbox: list[float] | tuple[float, float, float, float],
    bbox_anchor: str = 'topleft',
    orig_w: int | None = None,
    orig_h: int | None = None,
) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    anchor = bbox_anchor.strip().lower()
    if anchor == 'center':
        x1 = x - w / 2.0
        y1 = y - h / 2.0
    elif anchor == 'topleft':
        x1 = x
        y1 = y
    else:
        raise ValueError(
            f'Unsupported bbox_anchor={bbox_anchor!r}. Expected one of: (\'topleft\', \'center\').'
        )

    x2 = x1 + w
    y2 = y1 + h
    if orig_w is not None:
        x1 = max(0.0, min(float(orig_w), x1))
        x2 = max(0.0, min(float(orig_w), x2))
    if orig_h is not None:
        y1 = max(0.0, min(float(orig_h), y1))
        y2 = max(0.0, min(float(orig_h), y2))
    return float(x1), float(y1), max(0.0, float(x2 - x1)), max(0.0, float(y2 - y1))


def coco_bbox_to_normalized_cxcywh(
    bbox_coco: list[float] | tuple[float, float, float, float],
    orig_w: int,
    orig_h: int,
    bbox_anchor: str = 'topleft',
) -> torch.Tensor:
    x, y, w, h = anchored_bbox_to_original_xywh(bbox_coco, bbox_anchor=bbox_anchor)
    cx = x + w / 2.0
    cy = y + h / 2.0
    return torch.tensor(
        [cx / orig_w, cy / orig_h, w / orig_w, h / orig_h],
        dtype=torch.float32,
    )


def normalized_cxcywh_to_original_xywh(
    box: list[float] | tuple[float, float, float, float],
    orig_w: int,
    orig_h: int,
) -> tuple[float, float, float, float]:
    cx, cy, w_norm, h_norm = box
    x1 = (cx - w_norm / 2.0) * orig_w
    y1 = (cy - h_norm / 2.0) * orig_h
    x2 = (cx + w_norm / 2.0) * orig_w
    y2 = (cy + h_norm / 2.0) * orig_h

    x1 = max(0.0, min(float(orig_w), x1))
    y1 = max(0.0, min(float(orig_h), y1))
    x2 = max(0.0, min(float(orig_w), x2))
    y2 = max(0.0, min(float(orig_h), y2))
    return x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)


def compute_detection_scores(
    pred_logits: torch.Tensor,
    presence_logit: torch.Tensor | None = None,
) -> torch.Tensor:
    scores = pred_logits.squeeze(-1).sigmoid()
    if presence_logit is None:
        return scores

    presence_score = presence_logit.sigmoid()
    if presence_score.ndim == 0:
        return scores * presence_score

    if presence_score.ndim == 1:
        if presence_score.numel() == 1:
            return scores * presence_score.squeeze(0)
        return scores * presence_score

    if presence_score.ndim == 2 and presence_score.shape[-1] == 1:
        return scores * presence_score.squeeze(-1)

    raise ValueError(
        f"Unsupported presence_logit shape: {tuple(presence_logit.shape)}"
    )
