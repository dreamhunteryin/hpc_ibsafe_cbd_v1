from matplotlib import pyplot as plt
from matplotlib.patches import Patch, Polygon, Rectangle
import numpy as np
from .dataset_camma import CammaFrame

try:
    from pycocotools import mask as mask_utils
except ImportError:
    mask_utils = None


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[float, float, float] | np.ndarray | None = None,
    color_idx: int | None = None,
    cmap: str = "Set1",
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Overlay one Gaussian tube channel on the image,
    modifying only pixels where mask > threshold.

    Args:
        image: (H,W,3) uint8 or float image
        mask: (H,W) tensor; if not boolean, will be thresholded at > 0.
        cmap: matplotlib colormap
        alpha: blending strength
    """
    # Prepare image as float in [0,1]
    if image.dtype == np.uint8:
        img = image.astype(np.float32) / 255.0
    else:
        img = image.astype(np.float32).copy()
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    overlay = img.copy()
    pose_mask = mask > 0
    if not pose_mask.any():
        return overlay
    # Get colormap RGB (drop alpha channel)
    if color is None:
        idx = color_idx if color_idx is not None else 0
        colormap = plt.get_cmap(cmap)
        color = np.array(colormap(idx)[:3])
    else:
        color = np.array(color)
    overlay[pose_mask] = (1 - alpha) * img[pose_mask] + alpha * color[:3]
    return overlay


def overlay_masks(
    image: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray | None = None,
    # class_indices: list[int] | None = None,
    colors: list[tuple[float, float, float]] | np.ndarray | None = None,
    **kwargs
) -> np.ndarray:
    """
    Overlay multiple mask channels on the image.

    Args:
        image: (H,W,3) uint8 or float image
        masks: (C,H,W) float32 tensor
        class_indices: list of channel indices to visualize
        kwargs: passed to overlay_mask
    """
    overlay = image.copy()
    # if class_indices is None:
    #     class_indices = list(range(masks.shape[0]))
    for idx, mask in enumerate(masks):
        color_idx, color = None, None
        if labels is not None:
            color_idx = labels[idx]
        if colors is not None:
            color = colors[idx]
        overlay = overlay_mask(overlay, mask, color=color, color_idx=color_idx, **kwargs)
    return overlay


def add_line(ax, vertices, **kwargs):
    xs, ys = zip(*vertices)
    ax.plot(xs, ys, **kwargs)


def get_borders(pixel_array):
    if pixel_array.ndim == 3:
        nonzero = np.argwhere(np.any(pixel_array != 0, axis=-1))
    else:
        nonzero = np.argwhere(pixel_array != 0)

    if nonzero.size == 0:
        height, width = pixel_array.shape[:2]
        return 0, height - 1, 0, width - 1

    h_min, w_min = nonzero.min(axis=0)[:2]
    h_max, w_max = nonzero.max(axis=0)[:2]
    return int(h_min), int(h_max), int(w_min), int(w_max)


def _decode_segmentation(segmentation, height: int, width: int) -> np.ndarray:
    if isinstance(segmentation, np.ndarray):
        mask = segmentation
    else:
        if mask_utils is None:
            raise ImportError(
                "pycocotools is required to visualize COCO segmentations."
            )

        if isinstance(segmentation, dict):
            mask = mask_utils.decode(segmentation)
        elif isinstance(segmentation, (list, tuple)):
            if len(segmentation) == 0:
                return np.zeros((height, width), dtype=np.uint8)
            polygons = segmentation
            if isinstance(polygons[0], (int, float)):
                polygons = [polygons]
            rles = mask_utils.frPyObjects(polygons, height, width)
            mask = mask_utils.decode(rles)
        else:
            raise TypeError(
                "Unsupported segmentation format. Expected a COCO RLE, polygon list, "
                "or a dense numpy mask."
            )

    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = np.any(mask, axis=-1)
    return mask.astype(np.uint8)


def _normalize_obb(obb) -> np.ndarray:
    corners = np.asarray(obb, dtype=np.float32)
    if corners.shape == (4, 2):
        return corners
    if corners.size != 8:
        raise ValueError(f"Expected OBB with 8 values or shape (4, 2), got {corners.shape}.")
    return corners.reshape(4, 2)


def show_camma_frame_annotations(
    frame: CammaFrame,
    annotation_types: list[str] | tuple[str, ...],
    category_ids: list[int] | None = None,
    crop_black_borders: bool = False,
    bbox_anchor: str = "topleft", # or "center" 
    category_id_to_name: dict[int, str] | None = None,
    colormap: str = "Set1",
    mask_alpha: float = 0.5,
    figsize: tuple[float, float] = (10, 6),
    ax=None,
):
    valid_types = {"mask", "bbox", "obb"}
    annotation_types = tuple(dict.fromkeys(annotation_types))
    invalid_types = sorted(set(annotation_types) - valid_types)
    if invalid_types:
        raise ValueError(
            f"Unsupported annotation_types={invalid_types}. Expected values in {sorted(valid_types)}."
        )

    selected_category_ids = None if category_ids is None else set(category_ids)
    annotations = [
        annotation
        for annotation in frame.annotations
        if selected_category_ids is None
        or annotation["category_id"] in selected_category_ids
    ]

    pixel_array = frame.pixel_array
    if crop_black_borders:
        h_min, h_max, w_min, w_max = get_borders(pixel_array)
        h_slice = slice(h_min, h_max + 1)
        w_slice = slice(w_min, w_max + 1)
    else:
        h_min, w_min = 0, 0
        h_slice = slice(None)
        w_slice = slice(None)

    image = pixel_array[h_slice, w_slice].copy()
    displayed_category_ids = []
    displayed_category_ids_seen = set()

    def record_category(category_id: int) -> None:
        if category_id not in displayed_category_ids_seen:
            displayed_category_ids.append(category_id)
            displayed_category_ids_seen.add(category_id)

    if "mask" in annotation_types:
        masks = []
        labels = []
        for annotation in annotations:
            segmentation = annotation.get("segmentation")
            if segmentation is None:
                continue
            mask = _decode_segmentation(
                segmentation,
                height=pixel_array.shape[0],
                width=pixel_array.shape[1],
            )
            masks.append(mask[h_slice, w_slice])
            labels.append(annotation["category_id"])
            record_category(annotation["category_id"])

        if masks:
            image = overlay_masks(
                image,
                np.stack(masks),
                labels=np.asarray(labels),
                cmap=colormap,
                alpha=mask_alpha,
            )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    ax.imshow(image)

    cmap = plt.get_cmap(colormap)

    for annotation in annotations:
        category_id = annotation["category_id"]
        color = cmap(category_id)

        if "bbox" in annotation_types and annotation.get("bbox") is not None:
            if bbox_anchor == "center":
                x_center, y_center, width, height = annotation["bbox"]
                x_min = x_center - width / 2
                y_min = y_center - height / 2
            elif bbox_anchor == "topleft":
                x_min, y_min, width, height = annotation["bbox"]
            rect = Rectangle(
                (x_min - w_min, y_min - h_min),
                width,
                height,
                fill=False,
                edgecolor=color,
                linewidth=2,
            )
            ax.add_patch(rect)
            record_category(category_id)

        if "obb" in annotation_types and annotation.get("obb") is not None:
            corners = _normalize_obb(annotation["obb"]).copy()
            corners[:, 0] -= w_min
            corners[:, 1] -= h_min
            poly = Polygon(
                corners,
                closed=True,
                fill=False,
                edgecolor=color,
                linewidth=2,
            )
            ax.add_patch(poly)
            record_category(category_id)

    if category_id_to_name is not None and displayed_category_ids:
        legend_handles = [
            Patch(
                facecolor=cmap(category_id),
                edgecolor=cmap(category_id),
                label=category_id_to_name.get(category_id, str(category_id)),
                alpha=mask_alpha,
            )
            for category_id in displayed_category_ids
        ]
        ax.legend(handles=legend_handles, loc="upper right")

    ax.set_axis_off()
    plt.tight_layout()
    return fig, ax


# def show_rotated_boxes(sample, colormap: str = 'Set1'):
#     # sample: ICGSample
#     img = sample.image
#     if img is None:
#         raise ValueError("sample.image is None (use dataset with load_image=True).")

#     # Handle torch CHW tensors if your transform returned tensors
#     if "torch" in str(type(img)):
#         import torch
#         if isinstance(img, torch.Tensor):
#             if img.ndim == 3 and img.shape[0] in (1, 3):  # CHW -> HWC
#                 img = img.detach().cpu().permute(1, 2, 0).numpy()
#             else:
#                 img = img.detach().cpu().numpy()

#     fig, ax = plt.subplots(figsize=(10, 6))
#     ax.imshow(img)

#     cmap = plt.get_cmap(colormap)

#     for i, corners in enumerate(sample.obb_xy):  # corners shape: (4, 2), in pixels
#         poly = Polygon(corners, closed=True, fill=False, edgecolor=cmap(i), linewidth=2)
#         ax.add_patch(poly)

#         label = sample.class_names[i] if i < len(sample.class_names) else str(int(sample.class_ids[i]))
#         x, y = corners[0]
#         ax.text(
#             x, y, label,
#             color=cmap(i), fontsize=9,
#             bbox=dict(facecolor="black", alpha=0.5, pad=2)
#         )

#     ax.set_axis_off()
#     plt.tight_layout()
#     plt.show()
