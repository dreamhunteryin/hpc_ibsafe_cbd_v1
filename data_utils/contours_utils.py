import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage.morphology import remove_small_holes


def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    if mask.max() == 0:
        return mask.astype(bool)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask.astype(bool)

    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return labels == largest


def extract_boundary_avoiding_green_black(
    image,
    mask,
    max_hole_area: int = 1500,
    outer_band: int = 15,
    neighborhood: int = 31,
    bad_fraction_thresh: float = 0.8,
    black_v_max: int = 35,
    green_h_range: tuple[int, int] = (35, 95),
    green_s_min: int = 40,
    green_v_min: int = 30,
    keep_largest_boundary_piece: bool = True,
    return_debug: bool = False,
):
    """
    Parameters
    ----------
    image : PIL image or RGB numpy array, shape (H, W, 3)
    mask : binary mask, shape (H, W)

    HSV thresholds use OpenCV convention:
    H in [0, 179], S/V in [0, 255].
    """
    if hasattr(image, "mode"):
        image = np.asarray(image.convert("RGB"))
    else:
        image = np.asarray(image)

    mask = np.asarray(mask) > 0

    clean_mask = _largest_connected_component(mask)
    clean_mask = remove_small_holes(clean_mask, area_threshold=max_hole_area)

    boundary = clean_mask & ~ndi.binary_erosion(
        clean_mask, structure=np.ones((3, 3), dtype=bool)
    )

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    black = v <= black_v_max

    h0, h1 = green_h_range
    green = (h >= h0) & (h <= h1) & (s >= green_s_min) & (v >= green_v_min)

    bad_pixels = black | green

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * outer_band + 1, 2 * outer_band + 1)
    )
    outer_ring = cv2.dilate(clean_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    outer_ring &= ~clean_mask

    bad_outer = (bad_pixels & outer_ring).astype(np.float32)
    ring = outer_ring.astype(np.float32)

    ksize = (neighborhood, neighborhood)
    bad_sum = cv2.boxFilter(
        bad_outer, ddepth=-1, ksize=ksize, normalize=False, borderType=cv2.BORDER_REPLICATE
    )
    ring_sum = cv2.boxFilter(
        ring, ddepth=-1, ksize=ksize, normalize=False, borderType=cv2.BORDER_REPLICATE
    )

    frac_bad_outside = np.divide(
        bad_sum,
        ring_sum,
        out=np.zeros_like(bad_sum),
        where=ring_sum > 1e-6,
    )

    kept_boundary = boundary & (frac_bad_outside < bad_fraction_thresh)

    if keep_largest_boundary_piece and kept_boundary.any():
        kept_boundary = _largest_connected_component(kept_boundary)

    if return_debug:
        return kept_boundary, {
            "clean_mask": clean_mask,
            "boundary": boundary,
            "outer_ring": outer_ring,
            "bad_pixels": bad_pixels,
            "frac_bad_outside": frac_bad_outside,
        }

    return kept_boundary


# kept_boundary, dbg = extract_boundary_avoiding_green_black(
#     image,
#     liver_mask,
#     return_debug=True,
# )
# contours, _ = cv2.findContours(
#     kept_boundary.astype(np.uint8),
#     cv2.RETR_EXTERNAL,
#     cv2.CHAIN_APPROX_NONE,
# )
# line_xy = max(contours, key=len)[:, 0, :]