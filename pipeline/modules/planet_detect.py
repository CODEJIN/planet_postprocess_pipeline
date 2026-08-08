"""
Planet detection utilities for SER Crop preprocessing.

Detects the planet disk in a raw (mono/Bayer) or RGB frame, validates that
the planet is fully on-screen and not deformed, and returns its geometric
centre and bounding-box dimensions.

Key design decisions:
  - Geometric centroid (bounding-box centre) rather than brightness-weighted
    centroid avoids systematic drift caused by Jupiter's uneven atmospheric bands.
  - Triangle auto-threshold (Zack 1977) works well across a wide range of
    exposure levels without tuning.
  - Aspect-ratio and straight-edge checks catch partially-clipped frames that
    the boundary test alone might miss (e.g. when the planet is centred but
    one side is cut by a data-transfer error).
  - Ringed planets (e.g. Saturn) attach a wide, faint ring to the disk in the
    thresholded mask, which would otherwise fail the circularity checks. A
    percentile re-threshold isolates the brighter disk core before those
    checks run, while boundary/diameter/centroid still use the full
    disk+ring extent. Filters where the disk is *not* the brightest part of
    the blob (e.g. Saturn's CH4 band) should pass ``skip_shape_check=True``.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np


# ── Internal helpers ──────────────────────────────────────────────────────────

def _largest_component(
    mask: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (stats, centroid) for the largest connected component in *mask*.

    Returns ``(None, None)`` when:
      - no foreground components are found, OR
      - the largest component covers > 90 % of the image area
        (indicates a failed threshold — whole frame is "lit up").
    """
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    if num_labels <= 1:
        return None, None

    # Ignore background label 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = int(np.argmax(areas)) + 1  # +1 because we skipped label 0

    h, w = mask.shape
    if stats[largest_idx, cv2.CC_STAT_AREA] > w * h * 0.9:
        return None, None

    return stats[largest_idx], centroids[largest_idx]


# ── Public API ────────────────────────────────────────────────────────────────

def analyze_planet(
    image: np.ndarray,
    min_diameter: int = 0,
    padding: int = 1,
    aspect_ratio_limit: float = 0.2,
    straight_edge_limit: float = 0.5,
    core_percentile: float = 60.0,
    skip_shape_check: bool = False,
) -> Optional[Dict]:
    """Detect the planet in *image* and validate its shape.

    Accepts mono (H, W) or RGB/BGR (H, W, 3) arrays of any dtype.

    Parameters
    ----------
    image:               Input frame (mono or colour).
    min_diameter:        Minimum diameter in pixels; smaller detections are
                         rejected (set to 0 to disable).
    padding:             Pixels from the image edge within which the bounding
                         box must not reach (clipping check).
    aspect_ratio_limit:  Maximum deviation from 1:1; e.g. 0.2 means the
                         shorter axis must be ≥ 80 % of the longer axis.
    straight_edge_limit: Fraction of a bounding-box edge that may be lit
                         before the frame is considered clipped.
    core_percentile:     Intensity percentile (within the detected blob) used
                         to isolate the bright disk core from a fainter
                         attached ring before the shape checks run — see
                         "Disk-core isolation" below.
    skip_shape_check:    When True, bypass the aspect-ratio and straight-edge
                         checks entirely (still applies boundary + diameter).
                         Use for filters where the disk is *not* the brightest
                         part of the blob (e.g. Saturn's CH4/methane band,
                         where atmospheric absorption makes the globe darker
                         than its icy rings) — there, no brightness-based
                         core isolation can find a round "disk" to validate.

    Returns
    -------
    dict with keys ``centroid``, ``width``, ``height``, ``mask``
    or ``None`` if the frame should be rejected.
    """
    # ── Convert to 8-bit grayscale ─────────────────────────────────────────────
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    if gray.dtype == np.uint16:
        image_8 = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    else:
        image_8 = gray.astype(np.uint8)

    # ── Noise reduction + threshold ────────────────────────────────────────────
    blurred = cv2.GaussianBlur(image_8, (5, 5), 0)
    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE
    )

    # ── Largest connected component (disk, plus any attached rings) ───────────
    stats, _ = _largest_component(thresh)
    if stats is None:
        return None

    x, y, bw, bh, _ = (
        int(stats[cv2.CC_STAT_LEFT]),
        int(stats[cv2.CC_STAT_TOP]),
        int(stats[cv2.CC_STAT_WIDTH]),
        int(stats[cv2.CC_STAT_HEIGHT]),
        int(stats[cv2.CC_STAT_AREA]),
    )
    h, w = thresh.shape

    # 1. Boundary check — planet (incl. rings) must not touch the image edge
    if (
        x < padding
        or y < padding
        or (x + bw) > (w - padding)
        or (y + bh) > (h - padding)
    ):
        return None

    if not skip_shape_check:
        # ── Disk-core isolation ─────────────────────────────────────────────
        # Ringed planets (e.g. Saturn) attach a wide, faint ring to the disk in
        # the loose triangle-threshold mask, which blows out the blob's
        # bounding-box aspect ratio (rings span ~2.2x the disk diameter) and
        # would otherwise fail the circularity check below even for a good
        # frame. In continuum filters the disk is brighter than the ring, so
        # re-thresholding at a high percentile of the blob's own pixel values
        # isolates a compact, round core (the disk) and drops the fainter
        # ring. Shape-based stripping (e.g. morphological opening) is not
        # enough on heavily-blurred/noisy single frames, where the ring's
        # binarised footprint can be nearly as thick as the disk itself —
        # intensity is the more reliable signal here.
        # If nothing distinct survives (e.g. a solid Venus crescent has no
        # separate faint appendage to strip), fall back to the full blob.
        shape_x, shape_y, shape_bw, shape_bh = x, y, bw, bh
        core_mask_roi = None
        roi_vals = blurred[y : y + bh, x : x + bw]
        roi_mask = thresh[y : y + bh, x : x + bw] > 0
        vals = roi_vals[roi_mask]
        if vals.size > 0:
            core_thv = np.percentile(vals, core_percentile)
            core_bin = (((roi_vals >= core_thv) & roi_mask).astype(np.uint8)) * 255
            core_stats, _ = _largest_component(core_bin)
            if core_stats is not None:
                cx0 = int(core_stats[cv2.CC_STAT_LEFT])
                cy0 = int(core_stats[cv2.CC_STAT_TOP])
                shape_bw = int(core_stats[cv2.CC_STAT_WIDTH])
                shape_bh = int(core_stats[cv2.CC_STAT_HEIGHT])
                shape_x, shape_y = x + cx0, y + cy0
                core_mask_roi = core_bin[cy0 : cy0 + shape_bh, cx0 : cx0 + shape_bw]

        # 2. Aspect-ratio check — disk core (ring excluded) must be roughly circular
        aspect = shape_bw / shape_bh if shape_bw < shape_bh else shape_bh / shape_bw
        if aspect < (1.0 - aspect_ratio_limit):
            return None

        # 3. Straight-edge check — disk-core edges must be curved, not flat
        roi = core_mask_roi if core_mask_roi is not None else (
            thresh[shape_y : shape_y + shape_bh, shape_x : shape_x + shape_bw]
        )
        edge_ratios = [
            np.count_nonzero(roi[0, :]) / shape_bw,    # top edge
            np.count_nonzero(roi[-1, :]) / shape_bw,   # bottom edge
            np.count_nonzero(roi[:, 0]) / shape_bh,    # left edge
            np.count_nonzero(roi[:, -1]) / shape_bh,   # right edge
        ]
        if max(edge_ratios) > straight_edge_limit:
            return None

    # 4. Minimum diameter
    diameter = max(bw, bh)
    if diameter < min_diameter:
        return None

    # 5. Geometric centroid (bounding-box centre — avoids brightness bias)
    centroid_x = x + bw / 2.0
    centroid_y = y + bh / 2.0

    return {
        "centroid": (centroid_x, centroid_y),
        "width": bw,
        "height": bh,
        "mask": thresh,
    }


def get_cropped_frame(
    frame: np.ndarray,
    center: Tuple[float, float],
    size: int,
) -> np.ndarray:
    """Return a square crop of *frame* centred on *center*.

    Pixels outside the original frame are filled with zeros (black).
    Uses ``round()`` (not ``int()``) to avoid a systematic ½-pixel bias
    caused by always truncating towards zero.

    Parameters
    ----------
    frame:   Source image — mono (H, W) or colour (H, W, C).
    center:  (cx, cy) in pixel coordinates (may be fractional).
    size:    Side length of the output square in pixels.
    """
    h, w = frame.shape[:2]
    cx, cy = round(center[0]), round(center[1])

    x1, x2 = cx - size // 2, cx + size // 2
    y1, y2 = cy - size // 2, cy + size // 2

    # Clamp to image bounds
    sx1, sx2 = max(0, x1), min(w, x2)
    sy1, sy2 = max(0, y1), min(h, y2)

    # Allocate black output buffer
    out_shape = (size, size, frame.shape[2]) if frame.ndim == 3 else (size, size)
    out = np.zeros(out_shape, dtype=frame.dtype)

    # Destination offsets inside the output buffer
    dx1, dx2 = sx1 - x1, sx2 - x1
    dy1, dy2 = sy1 - y1, sy2 - y1

    if dy2 > dy1 and dx2 > dx1:
        out[dy1:dy2, dx1:dx2] = frame[sy1:sy2, sx1:sx2]

    return out
