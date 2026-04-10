"""
À trous undecimated wavelet sharpening with B3-spline kernel.

Replicates WaveSharp / Registax 6 wavelet sharpening.

Algorithm:
  output = clip(original + Σ(detail_i × gain_i), 0, full_range)

where detail_i is the à trous wavelet detail at scale 2^i pixels, and
gain_i is derived from the per-layer "amount" (0–200, WaveSharp-compatible).

MAX_GAINS: calibrated empirically from a WaveSharp reference output
  (sharpen_filter=0.1, power_function=1.0, amount=200 on layers 1–3).

Key properties:
  - Mean-preserving: the sharpening adds zero-mean detail, so the image
    brightness is unchanged (unlike auto-stretch approaches).
  - Fine-scale emphasis: finest detail layers carry the highest gain,
    matching human perception of "sharpness".
  - Soft threshold: optional per-layer noise gate (sharpen_filter) that
    suppresses very small coefficients before amplification.

References:
  Starck, J.-L. & Murtagh, F. (2006). Astronomical Image and Data Analysis.
  Bijaoui, A. (1991). Image restoration and the wavelet transform.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


# ── Constants ──────────────────────────────────────────────────────────────────

# B3-spline scaling function (5-tap, separable)
_B3 = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float64) / 16.0

# Maximum extra gain per level when amount=200 (WaveSharp-calibrated).
# Derived by reverse-engineering a WaveSharp reference output (amount=200,
# sharpen_filter=0.1, power=1.0) using single-level OLS regression on
# a real Jupiter stack.  These are "extra" gains: total multiplier on
# detail_i = (1 + gain_i).
#
# Level 0 = finest (~2 px),  Level 5 = coarsest (~64 px).
_MAX_GAINS = [29.15, 9.48, 0.0, 0.0, 0.0, 0.0]


# ── Low-level building blocks ──────────────────────────────────────────────────

def _build_atrous_kernel(level: int) -> np.ndarray:
    """Build the à trous B3-spline kernel for the given decomposition level.

    At level *i* the inter-tap spacing is 2^i, yielding a kernel of length
    (4 × 2^i + 1) with (2^i − 1) zeros inserted between each of the 5 taps.
    """
    step = 1 << level          # 2^level
    size = (len(_B3) - 1) * step + 1
    kernel = np.zeros(size, dtype=np.float64)
    kernel[::step] = _B3
    return kernel


def _convolve1d_reflect(arr: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """numpy drop-in for scipy.ndimage.convolve1d(arr, kernel, axis, mode='reflect')."""
    pad = len(kernel) // 2
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (pad, pad)
    padded = np.pad(arr, pad_width, mode="reflect")
    result = np.zeros_like(arr, dtype=np.float64)
    for i, k in enumerate(kernel):
        sl = [slice(None)] * arr.ndim
        sl[axis] = slice(i, i + arr.shape[axis])
        result += k * padded[tuple(sl)]
    return result


def _smooth(image: np.ndarray, level: int) -> np.ndarray:
    """Apply separable B3-spline smoothing at the given à trous level."""
    kernel = _build_atrous_kernel(level)
    out = _convolve1d_reflect(image, kernel, axis=0)
    out = _convolve1d_reflect(out,   kernel, axis=1)
    return out


def _soft_threshold(w: np.ndarray, threshold: float) -> np.ndarray:
    """Soft threshold: suppress |w| < threshold (WaveSharp 'sharpen filter').

    Implements Donoho-style soft thresholding:
        output = sign(w) × max(|w| - threshold, 0)

    This preserves large coefficients (edges) while attenuating small ones (noise).
    """
    if threshold <= 0.0:
        return w
    return np.sign(w) * np.maximum(np.abs(w) - threshold, 0.0)


def _noise_sigma(w: np.ndarray) -> float:
    """Estimate noise standard deviation from wavelet detail using MAD.

    sigma = MAD(w) / 0.6745

    The MAD-based estimator is robust to heavy-tailed distributions
    (edges/features) that are common in wavelet detail coefficients.
    Equivalent to std for Gaussian noise, more robust for non-Gaussian.
    """
    return float(np.median(np.abs(w)) / 0.6745)


# ── Border taper ──────────────────────────────────────────────────────────────

def border_taper(
    image: np.ndarray,
    top: int = 0,
    bottom: int = 0,
    left: int = 0,
    right: int = 0,
) -> np.ndarray:
    """Cosine-fade the outermost pixels on each side to zero.

    Designed to be applied **before** wavelet sharpening to eliminate
    stacking boundary gradients (from de-rotation warp BORDER_CONSTANT=0)
    before the wavelet can amplify them.

    Each side is tapered independently so the width can be clamped to the
    actual background margin on that side (use safe_taper_widths() to
    compute side-adaptive widths from the detected disk geometry).

    Why this works without ringing:
      - The taper boundary lies in the near-zero background region.
      - Background × taper ≈ 0 → wavelet sees no new high-contrast edge.
      - Only axis-aligned transitions, not circular, so no ring artifact.

    Args:
        image:   Float array (2-D or 3-D), any range.
        top:     Pixels to taper on the top edge    (0 = skip).
        bottom:  Pixels to taper on the bottom edge (0 = skip).
        left:    Pixels to taper on the left edge   (0 = skip).
        right:   Pixels to taper on the right edge  (0 = skip).

    Returns:
        Tapered float array, same dtype and shape as *image*.
    """
    if not any([top, bottom, left, right]):
        return image

    h, w = image.shape[:2]

    def _ramp(n: int) -> np.ndarray:
        return (0.5 * (1.0 - np.cos(np.pi * np.arange(n) / n))).astype(np.float32)

    mask = np.ones((h, w), dtype=np.float32)
    if top    > 0: mask[:top,    :] = np.minimum(mask[:top,    :], _ramp(top)[:, None])
    if bottom > 0: mask[-bottom:,:] = np.minimum(mask[-bottom:,:], _ramp(bottom)[::-1, None])
    if left   > 0: mask[:,  :left ] = np.minimum(mask[:,  :left ], _ramp(left)[None, :])
    if right  > 0: mask[:, -right:] = np.minimum(mask[:, -right:], _ramp(right)[None, ::-1])

    if image.ndim == 3:
        mask = mask[:, :, None]

    return (image * mask).astype(image.dtype)


def safe_taper_widths(
    image: np.ndarray,
    requested_px: int,
    safety_px: int = 5,
    content_threshold_frac: float = 0.05,
) -> tuple:
    """Compute per-side taper widths guaranteed not to overlap with the planet.

    Scans mean brightness profiles from each edge inward to find where actual
    image content (planet or sky glow) begins.  The taper on that side is
    limited to (content_start - safety_px) so it stays entirely in the
    zero/near-zero stacking gradient zone.

    If the planet extends to the image edge (no background strip), the taper
    for that side is 0 — no taper is applied rather than clipping the planet.

    Args:
        image:                  2-D float image.
        requested_px:           Desired maximum taper width per side.
        safety_px:              Extra gap between taper end and content start.
        content_threshold_frac: Fraction of image peak below which pixels are
                                considered background/artifact (default 0.05 =
                                5 % of max).  Increase if limb is very bright.

    Returns:
        (top, bottom, left, right) — per-side widths in pixels.
    """
    peak = float(image.max())
    if peak < 1e-6:
        return 0, 0, 0, 0
    threshold = peak * content_threshold_frac

    # Collapse each axis to a 1-D brightness profile
    col_profile = image.mean(axis=0)   # length W — brightness per column
    row_profile = image.mean(axis=1)   # length H — brightness per row

    def _first_above(arr: np.ndarray) -> int:
        """First index where arr exceeds threshold (scan from index 0)."""
        for i, v in enumerate(arr):
            if v > threshold:
                return i
        return len(arr)   # all background

    left_start   = _first_above(col_profile)
    right_start  = _first_above(col_profile[::-1])
    top_start    = _first_above(row_profile)
    bottom_start = _first_above(row_profile[::-1])

    def _width(content_px: int) -> int:
        return max(0, min(requested_px, content_px - safety_px))

    return _width(top_start), _width(bottom_start), _width(left_start), _width(right_start)


# ── Public API ─────────────────────────────────────────────────────────────────

def decompose(image: np.ndarray, levels: int = 6) -> List[np.ndarray]:
    """Decompose *image* into à trous wavelet coefficients.

    Args:
        image:  2-D float array (any range; float64 precision used internally).
        levels: Number of detail layers to extract.

    Returns:
        List of length ``levels + 1``:
        ``[detail_0, detail_1, ..., detail_{levels-1}, residual]``

        detail_i  = contribution at spatial scale ~2^i … 2^(i+1) pixels.
        residual  = low-frequency approximation after *levels* smoothings.
    """
    coeffs: List[np.ndarray] = []
    current = image.astype(np.float64)

    for i in range(levels):
        smoothed = _smooth(current, i)
        coeffs.append(current - smoothed)   # detail layer i
        current = smoothed

    coeffs.append(current)   # residual (low-frequency)
    return coeffs


def amounts_to_weights(
    amounts: List[float],
    power: float = 1.0,
    max_gains: Optional[List[float]] = None,
) -> List[float]:
    """Convert WaveSharp-style amounts (0–200) to internal extra-gain weights.

    Args:
        amounts:   Per-level amount values, same range as WaveSharp (0–200).
                   length must equal the number of wavelet levels.
        power:     WaveSharp 'power function' exponent (1.0 = linear).
                   Values > 1 give more aggressive sharpening at high amounts.
        max_gains: Override the calibrated _MAX_GAINS table.

    Returns:
        List of per-level extra-gain weights for use in :func:`sharpen`.
    """
    mg = max_gains if max_gains is not None else _MAX_GAINS
    weights = []
    for i, amt in enumerate(amounts):
        g = mg[i] if i < len(mg) else 0.0
        w = (amt / 200.0) ** power * g
        weights.append(w)
    return weights


def reconstruct(
    coeffs: List[np.ndarray],
    weights: List[float],
    sharpen_filter: float = 0.0,
) -> np.ndarray:
    """Reconstruct a sharpened image from wavelet coefficients.

    Formula:
        output = original + Σ( soft_threshold(detail_i, thr_i) × weights[i] )

    Args:
        coeffs:         Output of :func:`decompose`.
        weights:        Per-level extra-gain (length == levels).
        sharpen_filter: Soft-threshold factor (WaveSharp 'sharpen filter'),
                        applied as thr_i = sharpen_filter × std(detail_i).
                        0.0 (default) = no thresholding.

    Returns:
        Float64 array (same shape as input, **not yet clipped**).
    """
    # Start from the original (residual + all details = original)
    original = coeffs[-1].copy()
    for d in coeffs[:-1]:
        original = original + d   # reconstruct original exactly

    result = original.copy()
    for detail, w in zip(coeffs[:-1], weights):
        if w == 0.0:
            continue
        thr = sharpen_filter * _noise_sigma(detail) if sharpen_filter > 0.0 else 0.0
        d_thr = _soft_threshold(detail, thr)
        result = result + d_thr * w

    return result


def sharpen_color(
    image: np.ndarray,
    levels: int = 6,
    amounts: Optional[List[float]] = None,
    weights: Optional[List[float]] = None,
    power: float = 1.0,
    sharpen_filter: float = 0.0,
) -> np.ndarray:
    """Sharpen a colour (H, W, 3) RGB float [0, 1] image via L-channel sharpening.

    Converts RGB → Lab, sharpens only the L (luminance) channel using à trous
    wavelet sharpening, then converts back to RGB.  Chrominance (a, b) is
    preserved unchanged, so colour balance is unaffected.

    Args:
        image:          Float32 (H, W, 3) RGB array in [0, 1].
        levels:         Number of wavelet decomposition levels.
        amounts:        Per-level WaveSharp amounts (0–200).
        weights:        Raw per-level gain (overrides amounts if given).
        power:          WaveSharp power-function exponent.
        sharpen_filter: Soft-threshold noise-gate coefficient.

    Returns:
        Float32 (H, W, 3) RGB array in [0, 1], with sharpened luminance.
    """
    import cv2 as _cv2
    # RGB [0,1] → BGR → Lab (L in [0,100], a/b in [-127,127])
    bgr = _cv2.cvtColor(image.astype(np.float32), _cv2.COLOR_RGB2BGR)
    lab = _cv2.cvtColor(bgr, _cv2.COLOR_BGR2Lab)

    L = lab[:, :, 0] / 100.0           # [0, 1]
    L_sharp = sharpen(L, levels=levels, amounts=amounts, weights=weights,
                      power=power, sharpen_filter=sharpen_filter)
    lab[:, :, 0] = np.clip(L_sharp * 100.0, 0.0, 100.0)

    bgr_sharp = _cv2.cvtColor(lab, _cv2.COLOR_Lab2BGR)
    rgb_sharp = _cv2.cvtColor(bgr_sharp, _cv2.COLOR_BGR2RGB)
    return np.clip(rgb_sharp, 0.0, 1.0).astype(np.float32)


def estimate_limb_overshoot_px(
    original: np.ndarray,
    sharpened: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    n_angles: int = 36,
    threshold_frac: float = 0.10,
    max_scan_px: int = 50,
) -> float:
    """Measure inward extent of wavelet overshoot ring at the disk edge.

    Computes |sharpened - original| and samples radially inward from the disk
    edge at *n_angles* equally-spaced directions.  For each direction, finds
    how far inside the edge the diff remains above *threshold_frac* × (peak
    diff along that radial line).  Returns the 75th-percentile depth across
    all angles — a conservative, robust estimate of the ring width.

    Args:
        original:       Pre-wavelet 2-D float image.
        sharpened:      Post-wavelet 2-D float image (same shape).
        cx, cy:         Disk centre in pixels.
        radius:         Disk radius in pixels.
        n_angles:       Number of radial directions to sample.
        threshold_frac: Fraction of per-angle peak diff used as the
                        significance threshold (default 0.10 = 10 %).
        max_scan_px:    Maximum inward depth to scan in pixels.

    Returns:
        Estimated ring depth in pixels (float). Falls back to 12.0 if the
        measurement is unreliable.
    """
    diff = np.abs(original.astype(np.float64) - sharpened.astype(np.float64))
    if diff.ndim == 3:
        diff = diff.mean(axis=2)

    h, w = diff.shape
    max_scan = min(max_scan_px, int(radius * 0.30))
    if max_scan < 1:
        return 8.0

    depths: List[float] = []
    for angle in np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False):
        cos_a = float(np.cos(angle))
        sin_a = float(np.sin(angle))

        # Sample from disk edge (d=0) inward (d=max_scan)
        profile: List[float] = []
        for d in range(max_scan + 1):
            r = radius - d
            if r < 0:
                break
            xi = int(round(cx + r * cos_a))
            yi = int(round(cy + r * sin_a))
            if 0 <= xi < w and 0 <= yi < h:
                profile.append(float(diff[yi, xi]))
            else:
                profile.append(0.0)

        if not profile:
            continue

        peak = max(profile)
        if peak < 1e-8:
            depths.append(0.0)
            continue

        thr = peak * threshold_frac
        # Find the deepest index still above threshold
        depth = 0
        for d_idx, v in enumerate(profile):
            if v >= thr:
                depth = d_idx
        depths.append(float(depth))

    if not depths:
        return 12.0

    return float(np.percentile(depths, 75))


def blend_limb_taper(
    original: np.ndarray,
    sharpened: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    feather_px: float,
) -> np.ndarray:
    """Blend sharpened and original images with a soft disk-edge taper.

    Inside the blend zone (``radius - feather_px`` … ``radius``) the output
    transitions smoothly from fully-sharpened (disk interior) to the original
    pre-wavelet image (disk edge and background).  Because the original has no
    overshoot ring, this suppresses the ring without creating a new
    discontinuity — unlike multiplying by a mask that zeros out the edge.

        result = sharpened × mask + original × (1 − mask)

    where ``mask = clip((radius − dist) / feather_px, 0, 1)``.

    Args:
        original:   Pre-wavelet float array (2-D or 3-D, any range).
        sharpened:  Post-wavelet float array, same shape.
        cx, cy:     Disk centre in pixels.
        radius:     Disk radius in pixels.
        feather_px: Width of the blend zone in pixels (inward from edge).

    Returns:
        Blended float32 array, same shape as input.
    """
    h, w = original.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = np.clip((radius - dist) / max(float(feather_px), 1.0), 0.0, 1.0).astype(np.float32)

    if original.ndim == 3:
        mask = mask[:, :, np.newaxis]

    return (sharpened * mask + original * (1.0 - mask)).astype(np.float32)


def _make_disk_weight(
    h: int, w: int,
    cx: float, cy: float,
    radius: float,
    feather_px: float,
) -> np.ndarray:
    """Soft circular mask: 1.0 inside disk, linear fade to 0 over feather_px at edge."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return np.clip((radius - dist) / max(feather_px, 1.0), 0.0, 1.0).astype(np.float32)


def sharpen_disk_aware(
    image: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    levels: int = 6,
    amounts: Optional[List[float]] = None,
    weights: Optional[List[float]] = None,
    power: float = 1.0,
    sharpen_filter: float = 0.0,
    edge_feather_factor: float = 2.0,
) -> np.ndarray:
    """À trous wavelet sharpening with per-level spatial edge feathering.

    Each detail level L contributes:

        detail_L × gain_L × spatial_weight_L(pixel)

    where ``spatial_weight_L`` fades from 1.0 (disk interior) to 0.0 at the
    disk edge over a zone of width ``feather_L = 2^L × edge_feather_factor``
    pixels.

    Because finer levels (small L, narrow B3 kernel ~2^L px) have a smaller
    feather zone, they sharpen almost to the disk limb.  Coarser levels (large
    L, wide kernel) fade out further inward — suppressing their overshoot ring
    without widening the total blurred zone.  The total effective blurred zone
    equals the feather width of the *coarsest active* level, which is far
    narrower than the monolithic post-sharpen blend approach.

    With the default ``edge_feather_factor=2.0``:
        Level 0 (~2 px kernel):  feather = 2 px
        Level 1 (~4 px kernel):  feather = 4 px
        Level 2 (~8 px kernel):  feather = 8 px

    Args:
        image:               Float array in [0, 1], 2-D or 3-D.
        cx, cy:              Disk centre in pixels.
        radius:              Disk radius in pixels.
        levels:              Number of decomposition levels.
        amounts:             Per-level WaveSharp amounts (0–200).
        weights:             Raw per-level gain (overrides amounts).
        power:               WaveSharp power-function exponent.
        sharpen_filter:      Soft-threshold noise-gate coefficient.
        edge_feather_factor: Feather width multiplier. ``feather_L = 2^L × factor``.
                             Increase to suppress more aggressive ringing;
                             decrease to sharpen closer to the disk edge.

    Returns:
        Float32 array in [0, 1], same shape as input.
    """
    if weights is not None:
        if len(weights) != levels:
            raise ValueError(f"len(weights)={len(weights)} must equal levels={levels}")
        gains = list(weights)
    else:
        if amounts is None:
            amounts = [200.0, 200.0, 100.0, 0.0, 0.0, 0.0]
        if len(amounts) != levels:
            raise ValueError(f"len(amounts)={len(amounts)} must equal levels={levels}")
        gains = amounts_to_weights(amounts, power=power)

    # Multi-channel: sharpen each channel with the same disk geometry
    if image.ndim == 3:
        channels = [
            sharpen_disk_aware(
                image[:, :, c], cx, cy, radius,
                levels=levels, weights=gains,
                sharpen_filter=sharpen_filter,
                edge_feather_factor=edge_feather_factor,
            )
            for c in range(image.shape[2])
        ]
        return np.stack(channels, axis=2).astype(np.float32)

    h, w = image.shape
    coeffs = decompose(image.astype(np.float64), levels)

    # Reconstruct original (residual + all details)
    original = coeffs[-1].copy()
    for d in coeffs[:-1]:
        original = original + d

    result = original.copy()
    for level_idx, (detail, gain) in enumerate(zip(coeffs[:-1], gains)):
        if gain == 0.0:
            continue
        thr = sharpen_filter * _noise_sigma(detail) if sharpen_filter > 0.0 else 0.0
        d_thr = _soft_threshold(detail, thr)

        # Per-level spatial weight: finer levels have narrower fade zones
        feather_L = max((2 ** level_idx) * edge_feather_factor, 1.0)
        weight_map = _make_disk_weight(h, w, cx, cy, radius, feather_L)

        result = result + d_thr * gain * weight_map

    return np.clip(result, 0.0, 1.0).astype(np.float32)


def sharpen_color_disk_aware(
    image: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    levels: int = 6,
    amounts: Optional[List[float]] = None,
    weights: Optional[List[float]] = None,
    power: float = 1.0,
    sharpen_filter: float = 0.0,
    edge_feather_factor: float = 2.0,
) -> np.ndarray:
    """Disk-aware sharpening for colour (H, W, 3) RGB float images via Lab L-channel.

    Converts RGB → Lab, applies :func:`sharpen_disk_aware` to the L channel
    only, then converts back.  Chrominance is preserved unchanged.

    Args and returns: same as :func:`sharpen_color` plus disk geometry args.
    """
    import cv2 as _cv2
    bgr = _cv2.cvtColor(image.astype(np.float32), _cv2.COLOR_RGB2BGR)
    lab = _cv2.cvtColor(bgr, _cv2.COLOR_BGR2Lab)

    L = lab[:, :, 0] / 100.0
    L_sharp = sharpen_disk_aware(
        L, cx, cy, radius,
        levels=levels, amounts=amounts, weights=weights,
        power=power, sharpen_filter=sharpen_filter,
        edge_feather_factor=edge_feather_factor,
    )
    lab[:, :, 0] = np.clip(L_sharp * 100.0, 0.0, 100.0)

    bgr_sharp = _cv2.cvtColor(lab, _cv2.COLOR_Lab2BGR)
    rgb_sharp = _cv2.cvtColor(bgr_sharp, _cv2.COLOR_BGR2RGB)
    return np.clip(rgb_sharp, 0.0, 1.0).astype(np.float32)


def sharpen(
    image: np.ndarray,
    levels: int = 6,
    amounts: Optional[List[float]] = None,
    weights: Optional[List[float]] = None,
    power: float = 1.0,
    sharpen_filter: float = 0.0,
) -> np.ndarray:
    """Apply à trous wavelet sharpening to *image*.

    Accepts either WaveSharp-compatible *amounts* (preferred) or raw *weights*.
    Handles both 2-D (grayscale) and 3-D (multi-channel) inputs.

    Args:
        image:          Float array in [0, 1] (normalised 16-bit input).
        levels:         Number of decomposition levels (default 6).
        amounts:        Per-level WaveSharp amounts, 0–200 scale.
                        Default: [200, 200, 100, 0, 0, 0] (layers 1–3 active).
        weights:        Raw per-level extra-gain (overrides *amounts* if given).
        power:          WaveSharp 'power function' exponent (1.0 = linear).
        sharpen_filter: WaveSharp 'sharpen filter' — soft-threshold factor
                        relative to each level's std.  0.0 = no threshold.

    Returns:
        Float32 array in [0, 1], **same histogram shape** as *image*
        (mean-preserving; no auto-stretch applied).
    """
    # Resolve weights
    if weights is not None:
        if len(weights) != levels:
            raise ValueError(f"len(weights)={len(weights)} must equal levels={levels}")
        gains = list(weights)
    else:
        if amounts is None:
            amounts = [200.0, 200.0, 100.0, 0.0, 0.0, 0.0]
        if len(amounts) != levels:
            raise ValueError(f"len(amounts)={len(amounts)} must equal levels={levels}")
        gains = amounts_to_weights(amounts, power=power)

    # Multi-channel: sharpen each channel independently
    if image.ndim == 3:
        channels = [
            sharpen(image[:, :, c], levels=levels, weights=gains,
                    sharpen_filter=sharpen_filter)
            for c in range(image.shape[2])
        ]
        return np.stack(channels, axis=2).astype(np.float32)

    coeffs = decompose(image.astype(np.float64), levels)
    result = reconstruct(coeffs, gains, sharpen_filter=sharpen_filter)
    return np.clip(result, 0.0, 1.0).astype(np.float32)
