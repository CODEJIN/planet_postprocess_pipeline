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
  - Per-layer denoise: optional per-layer soft-threshold of each detail
    coefficient before amplification (WaveSharp-compatible, same unit as
    sharpen_filter; amount=0.1 removes coefficients < 10% of noise sigma).
  - Filter types: 'gaussian' (B3-spline à trous, default), 'zerogauss'
    (LoG-based detail extraction, more aggressive), 'bilateral'
    (edge-preserving à trous, reduces limb artifacts).

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


# ── Constants ──────────────────────────────────────────────────────────────────

# Valid filter_type values for decompose() and sharpen*() functions.
FILTER_TYPES = ('gaussian', 'zerogauss', 'bilateral')

# Denoise amount is a soft-threshold coefficient multiplied by MAD(detail).
# Same unit as sharpen_filter: amount=0.1 → threshold = 0.1 × noise_sigma.
# Typical range 0.0–2.0; WaveSharp default ≈ 0.1.
_DENOISE_MAX_COEFF = 3.0   # UI hard ceiling


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


def _noise_sigma(w: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Estimate noise standard deviation from wavelet detail using MAD.

    sigma = MAD(w[mask]) / 0.6745

    Args:
        w:    Wavelet detail coefficient array.
        mask: Boolean array (same shape as w, or flattened).  When provided,
              MAD is computed over mask=True pixels only — typically the planet
              disk region.  This avoids background-dominated estimates: when
              most pixels are dark sky the full-array median collapses to ≈ 0,
              making every threshold effectively zero regardless of amount.
              When None, falls back to all non-zero pixels.
    """
    flat = np.abs(w.ravel())
    if mask is not None:
        m = mask.ravel() if mask.ndim > 1 else mask
        signal = flat[m]
    else:
        signal = flat[flat > 0.0]
    if signal.size < 10:
        return float(np.median(flat) / 0.6745)
    return float(np.median(signal) / 0.6745)


def _log_detail(image: np.ndarray, level: int) -> np.ndarray:
    """Compute LoG (Laplacian of Gaussian) detail at à trous scale 2^level.

    Implements the ZeroGauss filter type.  Uses a separable approximation:
        LoG(x,y,σ) ≈ D2G(x)·G(y) + G(x)·D2G(y)
    where D2G is the second derivative of the Gaussian (zero-sum kernel).

    The result is zero-mean (zero-sum kernel), so sharpening with LoG details
    is mean-preserving.  LoG is more aggressive than DoG at the same scale —
    it sharpens edges without the broad halo produced by USM.

    Args:
        image: 2-D float64 array.
        level: À trous level (kernel sigma ≈ 2^level × 0.75).

    Returns:
        LoG-filtered detail, same shape as image.
    """
    sigma = float(1 << level) * 0.75
    size = max(5, int(6.0 * sigma + 1) | 1)
    half = size // 2
    x = np.arange(size, dtype=np.float64) - half

    g = np.exp(-x ** 2 / (2.0 * sigma ** 2))
    g /= g.sum()

    # D2G: (x²/σ⁴ - 1/σ²) × Gaussian — zero-sum by construction
    d2g = (x ** 2 / sigma ** 4 - 1.0 / sigma ** 2) * np.exp(-x ** 2 / (2.0 * sigma ** 2))
    # Normalise amplitude to same scale as B3-based detail
    d2g_norm = np.sqrt(np.sum(d2g ** 2))
    if d2g_norm > 1e-12:
        d2g /= d2g_norm

    # LoG ≈ D2G(x)·G(y) + G(x)·D2G(y)
    part_x = _convolve1d_reflect(image, d2g, axis=1)
    part_x = _convolve1d_reflect(part_x, g,   axis=0)

    part_y = _convolve1d_reflect(image, g,   axis=1)
    part_y = _convolve1d_reflect(part_y, d2g, axis=0)

    return part_x + part_y


def _bilateral_smooth(image: np.ndarray, level: int, sigma_color: float = 0.08) -> np.ndarray:
    """Bilateral filter as the à trous smooth step (ZeroGauss/Bilateral type).

    Replaces the B3-spline convolution with an edge-preserving bilateral filter
    at the given à trous level.  The resulting detail (image - bilateral_smooth)
    preserves edges without amplifying them, eliminating limb-overshoot artifacts.

    Args:
        image:       2-D float64 array in [0, 1].
        level:       À trous level; sigmaSpace ≈ 2^level.
        sigma_color: Bilateral color-space sigma (0.08 ≈ fine edge preservation).

    Returns:
        Smoothed array (same shape as image, float64).
    """
    import cv2 as _cv2
    sigma_space = float(1 << level) * 0.75
    # cv2.bilateralFilter requires float32; d=-1 lets sigmaSpace determine diameter.
    smoothed = _cv2.bilateralFilter(
        image.astype(np.float32), d=-1,
        sigmaColor=sigma_color, sigmaSpace=sigma_space,
    )
    return smoothed.astype(np.float64)


def _denoise_coeff(
    detail: np.ndarray,
    amount: float,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Per-layer soft-threshold denoise of a wavelet detail coefficient.

    Implements WaveSharp-compatible per-layer denoise: applies a MAD-based
    soft threshold scaled by *amount*, identical in mechanism to the global
    ``sharpen_filter`` but applied independently per layer before gain
    multiplication.

        threshold = amount × MAD(detail[mask]) / 0.6745
        output    = sign(detail) × max(|detail| − threshold, 0)

    Args:
        detail: 2-D wavelet detail coefficient array.
        amount: Soft-threshold coefficient (WaveSharp-compatible scale).
                0.0 = off; 0.1 = WaveSharp gentle default; 1.0 = strong.
        mask:   Boolean array indicating planet-disk pixels for noise
                estimation.  Pass the disk mask from sharpen_disk_aware, or
                a brightness-derived mask from reconstruct().  When None,
                falls back to non-zero pixels.

    Returns:
        Thresholded detail (same shape, float64).  Identical to input if
        amount ≤ 0.
    """
    if amount <= 0.0:
        return detail
    threshold = float(amount) * _noise_sigma(detail, mask=mask)
    return _soft_threshold(detail, threshold)


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

def decompose(
    image: np.ndarray,
    levels: int = 6,
    filter_type: str = 'gaussian',
) -> List[np.ndarray]:
    """Decompose *image* into à trous wavelet coefficients.

    Args:
        image:       2-D float array (any range; float64 precision internally).
        levels:      Number of detail layers to extract.
        filter_type: Decomposition kernel.
            'gaussian'  — B3-spline à trous (default, WaveSharp-compatible).
            'zerogauss' — LoG-based detail extracted directly from the original
                          image at each scale (more aggressive, zero-sum).
            'bilateral' — Edge-preserving à trous (bilateral smooth step);
                          reduces limb overshoot at planet boundaries.

    Returns:
        List of length ``levels + 1``:
        ``[detail_0, detail_1, ..., detail_{levels-1}, residual]``

        detail_i  = contribution at spatial scale ~2^i … 2^(i+1) pixels.
        residual  = low-frequency approximation (summing all with original
                    reconstructs the original exactly for all filter types).
    """
    coeffs: List[np.ndarray] = []
    current = image.astype(np.float64)

    if filter_type == 'gaussian':
        for i in range(levels):
            smoothed = _smooth(current, i)
            coeffs.append(current - smoothed)
            current = smoothed
        coeffs.append(current)

    elif filter_type == 'zerogauss':
        # LoG details are extracted from the *original* image (not cascaded),
        # so each scale is independent.  Residual = original − Σ details
        # guarantees exact reconstruction: residual + Σdetail_i = original.
        orig = current.copy()
        for i in range(levels):
            coeffs.append(_log_detail(orig, i))
        residual = orig.copy()
        for d in coeffs:
            residual = residual - d
        coeffs.append(residual)

    elif filter_type == 'bilateral':
        for i in range(levels):
            smoothed = _bilateral_smooth(current, i)
            coeffs.append(current - smoothed)
            current = smoothed
        coeffs.append(current)

    else:
        raise ValueError(f"filter_type must be one of {FILTER_TYPES}, got {filter_type!r}")

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
    denoise_amounts: Optional[List[float]] = None,
) -> np.ndarray:
    """Reconstruct a sharpened image from wavelet coefficients.

    Processing order per layer:
        detail_i → denoise (soft-threshold) → sharpen_filter (soft-threshold) → × gain → add

    Args:
        coeffs:          Output of :func:`decompose`.
        weights:         Per-level extra-gain (length == levels).
        sharpen_filter:  Global soft-threshold coefficient (WaveSharp 'sharpen
                         filter'): thr_i = sharpen_filter × MAD(detail_i).
                         0.0 = no thresholding.
        denoise_amounts: Per-level soft-threshold coefficient (WaveSharp scale).
                         0.0 = off; 0.1 = WaveSharp gentle default; 1.0 = strong.
                         Applied before the global sharpen_filter threshold.
                         Length must equal ``len(weights)`` or be None.

    Returns:
        Float64 array (same shape as input, **not yet clipped**).
    """
    original = coeffs[-1].copy()
    for d in coeffs[:-1]:
        original = original + d

    # Build a content mask for noise estimation: top 50% of pixels by brightness.
    # This approximates the planet disk without needing explicit disk geometry.
    # For images where most pixels are dark sky, the full-array MAD collapses
    # to near zero; using the brighter half keeps the estimator in the planet
    # region regardless of the exact background level.
    needs_mask = sharpen_filter > 0.0 or (
        denoise_amounts and any(x > 0.0 for x in denoise_amounts)
    )
    if needs_mask:
        orig_flat = original.ravel()
        p50 = float(np.percentile(orig_flat, 50))
        content_mask = orig_flat > max(p50, 1e-6)
        if content_mask.sum() < 10:
            content_mask = None
    else:
        content_mask = None

    result = original.copy()
    for i, (detail, w) in enumerate(zip(coeffs[:-1], weights)):
        if w == 0.0:
            continue

        # Per-layer denoise: MAD-based soft-threshold (WaveSharp-compatible)
        dn = denoise_amounts[i] if (denoise_amounts and i < len(denoise_amounts)) else 0.0
        d_proc = _denoise_coeff(detail, dn, mask=content_mask)

        # Global soft threshold (noise gate)
        thr = sharpen_filter * _noise_sigma(d_proc, mask=content_mask) if sharpen_filter > 0.0 else 0.0
        d_thr = _soft_threshold(d_proc, thr)

        result = result + d_thr * w

    return result


def sharpen_color(
    image: np.ndarray,
    levels: int = 6,
    amounts: Optional[List[float]] = None,
    weights: Optional[List[float]] = None,
    power: float = 1.0,
    sharpen_filter: float = 0.0,
    denoise_amounts: Optional[List[float]] = None,
    filter_type: str = 'gaussian',
) -> np.ndarray:
    """Sharpen a colour (H, W, 3) RGB float [0, 1] image via L-channel sharpening.

    Converts RGB → Lab, sharpens only the L (luminance) channel using à trous
    wavelet sharpening, then converts back to RGB.  Chrominance (a, b) is
    preserved unchanged, so colour balance is unaffected.

    Args:
        image:           Float32 (H, W, 3) RGB array in [0, 1].
        levels:          Number of wavelet decomposition levels.
        amounts:         Per-level WaveSharp amounts (0–200).
        weights:         Raw per-level gain (overrides amounts if given).
        power:           WaveSharp power-function exponent.
        sharpen_filter:  Soft-threshold noise-gate coefficient.
        denoise_amounts: Per-level soft-threshold coefficient (0.0=off, 0.1=gentle, 1.0=strong).
        filter_type:     Decomposition kernel ('gaussian', 'zerogauss',
                         'bilateral').

    Returns:
        Float32 (H, W, 3) RGB array in [0, 1], with sharpened luminance.
    """
    import cv2 as _cv2
    bgr = _cv2.cvtColor(image.astype(np.float32), _cv2.COLOR_RGB2BGR)
    lab = _cv2.cvtColor(bgr, _cv2.COLOR_BGR2Lab)

    L = lab[:, :, 0] / 100.0
    L_sharp = sharpen(L, levels=levels, amounts=amounts, weights=weights,
                      power=power, sharpen_filter=sharpen_filter,
                      denoise_amounts=denoise_amounts, filter_type=filter_type)
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


def _make_disk_weight_ellipse(
    h: int, w: int,
    cx: float, cy: float,
    rx: float, ry: float,
    angle_rad: float,
    feather_px: float,
) -> np.ndarray:
    """Soft elliptical mask: 1.0 inside disk, linear fade to 0 over feather_px at ellipse boundary.

    Uses the actual ellipse shape (rx=semi-major, ry=semi-minor, angle_rad=tilt)
    so that the feather zone follows Jupiter's oblate limb in every direction.
    The normalised elliptical distance (1.0 at boundary) is scaled by the
    geometric mean radius sqrt(rx*ry) to convert to pixels, preserving the same
    feather depth as the circular version while adapting to the ellipse shape.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    # Rotate to ellipse principal axes (semi-major along angle_rad from x-axis)
    cos_a = float(np.cos(angle_rad))
    sin_a = float(np.sin(angle_rad))
    dx_r =  cos_a * dx + sin_a * dy
    dy_r = -sin_a * dx + cos_a * dy
    # Normalised elliptical distance: exactly 1.0 at the ellipse boundary
    d_norm = np.sqrt((dx_r / rx) ** 2 + (dy_r / ry) ** 2)
    # Convert to approximate pixel distance from boundary.
    # (1 - d_norm) is dimensionless; scaling by sqrt(rx*ry) gives pixel units
    # consistent with the circular version when rx == ry.
    dist_from_boundary = (1.0 - d_norm) * float(np.sqrt(rx * ry))
    t = np.clip(dist_from_boundary / max(feather_px, 1.0), 0.0, 1.0)
    # Cosine S-curve: smoother at both endpoints than linear fade,
    # making the disk-edge transition less perceptible.
    return (0.5 * (1.0 - np.cos(np.pi * t))).astype(np.float32)


def _fill_outside_ellipse(
    image: np.ndarray,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    angle_rad: float,
) -> np.ndarray:
    """Fill pixels outside the ellipse with the nearest ellipse-boundary pixel value.

    Before applying the à trous wavelet, background pixels (near-zero) outside
    the disk are read by the B3 kernel and artificially inflate detail
    coefficients near the limb — creating a bright ring after sharpening.
    Replacing the outside region with a smooth extension (nearest limb pixel
    along each radial direction) makes the wavelet see a natural signal at the
    boundary, eliminating this artifact.

    Args:
        image:     2-D float array (single channel).
        cx, cy:    Disk centre in pixels.
        rx, ry:    Semi-major and semi-minor axes of the fill boundary.
        angle_rad: Ellipse tilt in radians (semi-major axis from x-axis).

    Returns:
        Copy of *image* with pixels outside the ellipse replaced by the value
        of their radially-projected nearest boundary pixel.
    """
    h, w = image.shape[0], image.shape[1]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    cos_a = float(np.cos(angle_rad))
    sin_a = float(np.sin(angle_rad))
    dx_r =  cos_a * dx + sin_a * dy
    dy_r = -sin_a * dx + cos_a * dy
    d_norm = np.sqrt((dx_r / rx) ** 2 + (dy_r / ry) ** 2)

    outside = d_norm > 1.0
    if not np.any(outside):
        return image

    # Project each outside pixel to the ellipse boundary along the radial
    # direction from the centre.  Dividing (dx_r, dy_r) by d_norm gives a
    # point (px_r, py_r) satisfying (px_r/rx)^2 + (py_r/ry)^2 = 1.
    d_safe = np.where(d_norm > 1e-6, d_norm, 1e-6)
    px_r = dx_r / d_safe          # projected, rotated frame
    py_r = dy_r / d_safe
    # Rotate back to image frame
    px = cos_a * px_r - sin_a * py_r   # dx from centre
    py = sin_a * px_r + cos_a * py_r   # dy from centre
    xi = np.clip(np.round(cx + px).astype(int), 0, w - 1)
    yi = np.clip(np.round(cy + py).astype(int), 0, h - 1)

    filled = image.copy()
    filled[outside] = image[yi[outside], xi[outside]]
    return filled


def auto_wavelet_params(
    image: np.ndarray,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    angle_rad: float,
    n_angles: int = 36,
    profile_ext_px: int = 10,
    profile_int_px: int = 25,
    visual_limb_frac: float = 0.05,
    grad_threshold_frac: float = 0.25,
) -> tuple:
    """Auto-estimate edge_feather_factor and disk_expand_px from a de-rotation stack.

    Measures two properties directly from the disk boundary in the stacked image:

    1. **expand_px**: Scaled proportionally to the geometric mean disk radius
       (``sqrt(rx × ry) × 0.0505``, calibrated from rx=102,ry=96→5.0 px).
       This corrects for find_disk_center's Otsu-threshold landing inside the
       true visual limb, which scales with planet size.  Visual-limb brightness
       measurement was found to over-estimate this offset (gives 2× the optimal
       value due to the coverage-gradient cross-scale consistency constraint).

    2. **eff (edge_feather_factor)**: Width of the coverage-gradient transition
       just inside the detected boundary.  The à trous wavelet amplifies any
       brightness step at this boundary; the feather zone must span at least
       half this width to suppress the artifact.  Measured as the radial
       width (in pixels) where |d_brightness/d_r| exceeds
       ``grad_threshold_frac × max_gradient``, then eff = gradient_width / 2.
       Falls back to geometric-mean-radius scaling
       (``sqrt(rx × ry) × 0.030``, calibrated from rx=102→eff=3.0) if the
       gradient measurement is unreliable.

    This function is designed to be called per-image (per filter per window)
    so that parameters adapt to filter-specific limb darkening (IR > R > G > B),
    seeing, and de-rotation coverage conditions.

    Args:
        image:               2-D or 3-D float array (values need not be normalised;
                             normalised internally).
        cx, cy:              Disk centre in pixels.
        rx, ry:              Semi-major / semi-minor ellipse radii (from
                             ``find_disk_center``).
        angle_rad:           Ellipse tilt in radians (semi-major from x-axis).
        n_angles:            Number of equally-spaced radial directions to sample
                             (default 36 = every 10°).
        profile_ext_px:      Pixels to sample outward past the detected boundary
                             (used only for gradient analysis, not expand_px).
        profile_int_px:      Pixels to sample inward from the detected boundary.
        visual_limb_frac:    Unused (kept for API compatibility).
        grad_threshold_frac: Fraction of the maximum radial |gradient| used to
                             determine the coverage-gradient zone width.

    Returns:
        ``(eff, expand_px)`` — both rounded to 1 decimal place.
    """
    h, w = image.shape[:2]
    lum = (image.mean(axis=2).astype(np.float64)
           if image.ndim == 3 else image.astype(np.float64))

    # Normalise to [0, 1] so thresholds are image-type agnostic
    lum_max = lum.max()
    if lum_max < 1e-8:
        return 3.0, 5.0
    lum = lum / lum_max

    # Background: median of image corners (assumed to be sky, far from planet)
    cs = max(10, int(min(h, w) * 0.05))
    bg = float(np.median(np.concatenate([
        lum[:cs, :cs].ravel(), lum[:cs, -cs:].ravel(),
        lum[-cs:, :cs].ravel(), lum[-cs:, -cs:].ravel(),
    ])))

    # Disk interior: median inside 50 % of the minor radius
    r_inner = int(0.5 * min(rx, ry))
    ya = max(0, int(cy) - r_inner); yb = min(h, int(cy) + r_inner + 1)
    xa = max(0, int(cx) - r_inner); xb = min(w, int(cx) + r_inner + 1)
    disk_val = float(np.median(lum[ya:yb, xa:xb]))

    if disk_val - bg < 0.02:
        # Degenerate image; return calibration-derived defaults
        return 3.0, 5.0

    cos_a = float(np.cos(angle_rad))
    sin_a = float(np.sin(angle_rad))

    gradient_widths: List[float] = []

    for theta in np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False):
        dx = float(np.cos(theta))
        dy = float(np.sin(theta))

        # Pixel distance to ellipse boundary along this image-space direction.
        # Rotate direction to ellipse principal-axis frame, then invert ellipse eq.
        dx_r =  cos_a * dx + sin_a * dy
        dy_r = -sin_a * dx + cos_a * dy
        denom = np.sqrt((dx_r / rx) ** 2 + (dy_r / ry) ** 2)
        if denom < 1e-8:
            continue
        r_ell = 1.0 / denom   # ellipse boundary radius in this direction

        # Build radial sample positions (inward side only for gradient analysis)
        r_start = max(1.0, r_ell - profile_int_px)
        rs = np.arange(r_start, r_ell + 1.0)
        xs = np.clip(np.round(cx + rs * dx).astype(int), 0, w - 1)
        ys = np.clip(np.round(cy + rs * dy).astype(int), 0, h - 1)
        profile = lum[ys, xs]

        # --- eff: gradient width on the inward side of the detected boundary ---
        # The coverage gradient lives inside the Otsu-detected rx, where fewer
        # de-rotation frames overlap.  We restrict the gradient analysis to the
        # interior portion so limb-darkening outside the boundary doesn't inflate
        # the estimate.
        if len(profile) < 5:
            continue
        deriv = np.abs(np.gradient(profile))
        max_d = deriv.max()
        if max_d < 1e-6:
            continue
        in_grad = deriv >= grad_threshold_frac * max_d
        gradient_widths.append(float(np.sum(in_grad)))

    # expand_px: proportional to geometric mean disk radius.
    # Calibrated from rx=102, ry=96 → optimal expand_px=5.0:
    #   sqrt(102*96) * 0.0505 ≈ 5.0
    # Visual-limb brightness measurement overestimates by ~2× due to the
    # cross-scale feather consistency constraint (Level-0 feather must stay
    # inside the Otsu boundary for all active wavelet scales to agree).
    expand_px = round(float(np.sqrt(rx * ry) * 0.0505), 1)

    # eff: measured gradient_width / 2 (Level-1 feather = 2*eff covers gradient)
    # Fallback: geometric-mean-radius scaling calibrated to our data (rx=102→eff=3.0)
    eff_fallback = float(np.sqrt(rx * ry) * 0.0303)
    if gradient_widths:
        grad_w = float(np.median(gradient_widths))
        eff_measured = grad_w / 2.0
        eff = eff_measured if 1.0 <= eff_measured <= 8.0 else eff_fallback
    else:
        eff = eff_fallback

    return round(float(max(1.0, eff)), 1), round(float(max(0.0, expand_px)), 1)


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
    ry: Optional[float] = None,
    angle: float = 0.0,
    expand_px: float = 0.0,
    denoise_amounts: Optional[List[float]] = None,
    filter_type: str = 'gaussian',
) -> np.ndarray:
    """À trous wavelet sharpening with per-level spatial edge feathering.

    Each detail level L contributes:

        detail_L → denoise → soft_threshold → × gain_L × spatial_weight_L

    where ``spatial_weight_L`` fades from 1.0 (disk interior) to 0.0 at the
    disk edge over a zone of width ``feather_L = 2^L × edge_feather_factor``
    pixels.

    When ``ry`` is provided, the feather zone follows the actual elliptical
    disk boundary (rx=radius semi-major, ry semi-minor, angle tilt in radians)
    rather than a circle.

    Args:
        image:               Float array in [0, 1], 2-D or 3-D.
        cx, cy:              Disk centre in pixels.
        radius:              Semi-major axis radius in pixels.
        levels:              Number of decomposition levels.
        amounts:             Per-level WaveSharp amounts (0–200).
        weights:             Raw per-level gain (overrides amounts).
        power:               WaveSharp power-function exponent.
        sharpen_filter:      Soft-threshold noise-gate coefficient.
        edge_feather_factor: Feather width multiplier.
        ry:                  Semi-minor axis radius (pixels). None = circular.
        angle:               Ellipse tilt angle in radians.
        expand_px:           Extra pixels to expand the mask boundary outward.
        denoise_amounts:     Per-level soft-threshold coefficient (0.0=off, 0.1=gentle, 1.0=strong).
        filter_type:         'gaussian', 'zerogauss', or 'bilateral'.

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

    use_ellipse = ry is not None and ry > 0.0

    if image.ndim == 3:
        channels = [
            sharpen_disk_aware(
                image[:, :, c], cx, cy, radius,
                levels=levels, weights=gains,
                sharpen_filter=sharpen_filter,
                edge_feather_factor=edge_feather_factor,
                ry=ry, angle=angle,
                expand_px=expand_px,
                denoise_amounts=denoise_amounts,
                filter_type=filter_type,
            )
            for c in range(image.shape[2])
        ]
        return np.stack(channels, axis=2).astype(np.float32)

    h, w = image.shape
    rx_m = radius + expand_px
    ry_m = (ry + expand_px) if use_ellipse else None

    coeffs = decompose(image.astype(np.float64), levels, filter_type=filter_type)

    original = coeffs[-1].copy()
    for d in coeffs[:-1]:
        original = original + d

    # Build a binary disk mask for noise estimation.  Using the actual disk
    # geometry gives a precise planet-region MAD without relying on brightness
    # thresholds or exact-zero assumptions about the background.
    Y_g, X_g = np.mgrid[:h, :w]
    if use_ellipse:
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        dx = X_g - cx
        dy = Y_g - cy
        rx_e = rx_m if rx_m > 0 else 1.0
        ry_e = ry_m if ry_m > 0 else rx_e
        disk_mask = (
            ((dx * cos_a + dy * sin_a) / rx_e) ** 2
            + ((dx * sin_a - dy * cos_a) / ry_e) ** 2
        ) <= 1.0
    else:
        disk_mask = (X_g - cx) ** 2 + (Y_g - cy) ** 2 <= rx_m ** 2
    disk_flat = disk_mask.ravel()
    if disk_flat.sum() < 10:
        disk_flat = None

    result = original.copy()
    for level_idx, (detail, gain) in enumerate(zip(coeffs[:-1], gains)):
        if gain == 0.0:
            continue

        dn = denoise_amounts[level_idx] if (denoise_amounts and level_idx < len(denoise_amounts)) else 0.0
        d_proc = _denoise_coeff(detail, dn, mask=disk_flat)

        thr = sharpen_filter * _noise_sigma(d_proc, mask=disk_flat) if sharpen_filter > 0.0 else 0.0
        d_thr = _soft_threshold(d_proc, thr)

        feather_L = max((2 ** level_idx) * edge_feather_factor, 1.0)
        if use_ellipse:
            weight_map = _make_disk_weight_ellipse(
                h, w, cx, cy, rx_m, ry_m, angle, feather_L
            )
        else:
            weight_map = _make_disk_weight(h, w, cx, cy, rx_m, feather_L)

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
    ry: Optional[float] = None,
    angle: float = 0.0,
    expand_px: float = 0.0,
    denoise_amounts: Optional[List[float]] = None,
    filter_type: str = 'gaussian',
) -> np.ndarray:
    """Disk-aware sharpening for colour (H, W, 3) RGB float images via Lab L-channel.

    Converts RGB → Lab, applies :func:`sharpen_disk_aware` to the L channel
    only, then converts back.  Chrominance is preserved unchanged.

    Args and returns: same as :func:`sharpen_color` plus disk geometry args
    and the new denoise_amounts / filter_type parameters.
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
        ry=ry, angle=angle,
        expand_px=expand_px,
        denoise_amounts=denoise_amounts,
        filter_type=filter_type,
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
    denoise_amounts: Optional[List[float]] = None,
    filter_type: str = 'gaussian',
) -> np.ndarray:
    """Apply à trous wavelet sharpening to *image*.

    Accepts either WaveSharp-compatible *amounts* (preferred) or raw *weights*.
    Handles both 2-D (grayscale) and 3-D (multi-channel) inputs.

    Args:
        image:           Float array in [0, 1] (normalised 16-bit input).
        levels:          Number of decomposition levels (default 6).
        amounts:         Per-level WaveSharp amounts, 0–200 scale.
                         Default: [200, 200, 100, 0, 0, 0].
        weights:         Raw per-level extra-gain (overrides *amounts*).
        power:           WaveSharp 'power function' exponent (1.0 = linear).
        sharpen_filter:  Soft-threshold factor relative to each level's MAD.
                         0.0 = no threshold.
        denoise_amounts: Per-level soft-threshold coefficient (0.0=off, 0.1=gentle, 1.0=strong).
        filter_type:     Decomposition kernel: 'gaussian' (default), 'zerogauss',
                         or 'bilateral'.

    Returns:
        Float32 array in [0, 1], mean-preserving.
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

    if image.ndim == 3:
        channels = [
            sharpen(image[:, :, c], levels=levels, weights=gains,
                    sharpen_filter=sharpen_filter,
                    denoise_amounts=denoise_amounts,
                    filter_type=filter_type)
            for c in range(image.shape[2])
        ]
        return np.stack(channels, axis=2).astype(np.float32)

    coeffs = decompose(image.astype(np.float64), levels, filter_type=filter_type)
    result = reconstruct(coeffs, gains, sharpen_filter=sharpen_filter,
                         denoise_amounts=denoise_amounts)
    return np.clip(result, 0.0, 1.0).astype(np.float32)
