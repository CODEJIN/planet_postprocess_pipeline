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

from typing import List, Optional, Tuple

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


def _bilateral_smooth(
    image: np.ndarray,
    level: int,
    sigma_color: Optional[float] = None,
    sigma_fine: float = 0.10,
    sigma_coarse: float = 0.12,
) -> np.ndarray:
    """Bilateral filter as the à trous smooth step (ZeroGauss/Bilateral type).

    Replaces the B3-spline convolution with an edge-preserving bilateral filter
    at the given à trous level.  The resulting detail (image - bilateral_smooth)
    preserves edges without amplifying them as much as a linear (gaussian)
    smooth step would -- but does NOT eliminate limb-overshoot artifacts on
    real planetary data at an acceptable cost (see the 2026-08-15 real-data
    validation in SATURN_RING_WAVELET_STATUS_2026-08-15.md: filter_type=
    'bilateral' cut Jupiter's white-rim ~42-45% and Saturn's asymmetric
    ring-limb ringing ~79%, but at a ~78% loss of real disk-interior sharpness
    -- belts / Cassini Division -- on both targets. Do not read this
    docstring's original "eliminating" claim as validated; it wasn't tested
    against real data until that investigation).

    sigma_color was a single fixed 0.08 for every level until 2026-08-15, when
    a grid search (using the real sharpen() pipeline, not a hand-reimplemented
    approximation) found a per-level-scaled schedule is a strict Pareto
    improvement at the SAME (zero) overshoot on the project's synthetic
    hard-edge test: retained detail-enhancement boost rose from 26.4% (flat
    0.08) to 28.7% of the unmasked-gaussian baseline. Fine levels (small
    spatial support, level 0) keep a slightly lower sigma_color close to the
    old default; coarser levels (level>=5) use a slightly higher one, since
    by the time the smoothing kernel's spatial support is large enough to
    span the disk's own hard edge, a marginally larger colour tolerance still
    fully preserves that edge while recovering more real texture at that
    scale. This is a within-'bilateral'-only tuning change -- filter_type=
    'gaussian' (the default for every existing caller) is completely
    unaffected either way.

    Args:
        image:        2-D float64 array in [0, 1].
        level:        À trous level; sigmaSpace ≈ 2^level.
        sigma_color:  Explicit override -- if given, used for every level
                      (bypasses sigma_fine/sigma_coarse interpolation).
                      None (default): interpolate from sigma_fine to
                      sigma_coarse across levels 0-5 (see below).
        sigma_fine:   sigma_color at level 0 (finest spatial scale).
        sigma_coarse: sigma_color at level >= 5 (coarsest active scale in
                      this project's typical 6-level decomposition).

    Returns:
        Smoothed array (same shape as image, float64).
    """
    import cv2 as _cv2
    if sigma_color is None:
        t = min(level / 5.0, 1.0)
        sigma_color = sigma_fine + (sigma_coarse - sigma_fine) * t
    sigma_space = float(1 << level) * 0.75
    # cv2.bilateralFilter requires float32; d=-1 lets sigmaSpace determine diameter.
    smoothed = _cv2.bilateralFilter(
        image.astype(np.float32), d=-1,
        sigmaColor=float(sigma_color), sigmaSpace=sigma_space,
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
                          reduces limb overshoot at planet boundaries, but
                          NOT a clean fix -- real-data validation
                          (2026-08-15, see SATURN_RING_WAVELET_STATUS_
                          2026-08-15.md) found it cuts overshoot 42-79%
                          while costing ~78% of real disk-interior sharpness
                          (belts/Cassini Division) on both Jupiter and
                          Saturn. Do not switch master_filter_type to this
                          as a default without accepting that trade-off.

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
    overshoot_clamp_radius_px: float = 0.0,
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
        overshoot_clamp_radius_px: See sharpen()'s docstring. 0.0 (default):
                         bit-identical. Passed straight through to the
                         L-channel sharpen() call (channel-agnostic).

    Returns:
        Float32 (H, W, 3) RGB array in [0, 1], with sharpened luminance.
    """
    import cv2 as _cv2
    bgr = _cv2.cvtColor(image.astype(np.float32), _cv2.COLOR_RGB2BGR)
    lab = _cv2.cvtColor(bgr, _cv2.COLOR_BGR2Lab)

    L = lab[:, :, 0] / 100.0
    L_sharp = sharpen(L, levels=levels, amounts=amounts, weights=weights,
                      power=power, sharpen_filter=sharpen_filter,
                      denoise_amounts=denoise_amounts, filter_type=filter_type,
                      overshoot_clamp_radius_px=overshoot_clamp_radius_px)
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


def _local_min_max(image: np.ndarray, radius_px: float) -> Tuple[np.ndarray, np.ndarray]:
    """Per-pixel local min/max of `image` over a circular neighborhood of
    radius `radius_px`, via grayscale morphological erosion (=local min)
    and dilation (=local max) with a flat structuring element.

    Used to clamp sharpened output against unsharp-mask overshoot/ringing:
    a pixel that ends up brighter (or darker) than every real pixel of the
    ORIGINAL, unsharpened image within this neighborhood is definitionally
    invented by the filter, not real detail. See `overshoot_clamp_radius_px`
    on `sharpen()`/`sharpen_disk_aware()` for the caller-facing rationale
    (2026-08-15 -- root-caused ringing at the disk limb after four
    mask-domain attempts, documented in project_ring_limb_ringing_bug
    memory, failed to escape the gray-halo/white-rim trade-off; this is an
    output-domain mechanism instead).

    No scipy dependency (this codebase deliberately avoids one -- see
    `_convolve1d_reflect`'s "numpy drop-in for scipy.ndimage" comment);
    `cv2`'s grayscale morphology ops give the same result on a flat
    structuring element and `cv2` is already a heavy dependency elsewhere
    in this module.
    """
    import cv2

    k = max(1, int(round(radius_px)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    img32 = image.astype(np.float32)
    return cv2.erode(img32, kernel), cv2.dilate(img32, kernel)


def coverage_to_confidence(n: np.ndarray, floor: float = 0.0) -> np.ndarray:
    """Shape a raw [0,1] per-pixel de-rotation coverage fraction (see
    derotation.compute_frame_coverage_mask / derotate_filter's coverage
    aggregation) into a monotonic [floor,1] sharpening-gain / stacking-blend
    multiplier, via a smoothstep (3n^2 - 2n^3).

    Smoothstep specifically (not a linear map) because it has zero
    derivative at n=0 AND n=1 -- the disk interior sits at n=1 exactly, so
    a linear map would leave a slope discontinuity (kink) right where
    coverage saturates, which unsharp-mask sharpening could itself
    amplify into a new, smaller-scale ringing artifact -- exactly the
    failure mode this feature exists to reduce (see the 2026-08-13 Saturn
    limb-ringing diagnosis: find_disk_center's ellipse fit has a measured
    ~0.5-0.9px asymmetric error vs the true photometric limb, and full-
    strength gain at that mismatch produces classic overshoot).

    Args:
        n: raw coverage fraction, any shape, expected in [0,1] (values
           outside are clipped).
        floor: minimum multiplier, reached at n=0. 0.0 (default) is
           appropriate for the S0/S_L stacking blend's alpha(x), where
           hitting exactly zero is required for "never worse than S0" to
           hold by construction. A floor > 0.0 is appropriate for
           sharpening gain reduction instead -- a hard zero-gain cliff
           would itself read as a new artifact (a flat, unsharpened patch
           right at the real limb), the same halo-avoidance principle
           already applied once in sharpen_disk_aware's extra_gap_px ramp.
    """
    n_c = np.clip(n, 0.0, 1.0).astype(np.float32)
    smooth = n_c * n_c * (3.0 - 2.0 * n_c)
    return (floor + (1.0 - floor) * smooth).astype(np.float32)


def _fill_outside_ellipse(
    image: np.ndarray,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    angle_rad: float,
    baseline_px: float = 3.0,
) -> np.ndarray:
    """Extend pixels outside the ellipse by continuing the LOCAL RADIAL
    GRADIENT measured just inside the boundary, rather than copying a flat
    boundary value.

    Before applying the à trous wavelet, background pixels (near-zero) outside
    the disk are read by the B3 kernel and artificially inflate detail
    coefficients near the limb — creating ringing after sharpening. Extending
    the outside region with a natural continuation of the boundary's own
    signal (rather than an abrupt value step) makes the wavelet see a natural
    signal at the boundary, eliminating this artifact.

    A first version of this (2026-08-15) copied a FLAT value (the boundary
    pixel's own brightness) outward. On real Saturn/Jupiter data this made
    things WORSE: real limb-darkening brightness keeps decreasing just
    outside the fitted boundary, so flattening it introduced a new abrupt
    first-derivative discontinuity (a kink) right where the flat region
    began — which the wavelet read as a new, more visible bright halo
    encircling the whole limb (worse than the original asymmetric ringing
    this was meant to fix). Continuing the measured LOCAL slope instead
    (clamped to non-increasing, since a measured local brightness INCREASE
    just inside the boundary is almost always noise or nearby unrelated
    structure, not a trend safe to extrapolate) avoids introducing that
    kink: the extension tapers down the same way the real signal already
    was, then floors at 0 once it would go negative.

    Args:
        image:       2-D float array (single channel).
        cx, cy:      Disk centre in pixels.
        rx, ry:      Semi-major and semi-minor axes of the fill boundary.
        angle_rad:   Ellipse tilt in radians (semi-major axis from x-axis).
        baseline_px: Pixel distance inward from the boundary used to
                     estimate the local outward slope. Small enough to stay
                     local (not confounded by unrelated features further
                     in), large enough to be somewhat robust to per-pixel
                     noise.

    Returns:
        Copy of *image* with pixels outside the ellipse replaced by a
        gradient-continuing extrapolation from their radially-projected
        boundary point.
    """
    import cv2

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
    boundary_x = (cx + px).astype(np.float32)
    boundary_y = (cy + py).astype(np.float32)

    # Outward unit vector from the boundary point through this pixel (same
    # ray as above, but measured from the boundary rather than the centre,
    # so it stays well-defined even very close to the boundary).
    seg_dx = xx - boundary_x
    seg_dy = yy - boundary_y
    t_out = np.sqrt(seg_dx ** 2 + seg_dy ** 2)          # pixel distance beyond boundary
    t_safe = np.where(t_out > 1e-6, t_out, 1.0)
    unit_x = seg_dx / t_safe
    unit_y = seg_dy / t_safe
    inward_x = boundary_x - baseline_px * unit_x
    inward_y = boundary_y - baseline_px * unit_y

    # Bilinear sampling (not nearest-integer rounding) is required here:
    # rounding to the nearest pixel made adjacent output pixels -- whose
    # continuous projected angle differs only infinitesimally -- snap to
    # different integer boundary pixels whenever real limb texture varies
    # along the ellipse, producing a fine quantisation/aliasing pattern.
    # The a trous wavelet then reads this as genuine high-frequency detail
    # and amplifies it into a visible checkerboard/moire artifact right at
    # the boundary (found on real Saturn/Jupiter data, 2026-08-15).
    img_f = image.astype(np.float32)
    val_boundary = cv2.remap(
        img_f, boundary_x, boundary_y,
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )
    val_inward = cv2.remap(
        img_f, inward_x, inward_y,
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )

    slope = np.minimum((val_boundary - val_inward) / baseline_px, 0.0)
    extrapolated = np.clip(val_boundary + slope * t_out, 0.0, val_boundary)

    filled = image.copy()
    filled[outside] = extrapolated[outside].astype(image.dtype)
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
    extra_rx: Optional[float] = None,
    extra_ry: Optional[float] = None,
    extra_angle: Optional[float] = None,
    extra_gap_px: Optional[float] = None,
    confidence_map: Optional[np.ndarray] = None,
    fill_outside_before_sharpen: bool = False,
    overshoot_clamp_radius_px: float = 0.0,
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

    When ``extra_rx`` is also provided, a SECOND co-centred ellipse is
    unioned into the mask — for a target with a coplanar structure that
    extends well past the primary disk (e.g. Saturn's rings), this lets that
    structure receive sharpening gain too, without which it sits at gain=0
    and is passed through completely unsharpened. This function has no
    notion of what the second shape physically represents; the caller
    supplies its geometry.

    IMPORTANT when the second shape sits right outside the primary disk (as
    a ring does): omitting ``extra_gap_px`` makes the union a solid ellipse
    reaching all the way to the centre, which backfills gain=1 immediately
    outside the primary disk's own feather zone -- defeating that feather's
    purpose of giving the disk's real limb gradient a "quiet" unboosted
    buffer before any further content is sharpened, and reintroducing a
    ringing artifact right at the limb (confirmed 2026-08-13 on real Saturn
    data). ``extra_gap_px`` fixes this by ramping the second shape's own
    weight smoothly from 0 (right at the primary disk's true boundary,
    measured by actual pixel distance, not a second ellipse) up to full
    strength over that many pixels -- a single continuous ramp, not a flat
    exclusion zone. Two earlier attempts at this were tried and rejected on
    real data (2026-08-13): (a) a separate, differently-eccentric "inner
    ellipse" (e.g. the ring system's own physical inner edge, a very flat
    ellipse vs. the near-circular globe) to carve a hole -- but that
    ellipse's boundary crosses the primary disk's boundary at some off-axis
    angle (any two co-centred ellipses of different eccentricity do), and
    right at that crossing the hole no longer reached the disk's actual
    edge, leaving a wedge of still-bright, still-unboosted pixels amid
    heavily sharpened neighbours (a visible dark cusp, confirmed by directly
    visualising the mask); (b) a HARD pixel-distance exclusion band (weight
    forced to exactly 0 for extra_gap_px pixels, then a separate fade-in
    beyond it) -- geometrically consistent (no crossing), but in real
    multi-frame Saturn stacks that "gap" isn't empty, it's the disk's own
    fairly bright limb-darkening tail, so a flat zero band there read as a
    visibly distinct unsharpened "halo" between two sharpened regions. The
    single continuous ramp has neither problem: no crossing (isotropic
    pixel distance from the disk's own true shape) and no flat segment for
    the eye to pick out (every pixel gets some, gradually increasing, gain).

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
        extra_rx:            Semi-major axis of an optional second, co-centred
                              ellipse to also include in the mask (union, via
                              per-level max). None = no second ellipse (default,
                              every existing caller's behaviour unchanged).
        extra_ry:            Semi-minor axis of the second ellipse. None/<=0
                              falls back to a circle of radius extra_rx.
        extra_angle:         Tilt angle (radians) of the second ellipse. None
                              falls back to reusing `angle`.
        extra_gap_px:        Width (pixels) of the continuous ramp the second
                              shape's own weight rises over, starting at 0
                              right at the primary disk's true boundary (see
                              above). None/<=0 = no ramp (solid union,
                              reaching full strength immediately at the
                              disk's own edge).
        confidence_map:      Optional (H, W) float array in [0, 1], same
                              spatial shape as `image`, multiplied into every
                              level's gain (see coverage_to_confidence()).
                              None (default) is bit-identical to every
                              existing caller -- multiplying by the scalar
                              1.0 rather than allocating a full-size array.
                              Intended for derotate_filter's per-pixel
                              de-rotation coverage signal: reduces
                              sharpening gain where the underlying multi-
                              frame stack is less reliable (near the limb,
                              where find_disk_center's ellipse fit has a
                              measured sub-pixel asymmetric error vs the
                              true photometric limb -- see the 2026-08-13
                              Saturn limb-ringing diagnosis), rather than
                              applying full-strength gain uniformly inside
                              the disk-feather mask regardless of coverage.
        fill_outside_before_sharpen: When True, replaces pixels outside the
                              primary ellipse (rx_m, ry_m) with their
                              radially-projected nearest-boundary value
                              (via _fill_outside_ellipse) before computing
                              the DETAIL coefficients used for the gain
                              correction -- but never for `original`, the
                              base the output is built from, which always
                              comes from the real unmodified image. This
                              stops the à trous kernel from reading
                              unrelated background/ring content and
                              inflating detail right at the limb (classic
                              Gibbs-ringing precursor), independent of and
                              complementary to confidence_map above (that
                              scales gain down near the limb; this removes
                              the intensity step the filter sees there in
                              the first place). Only the primary ellipse is
                              extended -- the extra_rx ring boundary is not
                              (a separate, differently-shaped problem).
                              The extension itself is capped to a bounded
                              band around the primary boundary (roughly
                              4x the widest ACTIVE level's own feather
                              width): filling the ENTIRE rest of the frame
                              with a flat radial projection was tried first
                              and found to corrupt real detail far from the
                              disk on real data -- when a co-centred
                              extra_rx shape (e.g. Saturn's rings) is also
                              present, its real texture sits well outside
                              this cap and must decompose from its own true
                              pixel values, not the globe-boundary's
                              artificial fill (confirmed via a visible
                              checkerboard/moiré artifact in the ring band
                              before this cap was added).
                              False (default): bit-identical, no second
                              decompose() call.
        overshoot_clamp_radius_px: When > 0, clamps the final assembled
                              result to the local min/max of the real input
                              `image` within this pixel radius (see
                              `_local_min_max`) -- suppresses unsharp-mask
                              overshoot/ringing (the white-rim class of
                              artifact) at any genuinely hard edge, most
                              notably the disk limb, regardless of how the
                              feather mask is shaped. A fundamentally
                              different (output-domain) mechanism from
                              edge_feather_factor/confidence_map/
                              fill_outside_before_sharpen above (all
                              input/mask-domain) -- see
                              WaveletConfig.master_overshoot_clamp_radius_px
                              for the full rationale. 0.0 (default):
                              bit-identical, no clamp applied.

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
                extra_rx=extra_rx, extra_ry=extra_ry, extra_angle=extra_angle,
                extra_gap_px=extra_gap_px,
                confidence_map=confidence_map,
                fill_outside_before_sharpen=fill_outside_before_sharpen,
                overshoot_clamp_radius_px=overshoot_clamp_radius_px,
            )
            for c in range(image.shape[2])
        ]
        return np.stack(channels, axis=2).astype(np.float32)

    h, w = image.shape
    if confidence_map is not None and confidence_map.shape != (h, w):
        raise ValueError(
            f"confidence_map.shape={confidence_map.shape} must match "
            f"image spatial shape ({h}, {w})"
        )
    _conf = confidence_map if confidence_map is not None else np.float32(1.0)
    rx_m = radius + expand_px
    ry_m = (ry + expand_px) if use_ellipse else None

    coeffs = decompose(image.astype(np.float64), levels, filter_type=filter_type)

    original = coeffs[-1].copy()
    for d in coeffs[:-1]:
        original = original + d

    if fill_outside_before_sharpen:
        fill_ry = ry_m if use_ellipse else rx_m
        filled = _fill_outside_ellipse(image, cx, cy, rx_m, fill_ry, angle)

        # Cap the extension to a bounded band around the primary boundary.
        # Filling the entire rest of the frame (unbounded) was tried first
        # and corrupts any co-centred extra_rx shape's (e.g. Saturn's rings)
        # own real detail far from the globe -- decomposing an artificial
        # radial extrapolation there instead of true pixel values produced
        # a visible checkerboard/moiré artifact once extra_weight_map pulled
        # those spurious coefficients back in. Cap width mirrors the same
        # "widest active feather" reasoning already used for extra_gap_px.
        active_feathers = [(2 ** i) * edge_feather_factor for i, g in enumerate(gains) if g != 0]
        cap_px = 4.0 * (max(active_feathers) if active_feathers else edge_feather_factor)
        if extra_rx is not None and extra_rx > 0 and extra_gap_px is not None and extra_gap_px > 0:
            # The ring's own inner ramp starts contributing real weight
            # starting right at the globe's true boundary (see extra_gap_px's
            # docstring) -- that zone's content is the globe's own real
            # limb-darkening tail, not background, and must decompose from
            # its own true pixel values. Stay safely clear of it.
            cap_px = min(cap_px, 0.5 * extra_gap_px)
        cap_rx = rx_m + cap_px
        cap_ry = (fill_ry + cap_px) if use_ellipse else cap_rx
        yy_cap, xx_cap = np.mgrid[0:h, 0:w].astype(np.float64)
        cos_c, sin_c = np.cos(angle), np.sin(angle)
        dxr = (xx_cap - cx) * cos_c + (yy_cap - cy) * sin_c
        dyr = -(xx_cap - cx) * sin_c + (yy_cap - cy) * cos_c
        beyond_cap = ((dxr / cap_rx) ** 2 + (dyr / cap_ry) ** 2) > 1.0
        filled[beyond_cap] = image[beyond_cap]

        detail_coeffs = decompose(filled.astype(np.float64), levels, filter_type=filter_type)[:-1]
    else:
        detail_coeffs = coeffs[:-1]

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

    # When a second shape (extra_rx) is present, its OUTER boundary is still
    # feathered with the normal per-shape normalised-ellipse method (a single
    # shape's own boundary, no crossing-with-anything-else concern). Its
    # INNER edge (where it meets the primary disk) is different: two
    # real-data bugs were found and fixed here on 2026-08-13 --
    #
    # (1) A solid second ellipse reaching the centre backfills gain=1
    #     immediately outside the disk's own feather zone, defeating that
    #     feather's purpose of giving the disk's real limb gradient a quiet,
    #     unboosted buffer -- visible ringing right at the border.
    # (2) Carving a hole with a SEPARATE ellipse at the extra shape's own
    #     (generally different) eccentricity crosses the disk's boundary at
    #     some off-axis angle (any two co-centred ellipses of different
    #     eccentricity do), and right at that crossing the hole stops
    #     reaching the disk's actual edge -- a wedge of still-bright,
    #     still-unboosted pixels amid sharpened neighbours (a visible cusp).
    #
    # A first fix for (2) used a HARD pixel-distance exclusion band (weight
    # forced to exactly 0 for a fixed width outside the disk, matching (1)'s
    # convention of "feather only fades on the inside, hard 0 immediately
    # outside"), applied via cv2.distanceTransform of the disk shape (no
    # crossing issue, unlike a second ellipse). That worked for the ringing,
    # but real-data review found the "gap" isn't actually empty in real
    # multi-frame Saturn stacks -- it's the disk's own fairly-bright
    # limb-darkening tail -- so a FLAT zero band there (plus the extra
    # shape's own separate fade-in just beyond it) reads as a visibly
    # distinct unsharpened "halo" sandwiched between two sharpened regions.
    #
    # Final fix: make the inner edge a single CONTINUOUS ramp with no flat
    # segment at all -- weight rises smoothly from 0 right at the disk's own
    # boundary (measured by real pixel distance, so still no eccentricity
    # crossing) up to full strength over extra_gap_px pixels. Every pixel in
    # that zone gets *some* gain, increasing gradually, rather than "zero
    # for a while, then a separate ramp" -- there is no longer a
    # perceptually-flat band for the eye to pick out.
    _disk_dist_out = None
    if extra_rx is not None and extra_rx > 0 and extra_gap_px is not None and extra_gap_px > 0:
        import cv2 as _cv2

        if use_ellipse:
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            dx_r = (X_g - cx) * cos_a + (Y_g - cy) * sin_a
            dy_r = -(X_g - cx) * sin_a + (Y_g - cy) * cos_a
            disk_core = ((dx_r / rx_m) ** 2 + (dy_r / ry_m) ** 2) <= 1.0
        else:
            disk_core = ((X_g - cx) ** 2 + (Y_g - cy) ** 2) <= rx_m ** 2
        _disk_dist_out = _cv2.distanceTransform(
            (~disk_core).astype(np.uint8), _cv2.DIST_L2, 5
        )

    result = original.copy()
    for level_idx, (detail, gain) in enumerate(zip(detail_coeffs, gains)):
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

        if extra_rx is not None and extra_rx > 0:
            _extra_ry = extra_ry if (extra_ry is not None and extra_ry > 0) else extra_rx
            _extra_angle = extra_angle if extra_angle is not None else angle
            extra_weight_map = _make_disk_weight_ellipse(
                h, w, cx, cy, extra_rx, _extra_ry, _extra_angle, feather_L
            )
            if _disk_dist_out is not None:
                t_inner = np.clip(_disk_dist_out / extra_gap_px, 0.0, 1.0)
                inner_ramp = (0.5 * (1.0 - np.cos(np.pi * t_inner))).astype(np.float32)
                extra_weight_map = np.minimum(extra_weight_map, inner_ramp)
            weight_map = np.maximum(weight_map, extra_weight_map)

        result = result + d_thr * gain * weight_map * _conf

    if overshoot_clamp_radius_px > 0:
        lo, hi = _local_min_max(image, overshoot_clamp_radius_px)
        result = np.clip(result, lo, hi)

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
    extra_rx: Optional[float] = None,
    extra_ry: Optional[float] = None,
    extra_angle: Optional[float] = None,
    extra_gap_px: Optional[float] = None,
    confidence_map: Optional[np.ndarray] = None,
    fill_outside_before_sharpen: bool = False,
    overshoot_clamp_radius_px: float = 0.0,
) -> np.ndarray:
    """Disk-aware sharpening for colour (H, W, 3) RGB float images via Lab L-channel.

    Converts RGB → Lab, applies :func:`sharpen_disk_aware` to the L channel
    only, then converts back.  Chrominance is preserved unchanged.

    Args and returns: same as :func:`sharpen_color` plus disk geometry args,
    the denoise_amounts / filter_type parameters, and the optional second-
    ellipse extra_rx/extra_ry/extra_angle/extra_gap_px, confidence_map,
    fill_outside_before_sharpen, and overshoot_clamp_radius_px (see
    sharpen_disk_aware's docstring) -- all channel-agnostic (pure spatial
    signals) so they are passed straight through to the single L-channel
    call, no per-channel handling needed.
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
        extra_rx=extra_rx, extra_ry=extra_ry, extra_angle=extra_angle,
        extra_gap_px=extra_gap_px,
        confidence_map=confidence_map,
        fill_outside_before_sharpen=fill_outside_before_sharpen,
        overshoot_clamp_radius_px=overshoot_clamp_radius_px,
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
    overshoot_clamp_radius_px: float = 0.0,
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
        overshoot_clamp_radius_px: When > 0, clamps the result to the local
                         min/max of *image* within this pixel radius --
                         suppresses unsharp-mask overshoot/ringing at hard
                         edges (see sharpen_disk_aware's docstring and
                         WaveletConfig.master_overshoot_clamp_radius_px for
                         the full rationale). 0.0 (default): bit-identical.

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
                    filter_type=filter_type,
                    overshoot_clamp_radius_px=overshoot_clamp_radius_px)
            for c in range(image.shape[2])
        ]
        return np.stack(channels, axis=2).astype(np.float32)

    coeffs = decompose(image.astype(np.float64), levels, filter_type=filter_type)
    result = reconstruct(coeffs, gains, sharpen_filter=sharpen_filter,
                         denoise_amounts=denoise_amounts)
    if overshoot_clamp_radius_px > 0:
        lo, hi = _local_min_max(image, overshoot_clamp_radius_px)
        result = np.clip(result, lo, hi)
    return np.clip(result, 0.0, 1.0).astype(np.float32)
