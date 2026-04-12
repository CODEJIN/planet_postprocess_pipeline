"""
Lucky stacking — AS!4-style local-warp frame stacking from SER video files.

Algorithm (per SER file):
  1. Score all frames with Laplacian variance on the planet disk.
  2. Select the top top_percent of frames.
  3. Build a high-SNR reference frame by global-aligning and mean-stacking
     the top reference_n_frames.
  4. Detect the planet disk center (cx, cy, radius) from the reference.
  5. Generate an AP (Alignment Point) grid across the disk.
  6. Pre-compute shared resources (Hann window, query grid, index map).
  7. For each selected frame:
       a. Global translation alignment via limb-center detection.
       b. Per-AP local shift estimation via phase correlation with Hann windowing.
       c. Interpolate RELIABLE AP shifts to a full-resolution warp map using
          Gaussian kernel regression (C∞-smooth, no triangle-edge artifacts).
       d. Apply local warp via cv2.remap (INTER_LINEAR, single interpolation
          combining global + local shift to avoid double-interpolation blur).
       e. Accumulate into quality-weighted sum with disk feather mask.
  8. Normalise and return the stacked float32 image.

Key differences from existing Step 5 de-rotation stack:
  - Step 5 stacks ~10-20 pre-stacked TIF images (one per filter cycle).
  - Lucky stacking stacks thousands of raw video frames from a single capture,
    correcting LOCAL atmospheric distortions via the AP warp — the same
    technique used by AutoStakkert!4.

Performance targets (280×280 px SER, 10 000 frames, 1 034 selected at 10%):
  Quality scoring      ~2 s  (every-other-frame, mask reuse)
  Reference build       ~0 s
  AP grid + resources   ~0 s
  AP warp loop         ~16 s  (phaseCorrelate 32×32 + Gaussian KR per frame)
  Total                ~18 s
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from pipeline.config import LuckyStackConfig
from pipeline.modules.derotation import (
    apply_shift,
    find_disk_center,
    limb_center_align,
    subpixel_align,
)
from pipeline.modules.quality import laplacian_var, planet_mask
from pipeline.modules.ser_io import SERReader


# ── 1. Quality scoring ─────────────────────────────────────────────────────────

def score_frames(
    reader: SERReader,
    cfg: LuckyStackConfig,
    score_step: int = 2,
) -> np.ndarray:
    """Compute Laplacian variance quality score for every frame.

    Samples every *score_step* frames for speed; linearly interpolates the rest.
    The planet disk mask is computed once from the first frame.

    Returns:
        float32 array of length FrameCount; higher = sharper/better seeing.
    """
    n_frames: int = reader.header["FrameCount"]

    # Quality mask: inner 80% of disk radius (excludes limb gradient zone).
    # The limb's intrinsic planet-sky edge creates a large gradient regardless
    # of seeing quality, biasing the quality score toward frames with sharper
    # limbs rather than sharper interior structure. Restricting to the inner
    # 80% measures only atmospheric sharpness on disk features (belts, zones).
    frame0 = reader.get_frame(0).astype(np.float32) / 255.0
    try:
        cx0, cy0, semi_a0, _, _ = find_disk_center(frame0)
        H, W = frame0.shape[:2]
        yy0, xx0 = np.mgrid[0:H, 0:W].astype(np.float32)
        dist0 = np.sqrt((xx0 - cx0) ** 2 + (yy0 - cy0) ** 2)
        mask = dist0 <= (float(semi_a0) * 0.80)
        if mask.sum() < 100:  # fallback if disk detection fails
            mask = planet_mask(frame0)
    except Exception:
        mask = planet_mask(frame0)

    sampled_idx: List[int] = list(range(0, n_frames, score_step))
    sampled_scores: List[float] = []

    for idx in sampled_idx:
        frame = reader.get_frame(idx).astype(np.float32) / 255.0
        # Mild Gaussian denoise before Laplacian (suppress shot noise)
        frame_b = cv2.GaussianBlur(frame, (0, 0), 1.2)
        sampled_scores.append(laplacian_var(frame_b, mask))

    # Interpolate to all frame indices
    scores = np.interp(
        np.arange(n_frames, dtype=np.float32),
        np.array(sampled_idx, dtype=np.float32),
        np.array(sampled_scores, dtype=np.float32),
    ).astype(np.float32)

    return scores


# ── 2. Reference frame construction ───────────────────────────────────────────

def build_reference_frame(
    reader: SERReader,
    scores: np.ndarray,
    cfg: LuckyStackConfig,
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Build a high-SNR reference frame from the top-scored frames.

    Global-aligns each of the best reference_n_frames to the single best frame
    via phase correlation, then returns their unweighted mean.

    Returns:
        (reference_f32, (disk_cx, disk_cy, disk_radius))

    Raises:
        RuntimeError if the planet disk cannot be reliably detected.
    """
    n = min(int(cfg.reference_n_frames), len(scores))
    best_indices = np.argsort(scores)[::-1][:n]

    best_idx = int(best_indices[0])
    best_frame = reader.get_frame(best_idx).astype(np.float32) / 255.0

    accum = best_frame.astype(np.float64)
    for idx in best_indices[1:]:
        frame = reader.get_frame(int(idx)).astype(np.float32) / 255.0
        dx, dy = subpixel_align(best_frame, frame)
        if abs(dx) > 20 or abs(dy) > 20:  # bad-frame guard
            accum += frame.astype(np.float64)
        else:
            accum += apply_shift(frame, dx, dy).astype(np.float64)

    reference = np.clip(accum / n, 0.0, 1.0).astype(np.float32)

    cx, cy, semi_a, _, _ = find_disk_center(reference)
    radius = float(semi_a)
    if radius < 10.0:
        raise RuntimeError(
            f"Disk not reliably detected in reference (radius={radius:.1f} px). "
            "Check that step01 produced a valid SER with the planet visible."
        )
    return reference, (float(cx), float(cy), radius)


# ── 3. AP grid generation ──────────────────────────────────────────────────────

def generate_ap_grid(
    disk_cx: float,
    disk_cy: float,
    disk_radius: float,
    reference: np.ndarray,
    cfg: LuckyStackConfig,
) -> List[Tuple[int, int]]:
    """Create an AP grid over the planet disk in the reference frame.

    Includes only APs whose patch:
    - is fully contained within the image
    - has its centre inside the disk
    - has local RMS contrast >= cfg.ap_min_contrast (rejects uniform sky)
    - has mean brightness >= cfg.ap_min_brightness (rejects very dark limb;
      equivalent to AS!4 Min Bright=50, i.e. 50/255≈0.196)

    Returns list of (ax, ay) integer AP centre coordinates.
    """
    H, W = reference.shape[:2]
    half = cfg.ap_size // 2
    min_bright = getattr(cfg, "ap_min_brightness", 0.0)
    valid_aps: List[Tuple[int, int]] = []

    for ay in range(half, H - half, cfg.ap_step):
        for ax in range(half, W - half, cfg.ap_step):
            dist = np.sqrt((ax - disk_cx) ** 2 + (ay - disk_cy) ** 2)
            if dist >= disk_radius:
                continue
            patch = reference[ay - half : ay + half, ax - half : ax + half]
            if float(patch.std()) < cfg.ap_min_contrast:
                continue
            if min_bright > 0.0 and float(patch.mean()) < min_bright:
                continue
            valid_aps.append((ax, ay))

    return valid_aps


# ── 4. Per-AP shift estimation ─────────────────────────────────────────────────

def _make_hann2d(size: int) -> np.ndarray:
    """Pre-compute a 2-D Hann window of shape (size, size)."""
    h = np.hanning(size).astype(np.float32)
    return np.outer(h, h)


def _estimate_ap_shift(
    ref_patch: np.ndarray,
    frm_patch: np.ndarray,
    hann2d: np.ndarray,
    cfg: LuckyStackConfig,
) -> Tuple[Optional[float], Optional[float], float]:
    """Estimate local shift via phase correlation with Hann windowing.

    cv2.phaseCorrelate(src1, src2) returns (dx, dy) such that:
        src2 ≈ src1 shifted by (dx, dy)
    To align src2 to src1, sample src2 at (x + dx, y + dy) →
    the remap map uses map_x = x + dx, map_y = y + dy.

    Returns (dx, dy, confidence) or (None, None, 0.0) if rejected.
    """
    ref_w = ref_patch * hann2d
    frm_w = frm_patch * hann2d

    (dx, dy), confidence = cv2.phaseCorrelate(ref_w, frm_w)
    confidence = float(confidence)

    if confidence < cfg.ap_confidence_threshold:
        return None, None, 0.0
    if abs(dx) > cfg.ap_search_range or abs(dy) > cfg.ap_search_range:
        return None, None, 0.0

    return float(dx), float(dy), confidence


# ── 5. Disk-masked global alignment ───────────────────────────────────────────

def _disk_phase_align(
    reference: np.ndarray,
    frame: np.ndarray,
    disk_cx: float,
    disk_cy: float,
    disk_radius: float,
    feather_frac: float = 0.15,
) -> Tuple[float, float]:
    """Sub-pixel global alignment via disk-masked phase correlation.

    More accurate than limb_center_align (≈±0.1 px vs ≈±0.5 px) because:
    - Only the planet disk is correlated — background noise cannot bias the result.
    - A soft (cosine) feathered mask avoids ringing from a hard disk boundary.

    The mask is centred on the reference disk.  For typical atmospheric shifts
    (±3–5 px) the planet disk in the target frame still lies mostly within the
    mask window, so the correlation is valid.

    Returns (dx, dy) shift to apply to *frame* to align it with *reference*.
    """
    H, W = reference.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dist = np.sqrt((xx - disk_cx) ** 2 + (yy - disk_cy) ** 2)

    feather_px = disk_radius * feather_frac
    # Cosine taper: 1.0 at disk centre, 0.0 outside disk_radius
    mask = np.clip((disk_radius - dist) / feather_px, 0.0, 1.0)
    mask = (0.5 - 0.5 * np.cos(mask * np.pi)).astype(np.float32)

    (dx, dy), _ = cv2.phaseCorrelate(
        (reference * mask).astype(np.float32),
        (frame     * mask).astype(np.float32),
    )
    return float(dx), float(dy)


# ── 6. Per-frame warp + remap ──────────────────────────────────────────────────

def _compute_warp_maps(
    frame_aligned: np.ndarray,
    reference: np.ndarray,
    ap_positions: List[Tuple[int, int]],
    hann2d: np.ndarray,
    query_pts: np.ndarray,    # kept for API compatibility; not used
    cfg: LuckyStackConfig,
) -> Tuple[np.ndarray, np.ndarray, int, List[Tuple[int, int, float, float, float]]]:
    """Compute per-AP shifts and build smooth full-resolution warp maps.

    Uses Gaussian kernel regression (Nadaraya-Watson estimator) instead of
    Delaunay triangulation.  Delaunay linear interpolation creates C0-continuous
    fields with persistent gradient discontinuities at triangle edges; over
    thousands of stacked frames these accumulate into a fine mesh artifact that
    wavelet sharpening (×200) amplifies to a visible pattern.

    Kernel regression produces a C∞-smooth field: each pixel's shift is the
    Gaussian-weighted average of nearby reliable APs.  Sigma = ap_step * 0.7
    is the minimum that guarantees continuous gradients between adjacent APs
    (requires σ ≥ ap_step/√2) while still preserving atmospheric corrections
    at scales above ~2×ap_step.  Pixels with < 5% of peak AP influence
    receive zero correction (background / limb fade-out).

    Returns:
        (map_dx, map_dy, n_good_aps)
        map_dx / map_dy: float32 [H, W] shift fields.
        n_good_aps:      number of APs with confident shifts.
    """
    H, W = frame_aligned.shape[:2]
    half = cfg.ap_size // 2

    # Sparse shift grids: place each reliable AP's shift at its pixel location
    shift_x = np.zeros((H, W), dtype=np.float32)
    shift_y = np.zeros((H, W), dtype=np.float32)
    weight   = np.zeros((H, W), dtype=np.float32)

    n_good = 0
    ap_shift_data: List[Tuple[int, int, float, float, float]] = []
    for ax, ay in ap_positions:
        ref_patch = reference[ay - half : ay + half, ax - half : ax + half].astype(np.float32)
        frm_patch = frame_aligned[ay - half : ay + half, ax - half : ax + half].astype(np.float32)

        dx, dy, conf = _estimate_ap_shift(ref_patch, frm_patch, hann2d, cfg)
        if dx is None:
            continue

        # Weight by confidence (Nadaraya-Watson with confidence as importance weight).
        # Replaces binary accept/reject: high-confidence APs dominate the warp field
        # at their location; marginal APs near the threshold contribute proportionally.
        # CRITICAL: store raw dx/dy as numerator values, conf as weight only.
        # GKR computes smooth(shift×weight)/smooth(weight) = confidence-weighted mean.
        # Previous bug: shift_x = dx*conf → result was dx*conf² / conf = dx*conf
        # (shifts systematically under-estimated by a factor of conf).
        shift_x[ay, ax] = float(dx)
        shift_y[ay, ax] = float(dy)
        weight[ay, ax]  = conf
        ap_shift_data.append((ax, ay, float(dx), float(dy), conf))
        n_good += 1

    if n_good < 3:
        zero = np.zeros((H, W), dtype=np.float32)
        return zero, zero, n_good, []

    # Gaussian kernel regression: smooth the shift × weight and weight maps,
    # then divide.  sigma = ap_step × ap_sigma_factor.  Must be ≥ ap_step/√2
    # ≈ 0.71 × ap_step to guarantee C∞-smooth gradients between adjacent APs.
    # Higher values reduce noise in AP shifts at the cost of spatial resolution.
    sigma = float(cfg.ap_step) * cfg.ap_sigma_factor
    ksize = int(6.0 * sigma + 1) | 1  # odd kernel, ≥ 6σ wide

    smooth_wx = cv2.GaussianBlur(shift_x * weight, (ksize, ksize), sigma)
    smooth_wy = cv2.GaussianBlur(shift_y * weight, (ksize, ksize), sigma)
    smooth_w  = cv2.GaussianBlur(weight,            (ksize, ksize), sigma)

    # Normalise; zero out pixels with negligible AP coverage (< 5% of peak)
    coverage_threshold = float(np.max(smooth_w)) * 0.05
    coverage_ok = smooth_w >= coverage_threshold

    map_dx = np.where(coverage_ok, smooth_wx / np.maximum(smooth_w, 1e-9), 0.0).astype(np.float32)
    map_dy = np.where(coverage_ok, smooth_wy / np.maximum(smooth_w, 1e-9), 0.0).astype(np.float32)

    return map_dx, map_dy, n_good, ap_shift_data


def _build_warp_maps_from_cache(
    H: int,
    W: int,
    ap_shift_data: List[Tuple[int, int, float, float, float]],
    cfg: LuckyStackConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build dense warp maps from cached AP shifts (no phaseCorrelate).

    Used in pass 2 of sigma-clipping to reconstruct warp maps from the shifts
    computed in pass 1.  Identical Gaussian KR as _compute_warp_maps but reads
    from a pre-computed list rather than calling cv2.phaseCorrelate.

    Returns:
        (map_dx, map_dy): float32 [H, W] dense shift fields.
    """
    shift_x = np.zeros((H, W), dtype=np.float32)
    shift_y = np.zeros((H, W), dtype=np.float32)
    weight   = np.zeros((H, W), dtype=np.float32)

    for ax, ay, dx, dy, conf in ap_shift_data:
        shift_x[ay, ax] = dx
        shift_y[ay, ax] = dy
        weight[ay, ax]  = conf

    if not ap_shift_data:
        return shift_x, shift_y

    sigma = float(cfg.ap_step) * cfg.ap_sigma_factor
    ksize = int(6.0 * sigma + 1) | 1

    smooth_wx = cv2.GaussianBlur(shift_x * weight, (ksize, ksize), sigma)
    smooth_wy = cv2.GaussianBlur(shift_y * weight, (ksize, ksize), sigma)
    smooth_w  = cv2.GaussianBlur(weight,            (ksize, ksize), sigma)

    coverage_threshold = float(np.max(smooth_w)) * 0.05
    coverage_ok = smooth_w >= coverage_threshold

    map_dx = np.where(coverage_ok, smooth_wx / np.maximum(smooth_w, 1e-9), 0.0).astype(np.float32)
    map_dy = np.where(coverage_ok, smooth_wy / np.maximum(smooth_w, 1e-9), 0.0).astype(np.float32)
    return map_dx, map_dy


# ── 6. Main stacking loop ──────────────────────────────────────────────────────

def apply_warp_and_stack(
    reader: SERReader,
    selected_indices: np.ndarray,
    scores: np.ndarray,
    reference: np.ndarray,
    disk_cx: float,
    disk_cy: float,
    disk_radius: float,
    ap_positions: List[Tuple[int, int]],
    cfg: LuckyStackConfig,
    progress_callback=None,
) -> Tuple[np.ndarray, Dict]:
    """Warp and accumulate all selected frames into a quality-weighted stack.

    For each selected frame:
      1. Global alignment via limb_center_align.
      2. Per-AP local shift estimation (phaseCorrelate with Hann window).
      3. Warp map via Gaussian kernel regression (C∞-smooth).
      4. Combined global+local warp via single cv2.remap (one interpolation).
      5. Quality-weighted accumulation with 2-pass sigma clipping.

    Returns:
        (stacked_image, stats_dict)
    """
    H, W = reference.shape[:2]
    n_selected = len(selected_indices)
    kappa = float(getattr(cfg, "sigma_clip_kappa", 3.0))

    hann2d = _make_hann2d(cfg.ap_size)
    query_pts = np.empty((0, 2), dtype=np.float64)

    xx_base = np.tile(np.arange(W, dtype=np.float32), (H, 1))
    yy_base = np.tile(np.arange(H, dtype=np.float32)[:, None], (1, W))

    accum      = np.zeros((H, W), dtype=np.float64)
    accum_sq   = np.zeros((H, W), dtype=np.float64)
    weight_sum = np.zeros((H, W), dtype=np.float64)

    frame_logs: List[Dict] = []
    n_global_only = 0

    # Caches for pass 2
    global_shifts: List[Tuple[float, float]] = []
    all_ap_shifts: List[List[Tuple[int, int, float, float, float]]] = []

    # ── Pass 1 ────────────────────────────────────────────────────────────
    for i, idx in enumerate(selected_indices):
        frame = reader.get_frame(int(idx)).astype(np.float32) / 255.0

        # Global alignment via limb centre
        dx_g, dy_g = limb_center_align(disk_cx, disk_cy, frame)
        max_g = disk_radius * 0.5
        if abs(dx_g) > max_g or abs(dy_g) > max_g:
            dx_g, dy_g = subpixel_align(reference, frame)
        align_method = "limb_center"

        frame_aligned = apply_shift(frame, dx_g, dy_g)

        map_dx, map_dy, n_good, ap_shift_data = _compute_warp_maps(
            frame_aligned, reference, ap_positions, hann2d, query_pts, cfg
        )
        if n_good < 3:
            n_global_only += 1

        global_shifts.append((dx_g, dy_g))
        all_ap_shifts.append(ap_shift_data)

        remap_x = (xx_base + map_dx - dx_g).astype(np.float32)
        remap_y = (yy_base + map_dy - dy_g).astype(np.float32)
        warped = cv2.remap(
            frame, remap_x, remap_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        warped = np.clip(warped, 0.0, 1.0)

        quality_w = max(float(scores[idx]) ** cfg.quality_weight_power, 1e-9)
        warped64 = warped.astype(np.float64)
        accum      += warped64 * quality_w
        accum_sq   += warped64 ** 2 * quality_w
        weight_sum += quality_w

        frame_logs.append({
            "frame_idx": int(idx),
            "quality_score": round(float(scores[idx]), 6),
            "global_shift_px": [round(float(dx_g), 3), round(float(dy_g), 3)],
            "align_method": align_method,
            "n_good_aps": n_good,
        })

        if progress_callback is not None and i % 50 == 0:
            progress_callback(i, n_selected)

    mean1 = np.where(weight_sum > 1e-12, accum / weight_sum, 0.0)
    var1  = np.maximum(np.where(weight_sum > 1e-12, accum_sq / weight_sum - mean1 ** 2, 0.0), 0.0)
    std1  = np.maximum(np.sqrt(var1), 1.0 / 255.0)

    # ── Pass 2: sigma-clip ────────────────────────────────────────────────
    if kappa > 0.0:
        accum2      = np.zeros((H, W), dtype=np.float64)
        weight_sum2 = np.zeros((H, W), dtype=np.float64)

        for i, idx in enumerate(selected_indices):
            frame = reader.get_frame(int(idx)).astype(np.float32) / 255.0

            dx_g, dy_g    = global_shifts[i]
            ap_shift_data = all_ap_shifts[i]
            map_dx, map_dy = _build_warp_maps_from_cache(H, W, ap_shift_data, cfg)

            remap_x = (xx_base + map_dx - dx_g).astype(np.float32)
            remap_y = (yy_base + map_dy - dy_g).astype(np.float32)
            warped = cv2.remap(
                frame, remap_x, remap_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            warped = np.clip(warped, 0.0, 1.0)

            quality_w  = max(float(scores[idx]) ** cfg.quality_weight_power, 1e-9)
            clip_mask  = (np.abs(warped - mean1) <= kappa * std1).astype(np.float64)
            warped64   = warped.astype(np.float64)
            accum2      += warped64 * quality_w * clip_mask
            weight_sum2 += quality_w * clip_mask

        stacked = np.where(
            weight_sum2 > 1e-12,
            accum2 / np.maximum(weight_sum2, 1e-12),
            mean1,
        ).astype(np.float32)
    else:
        stacked = mean1.astype(np.float32)

    stacked = np.clip(stacked, 0.0, 1.0)

    # Smooth the limb zone to suppress stacking jitter at the disk edge.
    # Uses a cosine feather (smooth, no hard inner boundary → no ring artifact).
    yy_s, xx_s = np.mgrid[0:H, 0:W].astype(np.float32)
    dist_s = np.sqrt((xx_s - disk_cx) ** 2 + (yy_s - disk_cy) ** 2)
    # blend weight: 0 inside (radius-20), rises to 1 at radius, cosine taper
    feather_w = np.clip((dist_s - (disk_radius - 20)) / 20.0, 0.0, 1.0)
    feather_w = (0.5 - 0.5 * np.cos(feather_w * np.pi)).astype(np.float32)
    stacked_blur = cv2.GaussianBlur(stacked, (0, 0), 3.0)
    stacked = (stacked * (1.0 - feather_w) + stacked_blur * feather_w).astype(np.float32)

    if frame_logs:
        avg_n_good = float(np.mean([f["n_good_aps"] for f in frame_logs]))
        n_total_aps = len(ap_positions)
        pct = 100.0 * avg_n_good / max(n_total_aps, 1)
        print(f"\n  AP acceptance: {avg_n_good:.0f}/{n_total_aps} ({pct:.0f}%)", flush=True)

    stats = {
        "n_stacked": n_selected,
        "n_global_only_frames": n_global_only,
        "n_aps": len(ap_positions),
        "disk_center_px": [round(disk_cx, 2), round(disk_cy, 2)],
        "disk_radius_px": round(disk_radius, 2),
        "frames": frame_logs,
    }
    return stacked, stats


# ── 7. Top-level entry point ───────────────────────────────────────────────────

def lucky_stack_ser(
    ser_path: Path,
    cfg: LuckyStackConfig,
    progress_callback=None,
) -> Tuple[np.ndarray, Dict]:
    """Run the full lucky stacking pipeline on a single SER file.

    Args:
        ser_path:          Path to a PIPP-preprocessed SER file.
        cfg:               LuckyStackConfig.
        progress_callback: Optional (done, total) callback for UI progress bars.

    Returns:
        (stacked_image, log_dict)
        stacked_image: float32 [0,1] 2-D array, ready for write_tif_16bit().
        log_dict: processing statistics (timing, AP counts, per-frame shifts).

    Raises:
        RuntimeError if SER is invalid or disk cannot be detected.
    """
    t0 = time.perf_counter()

    with SERReader(ser_path) as reader:
        n_frames: int = reader.header["FrameCount"]
        print(f"\n  SER: {ser_path.name}  ({n_frames} frames)", flush=True)

        if n_frames < cfg.min_frames:
            raise RuntimeError(
                f"SER has only {n_frames} frames (min_frames={cfg.min_frames}). "
                "Lower min_frames or use a longer capture."
            )

        # ── Phase 1: Quality scoring ──────────────────────────────────────
        print("  [1/4] Scoring frames…", end="\r", flush=True)
        scores = score_frames(reader, cfg, score_step=2)
        t1 = time.perf_counter()
        print(f"  [1/4] Scored {n_frames} frames          ({t1-t0:.1f}s)", flush=True)

        # ── Phase 2: Frame selection ──────────────────────────────────────
        n_select = max(cfg.min_frames, int(n_frames * cfg.top_percent))
        n_select = min(n_select, n_frames)
        selected_indices = np.argsort(scores)[::-1][:n_select]
        print(
            f"  [2/4] Selected {n_select}/{n_frames} frames "
            f"({100.0*n_select/n_frames:.1f}%)",
            flush=True,
        )

        # ── Phase 3: Reference frame ──────────────────────────────────────
        print("  [3/4] Building reference…", end="\r", flush=True)
        reference, (disk_cx, disk_cy, disk_radius) = build_reference_frame(
            reader, scores, cfg
        )
        t2 = time.perf_counter()
        print(
            f"  [3/4] Reference built   ({t2-t1:.1f}s)  "
            f"disk=({disk_cx:.1f},{disk_cy:.1f}) r={disk_radius:.1f}px",
            flush=True,
        )

        stacked: Optional[np.ndarray] = None
        stats: Dict = {}
        t_stack_total = 0.0
        n_iter = max(1, getattr(cfg, "n_iterations", 1))

        for iteration in range(n_iter):
            iter_label = f"iter {iteration+1}/{n_iter}" if n_iter > 1 else ""

            ap_positions = generate_ap_grid(disk_cx, disk_cy, disk_radius, reference, cfg)
            t3 = time.perf_counter()
            print(
                f"  [4/5] AP grid: {len(ap_positions)} points  ({t3-t2:.2f}s)"
                + (f"  [{iter_label}]" if iter_label else ""),
                flush=True,
            )

            step_label = "[5/5]" if n_iter == 1 else f"[{4+iteration+1}/{4+n_iter}]"
            print(f"  {step_label} Stacking {n_select} frames…", end="\r", flush=True)

            def _prog(done: int, total: int, _lbl=step_label) -> None:
                pct = 100 * done // max(total, 1)
                print(f"  {_lbl} Stacking {done}/{total} ({pct}%)…", end="\r", flush=True)
                if progress_callback is not None:
                    progress_callback(done, total)

            stacked, stats = apply_warp_and_stack(
                reader,
                selected_indices,
                scores,
                reference,
                disk_cx, disk_cy, disk_radius,
                ap_positions,
                cfg,
                progress_callback=_prog,
            )
            t4 = time.perf_counter()
            t_stack_total += t4 - t3
            print(
                f"\n  {step_label} Done  (stack {t4-t3:.1f}s"
                + (f"  [{iter_label}]" if iter_label else "") + ")",
                flush=True,
            )

            if iteration < n_iter - 1:
                reference = stacked

    t_total = time.perf_counter() - t0
    print(f"\n  Total: {t_total:.1f}s", flush=True)

    log: Dict = {
        "ser_path": str(ser_path),
        "n_frames_total": n_frames,
        "n_frames_selected": n_select,
        "top_percent": cfg.top_percent,
        "timing_s": {
            "scoring":   round(t1 - t0, 2),
            "reference": round(t2 - t1, 2),
            "stacking":  round(t_stack_total, 2),
            "total":     round(t_total, 2),
        },
        **stats,
    }
    return stacked, log
