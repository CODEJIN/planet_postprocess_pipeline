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

import multiprocessing as _mp
import threading as _threading
import time
from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor, as_completed as _as_completed
from queue import Empty as _QueueEmpty
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
    progress_callback=None,
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

    for i, idx in enumerate(sampled_idx):
        frame = reader.get_frame(idx).astype(np.float32) / 255.0
        sampled_scores.append(laplacian_var(frame, mask))
        if progress_callback is not None and i % 50 == 0:
            progress_callback(idx, n_frames)

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


# ── 3b. Adaptive AP grid (try14: LoG scale detection + dynamic AP sizes) ──────

_HANN_CACHE: Dict[int, np.ndarray] = {}


def _get_hann(ap_size: int) -> np.ndarray:
    """Return a cached 2-D Hann window for the given patch size."""
    if ap_size not in _HANN_CACHE:
        _HANN_CACHE[ap_size] = _make_hann2d(ap_size)
    return _HANN_CACHE[ap_size]


def local_log_energy(patch: np.ndarray, sigma: float) -> float:
    """Local Laplacian-of-Gaussian energy at scale sigma.

    Computes sigma^2 * mean(LoG^2) on the interior of the patch
    (margin = sigma*1.5) to exclude edge effects from the patch boundary.
    Higher values indicate stronger local feature energy at that scale.
    """
    blur = cv2.GaussianBlur(patch, (0, 0), sigma)
    lap  = cv2.Laplacian(blur, cv2.CV_32F)
    margin = max(1, int(sigma * 1.5))
    h, w = lap.shape
    center = lap[margin: h - margin, margin: w - margin]
    if center.size == 0:
        return 0.0
    return float((sigma ** 2) * np.mean(center ** 2))


def build_ap_size_candidates(disk_radius: float) -> List[int]:
    """Compute AP size candidates scaled to the planet disk radius.

    Maximum AP size = disk_radius * 1.28 (AP_half / disk_radius ≈ 0.64),
    rounded down to 8px. This ensures the largest AP covers the same fraction
    of the disk regardless of telescope scale:
      r=100px → max=128px   r=200px → max=256px   r=300px → max=384px

    Size steps:  24–64 at 8px intervals (fine), 64+ at 16px intervals (coarse).
    """
    max_ap_size = int(disk_radius * 1.28) // 8 * 8
    max_ap_size = max(max_ap_size, 64)

    fine   = list(range(24, 65, 8))               # 24,32,40,48,56,64
    coarse = list(range(80, max_ap_size + 1, 16)) # 80,96,...,max
    return fine + coarse


def generate_adaptive_ap_grid(
    disk_cx: float,
    disk_cy: float,
    disk_radius: float,
    reference: np.ndarray,
    cfg,
) -> List[Tuple[int, int, int]]:
    """Generate AP positions with per-point locally optimal AP size.

    Algorithm (try14 approach):
      1. Build AP size candidates scaled to disk_radius.
      2. Dense candidate search at ap_candidate_step spacing within disk.
      3. Per candidate: compute LoG energy at each size; select the size
         with highest energy as the natural scale for that position.
      4. NMS: sort by energy descending; winner's ap_size//2 radius suppresses
         ALL nearby candidates regardless of their size (cross-size suppression).
         This prevents multiple conflicting measurements at the same location.

    Returns list of (ax, ay, ap_size) triples.
    """
    H, W = reference.shape[:2]
    min_ap_size     = cfg.ap_size           # use ap_size as minimum (default 64)
    candidate_step  = int(getattr(cfg, "ap_candidate_step", 8))
    min_brightness  = getattr(cfg, "ap_min_brightness", 0.196)
    min_contrast    = cfg.ap_min_contrast

    all_sizes = build_ap_size_candidates(disk_radius)
    ap_sizes  = [s for s in all_sizes if s >= min_ap_size]
    if not ap_sizes:
        ap_sizes = [min_ap_size]

    sigmas   = {sz: sz / 4.0 for sz in ap_sizes}
    scan_half = max(ap_sizes) // 2   # narrow scan: search boundary = largest AP half

    raw: List[Dict] = []

    for ay in range(scan_half, H - scan_half, candidate_step):
        for ax in range(scan_half, W - scan_half, candidate_step):
            dist = np.sqrt((ax - disk_cx) ** 2 + (ay - disk_cy) ** 2)
            if dist >= disk_radius:
                continue

            # Brightness / contrast filter on the minimum-size patch
            mh = min_ap_size // 2
            base_patch = reference[ay - mh : ay + mh, ax - mh : ax + mh]
            if float(base_patch.mean()) < min_brightness:
                continue
            if float(base_patch.std()) < min_contrast:
                continue

            # LoG energy at each candidate size
            energies: Dict[int, float] = {}
            for sz in ap_sizes:
                half = sz // 2
                y0, y1 = ay - half, ay + half
                x0, x1 = ax - half, ax + half
                if y0 < 0 or y1 > H or x0 < 0 or x1 > W:
                    continue
                patch = reference[y0:y1, x0:x1]
                if patch.shape != (sz, sz):
                    continue
                energies[sz] = local_log_energy(patch, sigmas[sz])

            if not energies:
                continue

            natural_size = max(energies, key=energies.get)
            best_energy  = energies[natural_size]
            if best_energy < 1e-8:
                continue

            raw.append({
                "ax": ax, "ay": ay,
                "ap_size": natural_size,
                "score": best_energy,
            })

    # NMS: cross-size suppression — winner's ap_size//2 radius removes all nearby
    raw.sort(key=lambda c: c["score"], reverse=True)
    kept: List[Dict] = []
    suppressed: set = set()
    for i, c in enumerate(raw):
        if i in suppressed:
            continue
        kept.append(c)
        nms_r = c["ap_size"] // 2
        for j, c2 in enumerate(raw):
            if j <= i or j in suppressed:
                continue
            d = np.sqrt((c["ax"] - c2["ax"]) ** 2 + (c["ay"] - c2["ay"]) ** 2)
            if d < nms_r:
                suppressed.add(j)

    return [(c["ax"], c["ay"], c["ap_size"]) for c in kept]


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


# ── 5. Per-frame warp + remap ──────────────────────────────────────────────────

def _compute_warp_maps(
    frame_aligned: np.ndarray,
    reference: np.ndarray,
    ap_positions: List[Tuple[int, int]],
    hann2d: np.ndarray,
    query_pts: np.ndarray,    # kept for API compatibility; not used
    cfg: LuckyStackConfig,
) -> Tuple[np.ndarray, np.ndarray, int]:
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
    for ax, ay in ap_positions:
        ref_patch = reference[ay - half : ay + half, ax - half : ax + half].astype(np.float32)
        frm_patch = frame_aligned[ay - half : ay + half, ax - half : ax + half].astype(np.float32)

        dx, dy, conf = _estimate_ap_shift(ref_patch, frm_patch, hann2d, cfg)
        if dx is None:
            continue

        # Weight by confidence (Nadaraya-Watson with confidence as importance weight).
        # Replaces binary accept/reject: high-confidence APs dominate the warp field
        # at their location; marginal APs near the threshold contribute proportionally.
        shift_x[ay, ax] = float(dx) * conf
        shift_y[ay, ax] = float(dy) * conf
        weight[ay, ax]  = conf
        n_good += 1

    if n_good < 3:
        zero = np.zeros((H, W), dtype=np.float32)
        return zero, zero, n_good

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

    return map_dx, map_dy, n_good


# ── 5a. Adaptive warp maps (variable AP sizes + wide KR) ──────────────────────

def _compute_adaptive_warp_maps(
    frame_aligned: np.ndarray,
    reference: np.ndarray,
    ap_positions: List[Tuple[int, int, int]],
    cfg,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Adaptive-AP version of Gaussian kernel regression warp maps.

    Uses per-AP size (from generate_adaptive_ap_grid) and cfg.ap_kr_sigma (64px)
    for wide KR smoothing that covers the sparse AP set across the disk.

    Args:
        ap_positions: list of (ax, ay, ap_size) triples.
        cfg:          LuckyStackConfig with ap_kr_sigma, ap_confidence_threshold,
                      ap_search_range fields.

    Returns:
        (map_dx, map_dy, n_good_aps)
    """
    H, W = frame_aligned.shape[:2]

    shift_x = np.zeros((H, W), dtype=np.float32)
    shift_y = np.zeros((H, W), dtype=np.float32)
    weight  = np.zeros((H, W), dtype=np.float32)
    n_good  = 0

    for ax, ay, ap_size in ap_positions:
        half = ap_size // 2
        y0, y1 = ay - half, ay + half
        x0, x1 = ax - half, ax + half
        if y0 < 0 or y1 > H or x0 < 0 or x1 > W:
            continue

        ref_patch = reference[y0:y1, x0:x1].astype(np.float32)
        frm_patch = frame_aligned[y0:y1, x0:x1].astype(np.float32)
        hann      = _get_hann(ap_size)

        dx, dy, conf = _estimate_ap_shift(ref_patch, frm_patch, hann, cfg)
        if dx is None:
            continue

        shift_x[ay, ax] = float(dx) * conf
        shift_y[ay, ax] = float(dy) * conf
        weight[ay, ax]  = conf
        n_good += 1

    if n_good < 3:
        zero = np.zeros((H, W), dtype=np.float32)
        return zero, zero, n_good

    kr_sigma = float(getattr(cfg, "ap_kr_sigma", 64.0))
    ksize = int(6.0 * kr_sigma + 1) | 1

    sw_x = cv2.GaussianBlur(shift_x * weight, (ksize, ksize), kr_sigma)
    sw_y = cv2.GaussianBlur(shift_y * weight, (ksize, ksize), kr_sigma)
    sw   = cv2.GaussianBlur(weight,            (ksize, ksize), kr_sigma)

    coverage_threshold = float(np.max(sw)) * 0.05
    coverage_ok = sw >= coverage_threshold

    map_dx = np.where(coverage_ok, sw_x / np.maximum(sw, 1e-9), 0.0).astype(np.float32)
    map_dy = np.where(coverage_ok, sw_y / np.maximum(sw, 1e-9), 0.0).astype(np.float32)
    return map_dx, map_dy, n_good


# ── 5b. Worker state + worker function for multiprocessing ────────────────────
# _WORKER_STATE is set in the parent process immediately before Pool creation.
# fork workers inherit it via copy-on-write — no large-array pickling needed.

_WORKER_STATE: Dict = {}


def _worker_process_chunk(chunk_indices: List[int]) -> tuple:
    """Process a slice of pre-loaded selected frames (called in fork workers).

    Reads all data from _WORKER_STATE (inherited via fork, no pickling).
    Returns (local_accum, local_weight, local_logs, n_global_only).
    """
    frames           = _WORKER_STATE["frames"]            # (N_sel, H, W) float32
    reference        = _WORKER_STATE["reference"]         # (H, W) float32
    scores           = _WORKER_STATE["scores"]            # full float32 array
    selected_indices = _WORKER_STATE["selected_indices"]  # original frame indices
    ap_positions     = _WORKER_STATE["ap_positions"]
    hann2d           = _WORKER_STATE["hann2d"]
    cfg              = _WORKER_STATE["cfg"]
    disk_cx          = _WORKER_STATE["disk_cx"]
    disk_cy          = _WORKER_STATE["disk_cy"]
    disk_radius      = _WORKER_STATE["disk_radius"]
    xx_base          = _WORKER_STATE["xx_base"]
    yy_base          = _WORKER_STATE["yy_base"]

    H, W = reference.shape[:2]
    local_accum  = np.zeros((H, W), dtype=np.float64)
    local_weight = np.zeros((H, W), dtype=np.float64)
    local_logs: List[Dict] = []
    n_global_only = 0
    query_pts = np.empty((0, 2), dtype=np.float64)

    adaptive_mode = _WORKER_STATE.get("adaptive_mode", False)

    for local_i in chunk_indices:
        frame = frames[local_i]                   # float32 [0, 1]
        idx   = int(selected_indices[local_i])    # original index for score lookup

        # ── Global alignment ──────────────────────────────────────────────
        dx_g, dy_g = limb_center_align(disk_cx, disk_cy, frame)
        max_g = disk_radius * 0.5
        if abs(dx_g) > max_g or abs(dy_g) > max_g:
            dx_g, dy_g = subpixel_align(reference, frame)
            align_method = "phase_correlate"
        else:
            align_method = "limb_center"

        frame_aligned = apply_shift(frame, dx_g, dy_g)

        # ── AP warp maps ──────────────────────────────────────────────────
        if adaptive_mode:
            map_dx, map_dy, n_good = _compute_adaptive_warp_maps(
                frame_aligned, reference, ap_positions, cfg
            )
        else:
            map_dx, map_dy, n_good = _compute_warp_maps(
                frame_aligned, reference, ap_positions, hann2d, query_pts, cfg
            )
        if n_good < 3:
            n_global_only += 1

        # ── Combined remap ────────────────────────────────────────────────
        remap_x = (xx_base + map_dx - dx_g).astype(np.float32)
        remap_y = (yy_base + map_dy - dy_g).astype(np.float32)
        warped = cv2.remap(
            frame, remap_x, remap_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        warped = np.clip(warped, 0.0, 1.0)

        # ── Weighted accumulate ───────────────────────────────────────────
        quality_w = max(float(scores[idx]) ** cfg.quality_weight_power, 1e-9)
        local_accum  += warped.astype(np.float64) * quality_w
        local_weight += quality_w

        local_logs.append({
            "frame_idx":       idx,
            "quality_score":   round(float(scores[idx]), 6),
            "global_shift_px": [round(float(dx_g), 3), round(float(dy_g), 3)],
            "align_method":    align_method,
            "n_good_aps":      n_good,
        })

        # Per-frame progress.
        # Thread path: call _prog_cb directly (shared memory, no IPC).
        # Fork path: put to _prog_queue (IPC pipe); parent reader thread calls callback.
        # Both keys are absent in sequential path → no-op.
        _prog_cb = _WORKER_STATE.get("_prog_cb")
        if _prog_cb is not None:
            with _WORKER_STATE["_prog_lock"]:
                _WORKER_STATE["_prog_done"][0] += 1
                n_done = _WORKER_STATE["_prog_done"][0]
            _prog_cb(n_done, _WORKER_STATE["_prog_total"])
        else:
            _prog_q = _WORKER_STATE.get("_prog_queue")
            if _prog_q is not None:
                _prog_q.put_nowait(1)  # signal: one frame done

    return local_accum, local_weight, local_logs, n_global_only


# ── 6. Main stacking loop ──────────────────────────────────────────────────────

def apply_warp_and_stack(
    selected_frames: np.ndarray,
    selected_indices: np.ndarray,
    scores: np.ndarray,
    reference: np.ndarray,
    disk_cx: float,
    disk_cy: float,
    disk_radius: float,
    ap_positions: List[Tuple[int, int]],
    cfg: LuckyStackConfig,
    n_workers: int = 1,
    progress_callback=None,
) -> Tuple[np.ndarray, Dict]:
    """Warp and accumulate all selected frames into a quality-weighted stack.

    Args:
        selected_frames:   (N, H, W) float32 [0, 1] — pre-loaded from SER.
        selected_indices:  original frame indices in the SER file (for scores).
        n_workers:         1 = sequential; >1 = fork multiprocessing pool.

    For each frame:
      1. Global disk-centre alignment (limb_center_align → apply_shift).
      2. Per-AP local shift estimation (phaseCorrelate with Hann window).
      3. Warp map construction via Gaussian kernel regression (C∞-smooth).
      4. Combined global+local warp via single cv2.remap (one interpolation).
      5. Quality-weighted accumulation.

    Returns:
        (stacked_image, stats_dict)
    """
    global _WORKER_STATE

    H, W = reference.shape[:2]
    n_selected = len(selected_frames)

    # Detect adaptive mode: ap_positions are (ax, ay, ap_size) triples
    adaptive_mode = bool(ap_positions) and len(ap_positions[0]) == 3
    hann2d    = None if adaptive_mode else _make_hann2d(cfg.ap_size)
    xx_base   = np.tile(np.arange(W, dtype=np.float32), (H, 1))
    yy_base   = np.tile(np.arange(H, dtype=np.float32)[:, None], (1, W))
    query_pts = np.empty((0, 2), dtype=np.float64)  # API compat only

    accum      = np.zeros((H, W), dtype=np.float64)
    weight_sum = np.zeros((H, W), dtype=np.float64)
    frame_logs: List[Dict] = []
    n_global_only = 0

    if n_workers > 1:
        # ── Parallel path ──────────────────────────────────────────────────────
        # Set module-level state — workers read from it (no writes → no races).
        # fork: inherited via COW (no pickling).
        # threads: shared directly (same process memory, GIL released by OpenCV/numpy).
        _WORKER_STATE = {
            "frames":           selected_frames,
            "reference":        reference,
            "scores":           scores,
            "selected_indices": selected_indices,
            "ap_positions":     ap_positions,
            "hann2d":           hann2d,
            "cfg":              cfg,
            "disk_cx":          disk_cx,
            "disk_cy":          disk_cy,
            "disk_radius":      disk_radius,
            "xx_base":          xx_base,
            "yy_base":          yy_base,
            "adaptive_mode":    adaptive_mode,
        }

        # Split frame indices into equal-sized chunks
        all_local = list(range(n_selected))
        chunk_size = max(1, (n_selected + n_workers - 1) // n_workers)
        chunks = [all_local[i:i + chunk_size] for i in range(0, n_selected, chunk_size)]

        _fork_ok = "fork" in _mp.get_all_start_methods()
        completed = 0

        if _fork_ok:
            # Linux/macOS: fork pool — COW memory inheritance, fastest.
            # Per-frame progress via mp.Queue: workers put(1) after each frame;
            # a background reader thread reads and calls the callback so the GUI
            # updates continuously, not just when chunks finish.
            ctx = _mp.get_context("fork")
            _prog_queue = ctx.Queue()
            _WORKER_STATE["_prog_queue"] = _prog_queue

            _stop_reader = _threading.Event()
            _prog_done   = [0]

            def _queue_reader() -> None:
                while True:
                    try:
                        _prog_queue.get(timeout=0.05)
                        _prog_done[0] += 1
                        if progress_callback is not None:
                            progress_callback(_prog_done[0], n_selected)
                    except _QueueEmpty:
                        if _stop_reader.is_set():
                            break

            _reader = _threading.Thread(target=_queue_reader, daemon=True)
            _reader.start()

            with ctx.Pool(n_workers) as pool:
                for local_accum, local_weight, local_logs, local_n_global in pool.imap(
                    _worker_process_chunk, chunks
                ):
                    accum      += local_accum
                    weight_sum += local_weight
                    frame_logs.extend(local_logs)
                    n_global_only += local_n_global

            # Drain remaining queue items before stopping reader
            _stop_reader.set()
            _reader.join(timeout=3.0)
            if progress_callback is not None:
                progress_callback(n_selected, n_selected)  # guarantee 100%
        else:
            # Windows: thread pool — OpenCV/numpy release GIL → real parallelism.
            # Workers call progress_callback directly after each frame so the
            # progress bar updates continuously, not just when chunks finish.
            _prog_done = [0]
            _WORKER_STATE["_prog_cb"]    = progress_callback   # None-safe
            _WORKER_STATE["_prog_lock"]  = _threading.Lock()
            _WORKER_STATE["_prog_done"]  = _prog_done
            _WORKER_STATE["_prog_total"] = n_selected

            with _ThreadPoolExecutor(max_workers=n_workers) as executor:
                futs = [executor.submit(_worker_process_chunk, chunk) for chunk in chunks]
                for fut in _as_completed(futs):
                    local_accum, local_weight, local_logs, local_n_global = fut.result()
                    accum      += local_accum
                    weight_sum += local_weight
                    frame_logs.extend(local_logs)
                    n_global_only += local_n_global
                    # progress_callback already called per-frame by workers

    else:
        # ── Sequential path ────────────────────────────────────────────────────
        for i, (frame, idx) in enumerate(zip(selected_frames, selected_indices)):
            idx = int(idx)

            # ── Global alignment ──────────────────────────────────────────
            dx_g, dy_g = limb_center_align(disk_cx, disk_cy, frame)
            max_g = disk_radius * 0.5
            if abs(dx_g) > max_g or abs(dy_g) > max_g:
                dx_g, dy_g = subpixel_align(reference, frame)
                align_method = "phase_correlate"
            else:
                align_method = "limb_center"

            frame_aligned = apply_shift(frame, dx_g, dy_g)

            # ── AP warp maps ──────────────────────────────────────────────
            if adaptive_mode:
                map_dx, map_dy, n_good = _compute_adaptive_warp_maps(
                    frame_aligned, reference, ap_positions, cfg
                )
            else:
                map_dx, map_dy, n_good = _compute_warp_maps(
                    frame_aligned, reference, ap_positions, hann2d, query_pts, cfg
                )
            if n_good < 3:
                n_global_only += 1

            # ── Combined remap ────────────────────────────────────────────
            remap_x = (xx_base + map_dx - dx_g).astype(np.float32)
            remap_y = (yy_base + map_dy - dy_g).astype(np.float32)
            warped = cv2.remap(
                frame, remap_x, remap_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            warped = np.clip(warped, 0.0, 1.0)

            # ── Weighted accumulate ───────────────────────────────────────
            quality_w = max(float(scores[idx]) ** cfg.quality_weight_power, 1e-9)
            accum      += warped.astype(np.float64) * quality_w
            weight_sum += quality_w

            frame_logs.append({
                "frame_idx":       idx,
                "quality_score":   round(float(scores[idx]), 6),
                "global_shift_px": [round(float(dx_g), 3), round(float(dy_g), 3)],
                "align_method":    align_method,
                "n_good_aps":      n_good,
            })

            if progress_callback is not None and i % 50 == 0:
                progress_callback(i, n_selected)

    # Normalise
    stacked = np.where(weight_sum > 1e-12, accum / weight_sum, 0.0).astype(np.float32)
    stacked = np.clip(stacked, 0.0, 1.0)

    # Diagnostic: AP acceptance rate
    if frame_logs:
        avg_n_good = float(np.mean([f["n_good_aps"] for f in frame_logs]))
        n_total_aps = len(ap_positions)
        pct = 100.0 * avg_n_good / max(n_total_aps, 1)
        print(f"\n  AP acceptance: {avg_n_good:.0f}/{n_total_aps} ({pct:.0f}%)", flush=True)

    stats = {
        "n_stacked":           n_selected,
        "n_global_only_frames": n_global_only,
        "n_aps":               len(ap_positions),
        "disk_center_px":      [round(disk_cx, 2), round(disk_cy, 2)],
        "disk_radius_px":      round(disk_radius, 2),
        "frames":              frame_logs,
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
    n_iter = max(1, getattr(cfg, "n_iterations", 1))

    # ── Phase-aware progress mapping ──────────────────────────────────────────
    # Normalise the whole file to _PU (progress units) so callers always see a
    # consistent (done, _PU) denominator regardless of frame count or n_iter.
    # Scoring: 0 → _SCORE_END | Preload: _SCORE_END → _STACK_START |
    # Each stacking iteration: _STACK_START + i*_IT_PU → + (i+1)*_IT_PU
    _PU         = 1000
    _SCORE_END  = 150
    _STACK_START= 200
    _IT_PU      = (_PU - _STACK_START) // n_iter   # units per stacking iteration

    def _pu(v: int) -> None:
        """Emit normalised progress to the external callback."""
        if progress_callback is not None:
            progress_callback(v, _PU)

    with SERReader(ser_path) as reader:
        n_frames: int = reader.header["FrameCount"]
        print(f"\n  SER: {ser_path.name}  ({n_frames} frames)", flush=True)

        if n_frames < cfg.min_frames:
            raise RuntimeError(
                f"SER has only {n_frames} frames (min_frames={cfg.min_frames}). "
                "Lower min_frames or use a longer capture."
            )

        # ── Phase 1: Quality scoring (once for all iterations) ───────────
        print("  [1/5] Scoring frames…", end="\r", flush=True)

        def _score_prog(done: int, total: int) -> None:
            _pu(int(_SCORE_END * done / max(total, 1)))

        scores = score_frames(reader, cfg, score_step=2, progress_callback=_score_prog)
        t1 = time.perf_counter()
        _pu(_SCORE_END)
        print(f"  [1/5] Scored {n_frames} frames          ({t1-t0:.1f}s)", flush=True)

        # ── Phase 2: Frame selection (once for all iterations) ───────────
        n_select = max(cfg.min_frames, int(n_frames * cfg.top_percent))
        n_select = min(n_select, n_frames)
        selected_indices = np.argsort(scores)[::-1][:n_select]
        print(
            f"  [2/5] Selected {n_select}/{n_frames} frames "
            f"({100.0*n_select/n_frames:.1f}%)",
            flush=True,
        )

        # ── Phase 3: Initial reference frame ─────────────────────────────
        print("  [3/5] Building reference…", end="\r", flush=True)
        reference, (disk_cx, disk_cy, disk_radius) = build_reference_frame(
            reader, scores, cfg
        )
        t2 = time.perf_counter()
        print(
            f"  [3/5] Reference built   ({t2-t1:.1f}s)  "
            f"disk=({disk_cx:.1f},{disk_cy:.1f}) r={disk_radius:.1f}px",
            flush=True,
        )

        # ── Phase 3.5: Pre-load selected frames (once, shared across iterations) ─
        n_workers_cfg = int(getattr(cfg, "n_workers", 0))
        n_workers_use = n_workers_cfg if n_workers_cfg > 0 else _mp.cpu_count()
        n_workers_use = max(1, n_workers_use)

        print(
            f"  [3.5] Pre-loading {n_select} frames"
            f"  (workers: {n_workers_use})…",
            end="\r", flush=True,
        )
        t_pre0 = time.perf_counter()
        selected_frames = np.stack([
            reader.get_frame(int(idx)).astype(np.float32) / 255.0
            for idx in selected_indices
        ])  # (N_select, H, W) float32 [0, 1]
        t_pre1 = time.perf_counter()
        _pu(_STACK_START)
        print(
            f"  [3.5] Pre-loaded {n_select} frames  "
            f"({selected_frames.nbytes / 1e6:.0f} MB, {t_pre1-t_pre0:.1f}s)",
            flush=True,
        )

        stacked: Optional[np.ndarray] = None
        stats: Dict = {}
        t_stack_total = 0.0

        for iteration in range(n_iter):
            iter_label = f"iter {iteration+1}/{n_iter}" if n_iter > 1 else ""

            # ── Phase 4: AP grid ──────────────────────────────────────────
            # On iteration > 0, the reference is the previous stacked result
            # (much higher SNR → more accurate AP shifts).
            use_adaptive = bool(getattr(cfg, "use_adaptive_ap", True))
            if use_adaptive:
                ap_positions = generate_adaptive_ap_grid(
                    disk_cx, disk_cy, disk_radius, reference, cfg
                )
                from collections import Counter as _Counter
                ap_size_info = _Counter(sz for _, _, sz in ap_positions)
                ap_desc = " ".join(
                    f"{sz}px×{ap_size_info[sz]}" for sz in sorted(ap_size_info)
                )
            else:
                ap_positions = generate_ap_grid(
                    disk_cx, disk_cy, disk_radius, reference, cfg
                )
                ap_desc = f"size={cfg.ap_size}px"
            t3 = time.perf_counter()
            print(
                f"  [4/5] AP grid: {len(ap_positions)} points  "
                f"({t3-t2:.2f}s)"
                + (f"  [{ap_desc}]" if ap_desc else "")
                + (f"  [{iter_label}]" if iter_label else ""),
                flush=True,
            )
            if len(ap_positions) < 4:
                print("  WARNING: Too few APs — only global alignment will be applied.")

            # ── Phase 5: Warp + stack ─────────────────────────────────────
            step_label = f"[5/5]" if n_iter == 1 else f"[{4+iteration+1}/{4+n_iter}]"
            print(f"  {step_label} Stacking {n_select} frames…", end="\r", flush=True)

            def _prog(done: int, total: int, _lbl=step_label, _it=iteration) -> None:
                pct = 100 * done // max(total, 1)
                print(f"  {_lbl} Stacking {done}/{total} ({pct}%)…", end="\r", flush=True)
                offset = _STACK_START + _it * _IT_PU
                _pu(offset + int(_IT_PU * done / max(total, 1)))

            stacked, stats = apply_warp_and_stack(
                selected_frames,
                selected_indices,
                scores,
                reference,
                disk_cx, disk_cy, disk_radius,
                ap_positions,
                cfg,
                n_workers=n_workers_use,
                progress_callback=_prog,
            )
            t4 = time.perf_counter()
            t_stack_total += t4 - t3
            print(
                f"\n  {step_label} Done  (stack {t4-t3:.1f}s"
                + (f"  [{iter_label}]" if iter_label else "") + ")",
                flush=True,
            )

            # Prepare next iteration: use stacked result as new reference.
            # Disk geometry (disk_cx, disk_cy, disk_radius) is unchanged because
            # the stack is aligned to the original reference coordinate system.
            if iteration < n_iter - 1:
                reference = stacked  # float32 [0,1]; already clipped

    # Post-stack sub-pixel smoothing (try05: suppresses INTER_LINEAR aliasing at L1).
    if cfg.stack_blur_sigma > 0.0:
        stacked = cv2.GaussianBlur(stacked, (0, 0), cfg.stack_blur_sigma)
        stacked = np.clip(stacked, 0.0, 1.0)

    t_total = time.perf_counter() - t0
    print(f"\n  Total: {t_total:.1f}s  ({n_iter} iteration{'s' if n_iter>1 else ''})", flush=True)

    log: Dict = {
        "ser_path": str(ser_path),
        "n_frames_total": n_frames,
        "n_frames_selected": n_select,
        "top_percent": cfg.top_percent,
        "n_iterations": n_iter,
        "ap_size": cfg.ap_size,
        "ap_step": cfg.ap_step,
        "ap_confidence_threshold": cfg.ap_confidence_threshold,
        "timing_s": {
            "scoring":   round(t1 - t0, 2),
            "reference": round(t2 - t1, 2),
            "stacking":  round(t_stack_total, 2),
            "total":     round(t_total, 2),
        },
        **stats,
    }
    return stacked, log
