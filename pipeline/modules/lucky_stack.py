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
from collections import defaultdict
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
from pipeline.modules.quality import laplacian_var, tenengrad, planet_mask
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

    metric = getattr(cfg, "score_metric", "laplacian")
    score_fn = tenengrad if metric == "gradient" else laplacian_var

    sampled_idx: List[int] = list(range(0, n_frames, score_step))
    sampled_scores: List[float] = []

    for i, idx in enumerate(sampled_idx):
        frame = reader.get_frame(idx).astype(np.float32) / 255.0
        sampled_scores.append(score_fn(frame, mask))
        if progress_callback is not None and i % 50 == 0:
            progress_callback(idx, n_frames)

    # Interpolate to all frame indices
    scores = np.interp(
        np.arange(n_frames, dtype=np.float32),
        np.array(sampled_idx, dtype=np.float32),
        np.array(sampled_scores, dtype=np.float32),
    ).astype(np.float32)

    return scores


# ── 1b. Local gradient quality scoring (AS!4 quality_gradient_local=True) ──────

def score_frames_local(
    reader: SERReader,
    ap_positions: List[Tuple],
    cfg: LuckyStackConfig,
    score_step: int = 2,
    progress_callback=None,
) -> np.ndarray:
    """Score frames using mean local Tenengrad at AP patch positions.

    Matches AS!4's quality_type=Gradient + quality_gradient_local=True:
    for each frame, extract a patch at each AP center and compute Tenengrad
    (mean squared Sobel gradient). Frame score = mean over all AP patches.

    This is far more sensitive than global-disk Laplacian because:
      - AP patches (64px) target specific surface features (belts, zones, GRS)
      - Local sharpness of those features directly reflects momentary seeing

    Args:
        ap_positions: list of (ax, ay) pairs or (ax, ay, ap_size) triples.
        cfg:          LuckyStackConfig — uses ap_size as fallback patch size.

    Returns:
        float32 array of length FrameCount; higher = sharper.
    """
    n_frames: int = reader.header["FrameCount"]
    H: int = reader.header["Height"]
    W: int = reader.header["Width"]
    default_half = cfg.ap_size // 2

    sampled_idx: List[int] = list(range(0, n_frames, score_step))
    sampled_scores: List[float] = []

    for i, idx in enumerate(sampled_idx):
        frame = reader.get_frame(idx).astype(np.float32) / 255.0
        patch_scores: List[float] = []
        for ap in ap_positions:
            ax, ay = int(ap[0]), int(ap[1])
            half = int(ap[2]) // 2 if len(ap) >= 3 else default_half
            if ay - half < 0 or ay + half > H or ax - half < 0 or ax + half > W:
                continue
            patch = frame[ay - half: ay + half, ax - half: ax + half]
            # Use MAX gradient magnitude within patch (CV ~6%) vs mean (CV ~1.4%).
            # Max captures the sharpest feature in each AP; seeing variations affect
            # peak gradient far more than average gradient, giving 4× more frame
            # discrimination than mean-based approaches.
            _ksize = int(getattr(cfg, "quality_gradient_ksize", 3))
            # try55: Noise Robust pre-scoring blur — blur ONLY for quality scoring,
            # not for stacking. Matches AS!4's noise_robust parameter (NR=3 → σ=1.5px).
            _nr = int(getattr(cfg, "quality_noise_robust", 0))
            if _nr > 0:
                _sigma = _nr * 0.5
                patch = cv2.GaussianBlur(patch, (0, 0), _sigma)
            gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=_ksize)
            gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=_ksize)
            patch_scores.append(float((gx ** 2 + gy ** 2).max()))
        sampled_scores.append(float(np.mean(patch_scores)) if patch_scores else 0.0)
        if progress_callback is not None and i % 50 == 0:
            progress_callback(idx, n_frames)

    scores = np.interp(
        np.arange(n_frames, dtype=np.float32),
        np.array(sampled_idx, dtype=np.float32),
        np.array(sampled_scores, dtype=np.float32),
    ).astype(np.float32)

    return scores


# ── 1c. LoG disk quality scoring (AS!4 "lapl3" 방식) ─────────────────────────

def score_frames_log_disk(
    reader: SERReader,
    cfg: LuckyStackConfig,
    score_step: int = 2,
    progress_callback=None,
) -> np.ndarray:
    """Laplacian of Gaussian variance on planet disk — AS!4 'lapl3' 방식.

    AS!4가 내부적으로 사용하는 Laplacian 품질 지표를 역공학:
      1. 프레임 정규화 (float32 [0,1])
      2. GaussianBlur(sigma=log_disk_sigma) 적용
      3. Laplacian 계산
      4. 디스크 마스크(brightness > log_disk_threshold) 내 variance 반환

    score_correlation.py 분석 결과:
      - Spearman(AS!4) = 0.74  vs  local_gradient = 0.006
      - 최적 파라미터: sigma=3.0, threshold=0.25

    Returns:
        float32 array of length FrameCount; higher = sharper.
    """
    n_frames: int = reader.header["FrameCount"]
    sigma: float = float(getattr(cfg, "log_disk_sigma", 3.0))
    thr: float   = float(getattr(cfg, "log_disk_threshold", 0.25))

    sampled_idx: List[int] = list(range(0, n_frames, score_step))
    sampled_scores: List[float] = []

    for i, idx in enumerate(sampled_idx):
        raw = reader.get_frame(idx)
        f = raw.astype(np.float32) / raw.max()
        mask = f > thr
        if mask.sum() < 50:
            sampled_scores.append(0.0)
        else:
            blurred = cv2.GaussianBlur(f, (0, 0), sigma)
            lap = cv2.Laplacian(blurred, cv2.CV_32F, ksize=3)
            sampled_scores.append(float(lap[mask].var()))

        if progress_callback is not None and i % 50 == 0:
            progress_callback(idx, n_frames)

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
    ref_pct = float(getattr(cfg, "reference_percent", 0.0))
    if ref_pct > 0.0:
        n = max(1, min(int(len(scores) * ref_pct), len(scores)))
    else:
        n = min(int(cfg.reference_n_frames), len(scores))
    midpoint_pct = int(getattr(cfg, "reference_midpoint_percentage", 0))

    sorted_desc = np.argsort(scores)[::-1]  # best → worst

    if midpoint_pct <= 0:
        # Default: top n frames (highest quality)
        best_indices = sorted_desc[:n]
    else:
        # Centre the window at the frame at the given percentile from the bottom.
        # midpoint_pct=75 → 75th percentile from bottom = 25th percentile from top.
        total = len(scores)
        center_rank = int(round(total * (1.0 - midpoint_pct / 100.0)))  # rank in desc order
        half_n = n // 2
        lo = max(0, center_rank - half_n)
        hi = min(total, lo + n)
        lo = max(0, hi - n)  # clamp lo if hi hit the end
        best_indices = sorted_desc[lo:hi]

    # Sort best_indices by file position for sequential I/O, then process.
    # Mean is commutative — order of accumulation does not affect the result.
    _ref_sort = np.argsort(best_indices)
    best_indices_seq = best_indices[_ref_sort]

    best_idx = int(best_indices_seq[0])
    best_frame = reader.get_frame(best_idx).astype(np.float32) / 255.0

    accum = best_frame.astype(np.float64)
    for idx in best_indices_seq[1:]:
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


# ── 3b. Double AP grid (AS!4 double_ap_grid: s + s×1.5 + s×3 layers) ──────────

def generate_double_ap_grid(
    disk_cx: float,
    disk_cy: float,
    disk_radius: float,
    reference: np.ndarray,
    cfg,
) -> List[Tuple[int, int, int]]:
    """Generate AS!4-style multi-scale AP grid (double_ap_grid).

    Three layers scaled from cfg.ap_size (s):
      Layer 1 (s    px) : uniform grid at ap_step spacing
      Layer 2 (s×1.5px) : sparser grid at 2×ap_step spacing
      Layer 3 (s×3  px) : single center AP anchor

    For cfg.ap_size=64: 64 + 96 + 192 px  (matches AS!4 default)
    For cfg.ap_size=32: 32 + 48 + 96  px

    Same min_contrast / min_brightness filters as generate_ap_grid.
    Returns (ax, ay, ap_size) triples → uses adaptive warp path (KR sigma=ap_kr_sigma).
    """
    H, W = reference.shape[:2]
    min_bright   = getattr(cfg, "ap_min_brightness", 0.0)
    min_contrast = cfg.ap_min_contrast
    base_step    = cfg.ap_step
    s            = cfg.ap_size

    sz1 = s                          # e.g. 64
    sz2 = int(round(s * 1.5 / 8)) * 8  # e.g. 96  (rounded to 8px)
    sz3 = s * 3                      # e.g. 192

    aps: List[Tuple[int, int, int]] = []

    def _add_layer(ap_size: int, step: int) -> None:
        half = ap_size // 2
        for ay in range(half, H - half, step):
            for ax in range(half, W - half, step):
                dist = np.sqrt((ax - disk_cx) ** 2 + (ay - disk_cy) ** 2)
                if dist >= disk_radius:
                    continue
                patch = reference[ay - half : ay + half, ax - half : ax + half]
                if float(patch.std()) < min_contrast:
                    continue
                if min_bright > 0.0 and float(patch.mean()) < min_bright:
                    continue
                aps.append((ax, ay, ap_size))

    # Layer 1: s px, base_step
    _add_layer(sz1, base_step)

    # Layer 2: s×1.5 px, 2× base_step
    _add_layer(sz2, base_step * 2)

    # Layer 3: s×3 px center anchor (single AP)
    cx_i, cy_i = int(round(disk_cx)), int(round(disk_cy))
    half3 = sz3 // 2
    if (cy_i - half3 >= 0 and cy_i + half3 <= H and
            cx_i - half3 >= 0 and cx_i + half3 <= W):
        patch = reference[cy_i - half3 : cy_i + half3, cx_i - half3 : cx_i + half3]
        if float(patch.std()) >= min_contrast:
            aps.append((cx_i, cy_i, sz3))

    return aps


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


def generate_multiscale_ap_grid(
    disk_cx: float,
    disk_cy: float,
    disk_radius: float,
    reference: np.ndarray,
    cfg,
) -> List[Tuple[int, int, int]]:
    """Minimum-sufficient-size multi-scale AP grid (try64).

    For each candidate on a dense grid (spacing = ap_size // 2), find the
    smallest AP size that meets the contrast threshold.  Larger APs are placed
    only where smaller ones lack enough signal (smooth bright zones).
    Different scales can coexist at nearby positions; no explicit NMS is needed
    because the candidate grid spacing already prevents same-scale redundancy.

    Size progression: ap_size, ap_size*2, ap_size*4, ... up to disk_radius.
    """
    H, W = reference.shape[:2]
    min_ap          = cfg.ap_size
    min_contrast    = cfg.ap_min_contrast
    min_brightness  = getattr(cfg, "ap_min_brightness", 0.196)

    candidate_step = max(min_ap // 2, 8)

    # AP size candidates: doubling from min_ap, capped at disk_radius
    ap_sizes: List[int] = []
    sz = min_ap
    while sz <= int(disk_radius):
        ap_sizes.append(sz)
        sz *= 2
    if not ap_sizes:
        ap_sizes = [min_ap]

    min_half = min_ap // 2
    result: List[Tuple[int, int, int]] = []

    for ay in range(min_half, H - min_half, candidate_step):
        for ax in range(min_half, W - min_half, candidate_step):
            dist = np.sqrt((ax - disk_cx) ** 2 + (ay - disk_cy) ** 2)
            if dist >= disk_radius:
                continue

            for sz in ap_sizes:
                half = sz // 2
                y0, y1 = ay - half, ay + half
                x0, x1 = ax - half, ax + half
                if y0 < 0 or y1 > H or x0 < 0 or x1 > W:
                    break  # larger sizes also won't fit
                patch = reference[y0:y1, x0:x1]
                if float(patch.mean()) < min_brightness:
                    break  # dark region — larger patch won't help
                if float(patch.std()) >= min_contrast:
                    result.append((ax, ay, sz))
                    break  # found minimum sufficient size

    return result


# ── 3d. AS!4 greedy Poisson Disk Sampling AP grid ─────────────────────────────

def generate_as4_ap_grid(
    disk_cx: float,
    disk_cy: float,
    disk_radius: float,
    reference: np.ndarray,
    cfg,
) -> List[Tuple[int, int, int]]:
    """AS!4 greedy Poisson Disk Sampling AP grid (raster scan, reverse-engineered).

    Three independent layers: s, round(s×1.5/8)×8, s×3.
    Each layer scans top→bottom, left→right; a candidate is accepted when:
      1. centre inside disk  (dist to (disk_cx, disk_cy) < disk_radius)
      2. patch mean >= ap_min_brightness  (≈ AS!4 Min Bright = 50/255 ≈ 0.196)
      3. distance to every already-selected AP in this layer >= min_dist
         where min_dist = round(ap_size × 35/64)

    Match rate vs AS!4 ground truth: 100% on 260323 & 260415; 96-100% on 260407.
    Returns (ax, ay, ap_size) triples — same format as generate_double_ap_grid.
    """
    H, W = reference.shape[:2]
    min_bright = getattr(cfg, "ap_min_brightness", 0.196)
    s = cfg.ap_size

    sz1 = s
    sz2 = int(round(s * 1.5 / 8)) * 8
    sz3 = s * 3

    aps: List[Tuple[int, int, int]] = []

    # Integral image for O(1) patch mean — avoids per-pixel numpy slice overhead
    integ = cv2.integral(reference.astype(np.float64))

    disk_r2 = disk_radius * disk_radius

    def _greedy_pds_layer(ap_size: int) -> None:
        half = ap_size // 2
        min_dist    = int(round(ap_size * 35 / 64))
        min_dist_sq = min_dist ** 2
        inv_area    = 1.0 / (ap_size * ap_size)
        cell        = min_dist  # grid cell size = min_dist guarantees ≤3×3 cell check
        grid: Dict[Tuple[int, int], List[Tuple[int, int]]] = defaultdict(list)

        for ay in range(half, H - half):
            for ax in range(half, W - half):
                # Check 1: inside disk
                ddx = ax - disk_cx
                ddy = ay - disk_cy
                if ddx * ddx + ddy * ddy >= disk_r2:
                    continue
                # Check 2: brightness (integral image lookup)
                s_val = (integ[ay + half, ax + half]
                         - integ[ay - half, ax + half]
                         - integ[ay + half, ax - half]
                         + integ[ay - half, ax - half])
                if s_val * inv_area < min_bright:
                    continue
                # Check 3: min-dist — only inspect 3×3 neighbouring grid cells
                cx_cell = ax // cell
                cy_cell = ay // cell
                too_close = False
                for dcx in (-1, 0, 1):
                    for dcy in (-1, 0, 1):
                        for sx, sy in grid[(cx_cell + dcx, cy_cell + dcy)]:
                            if (ax - sx) * (ax - sx) + (ay - sy) * (ay - sy) < min_dist_sq:
                                too_close = True
                                break
                        if too_close:
                            break
                    if too_close:
                        break
                if not too_close:
                    grid[(cx_cell, cy_cell)].append((ax, ay))
                    aps.append((ax, ay, ap_size))

    _greedy_pds_layer(sz1)
    _greedy_pds_layer(sz2)
    _greedy_pds_layer(sz3)

    return aps


def compute_session_aps_from_ser(
    ser_path: Path,
    cfg,
) -> Tuple[List[Tuple[int, int, int]], float, float, float]:
    """Generate AS!4-style AP grid from the mid-frame of a reference SER.

    Returns (aps, disk_cx, disk_cy, disk_radius).
    Used by step02_lucky_stack to compute session-wide AP positions once.
    """
    with SERReader(ser_path) as reader:
        n = int(reader.header["FrameCount"])
        frame = reader.get_frame(n // 2).astype(np.float32) / 255.0
        if frame.ndim == 3:
            frame = frame.mean(axis=2).astype(np.float32)
    cx, cy, semi_a, _semi_b, _angle = find_disk_center(frame)
    aps = generate_as4_ap_grid(cx, cy, float(semi_a), frame, cfg)
    return aps, cx, cy, float(semi_a)


# ── 4. Per-AP shift estimation ─────────────────────────────────────────────────

def _make_hann2d(size: int) -> np.ndarray:
    """Pre-compute a 2-D Hann window of shape (size, size)."""
    h = np.hanning(size).astype(np.float32)
    return np.outer(h, h)


def _qsf_refine(cc: np.ndarray) -> Tuple[float, float, float]:
    """Sub-pixel peak via Quadratic Surface Fitting on 3×3 neighbourhood.

    Returns (dx, dy, peak_value) using the same sign convention as
    cv2.phaseCorrelate: positive (dx, dy) means frm is shifted right/down
    relative to ref.
    """
    H, W = cc.shape
    # Integer peak (with wrap-around periodicity)
    flat_idx = int(np.argmax(cc))
    py, px = divmod(flat_idx, W)

    # 3×3 neighbourhood with periodic wrap
    z = np.zeros((3, 3), dtype=np.float64)
    for di in range(-1, 2):
        for dj in range(-1, 2):
            z[di + 1, dj + 1] = cc[(py + di) % H, (px + dj) % W]

    # 1-D quadratic sub-pixel in x and y independently
    denom_x = z[1, 0] - 2.0 * z[1, 1] + z[1, 2]
    denom_y = z[0, 1] - 2.0 * z[1, 1] + z[2, 1]
    sub_x = -0.5 * (z[1, 2] - z[1, 0]) / (denom_x if abs(denom_x) > 1e-12 else 1e-12)
    sub_y = -0.5 * (z[2, 1] - z[0, 1]) / (denom_y if abs(denom_y) > 1e-12 else 1e-12)
    sub_x = float(np.clip(sub_x, -1.0, 1.0))
    sub_y = float(np.clip(sub_y, -1.0, 1.0))

    # Fractional peak position with wrap-around for negative shifts
    peak_x = float(px) + sub_x
    peak_y = float(py) + sub_y
    if peak_x > W / 2:
        peak_x -= W
    if peak_y > H / 2:
        peak_y -= H

    return peak_x, peak_y, float(cc[py, px])


def _ncc_shift(
    ref_patch: np.ndarray,
    frm_patch: np.ndarray,
    hann2d: np.ndarray,
    search_range: float,
) -> Tuple[Optional[float], Optional[float], float]:
    """AP shift via Normalized Cross-Correlation (NCC).

    NCC is robust where phase correlation fails:
    - uniform/featureless patches → low σ → low confidence (not random shift)
    - DC-offset differences between ref/frame → zeroed by mean subtraction

    Formula (FFT-based, O(N log N)):
        cc = IFFT(FFT(ref-μ) · conj(FFT(frm-μ))) / (N · σ_ref · σ_frm)
    Peak of cc ∈ [-1, 1]; sub-pixel position via QSF.

    Returns (dx, dy, confidence) or (None, None, 0.0).
    """
    # Zero-mean + Hann window
    ref_zm = (ref_patch - ref_patch.mean()) * hann2d
    frm_zm = (frm_patch - frm_patch.mean()) * hann2d

    sigma_ref = float(ref_zm.std())
    sigma_frm = float(frm_zm.std())
    N = ref_zm.size
    norm = N * sigma_ref * sigma_frm
    if norm < 1e-12:
        return None, None, 0.0

    F1 = np.fft.rfft2(ref_zm.astype(np.float64))
    F2 = np.fft.rfft2(frm_zm.astype(np.float64))
    # conj(F1)*F2 gives peak at +dx (same sign as phaseCorrelate convention).
    # F1*conj(F2) gives peak at -dx (opposite sign — do not use).
    cc = np.fft.irfft2(np.conj(F1) * F2, s=ref_zm.shape).astype(np.float32)
    cc /= norm

    dx, dy, peak = _qsf_refine(cc)

    # confidence = NCC peak, clamped to [0, 1]
    confidence = float(np.clip(peak, 0.0, 1.0))
    if abs(dx) > search_range or abs(dy) > search_range:
        return None, None, 0.0
    return dx, dy, confidence


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

    When cfg.use_qsf=True, replaces cv2.phaseCorrelate with manual
    normalized cross-power spectrum + quadratic surface fitting for
    sub-pixel accuracy (AS!4-style QSF).

    When cfg.use_ncc=True, uses NCC (see _ncc_shift above).

    Returns (dx, dy, confidence) or (None, None, 0.0) if rejected.
    """
    ref_w = ref_patch * hann2d
    frm_w = frm_patch * hann2d

    _use_ncc = bool(getattr(cfg, "use_ncc", False))
    _use_qsf = bool(getattr(cfg, "use_qsf", False))
    _use_pcc = bool(getattr(cfg, "use_pcc_upsample", False))

    if _use_ncc:
        dx, dy, confidence = _ncc_shift(
            ref_patch, frm_patch, hann2d, cfg.ap_search_range
        )
        # NCC peak ∈ [0,1] (true correlation coefficient).
        # ncc_confidence_threshold: explicit override; -1.0 = use ap_confidence_threshold.
        _ncc_thr_cfg = float(getattr(cfg, "ncc_confidence_threshold", -1.0))
        _ncc_thr = cfg.ap_confidence_threshold if _ncc_thr_cfg < 0.0 else _ncc_thr_cfg
        if dx is None or confidence < _ncc_thr:
            return None, None, 0.0
        return dx, dy, confidence

    if _use_pcc:
        # Gate with standard phaseCorrelate confidence (more reliable for rejection),
        # then refine shift with scikit-image DFT upsampling (0.1px precision).
        # PSS SubpixelRegistration equivalent (upsample_factor=10).
        (dx_pc, dy_pc), confidence = cv2.phaseCorrelate(ref_w, frm_w)
        confidence = float(confidence)
        if confidence >= cfg.ap_confidence_threshold and abs(dx_pc) <= cfg.ap_search_range and abs(dy_pc) <= cfg.ap_search_range:
            try:
                from skimage.registration import phase_cross_correlation as _pcc
                shift, _, _ = _pcc(
                    ref_w.astype(np.float64),
                    frm_w.astype(np.float64),
                    upsample_factor=10,
                )
                # skimage returns (row_shift, col_shift) = (dy, dx)
                dy, dx = float(shift[0]), float(shift[1])
            except Exception:
                dx, dy = float(dx_pc), float(dy_pc)
        else:
            dx, dy = float(dx_pc), float(dy_pc)
    elif _use_qsf:
        # Use cv2.phaseCorrelate for confidence gating (properly normalised [0,1]),
        # then QSF on the phase-correlation CC for sub-pixel accuracy.
        # This matches AS!4's approach: confidence gate on phaseCorrelate,
        # refine with QSF.  The previous pure-QSF confidence formula
        # (peak / (n * 0.01)) gave values ~0.012 for real data, always below
        # the 0.15 threshold — effectively disabling local warp for all experiments.
        (dx_pc, dy_pc), confidence = cv2.phaseCorrelate(ref_w, frm_w)
        confidence = float(confidence)
        if confidence >= cfg.ap_confidence_threshold:
            # QSF sub-pixel refinement on the phase-correlation CC
            F1 = np.fft.rfft2(ref_w.astype(np.float64))
            F2 = np.fft.rfft2(frm_w.astype(np.float64))
            cross = F1 * np.conj(F2)
            denom = np.abs(cross)
            denom[denom < 1e-12] = 1e-12
            cc = np.fft.irfft2(cross / denom, s=ref_w.shape)
            dx, dy, _ = _qsf_refine(cc)
        else:
            dx, dy = float(dx_pc), float(dy_pc)
    else:
        (dx, dy), confidence = cv2.phaseCorrelate(ref_w, frm_w)
        confidence = float(confidence)

    if confidence < cfg.ap_confidence_threshold:
        return None, None, 0.0
    if abs(dx) > cfg.ap_search_range or abs(dy) > cfg.ap_search_range:
        return None, None, 0.0

    return float(dx), float(dy), confidence


# ── 4a-opt. Pre-computation helpers for AP shift estimation ───────────────────

def _precompute_ap_ref_data(
    reference: np.ndarray,
    ap_positions: List,
    cfg: LuckyStackConfig,
) -> Optional[List]:
    """Pre-compute ref-patch-dependent data for AP shift estimation.

    reference is fixed for the entire stacking loop, so ref_patch extraction,
    Hann windowing, mean subtraction, std, and rfft2 are all constant per AP.
    Pre-computing them once eliminates N_frames × N_APs redundant operations.

    Returns a list (one entry per AP) of dicts, or None if the current config
    path does not benefit (pcc_upsample uses a black-box phaseCorrelate).
    Entries may be None for out-of-bounds APs.
    """
    _use_ncc = bool(getattr(cfg, "use_ncc", False))
    _use_qsf = bool(getattr(cfg, "use_qsf", False))
    if bool(getattr(cfg, "use_pcc_upsample", False)):
        return None  # skimage path: phaseCorrelate is a black box

    H, W = reference.shape[:2]
    precomp: List[Optional[Dict]] = []
    for ap in ap_positions:
        ax, ay = int(ap[0]), int(ap[1])
        ap_sz  = int(ap[2]) if len(ap) >= 3 else cfg.ap_size
        half   = ap_sz // 2
        if ay - half < 0 or ay + half > H or ax - half < 0 or ax + half > W:
            precomp.append(None)
            continue
        ref_patch = reference[ay - half: ay + half, ax - half: ax + half].astype(np.float32)
        hann      = _get_hann(ap_sz)
        if _use_ncc:
            ref_zm    = (ref_patch - ref_patch.mean()) * hann
            sigma_ref = float(ref_zm.std())
            F1        = np.fft.rfft2(ref_zm.astype(np.float64))
            precomp.append({"F1": F1, "sigma_ref": sigma_ref, "N": ref_zm.size})
        elif _use_qsf:
            ref_w  = ref_patch * hann
            F1_qsf = np.fft.rfft2(ref_w.astype(np.float64))
            precomp.append({"ref_w": ref_w, "F1": F1_qsf})
        else:
            precomp.append({"ref_w": ref_patch * hann})
    return precomp


def _estimate_ap_shift_precomp(
    frm_patch: np.ndarray,
    pc: Dict,
    hann2d: np.ndarray,
    cfg: LuckyStackConfig,
) -> Tuple[Optional[float], Optional[float], float]:
    """AP shift estimation using pre-computed ref patch data.

    Avoids re-extracting, re-windowing, and re-transforming the (constant)
    reference patch on every frame call.  pc must be a dict from
    _precompute_ap_ref_data; hann2d is the Hann window for the frame patch.
    """
    _use_ncc = bool(getattr(cfg, "use_ncc", False))
    _use_qsf = bool(getattr(cfg, "use_qsf", False))

    if _use_ncc:
        F1, sigma_ref, N = pc["F1"], pc["sigma_ref"], pc["N"]
        frm_zm    = (frm_patch - frm_patch.mean()) * hann2d
        sigma_frm = float(frm_zm.std())
        norm = N * sigma_ref * sigma_frm
        if norm < 1e-12:
            return None, None, 0.0
        F2 = np.fft.rfft2(frm_zm.astype(np.float64))
        cc = np.fft.irfft2(np.conj(F1) * F2, s=frm_zm.shape).astype(np.float32)
        cc /= norm
        dx, dy, peak = _qsf_refine(cc)
        confidence = float(np.clip(peak, 0.0, 1.0))
        if abs(dx) > cfg.ap_search_range or abs(dy) > cfg.ap_search_range:
            return None, None, 0.0
        _ncc_thr_cfg = float(getattr(cfg, "ncc_confidence_threshold", -1.0))
        _ncc_thr = cfg.ap_confidence_threshold if _ncc_thr_cfg < 0.0 else _ncc_thr_cfg
        if confidence < _ncc_thr:
            return None, None, 0.0
        return dx, dy, confidence

    ref_w = pc["ref_w"]
    frm_w = frm_patch * hann2d
    if _use_qsf:
        (dx_pc, dy_pc), confidence = cv2.phaseCorrelate(ref_w, frm_w)
        confidence = float(confidence)
        if confidence >= cfg.ap_confidence_threshold:
            F2    = np.fft.rfft2(frm_w.astype(np.float64))
            cross = pc["F1"] * np.conj(F2)
            denom = np.abs(cross)
            denom[denom < 1e-12] = 1e-12
            cc    = np.fft.irfft2(cross / denom, s=frm_w.shape)
            dx, dy, _ = _qsf_refine(cc)
        else:
            dx, dy = float(dx_pc), float(dy_pc)
    else:
        (dx, dy), confidence = cv2.phaseCorrelate(ref_w, frm_w)
        confidence = float(confidence)

    if confidence < cfg.ap_confidence_threshold:
        return None, None, 0.0
    if abs(dx) > cfg.ap_search_range or abs(dy) > cfg.ap_search_range:
        return None, None, 0.0
    return float(dx), float(dy), confidence


# ── 4b. try54: CoG (Centre-of-Gravity) global alignment ───────────────────────

def _cog_center_align(
    ref_cx: float,
    ref_cy: float,
    target: np.ndarray,
    max_shift_px: float = 15.0,
    fixed_threshold: int = 0,
) -> Tuple[float, float]:
    """Global stabilization via brightness-weighted centroid (CoG).

    AS!4 'Planet CoG' mode: uses cv2.moments() on the thresholded disk instead
    of ellipse fitting.  More robust when the limb is noisy or partially clipped,
    because the centroid uses ALL disk pixels rather than just the limb contour.

    Falls back to (0, 0) if centroid detection fails or gives an implausibly
    large shift (> max_shift_px).

    Args:
        ref_cx, ref_cy:  Reference frame disk center (pixels).
        target:          2-D float [0, 1] luminance of the current frame.
        max_shift_px:    Clamp for detection failures.
        fixed_threshold: Brightness threshold (0 = Otsu).
    """
    try:
        img_u8 = (np.clip(target, 0.0, 1.0) * 255.0).astype(np.uint8)
        if fixed_threshold > 0:
            _, binary = cv2.threshold(img_u8, fixed_threshold, 255, cv2.THRESH_BINARY)
        else:
            _, binary = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        M = cv2.moments(binary)
        if M["m00"] < 1.0:
            return 0.0, 0.0
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        dx = float(ref_cx - cx)
        dy = float(ref_cy - cy)
        if abs(dx) > max_shift_px or abs(dy) > max_shift_px:
            return 0.0, 0.0
        return dx, dy
    except Exception:
        return 0.0, 0.0


# ── 4c. try56: Per-AP quality weight map builder ───────────────────────────────

def _build_per_ap_quality_map(
    warped_frame: np.ndarray,
    ap_positions: List,
    cfg: LuckyStackConfig,
) -> np.ndarray:
    """Compute a 2-D quality weight map using per-AP local gradient scores.

    For each AP position, evaluate the max Sobel gradient in the patch of the
    WARPED frame (already aligned), then use Gaussian KR to interpolate to
    full resolution.  The result is used as a spatially varying accumulation
    weight instead of a single per-frame global quality scalar.

    Returns:
        float64 [H, W] weight map (unnormalized; caller accumulates weight_sum).
    """
    H, W = warped_frame.shape[:2]
    _ksize = int(getattr(cfg, "quality_gradient_ksize", 3))
    _nr    = int(getattr(cfg, "quality_noise_robust", 0))
    _power = float(cfg.quality_weight_power)

    q_map = np.zeros((H, W), dtype=np.float32)
    w_map = np.ones((H, W), dtype=np.float32)   # uniform 1.0 at each AP to normalize

    for ap in ap_positions:
        ax, ay = int(ap[0]), int(ap[1])
        half = int(ap[2]) // 2 if len(ap) >= 3 else cfg.ap_size // 2
        if ay - half < 0 or ay + half > H or ax - half < 0 or ax + half > W:
            continue
        patch = warped_frame[ay - half: ay + half, ax - half: ax + half]
        if _nr > 0:
            patch = cv2.GaussianBlur(patch, (0, 0), _nr * 0.5)
        gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=_ksize)
        gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=_ksize)
        q = float((gx ** 2 + gy ** 2).max()) ** _power
        q_map[ay, ax] = q

    # Gaussian KR: interpolate per-AP quality to full resolution
    sigma  = float(cfg.ap_step) * cfg.ap_sigma_factor
    ksize  = int(6.0 * sigma + 1) | 1
    smooth_qw = cv2.GaussianBlur(q_map * w_map, (ksize, ksize), sigma)
    smooth_w  = cv2.GaussianBlur(w_map,          (ksize, ksize), sigma)

    cov_thresh = float(np.max(smooth_w)) * 0.05
    cov_ok = smooth_w >= cov_thresh
    result = np.where(cov_ok, smooth_qw / np.maximum(smooth_w, 1e-9), 0.0)

    # Pixels outside all AP coverage: fall back to global median quality
    if not np.any(cov_ok):
        return np.full((H, W), 1e-9, dtype=np.float64)
    fallback = float(np.median(result[cov_ok]))
    result = np.where(cov_ok, result, max(fallback, 1e-9))

    return result.astype(np.float64)


# ── 5. Per-frame warp + remap ──────────────────────────────────────────────────

def _compute_warp_maps(
    frame_aligned: np.ndarray,
    reference: np.ndarray,
    ap_positions: List[Tuple[int, int]],
    hann2d: np.ndarray,
    query_pts: np.ndarray,    # kept for API compatibility; not used
    cfg: LuckyStackConfig,
    ref_precomp: Optional[List] = None,
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
    for i, (ax, ay) in enumerate(ap_positions):
        frm_patch = frame_aligned[ay - half : ay + half, ax - half : ax + half].astype(np.float32)

        if ref_precomp is not None and i < len(ref_precomp) and ref_precomp[i] is not None:
            dx, dy, conf = _estimate_ap_shift_precomp(frm_patch, ref_precomp[i], hann2d, cfg)
        else:
            ref_patch = reference[ay - half : ay + half, ax - half : ax + half].astype(np.float32)
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
        return zero, zero, n_good, zero

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

    # Per-pixel NCC confidence map: smooth_w normalized to [0,1].
    # Pixels inside well-covered AP regions → high; outside coverage → 0.
    # Used as per-pixel stacking weight when cfg.use_ncc=True.
    peak_w = float(np.max(smooth_w))
    conf_map = np.where(coverage_ok, smooth_w / max(peak_w, 1e-9), 0.0).astype(np.float32)

    return map_dx, map_dy, n_good, conf_map


# ── 5a. Adaptive warp maps (variable AP sizes + wide KR) ──────────────────────

def _compute_adaptive_warp_maps(
    frame_aligned: np.ndarray,
    reference: np.ndarray,
    ap_positions: List[Tuple[int, int, int]],
    cfg,
    ref_precomp: Optional[List] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Adaptive-AP version of Gaussian kernel regression warp maps.

    Uses fixed kr_sigma=64 Gaussian kernel regression. Each AP measurement
    is placed as a weighted point, then blurred with a global Gaussian to
    interpolate across the full disk.

    Args:
        ap_positions: list of (ax, ay, ap_size) triples.
        cfg:          LuckyStackConfig with ap_confidence_threshold,
                      ap_search_range fields.

    Returns:
        (map_dx, map_dy, n_good_aps)
    """
    H, W = frame_aligned.shape[:2]
    kr_sigma = float(getattr(cfg, "ap_kr_sigma", 64.0))

    shift_x = np.zeros((H, W), dtype=np.float32)
    shift_y = np.zeros((H, W), dtype=np.float32)
    weight  = np.zeros((H, W), dtype=np.float32)
    n_good  = 0

    for i, (ax, ay, ap_size) in enumerate(ap_positions):
        half = ap_size // 2
        y0, y1 = ay - half, ay + half
        x0, x1 = ax - half, ax + half
        if y0 < 0 or y1 > H or x0 < 0 or x1 > W:
            continue

        frm_patch = frame_aligned[y0:y1, x0:x1].astype(np.float32)
        hann      = _get_hann(ap_size)

        if ref_precomp is not None and i < len(ref_precomp) and ref_precomp[i] is not None:
            dx, dy, conf = _estimate_ap_shift_precomp(frm_patch, ref_precomp[i], hann, cfg)
        else:
            ref_patch = reference[y0:y1, x0:x1].astype(np.float32)
            dx, dy, conf = _estimate_ap_shift(ref_patch, frm_patch, hann, cfg)
        if dx is None:
            continue

        lap   = cv2.Laplacian(frm_patch, cv2.CV_32F)
        sharp = float(np.var(lap))
        w     = conf * float(np.log1p(sharp))

        shift_x[ay, ax] = float(dx) * w
        shift_y[ay, ax] = float(dy) * w
        weight[ay, ax]  = w
        n_good += 1

    if n_good < 3:
        zero = np.zeros((H, W), dtype=np.float32)
        return zero, zero, n_good, zero

    ksize  = int(6.0 * kr_sigma + 1) | 1
    sw_x   = cv2.GaussianBlur(shift_x * weight, (ksize, ksize), kr_sigma)
    sw_y   = cv2.GaussianBlur(shift_y * weight, (ksize, ksize), kr_sigma)
    sw     = cv2.GaussianBlur(weight,            (ksize, ksize), kr_sigma)
    cov_ok = sw >= float(np.max(sw)) * 0.05
    map_dx = np.where(cov_ok, sw_x / np.maximum(sw, 1e-9), 0.0).astype(np.float32)
    map_dy = np.where(cov_ok, sw_y / np.maximum(sw, 1e-9), 0.0).astype(np.float32)
    peak_sw = float(np.max(sw))
    conf_map = np.where(cov_ok, sw / max(peak_sw, 1e-9), 0.0).astype(np.float32)
    return map_dx, map_dy, n_good, conf_map


# ── 5b. TPS warp maps (try63: exact shift interpolation, no KR dilution) ──────

def _compute_warp_maps_tps(
    frame_aligned: np.ndarray,
    reference: np.ndarray,
    ap_positions: List,
    hann2d,          # pre-built Hann window (2-tuple path) or None (3-tuple path)
    cfg: LuckyStackConfig,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Thin Plate Spline warp maps — exact interpolation, no KR dilution.

    KR (Gaussian kernel regression) smooths AP shifts with a sigma=64px kernel,
    so a 3px local correction gets diluted to ~1-2px at the AP position.
    TPS passes EXACTLY through every reliable AP measurement: the 3px correction
    is applied as 3px, with C2-smooth variation between APs.

    This is the most likely interpolation approach used by AS!4 (no smoothing
    parameter exists in the AS!4 UI, consistent with exact interpolation).

    Performance: evaluates TPS on a coarse grid (ap_step//4 spacing) then
    bicubic-upsamples to full resolution — TPS is smooth so this loses nothing.

    Coverage mask: same Gaussian density map as KR, preventing wild TPS
    extrapolation in the sky/border region outside the AP convex hull.

    Args:
        ap_positions: (ax, ay) 2-tuples or (ax, ay, ap_size) 3-tuples.
        hann2d:       Hann window for 2-tuple path; None for 3-tuple (per-AP).

    Returns:
        (map_dx, map_dy, n_good_aps)
    """
    from scipy.interpolate import RBFInterpolator

    H, W = frame_aligned.shape[:2]
    adaptive = bool(ap_positions) and len(ap_positions[0]) == 3

    good_yx: List[List[float]] = []
    good_dx: List[float] = []
    good_dy: List[float] = []
    weight = np.zeros((H, W), dtype=np.float32)   # AP density for coverage mask

    for ap in ap_positions:
        ax, ay = int(ap[0]), int(ap[1])
        ap_sz  = int(ap[2]) if adaptive else cfg.ap_size
        half   = ap_sz // 2
        if ay - half < 0 or ay + half > H or ax - half < 0 or ax + half > W:
            continue

        ref_patch = reference[ay - half: ay + half, ax - half: ax + half].astype(np.float32)
        frm_patch = frame_aligned[ay - half: ay + half, ax - half: ax + half].astype(np.float32)
        _hann = _get_hann(ap_sz) if adaptive else hann2d

        dx, dy, conf = _estimate_ap_shift(ref_patch, frm_patch, _hann, cfg)
        if dx is None:
            continue

        good_yx.append([float(ay), float(ax)])
        good_dx.append(float(dx))
        good_dy.append(float(dy))
        weight[ay, ax] = conf

    n_good = len(good_dx)
    zero = np.zeros((H, W), dtype=np.float32)
    if n_good < 3:
        return zero, zero, n_good, zero

    pts = np.array(good_yx, dtype=np.float64)
    _smoothing = float(getattr(cfg, "tps_smoothing", 0.0))

    tps_dx = RBFInterpolator(pts, np.array(good_dx, dtype=np.float64),
                             kernel="thin_plate_spline", smoothing=_smoothing)
    tps_dy = RBFInterpolator(pts, np.array(good_dy, dtype=np.float64),
                             kernel="thin_plate_spline", smoothing=_smoothing)

    # Coarse grid evaluation → bicubic upsample (TPS is C2-smooth → no loss).
    # step = ap_step // 4 = 8px for default ap_step=32.
    _cstep = max(4, cfg.ap_step // 4)
    ys_c = np.arange(0, H, _cstep, dtype=np.float64)
    xs_c = np.arange(0, W, _cstep, dtype=np.float64)
    gx_c, gy_c = np.meshgrid(xs_c, ys_c)
    query = np.column_stack([gy_c.ravel(), gx_c.ravel()])

    coarse_dx = tps_dx(query).reshape(gy_c.shape).astype(np.float32)
    coarse_dy = tps_dy(query).reshape(gy_c.shape).astype(np.float32)

    map_dx = cv2.resize(coarse_dx, (W, H), interpolation=cv2.INTER_CUBIC)
    map_dy = cv2.resize(coarse_dy, (W, H), interpolation=cv2.INTER_CUBIC)

    # Coverage mask: zero out sky/border (same Gaussian density as KR).
    sigma  = float(cfg.ap_step) * cfg.ap_sigma_factor
    ksize  = int(6.0 * sigma + 1) | 1
    smooth_w = cv2.GaussianBlur(weight, (ksize, ksize), sigma)
    cov_ok   = smooth_w >= float(np.max(smooth_w)) * 0.05

    map_dx = np.where(cov_ok, map_dx, 0.0).astype(np.float32)
    map_dy = np.where(cov_ok, map_dy, 0.0).astype(np.float32)
    # TPS does not have a natural NCC confidence; return zeros (no conf-weighting).
    conf_map = np.zeros((H, W), dtype=np.float32)
    return map_dx, map_dy, n_good, conf_map


# ── 5c. True per-AP independent stacking (try68) ─────────────────────────────

def _spatial_per_ap_quality_stack(
    selected_frames: np.ndarray,
    selected_indices: np.ndarray,
    scores: np.ndarray,
    reference: np.ndarray,
    disk_cx: float,
    disk_cy: float,
    disk_radius: float,
    ap_positions: List,
    cfg,
    progress_callback=None,
) -> Tuple[np.ndarray, Dict]:
    """Per-AP quality-weighted stacking without patch boundaries.

    Streaming (single-pass) algorithm:
      For each frame:
        1. Sub-pixel global alignment.
        2. Per-AP Sobel quality score on the globally-aligned frame.
        3. Build smooth spatial quality weight map W_f via Gaussian KR
           (same sigma as the warp-field KR) — no hard patch boundaries.
        4. Compute full-frame KR warp → warped frame.
        5. Accumulate: accum += warped × W_f, weight += W_f.

    Why this eliminates wavelet grid artifacts:
      The patch-based approach (_per_ap_independent_stack) selects different
      frame subsets per AP. Adjacent APs can have slightly different mean
      brightness, creating a subtle grid that wavelet amplifies ×200.
      This function uses full KR-warped frames with spatially-varying weights,
      so every pixel is a smooth blend of all frames — no subset boundaries.

    Quality computed on the globally-aligned frame (before KR warp) gives a
    more authentic per-AP sharpness signal than computing on the warped frame,
    because the KR warp homogenises apparent quality across APs.
    """
    H, W = reference.shape[:2]
    n_sel = len(selected_frames)
    n_ap  = len(ap_positions)
    _ksize       = int(getattr(cfg, "quality_gradient_ksize", 3))
    # per_ap_quality_power: separate from global quality_weight_power.
    # Higher power → sharper per-AP selectivity (3–4 recommended).
    _per_ap_pow  = float(getattr(cfg, "per_ap_quality_power", 3.0))
    _stab_thresh = int(getattr(cfg, "stabilization_planet_threshold", 0))
    _interp      = getattr(cfg, "remap_interpolation", cv2.INTER_LINEAR)
    adaptive_mode = bool(ap_positions) and len(ap_positions[0]) == 3
    hann2d    = None if adaptive_mode else _make_hann2d(cfg.ap_size)
    xx_base   = np.tile(np.arange(W, dtype=np.float32), (H, 1))
    yy_base   = np.tile(np.arange(H, dtype=np.float32)[:, None], (1, W))
    query_pts = np.empty((0, 2), dtype=np.float64)

    # Pre-compute KR normalisation for the quality weight map.
    # G_sum[y,x] = Gaussian-weighted count of nearby APs — same sigma as warp KR.
    sigma_kr = float(cfg.ap_step) * cfg.ap_sigma_factor
    ksize_kr = int(6.0 * sigma_kr + 1) | 1
    g_ind = np.zeros((H, W), dtype=np.float32)
    for ap in ap_positions:
        ax, ay = int(ap[0]), int(ap[1])
        if 0 <= ay < H and 0 <= ax < W:
            g_ind[ay, ax] = 1.0
    G_sum      = cv2.GaussianBlur(g_ind, (ksize_kr, ksize_kr), sigma_kr)
    cov_thresh = float(G_sum.max()) * 0.05
    coverage_ok = G_sum >= cov_thresh
    G_sum_safe  = np.where(coverage_ok, G_sum, 1.0).astype(np.float64)

    accum      = np.zeros((H, W), dtype=np.float64)
    weight_sum = np.zeros((H, W), dtype=np.float64)
    frame_logs: List[Dict] = []
    n_global_only = 0

    for i, (frame, idx) in enumerate(zip(selected_frames, selected_indices)):
        idx = int(idx)

        # ── Global alignment ──────────────────────────────────────────────────
        dx_g, dy_g = limb_center_align(
            disk_cx, disk_cy, frame, fixed_threshold=_stab_thresh
        )
        if abs(dx_g) > disk_radius * 0.5 or abs(dy_g) > disk_radius * 0.5:
            dx_g, dy_g = subpixel_align(reference, frame)
            align_method = "phase_correlate"
        else:
            align_method = "limb_center"
        frame_aligned = apply_shift(frame, dx_g, dy_g)

        # ── Per-AP Sobel quality on globally-aligned frame ────────────────────
        q_map = np.zeros((H, W), dtype=np.float32)
        for ap in ap_positions:
            ax, ay = int(ap[0]), int(ap[1])
            half = (int(ap[2]) if len(ap) >= 3 else cfg.ap_size) // 2
            if ay - half < 0 or ay + half > H or ax - half < 0 or ax + half > W:
                continue
            patch = frame_aligned[ay - half: ay + half, ax - half: ax + half]
            gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=_ksize)
            gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=_ksize)
            q = float((gx ** 2 + gy ** 2).max()) ** _per_ap_pow
            q_map[ay, ax] = q

        # Gaussian KR → smooth spatial quality weight map for this frame
        smooth_qw = cv2.GaussianBlur(q_map, (ksize_kr, ksize_kr), sigma_kr)
        W_f = np.where(coverage_ok,
                       smooth_qw.astype(np.float64) / G_sum_safe,
                       0.0)

        # ── Full-frame KR warp ────────────────────────────────────────────────
        if adaptive_mode:
            map_dx, map_dy, n_good, _ = _compute_adaptive_warp_maps(
                frame_aligned, reference, ap_positions, cfg
            )
        else:
            map_dx, map_dy, n_good, _ = _compute_warp_maps(
                frame_aligned, reference, ap_positions, hann2d, query_pts, cfg
            )

        if n_good < 3:
            n_global_only += 1
            # No local warp available — fall back to global quality scalar
            W_f = np.full((H, W), max(float(scores[idx]) ** _per_ap_pow, 1e-9),
                          dtype=np.float64)
            map_dx = np.zeros((H, W), dtype=np.float32)
            map_dy = np.zeros((H, W), dtype=np.float32)

        remap_x = xx_base + map_dx
        remap_y = yy_base + map_dy
        warped = cv2.remap(
            frame_aligned, remap_x, remap_y,
            interpolation=_interp,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        warped = np.clip(warped, 0.0, 1.0)

        accum      += warped.astype(np.float64) * W_f
        weight_sum += W_f

        frame_logs.append({
            "frame_idx":       idx,
            "quality_score":   round(float(scores[idx]), 6),
            "global_shift_px": [round(float(dx_g), 3), round(float(dy_g), 3)],
            "align_method":    align_method,
            "n_good_aps":      n_good,
        })

        if progress_callback:
            progress_callback(i + 1, n_sel)

    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(weight_sum > 1e-12, accum / weight_sum, 0.0).astype(np.float32)
    result = np.clip(result, 0.0, 1.0)

    stats = {
        "n_stacked":            n_sel,
        "n_global_only_frames": n_global_only,
        "n_aps":                n_ap,
        "disk_center_px":       [round(disk_cx, 2), round(disk_cy, 2)],
        "disk_radius_px":       round(disk_radius, 2),
        "frames":               frame_logs,
    }
    return result, stats


def _per_ap_independent_stack(
    selected_frames: np.ndarray,
    selected_indices: np.ndarray,
    scores: np.ndarray,
    reference: np.ndarray,
    disk_cx: float,
    disk_cy: float,
    disk_radius: float,
    ap_positions: List,
    cfg,
    progress_callback=None,
) -> Tuple[np.ndarray, Dict]:
    """True per-AP independent lucky stacking (patch-based, legacy).

    For each AP, independently selects the best sub-frames by LOCAL quality at
    that AP location, then stacks only those patches. Different APs use different
    frame subsets — this is the core of true lucky imaging under atmospheric
    turbulence (isoplanatic patches are small; a frame sharp at AP[i] may be
    blurry at AP[j]).

    Wavelet grid artifact is suppressed by blending each AP with a 2× ap_size
    Gaussian mask (sigma = ap_size*2/3), so patches extend well past the planet
    disk edge and heavily overlap neighbors — brightness seams average away before
    wavelet sharpening can amplify them.

    Algorithm:
      Pass 1: Global-align all frames; compute per-AP quality score matrix
              [N_sel × N_ap] using Sobel gradient at each AP patch.
      Pass 2: For each AP, select top sub_percent frames by LOCAL score,
              estimate per-frame local shift (NCC / phaseCorrelate), extract
              sub-pixel patches via getRectSubPix, stack, blend with Gaussian mask.
    """
    H, W = reference.shape[:2]
    n_sel = len(selected_frames)
    n_ap  = len(ap_positions)
    sub_pct = float(getattr(cfg, "per_ap_stack_sub_percent", 0.5))
    _ksize  = int(getattr(cfg, "quality_gradient_ksize", 3))
    _power  = float(cfg.quality_weight_power)
    _stab_thresh = int(getattr(cfg, "stabilization_planet_threshold", 0))
    _interp = getattr(cfg, "remap_interpolation", cv2.INTER_CUBIC)
    _score_metric = str(getattr(cfg, "score_metric", "local_gradient"))
    _log_sigma    = float(getattr(cfg, "log_disk_sigma", 3.0))
    _log_thr      = float(getattr(cfg, "log_disk_threshold", 0.25))

    # ── Pass 1: global alignment + per-AP score matrix ────────────────────────
    global_shifts = np.zeros((n_sel, 2), dtype=np.float32)   # (dx_g, dy_g)
    score_matrix  = np.zeros((n_sel, n_ap), dtype=np.float32)

    n_workers = int(getattr(cfg, "n_workers", 1))
    if n_workers <= 0:
        n_workers = _mp.cpu_count()
    n_workers = max(1, n_workers)
    _fork_ok  = "fork" in _mp.get_all_start_methods()

    if n_workers > 1 and _fork_ok:
        global _WORKER_STATE
        _WORKER_STATE = {
            "frames":       selected_frames,
            "reference":    reference,
            "ap_positions": ap_positions,
            "cfg":          cfg,
            "disk_cx":      disk_cx,
            "disk_cy":      disk_cy,
            "disk_radius":  disk_radius,
        }
        all_idx  = list(range(n_sel))
        chunk_sz = max(1, (n_sel + n_workers - 1) // n_workers)
        chunks   = [all_idx[k:k + chunk_sz] for k in range(0, n_sel, chunk_sz)]
        ctx      = _mp.get_context("fork")
        with ctx.Pool(n_workers) as pool:
            results = pool.map(_per_ap_pass1_worker, chunks)
        for chunk_idx, (shifts_chunk, scores_chunk) in zip(chunks, results):
            for li, i in enumerate(chunk_idx):
                global_shifts[i] = shifts_chunk[li]
                score_matrix[i]  = scores_chunk[li]
        if progress_callback:
            progress_callback(n_sel, n_sel * 2)
    else:
        for i, frame in enumerate(selected_frames):
            dx_g, dy_g = limb_center_align(
                disk_cx, disk_cy, frame, fixed_threshold=_stab_thresh
            )
            if abs(dx_g) > disk_radius * 0.5 or abs(dy_g) > disk_radius * 0.5:
                dx_g, dy_g = subpixel_align(reference, frame)
            global_shifts[i] = (dx_g, dy_g)

            aligned = apply_shift(frame, dx_g, dy_g)
            for j, ap in enumerate(ap_positions):
                ax, ay = int(ap[0]), int(ap[1])
                half = (int(ap[2]) if len(ap) >= 3 else cfg.ap_size) // 2
                if ay - half < 0 or ay + half > H or ax - half < 0 or ax + half > W:
                    continue
                patch = aligned[ay - half: ay + half, ax - half: ax + half]
                if _score_metric == "log_disk":
                    pf = patch.astype(np.float32)
                    pm = float(pf.max())
                    if pm > 1e-9:
                        pf /= pm
                    mask_p = pf > _log_thr
                    if mask_p.sum() < 5:
                        score_matrix[i, j] = 0.0
                    else:
                        bl = cv2.GaussianBlur(pf, (0, 0), _log_sigma)
                        lp = cv2.Laplacian(bl, cv2.CV_32F, ksize=3)
                        score_matrix[i, j] = float(lp[mask_p].var())
                else:
                    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=_ksize)
                    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=_ksize)
                    score_matrix[i, j] = float((gx ** 2 + gy ** 2).max())

            if progress_callback and i % 100 == 0:
                progress_callback(i, n_sel * 2)

    # ── Intermediate: build sub-pixel globally-aligned frames ────────────────
    # Apply sub-pixel (bicubic) global shift to every selected frame so Pass 2
    # can extract patches without re-running warpAffine per frame×AP.
    aligned_frames = np.empty((n_sel, H, W), dtype=np.float32)
    for i, frame in enumerate(selected_frames):
        dx_g, dy_g = float(global_shifts[i, 0]), float(global_shifts[i, 1])
        aligned_frames[i] = apply_shift(frame, dx_g, dy_g)

    # ── Pass 2: per-AP independent patch stacking ────────────────────────────
    accum  = np.zeros((H, W), dtype=np.float64)
    weight = np.zeros((H, W), dtype=np.float64)
    n_top  = max(3, int(n_sel * sub_pct))
    frame_logs: List[Dict] = []
    _use_ncc_local = bool(getattr(cfg, "use_ncc", False))

    for j, ap in enumerate(ap_positions):
        ax, ay  = int(ap[0]), int(ap[1])
        ap_size = int(ap[2]) if len(ap) >= 3 else cfg.ap_size
        half    = ap_size // 2
        if ay - half < 0 or ay + half > H or ax - half < 0 or ax + half > W:
            continue

        # Blend region: 2× ap_size so patches extend well beyond the planet disk
        # edge and into neighboring APs — eliminates frame-subset brightness seams
        # that wavelet sharpening would amplify as a grid pattern.
        blend_size = ap_size * 2
        blend_half = blend_size // 2
        sigma_blend = blend_size / 3.0   # ~ap_size*2/3; ≈0.61 weight at ap_step distance
        g1d_b = cv2.getGaussianKernel(blend_size, sigma_blend)
        gmask = (g1d_b @ g1d_b.T).astype(np.float64)
        gmask /= gmask.max()

        hann_ap   = _get_hann(ap_size)
        ref_patch = reference[ay - half: ay + half, ax - half: ax + half].astype(np.float32)

        # Select top frames for this AP by local score
        ap_scores   = score_matrix[:, j]
        top_indices = np.argsort(ap_scores)[-n_top:]

        local_accum  = np.zeros((blend_size, blend_size), dtype=np.float64)
        local_weight = 0.0

        for li in top_indices:
            aligned_f = aligned_frames[li]  # sub-pixel globally aligned

            frm_patch = aligned_f[ay - half: ay + half, ax - half: ax + half].astype(np.float32)

            # Local shift at this AP (NCC or phaseCorrelate)
            if _use_ncc_local:
                dx_l, dy_l, _conf = _ncc_shift(
                    ref_patch, frm_patch, hann_ap, cfg.ap_search_range
                )
                if dx_l is None:
                    dx_l, dy_l = 0.0, 0.0
            else:
                (dx_l, dy_l), _ = cv2.phaseCorrelate(
                    ref_patch * hann_ap, frm_patch * hann_ap
                )
                dx_l, dy_l = float(dx_l), float(dy_l)
                if abs(dx_l) > cfg.ap_search_range or abs(dy_l) > cfg.ap_search_range:
                    dx_l, dy_l = 0.0, 0.0

            # Extract blend_size patch centered at the locally-shifted AP center
            cx = float(ax) + dx_l
            cy = float(ay) + dy_l
            patch = cv2.getRectSubPix(
                aligned_f, (blend_size, blend_size), (cx, cy)
            ).astype(np.float64)

            w = max(float(ap_scores[li]) ** _power, 1e-9)
            local_accum  += patch * w
            local_weight += w

        if local_weight < 1e-12:
            continue

        stacked_patch = local_accum / local_weight

        # Blend onto canvas — clamp to image bounds and slice gmask accordingly
        y0c = ay - blend_half;  y0g = 0
        y1c = ay + blend_half;  y1g = blend_size
        x0c = ax - blend_half;  x0g = 0
        x1c = ax + blend_half;  x1g = blend_size
        if y0c < 0:   y0g -= y0c; y0c = 0
        if y1c > H:   y1g -= (y1c - H); y1c = H
        if x0c < 0:   x0g -= x0c; x0c = 0
        if x1c > W:   x1g -= (x1c - W); x1c = W
        accum [y0c:y1c, x0c:x1c] += stacked_patch[y0g:y1g, x0g:x1g] * gmask[y0g:y1g, x0g:x1g]
        weight[y0c:y1c, x0c:x1c] += gmask[y0g:y1g, x0g:x1g]

        if progress_callback:
            progress_callback(n_sel + j, n_sel * 2)

    # Normalise; uncovered pixels → 0
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(weight > 1e-12, accum / weight, 0.0).astype(np.float32)
    result = np.clip(result, 0.0, 1.0)

    # Optional Fourier rolloff (low-pass noise suppression)
    _rolloff_sig = float(getattr(cfg, "fourier_rolloff_sigma", 0.0))
    if _rolloff_sig > 0.0:
        F  = np.fft.fft2(result.astype(np.float64))
        fy = np.fft.fftfreq(H)[:, None]
        fx = np.fft.fftfreq(W)[None, :]
        rolloff = np.exp(-0.5 * (np.sqrt(fy ** 2 + fx ** 2) / _rolloff_sig) ** 2)
        result  = np.fft.ifft2(F * rolloff).real.astype(np.float32)
        result  = np.clip(result, 0.0, 1.0)

    stats = {
        "n_stacked":            n_sel,
        "n_global_only_frames": 0,
        "n_aps":                n_ap,
        "disk_center_px":       [round(disk_cx, 2), round(disk_cy, 2)],
        "disk_radius_px":       round(disk_radius, 2),
        "frames":               frame_logs,
    }
    return result, stats


# ── 5c-2. Fourier-domain quality-weighted stacking (Mackay 2013 approach) ─────

def _fourier_pass_worker(chunk_indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Worker for _fourier_quality_stack: global-align + FFT accumulation per chunk.

    Called in fork workers; reads from _WORKER_STATE (no pickling needed).
    Returns (accum_F_chunk [H,W] complex128, weight_F_chunk [H,W] float64).
    """
    frames          = _WORKER_STATE["frames"]
    reference       = _WORKER_STATE["reference"]
    scores          = _WORKER_STATE["scores"]
    selected_indices = _WORKER_STATE["selected_indices"]
    cfg             = _WORKER_STATE["cfg"]
    disk_cx         = _WORKER_STATE["disk_cx"]
    disk_cy         = _WORKER_STATE["disk_cy"]
    disk_radius     = _WORKER_STATE["disk_radius"]

    H, W = reference.shape[:2]
    _stab_thresh    = int(getattr(cfg, "stabilization_planet_threshold", 0))
    power           = float(getattr(cfg, "fourier_quality_power", 1.0))
    _use_score      = bool(getattr(cfg, "per_ap_selection", False))
    _score_power    = float(getattr(cfg, "quality_weight_power", 1.0))

    accum_F  = np.zeros((H, W), dtype=np.complex128)
    weight_F = np.zeros((H, W), dtype=np.float64)

    for i in chunk_indices:
        frame = frames[i]
        dx_g, dy_g = limb_center_align(disk_cx, disk_cy, frame, fixed_threshold=_stab_thresh)
        if abs(dx_g) > disk_radius * 0.5 or abs(dy_g) > disk_radius * 0.5:
            dx_g, dy_g = subpixel_align(reference, frame)

        aligned = apply_shift(frame, dx_g, dy_g)
        F_n = np.fft.fft2(aligned.astype(np.float64))

        if _use_score:
            q_scalar = max(float(scores[int(selected_indices[i])]) ** _score_power, 1e-9)
            w_n = q_scalar * np.abs(F_n) ** power
        else:
            w_n = np.abs(F_n) ** power

        accum_F  += w_n * F_n
        weight_F += w_n

    return accum_F, weight_F


def _fourier_quality_stack(
    selected_frames: np.ndarray,
    selected_indices: np.ndarray,
    scores: np.ndarray,
    reference: np.ndarray,
    disk_cx: float,
    disk_cy: float,
    disk_radius: float,
    ap_positions: List,
    cfg,
    progress_callback=None,
    cancel_event=None,
    precomputed_noise_floor: "np.ndarray | None" = None,
) -> Tuple[np.ndarray, Dict]:
    """Fourier-domain quality-weighted stacking.

    For each spatial frequency f, frame n contributes with weight |FFT_n(f)|^power.
    Frames that are sharper (higher amplitude) at a given frequency automatically
    contribute more to that frequency — per-frequency lucky selection without
    spatial patch boundaries.

    Reference: Mackay 2013, arXiv:1303.5108 "High-Efficiency Lucky Imaging"

    Parallelised with ThreadPoolExecutor (closure captures read-only arrays,
    each chunk accumulates into local buffers then merged).  Works on Windows
    (no fork required) and avoids _WORKER_STATE global race conditions when
    multiple SER files are processed in parallel.
    """
    H, W = reference.shape[:2]
    n_sel = len(selected_frames)
    power        = float(getattr(cfg, "fourier_quality_power", 1.0))
    _stab_thresh = int(getattr(cfg, "stabilization_planet_threshold", 0))
    _use_score   = bool(getattr(cfg, "per_ap_selection", False))
    _score_power = float(getattr(cfg, "quality_weight_power", 1.0))

    # ── Noise-reduction options (require n_workers=1) ──────────────────────
    _snr_mask       = bool(getattr(cfg, "fourier_snr_mask", False))
    _snr_thresh     = float(getattr(cfg, "fourier_snr_threshold", 1.0))
    _rolloff_sig    = float(getattr(cfg, "fourier_rolloff_sigma", 0.0))
    _noise_floor_en = bool(getattr(cfg, "fourier_noise_floor", False))

    n_workers   = int(getattr(cfg, "n_workers", 1))
    if n_workers <= 0:
        n_workers = _mp.cpu_count()
    n_ser = max(1, int(getattr(cfg, "n_ser_parallel", 1)))
    n_workers = max(1, n_workers // n_ser)

    if n_workers > 1 and (_snr_mask or _rolloff_sig > 0.0 or _noise_floor_en):
        print("  [Fourier] snr_mask/rolloff/noise_floor require n_workers=1 — disabling", flush=True)
        _snr_mask = _noise_floor_en = False
        _rolloff_sig = 0.0

    # ── B: Rolloff mask (compute once) ────────────────────────────────────
    _rolloff_mask: np.ndarray | None = None
    if _rolloff_sig > 0.0:
        fy = np.fft.fftfreq(H)[:, None]
        fx = np.fft.fftfreq(W)[None, :]
        _rolloff_mask = np.exp(-0.5 * ((np.sqrt(fy**2 + fx**2)) / _rolloff_sig) ** 2)

    # ── C: Noise floor — use precomputed (from global bottom-25% frames) ──
    _noise_floor: "np.ndarray | None" = precomputed_noise_floor if _noise_floor_en else None

    accum_F  = np.zeros((H, W), dtype=np.complex128)
    weight_F = np.zeros((H, W), dtype=np.float64)

    # ── A: Extra accumulators for SNR mask ────────────────────────────────
    if _snr_mask:
        _sum_abs_F    = np.zeros((H, W), dtype=np.float64)
        _sum_abs_F_sq = np.zeros((H, W), dtype=np.float64)

    # Closure captures all read-only data — no global state needed.
    def _fourier_chunk(chunk_indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        local_accum  = np.zeros((H, W), dtype=np.complex128)
        local_weight = np.zeros((H, W), dtype=np.float64)
        for i in chunk_indices:
            if cancel_event is not None and cancel_event.is_set():
                break
            frame = selected_frames[i]
            dx_g, dy_g = limb_center_align(disk_cx, disk_cy, frame, fixed_threshold=_stab_thresh)
            if abs(dx_g) > disk_radius * 0.5 or abs(dy_g) > disk_radius * 0.5:
                dx_g, dy_g = subpixel_align(reference, frame)
            aligned = apply_shift(frame, dx_g, dy_g)
            F_n = np.fft.fft2(aligned.astype(np.float64))
            if _use_score:
                q_scalar = max(float(scores[int(selected_indices[i])]) ** _score_power, 1e-9)
                w_n = q_scalar * np.abs(F_n) ** power
            else:
                w_n = np.abs(F_n) ** power
            local_accum  += w_n * F_n
            local_weight += w_n
        return local_accum, local_weight

    if n_workers > 1:
        all_idx  = list(range(n_sel))
        chunk_sz = max(1, (n_sel + n_workers - 1) // n_workers)
        chunks   = [all_idx[k:k + chunk_sz] for k in range(0, n_sel, chunk_sz)]
        with _ThreadPoolExecutor(max_workers=n_workers) as executor:
            for local_accum, local_weight in executor.map(_fourier_chunk, chunks):
                accum_F  += local_accum
                weight_F += local_weight
        if progress_callback:
            progress_callback(n_sel, n_sel)
    else:
        for i, frame in enumerate(selected_frames):
            if cancel_event is not None and cancel_event.is_set():
                print(f"  [Fourier] Cancelled at frame {i}/{n_sel}", flush=True)
                break
            dx_g, dy_g = limb_center_align(disk_cx, disk_cy, frame, fixed_threshold=_stab_thresh)
            if abs(dx_g) > disk_radius * 0.5 or abs(dy_g) > disk_radius * 0.5:
                dx_g, dy_g = subpixel_align(reference, frame)
            aligned = apply_shift(frame, dx_g, dy_g)
            F_n = np.fft.fft2(aligned.astype(np.float64))
            abs_F = np.abs(F_n)

            # C: subtract noise floor before weighting
            abs_F_eff = np.maximum(abs_F - _noise_floor, 0.0) if _noise_floor is not None else abs_F

            if _use_score:
                q_scalar = max(float(scores[int(selected_indices[i])]) ** _score_power, 1e-9)
                w_n = q_scalar * abs_F_eff ** power
            else:
                w_n = abs_F_eff ** power
            accum_F  += w_n * F_n
            weight_F += w_n

            # A: accumulate for SNR mask
            if _snr_mask:
                _sum_abs_F    += abs_F
                _sum_abs_F_sq += abs_F ** 2

            if progress_callback and i % 100 == 0:
                progress_callback(i, n_sel)

    output_F = np.where(weight_F > 1e-12, accum_F / weight_F, 0.0)

    # ── A: Apply spectral SNR mask ─────────────────────────────────────────
    if _snr_mask and n_workers == 1:
        _mean_abs = _sum_abs_F / n_sel
        _var_abs  = np.maximum(_sum_abs_F_sq / n_sel - _mean_abs ** 2, 0.0)
        _snr      = _mean_abs / (np.sqrt(_var_abs) + 1e-9)
        output_F  = output_F * np.tanh(_snr / _snr_thresh)

    # ── B: Apply rolloff mask ─────────────────────────────────────────────
    if _rolloff_mask is not None:
        output_F = output_F * _rolloff_mask

    result = np.fft.ifft2(output_F).real.astype(np.float32)
    result = np.clip(result, 0.0, 1.0)

    if progress_callback:
        progress_callback(n_sel, n_sel)

    stats = {
        "n_stacked":            n_sel,
        "n_global_only_frames": 0,
        "n_aps":                len(ap_positions),
        "disk_center_px":       [round(disk_cx, 2), round(disk_cy, 2)],
        "disk_radius_px":       round(disk_radius, 2),
        "frames":               [],
    }
    return result, stats


# ── 5d. Worker state + worker function for multiprocessing ────────────────────
# _per_ap_pass1_worker is the Pass-1 parallel worker for _per_ap_independent_stack.
# _WORKER_STATE is set in the parent process immediately before Pool creation.
# fork workers inherit it via copy-on-write — no large-array pickling needed.

def _per_ap_pass1_worker(chunk_indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Pass 1 worker for _per_ap_independent_stack: global-align + per-AP scores.

    Called in fork workers; reads from _WORKER_STATE (no pickling needed).
    Returns (shifts_chunk [N, 2] float32, scores_chunk [N, n_ap] float32).
    """
    frames       = _WORKER_STATE["frames"]
    reference    = _WORKER_STATE["reference"]
    ap_positions = _WORKER_STATE["ap_positions"]
    cfg          = _WORKER_STATE["cfg"]
    disk_cx      = _WORKER_STATE["disk_cx"]
    disk_cy      = _WORKER_STATE["disk_cy"]
    disk_radius  = _WORKER_STATE["disk_radius"]

    H, W   = reference.shape[:2]
    n_ap   = len(ap_positions)
    _ksize = int(getattr(cfg, "quality_gradient_ksize", 3))
    _stab  = int(getattr(cfg, "stabilization_planet_threshold", 0))
    _score_metric = str(getattr(cfg, "score_metric", "local_gradient"))
    _log_sigma    = float(getattr(cfg, "log_disk_sigma", 3.0))
    _log_thr      = float(getattr(cfg, "log_disk_threshold", 0.25))

    shifts_chunk = np.zeros((len(chunk_indices), 2), dtype=np.float32)
    scores_chunk = np.zeros((len(chunk_indices), n_ap), dtype=np.float32)

    for li, i in enumerate(chunk_indices):
        frame = frames[i]
        dx_g, dy_g = limb_center_align(disk_cx, disk_cy, frame, fixed_threshold=_stab)
        if abs(dx_g) > disk_radius * 0.5 or abs(dy_g) > disk_radius * 0.5:
            dx_g, dy_g = subpixel_align(reference, frame)
        shifts_chunk[li] = (dx_g, dy_g)

        aligned = apply_shift(frame, dx_g, dy_g)
        for j, ap in enumerate(ap_positions):
            ax, ay = int(ap[0]), int(ap[1])
            half = (int(ap[2]) if len(ap) >= 3 else cfg.ap_size) // 2
            if ay - half < 0 or ay + half > H or ax - half < 0 or ax + half > W:
                continue
            patch = aligned[ay - half: ay + half, ax - half: ax + half]
            if _score_metric == "log_disk":
                pf = patch.astype(np.float32)
                pm = float(pf.max())
                if pm > 1e-9:
                    pf /= pm
                mask_p = pf > _log_thr
                if mask_p.sum() < 5:
                    scores_chunk[li, j] = 0.0
                else:
                    bl = cv2.GaussianBlur(pf, (0, 0), _log_sigma)
                    lp = cv2.Laplacian(bl, cv2.CV_32F, ksize=3)
                    scores_chunk[li, j] = float(lp[mask_p].var())
            else:
                gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=_ksize)
                gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=_ksize)
                scores_chunk[li, j] = float((gx ** 2 + gy ** 2).max())

    return shifts_chunk, scores_chunk


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
    ref_precomp   = _WORKER_STATE.get("ref_precomp")

    for local_i in chunk_indices:
        frame = frames[local_i]                   # float32 [0, 1]
        idx   = int(selected_indices[local_i])    # original index for score lookup

        # ── Global alignment ──────────────────────────────────────────────
        _stab_thresh = int(getattr(cfg, "stabilization_planet_threshold", 0))
        dx_g, dy_g = limb_center_align(disk_cx, disk_cy, frame, fixed_threshold=_stab_thresh)
        max_g = disk_radius * 0.5
        if abs(dx_g) > max_g or abs(dy_g) > max_g:
            dx_g, dy_g = subpixel_align(reference, frame)
            align_method = "phase_correlate"
        else:
            align_method = "limb_center"

        frame_aligned = apply_shift(frame, dx_g, dy_g)

        # ── AP warp maps ──────────────────────────────────────────────────
        if adaptive_mode:
            map_dx, map_dy, n_good, conf_map = _compute_adaptive_warp_maps(
                frame_aligned, reference, ap_positions, cfg, ref_precomp=ref_precomp
            )
        else:
            map_dx, map_dy, n_good, conf_map = _compute_warp_maps(
                frame_aligned, reference, ap_positions, hann2d, query_pts, cfg,
                ref_precomp=ref_precomp,
            )
        if n_good < 3:
            n_global_only += 1

        # ── Combined remap ────────────────────────────────────────────────
        remap_x = (xx_base + map_dx - dx_g).astype(np.float32)
        remap_y = (yy_base + map_dy - dy_g).astype(np.float32)
        _interp = getattr(cfg, "remap_interpolation", cv2.INTER_LINEAR)
        warped = cv2.remap(
            frame, remap_x, remap_y,
            interpolation=_interp,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        warped = np.clip(warped, 0.0, 1.0)

        # ── Weighted accumulate ───────────────────────────────────────────
        _per_ap_sel = bool(getattr(cfg, "per_ap_selection", False))
        quality_w   = max(float(scores[idx]) ** cfg.quality_weight_power, 1e-9)
        if _per_ap_sel and len(ap_positions) >= 3:
            q2d = _build_per_ap_quality_map(warped, ap_positions, cfg)
            local_accum  += warped.astype(np.float64) * q2d
            local_weight += q2d
        else:
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


# ── 6a. Sigma-clipping post-pass ──────────────────────────────────────────────

def _sigma_clip_stack(
    selected_frames: np.ndarray,
    selected_indices: np.ndarray,
    reference: np.ndarray,
    disk_cx: float,
    disk_cy: float,
    disk_radius: float,
    ap_positions: List[Tuple[int, int]],
    cfg,
    n_workers: int = 1,
    cancel_event=None,
) -> np.ndarray:
    """Sigma-clipping post-pass: warp all frames to *reference*, build per-pixel
    mean/std, discard |pixel − mean| > kappa × std, return nanmean.

    Called once after apply_warp_and_stack() finishes its n_iterations so the
    reference already has high SNR (good AP estimates).

    Memory: allocates one (N, H, W) float32 array — ~800 MB for N=2585, 280×280.
    Parallelised with ThreadPoolExecutor (numpy/cv2 release GIL → true concurrency,
    works on Windows where fork is unavailable).
    """
    N = len(selected_frames)
    H, W = reference.shape[:2]
    kappa = cfg.sigma_clip_kappa

    adaptive_mode = bool(ap_positions) and len(ap_positions[0]) == 3
    hann2d    = None if adaptive_mode else _make_hann2d(cfg.ap_size)
    xx_base   = np.tile(np.arange(W, dtype=np.float32), (H, 1))
    yy_base   = np.tile(np.arange(H, dtype=np.float32)[:, None], (1, W))
    query_pts = np.empty((0, 2), dtype=np.float64)
    _stab_thresh_sc = int(getattr(cfg, "stabilization_planet_threshold", 0))
    _interp = getattr(cfg, "remap_interpolation", cv2.INTER_LINEAR)

    warped_stack = np.zeros((N, H, W), dtype=np.float32)

    print(
        f"  [σ-clip] Warping {N} frames (kappa={kappa}, workers={n_workers})…",
        end="\r", flush=True,
    )

    # Pre-compute ref-patch FFTs once for the sigma-clip re-warp pass.
    _sc_ref_precomp = _precompute_ap_ref_data(reference, ap_positions, cfg)

    # Closure captures read-only shared state; each call writes to a unique row.
    def _warp_one(i: int) -> None:
        frame = selected_frames[i]
        dx_g, dy_g = limb_center_align(disk_cx, disk_cy, frame, fixed_threshold=_stab_thresh_sc)
        if abs(dx_g) > disk_radius * 0.5 or abs(dy_g) > disk_radius * 0.5:
            dx_g, dy_g = subpixel_align(reference, frame)
        frame_aligned = apply_shift(frame, dx_g, dy_g)

        if adaptive_mode:
            map_dx, map_dy, _, _cm = _compute_adaptive_warp_maps(
                frame_aligned, reference, ap_positions, cfg, ref_precomp=_sc_ref_precomp
            )
        else:
            map_dx, map_dy, _, _cm = _compute_warp_maps(
                frame_aligned, reference, ap_positions, hann2d, query_pts, cfg,
                ref_precomp=_sc_ref_precomp,
            )

        remap_x = (xx_base + map_dx - dx_g).astype(np.float32)
        remap_y = (yy_base + map_dy - dy_g).astype(np.float32)
        warped = cv2.remap(
            frame, remap_x, remap_y,
            interpolation=_interp,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        warped_stack[i] = np.clip(warped, 0.0, 1.0)

    if n_workers > 1:
        done = [0]
        lock = _threading.Lock()

        def _warp_tracked(i: int) -> None:
            if cancel_event is not None and cancel_event.is_set():
                return
            _warp_one(i)
            with lock:
                done[0] += 1
                if done[0] % 200 == 0 or done[0] == N:
                    print(f"  [σ-clip] Warping {done[0]}/{N}…", end="\r", flush=True)

        with _ThreadPoolExecutor(max_workers=n_workers) as executor:
            list(executor.map(_warp_tracked, range(N)))
    else:
        for i in range(N):
            if cancel_event is not None and cancel_event.is_set():
                print(f"  [σ-clip] Cancelled at frame {i}/{N}", flush=True)
                break
            _warp_one(i)
            if i % 200 == 0:
                print(f"  [σ-clip] Warping {i+1}/{N}…", end="\r", flush=True)

    mean = warped_stack.mean(axis=0)
    std  = warped_stack.std(axis=0)

    mask = np.abs(warped_stack - mean[np.newaxis]) <= kappa * np.maximum(std[np.newaxis], 1e-6)
    clip_pct = float(1.0 - mask.mean()) * 100.0
    print(f"  [σ-clip] Done — clipped {clip_pct:.1f}% of pixels          ", flush=True)

    warped_stack[~mask] = np.nan
    result = np.nanmean(warped_stack, axis=0)
    result = np.where(np.isnan(result), mean, result)
    return np.clip(result, 0.0, 1.0).astype(np.float32)


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
    cancel_event=None,
    precomputed_noise_floor: "np.ndarray | None" = None,
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

    # per_ap_selection → patch-based independent lucky stacking with wide-Gaussian
    # blending (2× ap_size blend region) to suppress wavelet grid artifacts.
    # use_per_ap_stack → same function, legacy alias.
    # Both bypass the parallel pool and manage their own workers / streaming.
    if bool(getattr(cfg, "per_ap_selection", False)) or bool(getattr(cfg, "use_per_ap_stack", False)):
        return _per_ap_independent_stack(
            selected_frames, selected_indices, scores, reference,
            disk_cx, disk_cy, disk_radius, ap_positions, cfg,
            progress_callback=progress_callback,
        )

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
        _ref_precomp_par = _precompute_ap_ref_data(reference, ap_positions, cfg)
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
            "ref_precomp":      _ref_precomp_par,
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
        _use_fourier_quality = bool(getattr(cfg, "use_fourier_quality", False))

        # try69: Fourier-domain quality-weighted stacking — separate early return
        if _use_fourier_quality:
            return _fourier_quality_stack(
                selected_frames, selected_indices, scores, reference,
                disk_cx, disk_cy, disk_radius, ap_positions, cfg,
                progress_callback=progress_callback,
                cancel_event=cancel_event,
                precomputed_noise_floor=precomputed_noise_floor,
            )

        _use_cog     = bool(getattr(cfg, "cog_align", False))
        _per_ap_sel  = bool(getattr(cfg, "per_ap_selection", False))
        _use_patch   = bool(getattr(cfg, "use_patch_blend", False))
        _use_tps     = bool(getattr(cfg, "use_tps", False))

        # Pre-compute ref-patch FFTs once — reused across all frames.
        # TPS path uses _compute_warp_maps_tps (different function) → skip.
        _ref_precomp = None if _use_tps else _precompute_ap_ref_data(reference, ap_positions, cfg)

        # try57 patch-blend: pre-build Gaussian mask
        if _use_patch:
            _pb_half = cfg.ap_size // 2
            _g1d = cv2.getGaussianKernel(cfg.ap_size, cfg.ap_size / 4.0)
            _pb_mask = (_g1d @ _g1d.T).astype(np.float64)
            _pb_mask /= _pb_mask.max()   # peak = 1.0

        for i, (frame, idx) in enumerate(zip(selected_frames, selected_indices)):
            idx = int(idx)

            # ── Global alignment ──────────────────────────────────────────
            _stab_thresh = int(getattr(cfg, "stabilization_planet_threshold", 0))
            if _use_cog:
                # try54: CoG (brightness-weighted centroid) instead of ellipse fit
                dx_g, dy_g = _cog_center_align(
                    disk_cx, disk_cy, frame,
                    max_shift_px=disk_radius * 0.5,
                    fixed_threshold=_stab_thresh,
                )
                align_method = "cog"
                # Fall back if CoG returns (0,0) due to failure
                if dx_g == 0.0 and dy_g == 0.0:
                    dx_g, dy_g = subpixel_align(reference, frame)
                    align_method = "phase_correlate"
            else:
                dx_g, dy_g = limb_center_align(disk_cx, disk_cy, frame, fixed_threshold=_stab_thresh)
                align_method = "limb_center"
                max_g = disk_radius * 0.5
                if abs(dx_g) > max_g or abs(dy_g) > max_g:
                    dx_g, dy_g = subpixel_align(reference, frame)
                    align_method = "phase_correlate"

            frame_aligned = apply_shift(frame, dx_g, dy_g)

            # ── AP warp maps ──────────────────────────────────────────────
            if _use_tps:
                # try63: TPS — exact shift interpolation, no KR dilution
                map_dx, map_dy, n_good, conf_map = _compute_warp_maps_tps(
                    frame_aligned, reference, ap_positions, hann2d, cfg
                )
            elif adaptive_mode:
                map_dx, map_dy, n_good, conf_map = _compute_adaptive_warp_maps(
                    frame_aligned, reference, ap_positions, cfg, ref_precomp=_ref_precomp
                )
            else:
                map_dx, map_dy, n_good, conf_map = _compute_warp_maps(
                    frame_aligned, reference, ap_positions, hann2d, query_pts, cfg,
                    ref_precomp=_ref_precomp,
                )
            if n_good < 3:
                n_global_only += 1

            if _use_patch:
                # ── try57/61: Patch blend — per-AP patch accumulation ─────
                _interp = getattr(cfg, "remap_interpolation", cv2.INTER_LINEAR)
                # try61: per-AP quality weighting inside patch blend.
                # Build per-AP quality map from the warped frame when
                # per_ap_selection is also enabled.
                if _per_ap_sel and len(ap_positions) >= 3:
                    _remap_x2 = (xx_base + map_dx - dx_g).astype(np.float32)
                    _remap_y2 = (yy_base + map_dy - dy_g).astype(np.float32)
                    _wf = cv2.remap(frame, _remap_x2, _remap_y2,
                                    interpolation=_interp,
                                    borderMode=cv2.BORDER_REFLECT_101)
                    _q2d = _build_per_ap_quality_map(
                        np.clip(_wf, 0.0, 1.0).astype(np.float32),
                        ap_positions, cfg,
                    )
                    _fill_base_w = float(np.mean(_q2d))
                else:
                    _q2d = None
                    quality_w = max(
                        float(scores[idx]) ** cfg.quality_weight_power, 1e-9)
                    _fill_base_w = quality_w

                for _ap in ap_positions:
                    _ax, _ay = int(_ap[0]), int(_ap[1])
                    _ap_sz   = int(_ap[2]) if len(_ap) >= 3 else cfg.ap_size
                    _ph      = _ap_sz // 2
                    if _ay - _ph < 0 or _ay + _ph > H or _ax - _ph < 0 or _ax + _ph > W:
                        continue
                    # AP-local shift from warp maps
                    _ldx = float(map_dx[_ay, _ax])
                    _ldy = float(map_dy[_ay, _ax])
                    # Source center in original frame: account for global shift
                    _src_cx = float(_ax) + _ldx - dx_g
                    _src_cy = float(_ay) + _ldy - dy_g
                    # Sub-pixel patch extraction
                    patch = cv2.getRectSubPix(
                        frame.astype(np.float32), (_ap_sz, _ap_sz),
                        (_src_cx, _src_cy)
                    ).astype(np.float64)
                    # Gaussian window mask × per-AP (or global) quality weight
                    _msk = _pb_mask if _ap_sz == cfg.ap_size else (
                        lambda s: (lambda g: (g @ g.T) / (g @ g.T).max())(
                            cv2.getGaussianKernel(s, s / 4.0))
                    )(_ap_sz)
                    _aq  = float(_q2d[_ay, _ax]) if _q2d is not None else quality_w
                    _w2d = _msk * _aq
                    _sy  = slice(_ay - _ph, _ay + _ph)
                    _sx  = slice(_ax - _ph, _ax + _ph)
                    accum[_sy, _sx]      += patch * _w2d
                    weight_sum[_sy, _sx] += _w2d
                # Fill-in fallback: remap at 0.1% weight fills zero-coverage
                # regions without affecting AP-covered areas (Gaussian peak=1.0,
                # fill=0.001 → <0.1% contribution where patches exist).
                _remap_x = (xx_base + map_dx - dx_g).astype(np.float32)
                _remap_y = (yy_base + map_dy - dy_g).astype(np.float32)
                _warped  = cv2.remap(frame, _remap_x, _remap_y,
                                     interpolation=_interp,
                                     borderMode=cv2.BORDER_REFLECT_101)
                _warped  = np.clip(_warped, 0.0, 1.0).astype(np.float64)
                _fw      = _fill_base_w * 1e-3   # 1000× less than AP patch peak
                accum      += _warped * _fw
                weight_sum += _fw
            else:
                # ── Combined remap ────────────────────────────────────────
                remap_x = (xx_base + map_dx - dx_g).astype(np.float32)
                remap_y = (yy_base + map_dy - dy_g).astype(np.float32)
                _interp = getattr(cfg, "remap_interpolation", cv2.INTER_LINEAR)
                warped = cv2.remap(
                    frame, remap_x, remap_y,
                    interpolation=_interp,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
                warped = np.clip(warped, 0.0, 1.0)

                # ── Weighted accumulate ───────────────────────────────────
                quality_w  = max(float(scores[idx]) ** cfg.quality_weight_power, 1e-9)
                if _per_ap_sel and len(ap_positions) >= 3:
                    # try56: spatially varying per-AP quality weight map
                    q2d = _build_per_ap_quality_map(warped, ap_positions, cfg)
                    accum      += warped.astype(np.float64) * q2d
                    weight_sum += q2d
                else:
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
    session_aps: Optional[List[Tuple[int, int, int]]] = None,
    session_ref_cx: float = 0.0,
    session_ref_cy: float = 0.0,
    cancel_event=None,
) -> Tuple[np.ndarray, Dict]:
    """Run the full lucky stacking pipeline on a single SER file.

    Args:
        ser_path:          Path to a PIPP-preprocessed SER file.
        cfg:               LuckyStackConfig.
        progress_callback: Optional (done, total) callback for UI progress bars.
        session_aps:       Pre-computed AP positions from reference SER (session-wide mode).
                           If provided, these APs are offset-corrected per this SER's disk
                           center and used directly (highest priority — overrides all grid modes).
        session_ref_cx:    Disk cx of the reference SER that generated session_aps.
        session_ref_cy:    Disk cy of the reference SER that generated session_aps.

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

        # ── Phase 0.5: Generate AP grid from middle frame for local scoring ──
        # When score_metric="local_gradient" but no AS3 APs are available,
        # derive a quick AP grid from the middle frame to feed score_frames_local().
        # This enables AS!4-style local gradient frame discrimination (CV ~4-6%)
        # without requiring external AS3 files (global Laplacian: CV ~1.4%).
        _score_metric = getattr(cfg, "score_metric", "laplacian")
        preloaded_ap_positions: Optional[List] = None
        if _score_metric == "local_gradient":
            try:
                _mid_idx = n_frames // 2
                _mid_frame = reader.get_frame(_mid_idx).astype(np.float32) / 255.0
                _cx, _cy, _semi_a, _, _ = find_disk_center(_mid_frame)
                _raw_aps = generate_ap_grid(_cx, _cy, float(_semi_a), _mid_frame, cfg)
                preloaded_ap_positions = [(ax, ay, cfg.ap_size) for ax, ay in _raw_aps]
                del _mid_frame
                print(
                    f"  [0.5] AP grid for local scoring: {len(preloaded_ap_positions)} pts  "
                    f"(disk cx={_cx:.0f} cy={_cy:.0f} r={_semi_a:.0f}px)",
                    flush=True,
                )
            except Exception as _e:
                print(
                    f"  [0.5] AP grid generation failed ({_e}) — falling back to global scorer",
                    flush=True,
                )

        def _score_prog(done: int, total: int) -> None:
            _pu(int(_SCORE_END * done / max(total, 1)))

        _use_local_scoring = (
            _score_metric == "local_gradient"
            and preloaded_ap_positions is not None
        )
        if _score_metric == "log_disk":
            print("  [1/5] LoG disk scoring (AS!4 lapl3 방식)…", end="\r", flush=True)
            scores = score_frames_log_disk(
                reader, cfg, score_step=2, progress_callback=_score_prog,
            )
            t1 = time.perf_counter()
            _pu(_SCORE_END)
            print(
                f"  [1/5] LoG-disk scored {n_frames} frames  "
                f"CV={scores.std()/max(scores.mean(),1e-9)*100:.1f}%  ({t1-t0:.1f}s)",
                flush=True,
            )
        elif _use_local_scoring:
            print("  [1/5] Local gradient scoring (AP patches)…", end="\r", flush=True)
            scores = score_frames_local(
                reader, preloaded_ap_positions, cfg,
                score_step=2, progress_callback=_score_prog,
            )
            t1 = time.perf_counter()
            _pu(_SCORE_END)
            print(
                f"  [1/5] Local scored {n_frames} frames  "
                f"CV={scores.std()/max(scores.mean(),1e-9)*100:.1f}%  ({t1-t0:.1f}s)",
                flush=True,
            )
        else:
            print("  [1/5] Scoring frames…", end="\r", flush=True)
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

        # ── Phase 3.1: Session AP offset correction ──────────────────────
        # If session_aps provided, shift by (disk_cx - ref_cx, disk_cy - ref_cy)
        # and re-filter to keep only APs whose centres land inside this disk.
        _session_ap_positions: Optional[List] = None
        if session_aps:
            H_img, W_img = reference.shape[:2]
            _dx = disk_cx - session_ref_cx
            _dy = disk_cy - session_ref_cy
            _r2 = disk_radius * disk_radius
            _shifted: List[Tuple[int, int, int]] = []
            for _ax, _ay, _sz in session_aps:
                _nax = int(round(_ax + _dx))
                _nay = int(round(_ay + _dy))
                _half = _sz // 2
                if (_nax - _half < 0 or _nax + _half > W_img
                        or _nay - _half < 0 or _nay + _half > H_img):
                    continue
                _ddx = _nax - disk_cx
                _ddy = _nay - disk_cy
                if _ddx * _ddx + _ddy * _ddy < _r2:
                    _shifted.append((_nax, _nay, _sz))
            _session_ap_positions = _shifted if _shifted else None
            from collections import Counter as _Ctr2
            if _session_ap_positions:
                _sc = _Ctr2(sz for _, _, sz in _session_ap_positions)
                print(
                    f"  [3.1] Session APs shifted (Δ={_dx:.1f},{_dy:.1f}): "
                    f"{len(_session_ap_positions)} pts  "
                    + " ".join(f"{sz}px×{_sc[sz]}" for sz in sorted(_sc)),
                    flush=True,
                )
            else:
                print(f"  [3.1] WARNING: No session APs survived disk filter — fallback to grid.", flush=True)

        # ── Phase 3.5: Pre-load selected frames (once, shared across iterations) ─
        n_workers_cfg = int(getattr(cfg, "n_workers", 0))
        n_workers_use = n_workers_cfg if n_workers_cfg > 0 else _mp.cpu_count()
        n_workers_use = max(1, n_workers_use)

        # use_patch_blend and use_tps are sequential-path-only features.
        # per_ap_selection now runs in both sequential and parallel paths.
        # use_per_ap_stack manages its own internal sub-pool via _per_ap_pass1_worker.
        _needs_seq = (
            bool(getattr(cfg, "use_patch_blend", False))
            or bool(getattr(cfg, "use_tps", False))
            or bool(getattr(cfg, "use_per_ap_stack", False))
            or bool(getattr(cfg, "use_fourier_quality", False))
        )
        if _needs_seq and n_workers_use > 1:
            _seq_reason = (
                "use_per_ap_stack (manages own sub-pool)"
                if getattr(cfg, "use_per_ap_stack", False)
                else "use_fourier_quality (manages own sub-pool)"
                if getattr(cfg, "use_fourier_quality", False)
                else "use_patch_blend/use_tps requires sequential path"
            )
            print(
                f"  [NOTE] {_seq_reason} "
                f"— outer n_workers forced to 1 (was {n_workers_use})",
                flush=True,
            )
            n_workers_use = 1

        print(
            f"  [3.5] Pre-loading {n_select} frames"
            f"  (workers: {n_workers_use})…",
            end="\r", flush=True,
        )
        t_pre0 = time.perf_counter()
        # Read frames sorted by file position (sequential I/O) then restore
        # quality order.  Random-order seeks are ~24× slower than sequential
        # on both HDD and cached SSD (benchmark: 7.17 ms vs 0.30 ms/frame).
        _sort_by_pos  = np.argsort(selected_indices)           # file-pos order
        _frames_bypos = np.stack([
            reader.get_frame(int(selected_indices[i])).astype(np.float32) / 255.0
            for i in _sort_by_pos
        ])
        _inv_order    = np.argsort(_sort_by_pos)               # restore quality order
        selected_frames = _frames_bypos[_inv_order]            # (N_select, H, W) float32 [0, 1]
        del _frames_bypos, _sort_by_pos, _inv_order
        t_pre1 = time.perf_counter()
        _pu(_STACK_START)
        print(
            f"  [3.5] Pre-loaded {n_select} frames  "
            f"({selected_frames.nbytes / 1e6:.0f} MB, {t_pre1-t_pre0:.1f}s)",
            flush=True,
        )

        # ── Phase 3.6: Fourier noise floor pre-pass (global bottom-25%) ──────
        # Compute mean FFT amplitude from the lowest-quality frames globally.
        # Uses frames NOT in selected_indices to avoid signal contamination.
        _precomputed_noise_floor: Optional[np.ndarray] = None
        if bool(getattr(cfg, "fourier_noise_floor", False)) and bool(getattr(cfg, "use_fourier_quality", False)):
            _stab_t = int(getattr(cfg, "stabilization_planet_threshold", 0))
            _noise_sorted_asc = np.argsort(scores)                # ascending quality (worst first)
            _n_noise = max(1, n_frames // 4)                      # bottom 25% of ALL frames
            # Exclude frames already in selected_indices to use truly bad frames
            _sel_set = set(selected_indices.tolist())
            _noise_cands = [int(i) for i in _noise_sorted_asc if i not in _sel_set]
            _n_noise = min(_n_noise, len(_noise_cands))
            if _n_noise > 0:
                _noise_indices = _noise_cands[:_n_noise]
                H_img, W_img = reference.shape[:2]
                _nf_sum = np.zeros((H_img, W_img), dtype=np.float64)
                print(f"  [3.6] Noise floor: loading {_n_noise} bottom frames…", end="\r", flush=True)
                for _ni in _noise_indices:
                    _fr = reader.get_frame(int(_ni)).astype(np.float32) / 255.0
                    _dx, _dy = limb_center_align(disk_cx, disk_cy, _fr, fixed_threshold=_stab_t)
                    if abs(_dx) > disk_radius * 0.5 or abs(_dy) > disk_radius * 0.5:
                        _dx, _dy = subpixel_align(reference, _fr)
                    _nf_sum += np.abs(np.fft.fft2(apply_shift(_fr, _dx, _dy).astype(np.float64)))
                _precomputed_noise_floor = _nf_sum / _n_noise
                print(f"  [3.6] Noise floor computed from {_n_noise} global-bottom frames", flush=True)

        stacked: Optional[np.ndarray] = None
        stats: Dict = {}
        t_stack_total = 0.0

        for iteration in range(n_iter):
            iter_label = f"iter {iteration+1}/{n_iter}" if n_iter > 1 else ""

            # ── Phase 4: AP grid ──────────────────────────────────────────
            # On iteration > 0, the reference is the previous stacked result
            # (much higher SNR → more accurate AP shifts).
            use_multiscale    = bool(getattr(cfg, "use_multiscale_ap", False))
            use_double        = bool(getattr(cfg, "use_double_ap_grid", False))
            use_adaptive      = bool(getattr(cfg, "use_adaptive_ap", True))
            use_as4_ap_grid   = bool(getattr(cfg, "use_as4_ap_grid", False))
            from collections import Counter as _Counter
            if _session_ap_positions is not None:
                # Session-wide AS4 AP mode: pre-shifted APs from reference SER.
                ap_positions = _session_ap_positions
                ap_size_info = _Counter(sz for _, _, sz in ap_positions)
                ap_desc = "as4_session: " + " ".join(
                    f"{sz}px×{ap_size_info[sz]}" for sz in sorted(ap_size_info)
                )
            elif preloaded_ap_positions is not None:
                # local_gradient scoring — reuse AP grid built in Phase 0.5.
                ap_positions = preloaded_ap_positions
                ap_size_info = _Counter(sz for _, _, sz in ap_positions)
                ap_desc = "local_q: " + " ".join(
                    f"{sz}px×{ap_size_info[sz]}" for sz in sorted(ap_size_info)
                )
            elif use_as4_ap_grid:
                # Standalone AS4 greedy PDS mode (no session sharing).
                ap_positions = generate_as4_ap_grid(
                    disk_cx, disk_cy, disk_radius, reference, cfg
                )
                ap_size_info = _Counter(sz for _, _, sz in ap_positions)
                ap_desc = "as4_pds: " + " ".join(
                    f"{sz}px×{ap_size_info[sz]}" for sz in sorted(ap_size_info)
                )
            elif use_multiscale:
                ap_positions = generate_multiscale_ap_grid(
                    disk_cx, disk_cy, disk_radius, reference, cfg
                )
                ap_size_info = _Counter(sz for _, _, sz in ap_positions)
                ap_desc = "multiscale: " + " ".join(
                    f"{sz}px×{ap_size_info[sz]}" for sz in sorted(ap_size_info)
                )
            elif use_double:
                ap_positions = generate_double_ap_grid(
                    disk_cx, disk_cy, disk_radius, reference, cfg
                )
                ap_size_info = _Counter(sz for _, _, sz in ap_positions)
                ap_desc = "double_ap_grid: " + " ".join(
                    f"{sz}px×{ap_size_info[sz]}" for sz in sorted(ap_size_info)
                )
            elif use_adaptive:
                ap_positions = generate_adaptive_ap_grid(
                    disk_cx, disk_cy, disk_radius, reference, cfg
                )
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
                cancel_event=cancel_event,
                precomputed_noise_floor=_precomputed_noise_floor,
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

    # Post-iteration sigma-clipping pass (optional).
    # Uses final stacked result as reference for re-warping all frames.
    if getattr(cfg, "sigma_clip", False) and getattr(cfg, "sigma_clip_kappa", 0.0) > 0.0:
        t_sc0 = time.perf_counter()
        _sc_workers_cfg = int(getattr(cfg, "n_workers", 0))
        if _sc_workers_cfg <= 0:
            _sc_workers_cfg = _mp.cpu_count()
        _sc_n_ser = max(1, int(getattr(cfg, "n_ser_parallel", 1)))
        _sc_workers = max(1, _sc_workers_cfg // _sc_n_ser)
        stacked = _sigma_clip_stack(
            selected_frames,
            selected_indices,
            stacked,           # best reference: final iteration result
            disk_cx, disk_cy, disk_radius,
            ap_positions,
            cfg,
            n_workers=_sc_workers,
            cancel_event=cancel_event,
        )
        print(f"  [σ-clip] {time.perf_counter()-t_sc0:.1f}s", flush=True)

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
