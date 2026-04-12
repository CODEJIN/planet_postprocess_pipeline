"""
Step 8 – Time-series RGB/LRGB compositing with sliding-window stacking.

Groups raw TIF frames (Step 2 lucky-stacking output, via config.input_dir)
into filter-cycle sets, applies de-rotation correction, stacks them, then
applies wavelet sharpening and composites each set into RGB/LRGB/false-colour
images.  The resulting time-ordered series is used as input to Step 9
(animated GIF).

Note: Step 3 wavelet-preview output (results_03) is intentionally bypassed
for the mono path.  Sharpening is applied AFTER stacking (stack → sharpen)
to maximise SNR before the sharpening step.

Two stacking modes (controlled by config.composite.stack_window_n):

  N = 1 (default — single-frame mode):
    For each filter cycle, pick the one frame per filter closest to the bin
    centre.  De-rotate to the centre time and composite.  This is the
    original behaviour.

  N > 1 (sliding-window stacking):
    For each output frame i, gather the N consecutive filter observations
    nearest to cycle i for every filter.  Frames whose quality score is below
    stack_min_quality (Laplacian-variance normalised per filter) are dropped.
    All surviving frames are de-rotated to the centre time of cycle i, then
    averaged (mean stack).  SNR improves by √N and seeing outliers are
    diluted, at the cost of slightly reduced temporal resolution.

Algorithm per output frame:
  1. Collect N frames per filter (sliding window around centre cycle).
  2. Score each frame via Laplacian variance; drop frames below threshold.
  3. De-rotate each surviving frame to the centre time.
  4. Mean-stack de-rotated frames per filter.
  5. Apply global per-filter normalisation if enabled (Pass 1 / Pass 2).
  6. Composite channels using CompositeSpec logic.

Output (when config.save_step08 is True):
    <output_base>/step08_series/
        frame_001_2026-03-20T10:46Z/
            RGB_composite.png
            IR-RGB_composite.png
            CH4-G-IR_composite.png
        frame_002_…/
        series_summary.txt
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from pipeline.config import PipelineConfig
from pipeline.modules import composite as comp_module
from pipeline.modules import image_io, wavelet as wavelet_module
from pipeline.modules.derotation import (
    apply_shift, find_disk_center, find_visual_limb_radius,
    spherical_derotation_warp,
)


# ── Disk-based Step 3 result loader ───────────────────────────────────────────

def _load_step03_from_disk(
    config: PipelineConfig,
) -> Dict[str, List[Tuple[Optional[Path], dict]]]:
    """Reconstruct Step 3 results by scanning the step03_wavelet_preview/ dir.

    Used when step08 is run standalone (Step 3 results not in memory).
    PNG filenames follow the same AS!4 naming convention as the source TIFs,
    so ``image_io.parse_filename`` can extract timestamps from them.
    """
    step03_dir = config.step_dir(3, "wavelet_preview")
    if not step03_dir.exists():
        return {}

    results: Dict[str, List[Tuple[Optional[Path], dict]]] = {}
    for filt_dir in sorted(step03_dir.iterdir()):
        if not filt_dir.is_dir():
            continue
        filt = filt_dir.name
        entries = []
        for png in sorted(filt_dir.glob("*_wavelet.png")):
            # Strip "_wavelet" suffix to recover the original AS!4 stem
            original_stem = png.stem[: -len("_wavelet")]
            meta = image_io.parse_filename(Path(original_stem + ".tif"))
            if meta is None:
                continue
            entries.append((png, meta))
        if entries:
            entries.sort(key=lambda x: x[1]["timestamp"])
            results[filt] = entries

    return results


# ── Raw TIF loader ────────────────────────────────────────────────────────────

def _load_raw_tifs(
    config: PipelineConfig,
) -> Dict[str, List[Tuple[Optional[Path], dict]]]:
    """Load raw input TIF files from config.input_dir, grouped by filter.

    Scans input_dir recursively for *.tif files.  Uses image_io.parse_filename
    to extract filter name and timestamp from the AS!4 filename convention.

    Returns:
        ``{filter: [(path, meta), ...]}`` sorted by timestamp per filter.
    """
    results: Dict[str, List[Tuple[Optional[Path], dict]]] = {}

    for tif_path in sorted(config.input_dir.rglob("*.tif")):
        meta = image_io.parse_filename(tif_path)
        if meta is None:
            continue
        filt = meta.get("filter")
        if filt is None or filt not in config.filters:
            continue
        results.setdefault(filt, []).append((tif_path, meta))

    for filt in results:
        results[filt].sort(key=lambda x: x[1]["timestamp"])

    return results


# ── Filter-cycle grouping ──────────────────────────────────────────────────────

def _group_into_cycles(
    filter_frames: Dict[str, List[Tuple[Optional[Path], dict]]],
    cycle_minutes: float,
    required_rgb: List[str],
) -> List[dict]:
    """Group frames into filter-cycle sets by time binning.

    Returns:
        List of cycle dicts:
            {"center_time": datetime, "frames": {filter: (path, meta)}}
        Sorted by center_time.
    """
    all_times = [
        meta["timestamp"]
        for entries in filter_frames.values()
        for _, meta in entries
        if _ is not None
    ]
    if not all_times:
        return []

    t_start = min(all_times)
    t_end   = max(all_times)
    cycle_sec = cycle_minutes * 60.0

    cycles: List[dict] = []
    t = t_start
    while t <= t_end:
        t_bin_end = t + timedelta(seconds=cycle_sec)
        t_center  = t + timedelta(seconds=cycle_sec / 2.0)

        group: Dict[str, Tuple[Optional[Path], dict]] = {}
        for filt, entries in filter_frames.items():
            candidates = [
                (path, meta) for path, meta in entries
                if path is not None and t <= meta["timestamp"] < t_bin_end
            ]
            if candidates:
                best = min(
                    candidates,
                    key=lambda x: abs((x[1]["timestamp"] - t_center).total_seconds()),
                )
                group[filt] = best

        if all(f in group for f in required_rgb):
            cycles.append({"center_time": t_center, "frames": group})

        t = t_bin_end

    return sorted(cycles, key=lambda c: c["center_time"])


# ── Quality scoring ────────────────────────────────────────────────────────────

def _laplacian_score(img: np.ndarray) -> float:
    """Return Laplacian variance as a sharpness proxy.

    Higher = sharper.  Computed on a uint8 version of the grayscale image.
    """
    gray = img if img.ndim == 2 else img.mean(axis=2).astype(np.float32)
    u8 = (gray * 255).clip(0, 255).astype(np.uint8)
    lap = cv2.Laplacian(u8, cv2.CV_64F)
    return float(lap.var())


def _compute_filter_quality_scores(
    filter_frames: Dict[str, List[Tuple[Optional[Path], dict]]],
) -> Dict[str, Dict[str, float]]:
    """Pre-compute normalised quality score [0,1] for every frame.

    Returns: {filter: {stem: normalised_score}}
    The score is Laplacian variance normalised by the per-filter maximum.
    """
    raw: Dict[str, Dict[str, float]] = {}
    for filt, entries in filter_frames.items():
        filt_scores: Dict[str, float] = {}
        for path, meta in entries:
            if path is None:
                continue
            try:
                img = image_io.read_tif(path)
                filt_scores[meta["stem"]] = _laplacian_score(img)
            except Exception:
                filt_scores[meta["stem"]] = 0.0
        raw[filt] = filt_scores

    # Normalise per filter
    norm: Dict[str, Dict[str, float]] = {}
    for filt, scores in raw.items():
        if not scores:
            norm[filt] = {}
            continue
        max_s = max(scores.values()) or 1.0
        norm[filt] = {stem: s / max_s for stem, s in scores.items()}
    return norm


# ── Per-frame de-rotation ──────────────────────────────────────────────────────

def _derotate_frame(
    frame_path: Path,
    frame_time: datetime,
    t_reference: datetime,
    period_hours: float,
    warp_scale: float,
    ref_cx: Optional[float] = None,
    ref_cy: Optional[float] = None,
    ref_semi_a: Optional[float] = None,
    polar_equatorial_ratio: float = 1.0,
) -> np.ndarray:
    """Load a raw TIF frame and apply de-rotation to t_reference.

    Args:
        ref_cx, ref_cy, ref_semi_a: Pre-computed disk centre (from reference
            frame).  If provided, these are used instead of per-frame
            ``find_disk_center()`` detection.  Passing a shared centre for all
            frames in a window eliminates warp-centre noise: because each frame
            would otherwise get a slightly different Otsu-detected (cx, cy), the
            post-warp positions differ per frame, the stacked channels have
            slightly different positions, and phase correlation in
            ``align_channels()`` applies an oscillating correction that causes
            GRS jitter in the animation.
        polar_equatorial_ratio: polar_radius / equatorial_radius from ellipse fit.
            1.0 = sphere; ~0.935 = Jupiter.

    Returns float [0, 1] 2-D array.
    """
    img = image_io.read_tif(frame_path)
    if img.ndim == 3:
        img = img.mean(axis=2).astype(np.float32)

    dt_sec = (frame_time - t_reference).total_seconds()
    if abs(dt_sec) < 1.0:
        return img

    if ref_cx is not None and ref_cy is not None and ref_semi_a is not None:
        cx, cy, semi_a = ref_cx, ref_cy, ref_semi_a
    else:
        cx, cy, semi_a, semi_b, _ = find_disk_center(img)
        polar_equatorial_ratio = float(np.clip(semi_b / max(semi_a, 1.0), 0.85, 1.0))

    warped = spherical_derotation_warp(
        img, dt_sec, cx, cy, semi_a,
        period_hours=period_hours,
        scale=warp_scale,
        polar_equatorial_ratio=polar_equatorial_ratio,
    )
    return warped


def _derotate_frame_color(
    frame_path: Path,
    frame_time: datetime,
    t_reference: datetime,
    period_hours: float,
    warp_scale: float,
    ref_cx: Optional[float] = None,
    ref_cy: Optional[float] = None,
    ref_semi_a: Optional[float] = None,
    polar_equatorial_ratio: float = 1.0,
) -> np.ndarray:
    """Load a color TIF and apply de-rotation to each RGB channel.

    The same warp parameters (disk centre + rotation) are applied to all
    three channels so relative channel alignment is preserved.

    Returns float32 [0, 1] array of shape (H, W, 3).
    """
    img = image_io.read_tif(frame_path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=2)
    img = img.astype(np.float32)

    dt_sec = (frame_time - t_reference).total_seconds()
    if abs(dt_sec) < 1.0:
        return img

    lum = img.mean(axis=2)
    if ref_cx is not None and ref_cy is not None and ref_semi_a is not None:
        cx, cy, semi_a = ref_cx, ref_cy, ref_semi_a
        per = polar_equatorial_ratio
    else:
        try:
            cx, cy, semi_a, semi_b, _ = find_disk_center(lum)
            per = float(np.clip(semi_b / max(semi_a, 1.0), 0.85, 1.0))
        except Exception:
            return img   # Can't detect disk — return unwarped

    out = np.zeros_like(img)
    for c in range(3):
        out[:, :, c] = spherical_derotation_warp(
            img[:, :, c], dt_sec, cx, cy, semi_a,
            period_hours=period_hours,
            scale=warp_scale,
            polar_equatorial_ratio=per,
        )
    return out


_CENTER_PREF_ORDER = ["IR", "R", "G", "CH4", "B"]


def _shared_center_derotated(derotated: Dict[str, np.ndarray]) -> None:
    """Shift ALL derotated filter images so the planet disk is centred.

    Uses the highest-quality filter (IR preferred) for disk detection and
    applies the same shift to every channel.  Modifies ``derotated`` in-place.
    """
    if not derotated:
        return

    ref_filt = next(
        (f for f in _CENTER_PREF_ORDER if f in derotated),
        next(iter(derotated)),
    )
    ref_img = derotated[ref_filt]
    h, w = ref_img.shape[:2]

    try:
        cx, cy, semi_a, *_ = find_disk_center(ref_img)
        if semi_a < 5:
            return
        dx = w * 0.5 - cx
        dy = h * 0.5 - cy
        for filt in derotated:
            derotated[filt] = apply_shift(derotated[filt], dx, dy)
    except Exception:
        pass


# ── Sliding-window stacking ────────────────────────────────────────────────────

def _stack_window_frames(
    cycles: List[dict],
    center_idx: int,
    window_n: int,
    quality_scores: Dict[str, Dict[str, float]],
    min_quality: float,
    t_center: datetime,
    period_hours: float,
    warp_scale: float,
    cycle_seconds: float = 0.0,
) -> Tuple[Dict[str, np.ndarray], dict]:
    """Collect, quality-filter, de-rotate, and stack frames for one output frame.

    Args:
        cycles:         All cycle dicts (sorted by time).
        center_idx:     Index of the centre cycle in ``cycles``.
        window_n:       Number of cycles to include in the window.
        quality_scores: {filter: {stem: normalised_score}} — from Pass 0.
        min_quality:    Minimum normalised quality score to include a frame.
        t_center:       Reference time for de-rotation (= centre cycle time).
        period_hours:   Planet rotation period.
        warp_scale:     De-rotation warp scale factor.
        cycle_seconds:  Expected duration of one filter cycle in seconds.
                        When > 0, the window is trimmed at any gap exceeding
                        2× this value, preventing temporally distant cycles
                        (e.g., across an observation break) from entering the
                        stack and introducing large de-rotation residuals.

    Returns:
        (derotated_dict, log_dict) where derotated_dict is {filter: 2-D array}.
    """
    half = window_n // 2
    start = max(0, center_idx - half)
    end   = min(len(cycles), center_idx + half + 1)

    # ── Time-based gap filtering ───────────────────────────────────────────────
    # When cycle_seconds > 0, expand the window outward from the centre cycle
    # one step at a time, stopping as soon as a gap between consecutive cycles
    # exceeds 2× the expected cycle duration.  This prevents cycles separated
    # by an observation gap (e.g. 18 minutes between 12:59 and 13:17) from
    # entering the same window and causing massive de-rotation residuals.
    if cycle_seconds > 0.0:
        max_gap_sec = cycle_seconds * 2.0
        filtered: List[dict] = [cycles[center_idx]]
        # Expand backward
        for i in range(center_idx - 1, start - 1, -1):
            gap = (cycles[i + 1]["center_time"] - cycles[i]["center_time"]).total_seconds()
            if gap > max_gap_sec:
                break
            filtered.insert(0, cycles[i])
        # Expand forward
        for i in range(center_idx + 1, end):
            gap = (cycles[i]["center_time"] - cycles[i - 1]["center_time"]).total_seconds()
            if gap > max_gap_sec:
                break
            filtered.append(cycles[i])
        window_cycles = filtered
    else:
        window_cycles = cycles[start:end]

    # Gather all frame entries per filter across the window
    window_frames: Dict[str, List[Tuple[Path, dict]]] = {}
    for cyc in window_cycles:
        for filt, (path, meta) in cyc["frames"].items():
            if path is not None:
                window_frames.setdefault(filt, []).append((path, meta))

    # Compute actual temporal midpoint from all frame timestamps in the window.
    # The cycle bin-based t_center (= bin_start + cycle_sec/2) may differ from
    # the actual midpoint when the observation cadence is irregular, when there
    # are gaps, or when the window is at the edge of the sequence.
    # Using the actual midpoint minimises the maximum |Δt| for any frame →
    # reduces de-rotation residual and keeps all channels equally displaced
    # in time → more consistent blurriness → phase correlation more reliable.
    all_window_timestamps = [
        meta["timestamp"]
        for entries_list in window_frames.values()
        for _, meta in entries_list
    ]
    if all_window_timestamps:
        t_min = min(all_window_timestamps)
        t_max = max(all_window_timestamps)
        t_center = t_min + (t_max - t_min) / 2   # override bin-based t_center

    # Detect the disk centre once from the centre cycle's highest-quality frame.
    #
    # WHY a shared warp centre matters:
    # _derotate_frame() calls find_disk_center() per frame when no reference is
    # supplied.  Otsu thresholding on low-SNR images (B, CH4) returns (cx, cy)
    # estimates that can differ by several pixels between frames.  As the
    # sliding window moves, the new frame that enters has a slightly different
    # Otsu result → the post-warp position of that frame differs from the rest
    # → after quality-weighted stacking the whole channel's effective position
    # shifts by ~(new frame's error) / N.  Different filters gain/lose different
    # frames as the window slides, so their effective positions shift by
    # different amounts each step → cross-filter position jitter → phase
    # correlation in align_channels() applies a different "correction" every
    # frame → GRS oscillation.
    #
    # Fix: use the same (cx, cy, semi_a) for every warp in this window.
    # All channels are then warped with an identical centre → inter-channel
    # warp-centre noise is zero → channel positions change consistently as
    # windows slide → phase correlation shift stays stable.
    ref_cx: Optional[float] = None
    ref_cy: Optional[float] = None
    ref_semi_a: Optional[float] = None
    ref_polar_eq_ratio: float = 1.0
    center_cycle = window_cycles[len(window_cycles) // 2]
    for pref_filt in _CENTER_PREF_ORDER:
        if pref_filt in center_cycle["frames"]:
            ref_path, _ = center_cycle["frames"][pref_filt]
            if ref_path is not None:
                try:
                    ref_img = image_io.read_tif(ref_path)
                    if ref_img.ndim == 3:
                        ref_img = ref_img.mean(axis=2).astype(np.float32)
                    ref_cx, ref_cy, ref_semi_a, ref_semi_b, _ = find_disk_center(ref_img)
                    if ref_semi_a >= 5:
                        ref_polar_eq_ratio = float(
                            np.clip(ref_semi_b / max(ref_semi_a, 1.0), 0.85, 1.0)
                        )
                        break          # reliable detection found
                    ref_cx = ref_cy = ref_semi_a = None
                except Exception:
                    ref_cx = ref_cy = ref_semi_a = None

    stacked: Dict[str, np.ndarray] = {}
    log: dict = {"window": [c["center_time"].strftime("%H:%M") for c in window_cycles],
                 "t_center_actual": t_center.strftime("%H:%M:%S"),
                 "filters": {}}

    for filt, entries in window_frames.items():
        # Quality-weighted stack: use ALL frames in the window; bad frames get a
        # low weight so they contribute minimally without being fully discarded.
        #
        # WHY not hard quality-threshold rejection:
        # When IR passes many frames (high SNR) but R/G/B pass only one or zero,
        # the resulting IR stack is a smooth multi-frame average while the other
        # channels are single, noisy frames.  Phase correlation between a blurry
        # IR stack and a sharp single-frame R/G/B image is unreliable — it finds
        # spurious cross-correlation peaks and applies a wrong shift, producing
        # visible colour fringing in the composite (several px offset).
        # By including all frames for every filter with quality weighting, each
        # channel's stack has the same temporal distribution → no differential
        # blurriness → phase correlation works correctly.

        n_total = len(entries)
        scored: List[Tuple[Path, dict, float]] = []
        n_below = 0
        for path, meta in entries:
            stem = meta["stem"]
            raw_score = quality_scores.get(filt, {}).get(stem, 1.0)
            scored.append((path, meta, raw_score))
            if raw_score < min_quality:
                n_below += 1

        # Weight: quality score ^ 2 so poor frames are strongly down-weighted.
        # Floor at 0.05 so even genuinely bad frames contribute a tiny bit
        # (avoids divide-by-zero and ensures all filters cover the same time span).
        weights: List[float] = [max(s ** 2, 0.05) for _, _, s in scored]
        w_sum = sum(weights)
        weights = [w / w_sum for w in weights]

        # De-rotate and apply per-frame residual correction, then accumulate.
        #
        # spherical_derotation_warp() applies only `warp_scale` (default 0.20)
        # of the full rotation correction.  The remaining (1 − warp_scale) = 80%
        # is applied here as a per-frame rigid horizontal shift BEFORE stacking.
        #
        # WHY per-frame (not post-stack) correction matters:
        # Post-stack correction must use the quality-weighted average dt, which
        # changes each time a dominant frame enters or exits the sliding window.
        # For example, if one B frame has quality score 1.0 while the others are
        # ~0.1, it determines ~83 % of the stack.  As this frame moves from the
        # rightmost to the leftmost position of the window (over 5 output frames),
        # the quality-weighted avg_dt swings from +480 s to −480 s — a total
        # 960 s shift that becomes a 6 px jump in the B channel correction the
        # moment this frame exits the window.  Applying the correction per frame
        # BEFORE stacking gives each frame exactly the right correction regardless
        # of its quality weight, and the accumulated stack is then already centred
        # on t_center with zero systematic residual.
        omega_rad_s = 2.0 * np.pi / (period_hours * 3600.0)
        planes: List[np.ndarray] = []
        for (path, meta, score), w in zip(scored, weights):
            warped = _derotate_frame(path, meta["timestamp"], t_center,
                                     period_hours, warp_scale,
                                     ref_cx, ref_cy, ref_semi_a,
                                     polar_equatorial_ratio=ref_polar_eq_ratio)
            # Per-frame residual: apply the remaining (1 − warp_scale) fraction
            # as a rigid shift so that each frame is fully de-rotated to t_center.
            if ref_semi_a is not None and ref_semi_a > 5:
                frame_dt = (meta["timestamp"] - t_center).total_seconds()
                if abs(frame_dt) > 0.5:
                    delta_lambda = frame_dt * omega_rad_s
                    per_frame_dx = (1.0 - warp_scale) * delta_lambda * ref_semi_a
                    warped = apply_shift(warped, per_frame_dx, 0.0)
            planes.append(warped * w)

        stacked[filt] = np.sum(planes, axis=0).astype(np.float32)

        # Log the quality-weighted avg_dt (informational only — no longer used
        # for a post-stack correction since each frame is already fully corrected).
        avg_dt_sec = sum(
            w * (meta["timestamp"] - t_center).total_seconds()
            for (_, meta, _score), w in zip(scored, weights)
        )

        log["filters"][filt] = {
            "n_used":    n_total,
            "n_below_threshold": n_below,
            "avg_dt_sec": round(avg_dt_sec, 1),
            "scores": [round(s, 3) for _, _, s in scored],
        }

    return stacked, log


# ── Color camera series ────────────────────────────────────────────────────────

def _run_color_series(
    config: PipelineConfig,
    results_03: Dict[str, List[Tuple[Optional[Path], dict]]],
    progress_callback=None,
) -> Dict[str, List[Tuple[Optional[Path], str]]]:
    """Step 8 for color camera: group → stack → sharpen → auto-correct → save.

    Unlike the mono path (which composites multiple filter channels), the color
    path receives a single color (Bayer or RGB) stream per time bin.  Each output
    frame is produced by:
      1. Grouping color TIFs into time bins of cycle_seconds width.
      2. Collecting N consecutive bins (sliding window).
      3. De-rotating each frame to the bin centre time.
      4. Mean-stacking all de-rotated frames (SNR ∝ √N).
      5. Wavelet sharpening (color-aware, same series_amounts as mono).
      6. Auto white balance + chromatic aberration correction per frame.
      7. Brightness scaling by series_scale.
    """
    from pipeline.steps.step07_rgb_composite import _auto_color_correct

    print("  [Color] Loading color TIF files from input_dir...")
    all_frames: List[Tuple[Path, dict]] = []
    for tif_path in sorted(config.input_dir.rglob("*.tif")):
        meta = image_io.parse_filename(tif_path)
        if meta is None:
            continue
        all_frames.append((tif_path, meta))
    all_frames.sort(key=lambda x: x[1]["timestamp"])

    if not all_frames:
        print("  [WARNING] No TIF files found — Step 8 (color) skipped.")
        return {}

    # ── Group into time bins ───────────────────────────────────────────────────
    cycle_sec = config.composite.cycle_seconds
    t_start = all_frames[0][1]["timestamp"]
    t_end   = all_frames[-1][1]["timestamp"]

    bins: List[List[Tuple[Path, dict]]] = []
    t = t_start
    while t <= t_end + timedelta(seconds=cycle_sec / 2.0):
        t_bin_end = t + timedelta(seconds=cycle_sec)
        group = [(p, m) for p, m in all_frames if t <= m["timestamp"] < t_bin_end]
        if group:
            bins.append(group)
        t = t_bin_end

    if not bins:
        print("  [WARNING] No time bins formed — Step 8 (color) skipped.")
        return {}

    window_n  = max(1, config.composite.stack_window_n)
    period    = config.derotation.rotation_period_hours
    warp_sc   = config.derotation.warp_scale

    print(f"  Found {len(all_frames)} color frames → {len(bins)} bins "
          f"(cycle={cycle_sec:.0f}s, window={window_n})")

    # ── Output directory ───────────────────────────────────────────────────────
    out_base: Optional[Path] = None
    if config.save_step08:
        out_base = config.step_dir(8, "series")
        out_base.mkdir(parents=True, exist_ok=True)
        print(f"  Output → {out_base}")
    else:
        print("  save_step08=False: results not written to disk")

    all_results: Dict[str, List[Tuple[Optional[Path], str]]] = {}
    total_written = 0

    for frame_idx, center_bin in enumerate(bins, start=1):
        mid = len(center_bin) // 2
        t_center   = center_bin[mid][1]["timestamp"]
        t_str      = t_center.strftime("%Y-%m-%d_%H-%M")
        frame_label = f"frame_{frame_idx:03d}_{t_str}"

        # Collect sliding-window frames
        half = window_n // 2
        lo   = max(0, frame_idx - 1 - half)
        hi   = min(len(bins), frame_idx - 1 + half + 1)
        window_frames: List[Tuple[Path, dict]] = [
            fr for b in bins[lo:hi] for fr in b
        ]

        if not window_frames:
            all_results[frame_label] = [(None, "COLOR")]
            continue

        # ── Shared disk centre from centre bin ────────────────────────────────
        ref_path    = center_bin[0][0]
        ref_raw     = image_io.read_tif(ref_path)
        ref_lum     = (ref_raw.mean(axis=2) if ref_raw.ndim == 3 else ref_raw).astype(np.float32)
        ref_cx = ref_cy = ref_semi_a = None
        per = 1.0
        try:
            cx, cy, semi_a, semi_b, _ = find_disk_center(ref_lum)
            if semi_a >= 5:
                ref_cx, ref_cy, ref_semi_a = cx, cy, semi_a
                per = float(np.clip(semi_b / max(semi_a, 1.0), 0.85, 1.0))
        except Exception:
            pass

        # ── De-rotate and stack ────────────────────────────────────────────────
        stacked_frames: List[np.ndarray] = []
        for tif_path, meta in window_frames:
            warped = _derotate_frame_color(
                tif_path, meta["timestamp"], t_center, period, warp_sc,
                ref_cx=ref_cx, ref_cy=ref_cy, ref_semi_a=ref_semi_a,
                polar_equatorial_ratio=per,
            )
            stacked_frames.append(warped)

        stacked = np.mean(stacked_frames, axis=0).astype(np.float32)

        # Centre the disk
        if ref_cx is not None and ref_semi_a is not None and ref_semi_a >= 5:
            h, w = stacked.shape[:2]
            dx = w * 0.5 - ref_cx
            dy = h * 0.5 - ref_cy
            for c in range(3):
                stacked[:, :, c] = apply_shift(stacked[:, :, c], dx, dy)

        # ── Wavelet sharpening (color-aware) ───────────────────────────────────
        lum2 = stacked.mean(axis=2)
        has_disk = False
        cx2 = cy2 = sr2 = ry2 = angle2 = 0.0
        try:
            cx2, cy2, sr2, ry2, angle2 = find_disk_center(lum2)
            has_disk = sr2 >= 5
        except Exception:
            pass

        if has_disk:
            _angle2_rad = np.radians(angle2)
            if config.wavelet.auto_params:
                _s8c_eff, _s8c_expand = wavelet_module.auto_wavelet_params(
                    lum2, cx2, cy2, sr2, ry2, _angle2_rad
                )
            else:
                _s8c_eff    = config.wavelet.series_edge_feather_factor
                _s8c_expand = config.wavelet.disk_expand_px
            sharpened = wavelet_module.sharpen_color_disk_aware(
                stacked, cx2, cy2, sr2,
                levels=config.wavelet.levels,
                amounts=config.wavelet.series_amounts,
                power=config.wavelet.series_power,
                sharpen_filter=config.wavelet.series_sharpen_filter,
                edge_feather_factor=_s8c_eff,
                ry=ry2, angle=_angle2_rad,
                expand_px=_s8c_expand,
            )
        else:
            sharpened = wavelet_module.sharpen_color(
                stacked,
                levels=config.wavelet.levels,
                amounts=config.wavelet.series_amounts,
                power=config.wavelet.series_power,
                sharpen_filter=config.wavelet.series_sharpen_filter,
            )

        # ── Auto WB + CA correction ────────────────────────────────────────────
        corrected, params = _auto_color_correct(sharpened)

        # ── Brightness scale ───────────────────────────────────────────────────
        corrected = np.clip(
            corrected * config.composite.series_scale, 0.0, 1.0
        ).astype(np.float32)

        # ── Save ───────────────────────────────────────────────────────────────
        out_path: Optional[Path] = None
        if out_base is not None:
            frame_out_dir = out_base / frame_label
            frame_out_dir.mkdir(exist_ok=True)
            out_path = frame_out_dir / "COLOR_composite.png"
            image_io.write_png_color_16bit(corrected, out_path)
            total_written += 1

        all_results[frame_label] = [(out_path, "COLOR")]

        if frame_idx % 5 == 0 or frame_idx == len(bins):
            print(f"  [{frame_idx:>3}/{len(bins)}] {t_str}  "
                  f"stack={len(stacked_frames)}  "
                  f"R×{params['r_gain']:.3f} B×{params['b_gain']:.3f}")
        if progress_callback is not None:
            progress_callback(frame_idx, len(bins))

    print(f"\n  Step 8 (color) complete: {total_written} color PNGs written")
    return all_results


# ── Main step ─────────────────────────────────────────────────────────────────

def run(
    config: PipelineConfig,
    results_03: Dict[str, List[Tuple[Optional[Path], dict]]],
    progress_callback=None,
) -> Dict[str, List[Tuple[Optional[Path], str]]]:
    """Run Step 8 for all filter-cycle sets found in the raw TIF input.

    Args:
        config:      Pipeline configuration.
        results_03:  Unused for mono camera (Step 3 PNGs are bypassed).
                     The mono path loads raw TIFs from config.input_dir
                     (stack → sharpen workflow).  For color camera mode,
                     results_03 is also unused; color TIFs are loaded from
                     config.input_dir directly.

    Returns:
        ``{frame_label: [(composite_path_or_None, composite_name), ...]}``
    """
    # Color camera: single-stream stacking + auto WB/CA correction
    if config.camera_mode == "color":
        print("  Color camera mode: stack → sharpen → auto-correct per frame")
        return _run_color_series(config, results_03, progress_callback)

    # Step 8 now reads raw TIFs directly, applying wavelet sharpening AFTER
    # stacking (stack → sharpen → composite).  This is physically correct:
    # stacking first improves SNR, then sharpening acts on the high-SNR stack.
    # results_03 (wavelet-sharpened PNGs from Step 3) is intentionally bypassed.
    print("  [INFO] Loading raw TIFs from input_dir (stack→sharpen workflow)...")
    raw_tif_frames = _load_raw_tifs(config)

    if not raw_tif_frames:
        print("  [WARNING] No raw TIF files found in input_dir — Step 8 skipped.")
        return {}

    # ── Group into filter cycles ───────────────────────────────────────────────
    # Use step-8-specific cycle_seconds (CompositeConfig), not step-4's QualityConfig.
    cycle_minutes = config.composite.cycle_seconds / 60.0
    # Determine which filters are required for compositing.  When series_specs
    # are configured, derive required channels from those specs; otherwise fall
    # back to the classic R/G/B set so that existing sessions are unaffected.
    active_specs = config.composite.series_specs or config.composite.specs
    if config.composite.series_specs:
        required_rgb = list({
            ch
            for spec in active_specs
            for ch in (spec.R, spec.G, spec.B, spec.L or "")
            if ch
        } & raw_tif_frames.keys())
    else:
        required_rgb = [f for f in ["R", "G", "B"] if f in raw_tif_frames]
    cycles = _group_into_cycles(raw_tif_frames, cycle_minutes, required_rgb)

    if not cycles:
        print("  [WARNING] No valid filter cycles found — Step 8 skipped.")
        return {}

    window_n    = max(1, config.composite.stack_window_n)
    min_quality = config.composite.stack_min_quality

    print(f"  Found {len(cycles)} filter-cycle sets  "
          f"(cycle={cycle_minutes:.2f} min, required: {required_rgb})")
    if window_n > 1:
        print(f"  Sliding-window stacking: N={window_n}, min_quality={min_quality:.2f}")
    else:
        print("  Single-frame mode (stack_window_n=1)")

    # ── Output directory ───────────────────────────────────────────────────────
    out_base: Optional[Path] = None
    if config.save_step08:
        out_base = config.step_dir(8, "series")
        out_base.mkdir(parents=True, exist_ok=True)
        print(f"  Output → {out_base}")
    else:
        print("  save_step08=False: results not written to disk")

    # Use series-specific specs when set (Step 8 GUI), else fall back to Step 7 specs.
    specs  = config.composite.series_specs or config.composite.specs
    align  = config.composite.align_channels
    plow   = config.composite.stretch_plow
    phigh  = config.composite.stretch_phigh
    period = config.derotation.rotation_period_hours
    scale  = config.derotation.warp_scale

    print(f"  Composites: {[s.name for s in specs]}")

    # ── Pass 0 (quality scoring — only when window_n > 1 and min_quality > 0) ─
    quality_scores: Dict[str, Dict[str, float]] = {}
    if window_n > 1 and min_quality > 0.0:
        print("  [Pass 0] 품질 점수 계산 중...")
        quality_scores = _compute_filter_quality_scores(raw_tif_frames)
        for filt, scores in quality_scores.items():
            if scores:
                mean_q = sum(scores.values()) / len(scores)
                below  = sum(1 for s in scores.values() if s < min_quality)
                print(f"    {filt}: {len(scores)} frames, "
                      f"mean={mean_q:.3f}, below_threshold={below}")
        print("  [Pass 0] 완료")

    # ── Pass 1 cache (global_filter_normalize=True) ───────────────────────────
    # When global_filter_normalize is on we need to normalise every frame using
    # statistics derived from ALL frames' post-wavelet data.  We therefore do a
    # two-phase approach:
    #   Phase A (main loop below): derotate + stack + wavelet → cache in memory
    #   Phase B (post-loop):       compute global lo/hi → normalise → compose/save
    # When global_filter_normalize is off the main loop does everything in one
    # pass and _pw_cache stays empty.
    _RGB_COLOUR_FILTS = {"R", "G", "B"}
    _pw_cache: List[dict] = []   # filled only when global_filter_normalize=True

    all_results: Dict[str, List[Tuple[Optional[Path], str]]] = {}
    total_written = 0
    summary_lines = [
        f"=== Step 8 Series Summary ===",
        f"stack_window_n={window_n}  min_quality={min_quality:.2f}\n",
    ]

    for frame_idx, cycle in enumerate(cycles, start=1):
        t_center    = cycle["center_time"]
        t_str       = t_center.strftime("%Y-%m-%d_%H-%M")
        frame_label = f"frame_{frame_idx:03d}_{t_str}"

        # ── Per-frame output directory ─────────────────────────────────────────
        frame_out_dir: Optional[Path] = None
        if out_base is not None:
            frame_out_dir = out_base / frame_label
            frame_out_dir.mkdir(exist_ok=True)

        # ── Collect, de-rotate, and stack frames ───────────────────────────────
        frame_log: dict = {"center_time": t_str}

        if window_n == 1:
            # Single-frame path (original behaviour)
            derotated: Dict[str, np.ndarray] = {}
            flog_filters: dict = {}
            for filt, (png_path, meta) in cycle["frames"].items():
                frame_time = meta["timestamp"]
                dt = (frame_time - t_center).total_seconds()
                warped = _derotate_frame(png_path, frame_time, t_center, period, scale)
                derotated[filt] = warped
                flog_filters[filt] = {
                    "stem": meta["stem"],
                    "timestamp": frame_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "dt_sec": round(dt, 1),
                }
            frame_log["filters"] = flog_filters
        else:
            # Sliding-window stacking path
            center_idx = frame_idx - 1   # 0-based index into cycles
            derotated, stack_log = _stack_window_frames(
                cycles, center_idx, window_n, quality_scores, min_quality,
                t_center, period, scale,
                cycle_seconds=config.composite.cycle_seconds,
            )
            frame_log.update(stack_log)

        # ── Shared centering ───────────────────────────────────────────────────
        _shared_center_derotated(derotated)

        # ── Wavelet sharpening (stack → sharpen → composite) ──────────────────
        # Applied after stacking so sharpening acts on the high-SNR stack,
        # not on noisy individual frames.  Uses master_amounts/power settings.

        # Elliptical disk-aware sharpening: detect disk ellipse from a reference
        # filter and use it for all filters in this window.  Feather zone follows
        # Jupiter's actual oblate shape, preventing over-blur at the equatorial
        # limb while still suppressing de-rotation coverage gradient ringing.
        _disk_cx = _disk_cy = _disk_rx = _disk_ry = _disk_angle = None
        for _ref_filt in _CENTER_PREF_ORDER:
            if _ref_filt in derotated:
                try:
                    _mc, _my, _mrx, _mry, _ma = find_disk_center(derotated[_ref_filt])
                    if _mrx >= 5:
                        _disk_cx, _disk_cy = _mc, _my
                        _disk_rx, _disk_ry, _disk_angle = _mrx, _mry, _ma
                        break
                except Exception:
                    pass

        for _filt in list(derotated.keys()):
            _img = derotated[_filt]
            if config.wavelet.border_taper_px > 0:
                _t, _b, _l, _r = wavelet_module.safe_taper_widths(
                    _img, config.wavelet.border_taper_px
                )
                _img = wavelet_module.border_taper(_img, top=_t, bottom=_b, left=_l, right=_r)

            if _disk_cx is not None:
                # find_disk_center returns angle in degrees; convert to radians
                _disk_angle_rad = np.radians(_disk_angle)
                if config.wavelet.auto_params:
                    _s8m_eff, _s8m_expand = wavelet_module.auto_wavelet_params(
                        _img, _disk_cx, _disk_cy, _disk_rx, _disk_ry,
                        _disk_angle_rad
                    )
                else:
                    _s8m_eff    = config.wavelet.series_edge_feather_factor
                    _s8m_expand = config.wavelet.disk_expand_px
                _sharpened = wavelet_module.sharpen_disk_aware(
                    _img, _disk_cx, _disk_cy, _disk_rx,
                    levels=config.wavelet.levels,
                    amounts=config.wavelet.series_amounts,
                    power=config.wavelet.series_power,
                    sharpen_filter=config.wavelet.series_sharpen_filter,
                    edge_feather_factor=_s8m_eff,
                    ry=_disk_ry, angle=_disk_angle_rad,
                    expand_px=_s8m_expand,
                )
            else:
                _sharpened = wavelet_module.sharpen(
                    _img,
                    levels=config.wavelet.levels,
                    amounts=config.wavelet.series_amounts,
                    power=config.wavelet.series_power,
                    sharpen_filter=config.wavelet.series_sharpen_filter,
                )
            derotated[_filt] = _sharpened

        # ── Global normalisation: cache post-wavelet data for Phase B ────────
        # Pass 1 statistics must be computed from post-wavelet data so that the
        # normalization reference matches what compose() would see.  We cache
        # the post-wavelet arrays here and defer compose+save to the post-loop
        # Phase B block below, where global lo/hi can be calculated from ALL
        # frames at once.
        if config.composite.global_filter_normalize:
            _pw_cache.append({
                "cycle":          cycle,
                "frame_idx":      frame_idx,
                "frame_label":    frame_label,
                "t_str":          t_str,
                "frame_out_dir":  frame_out_dir,
                "derotated":      derotated,
                "frame_log":      frame_log,
            })
            if frame_idx % 10 == 0 or frame_idx == len(cycles):
                print(f"  [{frame_idx:>3}/{len(cycles)}] {t_str}  캐시됨")
            if progress_callback is not None:
                progress_callback(frame_idx, len(cycles))
            continue   # compose+save happens in Phase B below

        # ── Composite each spec ────────────────────────────────────────────────
        frame_results: List[Tuple[Optional[Path], str]] = []

        # ── Per-filter monochrome frames (optional) ────────────────────────────
        if config.composite.save_mono_frames and frame_out_dir is not None:
            for filt in sorted(derotated.keys()):
                mono_path = frame_out_dir / f"{filt}_mono.png"
                image_io.write_png_16bit(derotated[filt], mono_path)
                frame_results.append((mono_path, f"{filt}_mono"))
                total_written += 1

        for spec in specs:
            required = {spec.R, spec.G, spec.B}
            if spec.L is not None:
                required.add(spec.L)

            if not required.issubset(derotated.keys()):
                missing = required - derotated.keys()
                frame_results.append((None, spec.name))
                frame_log.setdefault("skipped", []).append(
                    f"{spec.name}: missing {missing}"
                )
                continue

            try:
                # For sliding-window stacks (N > 1), disable phase-correlation
                # channel alignment.  align_channels() uses subpixel_align()
                # (cv2.phaseCorrelate) to detect inter-channel offsets and apply
                # a shift to each non-reference channel.  This works reliably for
                # single-frame composites (N=1) where all channels have the same
                # sharpness level.
                #
                # For stacked composites the detected shift is unstable across
                # output frames because:
                # 1. IR is an optional filter — some cycles may lack it, causing
                #    the IR stack to have fewer frames than R/G/B and therefore
                #    different (lower) blurriness level in those windows.
                # 2. Even with a shared warp centre, the blurriness ratio between
                #    the IR stack (luminance reference) and colour stacks changes
                #    when frames enter/leave the window with unequal per-filter
                #    frame counts.
                # 3. Phase correlation between images of unequal blurriness
                #    produces an unreliable cross-correlation peak that varies
                #    frame-to-frame by 1-3 px — causing the detected "correction"
                #    to oscillate, which in turn makes the GRS appear to jump.
                #
                # After de-rotation with a shared warp centre and
                # _shared_center_derotated(), all channels are already spatially
                # consistent within the tolerance of the intra-cycle timing
                # residual (~0.5-0.75 px for warp_scale=0.20).  This small fixed
                # offset is far less distracting than the 1-3 px oscillating
                # correction that align_channels introduces.
                #
                # N=1 mode is unaffected — single-frame composites use the
                # config value (True by default) because all channels are sharp
                # and uniform, making phase correlation reliable.
                _align = align if window_n == 1 else False
                # When global_filter_normalize is on the channels are
                # already rescaled by Pass 1.  A second stretch would
                # double-clip and brighten the result relative to Step 7.
                comp_img, clog = comp_module.compose(
                    spec,
                    {k: derotated[k] for k in required},
                    align=_align,
                    max_shift_px=config.composite.max_shift_px,
                    color_stretch_mode=config.composite.color_stretch_mode,
                    stretch_plow=plow,
                    stretch_phigh=phigh,
                )
            except Exception as exc:
                frame_results.append((None, spec.name))
                frame_log.setdefault("errors", []).append(
                    f"{spec.name}: {exc}"
                )
                continue

            # Apply brightness scale
            series_scale = config.composite.series_scale
            if abs(series_scale - 1.0) > 1e-6:
                comp_img = (comp_img * series_scale).clip(0.0, 1.0).astype(np.float32)

            out_path: Optional[Path] = None
            if frame_out_dir is not None:
                out_path = frame_out_dir / f"{spec.name}_composite.png"
                image_io.write_png_color_16bit(comp_img, out_path)
                total_written += 1

            frame_results.append((out_path, spec.name))

        all_results[frame_label] = frame_results

        # Summary line
        filter_list = ", ".join(
            f"{f}@{cycle['frames'][f][1]['timestamp'].strftime('%H:%M')}"
            for f in cycle["frames"]
        )
        composites_ok = sum(1 for p, _ in frame_results if p is not None)
        summary_lines.append(
            f"Frame {frame_idx:03d}  {t_str}  "
            f"composites={composites_ok}/{len(specs)}  [{filter_list}]"
        )

        if frame_idx % 10 == 0 or frame_idx == len(cycles):
            print(f"  [{frame_idx:>3}/{len(cycles)}] {t_str}  "
                  f"{composites_ok}/{len(specs)} composites written")

        if frame_out_dir is not None:
            with open(frame_out_dir / "frame_log.json", "w") as f:
                json.dump(frame_log, f, indent=2)

        if progress_callback is not None:
            progress_callback(frame_idx, len(cycles))

    # ── Phase B: global normalise + compose + save (global_filter_normalize=True)
    if _pw_cache:
        # Compute per-filter global mean and std from ALL frames' post-wavelet
        # pixel data.  Mean-std matching (z-score normalisation) is used instead
        # of percentile stretch because:
        #   • Percentile stretch always maps the Nth-percentile to 1.0, forcing
        #     the brightest pixels to saturate regardless of their true value.
        #   • Mean-std matching only adjusts the centre and spread of each
        #     frame's distribution to match the global reference, preserving
        #     within-frame contrast and avoiding forced saturation.
        # RGB filters share one joint global mean/std so their relative colour
        # balance is preserved across the normalisation.
        print("  [Pass 1] post-wavelet 글로벌 통계 계산 중 (mean-std 기반)...")
        _pix2: Dict[str, list] = {}
        for _entry in _pw_cache:
            for _f, _arr in _entry["derotated"].items():
                _pix2.setdefault(_f, []).append(_arr.ravel())

        # filter_stats stores (global_mean, global_std) per filter
        filter_stats: Dict[str, Tuple[float, float]] = {}
        _rgb_present = _RGB_COLOUR_FILTS & _pix2.keys()
        if _rgb_present:
            _combined = np.concatenate(
                [np.concatenate(_pix2[_f]) for _f in _rgb_present]
            )
            _g_mean = float(_combined.mean())
            _g_std  = float(_combined.std())
            if _g_std < 1e-7:
                _g_std = 1e-7
            for _f in _rgb_present:
                filter_stats[_f] = (_g_mean, _g_std)
            print(f"    RGB (joint): mean={_g_mean:.5f}  std={_g_std:.5f}")
        for _f, _plists in _pix2.items():
            if _f in _RGB_COLOUR_FILTS:
                continue
            _vals   = np.concatenate(_plists)
            _g_mean = float(_vals.mean())
            _g_std  = float(_vals.std())
            if _g_std < 1e-7:
                _g_std = 1e-7
            filter_stats[_f] = (_g_mean, _g_std)
            print(f"    {_f}: mean={_g_mean:.5f}  std={_g_std:.5f}")
        print(f"  [Pass 1] 완료 — {len(filter_stats)}개 필터 글로벌 분포 확정")

        print("  [Pass 2] 글로벌 정규화 + 합성 중...")
        for _entry in _pw_cache:
            cycle         = _entry["cycle"]
            frame_idx     = _entry["frame_idx"]
            frame_label   = _entry["frame_label"]
            t_str         = _entry["t_str"]
            frame_out_dir = _entry["frame_out_dir"]
            derotated     = _entry["derotated"]
            frame_log     = _entry["frame_log"]

            # Mean-std matching: shift and scale each frame so its mean and std
            # match the global reference, then clip to [0, 1].
            # Formula: out = (frame - frame_μ) / frame_σ * global_σ + global_μ
            for _filt in list(derotated.keys()):
                if _filt in filter_stats:
                    _g_mean2, _g_std2 = filter_stats[_filt]
                    _arr = derotated[_filt]
                    _f_mean = float(_arr.mean())
                    _f_std  = float(_arr.std())
                    if _f_std < 1e-7:
                        _f_std = 1e-7
                    derotated[_filt] = np.clip(
                        (_arr - _f_mean) / _f_std * _g_std2 + _g_mean2,
                        0.0, 1.0,
                    ).astype(np.float32)

            frame_results: List[Tuple[Optional[Path], str]] = []

            if config.composite.save_mono_frames and frame_out_dir is not None:
                for _filt in sorted(derotated.keys()):
                    _mono_path = frame_out_dir / f"{_filt}_mono.png"
                    image_io.write_png_16bit(derotated[_filt], _mono_path)
                    frame_results.append((_mono_path, f"{_filt}_mono"))
                    total_written += 1

            for spec in specs:
                _required = {spec.R, spec.G, spec.B}
                if spec.L is not None:
                    _required.add(spec.L)
                if not _required.issubset(derotated.keys()):
                    frame_results.append((None, spec.name))
                    frame_log.setdefault("skipped", []).append(
                        f"{spec.name}: missing {_required - derotated.keys()}"
                    )
                    continue
                try:
                    _align2 = align if window_n == 1 else False
                    _comp_img, _ = comp_module.compose(
                        spec,
                        {k: derotated[k] for k in _required},
                        align=_align2,
                        max_shift_px=config.composite.max_shift_px,
                        color_stretch_mode=config.composite.color_stretch_mode,
                        stretch_plow=0.0,    # already normalised; no re-stretch
                        stretch_phigh=100.0,
                    )
                except Exception as _exc:
                    frame_results.append((None, spec.name))
                    frame_log.setdefault("errors", []).append(
                        f"{spec.name}: {_exc}"
                    )
                    continue
                _series_scale = config.composite.series_scale
                if abs(_series_scale - 1.0) > 1e-6:
                    _comp_img = (_comp_img * _series_scale).clip(0.0, 1.0).astype(np.float32)
                _out_path: Optional[Path] = None
                if frame_out_dir is not None:
                    _out_path = frame_out_dir / f"{spec.name}_composite.png"
                    image_io.write_png_color_16bit(_comp_img, _out_path)
                    total_written += 1
                frame_results.append((_out_path, spec.name))

            all_results[frame_label] = frame_results

            _filter_list = ", ".join(
                f"{_f}@{cycle['frames'][_f][1]['timestamp'].strftime('%H:%M')}"
                for _f in cycle["frames"]
            )
            _composites_ok = sum(1 for _p, _ in frame_results if _p is not None)
            summary_lines.append(
                f"Frame {frame_idx:03d}  {t_str}  "
                f"composites={_composites_ok}/{len(specs)}  [{_filter_list}]"
            )
            if frame_idx % 10 == 0 or frame_idx == len(cycles):
                print(f"  [{frame_idx:>3}/{len(cycles)}] {t_str}  "
                      f"{_composites_ok}/{len(specs)} composites written")
            if frame_out_dir is not None:
                with open(frame_out_dir / "frame_log.json", "w") as _fh:
                    json.dump(frame_log, _fh, indent=2)
            if progress_callback is not None:
                progress_callback(frame_idx, len(cycles))

    # ── Save summary ───────────────────────────────────────────────────────────
    summary_text = "\n".join(summary_lines)
    if out_base is not None:
        txt_path = out_base / "series_summary.txt"
        txt_path.write_text(summary_text)
        print(f"  → {txt_path}")

    print(f"\n  Step 8 complete: {total_written} series PNGs written "
          f"({len(cycles)} frames × up to {len(specs)} composites)")
    return all_results
