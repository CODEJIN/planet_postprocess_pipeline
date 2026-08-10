"""
Step 4 – De-rotation stacking.

For each selected time window from Step 3:
  1. Pre-scan all windows to detect session-median image-space pole PA.
  2. Auto-detect camera orientation (flip_ns) from pole PA sign.
  3. Apply spherical de-rotation warp using the correct pole PA.
  4. Sub-pixel translate-align rotated frames via phase correlation.
  5. Combine with quality-weighted mean stack (weights = Step 3 norm_scores).
  6. If config.satellite.enabled: predict Galilean moon/shadow positions via
     JPL Horizons + Skyfield, refine with CV blob detection, and log positions.
  7. If config.satellite.composite_enabled: apply multi-rate compositing (exp9
     method) — overwrite planet TIFs with Europa+shadow composited stacks.

Output (when config.save_step04 is True):
    <output_base>/step04_derotated/
        window_01/
            IR_derotated.tif
            R_derotated.tif
            ...
            derotation_log.json   ← includes satellite positions when enabled
        derotation_summary.txt
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from pipeline.config import PipelineConfig
from pipeline.modules import derotation, image_io
from pipeline.modules.derotation import (
    auto_detect_ns_flip,
    auto_detect_pole_axis_flip,
    auto_detect_equator_pa,
    find_disk_center,
    equator_pa_from_disk_ellipse,
    query_horizons_np_ang,
    query_horizons_sub_observer_lat,
    spherical_derotation_warp,
    spherical_derotation_warp_3d,
)
from pipeline.steps.satellite_composite import (
    SatelliteTracker,
    _apply_satellite_composite,
    compute_session_disk_radius,
    resolve_tracker_flip_ns,
    run_plate_scale_calibration,
)

# Mono filters in priority order; "color" is appended for color-camera sessions.
_FILT_PREF      = ["IR", "R", "G", "B", "CH4"]
_FILT_PREF_EXT  = ["IR", "R", "G", "B", "CH4", "color"]



def _scan_session_pole_pa(
    scores: dict,
    config: PipelineConfig,
) -> Optional[float]:
    """Return the session-median image-space pole PA from all input frames.

    Iterates every frame in the preferred filter (from step03 scores dict,
    which covers all input TIFs regardless of window selection), computes
    per-frame belt-gradient PA, and returns the median.
    """
    # Pick the first preferred filter that has any scored frames.
    filt = next((f for f in _FILT_PREF_EXT if scores.get(f)), None)
    if filt is None:
        return None

    all_rows: List[dict] = sorted(scores[filt], key=lambda r: r["timestamp"])
    print(f"  [pole_pa] Pre-scanning {len(all_rows)} frame(s) for image-space pole PA…")

    # Detect disk geometry once from the middle frame (stable across session).
    mid_row = all_rows[len(all_rows) // 2]
    try:
        mid_raw = image_io.read_tif(mid_row["path"])
        mid_lum = mid_raw if mid_raw.ndim == 2 else mid_raw.mean(axis=2).astype(np.float32)
        cx, cy, semi_a, *_ = find_disk_center(mid_lum)
        if semi_a < 5:
            raise ValueError("disk too small")
    except Exception as exc:
        warnings.warn(f"  [pole_pa] disk detection failed: {exc}")
        return None

    raw_pas: List[float] = []
    for i, row in enumerate(all_rows):
        try:
            raw = image_io.read_tif(row["path"])
            lum = raw if raw.ndim == 2 else raw.mean(axis=2).astype(np.float32)
            pa = auto_detect_equator_pa(frames=[lum], cx=cx, cy=cy, disk_radius_px=semi_a)
            print(f"    frame {i+1}/{len(all_rows)}: raw pole_pa = {pa:.1f}° via {filt} [belt_gradient]")
            raw_pas.append(pa)
        except Exception as exc:
            try:
                raw = image_io.read_tif(row["path"])
                lum = raw if raw.ndim == 2 else raw.mean(axis=2).astype(np.float32)
                pa = equator_pa_from_disk_ellipse(lum)
                if pa is not None:
                    print(f"    frame {i+1}/{len(all_rows)}: raw pole_pa = {pa:.1f}° via {filt} [disk_ellipse]")
                    raw_pas.append(pa)
            except Exception:
                pass

    if not raw_pas:
        return None

    session_pa = float(np.median(raw_pas))
    raw_str = [f"{p:.1f}" for p in raw_pas]
    print(
        f"  [pole_pa] session pole_pa = {session_pa:.1f}° "
        f"(n={len(raw_pas)}, raw: {raw_str})"
    )
    return session_pa


def _detect_session_flip_ns(
    windows: List[dict],
    config: PipelineConfig,
    session_pole_pa: float,
) -> Tuple[bool, float, float]:
    """Detect de-rotation warp direction from atmospheric feature drift.

    Returns (derot_flip, ncc_flip_false, ncc_flip_true).
    derot_flip is passed as flip_direction to spherical_derotation_warp.
    NOTE: this does NOT determine satellite-tracker orientation — use
    sat_cfg.flip_ns for that (S-up cameras should set flip_ns=True there).
    Falls back to (False, 0.0, 0.0) when detection is ambiguous.

    Strategy: collect ALL frames from all windows (sorted by time) using the
    preferred filter, then slide pairs separated by window_frames positions.
    Each pair casts one vote; majority decides derot_flip.
    """
    print("  [derot_flip] Detecting de-rotation warp direction via drift test…")

    # Collect all frames across all windows using the preferred filter.
    filt = None
    all_rows: List[dict] = []
    for preferred in _FILT_PREF_EXT:
        for win in windows:
            pf = win.get("per_filter", {})
            if preferred in pf and pf[preferred].get("included"):
                all_rows.extend(pf[preferred]["included"])
        if len(all_rows) >= 2:
            filt = preferred
            break
        all_rows = []

    if len(all_rows) < 2:
        print("  [derot_flip] No suitable frames — defaulting to flip_direction=False")
        return False, 0.0, 0.0

    all_rows.sort(key=lambda r: r["timestamp"])

    # Load all frames as luminance arrays.
    loaded_frames: List[np.ndarray] = []
    loaded_ts: List = []
    for row in all_rows:
        raw = image_io.read_tif(row["path"])
        lum = raw if raw.ndim == 2 else raw.mean(axis=2).astype(np.float32)
        loaded_frames.append(lum)
        loaded_ts.append(row["timestamp"])

    # Find disk center from the middle frame (most representative).
    mid = len(loaded_frames) // 2
    try:
        cx, cy, semi_a, semi_b, _ = find_disk_center(loaded_frames[mid])
    except Exception:
        print("  [derot_flip] Disk detection failed — defaulting to flip_direction=False")
        return False, 0.0, 0.0
    if semi_a < 5:
        print("  [derot_flip] Disk too small — defaulting to flip_direction=False")
        return False, 0.0, 0.0
    polar_eq = float(np.clip(semi_b / max(semi_a, 1.0), 0.85, 1.0))

    # Step size = window_frames - 1: within one window of W frames, the max span
    # is index 0 ↔ index W-1, so pairs are separated by W-1 positions.
    W = config.quality.window_frames - 1
    t_center = loaded_ts[mid]

    votes: List[Tuple[bool, float, float, float]] = []  # (flip, confidence, ncc_f, ncc_t)

    for i in range(len(loaded_frames) - W):
        frames_pair = [loaded_frames[i], loaded_frames[i + W]]
        dt_pair = [
            (loaded_ts[i]     - t_center).total_seconds(),
            (loaded_ts[i + W] - t_center).total_seconds(),
        ]
        dt = (loaded_ts[i + W] - loaded_ts[i]).total_seconds()

        try:
            flip, ncc_f, ncc_t = auto_detect_ns_flip(
                frames=frames_pair,
                dt_sec_list=dt_pair,
                cx=cx, cy=cy,
                disk_radius_px=semi_a,
                period_hours=config.derotation.rotation_period_hours,
                warp_scale=config.derotation.warp_scale,
                pole_pa_deg=session_pole_pa,
                polar_equatorial_ratio=polar_eq,
            )
            confidence = abs(ncc_f - ncc_t)
            votes.append((flip, confidence, ncc_f, ncc_t))
            print(
                f"  [derot_flip] pair vote [{i}→{i+W}]: flip={flip}  confidence={confidence:.5f}"
                f"  [Δt={dt:.0f}s, filter={filt}]"
            )
        except Exception as exc:
            warnings.warn(f"  [derot_flip] pair [{i}→{i+W}] failed: {exc}")

    if not votes:
        print("  [derot_flip] No valid pairs — defaulting to flip_direction=False")
        return False, 0.0, 0.0

    n_true  = sum(1 for v, *_ in votes if v)
    n_false = len(votes) - n_true
    if n_true != n_false:
        derot_flip = n_true > n_false
    else:
        derot_flip = max(votes, key=lambda x: x[1])[0]

    best = max(votes, key=lambda x: x[1])
    ncc_f, ncc_t = best[2], best[3]

    print(
        f"  [derot_flip] → flip_direction={derot_flip}  "
        f"[{n_true}×True / {n_false}×False, {len(votes)} pair(s)]"
    )
    return derot_flip, ncc_f, ncc_t


def _detect_session_pole_axis_flip(
    windows: List[dict],
    config: PipelineConfig,
    session_pole_pa: float,
    derot_flip: bool,
) -> bool:
    """Auto-detect flip_pole_axis for the true 3D reprojection warp, from
    real atmospheric feature drift — same collect-all-frames/majority-vote
    structure as _detect_session_flip_ns(), extended to the reprojection's
    own sign ambiguity. Only called when use_true_reprojection is on.

    derot_flip must already be resolved (via _detect_session_flip_ns) — this
    searches the ORTHOGONAL ambiguity specific to modelling sub-observer
    latitude B explicitly, which doesn't exist in the linear warp at all.
    """
    print("  [flip_pole_axis] Detecting reprojection pole-axis sign via drift test…")

    all_rows: List[dict] = []
    for preferred in _FILT_PREF_EXT:
        for win in windows:
            pf = win.get("per_filter", {})
            if preferred in pf and pf[preferred].get("included"):
                all_rows.extend(pf[preferred]["included"])
        if len(all_rows) >= 2:
            break
        all_rows = []

    if len(all_rows) < 2:
        print("  [flip_pole_axis] No suitable frames — defaulting to False")
        return False
    all_rows.sort(key=lambda r: r["timestamp"])

    loaded_frames: List[np.ndarray] = []
    loaded_ts: List = []
    for row in all_rows:
        raw = image_io.read_tif(row["path"])
        lum = raw if raw.ndim == 2 else raw.mean(axis=2).astype(np.float32)
        loaded_frames.append(lum)
        loaded_ts.append(row["timestamp"])

    mid = len(loaded_frames) // 2
    try:
        cx, cy, semi_a, semi_b, _ = find_disk_center(loaded_frames[mid])
    except Exception:
        print("  [flip_pole_axis] Disk detection failed — defaulting to False")
        return False
    if semi_a < 5:
        print("  [flip_pole_axis] Disk too small — defaulting to False")
        return False

    t_center = loaded_ts[mid]
    sub_obs_lat = query_horizons_sub_observer_lat(
        horizons_id=config.derotation.horizons_id,
        t_utc=t_center,
        observer_code=config.derotation.observer_code,
    )
    sub_obs_lat = sub_obs_lat if sub_obs_lat is not None else 0.0

    W = config.quality.window_frames - 1

    votes: List[Tuple[bool, float]] = []  # (flip, confidence)
    for i in range(len(loaded_frames) - W):
        frames_pair = [loaded_frames[i], loaded_frames[i + W]]
        dt_pair = [
            (loaded_ts[i]     - t_center).total_seconds(),
            (loaded_ts[i + W] - t_center).total_seconds(),
        ]
        try:
            flip, ncc_f, ncc_t = auto_detect_pole_axis_flip(
                frames=frames_pair,
                dt_sec_list=dt_pair,
                cx=cx, cy=cy,
                disk_radius_px=semi_a,
                period_hours=config.derotation.rotation_period_hours,
                sub_observer_lat_deg=sub_obs_lat,
                warp_scale=config.derotation.warp_scale,
                pole_pa_deg=session_pole_pa,
                polar_equatorial_ratio_true=config.derotation.true_polar_equatorial_ratio,
                flip_direction=derot_flip,
            )
            votes.append((flip, abs(ncc_f - ncc_t)))
        except Exception as exc:
            warnings.warn(f"  [flip_pole_axis] pair [{i}→{i+W}] failed: {exc}")

    if not votes:
        print("  [flip_pole_axis] No valid pairs — defaulting to False")
        return False

    n_true  = sum(1 for v, _ in votes if v)
    n_false = len(votes) - n_true
    flip_pole_axis = (n_true > n_false) if n_true != n_false else max(votes, key=lambda x: x[1])[0]
    print(
        f"  [flip_pole_axis] → {flip_pole_axis}  "
        f"[{n_true}×True / {n_false}×False, {len(votes)} pair(s)]"
    )
    return flip_pole_axis


def _measure_derot_confidence(
    windows: List[dict],
    config: "PipelineConfig",
    session_pole_pa: float,
    flip_ns: bool,
    scale_min: float = 0.50,
    scale_max: float = 1.20,
    n_steps: int = 13,
    min_rotation_deg: float = 3.0,
    use_true_reprojection: bool = False,
    flip_pole_axis: bool = False,
) -> dict:
    """Measure de-rotation confidence via high-pass NCC sweep.

    warp_scale is a physical constant (empirically calibrated on best-seeing data,
    default 0.80) and is NOT derived from this sweep.  Instead the sweep answers:
    "given that we apply config.derotation.warp_scale, how much does the belt
    structure actually support the de-rotation?"

    Returns a dict:
        ncc_at_config_scale : NCC at config.derotation.warp_scale — primary
                              confidence metric.  Low (<0.3) means belt structure
                              is too blurry/absent for reliable de-rotation;
                              consider using a shorter window.
        estimated_peak_scale: scale where NCC peaks — diagnostic only, NOT used
                              to set warp_scale.
        best_ncc            : maximum NCC across sweep — diagnostic.
        rotation_deg        : rotation span used for measurement.
        measured            : False if measurement could not be performed.

    Forward prediction uses flip_direction = not flip_ns because de-rotation undoes
    the drift while forward-prediction replicates it.
    """
    config_scale = config.derotation.warp_scale
    fallback = {
        "ncc_at_config_scale":  0.0,
        "estimated_peak_scale": config_scale,
        "best_ncc":             0.0,
        "rotation_deg":         0.0,
        "measured":             False,
    }

    print("  [derot_conf] Measuring de-rotation confidence via NCC sweep…")

    # Select the window with the longest time span.
    best_win: Optional[Tuple[dict, str]] = None
    best_span = 0.0
    for win in windows:
        filt = next(
            (f for f in _FILT_PREF_EXT
             if f in win.get("per_filter", {}) and win["per_filter"][f].get("included")),
            None,
        )
        if filt is None:
            continue
        rows = win["per_filter"][filt]["included"]
        if len(rows) < 2:
            continue
        ts = [r["timestamp"] for r in rows]
        span = (max(ts) - min(ts)).total_seconds()
        if span > best_span:
            best_span = span
            best_win = (win, filt)

    if best_win is None:
        print("  [derot_conf] No suitable window → confidence unmeasured")
        return fallback

    win, filt = best_win
    period_sec = config.derotation.rotation_period_hours * 3600.0
    rotation_deg = best_span / period_sec * 360.0
    if rotation_deg < min_rotation_deg:
        print(
            f"  [derot_conf] Rotation {rotation_deg:.1f}° < {min_rotation_deg}° "
            "→ confidence unmeasured"
        )
        return {**fallback, "rotation_deg": rotation_deg}

    rows = sorted(win["per_filter"][filt]["included"], key=lambda r: r["timestamp"])
    try:
        raw_e = image_io.read_tif(rows[0]["path"])
        raw_l = image_io.read_tif(rows[-1]["path"])
    except Exception as exc:
        warnings.warn(f"  [derot_conf] Frame read failed: {exc}")
        return fallback

    def _lum(raw: np.ndarray) -> np.ndarray:
        img = raw if raw.ndim == 2 else raw.mean(axis=2).astype(np.float32)
        return img.astype(np.float32) / 65535.0 if img.dtype == np.uint16 else img.astype(np.float32)

    lum_e = _lum(raw_e)
    lum_l = _lum(raw_l)

    cx, cy, semi_a, semi_b, _ = find_disk_center(lum_e)
    if semi_a < 5:
        print("  [derot_conf] Disk detection failed → confidence unmeasured")
        return fallback

    polar_eq = float(np.clip(semi_b / max(semi_a, 1.0), 0.85, 1.0))
    dt = (rows[-1]["timestamp"] - rows[0]["timestamp"]).total_seconds()

    sub_observer_lat_deg = 0.0
    if use_true_reprojection:
        _t_mid = rows[0]["timestamp"] + (rows[-1]["timestamp"] - rows[0]["timestamp"]) / 2
        _b = query_horizons_sub_observer_lat(
            horizons_id=config.derotation.horizons_id,
            t_utc=_t_mid,
            observer_code=config.derotation.observer_code,
        )
        sub_observer_lat_deg = _b if _b is not None else 0.0

    # High-pass filter (σ=30 px) removes limb darkening before NCC.
    # Without it, the smooth radial limb-darkening gradient dominates and NCC
    # decreases monotonically with scale, so scale=0 always wins.
    _HP_SIGMA = 30.0

    def _highpass(img: np.ndarray) -> np.ndarray:
        return img - cv2.GaussianBlur(img, (0, 0), _HP_SIGMA)

    h, w = lum_e.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    disk_mask = ((xx - cx) ** 2 + (yy - cy) ** 2) < (0.7 * semi_a) ** 2
    ref_px = _highpass(lum_l)[disk_mask].astype(np.float64)
    if ref_px.std() < 1e-6:
        print("  [derot_conf] Reference frame featureless → confidence unmeasured")
        return fallback

    # Forward prediction replicates the drift (opposite of de-rotation direction).
    forward_flip = not flip_ns

    # Sweep points: uniform grid + config_scale explicitly included.
    sweep_scales = sorted(set(
        [float(s) for s in np.linspace(scale_min, scale_max, n_steps)]
        + [config_scale]
    ))

    best_ncc = -1.0
    estimated_peak_scale = config_scale
    ncc_at_config_scale = 0.0
    ncc_pairs: List[Tuple[float, float]] = []

    for scale in sweep_scales:
        if use_true_reprojection:
            warped = spherical_derotation_warp_3d(
                lum_e, dt, cx, cy, semi_a,
                period_hours=config.derotation.rotation_period_hours,
                sub_observer_lat_deg=sub_observer_lat_deg,
                pole_pa_deg=session_pole_pa,
                polar_equatorial_ratio_true=config.derotation.true_polar_equatorial_ratio,
                scale=scale,
                flip_direction=forward_flip,
                flip_pole_axis=flip_pole_axis,
            )
        else:
            warped = spherical_derotation_warp(
                lum_e, dt, cx, cy, semi_a,
                period_hours=config.derotation.rotation_period_hours,
                scale=scale,
                flip_direction=forward_flip,
                pole_pa_deg=session_pole_pa,
                polar_equatorial_ratio=polar_eq,
            )
        pred_px = _highpass(warped)[disk_mask].astype(np.float64)
        ncc = float(np.corrcoef(ref_px, pred_px)[0, 1]) if pred_px.std() > 1e-6 else 0.0
        ncc_pairs.append((scale, ncc))
        if ncc > best_ncc:
            best_ncc = ncc
            estimated_peak_scale = scale
        if abs(scale - config_scale) < 1e-9:
            ncc_at_config_scale = ncc

    ncc_str = "  ".join(f"{s:.2f}:{n:.4f}" for s, n in ncc_pairs)
    print(
        f"  [derot_conf] NCC sweep ({len(sweep_scales)} pts, Δt={dt:.0f}s, "
        f"{rotation_deg:.1f}°, {filt}):\n    {ncc_str}"
    )
    print(
        f"  [derot_conf] config_scale={config_scale:.2f}  "
        f"NCC@config={ncc_at_config_scale:.4f}  "
        f"peak_scale={estimated_peak_scale:.3f}  best_NCC={best_ncc:.4f}"
    )

    if ncc_at_config_scale < 0.30:
        print(
            f"  [derot_conf] WARNING: NCC={ncc_at_config_scale:.3f} at scale={config_scale:.2f} "
            "is low — belt structure may be too blurry for reliable de-rotation. "
            "Consider using a shorter window in Step 03."
        )

    return {
        "ncc_at_config_scale":  ncc_at_config_scale,
        "estimated_peak_scale": estimated_peak_scale,
        "best_ncc":             best_ncc,
        "rotation_deg":         rotation_deg,
        "measured":             True,
    }


def run(
    config: PipelineConfig,
    results_03: dict,
    progress_callback=None,
    cancel_event=None,
) -> Dict[str, List[Dict]]:
    """Run Step 4 de-rotation stacking.

    Args:
        config:      Pipeline configuration.
        results_03:  Output of step03_quality_assess.run().

    Returns:
        {"windows": [{window_index, center_time, outputs, log}, ...]}
    """
    windows: List[dict] = results_03.get("windows", [])
    if not windows:
        print("  [WARNING] No time windows from Step 3 — de-rotation skipped.")
        return {"windows": []}

    print(f"  Processing {len(windows)} window(s) × {len(config.filters)} filter(s)…")
    print(f"  Period: {config.derotation.rotation_period_hours}h  "
          f"|  sub-pixel alignment: enabled")

    # ── Session-level pole PA ──────────────────────────────────────────────────
    session_pole_pa = _scan_session_pole_pa(results_03.get("scores", {}), config)
    if session_pole_pa is None:
        session_pole_pa = 0.0
        print("  [WARNING] pole_pa scan failed — using 0.0°")

    # ── De-rotation warp direction (derot_flip) ───────────────────────────────
    # Determines flip_direction for spherical_derotation_warp via feature drift test.
    # For N-up AND pure NS-flip cameras this is almost always False (leftward drift).
    # sat_cfg.flip_ns is NOT used here — it controls satellite tracker only.
    derot_flip, _ncc_f, _ncc_t = _detect_session_flip_ns(windows, config, session_pole_pa)

    # ── Reprojection pole-axis sign (flip_pole_axis) ──────────────────────────
    # Same auto-detection principle as derot_flip above — never a manual GUI
    # toggle. Only computed when the true 3D reprojection warp is in use;
    # the linear warp has no such ambiguity.
    session_pole_axis_flip = False
    if config.derotation.use_true_reprojection:
        session_pole_axis_flip = _detect_session_pole_axis_flip(
            windows, config, session_pole_pa, derot_flip,
        )

    # ── Satellite tracker orientation (tracker_flip_ns) ───────────────────────
    sat_cfg = config.satellite
    tracker_flip_ns = resolve_tracker_flip_ns(config, windows, session_pole_pa, derot_flip)

    # ── warp_scale: fixed at config value (empirically calibrated from good-seeing data) ──
    # The physical rotation rate does not change with seeing; the NCC sweep result
    # is unreliable when belt structures are blurry (returns low scale instead of
    # the true ~0.80).  Confidence is measured separately and logged for diagnostics.
    warp_scale = config.derotation.warp_scale

    # ── De-rotation confidence (diagnostic, does not change warp_scale) ────────
    derot_conf = _measure_derot_confidence(
        windows, config, session_pole_pa, derot_flip,
        use_true_reprojection=bool(config.derotation.use_true_reprojection),
        flip_pole_axis=session_pole_axis_flip,
    )

    # ── SatelliteTracker ───────────────────────────────────────────────────────
    tracker = None
    if sat_cfg.enabled:
        tracker = SatelliteTracker(
            jupiter_horizons_id=config.derotation.horizons_id,
            observer_code=config.derotation.observer_code,
            flip_ew=sat_cfg.flip_ew,
            flip_ns=tracker_flip_ns,
        )
        print(f"  [satellite] tracker enabled  "
              f"(tracker_flip_ns={tracker_flip_ns}, flip_ew={sat_cfg.flip_ew})")

    # ── Session-wide median disk radius (for plate_scale stability) ────────────
    session_r_ref: Optional[float] = (
        compute_session_disk_radius(windows) if tracker is not None else None
    )

    # ── plate_scale auto-calibration from shadow transit (if present) ──────────
    calib_result: Optional[dict] = None
    if tracker is not None and session_r_ref is not None:
        calib_result = run_plate_scale_calibration(
            tracker, config, windows, results_03, session_r_ref, session_pole_pa
        )

    # ── Output directory ───────────────────────────────────────────────────────
    out_base: Optional[Path] = None
    if config.save_step04:
        out_base = config.step_dir(4, "derotated")
        out_base.mkdir(parents=True, exist_ok=True)
        print(f"  Output → {out_base}")
    else:
        print("  save_step04=False: results not written to disk")

    # ── Process each window ────────────────────────────────────────────────────
    all_results: List[dict] = []
    _conf_ncc  = derot_conf["ncc_at_config_scale"]
    _conf_peak = derot_conf["estimated_peak_scale"]
    _conf_ok   = derot_conf["measured"]

    summary_lines: List[str] = [
        "=== Step 4 De-rotation Summary ===",
        "",
        f"  pole_pa          : {session_pole_pa:.2f}°",
        f"  warp_scale       : {warp_scale:.4f}  (config fixed)",
        (
            f"  derot_confidence : {_conf_ncc:.4f}  "
            f"(NCC@scale={warp_scale:.2f};  est. peak={_conf_peak:.3f})"
            if _conf_ok else
            f"  derot_confidence : unmeasured"
        ),
        f"  derot_flip       : {derot_flip}",
        f"  tracker_flip_ns  : {tracker_flip_ns}",
        f"  ncc_flip_false   : {_ncc_f:.4f}",
        f"  ncc_flip_true    : {_ncc_t:.4f}",
        "",
    ]

    session_log = {
        "pole_pa_deg":             session_pole_pa,
        "warp_scale":              warp_scale,
        "derot_ncc_confidence":    _conf_ncc,
        "derot_estimated_scale":   _conf_peak,
        "derot_confidence_valid":  _conf_ok,
        "derot_flip":              derot_flip,
        "tracker_flip_ns":         tracker_flip_ns,
        "ncc_flip_false":          _ncc_f,
        "ncc_flip_true":           _ncc_t,
    }
    if calib_result is not None:
        session_log["plate_scale_calibration"] = {
            "ps_fit":     calib_result["ps_fit"],
            "ps_nom":     calib_result["ps_nom"],
            "dps_pct":    calib_result["dps_pct"],
            "cx_offset":  calib_result["cx_offset"],
            "n_frames":   calib_result["n"],
            "rmse_nom":   calib_result["rmse_nom"],
            "rmse_fit":   calib_result["rmse_fit"],
        }

    n_windows = len(windows)
    for win_idx, window in enumerate(windows, start=1):
        if cancel_event is not None and cancel_event.is_set():
            print("  [CANCELLED] Stopping Step 4.", flush=True)
            break
        if progress_callback is not None:
            progress_callback(win_idx - 1, n_windows)

        t_center = window["center_time"]
        t_center_str = t_center.strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"\n  Window {win_idx}  [{t_center_str}]  "
              f"quality={window['window_quality']:.4f}  "
              f"rotation={window['rotation_degrees']:.1f}°")

        # ── NP.ang from Horizons (celestial North Pole angle) ─────────────────
        np_ang = query_horizons_np_ang(
            horizons_id=config.derotation.horizons_id,
            t_utc=t_center,
            observer_code=config.derotation.observer_code,
        )
        np_ang_val = np_ang if np_ang is not None else 0.0
        if np_ang is None:
            print("    [WARNING] NP.ang not available → using 0.0°")
        else:
            print(f"  [NP.ang = {np_ang_val:.3f}° (celestial)]")

        # pole_pa for the WARP: image-space angle from auto_detect_equator_pa()
        # pole_pa for the TRACKER: pole_pa + NP.ang = camera rotation θ_cam
        pole_pa_for_warp = session_pole_pa
        print(f"  [pole_pa = {pole_pa_for_warp:.1f}° (image-space, for warp)]")

        # ── Sub-observer latitude B from Horizons (true 3D reprojection only) ──
        # Skipped unless opted in — avoids the extra lookup/network cost for
        # every session that still uses the default linear warp.
        use_true_reprojection = bool(config.derotation.use_true_reprojection)
        sub_observer_lat_val = 0.0
        if use_true_reprojection:
            sub_obs_lat = query_horizons_sub_observer_lat(
                horizons_id=config.derotation.horizons_id,
                t_utc=t_center,
                observer_code=config.derotation.observer_code,
            )
            sub_observer_lat_val = sub_obs_lat if sub_obs_lat is not None else 0.0
            if sub_obs_lat is None:
                print("    [WARNING] ObsSub-LAT not available → using 0.0°")
            else:
                print(f"  [ObsSub-LAT = {sub_observer_lat_val:.3f}° (sub-observer latitude B)]")

        # Create per-window output directory
        win_out_dir: Optional[Path] = None
        if out_base is not None:
            win_out_dir = out_base / f"window_{win_idx:02d}"
            win_out_dir.mkdir(parents=True, exist_ok=True)

        # ── Satellite position prediction ──────────────────────────────────────
        sat_log: Dict = {}
        if tracker is not None:
            sat_log = {
                "np_ang_deg":       np_ang_val,
                "pole_pa_deg":      pole_pa_for_warp,
                "derot_flip":       derot_flip,
                "tracker_flip_ns":  tracker_flip_ns,
            }

        # ── De-rotate all filters ──────────────────────────────────────────────
        filter_results = derotation.derotate_window(
            window=window,
            required_filters=(
                list(window["per_filter"].keys())
                if config.camera_mode == "color"
                else config.filters
            ),
            period_hours=config.derotation.rotation_period_hours,
            warp_scale=warp_scale,
            align=True,
            normalize_brightness=config.derotation.normalize_brightness,
            min_quality_threshold=config.derotation.min_quality_threshold,
            pole_pa_deg=pole_pa_for_warp,
            color_mode=(config.camera_mode == "color"),
            flip_direction=derot_flip,
            out_dir=win_out_dir,
            weight_power=config.derotation.stack_weight_power,
            use_true_reprojection=use_true_reprojection,
            sub_observer_lat_deg=sub_observer_lat_val,
            true_polar_equatorial_ratio=config.derotation.true_polar_equatorial_ratio,
            flip_pole_axis=session_pole_axis_flip,
        )

        # ── Satellite compositing (exp9 method) ───────────────────────────────
        if sat_cfg.composite_enabled and tracker is not None:
            print(f"  [satellite composite] Window {win_idx}…")
            disk_centers = _apply_satellite_composite(
                window=window,
                filter_results=filter_results,
                config=config,
                tracker=tracker,
                pole_pa_deg=pole_pa_for_warp,
                np_ang_deg=np_ang_val,
                r_ref=session_r_ref,
                use_true_reprojection=use_true_reprojection,
                sub_observer_lat_deg=sub_observer_lat_val,
                flip_pole_axis=session_pole_axis_flip,
            )
            if disk_centers and sat_log:
                sat_log["disk_centers"] = disk_centers

        # ── Build log and save JSON ────────────────────────────────────────────
        log_dict = derotation.derotation_log_to_json(win_idx, window, filter_results)
        log_dict["session"] = session_log
        if sat_log:
            log_dict["satellite"] = sat_log
        if win_out_dir is not None:
            json_path = win_out_dir / "derotation_log.json"
            with open(json_path, "w") as f:
                json.dump(log_dict, f, indent=2, default=str)
            print(f"    → {json_path.name}")

        # ── Summary ───────────────────────────────────────────────────────────
        summary_lines.append(
            f"Window {win_idx}  {t_center_str}  "
            f"quality={window['window_quality']:.4f}  "
            f"rotation_span={window['rotation_degrees']:.1f}°"
        )
        summary_filters = (
            list(filter_results.keys()) if config.camera_mode == "color"
            else config.filters
        )
        for filt in summary_filters:
            if filt in filter_results:
                out_path, flog = filter_results[filt]
                n = flog.get("n_stacked", 0)
                snr = round(float(n) ** 0.5, 2)
                fname = out_path.name if out_path else "—"
                summary_lines.append(
                    f"  {filt:>4}: {fname}  ({n} frames, SNR×{snr:.2f})"
                )
            else:
                summary_lines.append(f"  {filt:>4}: not available")
        summary_lines.append("")

        outputs = {filt: res[0] for filt, res in filter_results.items()}
        logs    = {filt: res[1] for filt, res in filter_results.items()}
        all_results.append({
            "window_index": win_idx,
            "center_time":  t_center_str,
            "outputs":      outputs,
            "log":          logs,
            "satellite":    sat_log,
        })

    if progress_callback is not None:
        progress_callback(n_windows, n_windows)

    # ── Save summary ───────────────────────────────────────────────────────────
    summary_text = "\n".join(summary_lines)
    print()
    print(summary_text)
    if out_base is not None:
        txt_path = out_base / "derotation_summary.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(summary_text)
        print(f"  → {txt_path}")

    return {
        "windows":                  all_results,
        "derot_ncc_confidence":     _conf_ncc,
        "derot_confidence_measured": _conf_ok,
        "session_pole_pa":          session_pole_pa,
    }
