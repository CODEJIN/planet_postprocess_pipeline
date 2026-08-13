"""
Unbiased re-measurement of the Saturn de-rotation NCC-vs-warp_scale curve,
with NO self-reference to config.derotation.warp_scale anywhere in the sweep.

Context: pipeline/steps/derotate_stack.py::_measure_derot_confidence sweeps
scale in [0.50, 1.20] (13 pts) and then explicitly INJECTS config_scale
(0.10) as an extra candidate point (see that function, ~line 466-468:
`sweep_scales = sorted(set([...linspace...] + [config_scale]))`). A prior
scratch check in this session (scratch_warp_scale_mask_sweep_results.json /
scratch_warp_scale_ring_excl_sweep_results.json) found that ~70% of 44
window x filter combos have NCC decreasing MONOTONICALLY across the whole
tested range -- meaning the injected config_scale point trivially "wins" as
estimated_peak_scale regardless of whether it bears any relation to
Saturn's true rotation. This script checks whether that's the whole story:

  - PURE uniform grid, scale in [-0.5, 2.0] step 0.05 (51 points).
    No config_scale injected anywhere.
  - Negative scale included as a symmetry sanity check: if NCC falls off
    symmetrically in both directions away from scale=0, that points to
    warp-interpolation/registration noise (any warp away from identity
    decorrelates, regardless of direction) rather than a real rotation
    signal, which should be asymmetric (favor the correct sign) and peak
    at some real positive interior scale.
  - Every one of the 44 real window x filter combos with rotation_deg>=3.0
    (min_rotation_deg, same threshold _measure_derot_confidence uses),
    using the LONGEST-baseline (first-vs-last) frame pair per combo, exactly
    as _measure_derot_confidence does.
  - The subset of those with the single LARGEST available rotation_deg in
    this dataset is additionally broken out and reported separately.
  - For 4 selected window x filter combos (5-frame windows, so 4 consecutive
    pairs available), ALL consecutive-frame pairs are also swept, in
    addition to the first-vs-last pair, to see whether the monotonic-decrease
    pattern holds across very different dt / rotation magnitudes within the
    same window, or is specific to the longest-baseline pair.

Everything except grid construction and bookkeeping is the real production
code, imported and called directly -- never reimplemented:
  - pipeline.modules.image_io.read_tif for raw frames
  - pipeline.modules.derotation.find_disk_center for disk geometry
  - pipeline.modules.derotation.compute_ring_crossing_mask for ring exclusion
    (carried over from the just-completed ring check in this session --
    already confirmed not to change the conclusion, kept for consistency)
  - pipeline.modules.derotation.spherical_derotation_warp for the warp itself
  - high-pass sigma=30 (removes limb darkening) + np.corrcoef-based NCC,
    copied verbatim from _measure_derot_confidence's inline logic (that
    logic is ~6 lines of numpy/cv2 glue around the real warp/mask calls,
    not something with its own importable function -- but the warp, mask,
    and disk-detection math itself is never hand-reimplemented).

Real data only: Saturn_Data/step02_lucky_stack raw TIFs, real timestamps
from Saturn_Data/step03_quality/windows.json, real per-window/per-filter
period_hours/pole_pa_deg/sub_observer_lat_deg/flip_direction from
Saturn_Data/step04_derotated/window_0N/derotation_log.json. Nothing
hardcoded or guessed.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from pipeline.modules import image_io
from pipeline.modules.derotation import (
    find_disk_center,
    spherical_derotation_warp,
    compute_ring_crossing_mask,
)

ROOT = Path("/data/planetflow/Saturn_Data")
WINDOWS_JSON = ROOT / "step03_quality" / "windows.json"
STEP02_DIR = ROOT / "step02_lucky_stack"
STEP04_DIR = ROOT / "step04_derotated"

OUT_RESULTS = Path("/data/planetflow/scratch_warp_scale_unbiased_sweep_results.json")

_HP_SIGMA = 30.0
MIN_ROTATION_DEG = 3.0

# Pure uniform grid, NO config_scale injected.
SCALE_MIN, SCALE_MAX, SCALE_STEP = -0.5, 2.0, 0.05
SWEEP_SCALES = [round(float(s), 10) for s in np.arange(SCALE_MIN, SCALE_MAX + 1e-9, SCALE_STEP)]

# Symmetry-check magnitudes: only values whose +X and -X both lie inside
# [SCALE_MIN, SCALE_MAX] can be compared (grid is asymmetric: -0.5..2.0).
SYMMETRY_X_VALUES = [round(0.05 * k, 2) for k in range(1, 11)]  # 0.05 .. 0.50

# Window x filter combos to additionally probe with multiple frame pairs
# (consecutive pairs, not just first-vs-last). Chosen as 4 distinct
# 5-frame, max-rotation combos spanning different filters.
MULTI_PAIR_COMBOS = [
    (3, "IR"),
    (4, "R"),
    (6, "G"),
    (7, "CH4"),
]


def _parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _lum(raw: np.ndarray) -> np.ndarray:
    img = raw if raw.ndim == 2 else raw.mean(axis=2).astype(np.float32)
    return img.astype(np.float32) / 65535.0 if img.dtype == np.uint16 else img.astype(np.float32)


def _highpass(img: np.ndarray) -> np.ndarray:
    return img - cv2.GaussianBlur(img, (0, 0), _HP_SIGMA)


def _frame_path(stem: str) -> Path:
    return STEP02_DIR / f"{stem}.tif"


def _classify_curve(curve: list) -> dict:
    """curve: list of (scale, ncc) sorted by scale ascending."""
    scales = np.array([s for s, _ in curve])
    nccs = np.array([n for _, n in curve])
    argmax_i = int(np.argmax(nccs))
    peak_scale = float(scales[argmax_i])
    peak_ncc = float(nccs[argmax_i])

    is_left_boundary = argmax_i == 0
    is_right_boundary = argmax_i == len(scales) - 1
    diffs = np.diff(nccs)
    monotonic_decreasing = bool(np.all(diffs <= 1e-9))
    monotonic_increasing = bool(np.all(diffs >= -1e-9))

    if is_left_boundary and monotonic_decreasing:
        shape = "monotonic_decreasing_full_range"
    elif is_right_boundary and monotonic_increasing:
        shape = "monotonic_increasing_full_range"
    elif not is_left_boundary and not is_right_boundary:
        shape = "interior_peak"
    else:
        shape = "boundary_peak_nonmonotonic"

    return {
        "peak_scale": peak_scale,
        "peak_ncc": peak_ncc,
        "argmax_at_left_boundary": is_left_boundary,
        "argmax_at_right_boundary": is_right_boundary,
        "strictly_monotonic_decreasing": monotonic_decreasing,
        "strictly_monotonic_increasing": monotonic_increasing,
        "shape": shape,
    }


def _symmetry_check(curve: list) -> dict:
    lookup = {round(s, 2): n for s, n in curve}
    out = {}
    for x in SYMMETRY_X_VALUES:
        n_pos = lookup.get(round(x, 2))
        n_neg = lookup.get(round(-x, 2))
        if n_pos is None or n_neg is None:
            continue
        out[f"{x:.2f}"] = {
            "ncc_pos": n_pos,
            "ncc_neg": n_neg,
            "diff_pos_minus_neg": n_pos - n_neg,
        }
    return out


def _sweep_pair(lum_e, lum_l, dt_sec, period_hours, pole_pa_deg,
                 sub_observer_lat_deg, flip_ns) -> dict:
    """Run the full pure-grid NCC sweep for one (early, late) frame pair.
    Mirrors _measure_derot_confidence's inline math exactly, just without
    the config_scale injection and over the wider [-0.5, 2.0] grid.
    """
    out = {"measured": False}

    cx, cy, semi_a, semi_b, _ = find_disk_center(lum_e)
    if semi_a < 5:
        out["skip_reason"] = "disk detection failed (semi_a < 5)"
        return out

    polar_eq = float(np.clip(semi_b / max(semi_a, 1.0), 0.85, 1.0))

    h, w = lum_e.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    disk_mask = ((xx - cx) ** 2 + (yy - cy) ** 2) < (0.7 * semi_a) ** 2

    ring_mask = compute_ring_crossing_mask(
        h, w, cx, cy, semi_a, semi_b,
        pole_pa_deg=pole_pa_deg,
        sub_observer_lat_deg=sub_observer_lat_deg,
    )
    disk_mask = disk_mask & ~ring_mask
    n_mask_px = int(disk_mask.sum())
    out["cx"], out["cy"], out["semi_a"], out["semi_b"] = cx, cy, semi_a, semi_b
    out["mask_pixels"] = n_mask_px
    if n_mask_px < 50:
        out["skip_reason"] = f"mask too small after ring exclusion ({n_mask_px} px)"
        return out

    ref_px = _highpass(lum_l)[disk_mask].astype(np.float64)
    if ref_px.std() < 1e-6:
        out["skip_reason"] = "reference frame featureless under mask"
        return out

    # Forward prediction replicates the drift -> opposite sense from de-rotation,
    # exactly as _measure_derot_confidence does.
    forward_flip = not flip_ns

    curve = []
    for scale in SWEEP_SCALES:
        warped = spherical_derotation_warp(
            lum_e, dt_sec, cx, cy, semi_a,
            period_hours=period_hours,
            scale=scale,
            flip_direction=forward_flip,
            pole_pa_deg=pole_pa_deg,
            polar_equatorial_ratio=polar_eq,
        )
        pred_px = _highpass(warped)[disk_mask].astype(np.float64)
        ncc = float(np.corrcoef(ref_px, pred_px)[0, 1]) if pred_px.std() > 1e-6 else 0.0
        curve.append((scale, ncc))

    out["measured"] = True
    out["curve"] = curve
    out.update(_classify_curve(curve))
    out["symmetry"] = _symmetry_check(curve)
    return out


def run_combo(win_idx: int, filt: str, rows: list, log_entry: dict) -> dict:
    """rows: sorted-by-timestamp [{'stem','timestamp':datetime}], first-vs-last pair only."""
    result = {"window_index": win_idx, "filter": filt, "n_frames": len(rows), "measured": False}

    if len(rows) < 2:
        result["skip_reason"] = f"only {len(rows)} frame(s), need >=2"
        return result

    period_hours = float(log_entry["period_hours"])
    pole_pa_deg = float(log_entry["pole_pa_deg"])
    sub_observer_lat_deg = float(log_entry["sub_observer_lat_deg"])
    flip_ns = bool(log_entry["flip_direction"])

    period_sec = period_hours * 3600.0
    ts_list = [r["timestamp"] for r in rows]
    span_sec = (max(ts_list) - min(ts_list)).total_seconds()
    rotation_deg = span_sec / period_sec * 360.0
    result.update(period_hours=period_hours, pole_pa_deg=pole_pa_deg,
                  sub_observer_lat_deg=sub_observer_lat_deg, flip_ns=flip_ns,
                  rotation_deg=rotation_deg)

    if rotation_deg < MIN_ROTATION_DEG:
        result["skip_reason"] = f"rotation_deg {rotation_deg:.2f} < {MIN_ROTATION_DEG}"
        return result

    rows_sorted = sorted(rows, key=lambda r: r["timestamp"])
    p_e, p_l = _frame_path(rows_sorted[0]["stem"]), _frame_path(rows_sorted[-1]["stem"])
    if not p_e.exists() or not p_l.exists():
        result["skip_reason"] = "missing raw frame(s)"
        return result

    try:
        lum_e = _lum(image_io.read_tif(p_e))
        lum_l = _lum(image_io.read_tif(p_l))
    except Exception as exc:
        result["skip_reason"] = f"read_tif failed: {exc}"
        return result

    dt = (rows_sorted[-1]["timestamp"] - rows_sorted[0]["timestamp"]).total_seconds()
    result["dt_sec"] = dt
    result["pair"] = f"{rows_sorted[0]['stem']} -> {rows_sorted[-1]['stem']}"

    sweep = _sweep_pair(lum_e, lum_l, dt, period_hours, pole_pa_deg, sub_observer_lat_deg, flip_ns)
    result.update(sweep)
    return result


def run_multi_pair(win_idx: int, filt: str, rows: list, log_entry: dict) -> list:
    """All consecutive pairs (i, i+1) plus first-vs-last, for one window/filter."""
    period_hours = float(log_entry["period_hours"])
    pole_pa_deg = float(log_entry["pole_pa_deg"])
    sub_observer_lat_deg = float(log_entry["sub_observer_lat_deg"])
    flip_ns = bool(log_entry["flip_direction"])
    period_sec = period_hours * 3600.0

    rows_sorted = sorted(rows, key=lambda r: r["timestamp"])
    pair_specs = [(i, i + 1, "consecutive") for i in range(len(rows_sorted) - 1)]
    pair_specs.append((0, len(rows_sorted) - 1, "first_vs_last"))

    out = []
    for i, j, label in pair_specs:
        r = {"window_index": win_idx, "filter": filt, "pair_label": label,
             "pair_idx": [i, j], "measured": False}
        p_e, p_l = _frame_path(rows_sorted[i]["stem"]), _frame_path(rows_sorted[j]["stem"])
        if not p_e.exists() or not p_l.exists():
            r["skip_reason"] = "missing raw frame(s)"
            out.append(r)
            continue
        try:
            lum_e = _lum(image_io.read_tif(p_e))
            lum_l = _lum(image_io.read_tif(p_l))
        except Exception as exc:
            r["skip_reason"] = f"read_tif failed: {exc}"
            out.append(r)
            continue
        dt = (rows_sorted[j]["timestamp"] - rows_sorted[i]["timestamp"]).total_seconds()
        rotation_deg = dt / period_sec * 360.0
        r["dt_sec"] = dt
        r["rotation_deg"] = rotation_deg
        r["pair"] = f"{rows_sorted[i]['stem']} -> {rows_sorted[j]['stem']}"
        if rotation_deg < MIN_ROTATION_DEG:
            r["skip_reason"] = f"rotation_deg {rotation_deg:.2f} < {MIN_ROTATION_DEG}"
            out.append(r)
            continue
        sweep = _sweep_pair(lum_e, lum_l, dt, period_hours, pole_pa_deg, sub_observer_lat_deg, flip_ns)
        r.update(sweep)
        out.append(r)
    return out


def main():
    windows = json.load(open(WINDOWS_JSON))["selected_windows"]

    main_results = []
    for win in windows:
        win_idx = win["window_index"]
        log_path = STEP04_DIR / f"window_{win_idx:02d}" / "derotation_log.json"
        if not log_path.exists():
            print(f"[window {win_idx}] no derotation_log.json -> skip entire window")
            continue
        log_filters = json.load(open(log_path)).get("filters", {})

        for filt, filt_data in win.get("per_filter", {}).items():
            if filt not in log_filters:
                continue
            rows = [{"stem": r["stem"], "timestamp": _parse_ts(r["timestamp"])}
                    for r in filt_data.get("included", [])]
            print(f"[window {win_idx}][{filt}] {len(rows)} frame(s), sweeping "
                  f"{len(SWEEP_SCALES)} pts over [{SCALE_MIN},{SCALE_MAX}]...", flush=True)
            res = run_combo(win_idx, filt, rows, log_filters[filt])
            if res.get("measured"):
                print(f"    -> shape={res['shape']}  peak_scale={res['peak_scale']:.3f}  "
                      f"peak_ncc={res['peak_ncc']:.4f}  rot={res['rotation_deg']:.2f}deg")
            else:
                print(f"    -> SKIP({res.get('skip_reason')})")
            main_results.append(res)

    measured = [r for r in main_results if r.get("measured")]
    print(f"\nMain sweep: {len(measured)}/{len(main_results)} combos measured "
          f"(rotation_deg >= {MIN_ROTATION_DEG})")

    shape_counts = {}
    for r in measured:
        shape_counts[r["shape"]] = shape_counts.get(r["shape"], 0) + 1
    print("Shape distribution:", shape_counts)

    max_rot = max((r["rotation_deg"] for r in measured), default=None)
    largest_rot_subset = [r for r in measured if max_rot is not None and
                           abs(r["rotation_deg"] - max_rot) < 1e-6]
    print(f"\nLargest-rotation subset: rotation_deg={max_rot:.4f}deg, "
          f"n={len(largest_rot_subset)} combos")
    lr_shape_counts = {}
    for r in largest_rot_subset:
        lr_shape_counts[r["shape"]] = lr_shape_counts.get(r["shape"], 0) + 1
    print("  shape distribution (largest-rotation subset):", lr_shape_counts)

    # Multi-pair probe for selected combos.
    multi_pair_results = {}
    for win_idx, filt in MULTI_PAIR_COMBOS:
        win = next((w for w in windows if w["window_index"] == win_idx), None)
        if win is None:
            continue
        log_path = STEP04_DIR / f"window_{win_idx:02d}" / "derotation_log.json"
        log_filters = json.load(open(log_path)).get("filters", {})
        if filt not in log_filters or filt not in win.get("per_filter", {}):
            continue
        rows = [{"stem": r["stem"], "timestamp": _parse_ts(r["timestamp"])}
                for r in win["per_filter"][filt].get("included", [])]
        print(f"\n[multi-pair] window {win_idx} {filt}: {len(rows)} frames, "
              f"sweeping all consecutive pairs + first-vs-last...", flush=True)
        pair_results = run_multi_pair(win_idx, filt, rows, log_filters[filt])
        for pr in pair_results:
            if pr.get("measured"):
                print(f"    [{pr['pair_label']} {pr['pair_idx']}] rot={pr['rotation_deg']:.2f}deg "
                      f"shape={pr['shape']} peak_scale={pr['peak_scale']:.3f} peak_ncc={pr['peak_ncc']:.4f}")
            else:
                print(f"    [{pr['pair_label']} {pr['pair_idx']}] SKIP({pr.get('skip_reason')})")
        multi_pair_results[f"window_{win_idx:02d}_{filt}"] = pair_results

    out = {
        "sweep_scales": SWEEP_SCALES,
        "min_rotation_deg": MIN_ROTATION_DEG,
        "symmetry_x_values": SYMMETRY_X_VALUES,
        "main_results": main_results,
        "max_rotation_deg": max_rot,
        "largest_rotation_subset_window_filter": [
            {"window_index": r["window_index"], "filter": r["filter"]} for r in largest_rot_subset
        ],
        "shape_distribution_all": shape_counts,
        "shape_distribution_largest_rotation_subset": lr_shape_counts,
        "multi_pair_results": multi_pair_results,
    }
    with open(OUT_RESULTS, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote results -> {OUT_RESULTS}")


if __name__ == "__main__":
    main()
