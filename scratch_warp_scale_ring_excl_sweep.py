"""
Re-run of the _measure_derot_confidence() NCC-vs-warp_scale sweep, across
EVERY real Saturn_Data window x filter combo, under two disk_mask conditions:

  (a) ORIGINAL   : plain circle, radius 0.7*semi_a, centered at (cx,cy)
                   (reproduces the historical sweep exactly - the mask as
                   written at derotate_stack.py:456, BEFORE this session's
                   ring-crossing-disk fix).
  (b) CORRECTED  : same circle, AND-ed with ~compute_ring_crossing_mask(...)
                   (real function from pipeline.modules.derotation, called
                   with this window's real cx/cy/semi_a/semi_b from
                   find_disk_center() on the earliest frame, and real
                   pole_pa_deg / sub_observer_lat_deg from that window's
                   production derotation_log.json).

Everything else (high-pass sigma=30, forward_flip = not flip_ns, corrcoef-
based NCC, the spherical_derotation_warp() call itself) is IDENTICAL to
pipeline/steps/derotate_stack.py::_measure_derot_confidence -- imported and
called for real, never reimplemented.

flip_ns: _measure_derot_confidence is called in run() as
    _measure_derot_confidence(windows, config, session_pole_pa, derot_flip, ...)
i.e. its 4th positional arg ("flip_ns" in the signature) is actually
derot_flip -- the SAME value stored per-filter in every production
derotation_log.json as "flip_direction" (confirmed constant, False, across
every window/filter checked). So flip_ns := that logged flip_direction, and
forward_flip = not flip_ns, exactly reproducing the production call.
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

OUT_SCRIPT_RESULTS = Path("/data/planetflow/scratch_warp_scale_ring_excl_sweep_results.json")

_HP_SIGMA = 30.0
MIN_ROTATION_DEG = 3.0
SCALE_MIN = 0.05
SCALE_MAX = 1.5
N_STEPS = 25  # >= 20 requested


def _parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _lum(raw: np.ndarray) -> np.ndarray:
    img = raw if raw.ndim == 2 else raw.mean(axis=2).astype(np.float32)
    return img.astype(np.float32) / 65535.0 if img.dtype == np.uint16 else img.astype(np.float32)


def _highpass(img: np.ndarray) -> np.ndarray:
    return img - cv2.GaussianBlur(img, (0, 0), _HP_SIGMA)


def _frame_path(stem: str) -> Path:
    return STEP02_DIR / f"{stem}.tif"


def run_sweep_for_window_filter(win_idx: int, filt: str, rows: list, log_entry: dict) -> dict:
    """rows: sorted-by-timestamp list of {'stem','timestamp':datetime,...} for this window/filter.
    log_entry: production derotation_log.json['filters'][filt] dict for this window (for
    period_hours, pole_pa_deg, sub_observer_lat_deg, flip_direction -- all real, logged values).
    """
    result = {
        "window_index": win_idx,
        "filter": filt,
        "n_frames": len(rows),
        "measured": False,
    }

    if len(rows) < 2:
        result["skip_reason"] = f"only {len(rows)} frame(s), need >=2"
        return result

    period_hours = float(log_entry["period_hours"])
    pole_pa_deg = float(log_entry["pole_pa_deg"])
    sub_observer_lat_deg = float(log_entry["sub_observer_lat_deg"])
    flip_ns = bool(log_entry["flip_direction"])  # see module docstring for why

    period_sec = period_hours * 3600.0
    ts_list = [r["timestamp"] for r in rows]
    span_sec = (max(ts_list) - min(ts_list)).total_seconds()
    rotation_deg = span_sec / period_sec * 360.0
    result["rotation_deg"] = rotation_deg
    result["period_hours"] = period_hours
    result["pole_pa_deg"] = pole_pa_deg
    result["sub_observer_lat_deg"] = sub_observer_lat_deg
    result["flip_ns"] = flip_ns

    if rotation_deg < MIN_ROTATION_DEG:
        result["skip_reason"] = f"rotation_deg {rotation_deg:.2f} < {MIN_ROTATION_DEG}"
        return result

    rows_sorted = sorted(rows, key=lambda r: r["timestamp"])
    p_e = _frame_path(rows_sorted[0]["stem"])
    p_l = _frame_path(rows_sorted[-1]["stem"])
    if not p_e.exists() or not p_l.exists():
        result["skip_reason"] = f"missing raw frame(s): {p_e.name if not p_e.exists() else ''} {p_l.name if not p_l.exists() else ''}".strip()
        return result

    try:
        raw_e = image_io.read_tif(p_e)
        raw_l = image_io.read_tif(p_l)
    except Exception as exc:
        result["skip_reason"] = f"read_tif failed: {exc}"
        return result

    lum_e = _lum(raw_e)
    lum_l = _lum(raw_l)

    cx, cy, semi_a, semi_b, _ = find_disk_center(lum_e)
    if semi_a < 5:
        result["skip_reason"] = "disk detection failed (semi_a < 5)"
        return result

    polar_eq = float(np.clip(semi_b / max(semi_a, 1.0), 0.85, 1.0))
    dt = (rows_sorted[-1]["timestamp"] - rows_sorted[0]["timestamp"]).total_seconds()

    h, w = lum_e.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    disk_mask_orig = ((xx - cx) ** 2 + (yy - cy) ** 2) < (0.7 * semi_a) ** 2

    ring_mask = compute_ring_crossing_mask(
        h, w, cx, cy, semi_a, semi_b,
        pole_pa_deg=pole_pa_deg,
        sub_observer_lat_deg=sub_observer_lat_deg,
    )
    disk_mask_corrected = disk_mask_orig & ~ring_mask

    n_orig = int(disk_mask_orig.sum())
    n_ring_excluded = int((disk_mask_orig & ring_mask).sum())
    n_corrected = int(disk_mask_corrected.sum())
    result["cx"], result["cy"], result["semi_a"], result["semi_b"] = cx, cy, semi_a, semi_b
    result["mask_pixels_original"] = n_orig
    result["mask_pixels_ring_excluded"] = n_ring_excluded
    result["mask_pixels_corrected"] = n_corrected
    result["ring_exclusion_fraction"] = n_ring_excluded / n_orig if n_orig > 0 else 0.0

    if n_corrected < 50:
        result["skip_reason"] = f"corrected mask too small ({n_corrected} px) after ring exclusion"
        return result

    forward_flip = not flip_ns
    sweep_scales = [float(s) for s in np.linspace(SCALE_MIN, SCALE_MAX, N_STEPS)]

    hp_l = _highpass(lum_l)
    ref_px_orig = hp_l[disk_mask_orig].astype(np.float64)
    ref_px_corr = hp_l[disk_mask_corrected].astype(np.float64)

    if ref_px_orig.std() < 1e-6 or ref_px_corr.std() < 1e-6:
        result["skip_reason"] = "reference frame featureless under one of the masks"
        return result

    curve_a, curve_b = [], []
    best_a = (-1.0, None)
    best_b = (-1.0, None)

    for scale in sweep_scales:
        warped = spherical_derotation_warp(
            lum_e, dt, cx, cy, semi_a,
            period_hours=period_hours,
            scale=scale,
            flip_direction=forward_flip,
            pole_pa_deg=pole_pa_deg,
            polar_equatorial_ratio=polar_eq,
        )
        hp_w = _highpass(warped)

        pred_a = hp_w[disk_mask_orig].astype(np.float64)
        ncc_a = float(np.corrcoef(ref_px_orig, pred_a)[0, 1]) if pred_a.std() > 1e-6 else 0.0
        curve_a.append((scale, ncc_a))
        if ncc_a > best_a[0]:
            best_a = (ncc_a, scale)

        pred_b = hp_w[disk_mask_corrected].astype(np.float64)
        ncc_b = float(np.corrcoef(ref_px_corr, pred_b)[0, 1]) if pred_b.std() > 1e-6 else 0.0
        curve_b.append((scale, ncc_b))
        if ncc_b > best_b[0]:
            best_b = (ncc_b, scale)

    result["measured"] = True
    result["dt_sec"] = dt
    result["peak_scale_original"] = best_a[1]
    result["best_ncc_original"] = best_a[0]
    result["peak_scale_corrected"] = best_b[1]
    result["best_ncc_corrected"] = best_b[0]
    result["moved_toward_1"] = abs(best_b[1] - 1.0) < abs(best_a[1] - 1.0)
    result["shift"] = best_b[1] - best_a[1]
    result["curve_original"] = curve_a
    result["curve_corrected"] = curve_b

    return result


def main():
    windows = json.load(open(WINDOWS_JSON))["selected_windows"]

    all_results = []
    for win in windows:
        win_idx = win["window_index"]
        log_path = STEP04_DIR / f"window_{win_idx:02d}" / "derotation_log.json"
        if not log_path.exists():
            print(f"[window {win_idx}] no derotation_log.json -> skip entire window")
            continue
        log = json.load(open(log_path))
        log_filters = log.get("filters", {})

        for filt, filt_data in win.get("per_filter", {}).items():
            if filt not in log_filters:
                print(f"[window {win_idx}][{filt}] not in production log -> skip")
                continue
            rows_raw = filt_data.get("included", [])
            rows = [
                {"stem": r["stem"], "timestamp": _parse_ts(r["timestamp"])}
                for r in rows_raw
            ]
            print(f"[window {win_idx}][{filt}] {len(rows)} frame(s) ...", flush=True)
            res = run_sweep_for_window_filter(win_idx, filt, rows, log_filters[filt])
            status = "OK" if res.get("measured") else f"SKIP({res.get('skip_reason')})"
            if res.get("measured"):
                print(
                    f"    -> peak_orig={res['peak_scale_original']:.3f} "
                    f"peak_corr={res['peak_scale_corrected']:.3f} "
                    f"ring_excl_frac={res['ring_exclusion_fraction']:.3f} "
                    f"moved_toward_1={res['moved_toward_1']}  [{status}]"
                )
            else:
                print(f"    -> {status}")
            all_results.append(res)

    with open(OUT_SCRIPT_RESULTS, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote {len(all_results)} result entries -> {OUT_SCRIPT_RESULTS}")


if __name__ == "__main__":
    main()
