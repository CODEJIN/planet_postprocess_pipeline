"""
Per-frame post-warp residual misalignment measurement, IR vs R, windows 1-9.

Adapted from scratch_align_hypothesis.py, extended to:
  - all 9 windows, filters IR and R (not just the few spot-checked before)
  - has_rings / ring_crossing_mask handling (compute_ring_crossing_mask),
    since several Saturn windows have has_rings=True
  - correct pre-warp-vs-post-warp shift application order, matching
    derotate_filter()'s real 2026-08-11 fix: frames logged with
    align_method == "pre_warp_center" had their align_shift_px applied to
    the RAW frame BEFORE spherical_derotation_warp(); frames logged with
    "limb_center" or "phase_correlate" had it applied AFTER warping (the
    old pattern). Using the wrong order for pre_warp_center frames would
    not reproduce what actually went into the real stack.

All geometry/params (disk_center_px, disk_radius_px, align_shift_px, dt_sec,
period_hours, warp_scale, pole_pa_deg, flip_direction, has_rings,
sub_observer_lat_deg) are read directly from each window's real
derotation_log.json -- never guessed or recomputed except:
  - ref_semi_b (needed for compute_ring_crossing_mask, and for
    polar_equatorial_ratio): NOT present in derotation_log.json (only
    disk_radius_px = semi_major is logged). Re-derived from the reference
    frame via the real dr.find_disk_center() (same function
    derotate_filter() itself calls) to get semi_b/semi_a, then rescaled so
    semi_a matches the logged disk_radius_px exactly (preserving the
    measured aspect ratio at the logged, possibly-radius-shared, scale).
    This is an approximation of the exact production ref_semi_b (which
    depends on shared_radius_px/shared_shape not fully recoverable from the
    log) -- flagged explicitly in the report.

Only real production functions are called: spherical_derotation_warp,
apply_shift, compute_ring_crossing_mask, find_disk_center, _to_luminance.
"""
import sys
sys.path.insert(0, '.')
import glob
import json

import cv2
import numpy as np

from pipeline.modules import derotation as dr
from pipeline.modules import image_io

WIN_DIR = "Saturn_Data/step04_derotated"


def find_path(stem):
    cands = glob.glob(f"Saturn_Data/step02_lucky_stack/{stem}*.tif")
    return cands[0] if cands else None


def crop_roi(lum, cx, cy, r):
    h, w = lum.shape
    ys, ye = max(0, int(cy - r)), min(h, int(cy + r))
    xs, xe = max(0, int(cx - r)), min(w, int(cx + r))
    return lum[ys:ye, xs:xe]


def phase_peak_response(ref_crop, tgt_crop):
    ref_f32 = ref_crop.astype(np.float32)
    tgt_f32 = tgt_crop.astype(np.float32)
    (raw_dx, raw_dy), resp = cv2.phaseCorrelate(ref_f32, tgt_f32)
    return (-float(raw_dx), -float(raw_dy)), float(resp)


def process(win_idx, filt):
    log_path = f"{WIN_DIR}/window_{win_idx:02d}/derotation_log.json"
    log = json.load(open(log_path))
    if filt not in log["filters"]:
        return None, "filter not in log"
    flog = log["filters"][filt]
    frames = flog["frames"]

    ref_frame = next((f for f in frames if f["align_method"] == "reference"), None)
    if ref_frame is None:
        return None, "no reference frame in log"
    ref_stem = ref_frame["stem"]
    ref_cx, ref_cy = ref_frame["disk_center_px"]
    disk_r = ref_frame["disk_radius_px"]  # == disk_radius_px for every frame (shared)

    period_hours = flog["period_hours"]
    warp_scale = flog["warp_scale"]
    pole_pa = flog["pole_pa_deg"]
    flip_dir = flog["flip_direction"]
    has_rings = flog.get("has_rings", False)
    sub_obs_lat = flog.get("sub_observer_lat_deg", 0.0)

    ref_path = find_path(ref_stem)
    if ref_path is None:
        return None, f"ref raw file not found for stem {ref_stem}"
    ref_raw = image_io.read_tif(ref_path)
    ref_lum = dr._to_luminance(ref_raw if ref_raw.ndim == 2 else ref_raw.mean(axis=2).astype(np.float32))

    # Re-derive aspect ratio (semi_b/semi_a) via the real find_disk_center(),
    # since derotation_log.json only stores the (possibly radius-shared)
    # semi_major, not semi_b. Rescale semi_b so semi_a == logged disk_r,
    # preserving the measured aspect ratio at the logged scale.
    _cx2, _cy2, semi_a2, semi_b2, _ang2 = dr.find_disk_center(ref_lum)
    aspect = float(semi_b2 / max(semi_a2, 1.0))
    ref_semi_b_est = disk_r * aspect
    polar_eq_ratio = float(np.clip(aspect, 0.85, 1.0))

    ring_crossing_mask = None
    if has_rings:
        h, w = ref_lum.shape[:2]
        ring_crossing_mask = dr.compute_ring_crossing_mask(
            h, w, ref_cx, ref_cy, disk_r, ref_semi_b_est, pole_pa, sub_obs_lat,
        )

    # Reference warped with dt=0 (the "target" every other frame should match).
    ref_warped = dr.spherical_derotation_warp(
        ref_lum, 0.0, ref_cx, ref_cy, disk_r,
        period_hours=period_hours, scale=warp_scale,
        flip_direction=flip_dir, pole_pa_deg=pole_pa,
        polar_equatorial_ratio=polar_eq_ratio,
        ring_crossing_mask=ring_crossing_mask,
    )
    rc = crop_roi(ref_warped, ref_cx, ref_cy, disk_r)

    results = []
    errors = []
    for f in frames:
        if f["align_method"] == "reference":
            continue
        stem = f["stem"]
        path = find_path(stem)
        if path is None:
            errors.append(f"{stem}: raw file not found")
            continue
        try:
            raw = image_io.read_tif(path)
            lum_raw = dr._to_luminance(raw if raw.ndim == 2 else raw.mean(axis=2).astype(np.float32))

            dt_sec = f["dt_sec"]
            dx, dy = f["align_shift_px"]
            method = f["align_method"]

            if method == "pre_warp_center":
                # Real production order (post-2026-08-11 fix): shift applied
                # to the RAW frame BEFORE warping.
                pre_shifted = dr.apply_shift(lum_raw, dx, dy)
                warped = dr.spherical_derotation_warp(
                    pre_shifted, dt_sec, ref_cx, ref_cy, disk_r,
                    period_hours=period_hours, scale=warp_scale,
                    flip_direction=flip_dir, pole_pa_deg=pole_pa,
                    polar_equatorial_ratio=polar_eq_ratio,
                    ring_crossing_mask=ring_crossing_mask,
                )
                corrected_lum = dr._to_luminance(warped)
            else:
                # limb_center / phase_correlate: shift applied AFTER warping.
                warped = dr.spherical_derotation_warp(
                    lum_raw, dt_sec, ref_cx, ref_cy, disk_r,
                    period_hours=period_hours, scale=warp_scale,
                    flip_direction=flip_dir, pole_pa_deg=pole_pa,
                    polar_equatorial_ratio=polar_eq_ratio,
                    ring_crossing_mask=ring_crossing_mask,
                )
                corrected = dr.apply_shift(warped, dx, dy)
                corrected_lum = dr._to_luminance(corrected)

            cc = crop_roi(corrected_lum, ref_cx, ref_cy, disk_r)
            (resid_dx, resid_dy), resid_resp = phase_peak_response(rc, cc)
            mag = float(np.hypot(resid_dx, resid_dy))

            results.append({
                "stem": stem,
                "align_method": method,
                "logged_shift": [round(dx, 3), round(dy, 3)],
                "residual_shift": [round(resid_dx, 3), round(resid_dy, 3)],
                "residual_mag_px": round(mag, 4),
                "phase_peak_response": round(resid_resp, 4),
            })
        except Exception as e:
            errors.append(f"{stem}: {type(e).__name__}: {e}")

    return {
        "window": win_idx,
        "filter": filt,
        "n_stacked": flog.get("n_stacked"),
        "has_rings": has_rings,
        "disk_r": disk_r,
        "polar_eq_ratio_used": round(polar_eq_ratio, 4),
        "results": results,
        "errors": errors,
    }, None


if __name__ == "__main__":
    all_out = {}
    for w in range(1, 10):
        for filt in ["IR", "R"]:
            key = f"window_{w:02d}_{filt}"
            out, err = process(w, filt)
            if err:
                print(f"== {key} == ERROR: {err}")
                all_out[key] = {"error": err}
                continue
            all_out[key] = out
            mags = [r["residual_mag_px"] for r in out["results"]]
            print(f"== {key} == n_stacked={out['n_stacked']} has_rings={out['has_rings']} "
                  f"n_measured={len(mags)} errors={len(out['errors'])}")
            for r in out["results"]:
                print(f"   {r['stem'][:35]:35s} method={r['align_method']:16s} "
                      f"logged_shift={r['logged_shift']} residual={r['residual_shift']} "
                      f"mag={r['residual_mag_px']:.3f} resp={r['phase_peak_response']}")
            for e in out["errors"]:
                print(f"   ERROR: {e}")
            if mags:
                print(f"   -> mean={np.mean(mags):.4f} median={np.median(mags):.4f} max={np.max(mags):.4f}")

    json.dump(all_out, open("scratch_residual_ir_vs_r_results.json", "w"), indent=2)
    print("\nWrote scratch_residual_ir_vs_r_results.json")
