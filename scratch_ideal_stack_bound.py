"""
Theoretical bound: how much sharpness/transition-width loss does
quality_weighted_stack() introduce PURELY from averaging, under IDEAL
conditions (zero real misalignment, zero real PSF/seeing difference)?

Ground truth: window_01 R filter's REFERENCE frame, run through the REAL
derotate_window() as a single-frame ("n=1") window -- i.e. it goes through
the exact same warp/align/scale code path production uses, just with
nothing to average against. This is our sharp "true" image.

Experiment A (ideal stack): three IDENTICAL COPIES of that same ground-truth
image, stacked via the REAL quality_weighted_stack() with the REAL
norm_score weights window_01/R actually used (0.8031, 0.8183, 0.9303), then
wavelet-sharpened via the same helper step05/wavelet_master.run() uses
internally (sharpen_disk_aware with master params). If quality_weighted_stack
is lossless, this must match the ground truth's own sharpened radial profile
to within float roundoff.

Experiment B (calibration curve): same three-copy construction, but frames 2
and 3 are sub-pixel shifted by +/- {0.3, 0.5, 1.0} px (frame 1 stays at 0,0,
mirroring how the real reference frame always has zero residual shift) via
the REAL apply_shift() bicubic function -- simulating pure residual
misalignment with NO real PSF difference. Measures transition-width
degradation as a function of shift magnitude alone.

Metric: transition width = smallest r/semi_a (searched 1.05..1.35) at which
the perpendicular-band radial profile (ring-plane axis, using window_01's
real pole_pa_deg) first drops to within 8% of the background floor
(estimated from the r/semi_a in [1.30,1.35] tail), averaged over both +x/-x
sides. Same threshold/definition applied uniformly to every condition so
comparisons are apples-to-apples.

No pipeline/ files are modified. Follows the hydration pattern from
scratch_warp_avg_isolate.py / scratch_cassini_scale_ab.py exactly (real
derotate_window(), real quality_weighted_stack(), real apply_shift(), real
sharpen_disk_aware()) -- nothing reimplemented.
"""
from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import map_coordinates

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.modules import image_io
from pipeline.modules import wavelet
from pipeline.config import WaveletConfig
from pipeline.modules.derotation import (
    derotate_window,
    find_disk_center,
    quality_weighted_stack,
    apply_shift,
)

STEP02_DIR = Path("Saturn_Data/step02_lucky_stack")
STEP04_DIR = Path("Saturn_Data/step04_derotated")
WINDOWS_JSON = Path("Saturn_Data/step03_quality/windows.json")
WINDOW_INDEX = 1
FILT = "R"

CROP_DIR = Path("scratch_investigation_crops2")
CROP_DIR.mkdir(exist_ok=True)

_WCFG = WaveletConfig()


def _parse_ts(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")


def _resolve_path(stem: str) -> str:
    matches = list(STEP02_DIR.glob(f"{stem}.tif"))
    if not matches:
        raise FileNotFoundError(stem)
    return str(matches[0])


def _wavelet_sharpen_like_step05(lum: np.ndarray) -> np.ndarray:
    """Mirror wavelet_master.run()'s real per-filter master-sharpen call
    (default config), matching scratch_cassini_scale_ab.py's helper."""
    cx, cy, rx, ry, angle_deg = find_disk_center(lum)
    if rx < 5:
        return lum
    return wavelet.sharpen_disk_aware(
        lum, cx, cy, rx,
        levels=_WCFG.levels,
        amounts=_WCFG.master_amounts,
        power=_WCFG.master_power,
        sharpen_filter=_WCFG.master_sharpen_filter,
        edge_feather_factor=_WCFG.edge_feather_factor,
        ry=ry, angle=np.radians(angle_deg),
        expand_px=_WCFG.disk_expand_px,
        denoise_amounts=_WCFG.master_denoise_amounts,
        filter_type=_WCFG.master_filter_type,
    )


def _hydrate_single(window: dict, stem: str) -> dict:
    """Hydrate a window containing ONLY the given stem for FILT (production
    single-frame path, exercises the real derotate_window() end to end)."""
    hydrated = {"center_time": _parse_ts(window["center_time"]), "per_filter": {}}
    pf = window["per_filter"][FILT]
    item = next(it for it in pf["included"] if it["stem"] == stem)
    hydrated["per_filter"][FILT] = {"included": [{
        "path": _resolve_path(item["stem"]),
        "stem": item["stem"],
        "timestamp": _parse_ts(item["timestamp"]),
        "norm_score": item["norm_score"],
    }]}
    return hydrated


def _run(hydrated, flog, tag, out_root):
    out_dir = out_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    return derotate_window(
        hydrated,
        required_filters=[FILT],
        period_hours=flog["period_hours"],
        warp_scale=flog["warp_scale"],
        align=flog["align_enabled"],
        normalize_brightness=flog["normalize_brightness"],
        min_quality_threshold=flog["min_quality_threshold"],
        pole_pa_deg=flog["pole_pa_deg"],
        color_mode=False,
        flip_direction=flog["flip_direction"],
        weight_power=flog["weight_power"],
        has_rings=flog["has_rings"],
        sub_observer_lat_deg=flog["sub_observer_lat_deg"],
        out_dir=out_dir,
    )


def _radial_profile(img2d, cx, cy, semi_a, pole_pa_deg, r_lo=1.0, r_hi=1.4, dr=0.002, perp_half=3):
    ang = np.radians(pole_pa_deg)
    cos_a, sin_a = np.cos(ang), np.sin(ang)
    r_values = np.arange(r_lo * semi_a, r_hi * semi_a, dr * semi_a)
    perp = np.arange(-perp_half, perp_half + 0.5, 1.0)
    profiles = {}
    for side, sign in (("+x", 1.0), ("-x", -1.0)):
        vals = []
        for r in r_values:
            xr = sign * r
            xs = xr * cos_a - perp * sin_a
            ys = xr * sin_a + perp * cos_a
            xx = cx + xs
            yy = cy + ys
            samp = map_coordinates(img2d, [yy, xx], order=1, mode="constant", cval=np.nan)
            vals.append(float(np.nanmean(samp)))
        profiles[side] = np.array(vals)
    return r_values / semi_a, profiles


def _transition_width(r_over_a, vals):
    """Smallest r/semi_a where vals first falls within 8% of the background
    floor (mean of the last 15% of samples, i.e. r/semi_a in [~1.30,1.4)).
    Linearly interpolates between the two bracketing samples so the result
    is not quantized to the sampling grid (needed since sub-pixel shifts can
    move the crossing by much less than one dr step)."""
    tail_n = max(3, len(vals) // 7)
    bg = np.nanmean(vals[-tail_n:])
    peak = np.nanmax(vals)
    thresh = bg + 0.08 * (peak - bg)
    prev_r, prev_v = None, None
    for r, v in zip(r_over_a, vals):
        if np.isnan(v):
            continue
        if v <= thresh:
            if prev_v is not None and prev_v > thresh:
                # interpolate crossing point between prev and current sample
                frac = (prev_v - thresh) / (prev_v - v)
                r_cross = prev_r + frac * (r - prev_r)
                return float(r_cross), float(bg), float(peak)
            return float(r), float(bg), float(peak)
        prev_r, prev_v = r, v
    return float(r_over_a[-1]), float(bg), float(peak)


def _avg_transition(img2d, cx, cy, semi_a, pole_pa_deg):
    r_over_a, profiles = _radial_profile(img2d, cx, cy, semi_a, pole_pa_deg)
    tw_plus, bg_p, pk_p = _transition_width(r_over_a, profiles["+x"])
    tw_minus, bg_m, pk_m = _transition_width(r_over_a, profiles["-x"])
    return (tw_plus + tw_minus) / 2.0, tw_plus, tw_minus


def _save_crop(img2d, cx, cy, semi_a, tag):
    half = int(semi_a * 1.6)
    y0, y1 = max(0, int(cy - half * 0.5)), min(img2d.shape[0], int(cy + half * 0.5))
    x0, x1 = max(0, int(cx - half)), min(img2d.shape[1], int(cx + half))
    crop = img2d[y0:y1, x0:x1]
    crop_norm = np.clip(crop / (np.percentile(crop, 99.5) + 1e-9), 0, 1)
    crop_u8 = (crop_norm * 255).astype(np.uint8)
    crop_u8 = cv2.resize(crop_u8, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_NEAREST)
    out_path = CROP_DIR / f"{tag}.png"
    cv2.imwrite(str(out_path), crop_u8)
    return out_path


def main():
    data = json.load(open(WINDOWS_JSON))
    window = {w["window_index"]: w for w in data["selected_windows"]}[WINDOW_INDEX]
    window_log = json.load(open(STEP04_DIR / f"window_{WINDOW_INDEX:02d}" / "derotation_log.json"))
    flog = window_log["filters"][FILT]

    ref_stem = flog["reference_stem"]
    real_weights = [fr["norm_score"] for fr in flog["frames"]]
    print(f"Window {WINDOW_INDEX} filter {FILT}: reference_stem={ref_stem}")
    print(f"Real norm_score weights (order = frames[] in log): {real_weights}")
    pole_pa = flog["pole_pa_deg"]

    with tempfile.TemporaryDirectory() as tmpdir:
        out_root = Path(tmpdir)

        # ── Ground truth: reference frame run through derotate_window() alone ──
        hydrated_ref = _hydrate_single(window, ref_stem)
        n1_results = _run(hydrated_ref, flog, "n1_ref", out_root)
        n1_path, n1_log = n1_results[FILT]
        gt_img = image_io.read_tif(str(n1_path))
        gt_lum = gt_img if gt_img.ndim == 2 else gt_img.mean(axis=2).astype(np.float32)
        cx, cy, semi_a, semi_b, angle = find_disk_center(gt_lum)
        print(f"Ground-truth (n=1, reference frame) disk fit: cx={cx:.2f} cy={cy:.2f} semi_a={semi_a:.2f}")

        gt_sharp = _wavelet_sharpen_like_step05(gt_lum)
        gt_tw, gt_tw_p, gt_tw_m = _avg_transition(gt_sharp, cx, cy, semi_a, pole_pa)
        print(f"\n[GROUND TRUTH n=1] transition width r/semi_a = {gt_tw:.4f}  (+x={gt_tw_p:.4f} -x={gt_tw_m:.4f})")
        _save_crop(gt_sharp, cx, cy, semi_a, "A_ground_truth_n1")

        # ── Real n=3 stack (for reference / sanity anchor) ──────────────────
        real_n3_path = STEP04_DIR / f"window_{WINDOW_INDEX:02d}" / f"{FILT}_derotated.tif"
        n3_img = image_io.read_tif(real_n3_path)
        n3_lum = n3_img if n3_img.ndim == 2 else n3_img.mean(axis=2).astype(np.float32)
        n3_sharp = _wavelet_sharpen_like_step05(n3_lum)
        n3_tw, n3_tw_p, n3_tw_m = _avg_transition(n3_sharp, cx, cy, semi_a, pole_pa)
        print(f"[REAL n=3 STACK (production)] transition width r/semi_a = {n3_tw:.4f}  (+x={n3_tw_p:.4f} -x={n3_tw_m:.4f})")
        _save_crop(n3_sharp, cx, cy, semi_a, "B_real_n3_production")

        # ── Experiment A: 3 IDENTICAL copies, real quality_weighted_stack ──
        ideal_stack = quality_weighted_stack([gt_lum, gt_lum, gt_lum], real_weights, weight_power=flog["weight_power"])
        max_abs_diff = float(np.max(np.abs(ideal_stack.astype(np.float64) - gt_lum.astype(np.float64))))
        print(f"\n[EXP A: 3 identical copies] max|stack - ground_truth| pre-sharpen = {max_abs_diff:.3e} (float32 range [0,1])")
        ideal_sharp = _wavelet_sharpen_like_step05(ideal_stack)
        ideal_tw, ideal_tw_p, ideal_tw_m = _avg_transition(ideal_sharp, cx, cy, semi_a, pole_pa)
        max_abs_diff_sharp = float(np.max(np.abs(ideal_sharp.astype(np.float64) - gt_sharp.astype(np.float64))))
        print(f"[EXP A: 3 identical copies, sharpened] transition width r/semi_a = {ideal_tw:.4f}  (+x={ideal_tw_p:.4f} -x={ideal_tw_m:.4f})")
        print(f"[EXP A] max|sharpened_stack - sharpened_ground_truth| = {max_abs_diff_sharp:.3e}")
        _save_crop(ideal_sharp, cx, cy, semi_a, "C_ideal_3x_identical_stack")

        # ── Experiment B: sub-pixel shift calibration curve ─────────────────
        print("\n[EXP B: sub-pixel misalignment calibration]")
        calib = []
        for shift_px in (0.3, 0.5, 1.0, 2.0, 3.0, 5.0):
            # Frame 1 (highest-ranked in real log order == reference-like) stays
            # put; frames 2/3 shifted +shift/-shift along x, mirroring the real
            # pre_warp_shift pattern of one frame at 0 and others displaced in
            # opposite directions.
            copies = [
                gt_lum,
                apply_shift(gt_lum, shift_px, 0.0),
                apply_shift(gt_lum, -shift_px, 0.0),
            ]
            shifted_stack = quality_weighted_stack(copies, real_weights, weight_power=flog["weight_power"])
            shifted_sharp = _wavelet_sharpen_like_step05(shifted_stack)
            tw, tw_p, tw_m = _avg_transition(shifted_sharp, cx, cy, semi_a, pole_pa)
            delta = tw - gt_tw
            print(f"  shift=+/-{shift_px}px: transition width r/semi_a = {tw:.4f}  (delta vs ground truth = {delta:+.4f})  (+x={tw_p:.4f} -x={tw_m:.4f})")
            calib.append({"shift_px": shift_px, "transition_width": tw, "delta_vs_gt": delta})
            _save_crop(shifted_sharp, cx, cy, semi_a, f"D_shift_{shift_px}px_stack")

        results = {
            "window": WINDOW_INDEX,
            "filter": FILT,
            "reference_stem": ref_stem,
            "real_weights": real_weights,
            "pole_pa_deg": pole_pa,
            "semi_a": semi_a,
            "ground_truth_n1_transition_width": gt_tw,
            "real_n3_production_transition_width": n3_tw,
            "ideal_3x_identical_stack_transition_width": ideal_tw,
            "ideal_stack_max_abs_diff_pre_sharpen": max_abs_diff,
            "ideal_stack_max_abs_diff_post_sharpen": max_abs_diff_sharp,
            "subpixel_shift_calibration": calib,
        }
        Path("scratch_ideal_stack_bound_results.json").write_text(json.dumps(results, indent=2, default=float))

    print(f"\nCrops saved under {CROP_DIR}/")


if __name__ == "__main__":
    main()
