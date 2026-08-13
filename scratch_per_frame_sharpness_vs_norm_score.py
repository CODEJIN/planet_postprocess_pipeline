"""Does individual per-frame sharpness at the globe limb vary in a way that
tracks norm_score, independent of stacking?

For each raw frame in a window (window_01 R+IR, plus 2 more windows chosen
for large norm_score spread), run it through derotate_window() as a
SINGLE-frame window (real pre-warp align/scale + real spherical de-rotation
warp, exactly as production would apply to that frame as part of a multi-
frame stack), then run it through wavelet_master's real sharpen_disk_aware
call (mirrors scratch_cassini_scale_ab.py's _wavelet_sharpen_like_step05).

For each individually-processed frame:
  - extract the same ring-plane-axis radial brightness profile used
    throughout this investigation (perpendicular-band decomposition,
    reused verbatim from scratch_step07_groundtruth.py's radial_profile),
    from r/semi_a ~0.95 to ~1.3, and report where it first drops below 0.15.
  - compute central-55%-of-semi_a Laplacian-variance sharpness (reused
    verbatim from scratch_window_level_sweep.py's _sharpness_central55) on
    the PRE-sharpen warped frame (avoids wavelet ringing throwing off the
    disk refit, per scratch_cassini_scale_ab.py's documented caveat).

Compares both metrics against the frame's own norm_score, within each
window, to see whether norm_score predicts raw per-frame edge sharpness at
all (it might not -- it could be measuring overall SNR/contrast instead).

No pipeline/ files are modified. Follows feedback_ab_test_via_real_pipeline:
always derotate_window(), never derotate_filter() directly.
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
from pipeline.modules.derotation import derotate_window, find_disk_center

STEP02_DIR = Path("Saturn_Data/step02_lucky_stack")
STEP04_DIR = Path("Saturn_Data/step04_derotated")
WINDOWS_JSON = Path("Saturn_Data/step03_quality/windows.json")
CROP_DIR = Path("scratch_investigation_crops2")
CROP_DIR.mkdir(exist_ok=True)

_WCFG = WaveletConfig()

# (window_index, filter) pairs to test. window_01 R/IR per task instructions;
# window_02 G and window_04 B chosen for their large norm_score spreads
# (0.9273 and 0.8075 respectively -- comparable to or larger than window_01's
# R spread of 0.1272), per inspection of windows.json across all 9 windows.
TARGETS = [(1, "R"), (1, "IR"), (2, "G"), (4, "B")]


def _parse_ts(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")


def _resolve_path(stem: str) -> str:
    matches = list(STEP02_DIR.glob(f"{stem}.tif"))
    if not matches:
        raise FileNotFoundError(stem)
    return str(matches[0])


def _hydrate_single_frame(window: dict, filt: str, item: dict) -> dict:
    """Hydrate a window dict containing ONLY this one frame, for this one
    filter -- so derotate_window() gives it the exact same pre-warp
    alignment + real spherical de-rotation warp treatment production would,
    with nothing to average against."""
    return {
        "center_time": _parse_ts(window["center_time"]),
        "per_filter": {
            filt: {"included": [{
                "path": _resolve_path(item["stem"]),
                "stem": item["stem"],
                "timestamp": _parse_ts(item["timestamp"]),
                "norm_score": item["norm_score"],
            }]}
        },
    }


def _run(hydrated, flog, filt, tag, out_root):
    out_dir = out_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    return derotate_window(
        hydrated,
        required_filters=[filt],
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


def _wavelet_sharpen_like_step05(lum: np.ndarray) -> np.ndarray:
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


def radial_profile(img: np.ndarray, cx, cy, semi_a, pole_pa_deg,
                    r0=0.95, r1=1.3, step_px=0.02, half_width=1.5):
    """Verbatim (param-adjusted range) from scratch_step07_groundtruth.py."""
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    ang = np.radians(pole_pa_deg)
    cos_a, sin_a = np.cos(ang), np.sin(ang)
    dx, dy = xx - cx, yy - cy
    xr = dx * cos_a + dy * sin_a
    yr = -dx * sin_a + dy * cos_a

    r_min_px, r_max_px = r0 * semi_a, r1 * semi_a
    radii_px = np.arange(r_min_px, r_max_px + 1e-9, step_px * semi_a)
    prof_pos, prof_neg = [], []
    for r in radii_px:
        band_pos = (np.abs(xr - r) < half_width) & (np.abs(yr) < half_width)
        band_neg = (np.abs(xr + r) < half_width) & (np.abs(yr) < half_width)
        prof_pos.append(float(img[band_pos].mean()) if band_pos.sum() > 0 else np.nan)
        prof_neg.append(float(img[band_neg].mean()) if band_neg.sum() > 0 else np.nan)
    return radii_px / semi_a, np.array(prof_pos), np.array(prof_neg)


def _transition_r(rr, prof, thresh=0.15):
    """First r/semi_a where profile drops below thresh and stays there
    (avoids catching a transient noise dip)."""
    valid = ~np.isnan(prof)
    for i in range(len(prof)):
        if not valid[i]:
            continue
        if prof[i] < thresh and np.all(prof[i:][~np.isnan(prof[i:])] < thresh + 0.05):
            return float(rr[i])
    return float("nan")


def _sharpness_central55(img2d: np.ndarray, cx, cy, semi_a) -> float:
    h, w = img2d.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = r < (0.55 * semi_a)
    lap = cv2.Laplacian(img2d.astype(np.float32), cv2.CV_32F, ksize=3)
    return float(np.var(lap[mask]))


def _save_crop(img2d, cx, cy, semi_a, name):
    half = int(semi_a * 2.3)
    y0, y1 = max(0, int(cy - half * 0.4)), min(img2d.shape[0], int(cy + half * 0.4))
    x0, x1 = max(0, int(cx - half)), min(img2d.shape[1], int(cx + half))
    crop = img2d[y0:y1, x0:x1]
    crop_norm = np.clip(crop / (np.percentile(crop, 99.5) + 1e-9), 0, 1)
    crop_u8 = (crop_norm * 255).astype(np.uint8)
    crop_u8 = cv2.resize(crop_u8, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_NEAREST)
    out_path = CROP_DIR / f"{name}.png"
    cv2.imwrite(str(out_path), crop_u8)
    return out_path


def main():
    data = json.load(open(WINDOWS_JSON))
    windows = {w["window_index"]: w for w in data["selected_windows"]}

    all_rows = []

    with tempfile.TemporaryDirectory() as tmpdir:
        out_root = Path(tmpdir)
        for wi, filt in TARGETS:
            window = windows[wi]
            window_log = json.load(open(STEP04_DIR / f"window_{wi:02d}" / "derotation_log.json"))
            flog = window_log["filters"][filt]
            included = window["per_filter"][filt]["included"]

            print(f"\n===== window_{wi:02d} {filt} (n={len(included)} frames, "
                  f"norm_scores={[it['norm_score'] for it in included]}) =====")

            frame_results = []
            for item in included:
                hydrated = _hydrate_single_frame(window, filt, item)
                tag = f"w{wi:02d}_{filt}_{item['stem']}"
                results = _run(hydrated, flog, filt, tag, out_root)
                out_path, log_dict = results[filt]
                if out_path is None:
                    print(f"  {item['stem']} norm_score={item['norm_score']}: SKIP no output ({log_dict.get('error')})")
                    continue

                img = image_io.read_tif(str(out_path))
                lum_warped = img if img.ndim == 2 else img.mean(axis=2).astype(np.float32)
                cx, cy, semi_a, semi_b, _angle = find_disk_center(lum_warped)
                if semi_a < 5:
                    print(f"  {item['stem']} norm_score={item['norm_score']}: SKIP disk detect failed")
                    continue

                sharp = _sharpness_central55(lum_warped, cx, cy, semi_a)

                lum_sharpened = _wavelet_sharpen_like_step05(lum_warped)
                rr, prof_pos, prof_neg = radial_profile(lum_sharpened, cx, cy, semi_a, flog["pole_pa_deg"])
                t_pos = _transition_r(rr, prof_pos)
                t_neg = _transition_r(rr, prof_neg)

                row = dict(window=wi, filter=filt, stem=item["stem"],
                           norm_score=item["norm_score"], semi_a=float(semi_a),
                           laplacian_var_central55=sharp,
                           transition_r_pos=t_pos, transition_r_neg=t_neg,
                           transition_r_mean=float(np.nanmean([t_pos, t_neg])))
                frame_results.append(row)
                all_rows.append(row)
                print(f"  {item['stem']} norm_score={item['norm_score']:.4f}: "
                      f"lap_var={sharp:.4e} transition_r(+x)={t_pos:.3f} "
                      f"transition_r(-x)={t_neg:.3f} mean={row['transition_r_mean']:.3f}")

            if len(frame_results) >= 2:
                by_score = sorted(frame_results, key=lambda r: r["norm_score"])
                lo, hi = by_score[0], by_score[-1]
                print(f"  lowest-score ({lo['stem']}, {lo['norm_score']:.4f}): "
                      f"lap_var={lo['laplacian_var_central55']:.4e} transition_r={lo['transition_r_mean']:.3f}")
                print(f"  highest-score ({hi['stem']}, {hi['norm_score']:.4f}): "
                      f"lap_var={hi['laplacian_var_central55']:.4e} transition_r={hi['transition_r_mean']:.3f}")

                # visual crops: sharpest vs least-sharp individual frame (by lap_var)
                by_lap = sorted(frame_results, key=lambda r: r["laplacian_var_central55"])
                softest, sharpest = by_lap[0], by_lap[-1]
                for tag_label, row in [("softest", softest), ("sharpest", sharpest)]:
                    out_path, _ = _run(
                        _hydrate_single_frame(window, filt,
                                               next(it for it in included if it["stem"] == row["stem"])),
                        flog, filt, f"crop_{wi:02d}_{filt}_{tag_label}", out_root)[filt]
                    img = image_io.read_tif(str(out_path))
                    lum_warped = img if img.ndim == 2 else img.mean(axis=2).astype(np.float32)
                    cx, cy, semi_a, _, _ = find_disk_center(lum_warped)
                    lum_sharpened = _wavelet_sharpen_like_step05(lum_warped)
                    crop_path = _save_crop(lum_sharpened, cx, cy, semi_a,
                                            f"w{wi:02d}_{filt}_{tag_label}_{row['stem']}_score{row['norm_score']:.3f}")
                    print(f"  crop [{tag_label}] saved: {crop_path}")

    out_json = Path("scratch_per_frame_sharpness_vs_norm_score_results.json")
    out_json.write_text(json.dumps(all_rows, indent=2))

    # Correlation check: does norm_score predict lap_var sharpness and/or
    # transition_r, across ALL frames pooled from all windows/filters tested?
    print("\n\n=== POOLED CORRELATION (all windows/filters) ===")
    scores = np.array([r["norm_score"] for r in all_rows])
    laps = np.array([r["laplacian_var_central55"] for r in all_rows])
    trans = np.array([r["transition_r_mean"] for r in all_rows])
    valid_t = ~np.isnan(trans)
    if len(scores) >= 3:
        corr_lap = np.corrcoef(scores, laps)[0, 1]
        print(f"corr(norm_score, laplacian_var) = {corr_lap:.3f}  (n={len(scores)})")
    if valid_t.sum() >= 3:
        corr_trans = np.corrcoef(scores[valid_t], trans[valid_t])[0, 1]
        print(f"corr(norm_score, transition_r)  = {corr_trans:.3f}  (n={valid_t.sum()}) "
              f"[negative = higher score -> smaller transition_r -> sharper, as hypothesized]")

    print(f"\nWrote {out_json}")
    print(f"Crops in {CROP_DIR}/")


if __name__ == "__main__":
    main()
