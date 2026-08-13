"""Test hypothesis: does raising quality_weighted_stack's weight_power (an
existing config knob, already wired through derotate_window()) recover
Saturn window_01's globe-limb sharpness deficit (r/semi_a ~1.15 single-frame
vs ~1.25 real 3-frame stack, per this session's established radial-profile
methodology)?

Follows established patterns exactly:
  - derotate_window() hydration: scratch_cassini_scale_ab.py / scratch_window_level_sweep.py
  - wavelet sharpen mirror of step05: scratch_cassini_scale_ab.py's
    _wavelet_sharpen_like_step05
  - radial profile along the ring-plane axis (perpendicular decomposition
    using pole_pa_deg): scratch_step07_groundtruth.py / scratch_warp_avg_isolate.py

Per feedback_ab_test_via_real_pipeline: uses derotate_window() (not
derotate_filter() directly), so shared_shape/consensus logic runs exactly as
production does. No pipeline/ files modified.
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
WINDOW_INDEX = 1
FILTERS = ["IR", "R"]
WEIGHT_POWERS = [1.0, 1.5, 2.0, 3.0, 4.0, 8.0]

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


def _hydrate_full(window: dict) -> dict:
    hydrated = {"center_time": _parse_ts(window["center_time"]), "per_filter": {}}
    for filt in FILTERS:
        pf = window["per_filter"][filt]
        included = [{
            "path": _resolve_path(it["stem"]),
            "stem": it["stem"],
            "timestamp": _parse_ts(it["timestamp"]),
            "norm_score": it["norm_score"],
        } for it in pf["included"]]
        hydrated["per_filter"][filt] = {"included": included}
    return hydrated


def _hydrate_single_best(window: dict):
    hydrated = {"center_time": _parse_ts(window["center_time"]), "per_filter": {}}
    best = {}
    for filt in FILTERS:
        pf = window["per_filter"][filt]
        item = max(pf["included"], key=lambda it: it["norm_score"])
        hydrated["per_filter"][filt] = {"included": [{
            "path": _resolve_path(item["stem"]),
            "stem": item["stem"],
            "timestamp": _parse_ts(item["timestamp"]),
            "norm_score": item["norm_score"],
        }]}
        best[filt] = item
    return hydrated, best


def _run(hydrated, flog, weight_power, tag, out_root):
    out_dir = out_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    return derotate_window(
        hydrated,
        required_filters=FILTERS,
        period_hours=flog["period_hours"],
        warp_scale=flog["warp_scale"],
        align=flog["align_enabled"],
        normalize_brightness=flog["normalize_brightness"],
        min_quality_threshold=flog["min_quality_threshold"],
        pole_pa_deg=flog["pole_pa_deg"],
        color_mode=False,
        flip_direction=flog["flip_direction"],
        weight_power=weight_power,
        has_rings=flog["has_rings"],
        sub_observer_lat_deg=flog["sub_observer_lat_deg"],
        out_dir=out_dir,
    )


def _wavelet_sharpen_like_step05(lum: np.ndarray, cx, cy, rx, ry, angle_deg) -> np.ndarray:
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


def radial_profile(img, cx, cy, semi_a, pole_pa_deg, r0=0.95, r1=1.35, step_px=0.5, half_width=2.0):
    """Same convention as scratch_step07_groundtruth.py / scratch_warp_avg_isolate.py:
    thin band along the ring-plane axis (pole_pa_deg direction), +x/-x sides."""
    ang = np.radians(pole_pa_deg)
    cos_a, sin_a = np.cos(ang), np.sin(ang)
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    dx, dy = xx - cx, yy - cy
    xr = dx * cos_a + dy * sin_a
    yr = -dx * sin_a + dy * cos_a

    r_min_px, r_max_px = r0 * semi_a, r1 * semi_a
    radii_px = np.arange(r_min_px, r_max_px + 1e-9, step_px)
    prof_pos, prof_neg = [], []
    for r in radii_px:
        band_pos = (np.abs(xr - r) < half_width) & (np.abs(yr) < half_width)
        band_neg = (np.abs(xr + r) < half_width) & (np.abs(yr) < half_width)
        prof_pos.append(float(img[band_pos].mean()) if band_pos.sum() > 0 else np.nan)
        prof_neg.append(float(img[band_neg].mean()) if band_neg.sum() > 0 else np.nan)
    return radii_px / semi_a, np.array(prof_pos), np.array(prof_neg)


def first_below(rr, prof, thresh=0.15):
    for r, v in zip(rr, prof):
        if not np.isnan(v) and v < thresh:
            return float(r)
    return None


def noise_proxy(img, cx, cy, semi_a, pole_pa_deg):
    """High-frequency residual std inside a smooth interior sector (0.25-0.45
    semi_a from center, restricted to a wedge AWAY from the ring-plane axis
    so we sample globe surface, not ring/limb structure) -- a graininess/SNR
    proxy independent of the limb-sharpness metric above."""
    ang = np.radians(pole_pa_deg)
    cos_a, sin_a = np.cos(ang), np.sin(ang)
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    dx, dy = xx - cx, yy - cy
    xr = dx * cos_a + dy * sin_a
    yr = -dx * sin_a + dy * cos_a
    r = np.sqrt(xr ** 2 + yr ** 2)
    # wedge near the pole axis (|xr| small relative to |yr|) inside the globe
    wedge = (r > 0.20 * semi_a) & (r < 0.55 * semi_a) & (np.abs(xr) < 0.35 * np.abs(yr) + 1.0)
    if wedge.sum() < 30:
        return None
    hp = img.astype(np.float32) - cv2.GaussianBlur(img.astype(np.float32), (0, 0), 3.0)
    return float(hp[wedge].std())


def save_crop(img, cx, cy, semi_a, pole_pa_deg, out_path, upscale=3):
    h, w = img.shape
    M = cv2.getRotationMatrix2D((cx, cy), pole_pa_deg, 1.0)
    rot = cv2.warpAffine(img.astype(np.float32), M, (w, h), flags=cv2.INTER_LINEAR)
    half = int(semi_a * 1.6)
    band = int(semi_a * 0.55)
    x0, x1 = max(int(cx - half), 0), min(int(cx + half), w)
    y0, y1 = max(int(cy - band), 0), min(int(cy + band), h)
    crop = rot[y0:y1, x0:x1]
    p995 = np.percentile(crop, 99.5)
    crop_n = np.clip(crop / max(p995, 1e-6), 0, 1)
    crop_u8 = (crop_n * 255).astype(np.uint8)
    crop_big = cv2.resize(crop_u8, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(out_path), crop_big)


def main():
    data = json.load(open(WINDOWS_JSON))
    window = {w["window_index"]: w for w in data["selected_windows"]}[WINDOW_INDEX]
    window_log = json.load(open(STEP04_DIR / f"window_{WINDOW_INDEX:02d}" / "derotation_log.json"))
    any_flog = window_log["filters"]["IR"]
    print(f"Confirmed default weight_power (both filters) = "
          f"{window_log['filters']['IR']['weight_power']} / {window_log['filters']['R']['weight_power']}")

    hydrated_full = _hydrate_full(window)
    hydrated_single, best = _hydrate_single_best(window)

    results = {filt: {} for filt in FILTERS}

    # Fixed reference geometry per filter, fit ONCE on the untouched raw
    # single-best frame -- reused for every variant's radial profile so that
    # small per-run disk-fit jitter on the (differently-sharpened) stacks
    # cannot masquerade as a sharpness difference. Matches
    # scratch_warp_avg_isolate.py's established practice ("disk itself
    # doesn't move -- only interior content changed").
    geom = {}
    for filt in FILTERS:
        raw = image_io.read_tif(_resolve_path(best[filt]["stem"]))
        raw_lum = raw if raw.ndim == 2 else raw.mean(axis=2).astype(np.float32)
        cx, cy, semi_a, semi_b, angle = find_disk_center(raw_lum)
        geom[filt] = (cx, cy, semi_a, semi_b, angle)
        print(f"[{filt}] fixed reference geometry (from raw best frame {best[filt]['stem']}): "
              f"cx={cx:.2f} cy={cy:.2f} semi_a={semi_a:.2f} semi_b={semi_b:.2f} angle={angle:.1f}")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_root = Path(tmpdir)

        # n=1 theoretical-best baseline
        n1_results = _run(hydrated_single, any_flog, any_flog["weight_power"], "n1", out_root)

        for filt in FILTERS:
            cx, cy, semi_a, semi_b, angle = geom[filt]
            n1_path, _ = n1_results[filt]
            n1_img = image_io.read_tif(str(n1_path))
            n1_lum = n1_img if n1_img.ndim == 2 else n1_img.mean(axis=2).astype(np.float32)
            n1_sharp = _wavelet_sharpen_like_step05(n1_lum, cx, cy, semi_a, semi_b, angle)
            rr, ppos, pneg = radial_profile(n1_sharp, cx, cy, semi_a, any_flog["pole_pa_deg"])
            t_pos, t_neg = first_below(rr, ppos), first_below(rr, pneg)
            results[filt]["n1_baseline"] = {"transition_pos": t_pos, "transition_neg": t_neg}
            print(f"[{filt}] n=1 baseline: transition +x={t_pos} -x={t_neg} "
                  f"(profile min={np.nanmin(np.concatenate([ppos, pneg])):.4f})")

        for wp in WEIGHT_POWERS:
            tag = f"wp_{wp}"
            wp_results = _run(hydrated_full, any_flog, wp, tag, out_root)
            for filt in FILTERS:
                cx, cy, semi_a, semi_b, angle = geom[filt]
                out_path, log_dict = wp_results[filt]
                img = image_io.read_tif(str(out_path))
                lum = img if img.ndim == 2 else img.mean(axis=2).astype(np.float32)
                sharpened = _wavelet_sharpen_like_step05(lum, cx, cy, semi_a, semi_b, angle)
                rr, ppos, pneg = radial_profile(sharpened, cx, cy, semi_a, any_flog["pole_pa_deg"])
                t_pos, t_neg = first_below(rr, ppos), first_below(rr, pneg)
                noise = noise_proxy(lum, cx, cy, semi_a, any_flog["pole_pa_deg"])
                results[filt][f"wp_{wp}"] = {
                    "transition_pos": t_pos, "transition_neg": t_neg,
                    "noise_proxy": noise, "n_stacked": log_dict.get("n_stacked"),
                    "profile_min": float(np.nanmin(np.concatenate([ppos, pneg]))),
                }
                print(f"[{filt}] wp={wp:4.1f}: transition +x={t_pos} -x={t_neg} "
                      f"noise_proxy={noise} profile_min={results[filt][f'wp_{wp}']['profile_min']:.4f}")

                if wp in (1.0, 2.0, 8.0):
                    save_crop(sharpened, cx, cy, semi_a, any_flog["pole_pa_deg"],
                              CROP_DIR / f"window01_{filt}_wp{wp}.png")

    Path("scratch_weight_power_radial_sweep_results.json").write_text(json.dumps(results, indent=2))

    print("\n=== SUMMARY (transition r/semi_a, averaged +x/-x, first drop below 0.15) ===")
    for filt in FILTERS:
        n1 = results[filt]["n1_baseline"]
        n1_avg = np.nanmean([v for v in (n1["transition_pos"], n1["transition_neg"]) if v is not None])
        print(f"\n{filt}: n=1 baseline avg transition = {n1_avg:.4f}")
        for wp in WEIGHT_POWERS:
            r = results[filt].get(f"wp_{wp}")
            if not r:
                continue
            avg = np.nanmean([v for v in (r["transition_pos"], r["transition_neg"]) if v is not None])
            print(f"  wp={wp:4.1f}: avg transition={avg:.4f}  noise_proxy={r['noise_proxy']:.5f}")

    print(f"\nCrops in {CROP_DIR}/")


if __name__ == "__main__":
    main()
