"""Isolate: does the Cassini Division get destroyed by multi-frame AVERAGING,
or by the per-frame spherical_derotation_warp itself, independent of the
already-fixed center+scale registration?

Three variants per filter (Saturn window_01, IR & R):
  A) raw       -- untouched step02 TIF of the single highest-norm_score frame
                  (true control, no warp, no stack)
  B) n1_warp   -- derotate_window() on a window containing ONLY that same
                  single frame (real warp runs, real pre-warp align/scale
                  runs against itself, but there is nothing to average)
  C) n3_stack  -- derotate_window() on the full window (n=3 frames), the
                  real production path (post scale-fix)

Follows scratch_cassini_scale_ab.py's hydration pattern exactly (same
STEP02_DIR globbing, same derotate_window() kwargs pulled from the real
derotation_log.json). No pipeline/ files are modified.
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
from pipeline.modules.derotation import derotate_window, find_disk_center

STEP02_DIR = Path("Saturn_Data/step02_lucky_stack")
STEP04_DIR = Path("Saturn_Data/step04_derotated")
WINDOWS_JSON = Path("Saturn_Data/step03_quality/windows.json")
WINDOW_INDEX = 1
FILTERS = ["IR", "R"]

CROP_DIR = Path("scratch_investigation_crops")
CROP_DIR.mkdir(exist_ok=True)


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


def _run(hydrated, flog, tag, out_root):
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
        weight_power=flog["weight_power"],
        has_rings=flog["has_rings"],
        sub_observer_lat_deg=flog["sub_observer_lat_deg"],
        out_dir=out_dir,
    )


def _radial_profile(img2d, cx, cy, semi_a, pole_pa_deg, r_lo=1.0, r_hi=2.4, dr=0.5, perp_half=3):
    """Sample mean brightness in a thin perpendicular band, along the
    ring-plane axis (rotated by pole_pa_deg), on both +x and -x sides."""
    ang = np.radians(pole_pa_deg)
    cos_a, sin_a = np.cos(ang), np.sin(ang)
    h, w = img2d.shape
    r_values = np.arange(r_lo * semi_a, r_hi * semi_a, dr)
    perp = np.arange(-perp_half, perp_half + 0.5, 1.0)
    profiles = {}
    for side, sign in (("+x", 1.0), ("-x", -1.0)):
        vals = []
        for r in r_values:
            xr = sign * r
            # invert xr = dx*cos+dy*sin, yr=-dx*sin+dy*cos  =>  dx = xr*cos - yr*sin, dy = xr*sin + yr*cos
            xs = xr * cos_a - perp * sin_a
            ys = xr * sin_a + perp * cos_a
            xx = cx + xs
            yy = cy + ys
            samp = map_coordinates(img2d, [yy, xx], order=1, mode="constant", cval=np.nan)
            vals.append(float(np.nanmean(samp)))
        profiles[side] = np.array(vals)
    return r_values / semi_a, profiles


def _extrema(r_over_a, vals):
    """Report empirical local min/max (simple neighbor comparison, ignoring NaN)."""
    out = []
    v = vals
    for i in range(2, len(v) - 2):
        if np.isnan(v[i]):
            continue
        window = v[i - 2:i + 3]
        if np.all(np.isnan(window)):
            continue
        if v[i] == np.nanmax(window) and v[i] >= v[i - 1] and v[i] >= v[i + 1]:
            out.append(("max", round(float(r_over_a[i]), 3), round(float(v[i]), 5)))
        elif v[i] == np.nanmin(window) and v[i] <= v[i - 1] and v[i] <= v[i + 1]:
            out.append(("min", round(float(r_over_a[i]), 3), round(float(v[i]), 5)))
    return out


def _save_crop(img2d, cx, cy, semi_a, tag, filt):
    half = int(semi_a * 2.3)
    y0, y1 = max(0, int(cy - half * 0.4)), min(img2d.shape[0], int(cy + half * 0.4))
    x0, x1 = max(0, int(cx - half)), min(img2d.shape[1], int(cx + half))
    crop = img2d[y0:y1, x0:x1]
    crop_norm = np.clip(crop / (np.percentile(crop, 99.5) + 1e-9), 0, 1)
    crop_u8 = (crop_norm * 255).astype(np.uint8)
    crop_u8 = cv2.resize(crop_u8, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_NEAREST)
    out_path = CROP_DIR / f"window01_{filt}_{tag}.png"
    cv2.imwrite(str(out_path), crop_u8)
    return out_path


def main():
    data = json.load(open(WINDOWS_JSON))
    window = {w["window_index"]: w for w in data["selected_windows"]}[WINDOW_INDEX]
    window_log = json.load(open(STEP04_DIR / f"window_{WINDOW_INDEX:02d}" / "derotation_log.json"))

    hydrated_full = _hydrate_full(window)
    hydrated_single, best = _hydrate_single_best(window)

    print("Single highest-norm_score frame chosen per filter:")
    for filt in FILTERS:
        it = best[filt]
        print(f"  {filt}: {it['stem']} norm_score={it['norm_score']}")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_root = Path(tmpdir)
        # flog differs slightly per filter for period/scale etc in this dataset,
        # but IR's/R's values are identical here -- use IR's as the shared call
        # config (matches scratch_cassini_scale_ab.py's `any_flog` pattern).
        any_flog = window_log["filters"]["IR"]
        n3_results = _run(hydrated_full, any_flog, "n3", out_root)
        n1_results = _run(hydrated_single, any_flog, "n1", out_root)

        for filt in FILTERS:
            print(f"\n===== {filt} =====")
            # --- raw control ---
            raw = image_io.read_tif(_resolve_path(best[filt]["stem"]))
            raw_lum = raw if raw.ndim == 2 else raw.mean(axis=2).astype(np.float32)
            r_cx, r_cy, r_semi_a, r_semi_b, r_angle = find_disk_center(raw_lum)

            # --- n1 warped single frame ---
            n1_path, n1_log = n1_results[filt]
            n1_img = image_io.read_tif(str(n1_path))
            n1_lum = n1_img if n1_img.ndim == 2 else n1_img.mean(axis=2).astype(np.float32)
            n1_cx, n1_cy, n1_semi_a, n1_semi_b, n1_angle = find_disk_center(n1_lum)

            # --- n3 stack ---
            n3_path, n3_log = n3_results[filt]
            n3_img = image_io.read_tif(str(n3_path))
            n3_lum = n3_img if n3_img.ndim == 2 else n3_img.mean(axis=2).astype(np.float32)
            n3_cx, n3_cy, n3_semi_a, n3_semi_b, n3_angle = find_disk_center(n3_lum)

            print(f"  fitted semi_a: raw={r_semi_a:.2f}px n1={n1_semi_a:.2f}px n3={n3_semi_a:.2f}px "
                  f"(agreement: {abs(n1_semi_a-r_semi_a)/r_semi_a*100:.2f}% / {abs(n3_semi_a-r_semi_a)/r_semi_a*100:.2f}%)")
            print(f"  n1 dt_sec={n1_log['frames'][0]['dt_sec']}  reference_stem={n1_log['reference_stem']}")
            print(f"  n3 reference_stem={n3_log['reference_stem']} n_stacked={n3_log['n_stacked']}")

            pole_pa = any_flog["pole_pa_deg"]
            # Use the RAW fit's geometry for ALL THREE (disk itself doesn't move --
            # only interior content changed -- confirms the sanity check above).
            cx, cy, semi_a = r_cx, r_cy, r_semi_a

            for tag, img2d in (("raw", raw_lum), ("n1_warp", n1_lum), ("n3_stack", n3_lum)):
                r_over_a, profiles = _radial_profile(img2d, cx, cy, semi_a, pole_pa)
                for side in ("+x", "-x"):
                    ext = _extrema(r_over_a, profiles[side])
                    print(f"  [{tag} {side}] extrema (type, r/semi_a, value): {ext}")
                crop_path = _save_crop(img2d, cx, cy, semi_a, tag, filt)
                print(f"  [{tag}] crop saved: {crop_path}")

    print(f"\nAll crops in {CROP_DIR}/")


if __name__ == "__main__":
    main()
