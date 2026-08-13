"""Mandatory post-wavelet visual check for the raw-sharpness-based frame
selection feature (2026-08-13 plan, section "real-data verification").

Runs derotate_window() with sharpness_selection_enabled OFF vs ON for one
representative Saturn window and one Jupiter window, applies the SAME
wavelet sharpening logic wavelet_master.py uses (including Saturn's
ring-aware extra-ellipse mask), and saves before/after crops for visual
inspection -- specifically checking for any new banding/discontinuity
artifact at the sharpness-exclusion boundary, since this project has
already been burned once this session by a stacking-weight feature that
looked fine pre-wavelet but produced a visible artifact after sharpening.
"""
from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.modules import image_io, wavelet
from pipeline.modules.derotation import (
    derotate_window,
    find_disk_center,
    _SATURN_RING_INNER_REQ,
    _SATURN_RING_OUTER_REQ,
)

MASTER_AMOUNTS = [200.0, 200.0, 200.0, 0.0, 0.0, 0.0]


def _parse_ts(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")


def _hydrate_window(window: dict, step02_dir: Path) -> dict:
    hydrated = {"center_time": _parse_ts(window["center_time"]), "per_filter": {}}
    for filt, pf in window["per_filter"].items():
        included = []
        for item in pf["included"]:
            matches = list(step02_dir.glob(f"{item['stem']}.tif"))
            if not matches:
                continue
            included.append({
                "path": str(matches[0]), "stem": item["stem"],
                "timestamp": _parse_ts(item["timestamp"]), "norm_score": item["norm_score"],
            })
        hydrated["per_filter"][filt] = {"included": included}
    return hydrated


def _wavelet_sharpen(img: np.ndarray, has_rings: bool, pole_pa_deg: float, sub_observer_lat_deg: float) -> np.ndarray:
    lum = img if img.ndim == 2 else img.mean(axis=2).astype(np.float32)
    cx, cy, rx, ry, angle = find_disk_center(lum)
    angle_rad = np.radians(pole_pa_deg) if has_rings else np.radians(angle)

    extra_rx = extra_ry = extra_angle = extra_gap_px = None
    if has_rings:
        sin_b = abs(np.sin(np.radians(sub_observer_lat_deg)))
        RING_MASK_SAFETY_FACTOR = 1.35
        extra_rx = rx * _SATURN_RING_OUTER_REQ * RING_MASK_SAFETY_FACTOR
        extra_ry = max(extra_rx * sin_b, 1e-6)
        extra_angle = angle_rad
        active_idxs = [i for i, a in enumerate(MASTER_AMOUNTS) if a != 0]
        max_active_level = max(active_idxs) if active_idxs else 0
        extra_gap_px = (2 ** max_active_level) * 2.0  # edge_feather_factor default 2.0

    return wavelet.sharpen_disk_aware(
        img, cx, cy, rx,
        levels=6, amounts=MASTER_AMOUNTS, power=1.0, sharpen_filter=0.0,
        edge_feather_factor=2.0, ry=ry, angle=angle_rad, expand_px=0.0,
        denoise_amounts=[0.0] * 6, filter_type="gaussian",
        extra_rx=extra_rx, extra_ry=extra_ry, extra_angle=extra_angle,
        extra_gap_px=extra_gap_px,
    )


def _save_crop(img2d: np.ndarray, cx: float, cy: float, semi_a: float, path: Path, scale: float = 2.0):
    vis = np.clip(img2d / (np.percentile(img2d, 99.5) + 1e-9), 0, 1)
    vis_u8 = (vis * 255).astype(np.uint8)
    half = int(semi_a * 1.5)
    y0, y1 = max(0, int(cy - half)), int(cy + half)
    x0, x1 = max(0, int(cx - half)), int(cx + half)
    crop = vis_u8[y0:y1, x0:x1]
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(path), crop)


def run_one(target_name, step02_dir, step04_dir, windows_json, wi, filt, has_rings, true_polar_ratio, out_dir):
    data = json.load(open(windows_json))
    windows = {w["window_index"]: w for w in data["selected_windows"]}
    window = windows[wi]
    window_log = json.load(open(step04_dir / f"window_{wi:02d}/derotation_log.json"))
    hydrated = _hydrate_window(window, step02_dir)
    flog = window_log["filters"][filt]

    for tag, enabled in [("off", False), ("on", True)]:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_out = Path(tmp)
            results = derotate_window(
                hydrated, required_filters=[filt],
                period_hours=flog["period_hours"], warp_scale=flog["warp_scale"],
                align=flog["align_enabled"], normalize_brightness=flog["normalize_brightness"],
                min_quality_threshold=0.0, pole_pa_deg=flog["pole_pa_deg"],
                color_mode=False, flip_direction=flog["flip_direction"],
                weight_power=flog["weight_power"], has_rings=has_rings,
                sub_observer_lat_deg=flog.get("sub_observer_lat_deg", 0.0),
                use_true_reprojection=True, true_polar_equatorial_ratio=true_polar_ratio,
                out_dir=tmp_out,
                sharpness_selection_enabled=enabled, sharpness_keep_fraction=0.5,
            )
            out_path, log_dict = results[filt]
            stacked = image_io.read_tif(out_path)

        sharpened = _wavelet_sharpen(stacked, has_rings, flog["pole_pa_deg"], flog.get("sub_observer_lat_deg", 0.0))
        lum = sharpened if sharpened.ndim == 2 else sharpened.mean(axis=2).astype(np.float32)
        cx, cy, rx, ry, _ = find_disk_center(lum)
        out_png = out_dir / f"{target_name}_w{wi:02d}_{filt}_{tag}.png"
        _save_crop(lum, cx, cy, rx, out_png)
        n_excl = sum(1 for f in log_dict["frames"] if f.get("sharpness_excluded"))
        print(f"{target_name} w{wi:02d} {filt} [{tag}]: n_stacked={log_dict['n_stacked']} "
              f"excluded={n_excl} -> {out_png.name}")


def main():
    out_dir = Path("scratch_sharpness_selection_wavelet_crops")
    out_dir.mkdir(exist_ok=True)

    run_one("Saturn", Path("Saturn_Data/step02_lucky_stack"), Path("Saturn_Data/step04_derotated"),
            Path("Saturn_Data/step03_quality/windows.json"), 1, "R", True, 0.9021, out_dir)
    run_one("Saturn", Path("Saturn_Data/step02_lucky_stack"), Path("Saturn_Data/step04_derotated"),
            Path("Saturn_Data/step03_quality/windows.json"), 1, "IR", True, 0.9021, out_dir)
    run_one("Jupiter", Path("Jupiter_Data/step02_lucky_stack"), Path("Jupiter_Data/step04_derotated"),
            Path("Jupiter_Data/step03_quality/windows.json"), 3, "IR", False, 0.9, out_dir)
    run_one("Jupiter", Path("Jupiter_Data/step02_lucky_stack"), Path("Jupiter_Data/step04_derotated"),
            Path("Jupiter_Data/step03_quality/windows.json"), 3, "R", False, 0.9, out_dir)

    print(f"\nCrops written to {out_dir}/")


if __name__ == "__main__":
    main()
