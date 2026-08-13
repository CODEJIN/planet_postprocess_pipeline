"""Verify the shared_shape-gating fix (2026-08-11): derotate_filter()'s
pre-warp alignment used to gate on the WINDOW-level `shared_shape is not
None` flag (true whenever ANY filter in the window -- typically CH4 --
failed its own shape detection), silently downgrading every OTHER filter
(IR/R/G/B, whose own find_disk_center() fit is reliable) from the precise
find_disk_center()-based pre-warp alignment to the less precise
subpixel_align()-based correlation fallback. Fixed to gate on this filter's
OWN _rshape_ok instead.

Reuses the exact same proxy methodology as scratch_weight_power_sweep.py
(pre-wavelet raw-TIF Laplacian variance, stack vs best single raw frame,
central 55%-of-radius disk region) across all 45 real window x filter
combos, with has_rings=True and each window's actual logged
sub_observer_lat_deg/pole_pa_deg -- i.e. reproducing the exact production
config already baked into Saturn_Data/step04_derotated logs, but through
the now-fixed derotate_filter().
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.modules import image_io
from pipeline.modules.derotation import derotate_filter, find_disk_center

STEP02_DIR = Path("Saturn_Data/step02_lucky_stack")
STEP04_DIR = Path("Saturn_Data/step04_derotated")
WINDOWS_JSON = Path("Saturn_Data/step03_quality/windows.json")

FILTERS = ["IR", "R", "G", "B", "CH4"]


def _parse_ts(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")


def _sharpness_central55(img2d: np.ndarray) -> float:
    cx, cy, semi_a, semi_b, _ = find_disk_center(img2d)
    if semi_a < 5:
        raise RuntimeError("disk detection failed")
    h, w = img2d.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = r < (0.55 * semi_a)
    lap = cv2.Laplacian(img2d.astype(np.float32), cv2.CV_32F, ksize=3)
    return float(np.var(lap[mask]))


def _build_included_rows(window: dict, filt: str) -> list[dict]:
    rows = []
    for item in window["per_filter"][filt]["included"]:
        stem = item["stem"]
        matches = list(STEP02_DIR.glob(f"{stem}.tif"))
        if not matches:
            raise FileNotFoundError(f"no step02 tif for stem {stem}")
        rows.append({
            "path": str(matches[0]),
            "stem": stem,
            "timestamp": _parse_ts(item["timestamp"]),
            "norm_score": item["norm_score"],
        })
    return rows


def main():
    data = json.load(open(WINDOWS_JSON))
    windows = {w["window_index"]: w for w in data["selected_windows"]}

    results = []
    for wi in range(1, 10):
        window = windows[wi]
        window_log = json.load(open(STEP04_DIR / f"window_{wi:02d}" / "derotation_log.json"))
        for filt in FILTERS:
            if filt not in window["per_filter"]:
                continue
            flog = window_log["filters"][filt]
            included_rows = _build_included_rows(window, filt)
            if len(included_rows) < 2:
                continue
            t_reference = _parse_ts(window["center_time"])

            best_row = max(included_rows, key=lambda r: r["norm_score"])
            best_raw = image_io.read_tif(best_row["path"])
            best_lum = best_raw if best_raw.ndim == 2 else best_raw.mean(axis=2).astype(np.float32)
            try:
                best_sharp = _sharpness_central55(best_lum)
            except RuntimeError as e:
                print(f"window_{wi:02d} {filt}: SKIP best-frame disk detect failed: {e}")
                continue

            try:
                stacked, log_dict = derotate_filter(
                    included_rows,
                    t_reference,
                    period_hours=flog["period_hours"],
                    warp_scale=flog["warp_scale"],
                    align=True,
                    normalize_brightness=flog["normalize_brightness"],
                    min_quality_threshold=flog["min_quality_threshold"],
                    pole_pa_deg=flog["pole_pa_deg"],
                    color_mode=False,
                    flip_direction=flog["flip_direction"],
                    weight_power=flog["weight_power"],
                    has_rings=flog["has_rings"],
                    sub_observer_lat_deg=flog["sub_observer_lat_deg"],
                )
            except Exception as e:
                print(f"window_{wi:02d} {filt}: ERROR derotate_filter: {e}")
                continue

            try:
                stack_sharp = _sharpness_central55(stacked)
            except RuntimeError as e:
                print(f"window_{wi:02d} {filt}: SKIP stack disk detect failed: {e}")
                continue

            ratio = stack_sharp / best_sharp
            results.append({
                "window": wi, "filter": filt,
                "n_stacked": log_dict["n_stacked"],
                "ring_crosses_disk": log_dict.get("ring_crosses_disk"),
                "stack_sharpness": stack_sharp,
                "best_single_sharpness": best_sharp,
                "ratio_vs_best_single": ratio,
            })
            print(f"window_{wi:02d} {filt:>4}: n={log_dict['n_stacked']} ratio={ratio:.4f}")

    out_path = Path("scratch_shared_shape_fix_sweep_results.json")
    out_path.write_text(json.dumps(results, indent=2))

    ratios = [r["ratio_vs_best_single"] for r in results]
    worse = sum(1 for r in ratios if r < 1.0)
    print(f"\n=== Summary: n={len(ratios)} median={np.median(ratios):.4f} "
          f"mean={np.mean(ratios):.4f} worse={worse}/{len(ratios)} ===")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
