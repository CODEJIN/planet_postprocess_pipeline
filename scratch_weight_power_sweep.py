"""One-off investigation script: test whether raising quality_weighted_stack's
weight_power recovers step05's sharpness deficit vs step07 (see task brief).

Uses REAL derotate_filter() (imported, not reimplemented) with the exact
params recorded in each window's real derotation_log.json. Not part of the
production pipeline -- exploratory/diagnostic only.
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
WINDOWS_JSON = Path("Saturn_Data/step03_quality/windows.json")

# Params confirmed identical across window_01/05/09 derotation_log.json
PERIOD_HOURS = 10.56
WARP_SCALE = 0.10
POLE_PA_DEG = -7.0
FLIP_DIRECTION = False
MIN_QUALITY_THRESHOLD = 0.05
NORMALIZE_BRIGHTNESS = True

WEIGHT_POWERS = [1.0, 2.0, 4.0, 8.0, 16.0]

TARGETS = [(1, "IR"), (1, "G"), (5, "IR"), (5, "G"), (9, "IR"), (9, "G")]


def _parse_ts(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")


def _sharpness_central55(img2d: np.ndarray) -> float:
    """Laplacian variance inside central 55%-of-radius disk region."""
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
    for wi, filt in TARGETS:
        window = windows[wi]
        included_rows = _build_included_rows(window, filt)
        t_reference = _parse_ts(window["center_time"])

        # Single best (highest norm_score) raw frame's own sharpness.
        best_row = max(included_rows, key=lambda r: r["norm_score"])
        best_raw = image_io.read_tif(best_row["path"])
        best_lum = best_raw if best_raw.ndim == 2 else best_raw.mean(axis=2).astype(np.float32)
        best_sharp = _sharpness_central55(best_lum)

        print(f"\n=== window_{wi:02d} {filt}: n={len(included_rows)} "
              f"scores={[round(r['norm_score'],3) for r in included_rows]} "
              f"best={best_row['stem']} best_sharp={best_sharp:.6e} ===")

        for wp in WEIGHT_POWERS:
            stacked, log_dict = derotate_filter(
                included_rows,
                t_reference,
                period_hours=PERIOD_HOURS,
                warp_scale=WARP_SCALE,
                align=True,
                normalize_brightness=NORMALIZE_BRIGHTNESS,
                min_quality_threshold=MIN_QUALITY_THRESHOLD,
                pole_pa_deg=POLE_PA_DEG,
                color_mode=False,
                flip_direction=FLIP_DIRECTION,
                weight_power=wp,
            )
            stack_sharp = _sharpness_central55(stacked)
            ratio = stack_sharp / best_sharp
            results.append({
                "window": wi, "filter": filt, "weight_power": wp,
                "n_stacked": log_dict["n_stacked"],
                "stack_sharpness": stack_sharp,
                "best_single_sharpness": best_sharp,
                "ratio_vs_best_single": ratio,
            })
            print(f"  weight_power={wp:5.1f}  n_stacked={log_dict['n_stacked']}  "
                  f"stack_sharp={stack_sharp:.6e}  ratio_vs_best_single={ratio:.4f}")

    out_path = Path("scratch_weight_power_sweep_results.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")

    print("\n=== Summary table ===")
    print(f"{'window':>6} {'filter':>6} {'weight_power':>12} {'stack_sharp':>14} {'ratio':>8}")
    for r in results:
        print(f"{r['window']:>6} {r['filter']:>6} {r['weight_power']:>12.1f} "
              f"{r['stack_sharpness']:>14.6e} {r['ratio_vs_best_single']:>8.4f}")


if __name__ == "__main__":
    main()
