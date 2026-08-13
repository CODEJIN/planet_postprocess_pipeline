"""Jupiter equivalent of scratch_window_level_sweep.py (the Saturn 45-combo
step05-vs-step07 sharpness sweep, see project_saturn_step05_sharpness_gap
memory: median ratio ended at ~0.862 after 3 real bug fixes, with IR/CH4
causes still unexplained).

User question (2026-08-13): "why doesn't Jupiter show the same outer-edge
blur if it uses the identical algorithm?" -- this has never actually been
measured for Jupiter with the same rigorous methodology; the Saturn number
came from derotate_window() (not derotate_filter() directly, which silently
skips shared_shape/shared_radius_px/filter_pose threading -- see
project_shared_shape_gating_bug memory). This script replicates that exact
harness for Jupiter_Data, passing use_true_reprojection=True/
true_polar_equatorial_ratio to match this user's ACTUAL saved profile
settings (~/.astropipe/profiles/mono.json has use_true_reprojection: true,
and Jupiter_Data's own existing derotation_log.json already shows a
nonzero sub_observer_lat_deg confirming that flag was on when this data was
originally produced).
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

from pipeline.modules import image_io
from pipeline.modules.derotation import derotate_window, find_disk_center

STEP02_DIR = Path("Jupiter_Data/step02_lucky_stack")
STEP04_DIR = Path("Jupiter_Data/step04_derotated")
WINDOWS_JSON = Path("Jupiter_Data/step03_quality/windows.json")

FILTERS = ["IR", "R", "G", "B", "CH4"]

# 2026-08-13: toggle for the raw-sharpness-based frame selection feature
# (see frame_sharpness_central() in pipeline/modules/derotation.py). Set
# via env var so this script can be re-run both ways without editing code:
#   SHARPNESS_SELECTION=1 python3 scratch_jupiter_step05_sharpness_sweep.py
import os
SHARPNESS_SELECTION_ENABLED = os.environ.get("SHARPNESS_SELECTION", "0") == "1"
SHARPNESS_KEEP_FRACTION = float(os.environ.get("SHARPNESS_KEEP_FRACTION", "0.5"))


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


def _hydrate_window(window: dict) -> dict:
    hydrated = {"center_time": _parse_ts(window["center_time"]), "per_filter": {}}
    for filt, pf in window["per_filter"].items():
        included = []
        for item in pf["included"]:
            matches = list(STEP02_DIR.glob(f"{item['stem']}.tif"))
            if not matches:
                raise FileNotFoundError(f"no step02 tif for stem {item['stem']}")
            included.append({
                "path": str(matches[0]),
                "stem": item["stem"],
                "timestamp": _parse_ts(item["timestamp"]),
                "norm_score": item["norm_score"],
            })
        hydrated["per_filter"][filt] = {"included": included}
    return hydrated


def main():
    print(f"sharpness_selection_enabled={SHARPNESS_SELECTION_ENABLED} "
          f"sharpness_keep_fraction={SHARPNESS_KEEP_FRACTION}")
    data = json.load(open(WINDOWS_JSON))
    windows = {w["window_index"]: w for w in data["selected_windows"]}

    results = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for wi in sorted(windows.keys())[:9]:
            window = windows[wi]
            log_path = STEP04_DIR / f"window_{wi:02d}" / "derotation_log.json"
            if not log_path.exists():
                continue
            window_log = json.load(open(log_path))
            hydrated = _hydrate_window(window)
            any_flog = next(iter(window_log["filters"].values()))

            out_dir = Path(tmpdir) / f"window_{wi:02d}"
            out_dir.mkdir(parents=True, exist_ok=True)

            window_results = derotate_window(
                hydrated,
                required_filters=[f for f in FILTERS if f in window["per_filter"]],
                period_hours=any_flog["period_hours"],
                warp_scale=any_flog["warp_scale"],
                align=any_flog["align_enabled"],
                normalize_brightness=any_flog["normalize_brightness"],
                min_quality_threshold=any_flog["min_quality_threshold"],
                pole_pa_deg=any_flog["pole_pa_deg"],
                color_mode=False,
                flip_direction=any_flog["flip_direction"],
                weight_power=any_flog["weight_power"],
                has_rings=any_flog.get("has_rings", False),
                sub_observer_lat_deg=any_flog.get("sub_observer_lat_deg", 0.0),
                use_true_reprojection=True,
                true_polar_equatorial_ratio=0.9,  # Jupiter's true Rpol/Req
                out_dir=out_dir,
                sharpness_selection_enabled=SHARPNESS_SELECTION_ENABLED,
                sharpness_keep_fraction=SHARPNESS_KEEP_FRACTION,
            )

            for filt, (out_path, log_dict) in window_results.items():
                if "error" in log_dict:
                    print(f"window_{wi:02d} {filt}: ERROR {log_dict['error']}")
                    continue
                if out_path is None:
                    print(f"window_{wi:02d} {filt}: SKIP no output written")
                    continue
                included_rows = hydrated["per_filter"][filt]["included"]
                if len(included_rows) < 2:
                    continue
                best_row = max(included_rows, key=lambda r: r["norm_score"])
                best_raw = image_io.read_tif(best_row["path"])
                best_lum = best_raw if best_raw.ndim == 2 else best_raw.mean(axis=2).astype(np.float32)
                try:
                    best_sharp = _sharpness_central55(best_lum)
                except RuntimeError as e:
                    print(f"window_{wi:02d} {filt}: SKIP best-frame disk detect failed: {e}")
                    continue

                stacked = image_io.read_tif(str(out_path))
                stacked_lum = stacked if stacked.ndim == 2 else stacked.mean(axis=2).astype(np.float32)
                try:
                    stack_sharp = _sharpness_central55(stacked_lum)
                except RuntimeError as e:
                    print(f"window_{wi:02d} {filt}: SKIP stack disk detect failed: {e}")
                    continue

                ratio = stack_sharp / best_sharp
                results.append({
                    "window": wi, "filter": filt,
                    "n_stacked": log_dict.get("n_stacked"),
                    "geometry_source": log_dict.get("geometry_source"),
                    "stack_sharpness": stack_sharp,
                    "best_single_sharpness": best_sharp,
                    "ratio_vs_best_single": ratio,
                })
                print(f"window_{wi:02d} {filt:>4}: n={log_dict.get('n_stacked')} "
                      f"geom={log_dict.get('geometry_source')} ratio={ratio:.4f}")

    out_path = Path("scratch_jupiter_step05_sharpness_sweep_results.json")
    out_path.write_text(json.dumps(results, indent=2))

    ratios = [r["ratio_vs_best_single"] for r in results]
    worse = sum(1 for r in ratios if r < 1.0)
    print(f"\n=== Summary: n={len(ratios)} median={np.median(ratios):.4f} "
          f"mean={np.mean(ratios):.4f} worse={worse}/{len(ratios)} ===")

    print("\n=== By filter ===")
    for f in FILTERS:
        fr = [r["ratio_vs_best_single"] for r in results if r["filter"] == f]
        if fr:
            print(f"  {f:>4}: n={len(fr)} median={np.median(fr):.4f} mean={np.mean(fr):.4f}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
