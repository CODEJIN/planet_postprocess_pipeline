"""Corrected 45-combo sharpness sweep (2026-08-11): calls the REAL
derotate_window() (not derotate_filter() directly per-filter) so that
shared_shape/shared_radius_px/filter_pose are computed and threaded through
exactly as production does (see derotate_window()'s "Ring-aware shared
shape/pose" block). scratch_shared_shape_fix_sweep.py called derotate_filter()
directly without ever passing shared_shape, so it silently bypassed BOTH the
shared_shape gating fix (2dc3773) and the aspect-ratio-disagreement fix
(this commit) -- its "0.865, unchanged before/after" result was invalid,
not a real verification. This script is the corrected replacement.
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
    data = json.load(open(WINDOWS_JSON))
    windows = {w["window_index"]: w for w in data["selected_windows"]}

    results = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for wi in range(1, 10):
            window = windows[wi]
            window_log = json.load(open(STEP04_DIR / f"window_{wi:02d}" / "derotation_log.json"))
            hydrated = _hydrate_window(window)

            # All filters in a window share these (confirmed identical across
            # filters in every window's real derotation_log.json).
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
                has_rings=any_flog["has_rings"],
                sub_observer_lat_deg=any_flog["sub_observer_lat_deg"],
                out_dir=out_dir,
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
                    "ring_crosses_disk": log_dict.get("ring_crosses_disk"),
                    "stack_sharpness": stack_sharp,
                    "best_single_sharpness": best_sharp,
                    "ratio_vs_best_single": ratio,
                })
                print(f"window_{wi:02d} {filt:>4}: n={log_dict.get('n_stacked')} "
                      f"geom={log_dict.get('geometry_source')} ratio={ratio:.4f}")

    out_path = Path("scratch_window_level_sweep_results.json")
    out_path.write_text(json.dumps(results, indent=2))

    ratios = [r["ratio_vs_best_single"] for r in results]
    worse = sum(1 for r in ratios if r < 1.0)
    print(f"\n=== Summary: n={len(ratios)} median={np.median(ratios):.4f} "
          f"mean={np.mean(ratios):.4f} worse={worse}/{len(ratios)} ===")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
