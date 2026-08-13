"""Direct A/B of Saturn warp_scale on REAL final stacked sharpness (not NCC),
via the real derotate_window() production path (same validated methodology as
scratch_window_level_sweep.py -- shared_shape/shared_radius_px/filter_pose all
threaded through correctly). This follows up the 2026-08-11 finding that the
NCC sweep used to justify warp_scale=0.10 is noise-dominated / self-referential
(no genuine interior peak, adversarially verified) -- NCC can't be trusted to
pick a scale, so compare actual stack-vs-single-frame sharpness directly for
each candidate scale, reusing the exact metric (_sharpness_central55) used for
the whole "step05 >= step07" acceptance bar this session.
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
SCALES_TO_TEST = [0.0, 0.05, 0.10, 0.25, 0.5, 1.0]


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


def run_for_scale(scale_override):
    data = json.load(open(WINDOWS_JSON))
    windows = {w["window_index"]: w for w in data["selected_windows"]}

    # Cache best-single-frame sharpness across scale runs (scale-independent).
    global _BEST_CACHE
    results = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for wi in range(1, 10):
            window = windows[wi]
            window_log = json.load(open(STEP04_DIR / f"window_{wi:02d}" / "derotation_log.json"))
            hydrated = _hydrate_window(window)
            any_flog = next(iter(window_log["filters"].values()))

            out_dir = Path(tmpdir) / f"window_{wi:02d}"
            out_dir.mkdir(parents=True, exist_ok=True)

            used_scale = any_flog["warp_scale"] if scale_override is None else scale_override

            window_results = derotate_window(
                hydrated,
                required_filters=[f for f in FILTERS if f in window["per_filter"]],
                period_hours=any_flog["period_hours"],
                warp_scale=used_scale,
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
                if "error" in log_dict or out_path is None:
                    continue
                included_rows = hydrated["per_filter"][filt]["included"]
                if len(included_rows) < 2:
                    continue

                cache_key = (wi, filt)
                if cache_key not in _BEST_CACHE:
                    best_row = max(included_rows, key=lambda r: r["norm_score"])
                    best_raw = image_io.read_tif(best_row["path"])
                    best_lum = best_raw if best_raw.ndim == 2 else best_raw.mean(axis=2).astype(np.float32)
                    try:
                        _BEST_CACHE[cache_key] = _sharpness_central55(best_lum)
                    except RuntimeError:
                        _BEST_CACHE[cache_key] = None
                best_sharp = _BEST_CACHE[cache_key]
                if best_sharp is None:
                    continue

                stacked = image_io.read_tif(str(out_path))
                stacked_lum = stacked if stacked.ndim == 2 else stacked.mean(axis=2).astype(np.float32)
                try:
                    stack_sharp = _sharpness_central55(stacked_lum)
                except RuntimeError:
                    continue

                results.append({
                    "window": wi, "filter": filt, "scale": used_scale,
                    "n_stacked": log_dict.get("n_stacked"),
                    "stack_sharpness": stack_sharp,
                    "best_single_sharpness": best_sharp,
                    "ratio_vs_best_single": stack_sharp / best_sharp,
                })
    return results


_BEST_CACHE: dict = {}


def main():
    all_results = {}
    for scale in SCALES_TO_TEST:
        print(f"\n=== scale={scale} ===")
        res = run_for_scale(scale)
        ratios = [r["ratio_vs_best_single"] for r in res]
        worse = sum(1 for r in ratios if r < 1.0)
        print(f"n={len(ratios)} median={np.median(ratios):.4f} mean={np.mean(ratios):.4f} worse={worse}/{len(ratios)}")
        all_results[str(scale)] = res

    out_path = Path("scratch_warp_scale_final_ab_results.json")
    out_path.write_text(json.dumps(all_results, indent=2))

    print("\n=== Summary table ===")
    print(f"{'scale':>8} {'median':>8} {'mean':>8} {'worse':>10}")
    for scale in SCALES_TO_TEST:
        ratios = [r["ratio_vs_best_single"] for r in all_results[str(scale)]]
        worse = sum(1 for r in ratios if r < 1.0)
        print(f"{scale:>8} {np.median(ratios):>8.4f} {np.mean(ratios):>8.4f} {worse:>6}/{len(ratios)}")

    print("\n=== Per-filter median by scale ===")
    for filt in FILTERS:
        row = []
        for scale in SCALES_TO_TEST:
            vals = [r["ratio_vs_best_single"] for r in all_results[str(scale)] if r["filter"] == filt]
            row.append(f"{np.median(vals):.3f}" if vals else "  -  ")
        print(f"{filt:>4}: " + "  ".join(row))

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
