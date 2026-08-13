"""Jupiter regression sweep for the new frame-to-frame rotation correction
(2026-08-14) -- A/B via monkeypatch (real production apply_shift_and_scale,
forced rotation_deg=0.0 for "before" vs real rotation for "after"), using
the SAME disk-sharpness metric and derotate_window()-based harness as
scratch_window_level_sweep.py (central 55% semi_a Laplacian variance ratio
vs the best single raw frame in the window).

Reports median, 10th percentile, and worst-case delta -- not just the
median, since a regression could hide in the tail even if the median holds.
"""
from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.modules import image_io
from pipeline.modules import derotation
from pipeline.modules.derotation import (
    derotate_window,
    find_disk_center,
    apply_shift_and_scale as _real_apply_shift_and_scale,
)

STEP02_DIR = Path("Jupiter_Data/step02_lucky_stack")
STEP04_DIR = Path("Jupiter_Data/step04_derotated")
WINDOWS_JSON = Path("Jupiter_Data/step03_quality/windows.json")

FILTERS = ["IR", "R", "G", "B", "CH4"]


def _force_no_rotation(image, target_cx, target_cy, ref_cx, ref_cy, scale, rotation_deg=0.0):
    return _real_apply_shift_and_scale(image, target_cx, target_cy, ref_cx, ref_cy, scale, rotation_deg=0.0)


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


def _run_sweep(rotation_enabled: bool, n_windows: int):
    data = json.load(open(WINDOWS_JSON))
    windows = {w["window_index"]: w for w in data["selected_windows"]}

    results = []
    all_rotations = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for wi in sorted(windows.keys())[:n_windows]:
            window = windows[wi]
            log_path = STEP04_DIR / f"window_{wi:02d}" / "derotation_log.json"
            if not log_path.exists():
                continue
            window_log = json.load(open(log_path))
            hydrated = _hydrate_window(window)
            any_flog = next(iter(window_log["filters"].values()))

            out_dir = Path(tmpdir) / f"window_{wi:02d}"
            out_dir.mkdir(parents=True, exist_ok=True)

            ctx = mock.patch.object(derotation, "apply_shift_and_scale", _force_no_rotation) \
                if not rotation_enabled else mock.MagicMock()
            if not rotation_enabled:
                with mock.patch.object(derotation, "apply_shift_and_scale", _force_no_rotation):
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
                        out_dir=out_dir,
                    )
            else:
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
                    out_dir=out_dir,
                )

            for filt, (out_path, log_dict) in window_results.items():
                if "error" in log_dict or out_path is None:
                    continue
                included_rows = hydrated["per_filter"][filt]["included"]
                if len(included_rows) < 2:
                    continue
                if rotation_enabled:
                    for f in log_dict.get("frames", []):
                        rot = f.get("pre_warp_rotation_deg")
                        if rot is not None:
                            all_rotations.append(rot)
                best_row = max(included_rows, key=lambda r: r["norm_score"])
                best_raw = image_io.read_tif(best_row["path"])
                best_lum = best_raw if best_raw.ndim == 2 else best_raw.mean(axis=2).astype(np.float32)
                try:
                    best_sharp = _sharpness_central55(best_lum)
                except RuntimeError:
                    continue

                stacked = image_io.read_tif(str(out_path))
                stacked_lum = stacked if stacked.ndim == 2 else stacked.mean(axis=2).astype(np.float32)
                try:
                    stack_sharp = _sharpness_central55(stacked_lum)
                except RuntimeError:
                    continue

                ratio = stack_sharp / best_sharp
                results.append({"window": wi, "filter": filt, "ratio": ratio})
    return results, all_rotations


def main():
    n_windows = int(sys.argv[1]) if len(sys.argv) > 1 else 28
    print(f"Running {n_windows} windows, rotation OFF (baseline)...")
    before, _ = _run_sweep(rotation_enabled=False, n_windows=n_windows)
    print(f"Running {n_windows} windows, rotation ON (new)...")
    after, rotations = _run_sweep(rotation_enabled=True, n_windows=n_windows)

    before_map = {(r["window"], r["filter"]): r["ratio"] for r in before}
    after_map = {(r["window"], r["filter"]): r["ratio"] for r in after}
    keys = sorted(set(before_map) & set(after_map))

    deltas = []
    print("\nwindow filter  before   after    delta")
    for k in keys:
        b, a = before_map[k], after_map[k]
        d = a - b
        deltas.append(d)
        print(f"{k[0]:>3}    {k[1]:<4}  {b:.4f}   {a:.4f}   {d:+.4f}")

    before_ratios = np.array([before_map[k] for k in keys])
    after_ratios = np.array([after_map[k] for k in keys])
    deltas = np.array(deltas)

    print(f"\n=== Summary (n={len(keys)}) ===")
    print(f"before: median={np.median(before_ratios):.4f} p10={np.percentile(before_ratios,10):.4f}")
    print(f"after:  median={np.median(after_ratios):.4f} p10={np.percentile(after_ratios,10):.4f}")
    print(f"delta:  median={np.median(deltas):+.4f} worst={deltas.min():+.4f} best={deltas.max():+.4f}")

    rotations = np.array(rotations)
    nonzero = rotations[rotations != 0.0]
    print(f"\n=== Rotation distribution (n={len(rotations)} frames) ===")
    print(f"applied (nonzero): {len(nonzero)}/{len(rotations)} ({100*len(nonzero)/max(len(rotations),1):.0f}%)")
    if len(nonzero):
        print(f"|rotation| deg: mean={np.abs(nonzero).mean():.3f} median={np.median(np.abs(nonzero)):.3f} "
              f"max={np.abs(nonzero).max():.3f}")


if __name__ == "__main__":
    main()
