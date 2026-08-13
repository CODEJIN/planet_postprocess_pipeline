"""Control test for scratch_raw_sharpness_selection_test.py's finding:
does excluding low-sharpness frames help because of WHICH frames are
excluded (content quality), or merely because FEWER frames are averaged
(trivial denominator effect)? Compares full-stack vs top-K-sharpest vs
bottom-K-sharpest (same K, i.e. same frame COUNT) for a few window/filter
combos across both targets.
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

TARGETS = {
    "Saturn": dict(
        step02=Path("Saturn_Data/step02_lucky_stack"),
        step04=Path("Saturn_Data/step04_derotated"),
        windows_json=Path("Saturn_Data/step03_quality/windows.json"),
        true_polar_ratio=0.9021,
    ),
    "Jupiter": dict(
        step02=Path("Jupiter_Data/step02_lucky_stack"),
        step04=Path("Jupiter_Data/step04_derotated"),
        windows_json=Path("Jupiter_Data/step03_quality/windows.json"),
        true_polar_ratio=0.9,
    ),
}


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


def _hydrate_window(window: dict, step02_dir: Path) -> dict:
    hydrated = {"center_time": _parse_ts(window["center_time"]), "per_filter": {}}
    for filt, pf in window["per_filter"].items():
        included = []
        for item in pf["included"]:
            matches = list(step02_dir.glob(f"{item['stem']}.tif"))
            if not matches:
                continue
            included.append({
                "path": str(matches[0]),
                "stem": item["stem"],
                "timestamp": _parse_ts(item["timestamp"]),
                "norm_score": item["norm_score"],
            })
        hydrated["per_filter"][filt] = {"included": included}
    return hydrated


def _run_stack(hydrated, filt, flog, included_rows, true_polar_ratio, out_dir):
    sub_hydrated = {"center_time": hydrated["center_time"], "per_filter": {filt: {"included": included_rows}}}
    results = derotate_window(
        sub_hydrated, required_filters=[filt],
        period_hours=flog["period_hours"], warp_scale=flog["warp_scale"],
        align=flog["align_enabled"], normalize_brightness=flog["normalize_brightness"],
        min_quality_threshold=0.0, pole_pa_deg=flog["pole_pa_deg"],
        color_mode=False, flip_direction=flog["flip_direction"],
        weight_power=flog["weight_power"], has_rings=flog.get("has_rings", False),
        sub_observer_lat_deg=flog.get("sub_observer_lat_deg", 0.0),
        use_true_reprojection=True, true_polar_equatorial_ratio=true_polar_ratio,
        out_dir=out_dir,
    )
    return results[filt]


def main():
    targets_to_test = [
        ("Saturn", 2, "IR"), ("Saturn", 4, "CH4"), ("Saturn", 6, "G"),
        ("Jupiter", 2, "R"), ("Jupiter", 3, "IR"), ("Jupiter", 4, "B"),
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        for target_name, wi, filt in targets_to_test:
            cfg = TARGETS[target_name]
            data = json.load(open(cfg["windows_json"]))
            windows = {w["window_index"]: w for w in data["selected_windows"]}
            window = windows[wi]
            window_log = json.load(open(cfg["step04"] / f"window_{wi:02d}/derotation_log.json"))
            hydrated = _hydrate_window(window, cfg["step02"])
            flog = window_log["filters"][filt]
            rows = hydrated["per_filter"][filt]["included"]

            sharpness = {}
            for row in rows:
                raw = image_io.read_tif(row["path"])
                lum = raw if raw.ndim == 2 else raw.mean(axis=2).astype(np.float32)
                sharpness[row["stem"]] = _sharpness_central55(lum)
            best_sharp = max(sharpness.values())
            sorted_stems = sorted(sharpness, key=sharpness.get, reverse=True)
            k = max(2, len(sorted_stems) // 2)
            top_stems = set(sorted_stems[:k])
            bottom_stems = set(sorted_stems[-k:])
            top_rows = [r for r in rows if r["stem"] in top_stems]
            bottom_rows = [r for r in rows if r["stem"] in bottom_stems]

            def ratio_for(subset_rows, tag):
                out_dir = Path(tmpdir) / f"{target_name}_{wi}_{filt}_{tag}"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path, log_dict = _run_stack(hydrated, filt, flog, subset_rows, cfg["true_polar_ratio"], out_dir)
                stacked = image_io.read_tif(str(out_path))
                slum = stacked if stacked.ndim == 2 else stacked.mean(axis=2).astype(np.float32)
                return _sharpness_central55(slum) / best_sharp

            r_full = ratio_for(rows, "full")
            r_top = ratio_for(top_rows, "top")
            r_bottom = ratio_for(bottom_rows, "bottom")
            print(f"{target_name} w{wi:02d} {filt}: n_all={len(rows)} k={k} | "
                  f"full={r_full:.4f}  top{k}={r_top:.4f}  bottom{k}={r_bottom:.4f}")


if __name__ == "__main__":
    main()
