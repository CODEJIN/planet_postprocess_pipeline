"""New hypothesis test (2026-08-13, continuing the interior-disk sharpness-
gap investigation after the Jupiter cross-check showed the ~0.85 median
ratio is NOT edge/coverage-specific -- it's present deep in the disk
interior (r<0.55*semi_a) where every frame has 100% valid data).

Earlier this session (scratch_per_frame_sharpness_vs_norm_score.py, only
ever run on window_01/R, n=3 frames) found real per-frame PSF/sharpness
variation (~4x spread in Laplacian variance) but concluded norm_score
doesn't predict which frame is sharper (corr ~ -0.10) -- that hypothesis
was framed narrowly as "can norm_score-based weight_power fix this" (no).
It did NOT test the more direct question: does selecting/excluding frames
by their OWN RAW measured sharpness (not norm_score) close the gap?

This is the textbook explanation for "any multi-frame average is softer
than its single sharpest member, everywhere, regardless of coordinate
system, target, or coverage": if frames have genuinely different
intrinsic PSF/sharpness (seeing varied during capture), a plain (or
norm_score-weighted, since norm_score != sharpness) average blends toward
something between the sharpest and blurriest member -- not the sharpest.

Test: for many window/filter combos (both Saturn and Jupiter, for a cross-
target check consistent with this session's other finding), compute:
  1. Each raw included frame's own sharpness (Laplacian var, central 55%
     of ITS OWN disk fit -- not the stack's).
  2. within-window sharpness spread (max/min ratio among included frames).
  3. current production stack's ratio vs single best frame (established
     methodology, unchanged).
  4. an alternative stack using ONLY the top-half sharpest frames (by raw
     measured sharpness, not norm_score) -- same derotate_window() call,
     just a pre-filtered included_rows list.
  5. correlation between (2) and (3): if PSF heterogeneity is the driver,
     windows with WIDER sharpness spread should show WORSE (lower)
     current-stack ratios, and the top-half-only stack should recover
     more of the loss than the norm_score-weighted full stack.
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
    all_results = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for target_name, cfg in TARGETS.items():
            data = json.load(open(cfg["windows_json"]))
            windows = {w["window_index"]: w for w in data["selected_windows"]}
            for wi in sorted(windows.keys())[:6]:
                window = windows[wi]
                log_path = cfg["step04"] / f"window_{wi:02d}" / "derotation_log.json"
                if not log_path.exists():
                    continue
                window_log = json.load(open(log_path))
                hydrated = _hydrate_window(window, cfg["step02"])

                for filt in FILTERS:
                    if filt not in window_log["filters"]:
                        continue
                    flog = window_log["filters"][filt]
                    rows = hydrated["per_filter"].get(filt, {}).get("included", [])
                    if len(rows) < 4:
                        continue  # need enough frames to split top-half meaningfully

                    # 1. raw per-frame sharpness (own disk geometry, pre-stack)
                    sharpness = {}
                    for row in rows:
                        try:
                            raw = image_io.read_tif(row["path"])
                            lum = raw if raw.ndim == 2 else raw.mean(axis=2).astype(np.float32)
                            sharpness[row["stem"]] = _sharpness_central55(lum)
                        except RuntimeError:
                            continue
                    if len(sharpness) < 4:
                        continue

                    vals = list(sharpness.values())
                    spread = max(vals) / max(min(vals), 1e-12)
                    best_stem = max(sharpness, key=sharpness.get)
                    best_sharp = sharpness[best_stem]

                    # 2. current production stack (all included frames)
                    out_full = Path(tmpdir) / f"{target_name}_{wi}_{filt}_full"
                    out_full.mkdir(parents=True, exist_ok=True)
                    try:
                        out_path, log_dict = _run_stack(hydrated, filt, flog, rows, cfg["true_polar_ratio"], out_full)
                        if out_path is None or "error" in log_dict:
                            continue
                        stacked = image_io.read_tif(str(out_path))
                        stacked_lum = stacked if stacked.ndim == 2 else stacked.mean(axis=2).astype(np.float32)
                        full_sharp = _sharpness_central55(stacked_lum)
                    except (RuntimeError, Exception) as e:
                        continue
                    full_ratio = full_sharp / best_sharp

                    # 3. top-half-by-raw-sharpness-only stack
                    sorted_stems = sorted(sharpness, key=sharpness.get, reverse=True)
                    top_half_stems = set(sorted_stems[:max(2, len(sorted_stems) // 2)])
                    top_rows = [r for r in rows if r["stem"] in top_half_stems]
                    out_top = Path(tmpdir) / f"{target_name}_{wi}_{filt}_top"
                    out_top.mkdir(parents=True, exist_ok=True)
                    try:
                        out_path2, log_dict2 = _run_stack(hydrated, filt, flog, top_rows, cfg["true_polar_ratio"], out_top)
                        if out_path2 is None or "error" in log_dict2:
                            continue
                        stacked2 = image_io.read_tif(str(out_path2))
                        stacked2_lum = stacked2 if stacked2.ndim == 2 else stacked2.mean(axis=2).astype(np.float32)
                        top_sharp = _sharpness_central55(stacked2_lum)
                    except (RuntimeError, Exception) as e:
                        continue
                    top_ratio = top_sharp / best_sharp

                    all_results.append({
                        "target": target_name, "window": wi, "filter": filt,
                        "n_all": len(rows), "n_top_half": len(top_rows),
                        "sharpness_spread": spread,
                        "full_stack_ratio": full_ratio,
                        "top_half_stack_ratio": top_ratio,
                    })
                    print(f"{target_name} w{wi:02d} {filt:>4}: n={len(rows)}->  spread={spread:.2f}x  "
                          f"full_ratio={full_ratio:.4f}  top_half_ratio={top_ratio:.4f}  "
                          f"delta={top_ratio-full_ratio:+.4f}")

    Path("scratch_raw_sharpness_selection_test_results.json").write_text(json.dumps(all_results, indent=2))

    if all_results:
        full = np.array([r["full_stack_ratio"] for r in all_results])
        top = np.array([r["top_half_stack_ratio"] for r in all_results])
        spread = np.array([r["sharpness_spread"] for r in all_results])
        print(f"\n=== n={len(all_results)} ===")
        print(f"full_stack_ratio:     median={np.median(full):.4f} mean={np.mean(full):.4f}")
        print(f"top_half_stack_ratio: median={np.median(top):.4f} mean={np.mean(top):.4f}")
        print(f"improvement (top_half - full): median={np.median(top-full):+.4f} mean={np.mean(top-full):+.4f}")
        print(f"correlation(spread, full_ratio): {np.corrcoef(spread, full)[0,1]:.3f}")
        print(f"correlation(spread, improvement): {np.corrcoef(spread, top-full)[0,1]:.3f}")


if __name__ == "__main__":
    main()
