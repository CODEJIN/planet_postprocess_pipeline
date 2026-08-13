"""
Per-frame shape-detection reliability check for IR vs R, across all 9 windows,
for BOTH reference and non-reference frames.

For every included frame (per Saturn_Data/step03_quality/windows.json), load the
raw TIF, compute luminance via the real pipeline._to_luminance, and call the real
pipeline._find_disk_center_impl directly to get confidence/shape_reliable/semi_a/semi_b.

Ground rules followed:
- Uses pipeline.modules.derotation._find_disk_center_impl (real production function),
  no hand-reimplementation.
- Uses real included-frame lists from windows.json (per_filter[filt]['included']).
- Raw TIFs read via pipeline.modules.image_io.read_tif.
- Reports only real measured numbers; records errors per-frame rather than guessing.
"""
import sys
sys.path.insert(0, '.')
import json
import glob
import numpy as np

from pipeline.modules import derotation as dr
from pipeline.modules import image_io

QUALITY_JSON = "Saturn_Data/step03_quality/windows.json"
RAW_DIR = "Saturn_Data/step02_lucky_stack"


def find_path(stem):
    cands = glob.glob(f"{RAW_DIR}/{stem}*.tif")
    return cands[0] if cands else None


def main():
    windows = json.load(open(QUALITY_JSON))["selected_windows"]

    per_window_filter = {}  # (win_idx, filt) -> list of frame records
    errors = []

    for w in windows:
        win_idx = w["window_index"]
        for filt in ("IR", "R"):
            pf = w["per_filter"].get(filt)
            if pf is None:
                continue
            included = pf.get("included", [])
            recs = []
            for entry in included:
                stem = entry["stem"]
                path = find_path(stem)
                if path is None:
                    errors.append({"window": win_idx, "filter": filt, "stem": stem,
                                    "error": "raw tif not found"})
                    continue
                try:
                    raw = image_io.read_tif(path)
                    lum = dr._to_luminance(raw)
                    cx, cy, semi_a, semi_b, angle, confidence, shape_reliable = (
                        dr._find_disk_center_impl(lum)
                    )
                    aspect = float(semi_b) / float(semi_a) if semi_a > 0 else float("nan")
                    recs.append({
                        "stem": stem,
                        "rank": entry.get("rank"),
                        "cx": float(cx), "cy": float(cy),
                        "semi_a": float(semi_a), "semi_b": float(semi_b),
                        "angle": float(angle),
                        "confidence": float(confidence),
                        "shape_reliable": bool(shape_reliable),
                        "aspect_ratio": aspect,
                    })
                except Exception as e:
                    errors.append({"window": win_idx, "filter": filt, "stem": stem,
                                    "error": repr(e)})
            per_window_filter[f"window_{win_idx:02d}_{filt}"] = recs

    # Aggregate per window x filter
    summary = []
    for key, recs in per_window_filter.items():
        win_idx_str, filt = key.rsplit("_", 1)
        win_idx = int(win_idx_str.replace("window_", ""))
        if not recs:
            summary.append({"window": win_idx, "filter": filt, "n_frames": 0,
                             "mean_confidence": None, "frac_shape_reliable": None,
                             "aspect_ratio_mean": None, "aspect_ratio_stdev": None})
            continue
        confs = [r["confidence"] for r in recs]
        reliables = [r["shape_reliable"] for r in recs]
        aspects = [r["aspect_ratio"] for r in recs if not np.isnan(r["aspect_ratio"])]
        summary.append({
            "window": win_idx,
            "filter": filt,
            "n_frames": len(recs),
            "mean_confidence": float(np.mean(confs)),
            "frac_shape_reliable": float(np.mean(reliables)),
            "aspect_ratio_mean": float(np.mean(aspects)) if aspects else None,
            "aspect_ratio_stdev": float(np.std(aspects)) if len(aspects) > 1 else 0.0,
        })

    out = {
        "per_frame": per_window_filter,
        "summary": summary,
        "errors": errors,
    }
    json.dump(out, open("scratch_per_frame_shape_reliability_results.json", "w"), indent=2)

    print("=== Summary (window, filter, n, mean_conf, frac_reliable, aspect_mean, aspect_stdev) ===")
    for s in sorted(summary, key=lambda x: (x["window"], x["filter"])):
        print(f"  W{s['window']} {s['filter']:>2} n={s['n_frames']:>2} "
              f"mean_conf={s['mean_confidence']} frac_reliable={s['frac_shape_reliable']} "
              f"aspect_mean={s['aspect_ratio_mean']} aspect_stdev={s['aspect_ratio_stdev']}")

    if errors:
        print(f"\n=== {len(errors)} ERRORS ===")
        for e in errors:
            print(f"  {e}")


if __name__ == "__main__":
    main()
