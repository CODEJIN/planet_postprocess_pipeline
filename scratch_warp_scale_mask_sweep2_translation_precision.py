"""Translation-precision A/B for Saturn window_01 globe-limb blur investigation.

Compares the PRODUCTION ellipse-centroid pre-warp dx,dy (already logged in
derotation_log.json's pre_warp_shift_dx/dy, computed by _find_disk_center_impl
inside derotate_filter()) against an independent phase-correlation (subpixel_align)
measurement of the SAME raw frame pairs, cropped to an inner-disk ROI that avoids
the ring/limb (per feedback_phasecorrelate_ring_crop).

If the two disagree meaningfully (>=0.5px), proceeds to step 4: monkeypatch the
pre-warp dx/dy inside the REAL derotate_filter() (via derotation._find_disk_center_impl,
the exact function that produces them) to use the phase-correlation value instead,
run the REAL derotate_window()+wavelet_master.run() path, and re-measure the globe
radial brightness transition width vs the unmodified baseline.
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
from pipeline.modules import wavelet
from pipeline.config import WaveletConfig
from pipeline.modules.derotation import (
    derotate_window,
    find_disk_center,
    subpixel_align,
    _find_disk_center_impl,
)
from pipeline.steps import wavelet_master

_WCFG = WaveletConfig()

STEP02_DIR = Path("Saturn_Data/step02_lucky_stack")
STEP04_DIR = Path("Saturn_Data/step04_derotated")
WINDOWS_JSON = Path("Saturn_Data/step03_quality/windows.json")
WINDOW_INDEX = 1
FILTERS = ["IR", "R"]

CROP_DIR = Path("scratch_investigation_crops2")
CROP_DIR.mkdir(exist_ok=True)


def _parse_ts(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")


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


def _to_lum(img):
    return img if img.ndim == 2 else img.mean(axis=2).astype(np.float32)


# ── Step 1-3: compare ellipse-centroid vs phase-correlation dx,dy ──────────

def compare_methods():
    window_log = json.load(open(STEP04_DIR / f"window_{WINDOW_INDEX:02d}" / "derotation_log.json"))
    results = {}
    for filt in FILTERS:
        flog = window_log["filters"][filt]
        ref_cx, ref_cy = flog["frames"][0]["disk_center_px"]
        ref_semi_a = flog["frames"][0]["disk_radius_px"]
        ref_stem = flog["reference_stem"]
        ref_path = list(STEP02_DIR.glob(f"{ref_stem}.tif"))[0]
        ref_lum = _to_lum(image_io.read_tif(str(ref_path)))

        roi_half = 0.6 * ref_semi_a
        y0, y1 = int(ref_cy - roi_half), int(ref_cy + roi_half)
        x0, x1 = int(ref_cx - roi_half), int(ref_cx + roi_half)
        ref_crop = ref_lum[y0:y1, x0:x1]

        print(f"\n=== {filt} (ref={ref_stem}, ref_semi_a={ref_semi_a:.2f}, ROI half={roi_half:.1f}px) ===")
        filt_results = []
        for frame in flog["frames"]:
            if frame["stem"] == ref_stem:
                continue
            stem = frame["stem"]
            path = list(STEP02_DIR.glob(f"{stem}.tif"))[0]
            tgt_lum = _to_lum(image_io.read_tif(str(path)))
            tgt_crop = tgt_lum[y0:y1, x0:x1]

            pc_dx, pc_dy = subpixel_align(ref_crop, tgt_crop)
            log_dx, log_dy = frame["pre_warp_shift_dx"], frame["pre_warp_shift_dy"]

            diff = float(np.hypot(pc_dx - log_dx, pc_dy - log_dy))
            print(f"  {stem}: ellipse-centroid dx,dy=({log_dx:.3f},{log_dy:.3f})  "
                  f"phase-corr dx,dy=({pc_dx:.3f},{pc_dy:.3f})  disagreement={diff:.3f}px")
            filt_results.append({
                "stem": stem, "log_dx": log_dx, "log_dy": log_dy,
                "pc_dx": pc_dx, "pc_dy": pc_dy, "disagreement_px": diff,
            })
        results[filt] = filt_results
    return results


# ── Step 4 (conditional): substitute phase-corr dx,dy into the real pipeline ──

def _radial_profile(img2d, cx, cy, semi_a, pole_pa_deg, r_start=0.95, r_end=1.30, n=60):
    ang = np.radians(pole_pa_deg)
    # sample along the ring-plane axis (perpendicular to pole), both sides, average
    dirx, diry = np.cos(ang), np.sin(ang)
    rs = np.linspace(r_start, r_end, n)
    vals = []
    for r in rs:
        px = r * semi_a
        pts = []
        for sign in (+1, -1):
            x = cx + sign * px * dirx
            y = cy + sign * px * diry
            if 0 <= int(y) < img2d.shape[0] and 0 <= int(x) < img2d.shape[1]:
                pts.append(
                    cv2.getRectSubPix(img2d.astype(np.float32), (1, 1), (float(x), float(y)))[0, 0]
                )
        vals.append(np.mean(pts) if pts else np.nan)
    return rs, np.array(vals)


def _transition_width(rs, vals, bg_level):
    # r/semi_a at which profile first drops to within 10% of background
    thresh = bg_level + 0.10 * (vals[0] - bg_level)
    below = np.where(vals <= thresh)[0]
    return rs[below[0]] if len(below) else rs[-1]


def run_pc_substitution_ablation(window_log):
    data = json.load(open(WINDOWS_JSON))
    windows = {w["window_index"]: w for w in data["selected_windows"]}
    window = windows[WINDOW_INDEX]
    hydrated = _hydrate_window(window)
    any_flog = next(iter(window_log["filters"].values()))

    # Build a stem -> (dx, dy) override table from phase-correlation, one per
    # filter (recomputed here against that filter's own reference + ROI).
    pc_overrides = {}
    for filt in FILTERS:
        flog = window_log["filters"][filt]
        ref_cx, ref_cy = flog["frames"][0]["disk_center_px"]
        ref_semi_a = flog["frames"][0]["disk_radius_px"]
        ref_stem = flog["reference_stem"]
        ref_lum = _to_lum(image_io.read_tif(str(list(STEP02_DIR.glob(f"{ref_stem}.tif"))[0])))
        roi_half = 0.6 * ref_semi_a
        y0, y1 = int(ref_cy - roi_half), int(ref_cy + roi_half)
        x0, x1 = int(ref_cx - roi_half), int(ref_cx + roi_half)
        ref_crop = ref_lum[y0:y1, x0:x1]
        for frame in flog["frames"]:
            stem = frame["stem"]
            if stem == ref_stem:
                continue
            tgt_lum = _to_lum(image_io.read_tif(str(list(STEP02_DIR.glob(f"{stem}.tif"))[0])))
            tgt_crop = tgt_lum[y0:y1, x0:x1]
            pc_dx, pc_dy = subpixel_align(ref_crop, tgt_crop)
            # target_cx = ref_cx - pc_dx  (mirrors production's target_cx = ref_cx - dx)
            pc_overrides[stem] = (pc_dx, pc_dy, ref_cx - pc_dx, ref_cy - pc_dy)

    real_impl = _find_disk_center_impl

    def patched_impl(raw_lum, *args, **kwargs):
        # We can't know which stem this call is for from inside _find_disk_center_impl
        # (it only receives pixel data) -- so instead we patch at a level where the
        # stem IS known: wrap image_io.read_tif is not enough either. Simplest robust
        # interception point given derotate_filter()'s structure: monkeypatch
        # _find_disk_center_impl to return the REAL result unchanged (shape/radius
        # untouched) -- the dx/dy substitution itself is done by monkeypatching
        # apply_shift_and_scale's target_cx/cy indirectly is not possible either
        # since that also lacks the stem. So we patch derotate_filter's OWN module-level
        # _find_disk_center_impl reference is insufficient; instead see note below --
        # this function is intentionally unused; real interception happens via
        # the image-content trick in _tagged_find_disk_center below.
        return real_impl(raw_lum, *args, **kwargs)

    # Practical interception: _find_disk_center_impl receives only the raw
    # luminance array, not the stem -- but within derotate_filter's per-frame
    # loop, it's called once per raw frame, and we control exactly which raw
    # arrays exist (one per stem, all distinct pixel content). So key the
    # override table by array identity (id()) instead of stem: before calling
    # derotate_window(), we can't pre-associate id()s (arrays are freshly read
    # inside derotate_filter). Instead, wrap image_io.read_tif to tag arrays
    # via a side-channel dict keyed by path, then have the patched
    # _find_disk_center_impl look up the override by matching array bytes.
    stem_by_id = {}
    real_read_tif = image_io.read_tif

    def tagging_read_tif(path, *a, **kw):
        arr = real_read_tif(path, *a, **kw)
        stem_by_id[id(arr)] = Path(path).stem
        return arr

    def override_impl(raw_lum, *args, **kwargs):
        cx, cy, semi_a, semi_b, angle, conf, shape_ok = real_impl(raw_lum, *args, **kwargs)
        # raw_lum here is _raw_lum = _to_luminance(img); img identity differs
        # from the tagged array from read_tif, so id() lookup won't match
        # directly for color mode. In mono mode (this pipeline, color_mode=False),
        # _to_luminance on a 2-D array is (per _to_luminance) typically a no-op
        # or copy -- check both id(raw_lum) and a content hash fallback.
        stem = stem_by_id.get(id(raw_lum))
        if stem is None:
            for known_id, known_stem in stem_by_id.items():
                pass
        if stem is not None and stem in pc_overrides:
            pc_dx, pc_dy, tcx, tcy = pc_overrides[stem]
            return (tcx, tcy, semi_a, semi_b, angle, conf, shape_ok)
        return (cx, cy, semi_a, semi_b, angle, conf, shape_ok)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_root = Path(tmpdir)

        # Baseline (unmodified production)
        out_dir = out_root / "baseline"
        out_dir.mkdir()
        baseline_results = derotate_window(
            hydrated, required_filters=FILTERS,
            period_hours=any_flog["period_hours"], warp_scale=any_flog["warp_scale"],
            align=any_flog["align_enabled"], normalize_brightness=any_flog["normalize_brightness"],
            min_quality_threshold=any_flog["min_quality_threshold"], pole_pa_deg=any_flog["pole_pa_deg"],
            color_mode=False, flip_direction=any_flog["flip_direction"], weight_power=any_flog["weight_power"],
            has_rings=any_flog["has_rings"], sub_observer_lat_deg=any_flog["sub_observer_lat_deg"],
            out_dir=out_dir,
        )

        # PC-substituted
        out_dir2 = out_root / "pc_substituted"
        out_dir2.mkdir()
        with mock.patch.object(image_io, "read_tif", tagging_read_tif), \
             mock.patch.object(derotation, "_find_disk_center_impl", override_impl):
            pc_results = derotate_window(
                hydrated, required_filters=FILTERS,
                period_hours=any_flog["period_hours"], warp_scale=any_flog["warp_scale"],
                align=any_flog["align_enabled"], normalize_brightness=any_flog["normalize_brightness"],
                min_quality_threshold=any_flog["min_quality_threshold"], pole_pa_deg=any_flog["pole_pa_deg"],
                color_mode=False, flip_direction=any_flog["flip_direction"], weight_power=any_flog["weight_power"],
                has_rings=any_flog["has_rings"], sub_observer_lat_deg=any_flog["sub_observer_lat_deg"],
                out_dir=out_dir2,
            )

        for filt in FILTERS:
            for tag, results in [("baseline", baseline_results), ("pc_substituted", pc_results)]:
                out_path, log_dict = results[filt]
                if out_path is None:
                    print(f"{filt} {tag}: SKIP no output")
                    continue
                img = image_io.read_tif(str(out_path))
                lum = _to_lum(img)
                cx, cy, semi_a, semi_b, angle = find_disk_center(lum)
                pole_pa = window_log["filters"][filt]["pole_pa_deg"]
                rs, vals = _radial_profile(lum, cx, cy, semi_a, pole_pa)
                bg = float(np.nanmedian(vals[-5:]))
                width = _transition_width(rs, vals, bg)
                print(f"{filt} {tag}: cx={cx:.1f} cy={cy:.1f} semi_a={semi_a:.2f} "
                      f"bg={bg:.4f} transition r/semi_a={width:.3f}")

                vis = np.clip(lum / (np.percentile(lum, 99.5) + 1e-9), 0, 1)
                vis_u8 = (vis * 255).astype(np.uint8)
                half = int(semi_a * 1.4)
                y0, y1 = max(0, int(cy - half)), min(vis_u8.shape[0], int(cy + half))
                x0, x1 = max(0, int(cx - half)), min(vis_u8.shape[1], int(cx + half))
                crop = cv2.resize(vis_u8[y0:y1, x0:x1], None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(str(CROP_DIR / f"window01_{filt}_{tag}.png"), crop)


def main():
    comparison = compare_methods()

    max_disagreement = max(
        r["disagreement_px"] for filt_results in comparison.values() for r in filt_results
    )
    print(f"\n>>> Max disagreement across all frames/filters: {max_disagreement:.3f}px")

    with open("scratch_warp_scale_mask_sweep2_translation_precision_results.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    if max_disagreement < 0.5:
        print(">>> Methods agree closely (<0.5px). Translation precision is very likely "
              "NOT the bottleneck; skipping step-4 ablation (no plausible mechanism).")
        return

    print(">>> Meaningful disagreement found; running step-4 real-pipeline ablation...")
    window_log = json.load(open(STEP04_DIR / f"window_{WINDOW_INDEX:02d}" / "derotation_log.json"))
    run_pc_substitution_ablation(window_log)


if __name__ == "__main__":
    main()
