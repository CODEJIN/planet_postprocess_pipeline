"""One-off investigation script: re-verify Saturn warp_scale under masks that
exclude progressively more of the outer disk (where the ring/Cassini
division/limb could contaminate the NCC metric), to check whether the
previously-found optimum of ~0.10 (vs Jupiter's validated 1.00) survives
when the measurement is restricted to a region far from the ring.

Uses real Step-2 lucky-stack output in Saturn_Data/step02_lucky_stack/
(50 frames, IR/R/G/B/CH4, 2026-08-07 16:14-17:29). Not part of the
production pipeline -- exploratory/diagnostic only.
"""
from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.modules import image_io
from pipeline.modules.derotation import (
    auto_detect_ns_flip,
    auto_detect_pole_pa,
    find_disk_center,
    spherical_derotation_warp,
)

DATA_DIR = Path("Saturn_Data/step02_lucky_stack")
PERIOD_HOURS = 10.56  # Saturn System III, per DerotationConfig docstring
HP_SIGMA = 30.0        # matches _measure_derot_confidence's high-pass sigma

FNAME_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})-(\d{2})(\d{2})_(\d)-U-([A-Za-z0-9]+)-Sat_ser_crop_lucky\.tif$"
)


def _parse_frame(path: Path):
    m = FNAME_RE.match(path.name)
    if not m:
        return None
    date_s, hh, mm, decimin, filt = m.groups()
    sec = int(decimin) * 6  # "_1" = 0.1 min = 6s
    dt = datetime.strptime(f"{date_s} {hh}:{mm}:{sec:02d}", "%Y-%m-%d %H:%M:%S")
    return {"path": path, "timestamp": dt, "filter": filt}


def _load_frames():
    frames = []
    for p in sorted(DATA_DIR.glob("*.tif")):
        info = _parse_frame(p)
        if info:
            frames.append(info)
    return frames


def _highpass(img: np.ndarray, sigma: float) -> np.ndarray:
    return img - cv2.GaussianBlur(img, (0, 0), sigma)


def _build_masks(h, w, cx, cy, semi_a):
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return {
        "baseline_0.70R":        r < (0.70 * semi_a),
        "modest_0.55R":          r < (0.55 * semi_a),
        "inner_annulus_0.35-0.60R": (r >= (0.35 * semi_a)) & (r < (0.60 * semi_a)),
    }


def _ncc_sweep_for_pair(early: dict, late: dict, scales, pole_pa_deg: float, flip_direction: bool):
    raw_e = image_io.read_tif(early["path"])
    raw_l = image_io.read_tif(late["path"])
    lum_e = raw_e if raw_e.ndim == 2 else raw_e.mean(axis=2).astype(np.float32)
    lum_l = raw_l if raw_l.ndim == 2 else raw_l.mean(axis=2).astype(np.float32)

    cx, cy, semi_a, semi_b, _ = find_disk_center(lum_e)
    if semi_a < 5:
        return None
    polar_eq = float(np.clip(semi_b / max(semi_a, 1.0), 0.85, 1.0))

    dt = (late["timestamp"] - early["timestamp"]).total_seconds()
    rotation_deg = dt / (PERIOD_HOURS * 3600.0) * 360.0

    h, w = lum_e.shape
    masks = _build_masks(h, w, cx, cy, semi_a)

    lum_l_hp = _highpass(lum_l, HP_SIGMA)
    ref_by_mask = {name: lum_l_hp[m].astype(np.float64) for name, m in masks.items()}
    for name, ref_px in ref_by_mask.items():
        if ref_px.size < 30 or ref_px.std() < 1e-6:
            return None

    result = {"filter": early["filter"], "dt_sec": dt, "rotation_deg": rotation_deg,
              "cx": cx, "cy": cy, "semi_a": semi_a, "semi_b": semi_b}

    # Forward prediction replicates the drift -> opposite sense from de-rotation.
    forward_flip = not flip_direction

    for name, ref_px in ref_by_mask.items():
        best_ncc, best_scale = -1.0, None
        ncc_at_010, ncc_at_100 = None, None
        for scale in scales:
            warped = spherical_derotation_warp(
                lum_e, dt, cx, cy, semi_a,
                period_hours=PERIOD_HOURS, scale=scale,
                flip_direction=forward_flip, pole_pa_deg=pole_pa_deg,
                polar_equatorial_ratio=polar_eq,
            )
            pred_px = _highpass(warped, HP_SIGMA)[masks[name]].astype(np.float64)
            if pred_px.std() < 1e-6:
                ncc = 0.0
            else:
                ncc = float(np.corrcoef(ref_px, pred_px)[0, 1])
            if ncc > best_ncc:
                best_ncc, best_scale = ncc, scale
            if abs(scale - 0.10) < 1e-9:
                ncc_at_010 = ncc
            if abs(scale - 1.00) < 1e-9:
                ncc_at_100 = ncc
        result[name] = {
            "best_scale": best_scale, "best_ncc": best_ncc,
            "ncc_at_0.10": ncc_at_010, "ncc_at_1.00": ncc_at_100,
        }
    return result


def main():
    frames = _load_frames()
    print(f"Loaded {len(frames)} frames")
    by_filter: dict[str, list[dict]] = {}
    for f in frames:
        by_filter.setdefault(f["filter"], []).append(f)
    for filt in by_filter:
        by_filter[filt].sort(key=lambda x: x["timestamp"])
        print(f"  {filt}: {len(by_filter[filt])} frames")

    scales = sorted(set(list(np.round(np.arange(0.0, 1.21, 0.05), 2)) + [0.10, 1.00]))

    # ── Auto-detect pole_pa and flip_direction from real IR frames, exactly
    # like the production pipeline (_scan_session_pole_pa / auto_detect_ns_flip)
    # -- a wrong flip direction would push every mask's "best_scale" toward 0
    # regardless of ring contamination, which would look identical to the
    # effect we're trying to isolate. Must get this right before trusting
    # any of the sweep results below.
    probe_filt = "IR" if "IR" in by_filter else next(iter(by_filter))
    probe_frames_info = by_filter[probe_filt]
    probe_imgs = []
    probe_lums = []
    for fi in probe_frames_info:
        raw = image_io.read_tif(fi["path"])
        lum = raw if raw.ndim == 2 else raw.mean(axis=2).astype(np.float32)
        probe_lums.append(lum)
    ref_cx, ref_cy, ref_semi_a, ref_semi_b, _ = find_disk_center(probe_lums[0])
    pole_pa_deg = auto_detect_pole_pa(probe_lums, ref_cx, ref_cy, ref_semi_a)
    t0 = probe_frames_info[0]["timestamp"]
    dt_list = [(fi["timestamp"] - t0).total_seconds() for fi in probe_frames_info]
    polar_eq0 = float(np.clip(ref_semi_b / max(ref_semi_a, 1.0), 0.85, 1.0))
    derot_flip, score_f, score_t = auto_detect_ns_flip(
        probe_lums, dt_list, ref_cx, ref_cy, ref_semi_a,
        period_hours=PERIOD_HOURS, warp_scale=0.5,
        pole_pa_deg=pole_pa_deg, polar_equatorial_ratio=polar_eq0,
    )
    print(f"\nAuto-detected: pole_pa_deg={pole_pa_deg:.1f}  derot_flip={derot_flip}  "
          f"(scores: false={score_f:.4f} true={score_t:.4f})  "
          f"disk cx={ref_cx:.1f} cy={ref_cy:.1f} semi_a={ref_semi_a:.1f} semi_b={ref_semi_b:.1f}\n")

    # Two pairing strategies:
    #  - "adjacent": consecutive same-filter frames (~7.6 min apart, ~4-5 deg)
    #  - "span3": 3 cycles apart (~23 min apart, ~13-14 deg) -- closer to the
    #    "rotation >= 13 deg" subset used in the original session sweep.
    pair_specs = [("adjacent", 1), ("span3", 3)]

    all_results = []
    for filt, flist in by_filter.items():
        if filt == "CH4":
            continue  # CH4 disk detection is independently known-unreliable; skip here
        for label, step in pair_specs:
            for i in range(len(flist) - step):
                early, late = flist[i], flist[i + step]
                try:
                    r = _ncc_sweep_for_pair(early, late, scales, pole_pa_deg, derot_flip)
                except Exception as exc:
                    print(f"  [skip] {filt} {label} idx={i}: {exc}")
                    continue
                if r is None:
                    continue
                r["pairing"] = label
                r["early"] = early["path"].name
                r["late"] = late["path"].name
                all_results.append(r)
                print(f"  {filt:4s} {label:9s} rot={r['rotation_deg']:5.1f}deg  "
                      + "  ".join(
                          f"{name}: best={v['best_scale']:.2f}(ncc={v['best_ncc']:.3f})"
                          for name, v in r.items() if isinstance(v, dict)
                      ))

    print(f"\nTotal usable pairs: {len(all_results)}")
    mask_names = ["baseline_0.70R", "modest_0.55R", "inner_annulus_0.35-0.60R"]
    print("\n=== Summary: median best_scale per mask (rotation >= 8 deg only) ===")
    for name in mask_names:
        vals = [r[name]["best_scale"] for r in all_results
                if r["rotation_deg"] >= 8.0 and r[name]["best_ncc"] > 0.3]
        if vals:
            print(f"  {name:28s} n={len(vals):3d}  median={np.median(vals):.3f}  "
                  f"mean={np.mean(vals):.3f}  range=[{min(vals):.2f},{max(vals):.2f}]")
        else:
            print(f"  {name:28s} n=0 (no pairs met NCC>0.3 threshold)")

    print("\n=== Summary: NCC@scale=0.10 vs NCC@scale=1.00, per mask (rotation >= 8 deg) ===")
    for name in mask_names:
        n010 = [r[name]["ncc_at_0.10"] for r in all_results if r["rotation_deg"] >= 8.0]
        n100 = [r[name]["ncc_at_1.00"] for r in all_results if r["rotation_deg"] >= 8.0]
        if n010:
            print(f"  {name:28s} n={len(n010):3d}  "
                  f"mean_NCC@0.10={np.mean(n010):.4f}  mean_NCC@1.00={np.mean(n100):.4f}  "
                  f"(0.10 wins in {sum(a>b for a,b in zip(n010,n100))}/{len(n010)})")

    import json
    out = {
        "scales": scales,
        "results": [
            {k: (v if not isinstance(v, dict) else v) for k, v in r.items()}
            for r in all_results
        ],
    }
    with open("scratch_warp_scale_mask_sweep_results.json", "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print("\nSaved raw results to scratch_warp_scale_mask_sweep_results.json")


if __name__ == "__main__":
    main()
