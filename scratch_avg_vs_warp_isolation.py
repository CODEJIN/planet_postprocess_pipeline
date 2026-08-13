"""
Isolate multi-frame-averaging loss from de-rotation-warp loss.

For window_01/05/09, filters IR and R:
  1. Per-frame sharpness of each raw step02 frame (own find_disk_center, own mask).
  2. Translation-only aligned average: align every included raw frame to the
     window's reference frame via subpixel_align()+apply_shift() (NO de-rotation
     warp at all), then quality_weighted_stack() with the real norm_score weights
     (weight_power=1.0, matching production default) -- isolates pure averaging
     loss from warp-induced blur.
  3. Compare against the REAL step04 de-rotated+stacked TIF already on disk.

All sharpness numbers use pipeline.modules.quality.laplacian_var() inside a
circular mask = central 55% of the REFERENCE frame's own find_disk_center()
radius (single consistent mask per window/filter, matching the earlier
investigation's methodology).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np

from pipeline.modules import image_io
from pipeline.modules.derotation import find_disk_center, subpixel_align, apply_shift, quality_weighted_stack
from pipeline.modules.quality import laplacian_var

ROOT = Path("Saturn_Data")
STEP02 = ROOT / "step02_lucky_stack"
STEP04 = ROOT / "step04_derotated"

WINDOWS_FILTERS = [
    ("window_01", "IR"), ("window_01", "R"),
    ("window_05", "IR"), ("window_05", "R"),
    ("window_09", "IR"), ("window_09", "R"),
]


def disk_mask(shape, cx, cy, radius, frac=0.55):
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    return dist2 <= (radius * frac) ** 2


def main():
    results = []
    for win, filt in WINDOWS_FILTERS:
        log = json.loads((STEP04 / win / "derotation_log.json").read_text())
        finfo = log["filters"][filt]
        ref_stem = finfo["reference_stem"]
        frames_meta = finfo["frames"]  # includes reference + others, with norm_score

        # Load all raw step02 frames (as used in derotate_filter, pre-warp)
        raw = {}
        for fm in frames_meta:
            stem = fm["stem"]
            tif = STEP02 / f"{stem}.tif"
            raw[stem] = image_io.read_tif(tif)

        ref_img = raw[ref_stem]
        ref_cx, ref_cy, ref_ra, ref_rb, _ = find_disk_center(ref_img)
        ref_radius = (ref_ra + ref_rb) / 2.0
        mask = disk_mask(ref_img.shape[:2], ref_cx, ref_cy, ref_radius, 0.55)

        # 1. Per-frame own sharpness (own find_disk_center + own 55% mask)
        per_frame_sharp = {}
        norm_scores = {}
        for fm in frames_meta:
            stem = fm["stem"]
            img = raw[stem]
            cx, cy, ra, rb, _ = find_disk_center(img)
            radius = (ra + rb) / 2.0
            m = disk_mask(img.shape[:2], cx, cy, radius, 0.55)
            per_frame_sharp[stem] = laplacian_var(img, m)
            norm_scores[stem] = fm["norm_score"]

        best_stem = max(norm_scores, key=norm_scores.get)
        best_single_sharp = per_frame_sharp[best_stem]

        # 2. Translation-only aligned average (no warp), same weights/weight_power
        #    as production quality_weighted_stack default (1.0).
        aligned_imgs = []
        weights = []
        for fm in frames_meta:
            stem = fm["stem"]
            img = raw[stem]
            if stem == ref_stem:
                aligned_imgs.append(img)
            else:
                dx, dy = subpixel_align(ref_img, img)
                aligned_imgs.append(apply_shift(img, dx, dy))
            weights.append(fm["norm_score"])

        translation_avg = quality_weighted_stack(aligned_imgs, weights, weight_power=1.0)
        translation_avg_sharp = laplacian_var(translation_avg, mask)

        # 3. Real step04 output, same mask (ref frame's own disk geometry)
        step04_path = STEP04 / win / f"{filt}_derotated.tif"
        step04_img = image_io.read_tif(step04_path)
        step04_sharp = laplacian_var(step04_img, mask)

        avg_loss_ratio = translation_avg_sharp / best_single_sharp
        warp_extra_ratio = step04_sharp / translation_avg_sharp

        results.append({
            "window": win,
            "filter": filt,
            "n_frames": len(frames_meta),
            "best_stem": best_stem,
            "best_single_sharpness": best_single_sharp,
            "translation_only_avg_sharpness": translation_avg_sharp,
            "real_step04_sharpness": step04_sharp,
            "avg_vs_best_ratio": avg_loss_ratio,
            "step04_vs_avg_ratio": warp_extra_ratio,
            "per_frame_sharp": per_frame_sharp,
        })

        print(f"\n=== {win} / {filt} (n={len(frames_meta)}, ref={ref_stem}) ===")
        for stem, s in per_frame_sharp.items():
            tag = " <- best/ref" if stem == best_stem else ""
            print(f"  frame {stem}: sharpness={s:.6e} norm_score={norm_scores[stem]:.3f}{tag}")
        print(f"  best single frame sharpness:        {best_single_sharp:.6e}")
        print(f"  translation-only aligned avg:        {translation_avg_sharp:.6e}  (ratio vs best: {avg_loss_ratio:.3f})")
        print(f"  REAL step04 (derotated+stacked):     {step04_sharp:.6e}  (ratio vs translation-avg: {warp_extra_ratio:.3f})")

    Path("scratch_avg_vs_warp_isolation_results.json").write_text(
        json.dumps(results, indent=2, default=float)
    )

    print("\n\n=== SUMMARY ===")
    avg_ratios = [r["avg_vs_best_ratio"] for r in results]
    warp_ratios = [r["step04_vs_avg_ratio"] for r in results]
    print(f"avg_vs_best_ratio: median={np.median(avg_ratios):.3f}  values={[round(x,3) for x in avg_ratios]}")
    print(f"step04_vs_avg_ratio: median={np.median(warp_ratios):.3f}  values={[round(x,3) for x in warp_ratios]}")


if __name__ == "__main__":
    main()
