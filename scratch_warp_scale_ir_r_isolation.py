"""
Isolate: does spherical_derotation_warp ALONE (no registration, no stacking)
blur IR content more than R content?

For window_02 and window_05 (worst IR ratio performers per known real stack
results), for filter IR and filter R, take each non-reference frame (real,
non-trivial dt_sec from the actual logged frame list) and produce:
  - original (unwarped) sharpness
  - scale=0.0 (pure identity warp -- same interpolation/remap pipeline, zero
    drift) sharpness
  - scale=0.10 (real production warp_scale) sharpness
all measured with Laplacian variance in the central 55%-of-radius disk
region, using ONLY real production functions from pipeline.modules.derotation.

Rerun fresh -- this script predates nothing, but per ground rules we do not
trust any stale output file, so this is a clean run every invocation.
"""
import sys, json, glob
sys.path.insert(0, '.')
import numpy as np
import cv2

from pipeline.modules import image_io
from pipeline.modules.derotation import (
    spherical_derotation_warp,
    find_disk_center,
)

STEP02_DIR = "Saturn_Data/step02_lucky_stack"
STEP04_DIR = "Saturn_Data/step04_derotated"

WINDOWS = ["window_02", "window_05"]
FILTERS = ["IR", "R"]


def find_step02_path(stem):
    matches = glob.glob(f"{STEP02_DIR}/{stem}.tif")
    if not matches:
        raise FileNotFoundError(stem)
    return matches[0]


def laplacian_var_central(img, cx, cy, radius, frac=0.55):
    """Laplacian-variance sharpness inside the central `frac` of disk radius."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = r <= (radius * frac)
    lap = cv2.Laplacian(img.astype(np.float32), cv2.CV_32F, ksize=3)
    vals = lap[mask]
    return float(np.var(vals))


def polar_equatorial_ratio_for_ref(ref_img, used_semi_a):
    """Reproduce derotate_filter's _polar_eq_ratio computation for the
    reference frame: own ellipse fit -> rescale semi_b to the (possibly
    radius-shared) used semi_a -> clip to [0.85, 1.0]."""
    cx, cy, semi_a, semi_b, _angle = find_disk_center(ref_img)
    if semi_a > 0:
        semi_b_scaled = semi_b * (used_semi_a / semi_a)
    else:
        semi_b_scaled = semi_b
    ratio = float(np.clip(semi_b_scaled / max(used_semi_a, 1.0), 0.85, 1.0))
    return ratio, cx, cy, semi_a, semi_b


results = []
errors = []

for win in WINDOWS:
    log_path = f"{STEP04_DIR}/{win}/derotation_log.json"
    log = json.load(open(log_path))
    filters_present = log["filters"]

    for filt in FILTERS:
        if filt not in filters_present:
            errors.append({"window": win, "filter": filt, "error": "filter not present in window"})
            continue
        try:
            fentry = filters_present[filt]
            period_hours = fentry["period_hours"]
            warp_scale_cfg = fentry["warp_scale"]
            pole_pa_deg = fentry["pole_pa_deg"]
            flip_direction = fentry["flip_direction"]
            ref_stem = fentry["reference_stem"]

            frames = fentry["frames"]
            ref_frame_entry = next(fr for fr in frames if fr["stem"] == ref_stem)
            used_cx, used_cy = ref_frame_entry["disk_center_px"]
            used_radius = ref_frame_entry["disk_radius_px"]

            ref_path = find_step02_path(ref_stem)
            ref_img = image_io.read_tif(ref_path)
            if ref_img.ndim == 3:
                ref_img = ref_img.mean(axis=2).astype(np.float32)

            ratio, own_cx, own_cy, own_semi_a, own_semi_b = polar_equatorial_ratio_for_ref(
                ref_img, used_radius
            )

            non_ref_frames = [fr for fr in frames if fr["dt_sec"] != 0.0]
            if not non_ref_frames:
                errors.append({"window": win, "filter": filt, "error": "no non-trivial dt_sec frames found"})
                continue

            for fr in non_ref_frames:
                stem = fr["stem"]
                dt_sec = fr["dt_sec"]
                path = find_step02_path(stem)
                img = image_io.read_tif(path)
                if img.ndim == 3:
                    img = img.mean(axis=2).astype(np.float32)

                raw_sharp = laplacian_var_central(img, used_cx, used_cy, used_radius)

                warped_010 = spherical_derotation_warp(
                    img, dt_sec, used_cx, used_cy, used_radius,
                    period_hours=period_hours,
                    scale=warp_scale_cfg,
                    flip_direction=flip_direction,
                    pole_pa_deg=pole_pa_deg,
                    polar_equatorial_ratio=ratio,
                )
                warped_000 = spherical_derotation_warp(
                    img, dt_sec, used_cx, used_cy, used_radius,
                    period_hours=period_hours,
                    scale=0.0,
                    flip_direction=flip_direction,
                    pole_pa_deg=pole_pa_deg,
                    polar_equatorial_ratio=ratio,
                )

                sharp_010 = laplacian_var_central(warped_010, used_cx, used_cy, used_radius)
                sharp_000 = laplacian_var_central(warped_000, used_cx, used_cy, used_radius)

                rec = {
                    "window": win,
                    "filter": filt,
                    "stem": stem,
                    "dt_sec": dt_sec,
                    "warp_scale_used": warp_scale_cfg,
                    "sharp_original": raw_sharp,
                    "sharp_identity_warp_scale0": sharp_000,
                    "sharp_real_warp_scale010": sharp_010,
                    "ratio_identity_over_original": sharp_000 / raw_sharp if raw_sharp > 0 else float("nan"),
                    "ratio_real_over_original": sharp_010 / raw_sharp if raw_sharp > 0 else float("nan"),
                    "ratio_real_over_identity": sharp_010 / sharp_000 if sharp_000 > 0 else float("nan"),
                }
                results.append(rec)
                print(
                    f"{win} {filt} {stem} dt={dt_sec:+.1f}s  "
                    f"orig={raw_sharp:.6e}  identity(scale=0)={sharp_000:.6e} (x{rec['ratio_identity_over_original']:.4f})  "
                    f"real(scale={warp_scale_cfg})={sharp_010:.6e} (x{rec['ratio_real_over_original']:.4f})  "
                    f"real/identity={rec['ratio_real_over_identity']:.4f}"
                )
        except Exception as e:
            errors.append({"window": win, "filter": filt, "error": repr(e)})
            print(f"== {win} {filt} == ERROR: {e!r}")

print()
print("=== Summary by filter ===")
for filt in FILTERS:
    sub = [r for r in results if r["filter"] == filt]
    if not sub:
        print(f"{filt}: no results")
        continue
    r_id = [r["ratio_identity_over_original"] for r in sub]
    r_real = [r["ratio_real_over_original"] for r in sub]
    r_ri = [r["ratio_real_over_identity"] for r in sub]
    print(f"{filt}: n={len(sub)}  "
          f"median identity/orig={np.median(r_id):.4f}  median real/orig={np.median(r_real):.4f}  "
          f"median real/identity={np.median(r_ri):.4f}")

if errors:
    print()
    print("=== Errors ===")
    for e in errors:
        print(e)

json.dump({"results": results, "errors": errors},
           open("scratch_warp_scale_ir_r_isolation_results.json", "w"), indent=2)
