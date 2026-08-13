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

WINDOWS = ["window_01", "window_05", "window_09"]
FILTERS_PREF = ["IR", "R", "G", "B"]  # try IR + first available of R/G/B


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

for win in WINDOWS:
    log_path = f"{STEP04_DIR}/{win}/derotation_log.json"
    log = json.load(open(log_path))
    filters_present = list(log["filters"].keys())
    chosen = []
    if "IR" in filters_present:
        chosen.append("IR")
    for f in ["R", "G", "B"]:
        if f in filters_present:
            chosen.append(f)
            break

    for filt in chosen:
        fentry = log["filters"][filt]
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
                "raw_sharp": raw_sharp,
                "warp010_sharp": sharp_010,
                "warp000_sharp": sharp_000,
                "ratio_010_over_raw": sharp_010 / raw_sharp if raw_sharp > 0 else float("nan"),
                "ratio_000_over_raw": sharp_000 / raw_sharp if raw_sharp > 0 else float("nan"),
            }
            results.append(rec)
            print(
                f"{win} {filt} {stem} dt={dt_sec:+.1f}s  "
                f"raw={raw_sharp:.6e}  warp0.10={sharp_010:.6e} (x{rec['ratio_010_over_raw']:.3f})  "
                f"warp0.00={sharp_000:.6e} (x{rec['ratio_000_over_raw']:.3f})"
            )

print()
print("=== Summary ===")
r010 = [r["ratio_010_over_raw"] for r in results]
r000 = [r["ratio_000_over_raw"] for r in results]
print(f"N frames tested: {len(results)}")
print(f"median warp(0.10)/raw ratio: {np.median(r010):.4f}")
print(f"mean   warp(0.10)/raw ratio: {np.mean(r010):.4f}")
print(f"median warp(0.00)/raw ratio: {np.median(r000):.4f}")
print(f"mean   warp(0.00)/raw ratio: {np.mean(r000):.4f}")

json.dump(results, open("scratch_warp_blur_test_results.json", "w"), indent=2)
