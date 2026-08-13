"""Controlled test of the external review's hypothesis: does composing the
pre-warp affine (apply_shift_and_scale) into spherical_derotation_warp's own
coordinate map -- so the RAW frame is resampled only once instead of twice
(once by warpAffine, again by the warp's internal cv2.remap) -- produce a
measurably sharper result on real Saturn data?

Captures the REAL production spherical_derotation_warp()'s map_x/map_y via a
cv2.remap monkeypatch (pass-through, doesn't change its behavior or return
value -- just records the arguments of the first call), then:
  (a) "current": raw -> apply_shift_and_scale (1 warpAffine) -> spherical_
      derotation_warp(aligned) (uses the captured map internally, 2 more
      remaps for its own cubic/linear blend) = production behavior, 3 total
      interpolation passes touching pixel data.
  (b) "composed": invert the SAME affine and apply it to the captured
      map_x/map_y directly, then resample the RAW (never-affine-transformed)
      image ONCE with that composed map using the identical cubic/linear
      blend the warp itself uses = 2 total interpolation passes (the
      blend's own two remaps merge naturally since they already read from
      whatever coordinates they're given).

Uses a REAL non-trivial dt_sec (so spherical_derotation_warp does actual
work, not an identity no-op) and a REAL non-trivial pre-warp shift+scale
(not identity) so the comparison isn't trivially degenerate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.modules import derotation
from pipeline.modules import image_io
from pipeline.modules.derotation import (
    find_disk_center,
    apply_shift_and_scale,
    spherical_derotation_warp,
)

STEP02_DIR = Path("Saturn_Data/step02_lucky_stack")
STEP04_DIR = Path("Saturn_Data/step04_derotated/window_01")


def _laplacian_var_central(img: np.ndarray, cx: float, cy: float, rx: float, frac: float = 0.55) -> float:
    h, w = img.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = r < (frac * rx)
    lap = cv2.Laplacian(img.astype(np.float32), cv2.CV_32F, ksize=3)
    return float(np.var(lap[mask]))


def main():
    window_log = json.load(open(STEP04_DIR / "derotation_log.json"))
    flog = window_log["filters"]["R"]

    # A real non-reference frame from this window: dt_sec=1020 (real warp,
    # not identity), and its real, already-measured pre-warp shift+scale.
    frame = next(f for f in flog["frames"] if f["dt_sec"] == 1020.0)
    stem = frame["stem"]
    raw_img = image_io.read_tif(list(STEP02_DIR.glob(f"{stem}.tif"))[0])
    raw_lum = raw_img if raw_img.ndim == 2 else raw_img.mean(axis=2).astype(np.float32)

    ref_cx, ref_cy = frame["disk_center_px"]
    ref_semi_a = frame["disk_radius_px"]
    pw_dx, pw_dy, pw_scale = frame["pre_warp_shift_dx"], frame["pre_warp_shift_dy"], frame["pre_warp_scale"]
    # target_center used by apply_shift_and_scale is this frame's own fit --
    # reconstruct it from ref_center - (dx, dy) is NOT right in general (that
    # identity only holds at scale=1); re-fit this frame's own centre directly.
    _, _, own_semi_i, _, _, _, _ = derotation._find_disk_center_impl(raw_lum)
    target_cx, target_cy, _, _, _, _, _ = derotation._find_disk_center_impl(raw_lum)
    print(f"frame={stem} dt_sec=1020 pw_dx={pw_dx} pw_dy={pw_dy} pw_scale={pw_scale}")
    print(f"ref_center=({ref_cx},{ref_cy}) ref_semi_a={ref_semi_a} target_center=({target_cx:.2f},{target_cy:.2f})")

    dt_sec = 1020.0
    pole_pa_deg = flog["pole_pa_deg"]
    period_hours = flog["period_hours"]
    warp_scale = flog["warp_scale"]
    flip_direction = flog["flip_direction"]

    # --- (a) CURRENT production path: affine first, then the real warp ---
    aligned = apply_shift_and_scale(raw_lum, target_cx, target_cy, ref_cx, ref_cy, pw_scale)

    captured = {}
    real_remap = cv2.remap
    def _capturing_remap(src, map_x, map_y, *args, **kwargs):
        if "map_x" not in captured:
            captured["map_x"] = map_x.copy()
            captured["map_y"] = map_y.copy()
        return real_remap(src, map_x, map_y, *args, **kwargs)

    with mock.patch.object(cv2, "remap", _capturing_remap):
        warped_current = spherical_derotation_warp(
            aligned, dt_sec, ref_cx, ref_cy, ref_semi_a,
            period_hours=period_hours, scale=warp_scale,
            flip_direction=flip_direction, pole_pa_deg=pole_pa_deg,
        )

    map_x, map_y = captured["map_x"], captured["map_y"]

    # --- (b) COMPOSED path: invert the affine, apply to the captured map,
    # then resample the RAW image directly with the SAME cubic/linear blend
    # spherical_derotation_warp itself uses (replicated here explicitly). ---
    s = max(float(pw_scale), 1e-8)
    raw_map_x = (float(target_cx) + (map_x - float(ref_cx)) / s).astype(np.float32)
    raw_map_y = (float(target_cy) + (map_y - float(ref_cy)) / s).astype(np.float32)

    src_f32 = raw_lum.astype(np.float32)
    warped_cubic = cv2.remap(src_f32, raw_map_x, raw_map_y, interpolation=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    warped_linear = cv2.remap(src_f32, raw_map_x, raw_map_y, interpolation=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    h, w = raw_lum.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist_from_center = np.sqrt((xx - ref_cx) ** 2 + (yy - ref_cy) ** 2)
    interp_feather_px = 12.0
    w_cubic = np.clip((ref_semi_a - dist_from_center) / interp_feather_px, 0.0, 1.0)
    warped_composed = w_cubic * warped_cubic + (1.0 - w_cubic) * warped_linear
    warped_composed = np.clip(warped_composed, 0.0, 1.0).astype(np.float32)

    sharp_current = _laplacian_var_central(warped_current, ref_cx, ref_cy, ref_semi_a)
    sharp_composed = _laplacian_var_central(warped_composed, ref_cx, ref_cy, ref_semi_a)
    sharp_raw = _laplacian_var_central(raw_lum, target_cx, target_cy, own_semi_i)

    print(f"\nraw (own geometry) sharpness:        {sharp_raw:.6e}")
    print(f"current (2-pass: affine+warp) sharpness: {sharp_current:.6e}")
    print(f"composed (1-pass) sharpness:          {sharp_composed:.6e}")
    print(f"composed/current ratio: {sharp_composed/sharp_current:.4f}")

    max_diff = float(np.abs(warped_current - warped_composed).max())
    mean_diff = float(np.abs(warped_current - warped_composed).mean())
    print(f"\nmax abs diff current vs composed: {max_diff:.5f}")
    print(f"mean abs diff current vs composed: {mean_diff:.5f}")

    # Save crops for visual comparison
    Path("scratch_composed_remap_crops").mkdir(exist_ok=True)
    for name, img in [("current", warped_current), ("composed", warped_composed)]:
        vis = np.clip(img / (np.percentile(img, 99.5) + 1e-9), 0, 1)
        vis_u8 = (vis * 255).astype(np.uint8)
        crop = vis_u8[int(ref_cy-100):int(ref_cy+100), int(ref_cx-100):int(ref_cx+150)]
        crop = cv2.resize(crop, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(f"scratch_composed_remap_crops/{name}.png", crop)
    print("\nCrops written to scratch_composed_remap_crops/")


if __name__ == "__main__":
    main()
