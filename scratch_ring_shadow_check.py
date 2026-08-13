"""Empirical check: is there a dark band/arc on the Saturn globe (derotated
window_01/IR, plus raw step02 lucky frame) that is NOT already covered by
compute_ring_crossing_mask()? Candidate explanation: ring shadow.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from pipeline.modules import image_io
from pipeline.modules.derotation import find_disk_center, compute_ring_crossing_mask

OUT = Path("/tmp/ring_shadow_check")
OUT.mkdir(exist_ok=True)


def to_lum(img):
    return img.astype(np.float64) if img.ndim == 2 else img.mean(axis=2)


def analyze(name, lum, pole_pa, sub_obs):
    h, w = lum.shape
    cx, cy, semi_a, semi_b, _ = find_disk_center(lum)
    ring_mask = compute_ring_crossing_mask(h, w, cx, cy, semi_a, semi_b, pole_pa, sub_obs)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    ang = math.radians(pole_pa)
    ca, sa = math.cos(ang), math.sin(ang)
    dx, dy = xx - cx, yy - cy
    xr = dx * ca + dy * sa
    yr = -dx * sa + dy * ca
    # elliptical radius (deprojected, 0..1 at globe limb)
    rell = np.sqrt((xr / semi_a) ** 2 + (yr / semi_b) ** 2)
    in_globe = rell <= 1.0
    central55 = np.sqrt(dx ** 2 + dy ** 2) < 0.55 * semi_a

    # Radial profile (elliptical rings), excluding known ring-crossing pixels,
    # to see limb-darkening baseline; then residual = lum - profile(rell)
    bins = np.linspace(0, 1.0, 41)
    idx = np.clip((rell * 40).astype(int), 0, 39)
    prof = np.full(40, np.nan)
    for b in range(40):
        sel = in_globe & (idx == b) & (~ring_mask)
        if sel.sum() > 5:
            prof[b] = np.median(lum[sel])
    # fill nan by interpolation
    good = ~np.isnan(prof)
    prof[~good] = np.interp(np.flatnonzero(~good), np.flatnonzero(good), prof[good])
    residual = np.zeros_like(lum)
    residual[in_globe] = lum[in_globe] - prof[idx[in_globe]]

    print(f"--- {name} ---")
    print(f"cx,cy={cx:.2f},{cy:.2f} semi_a={semi_a:.2f} semi_b={semi_b:.2f}")
    print(f"ring_mask px={ring_mask.sum()} globe px={in_globe.sum()} "
          f"frac={ring_mask.sum()/in_globe.sum():.3f}")
    print(f"central55 radius_px={0.55*semi_a:.2f} overlap(ring,central55)="
          f"{(ring_mask & central55).sum()} / central55={central55.sum()} "
          f"({(ring_mask & central55).sum()/central55.sum():.3f})")

    # Find darkest residual region OUTSIDE ring_mask, inside globe
    outside = in_globe & (~ring_mask)
    resid_out = np.where(outside, residual, np.nan)
    # smooth a bit to find coherent dark blobs, not single-pixel noise
    resid_smooth = cv2.GaussianBlur(np.nan_to_num(residual, nan=0.0).astype(np.float32), (5, 5), 0)
    resid_smooth_out = np.where(outside, resid_smooth, np.nan)
    flat_idx = np.nanargmin(resid_smooth_out)
    py, px = np.unravel_index(flat_idx, resid_smooth_out.shape)
    val = resid_smooth_out[py, px]
    # describe location: distance from center, angle, and rell (deprojected radius)
    dist = math.hypot(px - cx, py - cy)
    print(f"darkest non-ring-crossing residual pixel: (x={px},y={py}) "
          f"resid={val:.4f} dist_from_center_px={dist:.2f} rell={rell[py,px]:.3f} "
          f"in_central55={bool(central55[py,px])}")

    # Also report mean residual in central55 minus ring_mask (i.e. the region
    # that actually gets scored for sharpness) vs rest of globe.
    c55_excl_ring = central55 & (~ring_mask)
    rest_excl_ring = in_globe & (~central55) & (~ring_mask)
    print(f"mean residual central55(excl ring)={np.nanmean(np.where(c55_excl_ring, residual, np.nan)):.4f} "
          f"mean residual rest(excl ring)={np.nanmean(np.where(rest_excl_ring, residual, np.nan)):.4f}")

    # Save visualization
    disp = (255 * np.clip(lum / (np.percentile(lum[in_globe], 99) + 1e-9), 0, 1)).astype(np.uint8)
    disp_bgr = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
    overlay = disp_bgr.copy()
    overlay[ring_mask] = (0, 0, 255)  # red = known ring-crossing exclusion
    cv2.circle(overlay, (int(round(cx)), int(round(cy))), int(round(0.55 * semi_a)), (0, 255, 0), 1)
    blended = cv2.addWeighted(disp_bgr, 0.5, overlay, 0.5, 0)
    cv2.imwrite(str(OUT / f"{name}_overlay.png"), blended)

    resid_disp = residual.copy()
    resid_disp[~in_globe] = 0
    rmax = np.nanpercentile(np.abs(resid_disp[in_globe]), 99) + 1e-9
    resid_norm = np.clip((resid_disp / rmax) * 127 + 128, 0, 255).astype(np.uint8)
    resid_color = cv2.applyColorMap(resid_norm, cv2.COLORMAP_JET)
    resid_color[~in_globe] = 0
    cv2.imwrite(str(OUT / f"{name}_residual.png"), resid_color)
    return dict(cx=cx, cy=cy, semi_a=semi_a, semi_b=semi_b, ring_mask=ring_mask,
                in_globe=in_globe, central55=central55, residual=residual)


if __name__ == "__main__":
    d = json.load(open("Saturn_Data/step04_derotated/window_01/derotation_log.json"))
    pole_pa = d["filters"]["IR"]["pole_pa_deg"]
    sub_obs = d["filters"]["IR"]["sub_observer_lat_deg"]

    img = image_io.read_tif("Saturn_Data/step04_derotated/window_01/IR_derotated.tif")
    analyze("w01_IR_derotated", to_lum(img), pole_pa, sub_obs)

    raw = image_io.read_tif("Saturn_Data/step02_lucky_stack/2026-08-07-1621_8-U-IR-Sat_ser_crop_lucky.tif")
    analyze("raw_IR_1621_8", to_lum(raw), pole_pa, sub_obs)

    # also try R filter window_01 and CH4 window_05 for variety
    d5 = json.load(open("Saturn_Data/step04_derotated/window_05/derotation_log.json"))
    if "R" in d5["filters"]:
        pole_pa5 = d5["filters"]["R"]["pole_pa_deg"]
        sub_obs5 = d5["filters"]["R"]["sub_observer_lat_deg"]
        img5 = image_io.read_tif("Saturn_Data/step04_derotated/window_05/R_derotated.tif")
        analyze("w05_R_derotated", to_lum(img5), pole_pa5, sub_obs5)
