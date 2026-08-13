"""Visual diagnostic (per the approved plan, Phase "verify before trusting
the algebra"): render compute_ring_occlusion_weight()'s foreground/background
split on a real Saturn reference frame, so the sign/direction can be checked
by eye against what's actually visible before wiring this into production.

Not production code -- one-off diagnostic script.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.modules import image_io
from pipeline.modules.derotation import (
    _SATURN_RING_INNER_REQ,
    _SATURN_RING_OUTER_REQ,
    compute_ring_occlusion_weight,
    find_disk_center,
)

OUT = Path("/tmp/ring_occlusion_overlay")
OUT.mkdir(exist_ok=True)


def analyze(name, lum, pole_pa_deg, sub_observer_lat_deg):
    h, w = lum.shape
    cx, cy, semi_a, semi_b, _ = find_disk_center(lum)
    weight = compute_ring_occlusion_weight(h, w, cx, cy, semi_a, semi_b, pole_pa_deg, sub_observer_lat_deg)

    print(f"--- {name} ---")
    print(f"cx,cy={cx:.2f},{cy:.2f} semi_a={semi_a:.2f} semi_b={semi_b:.2f} "
          f"pole_pa={pole_pa_deg:.2f} B={sub_observer_lat_deg:.3f}")
    overlap = weight > 0.0
    # Recompute the raw footprint region (globe ellipse ∩ ring annulus)
    # independently of `weight` itself, so the diagnostic doesn't rely on
    # the same code path it's trying to verify for "where is the region at
    # all" -- only for "how is it split".
    sin_b = abs(math.sin(math.radians(sub_observer_lat_deg)))
    inner_a = semi_a * _SATURN_RING_INNER_REQ; inner_b = inner_a * sin_b
    outer_a = semi_a * _SATURN_RING_OUTER_REQ; outer_b = outer_a * sin_b
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    ang = math.radians(pole_pa_deg)
    ca, sa = math.cos(ang), math.sin(ang)
    ddx, ddy = xx - cx, yy - cy
    xr = ddx * ca + ddy * sa
    yr = -ddx * sa + ddy * ca
    in_globe = (xr / semi_a) ** 2 + (yr / semi_b) ** 2 <= 1.0
    in_outer = (xr / outer_a) ** 2 + (yr / max(outer_b, 1e-6)) ** 2 <= 1.0
    in_inner = (xr / inner_a) ** 2 + (yr / max(inner_b, 1e-6)) ** 2 <= 1.0
    footprint = in_globe & in_outer & ~in_inner

    print(f"footprint(globe∩annulus) px={footprint.sum()}  "
          f"foreground(w>0.5) px={(weight>0.5).sum()}  "
          f"background(w<=0.5, in footprint) px={(footprint & (weight<=0.5)).sum()}")

    disp = (255 * np.clip(lum / (np.percentile(lum, 99.5) + 1e-9), 0, 1)).astype(np.uint8)
    disp_bgr = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)

    # Globe ellipse (green)
    cv2.ellipse(disp_bgr, (int(cx), int(cy)), (int(semi_a), int(semi_b)),
                pole_pa_deg, 0, 360, (0, 255, 0), 1, cv2.LINE_AA)

    # Ring inner/outer ellipses (cyan)
    for req_ratio in (_SATURN_RING_INNER_REQ, _SATURN_RING_OUTER_REQ):
        ring_a = semi_a * req_ratio
        ring_b = ring_a * sin_b
        cv2.ellipse(disp_bgr, (int(cx), int(cy)), (int(ring_a), int(ring_b)),
                    pole_pa_deg, 0, 360, (255, 255, 0), 1, cv2.LINE_AA)

    # Occlusion weight heatmap overlay, evaluated over the WHOLE footprint
    # region regardless of weight value: red=foreground(exclude, weight->1),
    # blue=background(hidden behind globe, normal rotation, weight->0).
    heat = np.zeros_like(disp_bgr)
    heat[..., 2] = np.where(footprint, weight * 255, 0).astype(np.uint8)          # red = foreground weight
    heat[..., 0] = np.where(footprint, (1.0 - weight) * 255, 0).astype(np.uint8)  # blue = background weight
    blended = disp_bgr.copy()
    blended[footprint] = cv2.addWeighted(disp_bgr, 0.5, heat, 0.5, 0)[footprint]

    cv2.imwrite(str(OUT / f"{name}_overlay.png"), blended)
    return dict(cx=cx, cy=cy, semi_a=semi_a, semi_b=semi_b, weight=weight)


if __name__ == "__main__":
    d = json.load(open("Saturn_Data/step04_derotated/window_01/derotation_log.json"))
    pole_pa = d["filters"]["IR"]["pole_pa_deg"]
    sub_obs = d["filters"]["IR"]["sub_observer_lat_deg"]

    img = image_io.read_tif("Saturn_Data/step02_lucky_stack/2026-08-07-1621_8-U-IR-Sat_ser_crop_lucky.tif")
    lum = img.astype(np.float64) if img.ndim == 2 else img.mean(axis=2)
    analyze("w01_IR_raw", lum, pole_pa, sub_obs)

    d5 = json.load(open("Saturn_Data/step04_derotated/window_05/derotation_log.json"))
    pole_pa5 = d5["filters"]["R"]["pole_pa_deg"]
    sub_obs5 = d5["filters"]["R"]["sub_observer_lat_deg"]
    img5 = image_io.read_tif("Saturn_Data/step02_lucky_stack/2026-08-07-1652_7-U-R-Sat_ser_crop_lucky.tif") \
        if Path("Saturn_Data/step02_lucky_stack/2026-08-07-1652_7-U-R-Sat_ser_crop_lucky.tif").exists() else None
    print("\nDone. Overlays in", OUT)
