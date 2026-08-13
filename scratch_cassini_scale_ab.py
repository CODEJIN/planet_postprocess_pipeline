"""A/B real-pipeline verification of the frame-to-frame disk-scale correction
(2026-08-12) on window_01 IR/R -- the window/filters the user specifically
flagged as having a washed-out Cassini Division in step05 vs step07.

Per feedback_ab_test_via_real_pipeline: toggles the REAL production function
(apply_shift_and_scale) via monkeypatch to force scale=1.0 (== old
translation-only behaviour) for the "before" run, then restores it for the
"after" run -- both runs go through the exact same derotate_window() call,
same code path, only the scale correction itself differs.

Quantitative metric (purely analytic masks, no image content examined to
build them): Cassini contrast in the ansa sector only (away from globe
overlap), C_CD = 1 - I_gap / ((I_A + I_B) / 2).

Also saves zoomed ansa-region PNG crops for both runs so the user can
visually confirm (per this session's established practice of never trusting
a single scalar metric alone).
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
    apply_shift_and_scale as _real_apply_shift_and_scale,
    _SATURN_RING_INNER_REQ,
    _SATURN_RING_OUTER_REQ,
)

_WCFG = WaveletConfig()


def _wavelet_sharpen_like_step05(lum: np.ndarray) -> np.ndarray:
    """Mirror wavelet_master.run()'s real per-filter call (default config,
    auto_params=False path) so the A/B comparison matches what the user
    actually sees in step05, not the soft pre-wavelet stack."""
    cx, cy, rx, ry, angle_deg = find_disk_center(lum)
    if rx < 5:
        return lum
    return wavelet.sharpen_disk_aware(
        lum, cx, cy, rx,
        levels=_WCFG.levels,
        amounts=_WCFG.master_amounts,
        power=_WCFG.master_power,
        sharpen_filter=_WCFG.master_sharpen_filter,
        edge_feather_factor=_WCFG.edge_feather_factor,
        ry=ry, angle=np.radians(angle_deg),
        expand_px=_WCFG.disk_expand_px,
        denoise_amounts=_WCFG.master_denoise_amounts,
        filter_type=_WCFG.master_filter_type,
    )

STEP02_DIR = Path("Saturn_Data/step02_lucky_stack")
STEP04_DIR = Path("Saturn_Data/step04_derotated")
WINDOWS_JSON = Path("Saturn_Data/step03_quality/windows.json")

FILTERS = ["IR", "R"]
WINDOW_INDEX = 1


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


def _force_scale_one(image, target_cx, target_cy, ref_cx, ref_cy, scale):
    return _real_apply_shift_and_scale(image, target_cx, target_cy, ref_cx, ref_cy, 1.0)


def _run(hydrated, any_flog, tag: str, out_root: Path):
    out_dir = out_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    return derotate_window(
        hydrated,
        required_filters=FILTERS,
        period_hours=any_flog["period_hours"],
        warp_scale=any_flog["warp_scale"],
        align=any_flog["align_enabled"],
        normalize_brightness=any_flog["normalize_brightness"],
        min_quality_threshold=any_flog["min_quality_threshold"],
        pole_pa_deg=any_flog["pole_pa_deg"],
        color_mode=False,
        flip_direction=any_flog["flip_direction"],
        weight_power=any_flog["weight_power"],
        has_rings=any_flog["has_rings"],
        sub_observer_lat_deg=any_flog["sub_observer_lat_deg"],
        out_dir=out_dir,
    )


def _ansa_cassini_contrast(img2d: np.ndarray, cx, cy, semi_a, semi_b, pole_pa_deg):
    """Purely analytic ansa-sector Cassini contrast: C_CD = 1 - I_gap /
    ((I_A + I_B)/2), masks built only from known ring-system geometry
    (IAU radii ratios) and disk fit -- no image content examined to build
    the masks themselves (only to average brightness inside them)."""
    h, w = img2d.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    ang = np.radians(pole_pa_deg)
    cos_a, sin_a = np.cos(ang), np.sin(ang)
    dx, dy = xx - cx, yy - cy
    xr = dx * cos_a + dy * sin_a
    yr = -dx * sin_a + dy * cos_a

    # Ansa sector: near the ring-plane's projected minor axis is where the
    # rings open up widest away from the globe; restrict to |yr| small (near
    # the ring plane's own projected line) and |xr| large (away from the
    # globe silhouette itself).
    ansa = (np.abs(yr) < 6.0) & (np.abs(xr) > semi_a * 1.05)

    # Cassini Division / A-ring / B-ring radii as fractions of the globe's
    # equatorial radius (IAU), converted to this frame's own semi_a.
    cassini_req = 117_580.0 / 60_268.0  # IAU Cassini Division radius/Saturn Req
    a_ring_inner_req = _SATURN_RING_INNER_REQ + 0.35 * (_SATURN_RING_OUTER_REQ - _SATURN_RING_INNER_REQ)
    b_ring_outer_req = cassini_req - 0.05

    r_screen = np.abs(xr) / semi_a  # radial distance in disk-radius units along the ansa line

    a_band = ansa & (np.abs(r_screen - a_ring_inner_req) < 0.03)
    b_band = ansa & (np.abs(r_screen - b_ring_outer_req) < 0.03)
    gap_band = ansa & (np.abs(r_screen - cassini_req) < 0.015)

    if a_band.sum() < 5 or b_band.sum() < 5 or gap_band.sum() < 5:
        return None, (int(a_band.sum()), int(b_band.sum()), int(gap_band.sum()))

    i_a = float(img2d[a_band].mean())
    i_b = float(img2d[b_band].mean())
    i_gap = float(img2d[gap_band].mean())
    contrast = 1.0 - i_gap / ((i_a + i_b) / 2.0)
    return contrast, (int(a_band.sum()), int(b_band.sum()), int(gap_band.sum()))


def main():
    data = json.load(open(WINDOWS_JSON))
    windows = {w["window_index"]: w for w in data["selected_windows"]}
    window = windows[WINDOW_INDEX]
    window_log = json.load(open(STEP04_DIR / f"window_{WINDOW_INDEX:02d}" / "derotation_log.json"))
    hydrated = _hydrate_window(window)
    any_flog = next(iter(window_log["filters"].values()))

    crop_dir = Path("scratch_cassini_ab_crops")
    crop_dir.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_root = Path(tmpdir)

        with mock.patch.object(derotation, "apply_shift_and_scale", _force_scale_one):
            before_results = _run(hydrated, any_flog, "before_scale1", out_root)

        after_results = _run(hydrated, any_flog, "after_scalefix", out_root)

        for filt in FILTERS:
            for tag, results in [("BEFORE(scale=1)", before_results), ("AFTER(scale-corrected)", after_results)]:
                out_path, log_dict = results[filt]
                if out_path is None:
                    print(f"{filt} {tag}: SKIP no output")
                    continue
                img = image_io.read_tif(str(out_path))
                lum_raw = img if img.ndim == 2 else img.mean(axis=2).astype(np.float32)
                # Fit geometry on the PRE-sharpen stack (reliable) and reuse
                # it for the sharpened image -- wavelet ringing at the limb
                # can otherwise throw off find_disk_center() when run on the
                # sharpened result directly (confirmed: semi_a dropped from
                # ~67px to ~56px when re-fit post-sharpen).
                cx, cy, semi_a, semi_b, _angle = find_disk_center(lum_raw)
                if semi_a < 5:
                    print(f"{filt} {tag}: SKIP disk detect failed")
                    continue
                lum = _wavelet_sharpen_like_step05(lum_raw)
                contrast, counts = _ansa_cassini_contrast(lum, cx, cy, semi_a, semi_b, any_flog["pole_pa_deg"])
                print(f"{filt} {tag}: cx={cx:.1f} cy={cy:.1f} semi_a={semi_a:.2f} "
                      f"C_CD={contrast} band_px={counts}")

                # Save a zoomed ansa crop for visual inspection.
                half = int(semi_a * 2.3)
                y0, y1 = max(0, int(cy - half * 0.4)), min(lum.shape[0], int(cy + half * 0.4))
                x0, x1 = max(0, int(cx - half)), min(lum.shape[1], int(cx + half))
                crop = lum[y0:y1, x0:x1]
                crop_norm = np.clip(crop / (np.percentile(crop, 99.5) + 1e-9), 0, 1)
                crop_u8 = (crop_norm * 255).astype(np.uint8)
                crop_u8 = cv2.resize(crop_u8, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_NEAREST)
                safe_tag = tag.split("(")[0]
                cv2.imwrite(str(crop_dir / f"window01_{filt}_{safe_tag}.png"), crop_u8)

    print(f"\nCrops written to {crop_dir}/ -- inspect visually before trusting C_CD alone.")


if __name__ == "__main__":
    main()
