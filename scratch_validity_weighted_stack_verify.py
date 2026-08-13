"""Real-data verification of the 2026-08-13 coverage-aware (validity-
weighted) de-rotation stacking fix (see the approved plan at
dreamy-squishing-perlis.md).

Runs Saturn window_01 R and IR through the REAL production derotate_window()
end-to-end, once with the fix's validity weighting active (current code) and
once with it forcibly disabled (monkeypatch quality_weighted_stack to drop
valid_masks -- reproduces exact pre-fix behaviour without reverting code),
then compares the ansa-axis radial profile in bands consistent with this
session's established methodology:
  r/semi_a < 0.95   : must be unchanged (invalid fraction measured at 0% here)
  0.95-1.05         : main improvement band (non-reference stale content excluded)
  1.05-1.15         : reference frame's own limb/PSF tail must be preserved
  > 1.15            : background / no new cutoff-seam check

Also confirms real use_true_reprojection=True is actually in effect for this
session (per ~/.astropipe/session.json) and saves crops for visual review.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from functools import partial
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.modules import image_io
from pipeline.modules import derotation
from pipeline.modules.derotation import derotate_window, find_disk_center

STEP02_DIR = Path("Saturn_Data/step02_lucky_stack")
WINDOWS_JSON = Path("Saturn_Data/step03_quality/windows.json")
STEP04_DIR = Path("Saturn_Data/step04_derotated")

FILTERS = ["R", "IR"]


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


_original_qws = derotation.quality_weighted_stack


def _qws_force_no_validity(*args, **kwargs):
    kwargs["valid_masks"] = None
    return _original_qws(*args, **kwargs)


def _run(window_idx: int, force_no_validity: bool):
    data = json.load(open(WINDOWS_JSON))
    windows = {w["window_index"]: w for w in data["selected_windows"]}
    window = windows[window_idx]
    log_path = STEP04_DIR / f"window_{window_idx:02d}" / "derotation_log.json"
    window_log = json.load(open(log_path))
    hydrated = _hydrate_window(window)

    results = {}
    for filt in FILTERS:
        if filt not in window["per_filter"]:
            continue
        flog = window_log["filters"][filt]
        ctx = mock.patch.object(derotation, "quality_weighted_stack", _qws_force_no_validity) \
            if force_no_validity else mock.patch.object(derotation, "quality_weighted_stack", _original_qws)
        with ctx:
            window_results = derotate_window(
                hydrated,
                required_filters=[filt],
                period_hours=flog["period_hours"],
                warp_scale=flog["warp_scale"],
                align=flog["align_enabled"],
                normalize_brightness=flog["normalize_brightness"],
                min_quality_threshold=flog["min_quality_threshold"],
                pole_pa_deg=flog["pole_pa_deg"],
                color_mode=False,
                flip_direction=flog["flip_direction"],
                weight_power=flog["weight_power"],
                use_true_reprojection=True,
                sub_observer_lat_deg=flog["sub_observer_lat_deg"],
                true_polar_equatorial_ratio=0.9021,
                has_rings=flog.get("has_rings", True),
                out_dir=None,
            )
        out_path, log_dict = window_results[filt]
        assert "error" not in log_dict, log_dict.get("error")
        assert log_dict.get("use_true_reprojection") is True, "use_true_reprojection not active!"
        results[filt] = log_dict
    return results


def _radial_profile_ansa(img2d: np.ndarray, cx: float, cy: float, semi_a: float,
                          pole_pa_deg: float, r_over_a_range=(0.90, 1.35), n=90):
    """Sample along the ansa (perpendicular-to-pole) axis, both directions,
    averaged -- matches this session's established methodology."""
    ansa_angle_rad = np.radians(pole_pa_deg + 90.0)
    dirx, diry = np.cos(ansa_angle_rad), np.sin(ansa_angle_rad)
    r_over_a = np.linspace(*r_over_a_range, n)
    vals = []
    for roa in r_over_a:
        r_px = roa * semi_a
        samples = []
        for sign in (+1, -1):
            x = cx + sign * dirx * r_px
            y = cy + sign * diry * r_px
            if 0 <= int(y) < img2d.shape[0] - 1 and 0 <= int(x) < img2d.shape[1] - 1:
                samples.append(float(cv2.getRectSubPix(img2d.astype(np.float32), (1, 1), (x, y))[0, 0]))
        if samples:
            vals.append(np.mean(samples))
        else:
            vals.append(np.nan)
    return r_over_a, np.array(vals)


def main():
    window_idx = 1
    print(f"=== Running window_{window_idx:02d} WITH validity weighting (fix active) ===")
    with_fix = _run(window_idx, force_no_validity=False)
    print(f"=== Running window_{window_idx:02d} WITHOUT validity weighting (pre-fix behaviour) ===")
    without_fix = _run(window_idx, force_no_validity=True)

    Path("scratch_validity_weighted_stack_crops").mkdir(exist_ok=True)

    STEP04_WIN = STEP04_DIR / f"window_{window_idx:02d}"
    for filt in FILTERS:
        if filt not in with_fix:
            continue
        # Re-run once more each, this time saving the actual TIF, since the
        # above calls used out_dir=None to keep the comparison isolated from
        # disk state. Reuse the same monkeypatch pattern.
        flog_common = json.load(open(STEP04_WIN / "derotation_log.json"))["filters"][filt]
        data = json.load(open(WINDOWS_JSON))
        windows = {w["window_index"]: w for w in data["selected_windows"]}
        hydrated = _hydrate_window(windows[window_idx])

        tmp_out = Path("scratch_validity_weighted_stack_tmp") / filt
        tmp_out.mkdir(parents=True, exist_ok=True)
        for tag, force_no_validity in [("with_fix", False), ("without_fix", True)]:
            ctx = mock.patch.object(derotation, "quality_weighted_stack", _qws_force_no_validity) \
                if force_no_validity else mock.patch.object(derotation, "quality_weighted_stack", _original_qws)
            out_dir = tmp_out / tag
            out_dir.mkdir(parents=True, exist_ok=True)
            with ctx:
                window_results = derotate_window(
                    hydrated,
                    required_filters=[filt],
                    period_hours=flog_common["period_hours"],
                    warp_scale=flog_common["warp_scale"],
                    align=flog_common["align_enabled"],
                    normalize_brightness=flog_common["normalize_brightness"],
                    min_quality_threshold=flog_common["min_quality_threshold"],
                    pole_pa_deg=flog_common["pole_pa_deg"],
                    color_mode=False,
                    flip_direction=flog_common["flip_direction"],
                    weight_power=flog_common["weight_power"],
                    use_true_reprojection=True,
                    sub_observer_lat_deg=flog_common["sub_observer_lat_deg"],
                    true_polar_equatorial_ratio=0.9021,
                    has_rings=flog_common.get("has_rings", True),
                    out_dir=out_dir,
                )
            out_path, log_dict = window_results[filt]
            img = image_io.read_tif(out_path)
            img2d = img if img.ndim == 2 else img.mean(axis=2).astype(np.float32)
            cx, cy = log_dict["frames"][0]["disk_center_px"]
            semi_a = log_dict["frames"][0]["disk_radius_px"]
            pole_pa = log_dict["pole_pa_deg"]

            r_over_a, profile = _radial_profile_ansa(img2d, cx, cy, semi_a, pole_pa)
            if tag == "with_fix":
                profile_with = profile
            else:
                profile_without = profile

            # Save a crop for visual review
            vis = np.clip(img2d / (np.percentile(img2d, 99.5) + 1e-9), 0, 1)
            vis_u8 = (vis * 255).astype(np.uint8)
            half = int(semi_a * 1.4)
            y0, y1 = max(0, int(cy - half)), int(cy + half)
            x0, x1 = max(0, int(cx - half)), int(cx + half)
            crop = vis_u8[y0:y1, x0:x1]
            crop = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(f"scratch_validity_weighted_stack_crops/{filt}_{tag}.png", crop)

        print(f"\n--- {filt} radial profile (ansa axis, r/semi_a) ---")
        print(f"{'r/semi_a':>10} {'without_fix':>14} {'with_fix':>12} {'delta':>10}")
        for roa, pw, pwo in zip(r_over_a, profile_with, profile_without):
            band = ""
            if roa < 0.95:
                band = "[<0.95]"
            elif roa < 1.05:
                band = "[0.95-1.05]"
            elif roa < 1.15:
                band = "[1.05-1.15]"
            else:
                band = "[>1.15]"
            print(f"{roa:10.3f} {pwo:14.5f} {pw:12.5f} {pw-pwo:+10.5f}  {band}")

    print("\nCrops written to scratch_validity_weighted_stack_crops/")


if __name__ == "__main__":
    main()
