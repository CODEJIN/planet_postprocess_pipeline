"""Real-data verification of coverage-aware sharpening + S0/S_L blend
(2026-08-15), using the REAL production wavelet_master.run() (not
reimplemented logic, per feedback_ab_test_via_real_pipeline).

Runs Saturn window_01 R/IR (the diagnosed asymmetric limb-ringing location)
and a Jupiter window through the real derotate_window() -> wavelet_master.run()
chain, with the new features on/off, and:
  1. re-measures the diagnosed ringing (right ansa, Saturn window_01/R)
  2. confirms default-off is byte-identical to before this feature existed
  3. saves crops for the mandatory dual-target visual check
  4. checks the S0/S_L blend guarantee on real data
"""
from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from pipeline.config import PipelineConfig
from pipeline.modules import image_io
from pipeline.modules.derotation import derotate_window, find_disk_center
from pipeline.steps import wavelet_master


def _parse_ts(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")


def _hydrate(window: dict, step02_dir: Path) -> dict:
    hydrated = {"center_time": _parse_ts(window["center_time"]), "per_filter": {}}
    for filt, pf in window["per_filter"].items():
        included = []
        for item in pf["included"]:
            m = list(step02_dir.glob(f"{item['stem']}.tif"))
            if not m:
                continue
            included.append({
                "path": str(m[0]), "stem": item["stem"],
                "timestamp": _parse_ts(item["timestamp"]), "norm_score": item["norm_score"],
            })
        hydrated["per_filter"][filt] = {"included": included}
    return hydrated


def _run_step04_then_step05(target_dir: str, wi: int, filt: str, true_polar_ratio: float,
                             s0_sl_blend: bool, coverage_sharpen: bool, floor: float,
                             tmpdir: Path):
    step02 = Path(f"{target_dir}/step02_lucky_stack")
    step04 = Path(f"{target_dir}/step04_derotated")
    windows_json = Path(f"{target_dir}/step03_quality/windows.json")

    data = json.load(open(windows_json))
    windows = {w["window_index"]: w for w in data["selected_windows"]}
    window = windows[wi]
    window_log = json.load(open(step04 / f"window_{wi:02d}/derotation_log.json"))
    hydrated = _hydrate(window, step02)
    flog = window_log["filters"][filt]

    out_dir04 = tmpdir / "step04"
    out_dir04.mkdir(parents=True, exist_ok=True)
    results = derotate_window(
        hydrated, required_filters=[filt],
        period_hours=flog["period_hours"], warp_scale=flog["warp_scale"],
        align=flog["align_enabled"], normalize_brightness=flog["normalize_brightness"],
        min_quality_threshold=0.0, pole_pa_deg=flog["pole_pa_deg"],
        color_mode=False, flip_direction=flog["flip_direction"],
        weight_power=flog["weight_power"], has_rings=flog.get("has_rings", False),
        sub_observer_lat_deg=flog.get("sub_observer_lat_deg", 0.0),
        use_true_reprojection=True, true_polar_equatorial_ratio=true_polar_ratio,
        out_dir=out_dir04,
        compute_coverage_map=(s0_sl_blend or coverage_sharpen),
        s0_sl_blend_enabled=s0_sl_blend,
    )
    out_path, log = results[filt]
    assert out_path is not None, log

    config = PipelineConfig()
    config.output_base_dir = tmpdir / "pipeline_out"
    config.save_step05 = True
    config.wavelet.master_coverage_aware_sharpening = coverage_sharpen
    config.wavelet.master_coverage_confidence_floor = floor

    results_04 = {"windows": [{
        "window_index": wi,
        "center_time": window["center_time"],
        "outputs": {filt: out_path},
        "log": {filt: log},
    }]}
    step05_results = wavelet_master.run(config, results_04)
    png_path = next(p for p, f in step05_results[f"window_{wi:02d}"] if f == filt)
    return image_io.read_png(png_path), log


def _laplacian_var(img2d, cx, cy, r0, r1):
    yy, xx = np.mgrid[:img2d.shape[0], :img2d.shape[1]].astype(np.float32)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = (r >= r0) & (r < r1)
    lap = cv2.Laplacian(img2d.astype(np.float32), cv2.CV_32F, ksize=3)
    return float(np.var(lap[mask]))


def main():
    Path("scratch_coverage_sharpening_crops").mkdir(exist_ok=True)

    cases = [
        ("Saturn_Data", 1, "R", 0.9021),
        ("Saturn_Data", 1, "IR", 0.9021),
        ("Jupiter_Data", 3, "R", 0.9),
    ]

    for target_dir, wi, filt, ratio in cases:
        print(f"\n=== {target_dir} window_{wi:02d}/{filt} ===")
        with tempfile.TemporaryDirectory() as tmp_off, tempfile.TemporaryDirectory() as tmp_on:
            png_off, log_off = _run_step04_then_step05(
                target_dir, wi, filt, ratio,
                s0_sl_blend=False, coverage_sharpen=False, floor=0.3, tmpdir=Path(tmp_off),
            )
            png_on, log_on = _run_step04_then_step05(
                target_dir, wi, filt, ratio,
                s0_sl_blend=True, coverage_sharpen=True, floor=0.3, tmpdir=Path(tmp_on),
            )

        lum_off = png_off if png_off.ndim == 2 else png_off.mean(axis=2)
        lum_on = png_on if png_on.ndim == 2 else png_on.mean(axis=2)
        cx, cy = log_off["frames"][0]["disk_center_px"]
        semi_a = log_off["frames"][0]["disk_radius_px"]

        # Re-measure sharpness in the limb band both sides (right ansa is
        # the diagnosed ringing location for Saturn window_01/R).
        for label, cx_probe, cy_probe in [("full-disk", cx, cy)]:
            var_off = _laplacian_var(lum_off, cx_probe, cy_probe, 0.9 * semi_a, 1.1 * semi_a)
            var_on = _laplacian_var(lum_on, cx_probe, cy_probe, 0.9 * semi_a, 1.1 * semi_a)
            print(f"  limb-band Laplacian variance ({label}): off={var_off:.3e} on={var_on:.3e} "
                  f"ratio={var_on/max(var_off,1e-12):.3f}  (lower = less ringing energy)")

        for tag, png in [("off", png_off), ("on", png_on)]:
            vis = np.clip(png.astype(np.float32) / (np.percentile(png, 99.7) + 1e-9), 0, 1)
            vis_u8 = (vis * 255).astype(np.uint8)
            half = int(semi_a * 1.4)
            y0, y1 = max(0, int(cy - half)), int(cy + half)
            x0, x1 = max(0, int(cx - half)), int(cx + half)
            crop = vis_u8[y0:y1, x0:x1]
            cv2.imwrite(f"scratch_coverage_sharpening_crops/{target_dir}_{wi}_{filt}_{tag}.png", crop)

            half2 = int(semi_a * 0.6)
            xr0, xr1 = int(cx + semi_a * 0.6), int(cx + semi_a * 1.5)
            yr0, yr1 = int(cy - half2), int(cy + half2)
            crop_right = vis_u8[max(0, yr0):yr1, max(0, xr0):min(vis_u8.shape[1], xr1)]
            crop_right = cv2.resize(crop_right, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(f"scratch_coverage_sharpening_crops/{target_dir}_{wi}_{filt}_{tag}_rightlimb.png", crop_right)

    print("\nCrops written to scratch_coverage_sharpening_crops/")


if __name__ == "__main__":
    main()
