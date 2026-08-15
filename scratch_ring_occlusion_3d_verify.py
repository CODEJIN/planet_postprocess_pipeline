"""Real-data verification of ring-occlusion-in-3D-reprojection (2026-08-15),
using the REAL production derotate_window() -> wavelet_master.run() path
(per feedback_ab_test_via_real_pipeline -- no hand-rolled harness, no direct
derotate_filter() calls).

Background: has_rings=True's ring-occlusion mask (compute_ring_occlusion_
weight, validated 2026-08-11) was only ever wired into the legacy linear
warp. This session's actual production Saturn config (~/.astropipe/
session.json) has use_true_reprojection=True AND has_rings=True
simultaneously -- so that validated fix was silently inert. Fixed by adding
compute_ring_occlusion_weight_3d() (same physics, correct depth convention
for the 3D path) and wiring it into spherical_derotation_warp_3d().

This script runs Saturn window_01 (IR and R -- both logged ring_crosses_
disk=True in the real derotation_log.json) through derotate_window() twice,
has_rings=True vs has_rings=False, both with use_true_reprojection=True
(today's real setting), and:
  1. confirms has_rings now actually changes the output (it silently did
     NOT before this fix)
  2. confirms the change is localized near the ring-crossing band, not a
     global perturbation
  3. runs both through wavelet_master.run() and saves crops for the
     mandatory visual check (same crop convention as scratch_edge_
     extension_verify.py / scratch_coverage_sharpening_verify.py)
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
from pipeline.modules.derotation import derotate_window
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


def _run_step04_then_step05(wi: int, filt: str, true_polar_ratio: float,
                             has_rings: bool, tmpdir: Path):
    step02 = Path("Saturn_Data/step02_lucky_stack")
    step04 = Path("Saturn_Data/step04_derotated")
    windows_json = Path("Saturn_Data/step03_quality/windows.json")

    data = json.load(open(windows_json))
    windows = {w["window_index"]: w for w in data["selected_windows"]}
    window = windows[wi]
    window_log = json.load(open(step04 / f"window_{wi:02d}/derotation_log.json"))
    hydrated = _hydrate(window, step02)
    flog = window_log["filters"][filt]
    print(f"    [source log] pole_pa={flog['pole_pa_deg']} B={flog.get('sub_observer_lat_deg')} "
          f"flip_direction={flog['flip_direction']} ring_crosses_disk(logged)={flog.get('ring_crosses_disk')}")

    out_dir04 = tmpdir / "step04"
    out_dir04.mkdir(parents=True, exist_ok=True)
    results = derotate_window(
        hydrated, required_filters=[filt],
        period_hours=flog["period_hours"], warp_scale=flog["warp_scale"],
        align=flog["align_enabled"], normalize_brightness=flog["normalize_brightness"],
        min_quality_threshold=0.0, pole_pa_deg=flog["pole_pa_deg"],
        color_mode=False, flip_direction=flog["flip_direction"],
        weight_power=flog["weight_power"], has_rings=has_rings,
        sub_observer_lat_deg=flog.get("sub_observer_lat_deg", 0.0),
        use_true_reprojection=True, true_polar_equatorial_ratio=true_polar_ratio,
        flip_pole_axis=False,  # this session's real setting (~/.astropipe/session.json)
        out_dir=out_dir04,
    )
    out_path, log = results[filt]
    assert out_path is not None, log

    config = PipelineConfig()
    config.output_base_dir = tmpdir / "pipeline_out"
    config.save_step05 = True

    results_04 = {"windows": [{
        "window_index": wi,
        "center_time": window["center_time"],
        "outputs": {filt: out_path},
        "log": {filt: log},
    }]}
    step05_results = wavelet_master.run(config, results_04)
    png_path = next(p for p, f in step05_results[f"window_{wi:02d}"] if f == filt)
    return image_io.read_tif(out_path), image_io.read_png(png_path), log


def main():
    out_crops = Path("scratch_ring_occlusion_3d_crops")
    out_crops.mkdir(exist_ok=True)

    cases = [(1, "IR", 0.9021), (1, "R", 0.9021)]

    for wi, filt, ratio in cases:
        print(f"\n=== Saturn_Data window_{wi:02d}/{filt} ===")
        with tempfile.TemporaryDirectory() as tmp_off, tempfile.TemporaryDirectory() as tmp_on:
            step04_off, png_off, log_off = _run_step04_then_step05(
                wi, filt, ratio, has_rings=False, tmpdir=Path(tmp_off),
            )
            step04_on, png_on, log_on = _run_step04_then_step05(
                wi, filt, ratio, has_rings=True, tmpdir=Path(tmp_on),
            )

        assert log_off["ring_crosses_disk"] is False
        assert log_on["ring_crosses_disk"] is True

        cx, cy = log_on["frames"][0]["disk_center_px"]
        semi_a = log_on["frames"][0]["disk_radius_px"]

        diff04 = np.abs(step04_on.astype(np.float64) - step04_off.astype(np.float64))
        print(f"  [step04 stack] has_rings on vs off: mean|diff|={diff04.mean():.3e} "
              f"max|diff|={diff04.max():.3e} changed_px={int((diff04 > 1e-4).sum())}")
        assert diff04.max() > 1e-4, (
            "has_rings made NO difference to the step04 stack -- the ring mask "
            "is still not reaching spherical_derotation_warp_3d()"
        )

        # Localization check: the change should be concentrated within the
        # logged ring-crossing band, not scattered uniformly across the
        # whole frame (a real localized physical effect, not noise/a
        # different unrelated bug).
        yy, xx = np.mgrid[:diff04.shape[0], :diff04.shape[1]].astype(np.float64)
        r = np.hypot(xx - cx, yy - cy)
        near_disk = (r > 0.3 * semi_a) & (r < 1.3 * semi_a)
        far_from_disk = r > 1.6 * semi_a
        mean_near = diff04[near_disk].mean() if near_disk.any() else 0.0
        mean_far = diff04[far_from_disk].mean() if far_from_disk.any() else 0.0
        print(f"  mean|diff| near disk (0.3-1.3*semi_a)={mean_near:.3e} "
              f"vs far background (>1.6*semi_a)={mean_far:.3e}")
        assert mean_far < mean_near, "expected the change to be concentrated near the disk, not in the background"

        diff05 = np.abs(png_on.astype(np.float32) - png_off.astype(np.float32))
        print(f"  [step05 sharpened PNG] mean|diff|={diff05.mean():.3e} max|diff|={diff05.max():.3e}")

        for tag, png in [("off", png_off), ("on", png_on)]:
            vis = np.clip(png.astype(np.float32) / (np.percentile(png, 99.7) + 1e-9), 0, 1)
            vis_u8 = (vis * 255).astype(np.uint8)
            half = int(semi_a * 1.4)
            y0, y1 = max(0, int(cy - half)), int(cy + half)
            x0, x1 = max(0, int(cx - half)), int(cx + half)
            crop = vis_u8[y0:y1, x0:x1]
            cv2.imwrite(str(out_crops / f"w{wi:02d}_{filt}_{tag}.png"), crop)

        diff_vis = np.clip(diff05 / (diff05.max() + 1e-9) * 255, 0, 255).astype(np.uint8)
        half = int(semi_a * 1.4)
        y0, y1 = max(0, int(cy - half)), int(cy + half)
        x0, x1 = max(0, int(cx - half)), int(cx + half)
        cv2.imwrite(str(out_crops / f"w{wi:02d}_{filt}_diff.png"), diff_vis[y0:y1, x0:x1])

    print(f"\nCrops written to {out_crops}/")


if __name__ == "__main__":
    main()
