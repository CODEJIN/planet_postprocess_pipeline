"""A/B real-pipeline verification of the ring-aware wavelet sharpening mask
fix (2026-08-13) on window_01 IR/R -- calls the REAL pipeline.steps.
wavelet_master.run() (step05), not a hand-rolled reimplementation, feeding
it a results_04-shaped dict built from the real step04 output already on
disk plus the real derotation_log.json for this window.

Root cause fixed: sharpen_disk_aware()'s feather mask was zero beyond the
fitted disk radius, so step05 applied zero wavelet sharpening gain to
Saturn's entire ring system (r/semi_a 1.24-2.27), while step07 (plain,
unmasked sharpen()) sharpened the rings fully -- this, not stacking or
registration, is what the re-investigation workflow found explains the
Cassini Division vanishing in step05 vs step07.

A/B: monkeypatch wavelet.sharpen_disk_aware/sharpen_color_disk_aware to
force extra_rx=None (== old behaviour) for the "before" run, real function
for "after", exactly the monkeypatch-real-production-function A/B pattern
already used earlier this session (see scratch_cassini_scale_ab.py).
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.config import PipelineConfig
from pipeline.modules import wavelet
from pipeline.modules.derotation import find_disk_center
from pipeline.steps import wavelet_master

STEP04_DIR = Path("Saturn_Data/step04_derotated")
WINDOW_INDEX = 1
FILTERS = ["IR", "R"]

_real_sharpen_disk_aware = wavelet.sharpen_disk_aware


def _forced_no_extra(image, cx, cy, radius, **kwargs):
    kwargs = dict(kwargs)
    kwargs["extra_rx"] = None
    kwargs["extra_ry"] = None
    kwargs["extra_angle"] = None
    return _real_sharpen_disk_aware(image, cx, cy, radius, **kwargs)


def _build_results_04(window_index: int, filters):
    win_dir = STEP04_DIR / f"window_{window_index:02d}"
    window_log = json.load(open(win_dir / "derotation_log.json"))
    outputs = {}
    logs = {}
    for filt in filters:
        flog = window_log["filters"][filt]
        tif_candidates = list(win_dir.glob(f"{filt}_derotated.tif"))
        if not tif_candidates:
            tif_candidates = list(win_dir.glob(f"{filt}*.tif"))
        outputs[filt] = tif_candidates[0]
        logs[filt] = flog
    return {
        "window_index": window_index,
        "center_time": window_log.get("center_time", "2026-01-01T00:00:00Z"),
        "outputs": outputs,
        "log": logs,
    }


def main():
    win = _build_results_04(WINDOW_INDEX, FILTERS)
    print("has_rings per filter:", {f: win["log"][f].get("has_rings") for f in FILTERS})
    print("pole_pa_deg:", {f: win["log"][f].get("pole_pa_deg") for f in FILTERS})
    print("sub_observer_lat_deg:", {f: win["log"][f].get("sub_observer_lat_deg") for f in FILTERS})

    config = PipelineConfig()
    config.filters = FILTERS
    config.wavelet.auto_params = True  # matches real GUI (gui/main_window.py)
    config.save_step05 = True

    crop_dir = Path("scratch_wavelet_ring_mask_crops")
    crop_dir.mkdir(exist_ok=True)

    results = {}
    tmpdir = tempfile.mkdtemp()
    for tag, patch_target in [("before", _forced_no_extra), ("after", None)]:
        config.output_base_dir = Path(tmpdir) / tag
        if patch_target is not None:
            with mock.patch.object(wavelet, "sharpen_disk_aware", patch_target):
                out = wavelet_master.run(config, {"windows": [win]})
        else:
            out = wavelet_master.run(config, {"windows": [win]})
        results[tag] = out

    for filt in FILTERS:
        for tag in ["before", "after"]:
            win_results = results[tag][f"window_{WINDOW_INDEX:02d}"]
            png_path = None
            for p, f in win_results:
                if f == filt:
                    png_path = p
            if png_path is None or not png_path.exists():
                print(f"{filt} {tag}: no output")
                continue
            img = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"{filt} {tag}: failed to read {png_path}")
                continue
            lum = img.astype(np.float32) / (65535.0 if img.dtype == np.uint16 else 255.0)
            if lum.ndim == 3:
                lum = lum.mean(axis=2)
            cx, cy, rx, ry, angle = find_disk_center(lum)
            print(f"{filt} {tag}: cx={cx:.1f} cy={cy:.1f} rx={rx:.2f}")

            half = int(rx * 2.4)
            y0, y1 = max(0, int(cy - half * 0.4)), min(lum.shape[0], int(cy + half * 0.4))
            x0, x1 = max(0, int(cx - half)), min(lum.shape[1], int(cx + half))
            crop = lum[y0:y1, x0:x1]
            crop_norm = np.clip(crop / (np.percentile(crop, 99.5) + 1e-9), 0, 1)
            crop_u8 = (crop_norm * 255).astype(np.uint8)
            crop_u8 = cv2.resize(crop_u8, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(crop_dir / f"window01_{filt}_{tag}.png"), crop_u8)

    print(f"\nCrops written to {crop_dir}/")


if __name__ == "__main__":
    main()
