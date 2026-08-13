"""Regression tests for raw-sharpness-based frame selection (2026-08-13).

Background: a real-data investigation this session found that the final
multi-frame de-rotation stack is measurably blurrier than the single
sharpest raw frame in the same window -- a gap present deep in the disk
interior (unrelated to de-rotation warp coverage), confirmed on both
Saturn and Jupiter, so not a target-specific or coordinate-system issue.
The existing per-frame quality metric (`norm_score`, pipeline.modules.
quality) was found to correlate ~-0.10 with real per-frame sharpness. A
controlled experiment found that selecting only the top-half of a
window's frames by a NEW metric -- Laplacian variance over the central
55% of each frame's own disk radius, no pre-denoise -- raised the median
(stack/best-raw-frame) sharpness ratio from 0.733 to 0.878 across 51 real
window x filter combos (mixed Saturn/Jupiter), with a same-frame-count
"worst half" control performing WORSE than the unfiltered baseline,
proving the effect is about which frames are kept, not how many.

See frame_sharpness_central()'s docstring in pipeline/modules/derotation.py
for the full justification.

Run directly: python3 tests/test_frame_sharpness_central.py
Or via pytest: pytest tests/test_frame_sharpness_central.py -v
"""
from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.modules import image_io
from pipeline.modules.derotation import (
    _laplacian_var_central,
    derotate_filter,
    frame_sharpness_central,
)

SIZE = 300


def _textured_disk(cx: float, cy: float, r: float, amp: float = 0.15, seed: int = 0) -> np.ndarray:
    """Synthetic Otsu-detectable disk (base brightness 0.6) with fine
    random texture inside it, so a real Laplacian-variance difference
    exists between a sharp and a Gaussian-blurred copy -- a flat, uniform
    disk would give ~0 for both and couldn't demonstrate ranking."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    disk = (rr < r).astype(np.float64) * 0.6
    texture = amp * rng.standard_normal((SIZE, SIZE)) * (rr < r)
    return np.clip(disk + texture, 0.0, 1.0).astype(np.float32)


def test_frame_sharpness_central_ranks_sharp_over_blurred():
    sharp = _textured_disk(150.0, 150.0, 100.0, seed=1)
    blurred = cv2.GaussianBlur(sharp, (0, 0), sigmaX=3.0)

    s_sharp = frame_sharpness_central(sharp)
    s_blurred = frame_sharpness_central(blurred)

    assert s_sharp is not None and s_blurred is not None
    assert s_sharp > 2.0 * s_blurred, (
        f"expected sharp ({s_sharp}) to clearly exceed blurred ({s_blurred})"
    )


def test_laplacian_var_central_none_on_tiny_mask():
    """A degenerate (too-small) region must return None, not 0.0 --
    callers must not treat an unmeasurable frame as "worst"."""
    img = np.random.default_rng(0).standard_normal((50, 50)).astype(np.float32)
    assert _laplacian_var_central(img, 25.0, 25.0, 2.0) is None
    assert _laplacian_var_central(img, 25.0, 25.0, 20.0) is not None


def _write_window(tmp: Path, seeds_and_blur: list, t0: datetime, step_sec: float = 5.0):
    """Build `included_rows` for derotate_filter(): one synthetic-disk TIF
    per (seed, is_blurred) entry, all at the same nominal geometry (so
    de-rotation warp is near-identity and doesn't confound the sharpness
    comparison), equally-spaced timestamps, identical norm_score (so
    existing norm_score-based weighting can't itself explain any result)."""
    rows = []
    for i, (seed, blur) in enumerate(seeds_and_blur):
        img = _textured_disk(150.0, 150.0, 100.0, seed=seed)
        if blur:
            img = cv2.GaussianBlur(img, (0, 0), sigmaX=3.0)
        path = tmp / f"frame_{i}.tif"
        image_io.write_tif_16bit(img, path)
        rows.append({
            "path": str(path),
            "stem": f"frame_{i}",
            "timestamp": t0 + timedelta(seconds=i * step_sec),
            "norm_score": 0.9,
        })
    return rows


def test_derotate_filter_sharpness_selection_excludes_blurred_frame():
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        t0 = datetime(2026, 1, 1, 0, 0, 0)
        rows = _write_window(tmp, [(1, False), (2, False), (3, False), (4, True)], t0)
        t_ref = t0 + timedelta(seconds=5)  # frame_1 (sharp) is closest -> reference

        _, log_off = derotate_filter(rows, t_ref, period_hours=10.0, align=True)
        blurred_log_off = next(f for f in log_off["frames"] if f["stem"] == "frame_3")
        assert blurred_log_off.get("sharpness_excluded") is None  # feature off, no-op

        stacked_on, log_on = derotate_filter(
            rows, t_ref, period_hours=10.0, align=True,
            sharpness_selection_enabled=True, sharpness_keep_fraction=0.67,
        )
        blurred_log_on = next(f for f in log_on["frames"] if f["stem"] == "frame_3")
        assert blurred_log_on["sharpness_excluded"] is True
        assert log_on["n_stacked"] == 3

        stacked_off, _ = derotate_filter(rows, t_ref, period_hours=10.0, align=True)
        sharp_on = frame_sharpness_central(stacked_on)
        sharp_off = frame_sharpness_central(stacked_off)
        assert sharp_on > sharp_off, (
            f"expected excluding the blurred frame to sharpen the stack: "
            f"on={sharp_on} off={sharp_off}"
        )


def test_derotate_filter_never_excludes_reference_frame():
    """The reference frame must never be excluded by this mechanism, even
    if it is measurably the blurriest -- mirrors the outlier-shift-
    rejection block's own non_ref_mask invariant."""
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        t0 = datetime(2026, 1, 1, 0, 0, 0)
        # frame_0 (blurred) is closest to t_ref -> becomes the reference.
        rows = _write_window(tmp, [(1, True), (2, False), (3, False), (4, False)], t0)
        t_ref = t0

        _, log_on = derotate_filter(
            rows, t_ref, period_hours=10.0, align=True,
            sharpness_selection_enabled=True, sharpness_keep_fraction=0.5,
        )
        ref_log = next(f for f in log_on["frames"] if f["stem"] == "frame_0")
        assert ref_log["align_method"] == "reference"
        assert ref_log.get("sharpness_excluded") is None  # never a candidate at all
        assert any(f["stem"] == "frame_0" for f in log_on["frames"] if not f.get("sharpness_excluded"))


def test_derotate_filter_sharpness_default_off_matches_omitted():
    """Explicitly passing sharpness_selection_enabled=False,
    sharpness_keep_fraction=1.0 must be byte-identical to omitting both --
    the feature must be a complete no-op when off."""
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        t0 = datetime(2026, 1, 1, 0, 0, 0)
        rows = _write_window(tmp, [(1, False), (2, False), (3, False), (4, True)], t0)
        t_ref = t0 + timedelta(seconds=5)

        stacked_omitted, log_omitted = derotate_filter(rows, t_ref, period_hours=10.0, align=True)
        stacked_explicit, log_explicit = derotate_filter(
            rows, t_ref, period_hours=10.0, align=True,
            sharpness_selection_enabled=False, sharpness_keep_fraction=1.0,
        )

        np.testing.assert_array_equal(stacked_omitted, stacked_explicit)
        assert log_omitted["n_stacked"] == log_explicit["n_stacked"]
        for fl_a, fl_b in zip(log_omitted["frames"], log_explicit["frames"]):
            assert fl_a["raw_sharpness"] == fl_b["raw_sharpness"]
            assert fl_a.get("sharpness_excluded") is None
            assert fl_b.get("sharpness_excluded") is None


if __name__ == "__main__":
    test_frame_sharpness_central_ranks_sharp_over_blurred()
    print("frame_sharpness_central ranks sharp over blurred: OK")
    test_laplacian_var_central_none_on_tiny_mask()
    print("_laplacian_var_central returns None on tiny mask: OK")
    test_derotate_filter_sharpness_selection_excludes_blurred_frame()
    print("derotate_filter excludes blurred frame when enabled: OK")
    test_derotate_filter_never_excludes_reference_frame()
    print("derotate_filter never excludes the reference frame: OK")
    test_derotate_filter_sharpness_default_off_matches_omitted()
    print("derotate_filter default-off matches omitted params: OK")
    print("\nAll checks passed.")
