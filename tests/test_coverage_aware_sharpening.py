"""Regression tests for coverage-aware sharpening + S0/S_L pixel-adaptive
blend (2026-08-15).

Background: this session found the multi-frame de-rotation stack is
measurably blurrier than a single best frame across BOTH the disk interior
(unrelated to coverage -- fixed separately by frame_sharpness_central()
selection, see test_frame_sharpness_central.py) and specifically at the
limb, where a diagnosed Saturn asymmetric ringing artifact was traced to
find_disk_center()'s ellipse fit having a measured ~0.5-0.9px asymmetric
error vs the TRUE photometric limb (ring signal biasing the Otsu contour
fit). An external review proposed two remedies, implemented here:

  1. compute_frame_coverage_mask()/derotate_filter's coverage aggregation
     n(x): a per-pixel signal of how much of the final stack's blend mass
     at that pixel is genuine rotation-valid content (not identity-
     fallback from a stale epoch). Feeds:
  2. wavelet.coverage_to_confidence()/sharpen_disk_aware's confidence_map:
     reduces sharpening gain where n(x) is low.
  3. derotate_filter's s0_sl_blend_enabled: blends the stack toward the
     reference frame's own (dt=0) rendering wherever n(x) is low, so the
     result is never worse than the single-reference-frame baseline there
     by construction.

A near-identical per-pixel validity signal was built once already this
session (2026-08-13), found buggy in two ways, and fully reverted (zero
trace -- `git log -S"return_valid"` on derotation.py returns nothing):
(a) a symmetric signed-distance feather gave invalid pixels up to ~50%
weight -- fixed here by not feathering per-frame at all (summing several
frames' hard per-pixel booleans already grades the field; only the FINAL
aggregate gets one Gaussian blur); (b) conflating "off the modeled globe
domain entirely" (Saturn's rings/background, dt-independent) with "genuine
rotation invalidity" (dt-dependent) collapsed non-reference frames' weight
across the entire ring system -- fixed here by keeping on_globe_domain and
rotation_valid as separate checks (see compute_frame_coverage_mask's
docstring) and specifically regression-tested below.

Run directly: python3 tests/test_coverage_aware_sharpening.py
Or via pytest: pytest tests/test_coverage_aware_sharpening.py -v
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
    compute_frame_coverage_mask,
    derotate_filter,
)
from pipeline.modules.wavelet import coverage_to_confidence, sharpen_disk_aware

SIZE = 300


# ── (a) compute_frame_coverage_mask() correctness ───────────────────────────

def _coverage_scenario():
    """Same synthetic setup as test_reprojection.py::
    test_no_hard_ring_at_invalid_boundary -- already verified to produce a
    real far-side-invalid band at r in [0.985, 1.005]*disk_r."""
    h, w = 300, 300
    cx, cy, disk_r, period_h = 150.0, 150.0, 120.0, 10.0
    dt_sec = period_h * 3600.0 * 0.03
    kwargs = dict(
        cx=cx, cy=cy, disk_radius_px=disk_r, period_hours=period_h,
        sub_observer_lat_deg=20.0, pole_pa_deg=15.0,
        polar_equatorial_ratio_true=0.935, scale=1.0,
    )
    return h, w, dt_sec, cx, cy, disk_r, kwargs


def test_compute_frame_coverage_mask_invalid_band():
    h, w, dt_sec, cx, cy, disk_r, kwargs = _coverage_scenario()
    cov = compute_frame_coverage_mask(h, w, dt_sec, **kwargs)
    assert cov.dtype == bool
    assert cov.shape == (h, w)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    r = np.hypot(xx - cx, yy - cy)

    band = (r >= 0.985 * disk_r) & (r <= 1.005 * disk_r)
    assert band.sum() > 100
    assert not cov[band].all(), "expected some False (invalid) pixels in the known invalid band"

    interior = r < 0.5 * disk_r
    assert cov[interior].all(), "expected the disk interior to be fully valid"


def test_compute_frame_coverage_mask_off_domain_always_valid():
    """REGRESSION GUARD for the reverted attempt's exact bug: pixels
    entirely off the modeled globe domain (e.g. where Saturn's rings or
    background sky would be) must always report True (full coverage),
    never treated as rotation-invalid."""
    h, w, dt_sec, cx, cy, disk_r, kwargs = _coverage_scenario()
    cov = compute_frame_coverage_mask(h, w, dt_sec, **kwargs)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    r = np.hypot(xx - cx, yy - cy)
    far_off_domain = r > 1.3 * disk_r
    assert far_off_domain.sum() > 100
    assert cov[far_off_domain].all(), (
        "off-globe-domain pixels (rings/background) must always report full "
        "coverage -- they must never be treated as rotation-stale"
    )


# ── (b) sharpen_disk_aware's confidence_map ─────────────────────────────────

def _textured_disk(cx: float, cy: float, r: float, amp: float = 0.15, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    disk = (rr < r).astype(np.float64) * 0.6
    texture = amp * rng.standard_normal((SIZE, SIZE)) * (rr < r)
    return np.clip(disk + texture, 0.0, 1.0).astype(np.float32)


def test_confidence_map_none_matches_omitted():
    img = _textured_disk(150.0, 150.0, 100.0, seed=1)
    a = sharpen_disk_aware(img, 150.0, 150.0, 100.0, amounts=[200, 200, 100, 0, 0, 0])
    b = sharpen_disk_aware(img, 150.0, 150.0, 100.0, amounts=[200, 200, 100, 0, 0, 0], confidence_map=None)
    np.testing.assert_array_equal(a, b)


def test_confidence_map_zero_region_gets_no_gain():
    img = _textured_disk(150.0, 150.0, 100.0, seed=1)
    confidence = np.ones((SIZE, SIZE), dtype=np.float32)
    confidence[:, :150] = 0.0  # left half: no sharpening gain at all

    unsharpened_baseline = img  # gain=0 everywhere means output == input
    sharpened_full = sharpen_disk_aware(img, 150.0, 150.0, 100.0, amounts=[200, 200, 100, 0, 0, 0])
    sharpened_partial = sharpen_disk_aware(
        img, 150.0, 150.0, 100.0, amounts=[200, 200, 100, 0, 0, 0], confidence_map=confidence
    )

    # Left half (confidence=0): sharpened_partial must equal the ORIGINAL
    # image there (no gain applied), not the fully-sharpened version.
    left = slice(None), slice(0, 150)
    np.testing.assert_allclose(sharpened_partial[left], unsharpened_baseline[left], atol=1e-5)
    assert not np.allclose(sharpened_partial[left], sharpened_full[left], atol=1e-4), (
        "expected confidence=0 region to differ from the fully-sharpened result"
    )

    # Right half (confidence=1): must match the fully-sharpened result.
    right = slice(None), slice(150, SIZE)
    np.testing.assert_allclose(sharpened_partial[right], sharpened_full[right], atol=1e-5)


def test_confidence_map_shape_mismatch_raises():
    img = _textured_disk(150.0, 150.0, 100.0, seed=1)
    bad_shape = np.ones((10, 10), dtype=np.float32)
    try:
        sharpen_disk_aware(img, 150.0, 150.0, 100.0, confidence_map=bad_shape)
        assert False, "expected ValueError on shape mismatch"
    except ValueError:
        pass


# ── (c)/(d) derotate_filter S0/S_L blend ────────────────────────────────────

def _write_reprojection_window(tmp: Path, n_non_ref: int, dt_sec: float, t0: datetime):
    """Frames all sharing the same synthetic textured disk (so any stack
    difference is attributable to the blend/coverage mechanism, not real
    content differences), one reference (dt=0) plus n_non_ref frames all
    at the SAME large dt_sec (matching _coverage_scenario's proven invalid-
    band-producing ratio) -- driving n(x) down in the outer annulus for
    every non-reference frame simultaneously, well below the ~1.0 seen at
    the disk interior."""
    rows = []
    img = _textured_disk(150.0, 150.0, 100.0, seed=1)
    ref_path = tmp / "ref.tif"
    image_io.write_tif_16bit(img, ref_path)
    rows.append({"path": str(ref_path), "stem": "ref", "timestamp": t0, "norm_score": 0.9})
    for i in range(n_non_ref):
        path = tmp / f"frame_{i}.tif"
        image_io.write_tif_16bit(img, path)
        rows.append({
            "path": str(path), "stem": f"frame_{i}",
            "timestamp": t0 + timedelta(seconds=dt_sec), "norm_score": 0.9,
        })
    return rows


_REPROJ_KW = dict(
    period_hours=10.0,
    use_true_reprojection=True,
    sub_observer_lat_deg=20.0,
    pole_pa_deg=15.0,
    true_polar_equatorial_ratio=0.935,
)


def test_s0_sl_blend_recovers_reference_in_low_coverage_region():
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        t0 = datetime(2026, 1, 1, 0, 0, 0)
        dt_sec = 10.0 * 3600.0 * 0.03  # matches _coverage_scenario's ratio
        rows = _write_reprojection_window(tmp, n_non_ref=3, dt_sec=dt_sec, t0=t0)

        # Ground-truth S0: the reference frame alone, put through the same
        # pipeline (dt=0, trivially its own identity rendering).
        s0_ref, _ = derotate_filter(rows[:1], t0, align=False, **_REPROJ_KW)

        stacked_unblended, log_unblended = derotate_filter(
            rows, t0, align=True, compute_coverage_map=True,
            s0_sl_blend_enabled=False, **_REPROJ_KW,
        )
        stacked_blended, log_blended = derotate_filter(
            rows, t0, align=True, compute_coverage_map=True,
            s0_sl_blend_enabled=True, **_REPROJ_KW,
        )
        assert log_blended["s0_sl_blend_applied"] is True
        assert log_unblended["s0_sl_blend_applied"] is False

        cx, cy = log_blended["frames"][0]["disk_center_px"]
        semi_a = log_blended["frames"][0]["disk_radius_px"]
        yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
        r = np.hypot(xx - cx, yy - cy)

        # Outer annulus (low coverage for non-reference frames, per
        # _coverage_scenario's proven band location) -- blended must sit
        # measurably closer to S0 than the unblended stack does.
        outer = (r >= 0.90 * semi_a) & (r <= 1.0 * semi_a)
        assert outer.sum() > 50
        dist_blended = np.abs(stacked_blended[outer].astype(np.float64) - s0_ref[outer].astype(np.float64)).mean()
        dist_unblended = np.abs(stacked_unblended[outer].astype(np.float64) - s0_ref[outer].astype(np.float64)).mean()
        assert dist_blended < dist_unblended, (
            f"expected blended stack to be closer to S0 in the low-coverage "
            f"outer annulus: dist_blended={dist_blended:.5f} "
            f"dist_unblended={dist_unblended:.5f}"
        )

        # Disk interior (full coverage) -- blended and unblended should be
        # nearly identical (alpha -> 1 there).
        interior = r < 0.5 * semi_a
        assert interior.sum() > 50
        np.testing.assert_allclose(
            stacked_blended[interior], stacked_unblended[interior], atol=0.03
        )


def test_coverage_computed_without_blend_leaves_stack_unchanged():
    """compute_coverage_map=True with s0_sl_blend_enabled=False must
    compute/log n(x) (e.g. for step05 to consume) WITHOUT ever perturbing
    the stack itself."""
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        t0 = datetime(2026, 1, 1, 0, 0, 0)
        dt_sec = 10.0 * 3600.0 * 0.03
        rows = _write_reprojection_window(tmp, n_non_ref=3, dt_sec=dt_sec, t0=t0)

        stacked_a, log_a = derotate_filter(rows, t0, align=True, **_REPROJ_KW)
        stacked_b, log_b = derotate_filter(
            rows, t0, align=True, compute_coverage_map=True,
            s0_sl_blend_enabled=False, **_REPROJ_KW,
        )
        np.testing.assert_array_equal(stacked_a, stacked_b)
        assert log_a["coverage_computed"] is False
        assert log_b["coverage_computed"] is True
        assert log_b["coverage_mean"] is not None


def test_coverage_map_default_off_matches_omitted():
    """Omitting compute_coverage_map/s0_sl_blend_enabled must be byte-
    identical to explicitly passing their defaults (False, False)."""
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        t0 = datetime(2026, 1, 1, 0, 0, 0)
        dt_sec = 10.0 * 3600.0 * 0.03
        rows = _write_reprojection_window(tmp, n_non_ref=3, dt_sec=dt_sec, t0=t0)

        stacked_omitted, log_omitted = derotate_filter(rows, t0, align=True, **_REPROJ_KW)
        stacked_explicit, log_explicit = derotate_filter(
            rows, t0, align=True, compute_coverage_map=False,
            s0_sl_blend_enabled=False, **_REPROJ_KW,
        )
        np.testing.assert_array_equal(stacked_omitted, stacked_explicit)
        assert log_omitted["coverage_map"] is None
        assert log_explicit["coverage_map"] is None


def test_coverage_inert_when_use_true_reprojection_false():
    """The coverage signal doesn't exist for the linear warp path --
    requesting either feature without use_true_reprojection=True must be a
    silent no-op at the derotate_filter level (the hard warning/disable
    lives one layer up in derotate_stack.py; derotate_filter's own
    _do_coverage gate is what's tested directly here)."""
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        t0 = datetime(2026, 1, 1, 0, 0, 0)
        rows = _write_reprojection_window(tmp, n_non_ref=3, dt_sec=100.0, t0=t0)

        stacked_a, log_a = derotate_filter(rows, t0, align=True, period_hours=10.0)
        stacked_b, log_b = derotate_filter(
            rows, t0, align=True, period_hours=10.0,
            compute_coverage_map=True, s0_sl_blend_enabled=True,
        )
        np.testing.assert_array_equal(stacked_a, stacked_b)
        assert log_b["coverage_computed"] is False
        assert log_b["coverage_map"] is None


if __name__ == "__main__":
    test_compute_frame_coverage_mask_invalid_band()
    print("compute_frame_coverage_mask invalid band: OK")
    test_compute_frame_coverage_mask_off_domain_always_valid()
    print("compute_frame_coverage_mask off-domain always valid: OK")
    test_confidence_map_none_matches_omitted()
    print("confidence_map=None matches omitted: OK")
    test_confidence_map_zero_region_gets_no_gain()
    print("confidence_map=0 region gets no gain: OK")
    test_confidence_map_shape_mismatch_raises()
    print("confidence_map shape mismatch raises: OK")
    test_s0_sl_blend_recovers_reference_in_low_coverage_region()
    print("S0/S_L blend recovers reference in low-coverage region: OK")
    test_coverage_computed_without_blend_leaves_stack_unchanged()
    print("coverage computed without blend leaves stack unchanged: OK")
    test_coverage_map_default_off_matches_omitted()
    print("coverage map default-off matches omitted: OK")
    test_coverage_inert_when_use_true_reprojection_false()
    print("coverage inert when use_true_reprojection=False: OK")
    print("\nAll checks passed.")
