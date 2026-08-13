"""Regression tests for apply_shift_and_scale() (Saturn Cassini Division
frame-to-frame disk-scale correction, 2026-08-11/12 -- see
project_derotation_ring_occlusion_fix / the approved plan for the
frame-to-frame disk-scale investigation).

apply_shift_and_scale() maps a raw frame's own measured disk centre/radius
onto the reference frame's geometry: output = ref_center + scale *
(input - target_center). An earlier draft pivoted the scale around ref_center
and then added the existing translation-only (ref_cx-target_cx, ref_cy-
target_cy) shift -- algebraically NOT equivalent except at scale=1 (caught by
an external review before shipping, see the function's own docstring). These
tests exist specifically to catch that class of affine-composition bug if it
is ever reintroduced, plus the physically-motivated Cassini Division use case.

Run directly: python3 tests/test_apply_shift_and_scale.py
Or via pytest: pytest tests/test_apply_shift_and_scale.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.modules.derotation import apply_shift, apply_shift_and_scale

H, W = 300, 300


def _blob_peak(image: np.ndarray) -> tuple:
    """Sub-pixel centroid of the single brightest blob in *image*."""
    ys, xs = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    total = image.sum()
    return (float((xs * image).sum() / total), float((ys * image).sum() / total))


def _gaussian_blob(cx: float, cy: float, sigma: float = 4.0) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    return np.exp(-r2 / (2 * sigma * sigma)).astype(np.float32)


def test_target_center_maps_to_ref_center():
    """The single most important property (this is exactly what the earlier
    affine-composition bug got wrong for scale != 1): a blob sitting at
    target_center must land at ref_center after the transform, for a
    non-trivial scale."""
    target_cx, target_cy = 130.0, 140.0
    ref_cx, ref_cy = 160.0, 155.0
    scale = 1.08
    blob = _gaussian_blob(target_cx, target_cy)
    warped = apply_shift_and_scale(blob, target_cx, target_cy, ref_cx, ref_cy, scale)
    peak = _blob_peak(warped)
    assert abs(peak[0] - ref_cx) < 0.5, f"expected x={ref_cx}, got {peak[0]}"
    assert abs(peak[1] - ref_cy) < 0.5, f"expected y={ref_cy}, got {peak[1]}"


def test_radius_maps_by_scale_factor():
    """A point offset from target_center by (r, 0) must land offset from
    ref_center by (scale*r, 0) -- this is the property that actually fixes
    the Cassini Division (radius must be corrected, not just center)."""
    target_cx, target_cy = 150.0, 150.0
    ref_cx, ref_cy = 150.0, 150.0
    scale = 1.2
    r = 40.0
    blob = _gaussian_blob(target_cx + r, target_cy)
    warped = apply_shift_and_scale(blob, target_cx, target_cy, ref_cx, ref_cy, scale)
    peak = _blob_peak(warped)
    expected_x = ref_cx + scale * r
    assert abs(peak[0] - expected_x) < 0.5, f"expected x={expected_x}, got {peak[0]}"
    assert abs(peak[1] - ref_cy) < 0.5, f"expected y={ref_cy}, got {peak[1]}"


def test_scale_one_matches_plain_translation():
    """At scale=1.0, apply_shift_and_scale must reduce to the existing
    apply_shift() translation-only behaviour (not byte-identical -- the
    interpolation path differs slightly -- but numerically very close)."""
    target_cx, target_cy = 120.0, 135.0
    ref_cx, ref_cy = 145.0, 128.0
    blob = _gaussian_blob(target_cx, target_cy, sigma=6.0)
    via_scale_fn = apply_shift_and_scale(blob, target_cx, target_cy, ref_cx, ref_cy, 1.0)
    via_plain_shift = apply_shift(blob, ref_cx - target_cx, ref_cy - target_cy)
    np.testing.assert_allclose(via_scale_fn, via_plain_shift, atol=1e-4)


def test_reference_frame_identity():
    """target_center == ref_center with scale=1.0 must be a no-op (this is
    what a reference frame's own registration reduces to)."""
    cx, cy = 150.0, 150.0
    blob = _gaussian_blob(cx, cy)
    warped = apply_shift_and_scale(blob, cx, cy, cx, cy, 1.0)
    np.testing.assert_allclose(warped, blob, atol=1e-4)


def _ring_profile(r: np.ndarray) -> np.ndarray:
    """Purely analytic concentric A/Cassini-gap/B ring brightness profile,
    smoothed with a narrow logistic edge (avoids hard-edge aliasing without
    needing any image-content-based processing -- this is synthetic test
    data, not pipeline logic)."""
    def edge(x, center, width=0.6):
        return 1.0 / (1.0 + np.exp(-(x - center) / width))

    a_ring = edge(r, 95.0) * (1.0 - edge(r, 118.0))
    cassini_gap = edge(r, 118.0) * (1.0 - edge(r, 122.0))
    b_ring = edge(r, 122.0) * (1.0 - edge(r, 140.0))
    return 0.85 * a_ring + 0.08 * cassini_gap + 0.85 * b_ring


def _make_raw_frame(cx: float, cy: float, radius_scale: float) -> np.ndarray:
    """Independent construction (no apply_shift_and_scale involved) of a
    raw frame whose ring system is centred at (cx, cy) and apparently
    radius_scale times the canonical size -- models a frame whose own
    find_disk_center() measurement would report a disk radius_scale times
    the reference frame's."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / radius_scale
    return _ring_profile(r).astype(np.float32)


def _cassini_trough_depth(stack: np.ndarray, cx: float, cy: float) -> float:
    """Ansa-only (left/right of centre, away from top/bottom to dodge any
    residual centring noise) contrast between the ring peaks and the
    Cassini gap floor -- higher is a better-preserved (less blurred) gap."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    ansa = np.abs(yy - cy) < 8.0
    a_peak = stack[(np.abs(r - 106.0) < 3.0) & ansa].mean()
    b_peak = stack[(np.abs(r - 131.0) < 3.0) & ansa].mean()
    gap_floor = stack[(np.abs(r - 120.0) < 1.5) & ansa].mean()
    return 1.0 - gap_floor / ((a_peak + b_peak) / 2.0)


def test_scale_correction_preserves_cassini_gap_better_than_translation_only():
    """The actual physical motivation: stacking frames with a real ~3%
    frame-to-frame radius variation (matching this session's measured
    1.25-2.15%, doubled here for a clearer signal-to-interpolation-noise
    margin) washes out a thin radial feature (the Cassini gap) under
    translation-only registration; correcting scale as well as translation
    must preserve it noticeably better."""
    ref_cx, ref_cy = 150.0, 150.0
    rng_offsets = [
        (0.0, 0.0, 1.0),      # reference frame itself
        (2.0, -1.5, 1.03),
        (-1.0, 2.0, 0.97),
        (1.5, 1.0, 1.02),
        (-2.0, -0.5, 0.98),
    ]

    translation_only_stack = []
    scale_corrected_stack = []
    for dx, dy, radius_scale in rng_offsets:
        cx_k, cy_k = ref_cx + dx, ref_cy + dy
        raw = _make_raw_frame(cx_k, cy_k, radius_scale)

        translation_only_stack.append(apply_shift(raw, ref_cx - cx_k, ref_cy - cy_k))

        corrected_scale = 1.0 / radius_scale
        scale_corrected_stack.append(
            apply_shift_and_scale(raw, cx_k, cy_k, ref_cx, ref_cy, corrected_scale)
        )

    translation_only_mean = np.mean(translation_only_stack, axis=0)
    scale_corrected_mean = np.mean(scale_corrected_stack, axis=0)

    depth_translation_only = _cassini_trough_depth(translation_only_mean, ref_cx, ref_cy)
    depth_scale_corrected = _cassini_trough_depth(scale_corrected_mean, ref_cx, ref_cy)

    assert depth_scale_corrected > depth_translation_only, (
        f"expected scale correction to deepen the Cassini gap contrast: "
        f"translation-only={depth_translation_only:.4f}, "
        f"scale-corrected={depth_scale_corrected:.4f}"
    )
    # Scale correction should also come close to recovering the true
    # single-frame (radius_scale=1.0) gap contrast, not just "better than
    # the alternative" -- compare against the true reference frame alone.
    reference_alone = _make_raw_frame(ref_cx, ref_cy, 1.0)
    depth_reference_alone = _cassini_trough_depth(reference_alone, ref_cx, ref_cy)
    assert depth_scale_corrected > 0.7 * depth_reference_alone, (
        f"scale-corrected stack lost too much gap contrast vs single-frame "
        f"reference: {depth_scale_corrected:.4f} vs {depth_reference_alone:.4f}"
    )


if __name__ == "__main__":
    test_target_center_maps_to_ref_center()
    print("target center maps to ref center: OK")
    test_radius_maps_by_scale_factor()
    print("radius maps by scale factor: OK")
    test_scale_one_matches_plain_translation()
    print("scale=1.0 matches plain translation: OK")
    test_reference_frame_identity()
    print("reference frame identity: OK")
    test_scale_correction_preserves_cassini_gap_better_than_translation_only()
    print("scale correction preserves Cassini gap better than translation-only: OK")
    print("\nAll checks passed.")
