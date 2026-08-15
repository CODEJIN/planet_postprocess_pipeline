"""Regression tests for the overshoot clamp (`overshoot_clamp_radius_px`),
added 2026-08-15.

Root cause this addresses: wavelet unsharp-masking a genuinely steep edge
(most notably the disk limb) rings (Gibbs-phenomenon-like overshoot/
undershoot) regardless of how precisely a spatial feather mask's gain=0
point is placed -- four prior attempts this session (coverage-aware gain
reduction, flat-fill edge extension, gradient-aware edge extension, and a
robust ellipse refit for a more accurate mask boundary) all worked in the
INPUT/MASK domain and all hit the same gray-halo/white-rim trade-off. Per
explicit user ruling (feedback_white_rim_is_critical_defect memory), the
white-rim/overshoot class of artifact is a critical defect, not a tolerable
trade-off -- see project_ring_limb_ringing_bug memory for the full history.

This is a different, OUTPUT-domain mechanism: clamp the final sharpened
pixel to the local min/max of the real, pre-sharpen image within a small
neighborhood (`wavelet._local_min_max`). A pixel exceeding every real
nearby value is definitionally invented by the filter -- clamping it
removes exactly that overshoot on a hard edge (confirmed below).

UPDATE (same day, after further investigation): this does NOT preserve
real graded detail the way it was originally hoped to -- see
test_clamp_significantly_reduces_real_detail_enhancement below and
WaveletConfig.master_overshoot_clamp_radius_px's docstring. Kept as tested,
safe (default off), documented-limitation code rather than reverted --
matches this project's convention of keeping validated-but-insufficient
opt-in features (e.g. master_ring_extension_enabled) with an honest
"not recommended" docstring instead of deleting working code.

Run directly: python3 tests/test_overshoot_clamp.py
Or via pytest: pytest tests/test_overshoot_clamp.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.modules.wavelet import sharpen, sharpen_disk_aware, sharpen_color_disk_aware

H, W = 200, 200
CX, CY = 100.0, 100.0
DISK_RADIUS = 60.0
AMOUNTS = [200.0, 200.0, 200.0, 0.0, 0.0, 0.0]


def _hard_edge_disk(disk_val: float = 0.7, bg_val: float = 0.1) -> np.ndarray:
    """A genuine step-function disk (NO blur at all) -- deliberately
    provokes unsharp-mask ringing, unlike the Gaussian-blurred synthetic
    disks used elsewhere in this test suite (those are built to AVOID
    triggering wavelet edge effects; this one is built to trigger them)."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    r = np.sqrt((xx - CX) ** 2 + (yy - CY) ** 2)
    return np.where(r <= DISK_RADIUS, disk_val, bg_val).astype(np.float32)


def _textured_disk(seed: int = 0, amp: float = 0.12) -> np.ndarray:
    """Softly-blurred disk with fine random texture -- real, legitimate
    graded detail (not a hard edge) that sharpening is SUPPOSED to enhance,
    used to confirm the clamp doesn't defeat real detail enhancement."""
    import cv2
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    r = np.sqrt((xx - CX) ** 2 + (yy - CY) ** 2)
    disk = (r <= DISK_RADIUS).astype(np.float64) * 0.6
    texture = amp * rng.standard_normal((H, W)) * (r <= DISK_RADIUS)
    img = np.clip(disk + texture, 0.0, 1.0).astype(np.float32)
    return cv2.GaussianBlur(img, (5, 5), 1.2)


def test_radius_zero_is_unchanged_default_behaviour():
    """overshoot_clamp_radius_px=0.0 (the default) must be bit-identical to
    omitting it entirely, for both sharpen() and sharpen_disk_aware() --
    every existing caller is provably unaffected by this new parameter."""
    img = _hard_edge_disk()

    out_default = sharpen(img, amounts=AMOUNTS)
    out_explicit = sharpen(img, amounts=AMOUNTS, overshoot_clamp_radius_px=0.0)
    np.testing.assert_array_equal(out_default, out_explicit)

    out_da_default = sharpen_disk_aware(img, CX, CY, DISK_RADIUS + 30, amounts=AMOUNTS)
    out_da_explicit = sharpen_disk_aware(
        img, CX, CY, DISK_RADIUS + 30, amounts=AMOUNTS, overshoot_clamp_radius_px=0.0,
    )
    np.testing.assert_array_equal(out_da_default, out_da_explicit)


def test_hard_edge_disk_overshoots_without_clamp():
    """Sanity check that the test setup actually reproduces the artifact --
    otherwise a later 'no overshoot' assertion would be vacuous. Uses plain
    sharpen() (no disk-aware mask at all), matching this session's own
    established A/B/C/D finding that a hard edge rings under global uniform
    sharpening regardless of any spatial mask."""
    img = _hard_edge_disk()
    out = sharpen(img, amounts=AMOUNTS, overshoot_clamp_radius_px=0.0)

    band = (slice(int(CY - DISK_RADIUS - 10), int(CY - DISK_RADIUS + 10)), slice(0, W))
    true_max_near_edge = float(img[band].max())
    overshoot = float(out[band].max()) - true_max_near_edge
    assert overshoot > 0.01, (
        f"expected the hard edge to produce real overshoot (test setup sanity check), "
        f"got overshoot={overshoot:.4f}"
    )


def test_clamp_eliminates_overshoot_on_hard_edge():
    """Same input as above, clamp enabled: output must never exceed the
    real image's own local min/max within the clamp radius, anywhere."""
    img = _hard_edge_disk()
    radius = 3.0
    out = sharpen(img, amounts=AMOUNTS, overshoot_clamp_radius_px=radius)

    from pipeline.modules.wavelet import _local_min_max
    lo, hi = _local_min_max(img, radius)
    tol = 1e-4
    assert (out <= hi + tol).all(), (
        f"output exceeds local max by up to {float((out - hi).max()):.5f} "
        f"even with overshoot_clamp_radius_px={radius}"
    )
    assert (out >= lo - tol).all(), (
        f"output undershoots local min by up to {float((lo - out).max()):.5f} "
        f"even with overshoot_clamp_radius_px={radius}"
    )


def test_clamp_significantly_reduces_real_detail_enhancement():
    """KNOWN LIMITATION, characterized here rather than asserted away:
    the clamp does NOT preserve most legitimate local-contrast enhancement
    of real (non-hard-edge) graded texture -- it crushes the vast majority
    of it, even at a small radius. Confirmed independently twice on
    2026-08-15: once here with a hand-built belt-like texture, and again by
    an isolated investigation workflow using the real production sharpen()
    with the real calibrated gain table (this test's earlier version
    asserted the OPPOSITE -- that >70% would be retained -- which was wrong
    and is why this test now documents the actual, measured behavior).

    Root cause (see wavelet._local_min_max's docstring and
    project_ring_limb_ringing_bug memory): wavelet sharpening's whole point
    at fine levels is to push a pixel further from its immediate neighbors
    than the original data was -- which a small-neighborhood local min/max
    clamp cannot distinguish from unsharp-mask overshoot, because both are
    definitionally "value exceeds the immediate neighborhood's original
    range." This is not a bug in this implementation; it is a structural
    mismatch between a single-scale "prevent overshoot" clamp (the classic
    textbook technique this was modeled on) and a multi-scale wavelet
    sharpener whose finest levels operate at or below the clamp's own
    scale. This is why overshoot_clamp_radius_px is NOT recommended as a
    general-purpose fix -- see WaveletConfig.master_overshoot_clamp_radius_px
    docstring."""
    img = _textured_disk()
    interior = (
        (np.mgrid[0:H, 0:W][1].astype(np.float64) - CX) ** 2
        + (np.mgrid[0:H, 0:W][0].astype(np.float64) - CY) ** 2
    ) <= (DISK_RADIUS * 0.8) ** 2

    var_orig = float(np.var(img[interior]))
    out_noclamp = sharpen(img, amounts=AMOUNTS, overshoot_clamp_radius_px=0.0)
    out_clamped = sharpen(img, amounts=AMOUNTS, overshoot_clamp_radius_px=3.0)
    var_noclamp = float(np.var(out_noclamp[interior]))
    var_clamped = float(np.var(out_clamped[interior]))

    boost_noclamp = var_noclamp - var_orig
    boost_clamped = var_clamped - var_orig
    assert boost_noclamp > 0, "sharpening should increase interior variance (sanity check)"
    retained_fraction = boost_clamped / boost_noclamp
    assert retained_fraction < 0.3, (
        f"expected the clamp to significantly suppress real detail enhancement "
        f"(documented limitation) -- retained {retained_fraction:.2%}, which is "
        f"higher than previously measured; if this genuinely improved, update "
        f"WaveletConfig.master_overshoot_clamp_radius_px's docstring accordingly "
        f"instead of just loosening this assertion "
        f"(boost_noclamp={boost_noclamp:.5f}, boost_clamped={boost_clamped:.5f})"
    )


def test_sharpen_color_disk_aware_passes_through():
    """The color path threads overshoot_clamp_radius_px straight through to
    the L-channel sharpen_disk_aware() call -- default (0.0) is
    bit-identical, and a real radius measurably changes the output (proves
    the parameter isn't silently dropped somewhere in the RGB<->Lab
    round-trip)."""
    gray = _hard_edge_disk()
    rgb = np.stack([gray, gray, gray], axis=2).astype(np.float32)

    out_default = sharpen_color_disk_aware(rgb, CX, CY, DISK_RADIUS + 30, amounts=AMOUNTS)
    out_explicit = sharpen_color_disk_aware(
        rgb, CX, CY, DISK_RADIUS + 30, amounts=AMOUNTS, overshoot_clamp_radius_px=0.0,
    )
    np.testing.assert_array_equal(out_default, out_explicit)

    out_clamped = sharpen_color_disk_aware(
        rgb, CX, CY, DISK_RADIUS + 30, amounts=AMOUNTS, overshoot_clamp_radius_px=3.0,
    )
    assert not np.array_equal(out_default, out_clamped), (
        "overshoot_clamp_radius_px had no effect on sharpen_color_disk_aware's output"
    )


if __name__ == "__main__":
    test_radius_zero_is_unchanged_default_behaviour()
    print("radius=0 unchanged default behaviour: OK")
    test_hard_edge_disk_overshoots_without_clamp()
    print("hard edge disk overshoots without clamp (sanity check): OK")
    test_clamp_eliminates_overshoot_on_hard_edge()
    print("clamp eliminates overshoot on hard edge: OK")
    test_clamp_significantly_reduces_real_detail_enhancement()
    print("clamp significantly reduces real detail enhancement (documented limitation): OK")
    test_sharpen_color_disk_aware_passes_through()
    print("sharpen_color_disk_aware passes through: OK")
    print("\nAll checks passed.")
