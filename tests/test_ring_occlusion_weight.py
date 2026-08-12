"""Regression tests for compute_ring_occlusion_weight() (Saturn ring
foreground/background occlusion fix, 2026-08-11 -- see
project_derotation_ring_occlusion_fix memory and the approved plan in
current_review.md's wake).

Replaces test_ring_crossing_mask.py's coverage for compute_ring_crossing_mask
(deleted -- that function only tested 2D footprint overlap between the ring
annulus and the globe silhouette, with no notion of which side is nearer the
camera, so it excluded atmosphere de-rotation over the ring's hidden
(background) far side too, not just its true (foreground) near side).

compute_ring_occlusion_weight() keeps the same analytic-geometry philosophy
(no image content examined) but adds a closed-form line-of-sight depth
comparison: a ring-plane point (phi=0) at screen position (xr, yr) has depth
-yr / tan(B), independent of the ring radius; the globe's own near-surface
depth at the same screen position is the same formula spherical_derotation_
warp() itself uses. Whichever is nearer to the camera wins; the transition is
feathered by the depth difference itself, not an arbitrary pixel distance.

Run directly: python3 tests/test_ring_occlusion_weight.py
Or via pytest: pytest tests/test_ring_occlusion_weight.py -v
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.modules.derotation import (
    _SATURN_RING_INNER_REQ,
    _oblate_ortho_forward,
    compute_ring_occlusion_weight,
)

H, W = 300, 300
CX, CY = 150.0, 150.0
DISK_SEMI_A = 66.0
DISK_SEMI_B = 59.5  # ~Saturn's true 0.9021 oblateness at this semi_a


def test_low_tilt_has_both_foreground_and_background():
    """REGRESSION GUARD: at a low, realistic Saturn tilt (matching this
    session's real data, B=-11.07 deg), the footprint overlap must contain
    BOTH a genuine foreground region (weight>0.5) and a genuine background
    region (weight<0.5) -- this is exactly the split the old boolean mask
    could never produce (it always treated the whole footprint as
    foreground). Confirmed on real data this session: ~50/50 split,
    window_01/IR, 3394 overlap px."""
    weight = compute_ring_occlusion_weight(H, W, CX, CY, DISK_SEMI_A, DISK_SEMI_B, 0.0, -11.07)
    n_fg = int((weight > 0.5).sum())
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    dy = yy - CY
    # Background pixels get weight exactly 0.0 outside the feather zone, so
    # "distinct from foreground" must be checked within the ring-crossing
    # band itself, not just weight<0.5 anywhere in the image.
    band = (np.abs(dy) < 30.0) & (np.abs(xx - CX) < DISK_SEMI_A)
    n_bg_in_band = int(((weight < 0.5) & band).sum())
    assert n_fg > 0, "expected some genuine foreground pixels"
    assert n_bg_in_band > 0, "expected some genuine background pixels distinct from foreground"


def test_foreground_side_sign_flips_with_b_sign():
    """REGRESSION GUARD (pure algebra, no empirical assumptions): for a ring
    point (r, lam, phi=0), depth = r*cos(lam)*cos(B) and Y = -r*cos(lam)*
    sin(B), so depth = -Y/tan(B) -- Y and depth share the SAME sign when
    tan(B)<0 (B<0, e.g. this session's real data) and OPPOSITE signs when
    tan(B)>0 (B>0). This is a direct algebraic consequence of the already-
    validated _oblate_ortho_forward formula (see test_matches_oblate_ortho_
    forward_ring_depth below), independent of any assumption about which
    side "looks like" ring material in a real image -- an earlier version of
    this test tried to cross-check against a real-data shadow-darkening
    finding from a different investigation and got the expected sign
    backwards, because ring-SHADOW geometry (sub-solar latitude) and ring-
    OCCLUSION geometry (sub-observer latitude, what this function computes)
    are different physical questions that happen to share superficial
    structure -- don't conflate them again.

    Deep in the annulus, away from the limb-proximity edge effects near
    |xr| close to the footprint boundary (there, depth_globe drops toward 0
    and even the true background side can tip into "foreground" locally,
    which is physically real, not a bug -- restrict the sample to small
    |xr| to test the core classification cleanly)."""
    core_xr = 15.0
    for B_deg, pole_pa in [(-11.07, -7.0), (11.07, -7.0)]:
        weight = compute_ring_occlusion_weight(H, W, CX, CY, DISK_SEMI_A, DISK_SEMI_B, pole_pa, B_deg)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        ang = math.radians(pole_pa)
        cos_a, sin_a = math.cos(ang), math.sin(ang)
        dx, dy = xx - CX, yy - CY
        xr = dx * cos_a + dy * sin_a
        yr = -dx * sin_a + dy * cos_a
        near_side = (np.abs(yr - 22.0) < 1.0) & (np.abs(xr) < core_xr)
        far_side = (np.abs(yr + 22.0) < 1.0) & (np.abs(xr) < core_xr)
        tan_b = math.tan(math.radians(B_deg))
        if tan_b < 0:
            # yr>0 shares depth's sign -> foreground; yr<0 -> background.
            assert weight[near_side].mean() > 0.9, f"B={B_deg}: expected yr>0 foreground"
            assert weight[far_side].mean() < 0.1, f"B={B_deg}: expected yr<0 background"
        else:
            assert weight[near_side].mean() < 0.1, f"B={B_deg}: expected yr>0 background"
            assert weight[far_side].mean() > 0.9, f"B={B_deg}: expected yr<0 foreground"


def test_pole_on_tilt_no_occlusion_region():
    """REGRESSION GUARD: at a tilt far beyond Saturn's real range (B=85 deg,
    near pole-on), the ring's inner edge clears the globe entirely -- same
    early-exit as the old compute_ring_crossing_mask -- weight must be
    all-zero."""
    weight = compute_ring_occlusion_weight(H, W, CX, CY, DISK_SEMI_A, DISK_SEMI_B, 0.0, 85.0)
    assert not (weight > 0.0).any(), "expected no occlusion region at a near-pole-on tilt"
    assert weight.shape == (H, W)


def test_near_zero_b_falls_back_to_conservative_full_exclusion():
    """Near-exact edge-on (B~0) is numerically unstable for depth_ring=-yr/
    tan(B) (division blows up right where it matters) and physically
    ambiguous -- must fall back to the old, safe behaviour (treat the whole
    footprint as foreground/excluded) rather than risk a wrong comparison."""
    weight = compute_ring_occlusion_weight(H, W, CX, CY, DISK_SEMI_A, DISK_SEMI_B, 0.0, 1e-6)
    # Footprint region (if any) must be fully excluded, not partially split.
    if (weight > 0.0).any():
        occupied = weight[weight > 0.0]
        assert np.allclose(occupied, 1.0), "expected conservative full-exclusion fallback near B=0"


def test_matches_oblate_ortho_forward_ring_depth():
    """REGRESSION GUARD (algebra cross-check): a synthetic ring point at
    phi=0 (equatorial plane), projected via the general _oblate_ortho_
    forward primitive (already validated by this session's Jupiter 3D
    reprojection work), must have the SAME depth as this module's
    closed-form shortcut depth_ring = -yr / tan(B) at the same screen
    position -- catches any algebra mismatch between the two.
    """
    B = 15.0
    pole_pa = 20.0
    req_px = DISK_SEMI_A * _SATURN_RING_INNER_REQ

    for lam_deg in (10.0, 45.0, 90.0, 135.0, 200.0, 300.0):
        lam = math.radians(lam_deg)
        dx, dy, depth_expected = _oblate_ortho_forward(0.0, lam, B, pole_pa, req_px, req_px)
        # Recover yr the same way compute_ring_occlusion_weight does: undo
        # the pole_pa rotation on the projected screen offset.
        ang = math.radians(pole_pa)
        cos_a, sin_a = math.cos(ang), math.sin(ang)
        yr = -dx * sin_a + dy * cos_a
        depth_closed_form = -yr / math.tan(math.radians(B))
        assert abs(depth_closed_form - depth_expected) < 1e-6, (
            f"lam={lam_deg}: closed-form depth {depth_closed_form} != "
            f"_oblate_ortho_forward depth {depth_expected}"
        )


def test_feather_is_continuous_across_boundary():
    """The transition between foreground and background must be a smooth
    ramp (over _RING_DEPTH_FEATHER_PX of depth difference), not a hard
    step -- this is the whole point of the 2026-08-11 seam fix surviving
    into the occlusion-aware version."""
    weight = compute_ring_occlusion_weight(H, W, CX, CY, DISK_SEMI_A, DISK_SEMI_B, 0.0, -11.07)
    # Along a vertical scan through the crossing band, adjacent-pixel steps
    # should be small and gradual, not jump straight from 0 to 1.
    col = weight[:, int(CX)]
    nonzero_region = np.flatnonzero(col > 0.0)
    if nonzero_region.size > 2:
        diffs = np.abs(np.diff(col[nonzero_region.min():nonzero_region.max() + 1]))
        assert diffs.max() < 0.5, f"expected a smooth feather, got a step of {diffs.max():.3f}"


if __name__ == "__main__":
    test_low_tilt_has_both_foreground_and_background()
    print("low tilt has both foreground and background: OK")
    test_foreground_side_sign_flips_with_b_sign()
    print("foreground side sign flips with B sign: OK")
    test_pole_on_tilt_no_occlusion_region()
    print("near-pole-on tilt has no occlusion region: OK")
    test_near_zero_b_falls_back_to_conservative_full_exclusion()
    print("near-zero B conservative fallback: OK")
    test_matches_oblate_ortho_forward_ring_depth()
    print("matches _oblate_ortho_forward ring depth: OK")
    test_feather_is_continuous_across_boundary()
    print("feather is continuous: OK")
    print("\nAll checks passed.")
