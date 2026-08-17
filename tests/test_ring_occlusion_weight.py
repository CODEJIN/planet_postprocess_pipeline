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
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2

from _fixtures import textured_disk as _textured_disk
from pipeline.config import DerotationConfig
from pipeline.modules import image_io
from pipeline.modules.derotation import (
    _SATURN_RING_INNER_REQ,
    _SATURN_RING_OUTER_REQ,
    _feather_ring_foreground_boundary,
    _laplacian_var_central,
    _oblate_ortho_forward,
    _oblate_ortho_inverse,
    _ring_annulus_mask,
    _ring_registration_crop,
    compute_ring_occlusion_weight,
    compute_ring_occlusion_weight_3d,
    derotate_filter,
)

H, W = 300, 300
CX, CY = 150.0, 150.0
DISK_SEMI_A = 66.0
DISK_SEMI_B = 59.5  # ~Saturn's true 0.9021 oblateness at this semi_a
RATIO_TRUE = 0.9021  # Saturn's TRUE physical Rpol/Req (see spherical_derotation_warp_3d)


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


# ── compute_ring_occlusion_weight_3d() (2026-08-15) ─────────────────────────
#
# Wires the same physical occlusion test into the true 3D reprojection warp
# (spherical_derotation_warp_3d), which previously had no ring-awareness at
# all -- a real, currently-live gap: this session's production Saturn config
# (~/.astropipe/session.json / profiles/sat.json) has use_true_reprojection=
# True AND has_rings=True simultaneously, so the 2026-08-11 fix above was
# silently inert in practice. depth_ring needs no change (proven identical
# to _oblate_ortho_forward's own depth by test_matches_oblate_ortho_forward_
# ring_depth above); only depth_globe changes, from the linear warp's sqrt
# approximation to _oblate_ortho_inverse's own (exact, same-convention)
# depth. Two bugs found during design review, both invisible at pole_pa=0 --
# every test below uses a nonzero pole_pa for exactly that reason.

def test_matches_oblate_ortho_forward_ring_depth_3d():
    """BUG GUARD (flip_pole_axis sign): _oblate_ortho_forward negates Y
    AFTER depth is computed from the un-negated Y, so `yr` (recovered from
    screen position, same as the existing non-3D test above) is Y_USED
    (post-flip), not Y_raw. For a ring point (phi=0): depth = -Y_raw/tan(B).
    flip_pole_axis=False -> yr=Y_raw -> depth=-yr/tan(B) (matches the
    existing, flip-agnostic closed form). flip_pole_axis=True -> yr=-Y_raw
    -> depth=+yr/tan(B): the sign FLIPS. A naive port of the existing
    formula into the 3D-aware function would get this wrong for any session
    where flip_pole_axis is auto-detected True."""
    B = 15.0
    pole_pa = 20.0
    req_px = DISK_SEMI_A * _SATURN_RING_INNER_REQ

    for flip in (False, True):
        for lam_deg in (10.0, 45.0, 90.0, 135.0, 200.0, 300.0):
            lam = math.radians(lam_deg)
            dx, dy, depth_expected = _oblate_ortho_forward(
                0.0, lam, B, pole_pa, req_px, req_px, flip_pole_axis=flip
            )
            ang = math.radians(pole_pa)
            cos_a, sin_a = math.cos(ang), math.sin(ang)
            yr = -dx * sin_a + dy * cos_a
            sign = 1.0 if flip else -1.0
            depth_closed_form = sign * yr / math.tan(math.radians(B))
            assert abs(depth_closed_form - depth_expected) < 1e-6, (
                f"flip={flip} lam={lam_deg}: closed-form depth {depth_closed_form} "
                f"!= _oblate_ortho_forward depth {depth_expected}"
            )


def test_depth_globe_matches_oblate_ortho_inverse():
    """BUG GUARD (double-rotation): _oblate_ortho_inverse performs its own
    internal pole_pa un-rotation, so it MUST be called with RAW (dx, dy) =
    (xx-cx, yy-cy), never the already-rotated (xr, yr) also used here for
    the ellipse tests and depth_ring -- feeding it (xr, yr) double-rotates.
    This is invisible at pole_pa=0 (dx==xr, dy==yr there), so this test uses
    a nonzero pole_pa and explicitly confirms the double-rotated ("buggy")
    version actually disagrees with the correct one for this geometry --
    otherwise the test would pass vacuously and not actually guard anything.

    Reimplements compute_ring_occlusion_weight_3d's own logic independently
    and checks the real function's classification (weight>0.5, away from
    the feather boundary) against it, for both flip_pole_axis values (also
    exercises the sign convention from test_matches_oblate_ortho_forward_
    ring_depth_3d end-to-end, through the real function).

    Uses a larger pole_pa (-45 deg) and a polar_equatorial_ratio_true (0.6)
    further from the apparent fitted ratio (~0.9015) than this session's
    real Saturn geometry -- empirically confirmed to reliably produce a
    large double-rotation mismatch (hundreds of px); the real, more modest
    Saturn-like geometry (pole_pa=-7, ratio=0.9021) turns out to produce
    ZERO mismatch for this particular disk/ring configuration, which would
    make the bug guard below pass vacuously."""
    B = -11.07
    pole_pa = -45.0
    ratio_true = 0.6

    for flip in (False, True):
        weight = compute_ring_occlusion_weight_3d(
            H, W, CX, CY, DISK_SEMI_A, DISK_SEMI_B, pole_pa, B, ratio_true,
            flip_pole_axis=flip,
        )

        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        dx, dy = xx - CX, yy - CY
        ang = math.radians(pole_pa)
        cos_a, sin_a = math.cos(ang), math.sin(ang)
        xr = dx * cos_a + dy * sin_a
        yr = -dx * sin_a + dy * cos_a

        sin_b = abs(math.sin(math.radians(B)))
        inner_ring_semi_a = DISK_SEMI_A * _SATURN_RING_INNER_REQ
        inner_ring_semi_b = inner_ring_semi_a * sin_b
        outer_ring_semi_a = DISK_SEMI_A * _SATURN_RING_OUTER_REQ
        outer_ring_semi_b = outer_ring_semi_a * sin_b
        in_globe = (xr / DISK_SEMI_A) ** 2 + (yr / DISK_SEMI_B) ** 2 <= 1.0
        in_ring_outer = (xr / outer_ring_semi_a) ** 2 + (yr / outer_ring_semi_b) ** 2 <= 1.0
        in_ring_inner = (xr / inner_ring_semi_a) ** 2 + (yr / inner_ring_semi_b) ** 2 <= 1.0
        overlap = in_globe & (in_ring_outer & ~in_ring_inner)
        assert overlap.any()

        req_px = DISK_SEMI_A * 1.05
        rpol_px = req_px * ratio_true
        phi_correct, _lam_c, depth_globe_raw = _oblate_ortho_inverse(
            dx, dy, B, pole_pa, req_px, rpol_px, flip_pole_axis=flip,
        )
        depth_globe_correct = np.where(np.isnan(phi_correct), -np.inf, depth_globe_raw)
        sign = 1.0 if flip else -1.0
        depth_ring = sign * yr / math.tan(math.radians(B))
        expected_fg = overlap & (depth_ring > depth_globe_correct)

        # Sanity: confirm the double-rotation bug would actually give a
        # different answer here -- otherwise this test wouldn't catch it.
        phi_bug, _lam_b, depth_globe_bug_raw = _oblate_ortho_inverse(
            xr, yr, B, pole_pa, req_px, rpol_px, flip_pole_axis=flip,
        )
        depth_globe_bug = np.where(np.isnan(phi_bug), -np.inf, depth_globe_bug_raw)
        buggy_fg = overlap & (depth_ring > depth_globe_bug)
        assert int((buggy_fg != expected_fg).sum()) > 0, (
            f"flip={flip}: test geometry doesn't distinguish the double-"
            f"rotation bug -- strengthen the scenario"
        )

        core = overlap & (np.abs(depth_ring - depth_globe_correct) > 3.0)
        assert core.sum() > 50
        np.testing.assert_array_equal(weight[core] > 0.5, expected_fg[core])


def test_invalid_globe_depth_defaults_to_foreground():
    """BUG GUARD (invalid-depth sentinel): _oblate_ortho_inverse marks an
    unresolvable (no near-side solution) point with a literal depth=-1.0,
    not -inf/NaN. Comparing depth_ring directly against that raw sentinel
    would misclassify wherever depth_ring happens to fall in (-1.0, 0) --
    the opposite of the intended "unresolvable globe depth -> treat
    conservatively as foreground/occluded" policy already used by the B~0
    fallback. Using a small polar_equatorial_ratio_true (0.2, unrealistic
    for Saturn but a clean way to force the true 3D domain's polar extent
    well inside the apparent fitted ellipse) reliably produces many
    genuinely invalid points inside the footprint overlap on this
    geometry -- confirmed empirically, not asserted blindly."""
    B = -11.07
    pole_pa = -7.0
    ratio_true = 0.2

    weight = compute_ring_occlusion_weight_3d(
        H, W, CX, CY, DISK_SEMI_A, DISK_SEMI_B, pole_pa, B, ratio_true,
    )

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    dx, dy = xx - CX, yy - CY
    ang = math.radians(pole_pa)
    cos_a, sin_a = math.cos(ang), math.sin(ang)
    xr = dx * cos_a + dy * sin_a
    yr = -dx * sin_a + dy * cos_a

    sin_b = abs(math.sin(math.radians(B)))
    inner_ring_semi_a = DISK_SEMI_A * _SATURN_RING_INNER_REQ
    inner_ring_semi_b = inner_ring_semi_a * sin_b
    outer_ring_semi_a = DISK_SEMI_A * _SATURN_RING_OUTER_REQ
    outer_ring_semi_b = outer_ring_semi_a * sin_b
    in_globe = (xr / DISK_SEMI_A) ** 2 + (yr / DISK_SEMI_B) ** 2 <= 1.0
    in_ring_outer = (xr / outer_ring_semi_a) ** 2 + (yr / outer_ring_semi_b) ** 2 <= 1.0
    in_ring_inner = (xr / inner_ring_semi_a) ** 2 + (yr / inner_ring_semi_b) ** 2 <= 1.0
    overlap = in_globe & (in_ring_outer & ~in_ring_inner)

    req_px = DISK_SEMI_A * 1.05
    rpol_px = req_px * ratio_true
    phi_globe, _lam_globe, depth_globe_raw = _oblate_ortho_inverse(
        dx, dy, B, pole_pa, req_px, rpol_px,
    )
    invalid_in_overlap = overlap & np.isnan(phi_globe)
    assert invalid_in_overlap.sum() > 100, (
        "test geometry doesn't produce enough invalid globe points inside "
        "the footprint overlap to exercise the bug guard"
    )

    # Intended policy: unresolvable globe depth -> always foreground-leaning.
    # NOTE (2026-08-15): weight is no longer asserted to be exactly 1.0
    # everywhere here -- a separate real-data bug fix (see
    # _feather_ring_foreground_boundary's docstring) made this feathering
    # unconditional, so points near the edge of the invalid region now
    # correctly get a soft value instead of the old hard 1.0. The policy
    # guarantee this test actually cares about is that unresolvable points
    # never flip to background-leaning (<=0.5).
    assert weight[invalid_in_overlap].min() > 0.5, (
        "expected every unresolvable-globe-depth point to stay foreground-"
        "leaning (>0.5), even near the feather boundary"
    )

    # Sanity: confirm comparing against the RAW -1.0 sentinel instead (the
    # bug this guards against) would have disagreed at many of those same
    # points -- otherwise this test wouldn't actually catch the bug.
    depth_ring = -yr / math.tan(math.radians(B))
    buggy_fg = overlap & (depth_ring > depth_globe_raw)
    correct_fg = overlap & (depth_ring > -np.inf)
    mismatch = invalid_in_overlap & (buggy_fg != correct_fg)
    assert mismatch.sum() > 100, (
        "test geometry doesn't distinguish the raw-sentinel bug -- "
        "strengthen the scenario"
    )


def test_low_tilt_has_both_foreground_and_background_3d():
    """Direct 3D mirror of test_low_tilt_has_both_foreground_and_background."""
    weight = compute_ring_occlusion_weight_3d(
        H, W, CX, CY, DISK_SEMI_A, DISK_SEMI_B, 0.0, -11.07, RATIO_TRUE,
    )
    n_fg = int((weight > 0.5).sum())
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    dy = yy - CY
    band = (np.abs(dy) < 30.0) & (np.abs(xx - CX) < DISK_SEMI_A)
    n_bg_in_band = int(((weight < 0.5) & band).sum())
    assert n_fg > 0, "expected some genuine foreground pixels"
    assert n_bg_in_band > 0, "expected some genuine background pixels distinct from foreground"


def test_near_zero_b_falls_back_to_conservative_full_exclusion_3d():
    """Direct 3D mirror of test_near_zero_b_falls_back_to_conservative_full_exclusion."""
    weight = compute_ring_occlusion_weight_3d(
        H, W, CX, CY, DISK_SEMI_A, DISK_SEMI_B, 0.0, 1e-6, RATIO_TRUE,
    )
    if (weight > 0.0).any():
        occupied = weight[weight > 0.0]
        assert np.allclose(occupied, 1.0), "expected conservative full-exclusion fallback near B=0"


def test_pole_on_tilt_no_occlusion_region_3d():
    """Direct 3D mirror of test_pole_on_tilt_no_occlusion_region."""
    weight = compute_ring_occlusion_weight_3d(
        H, W, CX, CY, DISK_SEMI_A, DISK_SEMI_B, 0.0, 85.0, RATIO_TRUE,
    )
    assert not (weight > 0.0).any(), "expected no occlusion region at a near-pole-on tilt"
    assert weight.shape == (H, W)


def test_feather_is_continuous_across_boundary_3d():
    """Direct 3D mirror of test_feather_is_continuous_across_boundary."""
    weight = compute_ring_occlusion_weight_3d(
        H, W, CX, CY, DISK_SEMI_A, DISK_SEMI_B, 0.0, -11.07, RATIO_TRUE,
    )
    col = weight[:, int(CX)]
    nonzero_region = np.flatnonzero(col > 0.0)
    if nonzero_region.size > 2:
        diffs = np.abs(np.diff(col[nonzero_region.min():nonzero_region.max() + 1]))
        assert diffs.max() < 0.5, f"expected a smooth feather, got a step of {diffs.max():.3f}"


def test_feather_smooth_even_with_no_background_within_overlap():
    """BUG GUARD (2026-08-15, real-Saturn-data visual inspection): the
    original _feather_ring_foreground_boundary() only ran distance-
    transform feathering when the caller's overlap region contained BOTH
    foreground and background pixels; whenever `is_foreground` covered the
    ENTIRE region with no background portion at all (confirmed to occur on
    real data -- window_01/IR, pole_pa=-7, B=-11.07: overlap 1698px, 100%
    foreground), it silently fell back to a raw, unfeathered boolean mask.
    That hard edge, occurring right where the ring-crossing footprint
    coincides with the disk's own true limb, was amplified by wavelet
    sharpening into a visible bright wedge. Reproduces the degenerate case
    directly against the shared helper (no adjacent background carved out
    of the foreground region at all) and confirms the boundary is still a
    smooth ramp, matching the ordinary case's feather width."""
    h, w = 200, 200
    is_foreground = np.zeros((h, w), dtype=bool)
    is_foreground[80:120, 40:160] = True  # solid band, no background nearby
    weight = _feather_ring_foreground_boundary(h, w, is_foreground)

    row = weight[100, :]
    assert row[10] == 0.0, "expected zero far outside the foreground region"
    assert row[100] == 1.0, "expected full strength deep inside the foreground region"
    assert 0.4 < row[40] < 0.6, f"expected ~0.5 right at the boundary, got {row[40]:.3f}"
    diffs = np.abs(np.diff(row[28:53]))
    assert diffs.max() < 0.2, (
        f"expected a smooth ~{2*12}px-wide ramp across the boundary, "
        f"got a step of {diffs.max():.3f} (old bug: hard 0/1 step)"
    )


# ── Integration: derotate_filter actually applies the 3D ring mask ─────────

def _write_ring_crossing_window(tmp: Path, size: int, r: float, n_non_ref: int, dt_sec: float, t0: datetime):
    """A circular (non-oblate -- keeps the geometry simple) textured disk,
    one reference (dt=0) plus n_non_ref frames at the same nonzero dt_sec,
    all sharing the SAME source content -- any stack difference between
    has_rings=True/False is attributable purely to the ring-occlusion
    mechanism, not real per-frame content differences."""
    rows = []
    img = _textured_disk(size, size / 2.0, size / 2.0, r, seed=1)
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


def test_derotate_filter_applies_ring_occlusion_in_3d_path():
    """Integration regression guard for the actual gap this feature fixes:
    has_rings=True must change derotate_filter's output when
    use_true_reprojection=True (it silently did NOT before this fix -- the
    mask was computed but never passed to spherical_derotation_warp_3d()).
    A low-tilt B with rings crossing the disk (matching this session's real
    Saturn geometry) is used so the occlusion band actually falls inside
    the frame content."""
    size = 300
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        t0 = datetime(2026, 1, 1, 0, 0, 0)
        dt_sec = 9.9281 * 3600.0 * 0.05  # a real, non-tiny per-frame rotation
        rows = _write_ring_crossing_window(tmp, size, r=100.0, n_non_ref=3, dt_sec=dt_sec, t0=t0)

        common_kw = dict(
            period_hours=9.9281,
            use_true_reprojection=True,
            sub_observer_lat_deg=-11.07,
            pole_pa_deg=-7.0,
            true_polar_equatorial_ratio=RATIO_TRUE,
        )
        stacked_no_rings, log_no_rings = derotate_filter(
            rows, t0, align=True, has_rings=False, **common_kw,
        )
        stacked_with_rings, log_with_rings = derotate_filter(
            rows, t0, align=True, has_rings=True, **common_kw,
        )
        assert log_no_rings["ring_crosses_disk"] is False
        assert log_with_rings["ring_crosses_disk"] is True
        assert not np.allclose(stacked_no_rings, stacked_with_rings, atol=1e-4), (
            "has_rings=True made no difference to the use_true_reprojection=True "
            "stack -- the ring mask is still not actually reaching "
            "spherical_derotation_warp_3d()"
        )


# ── ring_crossing_mask persistence (2026-08-16, Phase 0 of
# project_ring_globe_layer_separation_roadmap) ─────────────────────────────
#
# derotate_filter() already computes ring_crossing_mask internally whenever
# has_rings=True (needed by the warp itself); this just stops it from being
# discarded once the function returns, so a future ring-only stacking/
# compositing stage can reuse it without recomputing the geometry. No new
# computation, no change to `stacked`/existing log fields -- verified by the
# byte-identical stack assertion above already covering that. derotate_
# window()'s companion-file-writing side of this (mirrors the existing,
# itself-untested-at-the-unit-level coverage_map_file mechanism) is
# validated against the real pipeline instead, per feedback_ab_test_via_
# real_pipeline -- see experiments/ for that run.

def test_ring_crossing_mask_present_in_log_when_has_rings():
    size = 300
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        t0 = datetime(2026, 1, 1, 0, 0, 0)
        dt_sec = 9.9281 * 3600.0 * 0.05
        rows = _write_ring_crossing_window(tmp, size, r=100.0, n_non_ref=3, dt_sec=dt_sec, t0=t0)
        common_kw = dict(
            period_hours=9.9281,
            use_true_reprojection=True,
            sub_observer_lat_deg=-11.07,
            pole_pa_deg=-7.0,
            true_polar_equatorial_ratio=RATIO_TRUE,
        )
        _, log_with_rings = derotate_filter(rows, t0, align=True, has_rings=True, **common_kw)
        _, log_no_rings = derotate_filter(rows, t0, align=True, has_rings=False, **common_kw)

        mask = log_with_rings["ring_crossing_mask"]
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (size, size)
        assert mask.min() >= 0.0 and mask.max() <= 1.0
        assert log_no_rings["ring_crossing_mask"] is None


# ── Ring-only stack (2026-08-16, Phase 1 of
# project_ring_globe_layer_separation_roadmap) ─────────────────────────────

def test_ring_annulus_mask_covers_only_the_annulus():
    """Sanity check on the new helper: nonzero only between the inner/outer
    ring radii, zero at the globe centre and zero well outside the ring."""
    disk_semi_a = 50.0
    mask = _ring_annulus_mask(H, W, CX, CY, disk_semi_a, 0.0, -25.0)
    assert mask.shape == (H, W)
    assert mask[int(CY), int(CX)] == 0.0, "expected zero at the globe centre"
    inner_r = disk_semi_a * _SATURN_RING_INNER_REQ
    outer_r = disk_semi_a * _SATURN_RING_OUTER_REQ
    mid_r = (inner_r + outer_r) / 2.0
    assert mask[int(CY), int(CX + mid_r)] > 0.9, "expected full weight mid-annulus along the major axis"
    assert mask[int(CY), int(CX + outer_r + 20)] == 0.0, "expected zero well outside the ring"
    # feather=False must return an exact hard boolean (only 0.0/1.0 values).
    hard = _ring_annulus_mask(H, W, CX, CY, disk_semi_a, 0.0, -25.0, feather=False)
    assert set(np.unique(hard).tolist()) <= {0.0, 1.0}


def _ring_textured_frame(size, cx, cy, disk_r, inner_semi_a, outer_semi_a, sub_observer_lat_deg,
                          ring_dx=0.0, ring_dy=0.0, seed=1):
    """Synthetic frame with a flat, STATIONARY globe disk (no texture, so
    the globe-based pre-warp registration always measures ~zero shift) and
    a textured ring ANNULUS ELLIPSE -- foreshortened by sin(|B|) exactly
    like _ring_annulus_mask()'s own geometry (pole_pa=0, so semi-major=x,
    semi-minor=y*sin(|B|)) -- so the real mask actually lines up with where
    the synthetic ring is. **A first version of this fixture drew a plain
    CIRCULAR annulus and silently failed to exercise the real (elliptical)
    mask at all** -- confirmed by direct debugging: _ring_annulus_mask's
    hard-mask pixel count (an ellipse) differed by more than 2x from a
    naive circular annulus at the same B, so a circular test ring and the
    real elliptical mask barely overlapped, and the measured shift was
    garbage as a result. Keep this elliptical, not circular.

    Band-limited noise (not a periodic pattern like sin(theta*N), which is
    a known-degenerate case for phase correlation independent of any
    masking, confirmed separately while debugging this test) can be
    independently shifted by (ring_dx, ring_dy) via cv2.remap resampling of
    the SAME underlying noise field (fixed seed) -- an exact, known
    ground-truth translation of the ring content, isolating whether the
    ring-only stack's registration is actually driven by the ring's own
    content, not the globe's.

    Globe brightness (0.9) is deliberately well above the ring's (peak
    ~0.5) -- confirmed by direct find_disk_center() debugging that making
    them comparable makes the Otsu-based fit latch onto the ring's OUTER
    edge as "the disk" instead of the globe (both are circular/elliptical
    around the same centre here, so the ring/disk squashed-aspect-ratio
    disambiguation in _find_disk_center_impl never triggers) -- exactly the
    brightness contrast real Saturn images have between globe and rings,
    which is what that fit relies on.
    """
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    rr_globe = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    globe = (rr_globe < disk_r).astype(np.float64) * 0.9

    rng = np.random.default_rng(seed)
    field = rng.standard_normal((size, size)).astype(np.float32)
    field = cv2.GaussianBlur(field, (0, 0), sigmaX=1.5)
    map_x = (xx - ring_dx).astype(np.float32)
    map_y = (yy - ring_dy).astype(np.float32)
    shifted_field = cv2.remap(field, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)

    sin_b = abs(math.sin(math.radians(sub_observer_lat_deg)))
    inner_semi_b = inner_semi_a * sin_b
    outer_semi_b = max(outer_semi_a * sin_b, 1e-6)
    xs, ys = xx - ring_dx - cx, yy - ring_dy - cy
    in_outer = (xs / outer_semi_a) ** 2 + (ys / outer_semi_b) ** 2 <= 1.0
    in_inner = (xs / inner_semi_a) ** 2 + (ys / max(inner_semi_b, 1e-6)) ** 2 <= 1.0
    in_annulus = in_outer & ~in_inner
    texture = 0.35 + 0.15 * shifted_field
    ring = np.where(in_annulus, texture, 0.0)
    return np.clip(np.maximum(globe, ring), 0.0, 1.0).astype(np.float32)


def _annulus_laplacian_var(image, cx, cy, inner_semi_a, outer_semi_a, sub_observer_lat_deg):
    sin_b = abs(math.sin(math.radians(sub_observer_lat_deg)))
    inner_semi_b = inner_semi_a * sin_b
    outer_semi_b = max(outer_semi_a * sin_b, 1e-6)
    yy, xx = np.mgrid[0:image.shape[0], 0:image.shape[1]].astype(np.float64)
    xr, yr = xx - cx, yy - cy
    margin = 5.0  # avoid the annulus's own hard edges
    in_outer = (xr / (outer_semi_a - margin)) ** 2 + (yr / max(outer_semi_b - margin, 1e-6)) ** 2 <= 1.0
    in_inner = (xr / (inner_semi_a + margin)) ** 2 + (yr / max(inner_semi_b + margin, 1e-6)) ** 2 <= 1.0
    band = in_outer & ~in_inner
    lap = cv2.Laplacian(image.astype(np.float32), cv2.CV_32F, ksize=3)
    return float(np.var(lap[band]))


def test_ring_only_stack_improves_on_modest_ring_specific_shift():
    """Realistic-magnitude test that the ring-only stack's registration is
    actually driven by the RING's own content, not just inheriting the
    globe's shift: the globe is IDENTICAL (stationary, textureless) in
    every frame, so the existing globe-based pre-warp registration measures
    ~zero shift -- but the ring's own texture is offset by a modest, known
    1px shift in every non-reference frame (deliberately kept small --
    empirically, unwindowed phase correlation on this annulus mask is only
    reliable up to about this magnitude; see the sanity-check fallback in
    derotate_filter()'s ring-only registration block and its comment for
    the larger-shift failure mode this test does NOT exercise). The plain
    atmosphere stack (registered only via the globe) therefore stacks
    slightly-misaligned copies of the ring texture (a bit blurred there),
    while ring_only_stack should measurably improve on that."""
    size = 300
    disk_r = 60.0
    inner_r = disk_r * _SATURN_RING_INNER_REQ
    outer_r = disk_r * _SATURN_RING_OUTER_REQ
    ring_dx, ring_dy = 1.0, 0.0
    B = -25.0

    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        t0 = datetime(2026, 1, 1, 0, 0, 0)
        ref_img = _ring_textured_frame(size, size / 2.0, size / 2.0, disk_r, inner_r, outer_r, B)
        ref_path = tmp / "ref.tif"
        image_io.write_tif_16bit(ref_img, ref_path)
        rows = [{"path": str(ref_path), "stem": "ref", "timestamp": t0, "norm_score": 0.9}]
        for i in range(3):
            img = _ring_textured_frame(
                size, size / 2.0, size / 2.0, disk_r, inner_r, outer_r, B,
                ring_dx=ring_dx, ring_dy=ring_dy,
            )
            path = tmp / f"frame_{i}.tif"
            image_io.write_tif_16bit(img, path)
            rows.append({
                "path": str(path), "stem": f"frame_{i}",
                "timestamp": t0 + timedelta(seconds=60.0), "norm_score": 0.9,
            })

        stacked, log = derotate_filter(
            rows, t0, align=True, has_rings=True, compute_ring_only_stack=True,
            sub_observer_lat_deg=B, pole_pa_deg=0.0,
        )
        ring_only = log["ring_only_stack"]
        assert ring_only is not None

        cx, cy = size / 2.0, size / 2.0
        var_atmosphere = _annulus_laplacian_var(stacked, cx, cy, inner_r, outer_r, B)
        var_ring_only = _annulus_laplacian_var(ring_only, cx, cy, inner_r, outer_r, B)

        assert var_ring_only > 1.1 * var_atmosphere, (
            f"expected ring-only registration to measurably improve on the "
            f"globe-registered atmosphere stack at this modest shift: "
            f"ring_only={var_ring_only:.4f} atmosphere={var_atmosphere:.4f}"
        )


def test_ring_only_stack_falls_back_to_globe_shift_when_ring_measurement_implausible(monkeypatch):
    """If the ring-annulus phase correlation returns a value that disagrees
    wildly with the (trusted) globe-based shift, the code must fall back to
    the globe-based shift rather than trust it -- this is the actual
    empirically-motivated safety net (see derotate_filter()'s ring-only
    registration block): unwindowed phase correlation on this annulus mask
    was found, via synthetic testing, to occasionally lock onto a badly
    wrong value at larger true shifts. Since ring and globe physically move
    together, a large disagreement means the ring lock failed, not that the
    ring truly moved independently -- ring_only_stack must never be worse
    than simply reusing the globe's own registration."""
    size = 300
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        t0 = datetime(2026, 1, 1, 0, 0, 0)
        # Identical content in every frame -- true shift is exactly zero
        # for both globe and ring.
        rows = _write_ring_crossing_window(tmp, size, r=60.0, n_non_ref=2, dt_sec=60.0, t0=t0)

        def _bogus_align(reference, target):
            return (37.0, -41.0)  # absurd; must never be used verbatim

        monkeypatch.setattr(
            "pipeline.modules.derotation.subpixel_align", _bogus_align,
        )
        stacked, log = derotate_filter(
            rows, t0, align=True, has_rings=True, compute_ring_only_stack=True,
            sub_observer_lat_deg=-25.0, pole_pa_deg=0.0,
        )
        ring_only = log["ring_only_stack"]
        assert ring_only is not None
        # Compare only OUTSIDE the globe disk: the atmosphere stack applies
        # real rotation-warp physics inside the disk (dt_sec != 0, matching
        # a realistic window), which ring_only never does by design (shift+
        # scale only) -- that's an intentional, expected difference there,
        # not a bug. Outside the disk there's no drawn ring material either
        # (this fixture has none), so a plausible (near-zero) fallback shift
        # should leave that region matching the plain stack, while an
        # un-clamped 37px "correction" would visibly shift border-replicated
        # content into it instead.
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
        outside_disk = np.sqrt((xx - size / 2.0) ** 2 + (yy - size / 2.0) ** 2) > 65.0
        assert np.allclose(ring_only[outside_disk], stacked[outside_disk], atol=0.05)


def test_ring_only_fallback_uses_genuine_nonzero_globe_shift_not_hardcoded_zero(monkeypatch):
    """Closes a gap an adversarial review found in the test above: with
    IDENTICAL content in every frame, the true globe shift is (0,0), so
    that test can't tell "fell back to the globe's real shift" apart from
    "fell back to a hardcoded (0.0, 0.0) default" -- both look the same.
    Here the globe (and ring, moving with it) has a REAL, known, NONZERO
    shift in every non-reference frame, the ring measurement is forced
    implausible (monkeypatched to an absurd constant), and the fallback
    must reproduce alignment consistent with the genuine globe shift, not
    an identity transform."""
    size = 300
    disk_r = 60.0
    inner_r = disk_r * _SATURN_RING_INNER_REQ
    outer_r = disk_r * _SATURN_RING_OUTER_REQ
    globe_dx, globe_dy = 4.0, 3.0  # real, nonzero, shared by globe+ring
    B = -25.0

    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        t0 = datetime(2026, 1, 1, 0, 0, 0)
        ref_img = _ring_textured_frame(size, size / 2.0, size / 2.0, disk_r, inner_r, outer_r, B)
        ref_path = tmp / "ref.tif"
        image_io.write_tif_16bit(ref_img, ref_path)
        rows = [{"path": str(ref_path), "stem": "ref", "timestamp": t0, "norm_score": 0.9}]
        for i in range(3):
            # Globe AND ring both drawn shifted by the same (globe_dx, globe_dy)
            # -- physically realistic (both move together on the sky).
            img = _ring_textured_frame(
                size, size / 2.0 + globe_dx, size / 2.0 + globe_dy, disk_r, inner_r, outer_r, B,
                ring_dx=globe_dx, ring_dy=globe_dy,
            )
            path = tmp / f"frame_{i}.tif"
            image_io.write_tif_16bit(img, path)
            rows.append({
                "path": str(path), "stem": f"frame_{i}",
                "timestamp": t0 + timedelta(seconds=60.0), "norm_score": 0.9,
            })

        def _bogus_align(reference, target):
            return (99.0, -99.0)  # absurd; forces the plausibility check to fail

        monkeypatch.setattr("pipeline.modules.derotation.subpixel_align", _bogus_align)
        stacked, log = derotate_filter(
            rows, t0, align=True, has_rings=True, compute_ring_only_stack=True,
            sub_observer_lat_deg=B, pole_pa_deg=0.0,
        )
        ring_only = log["ring_only_stack"]
        assert ring_only is not None

        # If the fallback correctly used the genuine (~-4,-3 in apply_shift_
        # and_scale's target-center convention) globe shift, ring_only_stack
        # should closely resemble a single well-aligned frame (the reference
        # itself) in the ring band -- an identity-transform fallback would
        # instead leave the un-recentred content misaligned by (4,3)px
        # relative to the reference, visibly different.
        cx, cy = size / 2.0, size / 2.0
        diff_if_fallback_worked = np.abs(ring_only - ref_img)
        # An unshifted (identity) composite of frames each offset by (4,3)px
        # would show a large, structured residual against the reference
        # specifically in the ring band (ghosting); a correct globe-shift
        # fallback should not.
        sin_b = math.sin(math.radians(B))
        band_yy, band_xx = np.mgrid[0:size, 0:size].astype(np.float64)
        rr = np.sqrt(((band_xx - cx)) ** 2 + ((band_yy - cy) / max(abs(sin_b), 1e-6)) ** 2)
        ring_band = (rr >= inner_r + 4) & (rr <= outer_r - 4)
        mean_diff_in_band = float(diff_if_fallback_worked[ring_band].mean())
        assert mean_diff_in_band < 0.05, (
            f"expected the fallback to use the genuine ({-globe_dx},{-globe_dy})px globe shift "
            f"(ring_only closely matching the reference in the ring band), not an identity "
            f"transform -- mean|diff|={mean_diff_in_band:.4f}"
        )


# ── Ring registration crop+taper fix (2026-08-17, fixes the ~1-2px-only
# reliability limit above by cropping tightly to the ring and applying a
# smooth taper instead of multiplying a hard mask against the full image)
# ────────────────────────────────────────────────────────────────────────

def test_ring_registration_crop_returns_none_when_geometry_degenerate():
    """Disk too close to the frame edge for the ideal crop half-size
    (outer_semi_a * margin_factor) to fit with at least a 20px margin --
    must return None (this module's standard "unmeasurable" convention),
    never a garbage-sized or badly-clipped array."""
    # Center placed far outside the image entirely (not just near an edge)
    # so the clamped crop window has essentially zero/negative overlap with
    # the actual image bounds.
    image = np.zeros((400, 400), dtype=np.float32)
    crop = _ring_registration_crop(image, cx=-500.0, cy=200.0, disk_semi_a=50.0,
                                    pole_pa_deg=0.0, sub_observer_lat_deg=-25.0)
    assert crop is None


def test_ring_registration_crop_is_zero_outside_the_annulus():
    """The crop must taper to (near-)zero both at the inner cavity and well
    outside the outer ring edge -- i.e. it must actually isolate ring
    content, not just be an arbitrary rectangular crop."""
    disk_semi_a = 60.0
    image = np.ones((400, 400), dtype=np.float32)
    crop = _ring_registration_crop(image, cx=200.0, cy=200.0, disk_semi_a=disk_semi_a,
                                    pole_pa_deg=0.0, sub_observer_lat_deg=-25.0)
    assert crop is not None
    cy0, cx0 = crop.shape[0] // 2, crop.shape[1] // 2
    assert crop[cy0, cx0] == 0.0, "expected ~zero at the crop centre (inner cavity)"
    assert crop[2, 2] == 0.0, "expected ~zero near the crop's own corner (outside the ring)"
    assert crop.max() > 0.5, "expected substantial nonzero weight somewhere in the annulus"


@pytest.mark.parametrize("ring_dx,ring_dy", [(3.0, -2.0), (5.0, 5.0)])
def test_ring_only_stack_improves_on_larger_ring_specific_shift(ring_dx, ring_dy):
    """Same scenario as test_ring_only_stack_improves_on_modest_ring_specific_shift
    (1px) but at the larger shift magnitudes the OLD full-image hard-mask
    approach was documented to get wrong at (3,-2)px specifically (see
    _ring_annulus_mask's docstring history) -- the crop+taper fix
    (_ring_registration_crop) must handle these too, not just 1px."""
    size = 300
    disk_r = 60.0
    inner_r = disk_r * _SATURN_RING_INNER_REQ
    outer_r = disk_r * _SATURN_RING_OUTER_REQ
    B = -25.0

    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        t0 = datetime(2026, 1, 1, 0, 0, 0)
        ref_img = _ring_textured_frame(size, size / 2.0, size / 2.0, disk_r, inner_r, outer_r, B)
        ref_path = tmp / "ref.tif"
        image_io.write_tif_16bit(ref_img, ref_path)
        rows = [{"path": str(ref_path), "stem": "ref", "timestamp": t0, "norm_score": 0.9}]
        for i in range(3):
            img = _ring_textured_frame(
                size, size / 2.0, size / 2.0, disk_r, inner_r, outer_r, B,
                ring_dx=ring_dx, ring_dy=ring_dy,
            )
            path = tmp / f"frame_{i}.tif"
            image_io.write_tif_16bit(img, path)
            rows.append({
                "path": str(path), "stem": f"frame_{i}",
                "timestamp": t0 + timedelta(seconds=60.0), "norm_score": 0.9,
            })

        stacked, log = derotate_filter(
            rows, t0, align=True, has_rings=True, compute_ring_only_stack=True,
            sub_observer_lat_deg=B, pole_pa_deg=0.0,
        )
        ring_only = log["ring_only_stack"]
        assert ring_only is not None

        cx, cy = size / 2.0, size / 2.0
        var_atmosphere = _annulus_laplacian_var(stacked, cx, cy, inner_r, outer_r, B)
        var_ring_only = _annulus_laplacian_var(ring_only, cx, cy, inner_r, outer_r, B)

        assert var_ring_only > 1.1 * var_atmosphere, (
            f"expected ring-only registration to measurably improve on the "
            f"globe-registered atmosphere stack at shift=({ring_dx},{ring_dy}): "
            f"ring_only={var_ring_only:.4f} atmosphere={var_atmosphere:.4f}"
        )


def test_ring_only_stack_excludes_frame_with_no_reliable_globe_prewarp():
    """2026-08-17 fix: when a frame's own globe-based pre-warp measurement
    fails outright (here: its globe center is drawn >15px from the
    reference's, so derotate_filter()'s own pre-warp sanity gate rejects
    it -- see the `abs(_dx) <= 15.0` checks), that frame must get WEIGHT
    ZERO in the ring-only stack (previously: a silent hardcoded identity
    (0,0,1.0) transform, contaminating the average with a badly misaligned
    copy). Verified by comparing against a stack built from only the two
    well-behaved frames -- if the bad frame were still contributing, the
    two stacks would visibly differ."""
    size = 300
    disk_r = 60.0
    inner_r = disk_r * _SATURN_RING_INNER_REQ
    outer_r = disk_r * _SATURN_RING_OUTER_REQ
    B = -25.0

    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        t0 = datetime(2026, 1, 1, 0, 0, 0)
        ref_img = _ring_textured_frame(size, size / 2.0, size / 2.0, disk_r, inner_r, outer_r, B)
        ref_path = tmp / "ref.tif"
        image_io.write_tif_16bit(ref_img, ref_path)

        good_rows = [{"path": str(ref_path), "stem": "ref", "timestamp": t0, "norm_score": 0.9}]
        for i in range(2):
            img = _ring_textured_frame(size, size / 2.0, size / 2.0, disk_r, inner_r, outer_r, B)
            path = tmp / f"good_{i}.tif"
            image_io.write_tif_16bit(img, path)
            good_rows.append({
                "path": str(path), "stem": f"good_{i}",
                "timestamp": t0 + timedelta(seconds=60.0), "norm_score": 0.9,
            })

        # A third, "bad" frame: globe drawn 25px off from the reference (>
        # the 15px pre-warp sanity gate) AND its ring texture shifted by a
        # large, different amount -- if it silently got an identity
        # transform + full weight (the old bug), it would visibly
        # contaminate the ring-only stack with ghosting.
        bad_img = _ring_textured_frame(
            size, size / 2.0 + 25.0, size / 2.0, disk_r, inner_r, outer_r, B,
            ring_dx=20.0, ring_dy=-15.0, seed=2,
        )
        bad_path = tmp / "bad.tif"
        image_io.write_tif_16bit(bad_img, bad_path)
        bad_row = {"path": str(bad_path), "stem": "bad",
                   "timestamp": t0 + timedelta(seconds=60.0), "norm_score": 0.9}

        _, log_good_only = derotate_filter(
            good_rows, t0, align=True, has_rings=True, compute_ring_only_stack=True,
            sub_observer_lat_deg=B, pole_pa_deg=0.0,
        )
        ring_only_good = log_good_only["ring_only_stack"]

        _, log_with_bad = derotate_filter(
            good_rows + [bad_row], t0, align=True, has_rings=True, compute_ring_only_stack=True,
            sub_observer_lat_deg=B, pole_pa_deg=0.0,
        )
        ring_only_with_bad = log_with_bad["ring_only_stack"]

        assert ring_only_good is not None and ring_only_with_bad is not None
        diff = np.abs(ring_only_good - ring_only_with_bad)
        assert float(diff.max()) < 0.02, (
            f"expected the bad frame (no reliable globe pre-warp) to contribute "
            f"~nothing (weight 0) to the ring-only stack -- max|diff| vs the "
            f"good-frames-only stack was {float(diff.max()):.4f}"
        )


def test_ring_only_stack_absent_when_disabled_or_ringless():
    size = 300
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        t0 = datetime(2026, 1, 1, 0, 0, 0)
        rows = _write_ring_crossing_window(tmp, size, r=60.0, n_non_ref=2, dt_sec=60.0, t0=t0)

        _, log_disabled = derotate_filter(
            rows, t0, align=True, has_rings=True, compute_ring_only_stack=False,
            sub_observer_lat_deg=-25.0, pole_pa_deg=0.0,
        )
        assert log_disabled["ring_only_stack"] is None

        _, log_ringless = derotate_filter(
            rows, t0, align=True, has_rings=False, compute_ring_only_stack=True,
            sub_observer_lat_deg=-25.0, pole_pa_deg=0.0,
        )
        assert log_ringless["ring_only_stack"] is None


def test_ring_only_stack_does_not_change_atmosphere_stack():
    """compute_ring_only_stack=True must not alter `stacked` at all --
    byte-identical to the flag being off."""
    size = 300
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        t0 = datetime(2026, 1, 1, 0, 0, 0)
        rows = _write_ring_crossing_window(tmp, size, r=60.0, n_non_ref=2, dt_sec=60.0, t0=t0)
        common_kw = dict(has_rings=True, sub_observer_lat_deg=-25.0, pole_pa_deg=0.0)

        stacked_off, _ = derotate_filter(rows, t0, align=True, compute_ring_only_stack=False, **common_kw)
        stacked_on, _ = derotate_filter(rows, t0, align=True, compute_ring_only_stack=True, **common_kw)
        assert np.array_equal(stacked_off, stacked_on)


def test_compute_ring_only_stack_defaults_false():
    assert DerotationConfig().compute_ring_only_stack is False


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
    test_matches_oblate_ortho_forward_ring_depth_3d()
    print("3d: matches _oblate_ortho_forward ring depth (flip_pole_axis-aware): OK")
    test_depth_globe_matches_oblate_ortho_inverse()
    print("3d: depth_globe matches _oblate_ortho_inverse: OK")
    test_invalid_globe_depth_defaults_to_foreground()
    print("3d: invalid globe depth defaults to foreground: OK")
    test_low_tilt_has_both_foreground_and_background_3d()
    print("3d: low tilt has both foreground and background: OK")
    test_near_zero_b_falls_back_to_conservative_full_exclusion_3d()
    print("3d: near-zero B conservative fallback: OK")
    test_pole_on_tilt_no_occlusion_region_3d()
    print("3d: near-pole-on tilt has no occlusion region: OK")
    test_feather_is_continuous_across_boundary_3d()
    print("3d: feather is continuous: OK")
    test_feather_smooth_even_with_no_background_within_overlap()
    print("feather smooth even with no background within overlap: OK")
    test_derotate_filter_applies_ring_occlusion_in_3d_path()
    print("integration: derotate_filter applies ring occlusion in 3D path: OK")
    test_ring_crossing_mask_present_in_log_when_has_rings()
    print("ring_crossing_mask present in log when has_rings: OK")
    test_ring_annulus_mask_covers_only_the_annulus()
    print("ring annulus mask covers only the annulus: OK")
    test_ring_only_stack_improves_on_modest_ring_specific_shift()
    print("ring-only stack improves on modest ring-specific shift: OK")
    test_ring_only_stack_absent_when_disabled_or_ringless()
    print("ring-only stack absent when disabled or ringless: OK")
    test_ring_only_stack_does_not_change_atmosphere_stack()
    print("ring-only stack does not change atmosphere stack: OK")
    test_compute_ring_only_stack_defaults_false()
    print("compute_ring_only_stack defaults False: OK")
    print("(ring-only fallback monkeypatch test requires pytest -- run via pytest)")
    print("\nAll checks passed.")
