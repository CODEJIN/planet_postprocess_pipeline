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

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.modules import image_io
from pipeline.modules.derotation import (
    _SATURN_RING_INNER_REQ,
    _SATURN_RING_OUTER_REQ,
    _feather_ring_foreground_boundary,
    _oblate_ortho_forward,
    _oblate_ortho_inverse,
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

def _textured_disk(size, cx, cy, r, amp=0.15, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    disk = (rr < r).astype(np.float64) * 0.6
    texture = amp * rng.standard_normal((size, size)) * (rr < r)
    return np.clip(disk + texture, 0.0, 1.0).astype(np.float32)


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
    print("\nAll checks passed.")
