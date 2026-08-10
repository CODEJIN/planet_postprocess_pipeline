"""Regression tests for the true oblate-spheroid reprojection de-rotation
warp (spherical_derotation_warp_3d, DerotationConfig.use_true_reprojection).

This is an additive, opt-in alternative to the validated linear
spherical_derotation_warp() — see the "True oblate-spheroid reprojection"
section in pipeline/modules/derotation.py for the derivation. These tests
guard the properties that were hardest to get right during design (all
found via an adversarial review before implementation):

  1. Forward/inverse round-trip self-consistency across a wide range of
     sub-observer latitude B, position angle P, and body coordinates.
  2. Numerical stability of the B->0 quadratic-vs-direct-solve branch —
     without it, longitude recovery explodes near B=0 (division by
     sin(B)); this must NOT regress silently.
  3. B=0 ground-truth sign/direction match against the live (validated)
     spherical_derotation_warp() — this is a REAL regression test: an
     earlier draft of _reprojected_position had the Delta-lambda sign
     backwards (it matched a *different*, unrelated sign check performed
     during planning, not the actual warp function), which was only caught
     by directly comparing pixel-shift ground truth between the two
     functions. If this sign flips back, both spherical_derotation_warp_3d
     and the satellite/shadow smearing-position correction in
     satellite_composite.py would silently move content in the wrong
     direction.
  4. Point-shift/full-image-warp consistency — _reprojection_point_shift
     (used by the satellite/shadow smearing correction) must agree with
     spherical_derotation_warp_3d (the full per-pixel image warp) at the
     same pixel, since they share the same underlying math but are called
     from two different, independently-maintained code paths.

Run directly: python3 tests/test_reprojection.py
Or via pytest: pytest tests/test_reprojection.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.modules.derotation import (
    _oblate_ortho_forward,
    _oblate_ortho_inverse,
    _reprojection_point_shift,
    auto_detect_equator_pa,
    spherical_derotation_warp,
    spherical_derotation_warp_3d,
)

REQ, RPOL = 100.0, 90.0  # synthetic test oblateness (not any real planet)


def test_round_trip_self_consistency():
    """forward(phi, lam) -> inverse -> forward must reproduce the same
    pixel offset across a wide sweep of B, P, phi, lam."""
    rng = np.random.default_rng(0)
    max_err = 0.0
    n_checked = 0
    for _ in range(20_000):
        B = rng.uniform(-60, 60)
        P = rng.uniform(-180, 180)
        phi = rng.uniform(-1.3, 1.3)
        lam = rng.uniform(-1.3, 1.3)
        dx, dy, depth = _oblate_ortho_forward(phi, lam, B, P, REQ, RPOL)
        if depth <= 0:
            continue  # not on the near/visible side — nothing to round-trip
        phi2, lam2, depth2 = _oblate_ortho_inverse(dx, dy, B, P, REQ, RPOL)
        assert np.isfinite(phi2), f"inverse failed for a valid forward point: B={B} P={P}"
        dx2, dy2, _ = _oblate_ortho_forward(phi2, lam2, B, P, REQ, RPOL)
        err = float(np.hypot(dx - dx2, dy - dy2))
        max_err = max(max_err, err)
        n_checked += 1
    assert n_checked > 15_000, f"too few valid (depth>0) samples: {n_checked}"
    assert max_err < 1e-4, f"round-trip pixel error too high: {max_err}"


def test_b_near_zero_numerical_stability():
    """The direct-solve branch for |B| < threshold must not regress into
    the general quadratic's division-by-sin(B) instability."""
    phi, lam, P = 0.3, 0.4, 10.0
    for B in (0.0, 1e-8, 1e-6, 1e-4, 0.01, 0.05, 0.1, 1.0):
        dx, dy, depth = _oblate_ortho_forward(phi, lam, B, P, REQ, RPOL)
        assert depth > 0
        phi2, lam2, depth2 = _oblate_ortho_inverse(dx, dy, B, P, REQ, RPOL)
        assert np.isfinite(phi2), f"B={B}: inverse produced NaN"
        assert abs(phi2 - phi) < 1e-6, f"B={B}: phi drifted ({phi2} vs {phi})"
        assert abs(lam2 - lam) < 1e-6, f"B={B}: lam drifted ({lam2} vs {lam}) — likely the sin(B) instability regressing"


def test_b_zero_matches_live_linear_warp_direction():
    """REGRESSION GUARD: at B=0, spherical_derotation_warp_3d must shift a
    real image's content in the SAME direction as the validated
    spherical_derotation_warp(). This is the ground truth an earlier
    implementation draft got backwards (see module docstring)."""
    h, w = 400, 400
    cx, cy, disk_r, period_h = 200.0, 200.0, 100.0, 10.0
    raw_x, raw_y = 230.0, 210.0
    dt_sec = 300.0

    img = np.zeros((h, w), dtype=np.float32)
    img[int(round(raw_y)), int(round(raw_x))] = 1.0

    for pole_pa in (0.0, 45.0, 120.0):
        out_linear = spherical_derotation_warp(
            img, dt_sec, cx, cy, disk_r,
            period_hours=period_h, scale=1.0,
            pole_pa_deg=pole_pa, polar_equatorial_ratio=0.935,
        )
        out_3d = spherical_derotation_warp_3d(
            img, dt_sec, cx, cy, disk_r, period_h,
            sub_observer_lat_deg=0.0, pole_pa_deg=pole_pa,
            polar_equatorial_ratio_true=0.935, scale=1.0,
        )
        yl, xl = np.unravel_index(np.argmax(out_linear), out_linear.shape)
        y3, x3 = np.unravel_index(np.argmax(out_3d), out_3d.shape)
        shift_linear = (xl - raw_x, yl - raw_y)
        shift_3d = (x3 - raw_x, y3 - raw_y)
        # At B=0 there is no true B-tilt effect, so linear and 3D must agree
        # on BOTH x and y at every pole_pa, not just pole_pa=0. An earlier
        # version of this test only checked y at pole_pa=0 (the pole-axis
        # sign was believed to be a separate, real ambiguity resolved by
        # flip_pole_axis) — that gap let a real bug through: the 3D warp's
        # pole_pa rotation was an IMPROPER rotation (determinant -1, see
        # test_pole_pa_rotation_is_proper below), which flip_pole_axis
        # cannot fix (it flips Y, not the reflection). Found via external
        # code review + confirmed against real Jupiter data (derot NCC
        # confidence dropped 0.877->0.620 with the bug present).
        assert abs(shift_linear[0] - shift_3d[0]) < 1.5, (
            f"pole_pa={pole_pa}: x-shift direction mismatch — "
            f"linear={shift_linear} 3d={shift_3d}"
        )
        assert abs(shift_linear[1] - shift_3d[1]) < 1.5, (
            f"pole_pa={pole_pa}: y-shift direction mismatch — "
            f"linear={shift_linear} 3d={shift_3d}"
        )


def test_pole_pa_rotation_is_proper():
    """REGRESSION GUARD: the pole_pa rotation inside _oblate_ortho_forward
    must be a proper rotation (determinant +1), not an improper one
    (determinant -1, i.e. a hidden reflection). This is a cheap, direct
    check for the exact bug class found above: computing dy with an extra
    outer sign flip turns the transform into a reflection, which no amount
    of toggling flip_pole_axis (which only flips Y, not the handedness) can
    undo — confirmed empirically as the cause of a real quality regression
    on Jupiter data (NCC confidence 0.877->0.620) before this was found and
    fixed via external code review.
    """
    req_px, rpol_px = 100.0, 93.0
    for pole_pa_deg in (0.0, -6.25, 30.0, 90.0, 179.0):
        # Probe the forward map's Jacobian at B=0 with two orthogonal unit
        # nudges in (X, Y) via finite differences on phi/lam is overkill —
        # instead, directly verify the *rotation step* by checking that
        # forward-then-inverse recovers the identity for a pure (X, Y) pair,
        # AND that swapping the two orthogonal basis directions preserves
        # handedness (cross product sign), which an improper rotation flips.
        phi1, lam1 = 0.1, 0.05
        phi2, lam2 = 0.1, 0.15
        dx1, dy1, _ = _oblate_ortho_forward(phi1, lam1, 0.0, pole_pa_deg, req_px, rpol_px)
        dx2, dy2, _ = _oblate_ortho_forward(phi2, lam2, 0.0, pole_pa_deg, req_px, rpol_px)
        dxo, dyo, _ = _oblate_ortho_forward(0.0, 0.0, 0.0, pole_pa_deg, req_px, rpol_px)
        v1 = np.array([dx1 - dxo, dy1 - dyo])
        v2 = np.array([dx2 - dxo, dy2 - dyo])
        # A proper rotation preserves the sign of the z-component of the
        # cross product of any two vectors under the SAME transform as their
        # pre-image (phi,lam differences here play the role of the
        # pre-transform basis) — an improper (reflecting) transform flips it.
        # Compare against pole_pa_deg=0 as the known-good reference sign.
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        dx1_0, dy1_0, _ = _oblate_ortho_forward(phi1, lam1, 0.0, 0.0, req_px, rpol_px)
        dx2_0, dy2_0, _ = _oblate_ortho_forward(phi2, lam2, 0.0, 0.0, req_px, rpol_px)
        dxo_0, dyo_0, _ = _oblate_ortho_forward(0.0, 0.0, 0.0, 0.0, req_px, rpol_px)
        v1_0 = np.array([dx1_0 - dxo_0, dy1_0 - dyo_0])
        v2_0 = np.array([dx2_0 - dxo_0, dy2_0 - dyo_0])
        cross_0 = v1_0[0] * v2_0[1] - v1_0[1] * v2_0[0]
        assert np.sign(cross) == np.sign(cross_0), (
            f"pole_pa={pole_pa_deg}: handedness flipped relative to pole_pa=0 "
            f"— rotation is improper (a reflection), not a proper rotation"
        )


def test_point_shift_matches_full_image_warp():
    """_reprojection_point_shift (used by the satellite/shadow smearing
    correction) must agree with spherical_derotation_warp_3d (the full
    per-pixel image warp) at the same pixel — they share the same math via
    _reprojected_position but are invoked from independent call sites."""
    h, w = 400, 400
    cx, cy, disk_r, period_h = 200.0, 200.0, 100.0, 10.0
    B, pole_pa, ratio, scale = -11.0, 25.0, 0.9021, 0.10
    dt_sec = 250.0

    for raw_x, raw_y in [(230.0, 210.0), (180.0, 240.0), (205.0, 205.0)]:
        img = np.zeros((h, w), dtype=np.float32)
        img[int(round(raw_y)), int(round(raw_x))] = 1.0
        out = spherical_derotation_warp_3d(
            img, dt_sec, cx, cy, disk_r, period_h,
            sub_observer_lat_deg=B, pole_pa_deg=pole_pa,
            polar_equatorial_ratio_true=ratio, scale=scale,
        )
        y_img, x_img = np.unravel_index(np.argmax(out), out.shape)

        # The image warp maps OUTPUT pixel -> SOURCE pixel (dst(x,y)=src(map)),
        # so the point-shift helper (same direction as the smearing
        # correction's usage: raw position -> its position after the SAME
        # warp the atmosphere underwent) needs -dt_sec, exactly as wired in
        # satellite_composite.py's _compute_smearing_map.
        wdx, wdy = _reprojection_point_shift(
            raw_x, raw_y, -dt_sec, cx, cy, disk_r, period_h,
            B, pole_pa, ratio, scale=scale,
        )
        pred_x, pred_y = raw_x + wdx, raw_y + wdy
        assert abs(pred_x - x_img) < 1.5 and abs(pred_y - y_img) < 1.5, (
            f"point={raw_x},{raw_y}: point-shift predicted ({pred_x:.1f},{pred_y:.1f}) "
            f"but full-image warp placed it at ({x_img},{y_img})"
        )


def test_discriminant_clamp_near_limb():
    """No NaN/garbage for points at or just outside the disk radius, across
    a B/P sweep — the discriminant and sqrt(xb_sq) clamps must hold."""
    rng = np.random.default_rng(1)
    for _ in range(2000):
        B = rng.uniform(-60, 60)
        P = rng.uniform(-180, 180)
        r_frac = rng.uniform(0.95, 1.10)  # at and just past the limb
        theta = rng.uniform(0, 2 * np.pi)
        dx = r_frac * REQ * np.cos(theta)
        dy = r_frac * REQ * np.sin(theta)
        phi, lam, depth = _oblate_ortho_inverse(dx, dy, B, P, REQ, RPOL)
        # Either a valid finite solution, or cleanly marked invalid (NaN) —
        # never a silent garbage value (e.g. complex-sqrt artifact leaking
        # through as a huge/NaN-adjacent float).
        if np.isfinite(phi):
            assert -np.pi / 2 - 1e-6 <= phi <= np.pi / 2 + 1e-6
        assert not (np.isfinite(phi) and depth <= 0), "finite phi with non-visible depth"


def test_positive_pole_pa_is_clockwise_on_screen():
    """REGRESSION GUARD (external review, 2026-08-10): positive pole_pa_deg
    must sweep CLOCKWISE as displayed on screen, not counter-clockwise —
    several docstrings previously claimed "Positive = CCW", which is wrong
    for this module's plain (x right, y down) pixel-coordinate convention.
    Verified directly: a point placed to the right of disk centre, warped
    with increasing pole_pa_deg, must move increasingly DOWNWARD (+y) —
    3 o'clock toward 6 o'clock is clockwise as displayed.
    """
    h, w = 400, 400
    cx, cy, disk_r, period_h = 200.0, 200.0, 150.0, 10.0
    dt_sec = 3000.0
    px, py = 210.0, 200.0

    prev_dy = -1.0  # dy must strictly increase (more positive / more downward)
    for pole_pa in (0.0, 30.0, 60.0, 90.0):
        img = np.zeros((h, w), dtype=np.float32)
        img[int(py), int(px)] = 1.0
        warped = spherical_derotation_warp(
            img, dt_sec, cx, cy, disk_r, period_hours=period_h,
            scale=1.0, pole_pa_deg=pole_pa, polar_equatorial_ratio=1.0,
        )
        yy, xx = np.unravel_index(np.argmax(warped), warped.shape)
        dx, dy = xx - px, yy - py
        if pole_pa == 0.0:
            assert dx > 10 and abs(dy) < 2, f"pole_pa=0 should drift rightward: dx={dx} dy={dy}"
        elif pole_pa == 90.0:
            assert dy > 10 and abs(dx) < 2, f"pole_pa=90 should drift downward: dx={dx} dy={dy}"
        else:
            assert dy > prev_dy, (
                f"pole_pa={pole_pa}: dy={dy} did not increase from previous "
                f"pole_pa's dy={prev_dy} — positive pole_pa is not sweeping "
                f"clockwise (toward +y) as it should"
            )
        prev_dy = dy


def test_auto_detect_equator_pa_matches_warp_sign_convention():
    """REGRESSION GUARD: auto_detect_equator_pa's returned angle must use
    the SAME sign convention as the warp's own pole_pa_deg (positive =
    clockwise as displayed) — verified by rotating a synthetic belted
    image with cv2's own independently-defined rotation direction
    (cv2.getRotationMatrix2D: positive angle = CCW as displayed) and
    checking the detector reports the opposite sign."""
    import cv2

    h, w = 300, 300
    cx, cy, disk_r = 150.0, 150.0, 120.0

    def make_belted_image(tilt_deg):
        img = np.zeros((h, w), dtype=np.float32)
        yy_row = np.arange(h)[:, None]
        img[:, :] = 0.5 + 0.4 * np.sin(yy_row / 15.0)
        yy, xx = np.mgrid[0:h, 0:w]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < disk_r ** 2
        img = (img * mask).astype(np.float32)
        if tilt_deg != 0:
            M = cv2.getRotationMatrix2D((cx, cy), tilt_deg, 1.0)  # + = CCW on screen
            img = cv2.warpAffine(img, M, (w, h))
        return img.astype(np.float32)

    pa_ccw15 = auto_detect_equator_pa(frames=[make_belted_image(15.0)], cx=cx, cy=cy, disk_radius_px=disk_r)
    pa_cw15 = auto_detect_equator_pa(frames=[make_belted_image(-15.0)], cx=cx, cy=cy, disk_radius_px=disk_r)
    # Image rotated CCW (cv2 +15) must be detected as NEGATIVE (since positive
    # here means clockwise); image rotated CW (cv2 -15) must be POSITIVE.
    assert pa_ccw15 < -5, f"CCW-rotated image should give a clearly negative equator_pa, got {pa_ccw15}"
    assert pa_cw15 > 5, f"CW-rotated image should give a clearly positive equator_pa, got {pa_cw15}"


if __name__ == "__main__":
    test_round_trip_self_consistency()
    print("round-trip self-consistency: OK")
    test_b_near_zero_numerical_stability()
    print("B->0 numerical stability: OK")
    test_b_zero_matches_live_linear_warp_direction()
    print("B=0 direction match vs live linear warp (all pole_pa): OK")
    test_pole_pa_rotation_is_proper()
    print("pole_pa rotation is proper (determinant/handedness check): OK")
    test_point_shift_matches_full_image_warp()
    print("point-shift vs full-image-warp consistency: OK")
    test_discriminant_clamp_near_limb()
    print("discriminant clamp near limb: OK")
    test_positive_pole_pa_is_clockwise_on_screen()
    print("positive pole_pa is clockwise on screen: OK")
    test_auto_detect_equator_pa_matches_warp_sign_convention()
    print("auto_detect_equator_pa sign matches warp convention: OK")
    print("\nAll checks passed.")
