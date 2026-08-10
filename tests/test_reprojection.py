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
        # At B=0 the x-component (equatorial/Delta-lambda direction) must
        # match closely. The y-component (pole-axis direction) is a
        # SEPARATE, not-yet-empirically-resolved sign ambiguity at nonzero
        # pole_pa (see DerotationConfig.flip_pole_axis) — not checked here.
        assert abs(shift_linear[0] - shift_3d[0]) < 1.5, (
            f"pole_pa={pole_pa}: x-shift direction mismatch — "
            f"linear={shift_linear} 3d={shift_3d}"
        )
        if pole_pa == 0.0:
            assert abs(shift_linear[1] - shift_3d[1]) < 1.5, (
                f"pole_pa=0: y-shift should also match — linear={shift_linear} 3d={shift_3d}"
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


if __name__ == "__main__":
    test_round_trip_self_consistency()
    print("round-trip self-consistency: OK")
    test_b_near_zero_numerical_stability()
    print("B->0 numerical stability: OK")
    test_b_zero_matches_live_linear_warp_direction()
    print("B=0 direction match vs live linear warp: OK")
    test_point_shift_matches_full_image_warp()
    print("point-shift vs full-image-warp consistency: OK")
    test_discriminant_clamp_near_limb()
    print("discriminant clamp near limb: OK")
    print("\nAll checks passed.")
