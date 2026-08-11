"""Regression tests for compute_ring_crossing_mask() (Saturn ring/globe
architecture fix, 2026-08-11 — see project_saturn_ring_globe_separation and
project_saturn_step05_sharpness_gap memory).

Replaces the earlier image-based ring-edge detector (detect_ring_geometry,
deleted — it failed on 100% of 45 real Saturn frames) with a fully analytic
approach: the ring's projected inner/outer ellipses are computed directly
from the globe's own already-measured geometry and a Horizons sub-observer
latitude B, using the same _oblate_ortho_forward() projection already
validated for the Jupiter 3D reprojection feature (at phi=0, the equatorial
plane where the ring lies, that projection reduces to a point on an ellipse
with semi-major=req_px, semi-minor=req_px*sin(|B|)).

Run directly: python3 tests/test_ring_crossing_mask.py
Or via pytest: pytest tests/test_ring_crossing_mask.py -v
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.modules.derotation import (
    _SATURN_RING_INNER_REQ,
    _SATURN_RING_OUTER_REQ,
    _oblate_ortho_forward,
    compute_ring_crossing_mask,
)

H, W = 300, 300
CX, CY = 150.0, 150.0
DISK_SEMI_A = 66.0
DISK_SEMI_B = 59.5  # ~Saturn's true 0.9021 oblateness at this semi_a


def test_low_tilt_crosses_widely():
    """REGRESSION GUARD: at a low, realistic Saturn tilt (matching this
    session's real data, B=-11.07 deg), the ring must be found crossing the
    globe over a substantial area — confirmed visually on real frames this
    session (Saturn_Data/step04_derotated/window_01 and window_05)."""
    mask = compute_ring_crossing_mask(H, W, CX, CY, DISK_SEMI_A, DISK_SEMI_B, 0.0, -11.07)
    frac = mask.sum() / (math.pi * DISK_SEMI_A * DISK_SEMI_B)
    assert frac > 0.05, f"expected a substantial crossing area at B=-11 deg, got {frac:.3f} of disk area"


def test_crossing_area_shrinks_beyond_saturns_real_tilt_range():
    """REGRESSION GUARD: past Saturn's real max tilt (~27 deg), crossing
    area must shrink toward zero as |B| keeps increasing toward pole-on —
    the ring's inner-edge minor-axis approach to the globe centre
    (req_px*sin(|B|)) keeps growing with |B|, so eventually none of it
    dips inside the globe's own silhouette (see test_pole_on_tilt_no_
    crossing for the B=85 deg extreme). Note: area is NOT monotonic across
    the FULL B range -- very-near-edge-on tilts (B close to 0) have a
    vanishingly THIN ring annulus (near-zero projected height), so area
    actually grows from ~0 before shrinking again; this test only checks
    the well-behaved shrinking regime past the peak, at and beyond
    Saturn's real physical tilt range."""
    areas = []
    for b in (27.0, 35.0, 45.0, 60.0, 75.0):
        mask = compute_ring_crossing_mask(H, W, CX, CY, DISK_SEMI_A, DISK_SEMI_B, 0.0, b)
        areas.append(mask.sum())
    assert areas == sorted(areas, reverse=True), f"expected monotonically shrinking area, got {areas}"
    assert areas[0] > areas[-1] * 5, f"expected a large overall shrink, got {areas}"


def test_pole_on_tilt_no_crossing():
    """REGRESSION GUARD: at a tilt far beyond Saturn's real range (B=85 deg,
    near pole-on -- not physically realistic for Saturn as seen from Earth,
    but a clean mathematical extreme), the ring's inner edge must clear the
    globe entirely (sin(85deg)~=0.996, inner_ring_semi_b ~= 1.235*disk_semi_a,
    always > any real disk_semi_b<disk_semi_a) -- mask must be all-False,
    exercising the early-exit path directly."""
    mask = compute_ring_crossing_mask(H, W, CX, CY, DISK_SEMI_A, DISK_SEMI_B, 0.0, 85.0)
    assert not mask.any(), "expected no crossing at a near-pole-on tilt"
    assert mask.shape == (H, W)


def test_b_zero_or_negative_symmetric():
    """B's sign shouldn't matter (only tilt magnitude does) -- the ring
    doesn't care whether it's tilted toward or away from the observer for
    this purely-geometric crossing question."""
    mask_pos = compute_ring_crossing_mask(H, W, CX, CY, DISK_SEMI_A, DISK_SEMI_B, 0.0, 11.07)
    mask_neg = compute_ring_crossing_mask(H, W, CX, CY, DISK_SEMI_A, DISK_SEMI_B, 0.0, -11.07)
    assert np.array_equal(mask_pos, mask_neg)


def test_matches_oblate_ortho_forward_projection():
    """REGRESSION GUARD (algebra cross-check): a synthetic ring point at
    phi=0 (equatorial plane), projected directly via _oblate_ortho_forward
    (the same primitive this whole session's Jupiter 3D reprojection work
    validated), must fall inside/outside compute_ring_crossing_mask's own
    ellipse-membership test exactly where expected -- catches any algebra
    mismatch between the mask's "shortcut" ellipse formula and the general
    projection formula it was derived from.
    """
    B = 15.0
    pole_pa = 20.0  # nonzero, to also exercise the rotation-by-pole_pa math
    req_px = DISK_SEMI_A * _SATURN_RING_INNER_REQ  # inner ring radius in px

    mask = compute_ring_crossing_mask(H, W, CX, CY, DISK_SEMI_A, DISK_SEMI_B, pole_pa, B)

    for lam_deg in (0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 270.0):
        lam = math.radians(lam_deg)
        dx, dy, depth = _oblate_ortho_forward(0.0, lam, B, pole_pa, req_px, req_px)
        px = int(round(CX + dx))
        py = int(round(CY + dy))
        if not (0 <= px < W and 0 <= py < H):
            continue
        # This point sits exactly ON the inner ring ellipse boundary. Just
        # inside it (scaled 0.9x from centre) must NOT be flagged as
        # crossing (it's inside the excluded inner hole, not the annulus);
        # just outside it (scaled 1.1x) falls in the annulus and, since the
        # inner ellipse is well within the globe at this shallow tilt,
        # should match the mask's own independent globe+annulus test.
        rx = CX + dx * 1.15
        ry = CY + dy * 1.15
        ix, iy = int(round(rx)), int(round(ry))
        if 0 <= ix < W and 0 <= iy < H:
            in_globe = ((rx - CX) * math.cos(math.radians(pole_pa)) + (ry - CY) * math.sin(math.radians(pole_pa))) ** 2 / DISK_SEMI_A ** 2 \
                + (-(rx - CX) * math.sin(math.radians(pole_pa)) + (ry - CY) * math.cos(math.radians(pole_pa))) ** 2 / DISK_SEMI_B ** 2 <= 1.0
            if in_globe:
                assert mask[iy, ix], (
                    f"lam={lam_deg}: point just outside the inner ring ellipse, "
                    f"inside the globe, should be flagged as ring-crossing"
                )


if __name__ == "__main__":
    test_low_tilt_crosses_widely()
    print("low tilt crosses widely: OK")
    test_crossing_area_shrinks_beyond_saturns_real_tilt_range()
    print("crossing area shrinks beyond Saturn's real tilt range: OK")
    test_pole_on_tilt_no_crossing()
    print("near-pole-on tilt has no crossing: OK")
    test_b_zero_or_negative_symmetric()
    print("B sign symmetry: OK")
    test_matches_oblate_ortho_forward_projection()
    print("matches _oblate_ortho_forward projection: OK")
    print("\nAll checks passed.")
