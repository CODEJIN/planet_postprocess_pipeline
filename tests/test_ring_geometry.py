"""Regression tests for Saturn ring geometry detection (Phase A, 2026-08-11
plan — see project_saturn_composite_alignment_bug memory).

detect_ring_geometry() extends _gradient_disk_r()'s radial-ray steepest-edge
methodology outward past the globe to find the ring's outer-edge ellipse.
This is purely additive — nothing existing calls it yet. These tests validate
the algorithm on synthetic data only; real-data detection accuracy must be
confirmed by the user on actual Saturn frames before any downstream pipeline
step is allowed to depend on this (see the plan's Phase B gate).

Run directly: python3 tests/test_ring_geometry.py
Or via pytest: pytest tests/test_ring_geometry.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.modules.derotation import (
    _SATURN_RING_INNER_OUTER_RATIO,
    detect_ring_geometry,
)


def _make_saturn_like(
    h=460, w=460, cx=230.0, cy=230.0, req_px=70.0, polar_ratio=0.9021,
    ring_outer_frac=2.27, ring_ar=0.6, ring_angle_deg=0.0,
    ring_brightness=0.5, globe_brightness=0.9, background=0.02,
):
    """Synthetic Saturn-like frame: filled ring annulus, globe ellipse drawn
    on top (so it occludes the ring where they overlap, matching how a real
    tilted ring visually crosses in front of/behind the globe)."""
    img = np.full((h, w), background, dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    dx, dy = xx - cx, yy - cy
    ang = np.radians(ring_angle_deg)
    cos_a, sin_a = np.cos(ang), np.sin(ang)
    xr = dx * cos_a + dy * sin_a
    yr = -dx * sin_a + dy * cos_a

    ring_outer_a = req_px * ring_outer_frac
    ring_outer_b = ring_outer_a * ring_ar
    ring_inner_a = ring_outer_a * _SATURN_RING_INNER_OUTER_RATIO
    ring_inner_b = ring_outer_b * _SATURN_RING_INNER_OUTER_RATIO
    outer_val = (xr / ring_outer_a) ** 2 + (yr / ring_outer_b) ** 2
    inner_val = (xr / ring_inner_a) ** 2 + (yr / ring_inner_b) ** 2
    img[(outer_val <= 1.0) & (inner_val >= 1.0)] = ring_brightness

    globe_b = req_px * polar_ratio
    globe_val = (xr / req_px) ** 2 + (yr / globe_b) ** 2
    img[globe_val <= 1.0] = globe_brightness

    return cv2.GaussianBlur(img, (0, 0), 1.0).astype(np.float32)


def test_recovers_outer_ellipse_across_tilt_sweep():
    """REGRESSION GUARD: outer ring radius/angle recovered within a few
    percent across a realistic tilt (ring aspect ratio) sweep."""
    cx, cy, req_px = 230.0, 230.0, 70.0
    for ring_ar in (0.3, 0.5, 0.7, 0.9):
        img = _make_saturn_like(cx=cx, cy=cy, req_px=req_px, ring_ar=ring_ar, ring_angle_deg=10.0)
        ring = detect_ring_geometry(img, cx, cy, req_px, req_px * 0.9021)
        assert ring is not None, f"ring_ar={ring_ar}: failed to detect ring at all"
        true_outer_a = req_px * 2.27
        true_outer_b = true_outer_a * ring_ar
        assert abs(ring.outer_semi_a - true_outer_a) / true_outer_a < 0.06, (
            f"ring_ar={ring_ar}: outer_semi_a={ring.outer_semi_a:.1f} vs true {true_outer_a:.1f}"
        )
        assert abs(ring.outer_semi_b - true_outer_b) / true_outer_b < 0.08, (
            f"ring_ar={ring_ar}: outer_semi_b={ring.outer_semi_b:.1f} vs true {true_outer_b:.1f}"
        )
        # Ellipse angle is only meaningful mod 180 -- shortest circular
        # distance to the true angle 10.0.
        diff = (ring.angle_deg - 10.0) % 180.0
        angle_err = min(diff, 180.0 - diff)
        assert angle_err < 5.0, (
            f"ring_ar={ring_ar}: angle_deg={ring.angle_deg:.1f} not close to 10.0 "
            f"(circular error {angle_err:.1f})"
        )


def test_crosses_disk_flag_matches_tilt():
    """REGRESSION GUARD: crosses_disk must flag True for thin/edge-on-like
    ring tilts and False for more open ones, matching real Saturn geometry
    (rings visually cross the globe only near edge-on/ring-plane-crossing)."""
    cx, cy, req_px = 230.0, 230.0, 70.0

    img_thin = _make_saturn_like(cx=cx, cy=cy, req_px=req_px, ring_ar=0.35)
    ring_thin = detect_ring_geometry(img_thin, cx, cy, req_px, req_px * 0.9021)
    assert ring_thin is not None
    assert ring_thin.crosses_disk is True, "thin/edge-on-like ring should cross the globe silhouette"

    img_open = _make_saturn_like(cx=cx, cy=cy, req_px=req_px, ring_ar=0.95)
    ring_open = detect_ring_geometry(img_open, cx, cy, req_px, req_px * 0.9021)
    assert ring_open is not None
    assert ring_open.crosses_disk is False, "near-face-on ring should NOT cross the globe silhouette"


def test_ringless_disk_returns_none():
    """REGRESSION GUARD: a plain globe with no ring (Jupiter-like) must
    return None, not a spurious ring fit from noise/limb darkening."""
    h, w = 300, 300
    cx, cy, req_px = 150.0, 150.0, 100.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    img = np.full((h, w), 0.02, dtype=np.float32)
    mask = ((xx - cx) / req_px) ** 2 + ((yy - cy) / (req_px * 0.935)) ** 2 <= 1.0
    img[mask] = 0.8
    img = cv2.GaussianBlur(img, (0, 0), 1.5).astype(np.float32)

    ring = detect_ring_geometry(img, cx, cy, req_px, req_px * 0.935)
    assert ring is None, "a ringless disk must not produce a spurious ring detection"


def test_noise_only_image_returns_none():
    """REGRESSION GUARD: pure noise must not produce a plausible-looking
    but meaningless ring fit."""
    rng = np.random.default_rng(0)
    img = rng.uniform(0.0, 0.1, size=(460, 460)).astype(np.float32)
    ring = detect_ring_geometry(img, 230.0, 230.0, 70.0, 63.0)
    assert ring is None, f"expected None on pure noise, got {ring}"


if __name__ == "__main__":
    test_recovers_outer_ellipse_across_tilt_sweep()
    print("outer ellipse recovered across tilt sweep: OK")
    test_crosses_disk_flag_matches_tilt()
    print("crosses_disk flag matches tilt: OK")
    test_ringless_disk_returns_none()
    print("ringless disk returns None: OK")
    test_noise_only_image_returns_none()
    print("noise-only image returns None: OK")
    print("\nAll checks passed.")
