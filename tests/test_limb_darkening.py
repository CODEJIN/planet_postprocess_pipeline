"""Tests for pipeline/modules/limb_darkening.py (Phase A of the Minnaert
limb-darkening confidence_map roadmap, project_map_space_derotation_roadmap
memory's WinJUPOS follow-up). Pure measurement/fitting -- no pipeline wiring
yet, see the module docstring.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.modules.limb_darkening import (
    _ellipse_normalized_radius,
    _minnaert_model,
    measure_radial_brightness_profile,
    fit_limb_darkening_curve,
    evaluate_limb_darkening_curve,
    build_confidence_map,
    RadialProfile,
)

SIZE = 300
CX, CY = 150.0, 150.0
RX, RY = 100.0, 90.0  # oblate, matching how a real Jupiter/Saturn disk fit looks
ANGLE_DEG = 12.0


def _synthetic_disk(i0=0.8, m=0.6, noise_std=0.01, seed=0, belts=True):
    """A synthetic oblate disk following I(theta) = i0 * cos(theta)^m
    exactly, optionally with fake bright/dark belt stripes overlaid (to
    exercise the per-bin-median robustness) and Gaussian noise."""
    rng = np.random.default_rng(seed)
    r_norm = _ellipse_normalized_radius((SIZE, SIZE), CX, CY, RX, RY, ANGLE_DEG)
    cos_theta = np.sqrt(np.clip(1.0 - r_norm ** 2, 0.0, 1.0))
    img = i0 * np.power(np.maximum(cos_theta, 1e-6), m)
    img = img.astype(np.float64)
    on_disk = r_norm <= 1.0

    if belts:
        yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
        # A few horizontal-ish bright/dark stripes (crude belts), offset
        # away from the disk center -- each covers only a minority of the
        # azimuthal ring at any given radius far enough from center, which
        # the per-bin median should reject. (Right at the very center,
        # ANY fixed-width stripe crossing it covers essentially the whole
        # tiny ring there by construction -- not a realistic belt scenario,
        # so tests compare only bins away from the center.)
        for band_y, delta in [(110, 0.15), (130, -0.12), (175, 0.10)]:
            band = np.abs(yy - band_y) < 4
            img = np.where(band & on_disk, np.clip(img + delta, 0.0, 1.0), img)

    img = img + rng.normal(0.0, noise_std, size=img.shape)
    img = np.where(on_disk, img, 0.0).astype(np.float32)
    return img


def test_ellipse_normalized_radius_is_one_on_ellipse_boundary():
    r_norm = _ellipse_normalized_radius((SIZE, SIZE), CX, CY, RX, RY, ANGLE_DEG)
    # A point exactly RX away along the (unrotated) semi-major axis, then
    # rotated into the ellipse's own tilted frame, should land at r_norm~1.
    ang = np.radians(ANGLE_DEG)
    px = int(round(CX + RX * np.cos(ang)))
    py = int(round(CY + RX * np.sin(ang)))
    assert abs(r_norm[py, px] - 1.0) < 0.02


def test_ellipse_normalized_radius_is_zero_at_center():
    r_norm = _ellipse_normalized_radius((SIZE, SIZE), CX, CY, RX, RY, ANGLE_DEG)
    assert r_norm[int(CY), int(CX)] < 1e-9


def test_measure_radial_brightness_profile_monotonic_for_darkening_disk():
    img = _synthetic_disk(m=0.6, noise_std=0.0, belts=False)
    profile = measure_radial_brightness_profile(img, CX, CY, RX, RY, ANGLE_DEG)
    assert profile.r_norm.size > 20
    # A darkening (m>0) profile should decrease (allow tiny numerical noise).
    assert np.all(np.diff(profile.brightness) < 1e-6)


def test_measure_radial_brightness_profile_rejects_belts_via_median():
    clean = _synthetic_disk(m=0.6, noise_std=0.0, belts=False)
    banded = _synthetic_disk(m=0.6, noise_std=0.0, belts=True)
    p_clean = measure_radial_brightness_profile(clean, CX, CY, RX, RY, ANGLE_DEG)
    p_banded = measure_radial_brightness_profile(banded, CX, CY, RX, RY, ANGLE_DEG)
    # Bins overlapping a belt should still be close to the belt-free profile
    # (median rejects the minority-coverage stripe), not shifted by the
    # belt's full amplitude (0.10-0.15).
    n = min(p_clean.r_norm.size, p_banded.r_norm.size)
    # Skip the handful of bins closest to the center: a ring that small has
    # so little azimuthal extent that any fixed-width stripe crossing near
    # the center covers most/all of it regardless of median -- not what
    # this test is checking (real belts sit at moderate-to-large radius).
    far_enough = p_clean.r_norm[:n] > 0.2
    diff = np.abs(p_clean.brightness[:n][far_enough] - p_banded.brightness[:n][far_enough])
    assert diff.max() < 0.05, f"max diff {diff.max():.4f} -- belts leaking into profile"


def test_measure_radial_brightness_profile_exclude_mask_removes_pixels():
    img = _synthetic_disk(m=0.6, noise_std=0.0, belts=False)
    # Fake "ring" region: a bright annulus just outside the globe, at radii
    # this profile call will otherwise include (r_max_factor default 1.05).
    r_norm = _ellipse_normalized_radius((SIZE, SIZE), CX, CY, RX, RY, ANGLE_DEG)
    ring_region = (r_norm > 1.0) & (r_norm < 1.05)
    img_with_ring = img.copy()
    img_with_ring[ring_region] = 5.0  # absurdly bright, would corrupt outer bins

    no_exclude = measure_radial_brightness_profile(
        img_with_ring, CX, CY, RX, RY, ANGLE_DEG, r_max_factor=1.05,
    )
    excluded = measure_radial_brightness_profile(
        img_with_ring, CX, CY, RX, RY, ANGLE_DEG, r_max_factor=1.05,
        exclude_mask=ring_region,
    )
    # Outermost surviving bin: without exclusion it's dominated by the fake
    # ring; with exclusion it should be back near the true (dark, near-limb)
    # disk brightness.
    assert no_exclude.brightness[-1] > 1.0
    assert excluded.brightness[-1] < 0.5


def test_measure_radial_brightness_profile_drops_sparse_bins():
    img = _synthetic_disk(m=0.6, noise_std=0.0, belts=False)
    profile = measure_radial_brightness_profile(
        img, CX, CY, RX, RY, ANGLE_DEG, n_bins=100, min_pixels_per_bin=20,
    )
    assert np.all(profile.counts >= 20)


def test_fit_limb_darkening_curve_recovers_known_exponent():
    for true_m in [0.3, 0.8, 1.5]:
        img = _synthetic_disk(i0=0.8, m=true_m, noise_std=0.005, seed=1, belts=True)
        profile = measure_radial_brightness_profile(img, CX, CY, RX, RY, ANGLE_DEG)
        fit = fit_limb_darkening_curve(profile)
        assert abs(fit.exponent - true_m) < 0.1, (
            f"true m={true_m} fitted m={fit.exponent:.3f}"
        )
        assert abs(fit.i0 - 0.8) < 0.05


def test_fit_limb_darkening_curve_recovers_negative_exponent_limb_brightening():
    """CH4-band-like case: brightness INCREASES toward the limb (negative m)
    -- must fit with no special-casing, per project_filter_agnostic_design."""
    img = _synthetic_disk(i0=0.5, m=-0.4, noise_std=0.005, seed=2, belts=False)
    profile = measure_radial_brightness_profile(img, CX, CY, RX, RY, ANGLE_DEG)
    fit = fit_limb_darkening_curve(profile)
    assert fit.exponent < 0.0
    assert abs(fit.exponent - (-0.4)) < 0.15


def test_fit_limb_darkening_curve_raises_on_too_few_points():
    profile = RadialProfile(
        r_norm=np.array([0.1, 0.2, 0.3]),
        brightness=np.array([0.8, 0.7, 0.6]),
        counts=np.array([100, 100, 100]),
    )
    with pytest.raises(ValueError):
        fit_limb_darkening_curve(profile)


def test_evaluate_limb_darkening_curve_matches_underlying_model():
    """evaluate_limb_darkening_curve is a thin wrapper around the same
    _minnaert_model the fit itself uses -- this checks that wiring, not the
    fit's own residual (which has an expected small nonzero floor from
    per-bin discretization of a nonlinear function, covered by the
    recovers_known_exponent tests instead)."""
    img = _synthetic_disk(i0=0.8, m=0.6, noise_std=0.0, belts=False)
    profile = measure_radial_brightness_profile(img, CX, CY, RX, RY, ANGLE_DEG)
    fit = fit_limb_darkening_curve(profile)
    r = np.linspace(0.0, 1.0, 50)
    predicted = evaluate_limb_darkening_curve(r, fit)
    expected = _minnaert_model(r, fit.i0, fit.exponent)
    assert np.array_equal(predicted, expected)


def test_evaluate_limb_darkening_curve_extends_to_true_limb():
    img = _synthetic_disk(i0=0.8, m=0.6, noise_std=0.0, belts=False)
    profile = measure_radial_brightness_profile(img, CX, CY, RX, RY, ANGLE_DEG)
    fit = fit_limb_darkening_curve(profile)
    at_limb = evaluate_limb_darkening_curve(np.array([1.0]), fit)
    assert np.isfinite(at_limb[0])
    assert at_limb[0] >= 0.0


def test_build_confidence_map_is_one_at_center_and_decreases_for_darkening():
    img = _synthetic_disk(i0=0.8, m=0.6, noise_std=0.0, belts=False)
    profile = measure_radial_brightness_profile(img, CX, CY, RX, RY, ANGLE_DEG)
    fit = fit_limb_darkening_curve(profile)
    conf = build_confidence_map((SIZE, SIZE), CX, CY, RX, RY, ANGLE_DEG, fit)
    assert conf.shape == (SIZE, SIZE)
    assert abs(conf[int(CY), int(CX)] - 1.0) < 1e-3
    assert conf.min() >= 0.0 and conf.max() <= 1.0
    # Along +x from center, confidence should not increase (darkening fit).
    row = conf[int(CY), int(CX):int(CX) + int(RX)]
    assert np.all(np.diff(row) <= 1e-6)


def test_build_confidence_map_stays_near_one_for_brightening_fit():
    img = _synthetic_disk(i0=0.5, m=-0.4, noise_std=0.0, belts=False)
    profile = measure_radial_brightness_profile(img, CX, CY, RX, RY, ANGLE_DEG)
    fit = fit_limb_darkening_curve(profile)
    conf = build_confidence_map((SIZE, SIZE), CX, CY, RX, RY, ANGLE_DEG, fit)
    # A brightening fit should clip to ~1.0 almost everywhere on the disk,
    # not get scaled down for being "dim" when it isn't.
    on_disk = _ellipse_normalized_radius((SIZE, SIZE), CX, CY, RX, RY, ANGLE_DEG) < 0.9
    assert conf[on_disk].min() > 0.95


if __name__ == "__main__":
    import inspect
    mod = sys.modules[__name__]
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        if name.startswith("test_"):
            fn()
            print(f"{name}: OK")
    print("\nAll checks passed.")
