"""Regression tests for the navigation-constrained limb fit (2026-08-16, opt-in).

Background: find_disk_center()'s ellipse fit has a measured ~0.5-0.9px
ASYMMETRIC error vs. the true photometric limb near Saturn's ring ansae --
the root cause documented in SATURN_RING_WAVELET_STATUS_2026-08-15.md and
project_ring_limb_ringing_bug memory. The statistical fix
(_robust_ellipse_refit, MAD-based) is a documented near-no-op on Saturn:
ring contamination is a large CONTIGUOUS ~40% angular arc, not scattered
points, so point-wise robust statistics can't tell "contaminated majority"
from "consensus". A prior scratch experiment
(experiments/scratch_globe_fit_asymmetry_diagnosis.py) also tried excluding
ring-adjacent rays before a free 5-parameter re-fit -- insufficient.

_navigation_constrained_ellipse_fit() instead fixes orientation (pole_pa_deg,
already measured independent of any ellipse fit) and apparent aspect ratio
(analytically predicted from Horizons B + the planet's TRUE physical
oblateness, via _predicted_apparent_ratio()) BEFORE looking at any ray data,
then fits only (cx, cy, scale) from the ring-free angular sectors -- a
heavily over-determined, well-conditioned problem regardless of which
contiguous arc is missing. Wired into wavelet_master.py behind
WaveletConfig.master_navigation_limb_fit_enabled (default False), applied
only when has_rings is True (the has_rings=True counterpart to
master_limb_fit_refinement_enabled, which is has_rings=False only).

Run directly: python3 tests/test_navigation_limb_fit.py
Or via pytest: pytest tests/test_navigation_limb_fit.py -v
"""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import PipelineConfig, WaveletConfig
from pipeline.modules import image_io
from pipeline.modules.derotation import (
    _predicted_apparent_ratio,
    _ring_contaminated_theta_mask,
    _fixed_shape_circle_fit,
    _navigation_constrained_ellipse_fit,
    _SATURN_RING_INNER_REQ,
    _SATURN_RING_OUTER_REQ,
)
from pipeline.steps import wavelet_master

SIZE = 300
TRUE_CX, TRUE_CY = 150.0, 148.0
TRUE_SEMI_A = 60.0  # ring outer edge (~2.27x) stays within the 300px canvas
TRUE_RATIO_SATURN = 0.9021  # Saturn's real Rpol/Req


# ── 1. _predicted_apparent_ratio ───────────────────────────────────────────────

def test_apparent_ratio_at_b_zero_equals_true_ratio():
    """Edge-on (B=0): apparent ratio should equal the true physical ratio
    exactly (maximal flattening, no foreshortening relief)."""
    r = _predicted_apparent_ratio(TRUE_RATIO_SATURN, 0.0)
    assert abs(r - TRUE_RATIO_SATURN) < 1e-9


def test_apparent_ratio_at_b_ninety_is_circular():
    """Pole-on (B=90 deg): apparent ratio should be exactly 1.0 (circular)."""
    r = _predicted_apparent_ratio(TRUE_RATIO_SATURN, 90.0)
    assert abs(r - 1.0) < 1e-9


def test_apparent_ratio_is_symmetric_in_sign_of_b():
    r_pos = _predicted_apparent_ratio(TRUE_RATIO_SATURN, 20.0)
    r_neg = _predicted_apparent_ratio(TRUE_RATIO_SATURN, -20.0)
    assert abs(r_pos - r_neg) < 1e-12


def test_apparent_ratio_is_monotonic_in_b():
    bs = [0.0, 10.0, 20.0, 30.0, 45.0, 60.0, 75.0, 90.0]
    ratios = [_predicted_apparent_ratio(TRUE_RATIO_SATURN, b) for b in bs]
    assert all(a <= b + 1e-12 for a, b in zip(ratios, ratios[1:]))
    assert ratios[0] < ratios[-1]


def test_apparent_ratio_matches_oblate_ortho_forward_numeric_envelope():
    """Independent cross-check against the codebase's own already-tested
    3D projection (_oblate_ortho_forward), not just this function's own
    algebra: numerically trace the projected silhouette envelope (max |dx|,
    max |dy| over the visible surface) and compare its ratio to the
    closed-form prediction."""
    from pipeline.modules.derotation import _oblate_ortho_forward

    req, rpol = 100.0, 100.0 * TRUE_RATIO_SATURN
    phi = np.linspace(-math.pi / 2, math.pi / 2, 361)
    lam = np.linspace(0.0, 2.0 * math.pi, 721)
    PHI, LAM = np.meshgrid(phi, lam, indexing="ij")
    for B in (0.0, 15.0, 45.0, 75.0, 90.0, -30.0):
        dx, dy, depth = _oblate_ortho_forward(PHI, LAM, B, 0.0, req, rpol)
        visible = depth >= 0
        numeric_ratio = float(np.abs(dy[visible]).max() / np.abs(dx[visible]).max())
        formula_ratio = _predicted_apparent_ratio(TRUE_RATIO_SATURN, B)
        assert abs(numeric_ratio - formula_ratio) < 1e-3, (B, numeric_ratio, formula_ratio)


# ── 2. _ring_contaminated_theta_mask ───────────────────────────────────────────

def _reference_theta_exclusion(
    thetas_deg, cx, cy, disk_semi_a, disk_semi_b, pole_pa_deg, sub_observer_lat_deg,
    outer_safety_factor=1.35,
):
    """Independent re-implementation of the ring-annulus membership test at
    the seed ellipse's own boundary point, mirroring `_ring_globe_overlap_
    ellipses`/`compute_ring_sharpening_mask`'s pixel-grid formulas applied
    to a single point per theta -- used to check `_ring_contaminated_theta_
    mask` against the definition it's supposed to implement, independent of
    its own (possibly buggy) code path."""
    sin_b = abs(math.sin(math.radians(sub_observer_lat_deg)))
    inner_a = disk_semi_a * _SATURN_RING_INNER_REQ
    inner_b = inner_a * sin_b
    outer_a = disk_semi_a * _SATURN_RING_OUTER_REQ * outer_safety_factor
    outer_b = max(outer_a * sin_b, 1e-6)
    out = []
    for theta_deg in thetas_deg:
        theta = math.radians(theta_deg)
        # Boundary point of the (pole_pa-oriented) seed ellipse at this
        # image-frame direction, expressed directly in the ellipse-aligned
        # frame (pole_pa_deg cancels out of this local test by construction
        # since both the seed ellipse and the ring share the same pole_pa).
        ang = math.radians(pole_pa_deg)
        dxu, dyu = math.cos(theta), math.sin(theta)
        dxr = math.cos(ang) * dxu + math.sin(ang) * dyu
        dyr = -math.sin(ang) * dxu + math.cos(ang) * dyu
        r_ell = 1.0 / math.sqrt((dxr / disk_semi_a) ** 2 + (dyr / disk_semi_b) ** 2)
        xr, yr = dxr * r_ell, dyr * r_ell
        in_outer = (xr / outer_a) ** 2 + (yr / outer_b) ** 2 <= 1.0
        in_inner = (xr / inner_a) ** 2 + (yr / max(inner_b, 1e-6)) ** 2 <= 1.0
        out.append(in_outer and not in_inner)
    return np.array(out)


def test_ring_mask_matches_independent_reference_computation():
    """Check `_ring_contaminated_theta_mask` against an independently
    re-derived reference at real-Saturn-like geometry (B=-11.07 deg,
    ratio=0.9021, matching the documented window_01 case) -- exact agreement
    at every sampled theta, not just a hand-picked subset."""
    thetas = np.arange(0.0, 360.0, 5.0)
    excluded = _ring_contaminated_theta_mask(
        thetas, cx=0.0, cy=0.0, disk_semi_a=100.0, disk_semi_b=90.21,
        pole_pa_deg=0.0, sub_observer_lat_deg=-11.07,
    )
    expected = _reference_theta_exclusion(
        thetas, cx=0.0, cy=0.0, disk_semi_a=100.0, disk_semi_b=90.21,
        pole_pa_deg=0.0, sub_observer_lat_deg=-11.07,
    )
    assert np.array_equal(excluded, expected)
    # Sanity: this geometry should exclude a real, non-trivial fraction of
    # rays (contamination exists) but not all of them (poles stay clean).
    assert 0 < excluded.sum() < len(thetas)
    # And it should be excluded near the ring/globe crossing band found by
    # direct inspection (~10-27 deg from the equator on each side/hemisphere)
    # rather than nowhere at all.
    assert excluded[thetas == 15.0][0]
    assert not excluded[thetas == 90.0][0]  # pole: never contaminated here


def test_ring_mask_is_symmetric_under_pole_pa_rotation():
    """Rotating pole_pa_deg by a fixed amount should rotate which thetas get
    excluded by the same amount, since the test only depends on the angle
    relative to pole_pa -- checks the rotation convention is applied
    consistently to both the seed ellipse and the ring geometry."""
    thetas = np.arange(0.0, 360.0, 5.0)
    base = _ring_contaminated_theta_mask(
        thetas, cx=0.0, cy=0.0, disk_semi_a=100.0, disk_semi_b=90.21,
        pole_pa_deg=0.0, sub_observer_lat_deg=-11.07,
    )
    rotated = _ring_contaminated_theta_mask(
        thetas, cx=0.0, cy=0.0, disk_semi_a=100.0, disk_semi_b=90.21,
        pole_pa_deg=40.0, sub_observer_lat_deg=-11.07,
    )
    shift = int(round(40.0 / 5.0))
    assert np.array_equal(rotated, np.roll(base, shift))


def test_ring_mask_all_false_when_ring_plane_edge_on_pole_on_view():
    """B=90 (pole-on): sin(B)=1 so the ring's projected minor axis is
    largest, but the ring's inner/outer radii (>1.0x disk_semi_a) still sit
    entirely outside the globe silhouette at every angle when disk_semi_b
    equals disk_semi_a (circular globe, pole-on) -- sanity check that the
    mask doesn't spuriously exclude everything."""
    thetas = np.arange(0.0, 360.0, 30.0)
    excluded = _ring_contaminated_theta_mask(
        thetas, cx=0.0, cy=0.0, disk_semi_a=100.0, disk_semi_b=100.0,
        pole_pa_deg=0.0, sub_observer_lat_deg=90.0,
    )
    assert not excluded.any()


# ── 3. _fixed_shape_circle_fit ─────────────────────────────────────────────────

def test_fixed_shape_circle_fit_recovers_known_ellipse():
    ratio = 0.85
    angle_deg = 20.0
    cx, cy, semi_a = 150.0, 140.0, 80.0
    thetas = np.radians(np.linspace(0.0, 360.0, 72, endpoint=False))
    ang = math.radians(angle_deg)
    # Points on an ellipse with the given angle/ratio, in image coordinates.
    xr = semi_a * np.cos(thetas)
    yr = semi_a * ratio * np.sin(thetas)
    x = cx + xr * math.cos(ang) - yr * math.sin(ang)
    y = cy + xr * math.sin(ang) + yr * math.cos(ang)
    pts = np.column_stack([x, y])

    fit = _fixed_shape_circle_fit(pts, angle_deg, ratio)
    assert fit is not None
    fcx, fcy, fsemi_a = fit
    assert abs(fcx - cx) < 1e-6
    assert abs(fcy - cy) < 1e-6
    assert abs(fsemi_a - semi_a) < 1e-6


def test_fixed_shape_circle_fit_returns_none_on_too_few_points():
    pts = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    assert _fixed_shape_circle_fit(pts, 0.0, 0.9) is None


# ── 4. _navigation_constrained_ellipse_fit (synthetic end-to-end) ─────────────

def _synthetic_saturn_disk(
    cx: float, cy: float, semi_a: float, ratio: float, pole_pa_deg: float,
    sub_observer_lat_deg: float, size: int = SIZE, ring_val: float = 0.4,
    disk_val: float = 0.6,
) -> np.ndarray:
    """Smooth-edged oblate disk (known true geometry) with a bright Saturn
    ring annulus (real IAU inner/outer radius ratios) drawn crossing the
    ansae -- reproduces the real contamination geometry that biases
    cv2.fitEllipse near theta=0/180, following test_disk_polarity.py's
    _make_ring_crossing_synthetic pattern."""
    semi_b = semi_a * ratio
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    ang = math.radians(pole_pa_deg)
    dx = (xx - cx) * math.cos(ang) + (yy - cy) * math.sin(ang)
    dy = -(xx - cx) * math.sin(ang) + (yy - cy) * math.cos(ang)
    disk = ((dx / semi_a) ** 2 + (dy / semi_b) ** 2 <= 1.0).astype(np.float32) * disk_val

    sin_b = abs(math.sin(math.radians(sub_observer_lat_deg)))
    inner_a = semi_a * _SATURN_RING_INNER_REQ
    inner_b = inner_a * sin_b
    outer_a = semi_a * _SATURN_RING_OUTER_REQ
    outer_b = outer_a * sin_b
    in_outer = (dx / outer_a) ** 2 + (dy / max(outer_b, 1e-6)) ** 2 <= 1.0
    in_inner = (dx / inner_a) ** 2 + (dy / max(inner_b, 1e-6)) ** 2 <= 1.0
    ring = (in_outer & ~in_inner).astype(np.float32) * ring_val

    img = np.maximum(disk, ring)
    return cv2.GaussianBlur(img, (9, 9), 1.5)


def _biased_seed_from_image(img: np.ndarray):
    """Reproduce the real failure mode: a plain cv2.fitEllipse on the
    ring-contaminated blob, biased away from the true globe geometry."""
    from pipeline.modules.derotation import find_disk_center
    return find_disk_center(img)


def test_navigation_fit_recovers_true_geometry_despite_ring_contamination():
    pole_pa_deg = 0.0
    B = -11.07  # matches the real documented Saturn window_01 geometry
    true_ratio = _predicted_apparent_ratio(TRUE_RATIO_SATURN, B)
    img = _synthetic_saturn_disk(
        TRUE_CX, TRUE_CY, TRUE_SEMI_A, true_ratio, pole_pa_deg, B,
    )
    seed_cx, seed_cy, seed_a, seed_b, _seed_angle = _biased_seed_from_image(img)

    # The seed (plain cv2.fitEllipse on the ring-contaminated blob) should
    # itself be measurably biased vs. the true geometry -- otherwise this
    # test isn't exercising the failure mode it claims to.
    assert abs(seed_a - TRUE_SEMI_A) > 1.0 or abs(seed_b - TRUE_SEMI_A * true_ratio) > 1.0

    fit = _navigation_constrained_ellipse_fit(
        img, seed_cx, seed_cy, seed_a, seed_b, pole_pa_deg, B, TRUE_RATIO_SATURN,
    )
    assert fit is not None
    fcx, fcy, fsemi_a, fsemi_b, fangle, n_kept = fit
    assert abs(fcx - TRUE_CX) < 1.5
    assert abs(fcy - TRUE_CY) < 1.5
    assert abs(fsemi_a - TRUE_SEMI_A) < 1.5
    assert abs(fsemi_b - TRUE_SEMI_A * true_ratio) < 1.5
    assert n_kept >= 20


def test_navigation_fit_returns_none_when_too_few_rays_survive():
    """Seed placed near a corner of a small frame so most ray search windows
    exit the image bounds -- must return None (keep the caller's seed),
    never a fit built from too little data."""
    fit = _navigation_constrained_ellipse_fit(
        np.zeros((100, 100), dtype=np.float32),
        15.0, 15.0, 30.0, 27.0, 0.0, 20.0, TRUE_RATIO_SATURN, min_keep=20,
    )
    assert fit is None


# ── 5. Config default + wiring gating ──────────────────────────────────────────

def test_master_navigation_limb_fit_enabled_defaults_false():
    assert WaveletConfig().master_navigation_limb_fit_enabled is False


def _make_results_04(tmp_path: Path, has_rings: bool) -> dict:
    img = _synthetic_saturn_disk(TRUE_CX, TRUE_CY, TRUE_SEMI_A, 0.9, 0.0, 20.0) \
        if has_rings else \
        cv2.GaussianBlur(
            (np.sqrt(((np.mgrid[:SIZE, :SIZE][1] - TRUE_CX) / TRUE_SEMI_A) ** 2
                      + ((np.mgrid[:SIZE, :SIZE][0] - TRUE_CY) / (TRUE_SEMI_A * 0.9)) ** 2) <= 1.0
             ).astype(np.float32) * 0.6, (9, 9), 1.5)
    tif_path = tmp_path / "IR_derot.tif"
    image_io.write_tif_16bit(img, tif_path)
    return {
        "windows": [
            {
                "window_index": 1,
                "center_time": "2026-08-16T00:00:00Z",
                "outputs": {"IR": tif_path},
                "log": {"IR": {"has_rings": has_rings, "pole_pa_deg": 0.0,
                                "sub_observer_lat_deg": 20.0}},
            }
        ]
    }


def _run_step05(tmp_path: Path, has_rings: bool, flag_enabled: bool):
    config = PipelineConfig()
    config.output_base_dir = tmp_path / "out"
    config.save_step05 = False
    config.filters = ["IR"]
    config.wavelet.master_navigation_limb_fit_enabled = flag_enabled
    results_04 = _make_results_04(tmp_path, has_rings=has_rings)
    return wavelet_master.run(config, results_04)


def test_navigation_fit_skipped_when_flag_disabled(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        def _boom(*args, **kwargs):
            raise AssertionError("_navigation_constrained_ellipse_fit must not be "
                                  "called when the flag is off")

        monkeypatch.setattr(wavelet_master, "_navigation_constrained_ellipse_fit", _boom)
        results = _run_step05(tmp_path, has_rings=True, flag_enabled=False)
        assert results


def test_navigation_fit_skipped_for_ringless_targets(monkeypatch):
    """Flag on but has_rings=False (Jupiter): must still be skipped -- this
    is the has_rings=True counterpart to master_limb_fit_refinement_enabled,
    which owns the ringless case."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        def _boom(*args, **kwargs):
            raise AssertionError("_navigation_constrained_ellipse_fit must not be "
                                  "called for has_rings=False")

        monkeypatch.setattr(wavelet_master, "_navigation_constrained_ellipse_fit", _boom)
        results = _run_step05(tmp_path, has_rings=False, flag_enabled=True)
        assert results


def test_navigation_fit_invoked_when_enabled_and_has_rings(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        calls = []
        real_fn = wavelet_master._navigation_constrained_ellipse_fit

        def _spy(*args, **kwargs):
            calls.append(1)
            return real_fn(*args, **kwargs)

        monkeypatch.setattr(wavelet_master, "_navigation_constrained_ellipse_fit", _spy)
        _run_step05(tmp_path, has_rings=True, flag_enabled=True)
        assert calls, "_navigation_constrained_ellipse_fit was not called when it should have been"


if __name__ == "__main__":
    test_apparent_ratio_at_b_zero_equals_true_ratio()
    print("apparent ratio at B=0 equals true ratio: OK")
    test_apparent_ratio_at_b_ninety_is_circular()
    print("apparent ratio at B=90 is circular: OK")
    test_apparent_ratio_is_symmetric_in_sign_of_b()
    print("apparent ratio symmetric in sign of B: OK")
    test_apparent_ratio_is_monotonic_in_b()
    print("apparent ratio monotonic in B: OK")
    test_apparent_ratio_matches_oblate_ortho_forward_numeric_envelope()
    print("apparent ratio matches _oblate_ortho_forward numeric envelope: OK")
    test_ring_mask_matches_independent_reference_computation()
    print("ring mask matches independent reference computation: OK")
    test_ring_mask_is_symmetric_under_pole_pa_rotation()
    print("ring mask symmetric under pole_pa rotation: OK")
    test_ring_mask_all_false_when_ring_plane_edge_on_pole_on_view()
    print("ring mask sane at pole-on view: OK")
    test_fixed_shape_circle_fit_recovers_known_ellipse()
    print("fixed-shape circle fit recovers known ellipse: OK")
    test_fixed_shape_circle_fit_returns_none_on_too_few_points()
    print("fixed-shape circle fit returns None on too few points: OK")
    test_navigation_fit_recovers_true_geometry_despite_ring_contamination()
    print("navigation fit recovers true geometry despite ring contamination: OK")
    test_navigation_fit_returns_none_when_too_few_rays_survive()
    print("navigation fit returns None when too few rays survive: OK")
    test_master_navigation_limb_fit_enabled_defaults_false()
    print("master_navigation_limb_fit_enabled defaults False: OK")
    print("(monkeypatch-based wiring tests require pytest -- run via pytest)")
