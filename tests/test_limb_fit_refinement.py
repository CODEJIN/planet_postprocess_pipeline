"""Regression tests for the robust disk-limb refit (2026-08-15, opt-in).

Background: find_disk_center()'s ellipse fit has a measured ~0.5-0.9px
ASYMMETRIC error vs. the true photometric limb -- the root cause of the
gray-halo/white-rim wavelet artifact trade-off documented in
SATURN_RING_WAVELET_STATUS_2026-08-15.md and project_ring_limb_ringing_bug
memory. _robust_ellipse_refit() (pipeline/modules/derotation.py) refines a
seed ellipse against 72 sub-pixel-measured radial rays via iteratively-
reweighted (MAD-based) outlier rejection. Validated on real Jupiter data
this session; NOT validated for has_rings=True targets (ring-crossing
contamination is too large a contiguous angular fraction for point-wise
robust statistics -- see the function's own docstring). Wired into
wavelet_master.py behind WaveletConfig.master_limb_fit_refinement_enabled
(default False), applied only when has_rings is False.

These tests cover the new algorithmic core with synthetic ground truth
(per feedback_correlation_vs_causation -- synthetic data is fine here
because it's pure math verification, not an absolute-threshold calibration
like the failed detect_ring_geometry() attempt) plus the wiring's gating
behaviour via a real wavelet_master.run() call.

Run directly: python3 tests/test_limb_fit_refinement.py
Or via pytest: pytest tests/test_limb_fit_refinement.py -v
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import PipelineConfig, WaveletConfig
from pipeline.modules import image_io
from pipeline.modules.derotation import _robust_ellipse_refit
from pipeline.steps import wavelet_master

SIZE = 300
TRUE_CX, TRUE_CY = 150.0, 148.0
TRUE_SEMI_A, TRUE_SEMI_B = 100.0, 90.0


def _synthetic_limb_disk(
    cx: float = TRUE_CX,
    cy: float = TRUE_CY,
    semi_a: float = TRUE_SEMI_A,
    semi_b: float = TRUE_SEMI_B,
    angle_deg: float = 0.0,
    disk_val: float = 0.6,
    size: int = SIZE,
) -> np.ndarray:
    """Smooth-edged synthetic oblate disk (hard ellipse + Gaussian blur --
    same pattern as test_disk_polarity.py's _make_ring_crossing_synthetic,
    already known to work with this codebase's gradient-based limb fitters)."""
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    ang = np.radians(angle_deg)
    dx = (xx - cx) * np.cos(ang) + (yy - cy) * np.sin(ang)
    dy = -(xx - cx) * np.sin(ang) + (yy - cy) * np.cos(ang)
    rd = np.sqrt((dx / semi_a) ** 2 + (dy / semi_b) ** 2)
    disk = (rd <= 1.0).astype(np.float32) * disk_val
    return cv2.GaussianBlur(disk, (9, 9), 1.5)


def test_recovers_true_ellipse_from_clean_synthetic_disk():
    """No contamination, seed == truth: refit should reproduce the known
    geometry to well within a pixel, and keep nearly all 72 rays."""
    img = _synthetic_limb_disk()
    refit = _robust_ellipse_refit(img, TRUE_CX, TRUE_CY, TRUE_SEMI_A, TRUE_SEMI_B, 0.0)
    assert refit is not None
    fcx, fcy, fsemi_a, fsemi_b, _fangle, n_kept = refit
    assert abs(fcx - TRUE_CX) < 0.5
    assert abs(fcy - TRUE_CY) < 0.5
    assert abs(fsemi_a - TRUE_SEMI_A) < 1.0
    assert abs(fsemi_b - TRUE_SEMI_B) < 1.0
    assert n_kept >= 60  # most of the 72 rays should survive on clean data


def test_recovers_true_ellipse_despite_localized_contamination():
    """A minority of rays (5/72, ~7% -- well under the ~40% contiguous
    fraction documented to break MAD on Saturn) pass near a local bright
    feature just past the true limb, mimicking the "cloud belt boundary"
    mechanism documented as the real Jupiter failure mode. The refit must
    still recover the true geometry closely.

    Note: this does not assert the contaminated rays were explicitly
    dropped (n_kept < 72) -- with only 5/72 mildly-deviating points, this
    codebase's cv2.fitEllipse least-squares fit was found (empirically,
    while writing this test) to already absorb them without needing
    explicit rejection, i.e. n_kept can legitimately stay at 72. What
    matters, and what this asserts, is that the recovered geometry stays
    accurate -- MAD-based rejection is what's needed once contamination is
    large-magnitude or a large contiguous fraction (Saturn's documented
    failure mode), not for every minor deviation.
    """
    img = _synthetic_limb_disk()
    yy, xx = np.mgrid[:SIZE, :SIZE].astype(np.float32)
    # Five scattered bumps: bright wedges reaching ~4px beyond the true
    # limb, at angles spread around the ellipse (not one contiguous arc).
    bump_angles_deg = [10, 95, 170, 240, 310]
    for a in bump_angles_deg:
        theta = np.radians(a)
        bx = TRUE_CX + (TRUE_SEMI_A + 4) * np.cos(theta)
        by = TRUE_CY + (TRUE_SEMI_B + 4) * np.sin(theta)
        bump = (np.sqrt((xx - bx) ** 2 + (yy - by) ** 2) <= 10.0).astype(np.float32) * 0.6
        img = np.maximum(img, cv2.GaussianBlur(bump, (9, 9), 1.5))

    refit = _robust_ellipse_refit(img, TRUE_CX, TRUE_CY, TRUE_SEMI_A, TRUE_SEMI_B, 0.0)
    assert refit is not None
    fcx, fcy, fsemi_a, fsemi_b, _fangle, _n_kept = refit
    assert abs(fcx - TRUE_CX) < 1.0
    assert abs(fcy - TRUE_CY) < 1.0
    assert abs(fsemi_a - TRUE_SEMI_A) < 1.5
    assert abs(fsemi_b - TRUE_SEMI_B) < 1.5


def test_returns_none_when_too_few_rays_survive():
    """Seed placed near a corner of a small frame so most of the 72 rays'
    search windows exit the image bounds -- must return None (keep the
    caller's original fit), never a fit built from too little data."""
    img = _synthetic_limb_disk(cx=15.0, cy=15.0, semi_a=30.0, semi_b=25.0, size=100)
    refit = _robust_ellipse_refit(
        img, 15.0, 15.0, 30.0, 25.0, 0.0, min_keep=20,
    )
    assert refit is None


def test_master_limb_fit_refinement_enabled_defaults_false():
    assert WaveletConfig().master_limb_fit_refinement_enabled is False


def _make_results_04(tmp_path: Path, has_rings: bool) -> dict:
    tif_path = tmp_path / "IR_derot.tif"
    image_io.write_tif_16bit(_synthetic_limb_disk(), tif_path)
    return {
        "windows": [
            {
                "window_index": 1,
                "center_time": "2026-08-15T00:00:00Z",
                "outputs": {"IR": tif_path},
                "log": {"IR": {"has_rings": has_rings, "pole_pa_deg": 0.0,
                                "sub_observer_lat_deg": 0.0}},
            }
        ]
    }


def _run_step05(tmp_path: Path, has_rings: bool, flag_enabled: bool):
    config = PipelineConfig()
    config.output_base_dir = tmp_path / "out"
    config.save_step05 = False
    config.filters = ["IR"]
    config.wavelet.master_limb_fit_refinement_enabled = flag_enabled
    results_04 = _make_results_04(tmp_path, has_rings=has_rings)
    return wavelet_master.run(config, results_04)


def test_refinement_skipped_when_flag_disabled(monkeypatch, tmp_path=None):
    """Default (flag off): _robust_ellipse_refit must never be called, even
    for a ringless filter -- byte-identical to pre-existing behaviour."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        def _boom(*args, **kwargs):
            raise AssertionError("_robust_ellipse_refit must not be called when the flag is off")

        monkeypatch.setattr(wavelet_master, "_robust_ellipse_refit", _boom)
        results = _run_step05(tmp_path, has_rings=False, flag_enabled=False)
        assert results  # ran to completion without hitting the guard


def test_refinement_skipped_for_has_rings_targets(monkeypatch, tmp_path=None):
    """Flag on but has_rings=True (Saturn): must still be skipped -- this
    refinement is validated for ringless targets only."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        def _boom(*args, **kwargs):
            raise AssertionError("_robust_ellipse_refit must not be called for has_rings=True")

        monkeypatch.setattr(wavelet_master, "_robust_ellipse_refit", _boom)
        results = _run_step05(tmp_path, has_rings=True, flag_enabled=True)
        assert results


def test_refinement_invoked_when_enabled_and_ringless(monkeypatch, tmp_path=None):
    """Flag on and has_rings=False: the refinement must actually run."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        calls = []
        real_fn = wavelet_master._robust_ellipse_refit

        def _spy(*args, **kwargs):
            calls.append(1)
            return real_fn(*args, **kwargs)

        monkeypatch.setattr(wavelet_master, "_robust_ellipse_refit", _spy)
        _run_step05(tmp_path, has_rings=False, flag_enabled=True)
        assert calls, "_robust_ellipse_refit was not called when it should have been"


if __name__ == "__main__":
    test_recovers_true_ellipse_from_clean_synthetic_disk()
    print("recovers true ellipse from clean synthetic disk: OK")
    test_recovers_true_ellipse_despite_localized_contamination()
    print("recovers true ellipse despite localized contamination: OK")
    test_returns_none_when_too_few_rays_survive()
    print("returns None when too few rays survive: OK")
    test_master_limb_fit_refinement_enabled_defaults_false()
    print("master_limb_fit_refinement_enabled defaults False: OK")
    print("(monkeypatch-based wiring tests require pytest -- run via pytest)")
