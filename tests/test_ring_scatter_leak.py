"""Tests for estimate_ring_scatter_leak() (pipeline/modules/derotation.py,
2026-08-17, opt-in) and its wiring in wavelet_master.py behind
WaveletConfig.master_ring_scatter_subtraction_enabled.

Background: a prior investigation (project_limb_darkening_confidence_map
memory's "PSF 산란광 가설", reproduced 2026-08-17 across 9 independent
same-night Saturn stacks, IR/R/G/B) found a real, reproducible photometric
excess on Saturn's globe near the ring ansae (phi=0/180 in the pole_pa_deg
frame) over what a symmetric Minnaert limb-darkening fit predicts -- most
likely PSF/optical scattering leaking from the adjacent bright ring. This is
a DIFFERENT phenomenon from the ellipse-fit-asymmetry ringing bug
(project_ring_limb_ringing_bug memory, ~10 rejected mask/gain/filter
attempts) -- this feature is not expected to fix that older bug.

Run directly: python3 tests/test_ring_scatter_leak.py
Or via pytest: pytest tests/test_ring_scatter_leak.py -v
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.modules.derotation import (
    _raised_cosine_falloff,
    _raised_cosine_rise,
    estimate_ring_scatter_leak,
)
from pipeline.modules.limb_darkening import (
    _ellipse_normalized_radius,
    fit_limb_darkening_curve,
    evaluate_limb_darkening_curve,
    measure_radial_brightness_profile,
    LimbDarkeningFit,
)
from pipeline.config import PipelineConfig, WaveletConfig
from pipeline.modules import image_io
from pipeline.steps import wavelet_master

SIZE = 300
CX, CY = 150.0, 150.0
RX, RY = 100.0, 90.0
I0, M = 0.8, 0.6


def _rotated_phi_r(pole_pa_deg=0.0):
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
    ang = np.radians(pole_pa_deg)
    dx, dy = xx - CX, yy - CY
    xr = dx * np.cos(ang) + dy * np.sin(ang)
    yr = -dx * np.sin(ang) + dy * np.cos(ang)
    r_norm = np.sqrt((xr / RX) ** 2 + (yr / RY) ** 2)
    phi_deg = np.degrees(np.arctan2(yr / RY, xr / RX))
    return r_norm, phi_deg


def _clean_minnaert_disk():
    r_norm = _ellipse_normalized_radius((SIZE, SIZE), CX, CY, RX, RY, 0.0)
    cos_theta = np.sqrt(np.clip(1.0 - r_norm ** 2, 0.0, 1.0))
    img = (I0 * np.power(np.maximum(cos_theta, 1e-6), M)).astype(np.float64)
    return np.where(r_norm <= 1.0, img, 0.0).astype(np.float32)


def _fit_from_clean_disk():
    img = _clean_minnaert_disk()
    profile = measure_radial_brightness_profile(img, CX, CY, RX, RY, 0.0)
    return fit_limb_darkening_curve(profile)


def _disk_with_bump(r_norm_center, phi_center_deg, amplitude, width_deg=8.0, width_r=0.02):
    """Clean Minnaert disk plus a localized Gaussian-ish brightness bump at
    a given (r_norm, phi) location in the pole_pa_deg=0 frame."""
    img = _clean_minnaert_disk().astype(np.float64)
    r_norm, phi_deg = _rotated_phi_r(0.0)
    d_phi = np.abs(((phi_deg - phi_center_deg + 180) % 360) - 180)
    bump = amplitude * np.exp(-0.5 * (d_phi / width_deg) ** 2) * \
        np.exp(-0.5 * ((r_norm - r_norm_center) / width_r) ** 2)
    img = np.where(r_norm <= 1.0, img + bump, img)
    return img.astype(np.float32)


# ── 1. _raised_cosine_falloff / _raised_cosine_rise ────────────────────────────

def test_raised_cosine_falloff_endpoints_and_monotonic():
    v = np.linspace(0.0, 10.0, 101)
    f = _raised_cosine_falloff(v, inner=2.0, outer=8.0)
    assert np.all(f[v <= 2.0] == 1.0)
    assert np.all(f[v >= 8.0] == 0.0)
    mid = f[(v > 2.0) & (v < 8.0)]
    assert np.all(np.diff(mid) <= 1e-9)  # non-increasing


def test_raised_cosine_rise_is_one_minus_falloff():
    v = np.linspace(-5.0, 15.0, 101)
    f = _raised_cosine_falloff(v, inner=2.0, outer=8.0)
    r = _raised_cosine_rise(v, inner=2.0, outer=8.0)
    assert np.allclose(f + r, 1.0)


# ── 2. estimate_ring_scatter_leak: pure-function behavior ──────────────────────

def test_leak_is_zero_when_image_matches_ld_model_exactly():
    img = _clean_minnaert_disk()
    fit = _fit_from_clean_disk()
    leak = estimate_ring_scatter_leak(img, CX, CY, RX, RY, 0.0, fit)
    assert leak.max() < 1e-6


def test_leak_removes_most_of_injected_ansa_bump():
    # width_r=0.05 (~5px at RX=100) is wide enough that the function's own
    # blur_sigma_px=2.5 default doesn't itself attenuate the peak much --
    # narrower bumps are legitimately blurred down by that step, this test
    # isolates the windowing/clamp behavior instead.
    amplitude = 0.08
    img = _disk_with_bump(r_norm_center=0.95, phi_center_deg=0.0, amplitude=amplitude,
                           width_deg=10.0, width_r=0.05)
    fit = _fit_from_clean_disk()
    leak = estimate_ring_scatter_leak(img, CX, CY, RX, RY, 0.0, fit)
    ang = np.radians(0.0)
    px = int(round(CX + 0.95 * RX * np.cos(ang)))
    py = int(round(CY + 0.95 * RY * np.sin(ang)))
    window = leak[py - 3:py + 4, px - 3:px + 4]
    assert window.max() >= 0.7 * amplitude


def test_leak_is_near_zero_far_from_ansa():
    """Same bump, but at phi=90 (polar direction) -- the angular window
    should suppress it even though raw excess there is just as large,
    proving the window (not the excess computation) gates the effect."""
    amplitude = 0.08
    img = _disk_with_bump(r_norm_center=0.95, phi_center_deg=90.0, amplitude=amplitude)
    fit = _fit_from_clean_disk()
    leak = estimate_ring_scatter_leak(img, CX, CY, RX, RY, 0.0, fit)
    px = int(round(CX))
    py = int(round(CY + 0.95 * RY))
    window = leak[py - 3:py + 4, px - 3:px + 4]
    assert window.max() < 0.05 * amplitude


def test_leak_is_near_zero_outside_radial_band_inward():
    """Bump at the ansa angle but deep inside the disk (r_norm=0.5) --
    outside the radial window -- should be left alone."""
    amplitude = 0.08
    img = _disk_with_bump(r_norm_center=0.5, phi_center_deg=0.0, amplitude=amplitude, width_r=0.03)
    fit = _fit_from_clean_disk()
    leak = estimate_ring_scatter_leak(img, CX, CY, RX, RY, 0.0, fit)
    ang = np.radians(0.0)
    px = int(round(CX + 0.5 * RX * np.cos(ang)))
    py = int(round(CY + 0.5 * RY * np.sin(ang)))
    window = leak[py - 3:py + 4, px - 3:px + 4]
    assert window.max() < 0.05 * amplitude


def test_leak_never_exceeds_pointwise_excess():
    """Core safety-clamp property: the Gaussian blur inside the function
    must never let the returned leak exceed a pixel's own raw excess over
    the model, even with an irregular synthetic excess field."""
    rng = np.random.default_rng(0)
    img = _clean_minnaert_disk().astype(np.float64)
    r_norm, _ = _rotated_phi_r(0.0)
    bumps = rng.uniform(0.0, 0.1, size=img.shape) * (r_norm <= 1.0)
    img = (img + bumps).astype(np.float32)
    fit = _fit_from_clean_disk()

    predicted = evaluate_limb_darkening_curve(r_norm, fit)
    excess = np.maximum(0.0, img.astype(np.float64) - predicted)

    leak = estimate_ring_scatter_leak(img, CX, CY, RX, RY, 0.0, fit)
    assert np.all(leak <= excess + 1e-6)


def test_corrected_image_bounded_by_predicted_and_original():
    amplitude = 0.08
    img = _disk_with_bump(r_norm_center=0.95, phi_center_deg=180.0, amplitude=amplitude)
    fit = _fit_from_clean_disk()
    leak = estimate_ring_scatter_leak(img, CX, CY, RX, RY, 0.0, fit)
    r_norm, _ = _rotated_phi_r(0.0)
    predicted = evaluate_limb_darkening_curve(r_norm, fit)
    corrected = img.astype(np.float64) - leak
    on_disk = (r_norm <= 1.0)
    # The precise guaranteed bound is corrected >= min(image, predicted), not
    # corrected >= predicted unconditionally: where image is ALREADY below
    # the (imperfectly-fit) model -- e.g. the fit's own residual_rms scatter,
    # nothing to do with this bump -- excess is 0 there by definition, so
    # leak is 0 and corrected==image, which can be (very slightly) below
    # predicted with no leak subtraction involved at all. Where image>=
    # predicted (the actual excess region this function targets), the
    # stronger corrected>=predicted bound does hold, checked separately.
    floor = np.minimum(img.astype(np.float64), predicted)
    assert np.all(corrected[on_disk] >= floor[on_disk] - 1e-6)
    assert np.all(corrected[on_disk] <= img[on_disk] + 1e-6)
    excess_region = on_disk & (img.astype(np.float64) >= predicted)
    assert np.all(corrected[excess_region] >= predicted[excess_region] - 1e-6)


def test_estimate_ring_scatter_leak_rejects_3d_input():
    img = np.zeros((SIZE, SIZE, 3), dtype=np.float32)
    fit = LimbDarkeningFit(i0=0.8, exponent=0.6, r_norm_fit_max=0.98, residual_rms=0.0, n_points=50)
    try:
        estimate_ring_scatter_leak(img, CX, CY, RX, RY, 0.0, fit)
        assert False, "expected ValueError on 3-D input"
    except ValueError:
        pass


# ── 3. Config default + wiring gating ──────────────────────────────────────────

def test_master_ring_scatter_subtraction_enabled_defaults_false():
    assert WaveletConfig().master_ring_scatter_subtraction_enabled is False


def test_master_ring_scatter_subtraction_strength_defaults_to_one():
    assert WaveletConfig().master_ring_scatter_subtraction_strength == 1.0


def _synthetic_saturn_like(has_rings: bool, color: bool = False):
    r_norm = _ellipse_normalized_radius((SIZE, SIZE), CX, CY, RX, RY, 0.0)
    cos_theta = np.sqrt(np.clip(1.0 - r_norm ** 2, 0.0, 1.0))
    img = (I0 * np.power(np.maximum(cos_theta, 1e-6), M)).astype(np.float64)
    if has_rings:
        r_norm2, phi_deg = _rotated_phi_r(0.0)
        d0 = np.abs(((phi_deg - 0.0 + 180) % 360) - 180)
        d180 = np.abs(((phi_deg - 180.0 + 180) % 360) - 180)
        d_ansa = np.minimum(d0, d180)
        bump = 0.06 * np.exp(-0.5 * (d_ansa / 8.0) ** 2) * \
            np.exp(-0.5 * ((r_norm2 - 0.95) / 0.02) ** 2)
        img = img + bump
    img = np.where(r_norm <= 1.0, img, 0.0).astype(np.float32)
    img = cv2.GaussianBlur(img, (5, 5), 1.0)
    if color:
        img = np.stack([img, img, img], axis=-1)
    return img


def _make_results_04(tmp_path: Path, has_rings: bool, color: bool = False) -> dict:
    img = _synthetic_saturn_like(has_rings=has_rings, color=color)
    tif_path = tmp_path / "IR_derot.tif"
    image_io.write_tif_16bit(img, tif_path)
    return {
        "windows": [
            {
                "window_index": 1,
                "center_time": "2026-08-17T00:00:00Z",
                "outputs": {"IR": tif_path},
                "log": {"IR": {"has_rings": has_rings, "pole_pa_deg": 0.0,
                                "sub_observer_lat_deg": 20.0}},
            }
        ]
    }


def _run_step05(tmp_path: Path, has_rings: bool, flag_enabled: bool,
                 ld_enabled: bool = False, color: bool = False, save: bool = False):
    config = PipelineConfig()
    config.output_base_dir = tmp_path / "out"
    config.save_step05 = save
    config.filters = ["IR"]
    config.camera_mode = "color" if color else "mono"
    config.wavelet.master_ring_scatter_subtraction_enabled = flag_enabled
    config.wavelet.master_limb_darkening_confidence_enabled = ld_enabled
    results_04 = _make_results_04(tmp_path, has_rings=has_rings, color=color)
    return wavelet_master.run(config, results_04)


def test_ring_scatter_subtraction_skipped_when_flag_disabled(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        def _boom(*args, **kwargs):
            raise AssertionError("estimate_ring_scatter_leak must not be called when the flag is off")

        monkeypatch.setattr(wavelet_master, "estimate_ring_scatter_leak", _boom)
        results = _run_step05(tmp_path, has_rings=True, flag_enabled=False)
        assert results


def test_ring_scatter_subtraction_skipped_for_ringless_targets(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        def _boom(*args, **kwargs):
            raise AssertionError("estimate_ring_scatter_leak must not be called for has_rings=False")

        monkeypatch.setattr(wavelet_master, "estimate_ring_scatter_leak", _boom)
        results = _run_step05(tmp_path, has_rings=False, flag_enabled=True)
        assert results


def test_ring_scatter_subtraction_invoked_when_enabled_and_has_rings(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        calls = []
        real_fn = wavelet_master.estimate_ring_scatter_leak

        def _spy(*args, **kwargs):
            calls.append(1)
            return real_fn(*args, **kwargs)

        monkeypatch.setattr(wavelet_master, "estimate_ring_scatter_leak", _spy)

        off_dir = tmp_path / "off"
        off_dir.mkdir()
        results_off = _run_step05(off_dir, has_rings=True, flag_enabled=False, save=True)

        on_dir = tmp_path / "on"
        on_dir.mkdir()
        results_on = _run_step05(on_dir, has_rings=True, flag_enabled=True, save=True)

        assert calls, "estimate_ring_scatter_leak was not called when it should have been"

        png_off = results_off["window_01"][0][0]
        png_on = results_on["window_01"][0][0]
        assert png_off is not None and png_on is not None
        img_off = image_io.read_tif(png_off) if str(png_off).endswith(".tif") else cv2.imread(str(png_off), cv2.IMREAD_UNCHANGED)
        img_on = image_io.read_tif(png_on) if str(png_on).endswith(".tif") else cv2.imread(str(png_on), cv2.IMREAD_UNCHANGED)
        assert not np.array_equal(img_off, img_on), "flag on/off produced identical output"


def test_ring_scatter_subtraction_shares_fit_with_ld_confidence(monkeypatch):
    """Both flags on together must fit the LD curve exactly once per filter
    -- proves the shared gating restructuring, guards against a future
    regression re-duplicating the fit."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        calls = []
        real_fn = wavelet_master.fit_limb_darkening_curve

        def _counting(*args, **kwargs):
            calls.append(1)
            return real_fn(*args, **kwargs)

        monkeypatch.setattr(wavelet_master, "fit_limb_darkening_curve", _counting)
        _run_step05(tmp_path, has_rings=True, flag_enabled=True, ld_enabled=True)
        assert len(calls) == 1, f"expected exactly 1 fit call, got {len(calls)}"


def test_ring_scatter_subtraction_degrades_gracefully_on_fit_failure(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("synthetic fit failure")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        monkeypatch.setattr(wavelet_master, "fit_limb_darkening_curve", _boom)
        results = _run_step05(tmp_path, has_rings=True, flag_enabled=True)
        assert results  # run() completes without raising


def test_ring_scatter_subtraction_skips_color_mode(capsys):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        results = _run_step05(tmp_path, has_rings=True, flag_enabled=True, color=True)
        assert results
        captured = capsys.readouterr()
        assert "color_mode" in captured.out or "img.ndim==3" in captured.out


if __name__ == "__main__":
    test_raised_cosine_falloff_endpoints_and_monotonic()
    print("raised cosine falloff endpoints/monotonic: OK")
    test_raised_cosine_rise_is_one_minus_falloff()
    print("raised cosine rise = 1 - falloff: OK")
    test_leak_is_zero_when_image_matches_ld_model_exactly()
    print("leak is zero for clean disk: OK")
    test_leak_removes_most_of_injected_ansa_bump()
    print("leak removes most of injected ansa bump: OK")
    test_leak_is_near_zero_far_from_ansa()
    print("leak near zero far from ansa: OK")
    test_leak_is_near_zero_outside_radial_band_inward()
    print("leak near zero outside radial band: OK")
    test_leak_never_exceeds_pointwise_excess()
    print("leak never exceeds pointwise excess: OK")
    test_corrected_image_bounded_by_predicted_and_original()
    print("corrected image bounded by predicted/original: OK")
    test_estimate_ring_scatter_leak_rejects_3d_input()
    print("rejects 3-D input: OK")
    test_master_ring_scatter_subtraction_enabled_defaults_false()
    print("config defaults False: OK")
    test_master_ring_scatter_subtraction_strength_defaults_to_one()
    print("config strength defaults 1.0: OK")
    print("(monkeypatch-based wiring tests require pytest -- run via pytest)")
