"""Photometric limb-darkening measurement/fitting (Minnaert-style, target- and
filter-agnostic).

Motivation (see project_map_space_derotation_roadmap memory / WinJUPOS
research, 2026-08-16): this project's history is full of white-rim/ringing
artifacts near the disk limb (0b9bc45, 5e6025f, this session's map-space
Phase B) that all trace back to the same root cause -- sharpen_disk_aware's
edge feathering is a purely GEOMETRIC ramp (pixel distance from an ellipse),
while the real brightness falloff at a planet's limb is a PHOTOMETRIC
phenomenon (limb darkening + optical/seeing PSF blur) with a different shape.
Approximating a smooth photometric curve with a hand-tuned geometric ramp
keeps leaving small value/derivative mismatches that wavelet sharpening
amplifies into visible defects.

This module measures the REAL radial brightness profile directly from a
stacked disk image and fits a smooth analytic curve to it (a simplified,
single-free-exponent Minnaert law). Being fit from the actual data rather
than a hardcoded per-target/per-filter constant, it naturally handles cases
like Jupiter/Saturn's methane-absorption band (~889nm), which is known to
show LIMB BRIGHTENING (high-altitude haze scattering) rather than darkening
-- no filter-name special-casing needed (project_filter_agnostic_design).

Phase A only (see plan): pure measurement + fitting, no pipeline wiring.
Turning a fitted curve into a sharpen_disk_aware() confidence_map is a later
phase.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit


def _ellipse_normalized_radius(
    shape: Tuple[int, int],
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    angle_deg: float,
) -> np.ndarray:
    """Per-pixel radius normalized so the fitted disk ellipse boundary is
    exactly at 1.0. Under the same orthographic-projection convention
    _oblate_ortho_forward/_oblate_ortho_inverse (derotation.py) already use,
    this normalized radius equals sin(theta), theta = angular distance from
    the sub-observer point -- so cos(theta) = sqrt(max(0, 1 - r_norm^2)) is
    exactly the Minnaert model's cosine term, with no extra derivation
    needed here.
    """
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    dx = xx - cx
    dy = yy - cy
    ang = math.radians(angle_deg)
    ca, sa = math.cos(ang), math.sin(ang)
    x_rot = dx * ca + dy * sa
    y_rot = -dx * sa + dy * ca
    return np.sqrt((x_rot / rx) ** 2 + (y_rot / ry) ** 2)


@dataclass
class RadialProfile:
    r_norm: np.ndarray      # bin centers (ellipse-normalized radius)
    brightness: np.ndarray  # robust (median) brightness per bin
    counts: np.ndarray      # number of pixels contributing to each bin


def measure_radial_brightness_profile(
    image: np.ndarray,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    angle_deg: float,
    r_max_factor: float = 1.05,
    n_bins: int = 100,
    exclude_mask: Optional[np.ndarray] = None,
    min_pixels_per_bin: int = 20,
) -> RadialProfile:
    """Robust (median-per-bin), azimuthally-averaged radial brightness
    profile of a disk image, normalized so the fitted ellipse boundary sits
    at r_norm=1.0 (see _ellipse_normalized_radius).

    Real surface features (belts, storms) are not explicitly detected --
    the per-bin MEDIAN already rejects them as long as they cover a
    minority of each bin's azimuthal extent, true for real planetary
    bands/storms relative to a full ring of pixels at fixed radius.

    exclude_mask (e.g. a Saturn ring annulus, built by the caller via
    derotation.py's _ring_annulus_mask) removes pixels entirely before
    binning -- required whenever a physically distinct, much brighter
    structure overlaps the profiled region. This function has no notion of
    what a ring is; keeping that out keeps this module target-agnostic.

    Bins with fewer than min_pixels_per_bin surviving pixels are dropped
    entirely (returned arrays can be shorter than n_bins) rather than
    reporting a noisy/meaningless median from a handful of pixels.
    """
    if image.ndim == 3:
        image = image.mean(axis=2)
    r_norm_grid = _ellipse_normalized_radius(image.shape, cx, cy, rx, ry, angle_deg)

    valid = np.isfinite(image) & (r_norm_grid <= r_max_factor)
    if exclude_mask is not None:
        valid &= ~(exclude_mask > 0.5)

    edges = np.linspace(0.0, r_max_factor, n_bins + 1)
    bin_idx = np.clip(np.digitize(r_norm_grid[valid], edges) - 1, 0, n_bins - 1)
    vals = image[valid]

    r_centers, medians, counts = [], [], []
    for b in range(n_bins):
        sel = bin_idx == b
        n = int(sel.sum())
        if n < min_pixels_per_bin:
            continue
        r_centers.append(0.5 * (edges[b] + edges[b + 1]))
        medians.append(float(np.median(vals[sel])))
        counts.append(n)

    return RadialProfile(
        r_norm=np.array(r_centers, dtype=np.float64),
        brightness=np.array(medians, dtype=np.float64),
        counts=np.array(counts, dtype=np.int64),
    )


@dataclass
class LimbDarkeningFit:
    i0: float
    exponent: float  # "m" in I(theta) = I0 * cos(theta)^m
    r_norm_fit_max: float
    residual_rms: float
    n_points: int


def _minnaert_model(r_norm: np.ndarray, i0: float, exponent: float) -> np.ndarray:
    cos_theta = np.sqrt(np.clip(1.0 - r_norm ** 2, 0.0, 1.0))
    # Guard the base away from exactly 0.0 so negative exponents (limb
    # brightening, e.g. CH4 band) stay finite right at r_norm=1.
    return i0 * np.power(np.maximum(cos_theta, 1e-6), exponent)


def fit_limb_darkening_curve(
    profile: RadialProfile,
    r_norm_fit_max: float = 0.98,
    initial_exponent: float = 0.5,
) -> LimbDarkeningFit:
    """Non-linear least-squares fit of a simplified, single-free-exponent
    Minnaert law I(theta) = I0 * cos(theta)^m to a measured radial profile.

    This uses the near-opposition approximation mu ~= mu0 (incidence angle
    ~= emission angle), standard practice for a single amateur image where
    the two angles can't be separated anyway (see this project's Minnaert/
    WinJUPOS research notes, 2026-08-16). `m` is a free-fit EFFECTIVE
    exponent, not the textbook Minnaert k -- it absorbs both the true k and
    any residual phase-angle asymmetry the near-opposition approximation
    doesn't capture (WinJUPOS's separate "LD angle" parameter; out of scope
    here, see roadmap Phase C). A negative fitted m means the profile gets
    BRIGHTER toward the limb (e.g. expected for a methane-absorption band)
    and falls out of the same fit with no special-casing.

    r_norm_fit_max excludes points very close to the true limb (r_norm->1,
    where cos(theta)->0 and the model is numerically stiff and most
    sensitive to residual ellipse-fit error) from the FIT itself --
    evaluate_limb_darkening_curve() can still be evaluated all the way to
    the edge by the caller.

    Raises ValueError if fewer than 8 profile points survive the
    r_norm_fit_max cut (too little data to trust a 2-parameter fit).
    """
    sel = profile.r_norm <= r_norm_fit_max
    r = profile.r_norm[sel]
    b = profile.brightness[sel]
    if r.size < 8:
        raise ValueError(
            f"only {r.size} profile points <= r_norm_fit_max={r_norm_fit_max}, need >= 8"
        )

    i0_guess = float(b[np.argmin(r)]) if r.size else 1.0
    popt, _ = curve_fit(
        _minnaert_model, r, b, p0=[i0_guess, initial_exponent],
        bounds=([0.0, -5.0], [np.inf, 20.0]), maxfev=10000,
    )
    i0_fit, m_fit = float(popt[0]), float(popt[1])
    residual = b - _minnaert_model(r, i0_fit, m_fit)
    residual_rms = float(np.sqrt(np.mean(residual ** 2)))

    return LimbDarkeningFit(
        i0=i0_fit, exponent=m_fit, r_norm_fit_max=r_norm_fit_max,
        residual_rms=residual_rms, n_points=int(r.size),
    )


def evaluate_limb_darkening_curve(r_norm: np.ndarray, fit: LimbDarkeningFit) -> np.ndarray:
    """Evaluate a fitted curve (see fit_limb_darkening_curve) at arbitrary
    r_norm values -- e.g. all the way to the true limb (r_norm=1) for
    plotting/diagnostics or building a confidence map (see
    build_confidence_map), even though the fit itself excluded that
    unstable tail."""
    return _minnaert_model(np.asarray(r_norm, dtype=np.float64), fit.i0, fit.exponent)


def build_confidence_map(
    shape: Tuple[int, int],
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    angle_deg: float,
    fit: LimbDarkeningFit,
) -> np.ndarray:
    """Evaluate a fitted limb-darkening curve over every pixel of an image
    of the given shape, normalized to 1.0 at disk centre and clipped to
    [0,1] -- a raw confidence signal proportional to the model's predicted
    relative brightness at that pixel.

    Intended as a sharpen_disk_aware() confidence_map input (see
    project_limb_darkening_confidence_map memory, Phase B) -- typically
    reshaped through wavelet.coverage_to_confidence() afterwards for its
    smoothstep + floor, exactly like the existing coverage-based confidence
    map, rather than used raw.

    For a normal darkening fit (exponent > 0) this decreases smoothly from
    1.0 at the centre toward 0.0 at the limb. For a brightening fit
    (exponent < 0, e.g. a methane-absorption band, per this module's
    design -- see fit_limb_darkening_curve) the ratio exceeds 1.0 away from
    centre and gets clipped back to 1.0, i.e. full confidence everywhere
    except literally at the centre -- the correct behaviour with no special
    casing: nothing here should have its gain reduced for being dim if
    dimness isn't actually the local condition.

    Pixels well outside the disk are evaluated by the same smooth formula
    (no special masking) -- the caller's own geometric feather
    (sharpen_disk_aware's weight_map) already zeroes gain out there; this
    map's job is only to shape gain INSIDE that boundary.
    """
    r_norm = _ellipse_normalized_radius(shape, cx, cy, rx, ry, angle_deg)
    predicted = evaluate_limb_darkening_curve(r_norm, fit)
    if fit.i0 > 1e-9:
        conf = predicted / fit.i0
    else:
        conf = np.ones_like(predicted)
    return np.clip(conf, 0.0, 1.0).astype(np.float32)
