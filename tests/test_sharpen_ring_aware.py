"""Regression tests for sharpen_disk_aware()'s optional second-ellipse
support (extra_rx/extra_ry/extra_angle/extra_gap_px), added 2026-08-12/13.

Root cause this fixes: sharpen_disk_aware()'s spatial feather mask is zero
beyond the primary disk radius (radius+expand_px), which silently zeroed all
wavelet sharpening gain over Saturn's rings in step05 (wavelet_master.py) --
the actual cause of the Cassini Division vanishing in the de-rotated stack
while remaining visible in step07 (which uses plain, unmasked sharpen()).
See project_derotation_ring_occlusion_fix memory and this session's
re-investigation workflow for the full diagnosis.

Three more real-data issues surfaced after the first fix shipped, in
sequence, each fixed before the next was found:

1. Covering the ring as a SOLID second ellipse (reaching all the way to the
   centre) backfills gain=1 immediately outside the primary disk's own
   feather zone, defeating that feather's purpose of giving the disk's real
   limb gradient a quiet, unboosted buffer -- visible ringing at the
   globe/ring border ("abnormal line", user-reported) that wasn't present in
   the pre-sharpen data at all.
2. A fix attempt carved a hole with a SEPARATE ellipse at the ring's own
   (very flat) eccentricity -- but any two co-centred ellipses of different
   eccentricity cross at some off-axis angle, and right at that crossing the
   hole stopped reaching the near-circular disk's actual edge, leaving a
   wedge of still-bright, still-unboosted pixels amid sharpened neighbours
   (a visible cusp, confirmed by directly visualising the mask).
3. Replacing the hole with a uniform pixel-distance HARD exclusion band
   (extra_gap_px, weight forced to exactly 0 for that many pixels, then a
   separate fade-in beyond it) fixed the crossing issue, but on real
   multi-frame Saturn stacks that "gap" isn't actually empty -- it's the
   disk's own fairly bright limb-darkening tail -- so the flat zero band
   read as its own visibly distinct unsharpened "halo" sitting between two
   sharpened regions (user-reported).

Fixed by making extra_gap_px a single CONTINUOUS ramp instead: the second
shape's weight rises smoothly from 0 right at the primary disk's true
boundary (measured by uniform pixel distance, not a second ellipse) up to
full strength over extra_gap_px pixels -- no crossing (isotropic distance)
and no flat segment (every pixel gets some, gradually increasing, gain).

This function itself has no notion of "rings" -- it just accepts an optional
second, co-centred ellipse to union into the mask, with a smooth inner ramp.
The Saturn-specific geometry (IAU ring radius ratios, pole_pa_deg,
sub_observer_lat_deg) lives in pipeline/steps/wavelet_master.py, not here.

Run directly: python3 tests/test_sharpen_ring_aware.py
Or via pytest: pytest tests/test_sharpen_ring_aware.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.modules.wavelet import sharpen_disk_aware

H, W = 200, 200
CX, CY = 100.0, 100.0
DISK_RADIUS = 30.0
AMOUNTS = [200.0, 200.0, 200.0, 0.0, 0.0, 0.0]


def _make_image_with_outer_ring_detail() -> np.ndarray:
    """Uniform disk interior plus a thin, high-frequency sinusoidal ring
    pattern well OUTSIDE the primary disk radius (in [50, 70]px), simulating
    Saturn's rings sitting outside the fitted globe radius."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    r = np.sqrt((xx - CX) ** 2 + (yy - CY) ** 2)
    theta = np.arctan2(yy - CY, xx - CX)
    img = np.zeros((H, W), dtype=np.float64)
    img[r <= DISK_RADIUS] = 0.5
    ring_band = (r >= 50.0) & (r <= 70.0)
    img[ring_band] = 0.4 + 0.1 * np.sin(theta[ring_band] * 24.0)
    return img.astype(np.float32)


def test_extra_rx_none_is_unchanged_default_behaviour():
    """Explicitly passing extra_rx=None must be bit-identical to omitting it
    -- the new parameter is purely additive and gated, so every existing
    caller (Jupiter etc.) is provably unaffected."""
    img = _make_image_with_outer_ring_detail()
    out_default = sharpen_disk_aware(img, CX, CY, DISK_RADIUS, amounts=AMOUNTS)
    out_explicit_none = sharpen_disk_aware(
        img, CX, CY, DISK_RADIUS, amounts=AMOUNTS,
        extra_rx=None, extra_ry=None, extra_angle=None,
    )
    np.testing.assert_array_equal(out_default, out_explicit_none)


def test_outer_ring_gets_zero_gain_without_extra_ellipse():
    """Without extra_rx, detail outside the primary disk radius must receive
    no sharpening boost -- the mask is 0 there, so the result in that region
    should closely track the (unmodified) input, not an amplified version."""
    img = _make_image_with_outer_ring_detail()
    out = sharpen_disk_aware(img, CX, CY, DISK_RADIUS, amounts=AMOUNTS)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    r = np.sqrt((xx - CX) ** 2 + (yy - CY) ** 2)
    ring_band = (r >= 50.0) & (r <= 70.0)
    deviation = out[ring_band] - img[ring_band]
    assert np.abs(deviation).max() < 0.01, (
        f"expected near-zero sharpening boost outside disk radius, "
        f"got max deviation {np.abs(deviation).max():.4f}"
    )


def test_extra_ellipse_restores_sharpening_gain_over_ring_region():
    """With extra_rx/extra_ry covering the ring band, that region must
    actually receive a measurable sharpening boost -- this is the whole
    point of the fix."""
    img = _make_image_with_outer_ring_detail()
    out_without = sharpen_disk_aware(img, CX, CY, DISK_RADIUS, amounts=AMOUNTS)
    out_with = sharpen_disk_aware(
        img, CX, CY, DISK_RADIUS, amounts=AMOUNTS,
        extra_rx=75.0, extra_ry=75.0, extra_angle=0.0,
    )
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    r = np.sqrt((xx - CX) ** 2 + (yy - CY) ** 2)
    ring_band = (r >= 50.0) & (r <= 70.0)
    deviation_without = np.abs(out_without[ring_band] - img[ring_band]).mean()
    deviation_with = np.abs(out_with[ring_band] - img[ring_band]).mean()
    assert deviation_with > 5.0 * deviation_without, (
        f"expected a large sharpening boost with extra_rx set: "
        f"without={deviation_without:.5f}, with={deviation_with:.5f}"
    )


def test_region_outside_both_ellipses_stays_suppressed():
    """A pixel far outside BOTH the primary disk and the extra ellipse
    (plain background) must remain unboosted regardless of extra_rx."""
    img = _make_image_with_outer_ring_detail()
    # Inject detail far outside both ellipses (corner background) to have
    # something to measure. Kept within [0, 1] (this module's documented
    # input/output convention -- the function's own final np.clip(0, 1)
    # would otherwise clip negative background values and be mistaken for
    # a suppression effect).
    img_bg = img.copy()
    img_bg[5:15, 5:15] = 0.1 + 0.05 * np.sin(np.linspace(0, 20, 100)).reshape(10, 10)
    out = sharpen_disk_aware(
        img_bg, CX, CY, DISK_RADIUS, amounts=AMOUNTS,
        extra_rx=75.0, extra_ry=75.0, extra_angle=0.0,
    )
    deviation = np.abs(out[5:15, 5:15] - img_bg[5:15, 5:15]).max()
    assert deviation < 0.01, (
        f"expected background far outside both ellipses to stay unboosted, "
        f"got max deviation {deviation:.4f}"
    )


def test_extra_gap_px_ramps_continuously_from_disk_edge():
    """The final 2026-08-13 fix: extra_gap_px must be a single CONTINUOUS
    ramp (0 right at the disk's true boundary, rising smoothly to full
    strength over extra_gap_px pixels), not a flat exclusion zone. A flat
    zone (a rejected intermediate design) reads as a visibly distinct
    unsharpened "halo" band sandwiched between two sharpened regions on
    real data, since the real disk-to-ring gap isn't empty -- it's the
    disk's own fairly bright limb-darkening tail.

    Uses a SMOOTH radial texture envelope (no hard step at the ring's inner
    boundary) so the measured boost reflects the mask's own ramp, not a
    wavelet-decomposition edge effect from an artificial discontinuity in
    the test image itself (an earlier version of this test had exactly that
    confound). Checks boost right at the disk's edge is small relative to
    boost well past the ramp -- a flat-exclusion design would instead show
    exactly zero right up to extra_gap_px and then jump, which a coarser
    "near < far" check alone wouldn't catch, so this also checks a
    monotonic rise through an intermediate point."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    r = np.sqrt((xx - CX) ** 2 + (yy - CY) ** 2)
    theta = np.arctan2(yy - CY, xx - CX)
    # Smooth onset: texture amplitude ramps up via its own sigmoid between
    # r=30 (disk edge) and r=50, well past the 10px mask ramp being tested,
    # so the mask's ramp is the dominant effect in [31, 40] rather than the
    # texture's own onset shape.
    envelope = 1.0 / (1.0 + np.exp(-(r - 40.0) / 5.0))
    img = 0.45 + 0.05 * envelope * np.sin(theta * 24.0)

    out = sharpen_disk_aware(
        img, CX, CY, DISK_RADIUS, amounts=AMOUNTS,
        extra_rx=75.0, extra_ry=75.0, extra_angle=0.0,
        extra_gap_px=10.0,
    )
    deviation = np.abs(out - img)

    def band_deviation(r_lo, r_hi):
        band = (r >= r_lo) & (r <= r_hi)
        return deviation[band].mean()

    near_edge = band_deviation(31.0, 32.0)     # ~1-2px into the 10px ramp
    mid_ramp = band_deviation(35.0, 36.0)      # ~half-way through the ramp
    end_ramp = band_deviation(39.0, 41.0)      # ramp complete (extra_gap_px=10 -> r=40)

    assert near_edge < mid_ramp < end_ramp, (
        f"expected a monotonically increasing ramp with no flat segment: "
        f"near_edge={near_edge:.4f}, mid_ramp={mid_ramp:.4f}, end_ramp={end_ramp:.4f}"
    )
    assert end_ramp > 3.0 * near_edge, (
        f"expected substantially more boost at the end of the ramp than right "
        f"at the disk edge: near_edge={near_edge:.4f}, end_ramp={end_ramp:.4f}"
    )


def test_extra_gap_px_is_isotropic_even_with_eccentric_primary_and_extra_shapes():
    """REGRESSION GUARD for the exact bug this session hit: an EARLIER fix
    attempt carved the gap using a second ellipse at the extra shape's own
    (very flat) eccentricity -- since the primary disk here is much rounder
    than that ellipse, their boundaries crossed at an off-axis angle, and
    right at that crossing the "gap" shrank to zero width, so a point at a
    fixed small offset from the disk's true edge would get FULL boost at
    the crossing angle while getting near-zero boost at other angles (the
    ramp's starting point depended on angle, not just distance). extra_gap_px
    must ramp from a UNIFORM-width (isotropic) distance to the disk's own
    shape regardless of angle -- verified here by checking the SAME
    fixed-offset test point (1px outside the true boundary, i.e. barely
    into a 12px ramp) at several angles around a genuinely oblate primary
    disk combined with a genuinely flat extra ellipse (the exact
    eccentricity mismatch that broke the old approach), and requiring the
    measured boost to be consistent across angles -- not requiring a
    specific absolute magnitude (the wavelet decomposition of this test
    image's own edges contributes some deviation regardless of the mask;
    what must NOT happen is one angle showing dramatically more boost than
    the rest)."""
    rx, ry = 30.0, 22.0  # oblate primary disk, Saturn-globe-like ratio ~0.73
    # extra_ry chosen so the "ring" still extends at least 5px beyond the
    # disk at every tested angle (checked numerically) -- otherwise, near
    # the pole, the test point would sit outside the ring's own extent
    # entirely and show ~0 boost regardless of the gap-ramp logic being
    # tested, which would look like a false pass.
    extra_rx, extra_ry = 90.0, 25.0
    gap_px = 12.0

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    dx, dy = xx - CX, yy - CY
    theta_grid = np.arctan2(dy, dx)

    deviations = {}
    for theta_deg in (0.0, 15.0, 30.0, 45.0, 60.0):
        theta = np.radians(theta_deg)
        disk_r_here = 1.0 / np.sqrt((np.cos(theta) / rx) ** 2 + (np.sin(theta) / ry) ** 2)
        test_r = disk_r_here + 1.0  # just outside the true edge, near-start of the ramp

        img = np.full((H, W), 0.5, dtype=np.float64)
        band = (np.abs(np.sqrt(dx ** 2 + dy ** 2) - test_r) < 1.0) & (np.abs(theta_grid - theta) < 0.05)
        img[band] = 0.1 + 0.05 * np.sin(np.linspace(0, 30, int(band.sum())))

        out = sharpen_disk_aware(
            img, CX, CY, rx, amounts=AMOUNTS, ry=ry, angle=0.0,
            extra_rx=extra_rx, extra_ry=extra_ry, extra_angle=0.0,
            extra_gap_px=gap_px,
        )
        deviations[theta_deg] = np.abs(out[band] - img[band]).mean()

    lo, hi = min(deviations.values()), max(deviations.values())
    assert hi < 3.0 * lo + 0.02, (
        f"expected consistent (isotropic) boost across angles at this fixed "
        f"offset from the disk edge, got a wide spread: {deviations} "
        f"(min={lo:.4f}, max={hi:.4f})"
    )


def test_degenerate_near_zero_extra_ry_stays_finite():
    """extra_ry can be driven arbitrarily small by an edge-on ring tilt
    (sub_observer_lat_deg -> 0); the caller is expected to clamp it (as
    wavelet_master.py does, max(..., 1e-6)) but this function must not
    produce NaN/Inf even at that extreme."""
    img = _make_image_with_outer_ring_detail()
    out = sharpen_disk_aware(
        img, CX, CY, DISK_RADIUS, amounts=AMOUNTS,
        extra_rx=75.0, extra_ry=1e-6, extra_angle=0.3,
    )
    assert np.isfinite(out).all(), "expected a finite result even with a near-degenerate extra ellipse"


if __name__ == "__main__":
    test_extra_rx_none_is_unchanged_default_behaviour()
    print("extra_rx=None matches default: OK")
    test_outer_ring_gets_zero_gain_without_extra_ellipse()
    print("outer ring gets zero gain without extra ellipse: OK")
    test_extra_ellipse_restores_sharpening_gain_over_ring_region()
    print("extra ellipse restores sharpening gain over ring region: OK")
    test_region_outside_both_ellipses_stays_suppressed()
    print("region outside both ellipses stays suppressed: OK")
    test_extra_gap_px_ramps_continuously_from_disk_edge()
    print("extra_gap_px ramps continuously from disk edge: OK")
    test_extra_gap_px_is_isotropic_even_with_eccentric_primary_and_extra_shapes()
    print("extra_gap_px is isotropic even with eccentric primary/extra shapes: OK")
    test_degenerate_near_zero_extra_ry_stays_finite()
    print("degenerate near-zero extra_ry stays finite: OK")
    print("\nAll checks passed.")
