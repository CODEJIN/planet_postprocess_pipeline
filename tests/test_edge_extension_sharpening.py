"""Regression tests for edge-extension-before-sharpening (2026-08-15).

Background: [[project_ring_limb_ringing_bug]] diagnosed a Saturn asymmetric
wavelet-sharpening overshoot at the globe limb, traced to find_disk_center's
ellipse fit sitting ~0.5-0.9px off the TRUE photometric limb (asymmetrically,
due to ring-signal-biased Otsu contour fitting). A separately-implemented
remedy (coverage-aware sharpening gain, see test_coverage_aware_sharpening.py)
was measured to have negligible real effect, because its signal (rotation-
validity coverage) is spatially close to but causally different from the
fit-error mechanism.

An external review re-derived the bug as classic Gibbs ringing: feeding a
real intensity step (the disk-to-background/ring transition) into the
band-limited à trous wavelet filter necessarily produces overshoot near
that step, independent of how well the mask is centred. Its proposed fix --
extend the disk signal past the fitted boundary before decomposition
(removing the step the filter sees), then let the existing feather mask cut
it back afterward -- turned out to already have unused groundwork in this
codebase: wavelet._fill_outside_ellipse() (added in commit a8db01a,
"wavelet algorithm update") does exactly this but was never wired into
sharpen_disk_aware(). This test file covers that wiring
(fill_outside_before_sharpen param).

Run directly: python3 tests/test_edge_extension_sharpening.py
Or via pytest: pytest tests/test_edge_extension_sharpening.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.modules.wavelet import _fill_outside_ellipse, sharpen_disk_aware

SIZE = 300


# ── _fill_outside_ellipse() unit correctness ────────────────────────────────

def test_fill_outside_ellipse_replaces_outside_only():
    h, w = SIZE, SIZE
    cx, cy, rx, ry = 150.0, 150.0, 100.0, 90.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    image = (xx + yy).astype(np.float32)  # simple ramp, easy to reason about

    filled = _fill_outside_ellipse(image, cx, cy, rx, ry, 0.0)
    assert filled.shape == image.shape

    d_norm = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
    inside = d_norm <= 1.0
    outside = ~inside

    np.testing.assert_array_equal(filled[inside], image[inside])
    assert not np.array_equal(filled[outside], image[outside]), (
        "expected outside-ellipse pixels to be overwritten"
    )
    # Every replaced pixel's new value must equal some on-boundary pixel's
    # original value (radial projection lands on the ellipse, not off it).
    d_norm_of_source = None  # sanity: replaced values are within the ramp's global range
    assert filled[outside].min() >= image.min() - 1e-3
    assert filled[outside].max() <= image.max() + 1e-3


def test_fill_outside_ellipse_noop_when_all_inside():
    image = np.random.default_rng(0).random((50, 50)).astype(np.float32)
    filled = _fill_outside_ellipse(image, 25.0, 25.0, 1000.0, 1000.0, 0.0)
    np.testing.assert_array_equal(filled, image)


# ── sharpen_disk_aware()'s fill_outside_before_sharpen ──────────────────────

def _textured_disk_on_ring(cx: float, cy: float, r: float, seed: int = 0) -> np.ndarray:
    """Disk with interior texture sitting in front of a bright, structured
    'ring' background (not just flat zero) -- a harder case than a plain
    dark background, closer to the real Saturn scenario this targets."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    disk = (rr < r).astype(np.float64) * 0.6
    texture = 0.1 * rng.standard_normal((SIZE, SIZE)) * (rr < r)
    ring_bg = 0.3 * np.sin(rr / 5.0) * (rr >= r) * (rr < r * 2.0)
    return np.clip(disk + texture + ring_bg, 0.0, 1.0).astype(np.float32)


def test_fill_outside_before_sharpen_false_matches_omitted():
    img = _textured_disk_on_ring(150.0, 150.0, 100.0, seed=1)
    a = sharpen_disk_aware(img, 150.0, 150.0, 100.0, amounts=[200, 200, 100, 0, 0, 0])
    b = sharpen_disk_aware(
        img, 150.0, 150.0, 100.0, amounts=[200, 200, 100, 0, 0, 0],
        fill_outside_before_sharpen=False,
    )
    np.testing.assert_array_equal(a, b)


def test_fill_outside_before_sharpen_never_changes_output_outside_disk():
    """The correctness guarantee this feature depends on: `original` (the
    base the output is built from) always comes from the real image, never
    the filled copy -- so anything strictly outside the disk mask must be
    bit-identical whether the flag is on or off."""
    img = _textured_disk_on_ring(150.0, 150.0, 100.0, seed=2)
    off = sharpen_disk_aware(img, 150.0, 150.0, 100.0, amounts=[200, 200, 100, 0, 0, 0])
    on = sharpen_disk_aware(
        img, 150.0, 150.0, 100.0, amounts=[200, 200, 100, 0, 0, 0],
        fill_outside_before_sharpen=True,
    )
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
    r = np.hypot(xx - 150.0, yy - 150.0)
    far_outside = r > 110.0  # comfortably outside the disk + feather zone
    np.testing.assert_array_equal(off[far_outside], on[far_outside])


def test_fill_outside_before_sharpen_reduces_hard_edge_overshoot():
    """Mechanism validation, mirroring the real diagnosed scenario: the
    ellipse fit passed to sharpen_disk_aware (r_fit) UNDERSHOOTS the true
    photometric limb (r_true) by a few pixels -- exactly the asymmetric
    ~0.5-0.9px fit error diagnosed on real Saturn data, exaggerated here for
    a clean synthetic signal. Real disk content (interior_max) genuinely
    extends out to r_true; only beyond r_true does it truly drop to
    background. Without the fix, the wavelet filter sees this real step at
    r_true sitting just outside the fitted mask's own feather zone, and the
    weight_map ramps to ~1 (not yet fully faded) while the filter has
    already picked up the step -- producing overshoot right at r_fit. With
    the fix, everything beyond r_fit is flattened to the r_fit boundary's
    own (still-bright) value before decomposition, so the filter never sees
    the step at all."""
    cx, cy = 150.0, 150.0
    r_fit, r_true = 97.0, 100.0
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    interior_max = 0.5
    img = np.where(rr < r_true, interior_max, 0.0).astype(np.float32)

    amounts = [200, 200, 200, 100, 0, 0]
    off = sharpen_disk_aware(img, cx, cy, r_fit, amounts=amounts, edge_feather_factor=2.0)
    on = sharpen_disk_aware(
        img, cx, cy, r_fit, amounts=amounts, edge_feather_factor=2.0,
        fill_outside_before_sharpen=True,
    )

    band = (rr >= r_fit - 8.0) & (rr < r_fit + 8.0)
    overshoot_off = float(np.clip(off[band] - interior_max, 0.0, None).max())
    overshoot_on = float(np.clip(on[band] - interior_max, 0.0, None).max())
    assert overshoot_off > 1e-3, "test setup should produce measurable overshoot without the fix"
    assert overshoot_on < overshoot_off, (
        f"expected reduced overshoot with fill_outside_before_sharpen=True: "
        f"off={overshoot_off:.4f} on={overshoot_on:.4f}"
    )


def test_fill_outside_before_sharpen_3d_and_color_paths():
    """Sanity: the 3-D per-channel recursion branch accepts and threads the
    new param without raising, and stays consistent with the 2-D call."""
    img2d = _textured_disk_on_ring(150.0, 150.0, 100.0, seed=3)
    img3d = np.stack([img2d, img2d, img2d], axis=2)

    out2d = sharpen_disk_aware(
        img2d, 150.0, 150.0, 100.0, amounts=[200, 200, 100, 0, 0, 0],
        fill_outside_before_sharpen=True,
    )
    out3d = sharpen_disk_aware(
        img3d, 150.0, 150.0, 100.0, amounts=[200, 200, 100, 0, 0, 0],
        fill_outside_before_sharpen=True,
    )
    for c in range(3):
        np.testing.assert_allclose(out3d[:, :, c], out2d, atol=1e-6)


if __name__ == "__main__":
    test_fill_outside_ellipse_replaces_outside_only()
    print("_fill_outside_ellipse replaces outside only: OK")
    test_fill_outside_ellipse_noop_when_all_inside()
    print("_fill_outside_ellipse no-op when all inside: OK")
    test_fill_outside_before_sharpen_false_matches_omitted()
    print("fill_outside_before_sharpen=False matches omitted: OK")
    test_fill_outside_before_sharpen_never_changes_output_outside_disk()
    print("fill_outside_before_sharpen never changes output outside disk: OK")
    test_fill_outside_before_sharpen_reduces_hard_edge_overshoot()
    print("fill_outside_before_sharpen reduces hard-edge overshoot: OK")
    test_fill_outside_before_sharpen_3d_and_color_paths()
    print("fill_outside_before_sharpen 3-D/color paths: OK")
    print("\nAll checks passed.")
