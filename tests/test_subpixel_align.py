"""Regression test for the subpixel_align() sign convention.

cv2.phaseCorrelate(reference, target) returns the *forward* content shift of
target relative to reference — i.e. if target's content sits at
reference's content moved by (true_dx, true_dy), phaseCorrelate returns
approximately (true_dx, true_dy), not its negation.

Every caller in this codebase uses the pattern:

    dx, dy = subpixel_align(reference, target)
    aligned = apply_shift(target, dx, dy)

apply_shift()/cv2.warpAffine with M=[[1,0,dx],[0,1,dy]] moves image content
by (+dx, +dy). To move target's content back onto reference (undo a forward
shift of (true_dx, true_dy)), apply_shift must be called with
(-true_dx, -true_dy) — the *negation* of what phaseCorrelate reports.

This test builds a synthetic image, shifts it by a known sub-pixel amount
using scipy.ndimage.shift (an interpolation implementation independent of
cv2.warpAffine, so this isn't circular), and verifies end-to-end that:

  1. subpixel_align() + apply_shift() actually recovers the reference
     (small residual MSE, not just "any" MSE).
  2. Using subpixel_align()'s output is a large improvement over doing
     nothing.
  3. Using the *un-negated* raw cv2.phaseCorrelate output instead would
     have made things WORSE than doing nothing at all — this is the
     concrete evidence that the sign bug this test guards against is a
     real regression, not just a style/precision nitpick.

Run directly: python3 tests/test_subpixel_align.py
Or via pytest: pytest tests/test_subpixel_align.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import shift as scipy_shift

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.modules.derotation import apply_shift, subpixel_align

# (dy, dx) test vectors — deliberately asymmetric and covering both signs on
# both axes, plus a near-zero case. scipy.ndimage.shift takes (row, col) =
# (dy, dx) order.
SHIFT_VECTORS_DY_DX = [
    (3.7, -2.3),
    (-5.1, 4.6),
    (0.4, 0.6),
    (-6.8, -6.2),
    (0.0, 0.0),
]


def _make_reference_image(size: int = 160, seed: int = 0) -> np.ndarray:
    """Synthetic float32 [0,1] image with rich, asymmetric texture.

    A pure Gaussian blob is radially symmetric and would make phase
    correlation direction-ambiguous for a naive sign-flip bug (wrong-sign
    error on a symmetric target can look deceptively small). Combining a
    blob with an off-center bright patch and mild structured noise gives
    phase correlation real, direction-sensitive structure to lock onto.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)

    cy, cx = size * 0.5, size * 0.5
    blob = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (size * 0.18) ** 2)))

    # Off-center asymmetric feature so up/down and left/right are distinguishable.
    patch_cy, patch_cx = size * 0.3, size * 0.65
    patch = 0.6 * np.exp(-(((xx - patch_cx) ** 2 + (yy - patch_cy) ** 2) / (2 * (size * 0.06) ** 2)))

    texture = rng.normal(0.0, 0.03, size=(size, size)).astype(np.float32)
    texture = cv2.GaussianBlur(texture, (5, 5), 1.0)

    img = blob + patch + texture
    img = np.clip(img, 0.0, None)
    img /= img.max()
    return img.astype(np.float32)


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def _check_one_shift(reference: np.ndarray, true_dy: float, true_dx: float) -> dict:
    """Run the full round-trip for one (true_dy, true_dx) and return diagnostics."""
    # Independent ground-truth shift: scipy spline interpolation, NOT cv2.warpAffine.
    target = scipy_shift(
        reference, shift=(true_dy, true_dx), order=3, mode="nearest"
    ).astype(np.float32)

    # Raw (un-negated) phase correlation output, for the "what if the sign
    # bug were still present" comparison.
    ref_f32 = reference.astype(np.float32)
    tgt_f32 = target.astype(np.float32)
    (raw_dx, raw_dy), _ = cv2.phaseCorrelate(ref_f32, tgt_f32)

    # Current (fixed) code path.
    dx, dy = subpixel_align(reference, target)
    aligned_fixed = apply_shift(target, dx, dy)

    # What the old, buggy (un-negated) code path would have produced.
    aligned_buggy = apply_shift(target, raw_dx, raw_dy)

    mse_unaligned = _mse(reference, target)
    mse_fixed = _mse(reference, aligned_fixed)
    mse_buggy = _mse(reference, aligned_buggy)

    return {
        "true_dy": true_dy,
        "true_dx": true_dx,
        "raw_phasecorrelate_dxdy": (raw_dx, raw_dy),
        "subpixel_align_dxdy": (dx, dy),
        "mse_unaligned": mse_unaligned,
        "mse_fixed": mse_fixed,
        "mse_buggy": mse_buggy,
    }


def test_subpixel_align_recovers_reference():
    """Fixed sign convention must bring target close to reference."""
    reference = _make_reference_image()
    for true_dy, true_dx in SHIFT_VECTORS_DY_DX:
        result = _check_one_shift(reference, true_dy, true_dx)
        # Small residual expected (interpolation blur from two independent
        # resampling steps: scipy shift then cv2 warpAffine), not exact zero.
        assert result["mse_fixed"] < 5e-4, (
            f"shift=({true_dy},{true_dx}): fixed-path MSE too high: {result}"
        )


def test_subpixel_align_beats_no_alignment():
    """Fixed sign convention must be a large improvement over doing nothing,
    for every non-trivial shift."""
    reference = _make_reference_image()
    for true_dy, true_dx in SHIFT_VECTORS_DY_DX:
        if true_dy == 0.0 and true_dx == 0.0:
            continue
        result = _check_one_shift(reference, true_dy, true_dx)
        assert result["mse_fixed"] < result["mse_unaligned"] * 0.1, (
            f"shift=({true_dy},{true_dx}): fixed path did not clearly beat "
            f"no alignment: {result}"
        )


def test_unfixed_sign_would_be_worse_than_no_alignment():
    """The specific regression this test guards against: applying the raw,
    un-negated cv2.phaseCorrelate output (the pre-fix behaviour) must be
    WORSE than doing no alignment at all, for every non-trivial shift."""
    reference = _make_reference_image()
    for true_dy, true_dx in SHIFT_VECTORS_DY_DX:
        if true_dy == 0.0 and true_dx == 0.0:
            continue
        result = _check_one_shift(reference, true_dy, true_dx)
        assert result["mse_buggy"] > result["mse_unaligned"], (
            f"shift=({true_dy},{true_dx}): un-negated path was NOT worse "
            f"than no alignment — sign convention assumption may no longer "
            f"hold for this cv2 version: {result}"
        )
        assert result["mse_fixed"] < result["mse_buggy"], (
            f"shift=({true_dy},{true_dx}): fixed path did not beat the "
            f"un-negated (buggy) path: {result}"
        )


if __name__ == "__main__":
    reference = _make_reference_image()
    print(f"{'true_dy':>8} {'true_dx':>8} {'raw_dx':>9} {'raw_dy':>9} "
          f"{'mse_unaligned':>14} {'mse_fixed':>12} {'mse_buggy':>12}")
    for true_dy, true_dx in SHIFT_VECTORS_DY_DX:
        r = _check_one_shift(reference, true_dy, true_dx)
        raw_dx, raw_dy = r["raw_phasecorrelate_dxdy"]
        print(f"{true_dy:8.2f} {true_dx:8.2f} {raw_dx:9.3f} {raw_dy:9.3f} "
              f"{r['mse_unaligned']:14.6f} {r['mse_fixed']:12.6f} {r['mse_buggy']:12.6f}")

    test_subpixel_align_recovers_reference()
    test_subpixel_align_beats_no_alignment()
    test_unfixed_sign_would_be_worse_than_no_alignment()
    print("\nAll checks passed.")
