"""Regression tests for composite.align_channels()'s quality-gated shift
selection.

Background: align_channels() used to simply trust cv2.phaseCorrelate()
(optionally restricted to a disk-only ROI, see git history of commit
e958140 and its revert). Real Jupiter data (28 windows, one session) showed
phase correlation confidently reporting a WRONG shift for the B channel
(weak correlated signal vs. the IR reference) in ~25% of windows -- e.g.
dy=+2.16 reported when every neighbouring window measured dy in -0.4..-0.9
-- producing a visible limb colour fringe that the old max_shift_px
magnitude-only gate does not catch (the bad shift is well under the gate).

align_channels() now considers (0,0)/whole-frame/disk-ROI as candidates and
picks whichever gives the best post-hoc disk-region NCC. This does NOT
recover the true offset when correlation itself has no way to find it (that
residual is a real, currently-unresolved limitation -- see
project_saturn_composite_alignment_bug memory) -- it only prevents applying
a confidently-wrong shift that makes things worse than doing nothing.

Run directly: python3 tests/test_align_channels.py
Or via pytest: pytest tests/test_align_channels.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.modules.composite import align_channels, apply_shift


def _make_disk(h=200, w=200, cx=100.0, cy=100.0, r=80.0, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.hypot(xx - cx, yy - cy)
    mask = dist < r
    belts = 0.5 + 0.3 * np.sin(yy / 9.0) + 0.05 * rng.standard_normal((h, w))
    img = np.where(mask, belts, 0.0).astype(np.float32)
    return np.clip(img, 0.0, 1.0)


def test_good_shift_is_recovered():
    """A channel with a real, correlated small shift must still be corrected
    (the quality gate must not block genuinely-good phase correlation)."""
    ref = _make_disk(seed=1)
    true_dx, true_dy = -1.3, 0.7
    shifted = apply_shift(ref, -true_dx, -true_dy)  # shifted + true shift back == ref

    aligned, shifts = align_channels({"ref": ref, "ch": shifted}, "ref", max_shift_px=10.0)
    dx, dy = shifts["ch"]
    assert abs(dx) > 0.3 or abs(dy) > 0.3, f"expected a real correction, got {shifts['ch']}"
    # Aligned channel should now closely match ref near the disk centre.
    diff = np.abs(aligned["ch"][60:140, 60:140] - ref[60:140, 60:140]).mean()
    diff_unaligned = np.abs(shifted[60:140, 60:140] - ref[60:140, 60:140]).mean()
    assert diff < diff_unaligned, "aligned channel should match ref better than the raw shifted one"


def test_uncorrelated_channel_falls_back_to_no_shift():
    """REGRESSION GUARD: when a channel has no real correlated structure
    with the reference (simulating the real B-channel failure mode), the
    quality gate must prefer (0,0) over whatever plausible-looking shift
    phase correlation reports, rather than blindly trusting it."""
    ref = _make_disk(seed=2)
    # A channel with the SAME disk footprint (so find_disk_center still
    # works) but uncorrelated internal structure -- phase correlation has
    # no true peak to find here, only noise.
    bad = _make_disk(seed=99)

    aligned, shifts = align_channels({"ref": ref, "ch": bad}, "ref", max_shift_px=15.0)
    dx, dy = shifts["ch"]

    # Whatever shift (if any) was tried, it must not have been picked
    # unless it genuinely beat (0,0) on quality -- assert the final choice
    # scores at least as well against ref as leaving it unshifted would.
    from pipeline.modules.composite import _disk_region_quality
    from pipeline.modules.derotation import find_disk_center

    cx, cy, sr, _, _ = find_disk_center(ref)
    q_none = _disk_region_quality(ref, bad, cx, cy, sr)
    q_chosen = _disk_region_quality(
        ref, bad if (dx, dy) == (0.0, 0.0) else aligned["ch"], cx, cy, sr
    )
    assert q_chosen >= q_none - 1e-6, (
        f"align_channels picked shift {shifts['ch']} with quality {q_chosen:.3f}, "
        f"worse than doing nothing ({q_none:.3f}) -- the quality gate should never "
        f"choose a candidate worse than (0,0)"
    )


def test_reference_channel_always_zero_shift():
    ref = _make_disk(seed=3)
    other = _make_disk(seed=4)
    _, shifts = align_channels({"ref": ref, "other": other}, "ref", max_shift_px=10.0)
    assert shifts["ref"] == (0.0, 0.0)


if __name__ == "__main__":
    test_good_shift_is_recovered()
    print("good shift is recovered: OK")
    test_uncorrelated_channel_falls_back_to_no_shift()
    print("uncorrelated channel never worse than no-shift: OK")
    test_reference_channel_always_zero_shift()
    print("reference channel always zero shift: OK")
    print("\nAll checks passed.")
