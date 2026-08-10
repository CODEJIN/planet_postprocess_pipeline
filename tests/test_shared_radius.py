"""Regression tests for resolve_shared_radius() (derotation.py).

Background: real Jupiter data showed each filter's independently-detected
semi_major_px (used as spherical_derotation_warp()'s disk_radius_px)
differs from its siblings by 1-3px within the same window — pure
Otsu-threshold noise, not a real size difference. The warp treats anything
beyond disk_radius_px*1.05 as invalid and forces it to background, so this
noise makes each filter's de-rotated stack go to zero at a slightly
different radius; composited, that reads as a limb colour fringe, already
present at the per-filter de-rotation stage (before composite or wavelet
run) — confirmed via a real 2026-08-11 investigation (see
project_saturn_composite_alignment_bug memory).

resolve_shared_radius() computes a window-wide consensus (median); each
filter only snaps to it if its own value already agrees within
_RADIUS_SHARE_REL_TOL, so a filter with a genuinely different apparent
size (e.g. an absorption band) is left alone.

Run directly: python3 tests/test_shared_radius.py
Or via pytest: pytest tests/test_shared_radius.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.modules.derotation import _RADIUS_SHARE_REL_TOL, resolve_shared_radius


def _fit(semi_a: float, confidence: float = 1.0) -> tuple:
    # (cx, cy, semi_a, semi_b, angle, confidence, shape_reliable)
    return (140.0, 140.0, semi_a, semi_a * 0.93, 175.0, confidence, True)


def test_returns_median_of_close_fits():
    fits = {"R": _fit(104.40), "G": _fit(104.86), "B": _fit(103.89), "IR": _fit(103.72)}
    result = resolve_shared_radius(fits)
    assert result is not None
    values = sorted(f[2] for f in fits.values())
    assert result == values[len(values) // 2] or abs(result - values[1]) < 1e-9 or abs(result - sum(values[1:3]) / 2) < 1e-9


def test_none_with_fewer_than_two_confident_fits():
    assert resolve_shared_radius({"IR": _fit(103.72)}) is None
    assert resolve_shared_radius({}) is None
    assert resolve_shared_radius({"IR": _fit(103.72, confidence=0.0), "R": _fit(104.0)}) is None


def test_zero_confidence_fits_excluded_from_median():
    fits = {
        "IR": _fit(103.72),
        "R": _fit(104.40),
        "CH4": _fit(65.0, confidence=0.0),  # failed detection, must not skew the median
    }
    result = resolve_shared_radius(fits)
    assert result is not None
    assert 103.0 < result < 105.0, f"a zero-confidence outlier leaked into the median: {result}"


def test_outlier_filter_not_forced_by_caller_threshold():
    """Not resolve_shared_radius's own job (it just returns the median) --
    but confirm the documented threshold constant is sane and that a
    genuinely different filter (e.g. CH4-like, ~35% smaller) would fail the
    caller-side acceptance check in derotate_filter()."""
    fits = {"R": _fit(104.40), "G": _fit(104.86), "B": _fit(103.89), "IR": _fit(103.72)}
    shared = resolve_shared_radius(fits)
    ch4_semi_a = 68.0  # a genuinely different apparent size
    assert abs(ch4_semi_a - shared) > _RADIUS_SHARE_REL_TOL * shared, (
        "a genuinely different filter size should fail the acceptance gate"
    )
    close_semi_a = 104.86  # within the group
    assert abs(close_semi_a - shared) <= _RADIUS_SHARE_REL_TOL * shared


if __name__ == "__main__":
    test_returns_median_of_close_fits()
    print("median of close fits: OK")
    test_none_with_fewer_than_two_confident_fits()
    print("None with <2 confident fits: OK")
    test_zero_confidence_fits_excluded_from_median()
    print("zero-confidence fits excluded: OK")
    test_outlier_filter_not_forced_by_caller_threshold()
    print("outlier filter correctly fails acceptance gate: OK")
    print("\nAll checks passed.")
