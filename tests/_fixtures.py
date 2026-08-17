"""Shared synthetic-image fixtures for the wavelet/sharpening test suite.

Extracted 2026-08-17 (pure cleanup, no behavior change) from 3 near-identical
copy-pasted `_textured_disk` helpers that had drifted only in cosmetic ways
(some took an explicit `size` argument, some used a module-level `SIZE`/`H,W`
constant; one applied a Gaussian blur afterward, one used an inclusive `<=`
radius test instead of the others' strict `<`). All three behaviors are
preserved here as explicit parameters -- no new default was invented, each
call site below passes exactly what it used to compute inline.
"""
from __future__ import annotations

import cv2
import numpy as np


def textured_disk(
    size: int,
    cx: float,
    cy: float,
    r: float,
    amp: float = 0.15,
    seed: int = 0,
    inclusive: bool = False,
    blur_sigma: float = 0.0,
) -> np.ndarray:
    """Uniform disk (brightness 0.6) plus random speckle texture, clipped to
    [0, 1] -- real, legitimate graded detail (not a hard edge) used across
    several sharpening tests to confirm gain/clamp/confidence behavior
    doesn't defeat genuine detail enhancement.

    inclusive: False (default) uses `rr < r` (test_ring_occlusion_weight.py/
    test_coverage_aware_sharpening.py's original convention); True uses
    `rr <= r` (test_overshoot_clamp.py's original convention).
    blur_sigma: 0.0 (default) returns the raw disk+texture unblurred; > 0.0
    applies cv2.GaussianBlur(img, (5, 5), blur_sigma) afterward
    (test_overshoot_clamp.py's original convention).
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    on_disk = (rr <= r) if inclusive else (rr < r)
    disk = on_disk.astype(np.float64) * 0.6
    texture = amp * rng.standard_normal((size, size)) * on_disk
    img = np.clip(disk + texture, 0.0, 1.0).astype(np.float32)
    if blur_sigma > 0.0:
        img = cv2.GaussianBlur(img, (5, 5), blur_sigma)
    return img
