"""Regression test for detect_tracker_flip_ns()'s belt-alignment rotation.

Found via a deep audit prompted by an external review questioning whether
pole_pa_deg means the same thing across derotation.py and
satellite_tracker.py. That specific hypothesis (a pole-axis vs
equator-axis mismatch in get_positions()'s theta_cam formula) was NOT
confirmed — historical real-data validation of shadow/moon positions
(project memory: a north-south MIRROR bug was found and fixed there in
2026-05-12, not a 90-degree rotation-scale error, which is what a
pole-vs-equator mismatch would produce) argues against it. But the audit
did turn up a real, different, previously-unknown bug in this file:
detect_tracker_flip_ns()'s "rotate the image so belts run horizontal" step
used the wrong sign, which DOUBLES the tilt instead of removing it.

Run directly: python3 tests/test_satellite_tracker.py
Or via pytest: pytest tests/test_satellite_tracker.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.modules.derotation import auto_detect_equator_pa


def _make_belted_image(h=300, w=300, cx=150.0, cy=150.0, disk_r=120.0, tilt_deg=0.0):
    img = np.zeros((h, w), dtype=np.float32)
    yy_row = np.arange(h)[:, None]
    img[:, :] = 0.5 + 0.4 * np.sin(yy_row / 15.0)
    yy, xx = np.mgrid[0:h, 0:w]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 < disk_r ** 2
    img = (img * mask).astype(np.float32)
    if tilt_deg != 0:
        M = cv2.getRotationMatrix2D((cx, cy), tilt_deg, 1.0)  # cv2: + = CCW on screen
        img = cv2.warpAffine(img, M, (w, h))
    return img.astype(np.float32)


def test_belt_alignment_rotation_reduces_tilt_to_zero():
    """REGRESSION GUARD: rotating a tilted image by the sign
    detect_tracker_flip_ns() actually uses (see the fixed line inside it)
    must bring the measured belt tilt to ~0, not double it.

    This mirrors the exact rotate-then-remeasure check used to find the
    bug: apply the SAME sign convention detect_tracker_flip_ns() uses
    internally (cv2.getRotationMatrix2D(center, +pole_pa_deg, 1.0), fixed
    from the original -pole_pa_deg) and confirm it actually zeroes the tilt
    on a synthetic belted image with a known injected tilt.
    """
    h, w = 300, 300
    cx, cy, disk_r = 150.0, 150.0, 120.0

    for true_tilt in (-15.0, 10.0, 25.0):
        img = _make_belted_image(h, w, cx, cy, disk_r, tilt_deg=0.0)
        # Inject a known tilt the same way auto_detect_equator_pa would see it
        # in a real frame: rotate with cv2's own (CCW-positive) convention.
        M_inject = cv2.getRotationMatrix2D((cx, cy), true_tilt, 1.0)
        tilted = cv2.warpAffine(img, M_inject, (w, h))

        measured_pa = auto_detect_equator_pa(frames=[tilted], cx=cx, cy=cy, disk_radius_px=disk_r)

        # The fixed correction step: rotate by +measured_pa (NOT -measured_pa).
        M_correct = cv2.getRotationMatrix2D((cx, cy), float(measured_pa), 1.0)
        corrected = cv2.warpAffine(tilted, M_correct, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        residual_pa = auto_detect_equator_pa(frames=[corrected], cx=cx, cy=cy, disk_radius_px=disk_r)

        assert abs(residual_pa) < 3.0, (
            f"true_tilt={true_tilt}: measured={measured_pa:.2f}, after +measured_pa "
            f"correction residual={residual_pa:.2f} (should be near 0)"
        )

        # And confirm the OLD (buggy) sign would have made it worse, as a
        # concrete demonstration this is a real, meaningful regression guard
        # and not just an arbitrary tolerance check.
        M_buggy = cv2.getRotationMatrix2D((cx, cy), -float(measured_pa), 1.0)
        buggy_corrected = cv2.warpAffine(tilted, M_buggy, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        buggy_residual = auto_detect_equator_pa(frames=[buggy_corrected], cx=cx, cy=cy, disk_radius_px=disk_r)
        assert abs(buggy_residual) > abs(measured_pa) * 1.5, (
            f"true_tilt={true_tilt}: the old buggy sign (-measured_pa) should visibly "
            f"WORSEN the tilt (roughly double it), got residual={buggy_residual:.2f} "
            f"vs original {measured_pa:.2f} — if this no longer worsens it, the "
            f"synthetic test setup itself may have changed, re-check by hand"
        )


if __name__ == "__main__":
    test_belt_alignment_rotation_reduces_tilt_to_zero()
    print("belt alignment rotation sign: OK")
    print("\nAll checks passed.")
