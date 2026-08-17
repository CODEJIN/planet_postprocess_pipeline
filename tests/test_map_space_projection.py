"""Regression tests for the map-space (lat/lon) projection primitives
(_disk_to_map, _map_to_disk) — Phase A of project_map_space_derotation_roadmap.

These are purely additive, not called from any existing production path yet
(Phase A is the core projection primitive only; wiring into the actual
de-rotation pipeline is a later phase). They reuse _oblate_ortho_forward/
_oblate_ortho_inverse exactly as-is (see tests/test_reprojection.py for
those functions' own round-trip/sign/stability tests) -- these tests cover
only the NEW layer: rendering a full disk<->map image via cv2.remap, and the
map-pixel-coordinate convention the two functions must agree on.

Run directly: python3 tests/test_map_space_projection.py
Or via pytest: pytest tests/test_map_space_projection.py -v
"""
from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.modules import image_io
from pipeline.modules.derotation import _disk_to_map, _map_to_disk, map_space_window_stack

SIZE = 300
CX, CY = 150.0, 150.0
DISK_SEMI_A = 100.0
REQ_PX = DISK_SEMI_A * 1.05   # same 5% padding convention as _reprojected_position
RPOL_PX = REQ_PX * 0.90       # arbitrary synthetic oblateness, not any real planet


def _synthetic_globe_texture(size=SIZE, cx=CX, cy=CY, radius=DISK_SEMI_A, seed=0):
    """Band-limited noise texture filling the globe silhouette -- avoids the
    periodic-pattern phase-correlation pitfall documented elsewhere in this
    project (irrelevant here, no correlation involved, but noise texture is
    also simply a good generic "has real structure everywhere" choice for a
    pixel-value round-trip comparison)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    field = rng.standard_normal((size, size)).astype(np.float32)
    field = cv2.GaussianBlur(field, (0, 0), sigmaX=3.0)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    disk = (0.5 + 0.3 * field) * (rr < radius)
    return np.clip(disk, 0.0, 1.0).astype(np.float32)


def test_disk_to_map_valid_mask_is_roughly_half_the_globe():
    """At a typical (non-degenerate) B, roughly half the globe's surface
    should be on the near (visible) side -- a sanity check that depth>0
    validity isn't accidentally all-true or all-false."""
    img = _synthetic_globe_texture()
    _map_img, valid_mask = _disk_to_map(
        img, CX, CY, REQ_PX, RPOL_PX, pole_pa_deg=15.0, sub_observer_lat_deg=20.0,
    )
    assert 0.3 < valid_mask.mean() < 0.7


def test_disk_to_map_to_disk_round_trip_no_rotation():
    """disk -> map -> disk at the SAME orientation (no longitude shift)
    must reproduce the original image closely, well inside the limb (a
    comfortable margin avoids limb/pole numerical edge effects that are a
    known, documented Phase A limitation, not what this test is checking)."""
    img = _synthetic_globe_texture()
    map_img, valid_mask = _disk_to_map(
        img, CX, CY, REQ_PX, RPOL_PX, pole_pa_deg=15.0, sub_observer_lat_deg=20.0,
    )
    recovered = _map_to_disk(
        map_img, valid_mask, (SIZE, SIZE), CX, CY, REQ_PX, RPOL_PX,
        pole_pa_deg=15.0, sub_observer_lat_deg=20.0,
    )

    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
    rr = np.sqrt((xx - CX) ** 2 + (yy - CY) ** 2)
    interior = rr < DISK_SEMI_A * 0.85
    diff = np.abs(recovered[interior] - img[interior])
    assert diff.mean() < 0.02, f"mean abs diff too high: {diff.mean():.4f}"
    assert np.percentile(diff, 95) < 0.05, f"95th percentile diff too high: {np.percentile(diff, 95):.4f}"


def test_round_trip_holds_across_b_and_pole_pa_sweep():
    """Same round-trip check across a spread of B/pole_pa combinations,
    including near edge-on (small B) and near pole-on (large B) -- the
    values _oblate_ortho_inverse's own branch selection (B~0 direct solve
    vs quadratic) is most sensitive to."""
    img = _synthetic_globe_texture()
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
    rr = np.sqrt((xx - CX) ** 2 + (yy - CY) ** 2)
    interior = rr < DISK_SEMI_A * 0.85

    for B, pole_pa in [(1.0, 0.0), (20.0, 15.0), (60.0, -40.0), (85.0, 170.0), (-30.0, 90.0)]:
        map_img, valid_mask = _disk_to_map(img, CX, CY, REQ_PX, RPOL_PX, pole_pa, B)
        recovered = _map_to_disk(
            map_img, valid_mask, (SIZE, SIZE), CX, CY, REQ_PX, RPOL_PX, pole_pa, B,
        )
        diff = np.abs(recovered[interior] - img[interior])
        assert diff.mean() < 0.03, f"B={B} pole_pa={pole_pa}: mean abs diff too high: {diff.mean():.4f}"


def test_map_to_disk_never_leaks_content_from_invalid_map_cells():
    """A disk pixel whose (phi,lam) lands on a map cell with valid_mask=0
    must come back as 0, not some interpolated/extrapolated garbage --
    guards the valid_sampled>0.5 gating in _map_to_disk()."""
    img = _synthetic_globe_texture()
    map_img, _real_valid_mask = _disk_to_map(img, CX, CY, REQ_PX, RPOL_PX, 15.0, 20.0)
    all_invalid = np.zeros_like(_real_valid_mask)
    recovered = _map_to_disk(map_img, all_invalid, (SIZE, SIZE), CX, CY, REQ_PX, RPOL_PX, 15.0, 20.0)
    assert not recovered.any()


def test_disk_to_map_zero_outside_valid_region():
    """map_image itself must be exactly 0 wherever valid_mask is 0 (not a
    stray sample from cv2.remap's border handling) -- callers combining
    multiple frames' maps must be able to trust valid_mask alone, without
    also checking map_image for "accidentally nonzero invalid" values."""
    img = _synthetic_globe_texture()
    map_img, valid_mask = _disk_to_map(img, CX, CY, REQ_PX, RPOL_PX, 15.0, 20.0)
    assert not map_img[valid_mask < 0.5].any()


PERIOD_HOURS = 10.0
B_DEG = 20.0
POLE_PA_DEG = 15.0
POLAR_EQ_RATIO = RPOL_PX / REQ_PX


def _write_frame(tmp_dir, name, image):
    path = Path(tmp_dir) / name
    image_io.write_tif_16bit(np.clip(image, 0.0, 1.0), path)
    return path


def test_map_space_window_stack_single_frame_matches_input_in_interior():
    """A single-row window (dt_sec=0) should reduce to that frame's own
    disk_to_map/map_to_disk round trip (already validated by the Phase A
    tests above) times the limb feather -- i.e. no rotation logic can be
    exercised with one frame, this just confirms map_space_window_stack's
    plumbing (file I/O, single-row averaging, feathering) doesn't corrupt
    the trivial case."""
    img = _synthetic_globe_texture()
    t_ref = datetime(2026, 8, 16, 0, 0, 0, tzinfo=timezone.utc)
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_frame(tmp, "frame0.tif", img)
        rows = [{"path": path, "timestamp": t_ref}]
        disk_out, info = map_space_window_stack(
            rows, t_ref, PERIOD_HOURS, CX, CY, DISK_SEMI_A,
            POLE_PA_DEG, B_DEG, POLAR_EQ_RATIO,
        )

    assert info["n_stacked"] == 1
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
    rr = np.sqrt((xx - CX) ** 2 + (yy - CY) ** 2)
    interior = rr < DISK_SEMI_A * 0.85
    diff = np.abs(disk_out[interior] - img[interior])
    assert diff.mean() < 0.03, f"mean abs diff too high: {diff.mean():.4f}"


def test_map_space_window_stack_recovers_reference_frame_after_rotation():
    """Core hypothesis test for Phase B's shift convention: synthesize a
    second "frame" by applying the EXACT INVERSE of the shift map_space_
    window_stack() will itself apply (using _map_to_disk's new lam_shift
    parameter, added specifically to make this self-consistency check
    possible), then confirm stacking it together with the true t_reference
    frame recovers the original image. This uses only the codebase's own
    already-validated primitives to build the ground truth (never an
    independently-authored renderer, which risks encoding the same sign
    bug in both the test and the code and rubber-stamping it) -- the same
    round-trip philosophy tests/test_reprojection.py already uses for
    _reprojected_position()."""
    img = _synthetic_globe_texture()
    ref_map, ref_valid = _disk_to_map(img, CX, CY, REQ_PX, RPOL_PX, POLE_PA_DEG, B_DEG)

    t_ref = datetime(2026, 8, 16, 0, 0, 0, tzinfo=timezone.utc)
    dt_sec = 3600.0
    period_sec = PERIOD_HOURS * 3600.0
    delta_lambda = (dt_sec / period_sec) * 2.0 * np.pi
    sign = -1.0  # flip_direction=False, matching map_space_window_stack's own convention
    shift = sign * delta_lambda

    # Build "what a frame captured at dt_sec would look like": the exact
    # inverse of the lam_shift map_space_window_stack() will apply when it
    # samples this frame back out (see _map_to_disk's docstring).
    frame_at_dt = _map_to_disk(
        ref_map, ref_valid, (SIZE, SIZE), CX, CY, REQ_PX, RPOL_PX,
        POLE_PA_DEG, B_DEG, lam_shift=-shift,
    )

    with tempfile.TemporaryDirectory() as tmp:
        path_ref = _write_frame(tmp, "frame_ref.tif", img)
        path_dt = _write_frame(tmp, "frame_dt.tif", frame_at_dt)
        rows = [
            {"path": path_ref, "timestamp": t_ref},
            {"path": path_dt, "timestamp": t_ref + timedelta(seconds=dt_sec)},
        ]
        disk_out, info = map_space_window_stack(
            rows, t_ref, PERIOD_HOURS, CX, CY, DISK_SEMI_A,
            POLE_PA_DEG, B_DEG, POLAR_EQ_RATIO,
        )

    assert info["n_stacked"] == 2
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
    rr = np.sqrt((xx - CX) ** 2 + (yy - CY) ** 2)
    interior = rr < DISK_SEMI_A * 0.85
    diff = np.abs(disk_out[interior] - img[interior])
    # Looser than the single-round-trip Phase A tolerance: this path goes
    # through two extra map round trips (constructing frame_at_dt, then
    # re-deriving it) plus 16-bit tif quantization.
    assert diff.mean() < 0.04, f"mean abs diff too high: {diff.mean():.4f}"
    assert np.percentile(diff, 95) < 0.10, f"95th percentile diff too high: {np.percentile(diff, 95):.4f}"


def test_map_space_window_stack_limb_feather_is_gradual_not_a_hard_step():
    """Phase A's real-data validation found _map_to_disk alone produces a
    hard boolean cutoff at the limb (see project_map_space_derotation_
    roadmap memory) -- map_space_window_stack's _MAP_SPACE_LIMB_FEATHER_PX
    taper is the fix. Confirm the taper actually exists structurally: the
    band between disk_radius_px and disk_radius_px+feather must contain
    intermediate values (not jump straight from interior-level to 0), and
    must reach exactly 0 beyond the feather band."""
    img = _synthetic_globe_texture()
    t_ref = datetime(2026, 8, 16, 0, 0, 0, tzinfo=timezone.utc)
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_frame(tmp, "frame0.tif", img)
        rows = [{"path": path, "timestamp": t_ref}]
        disk_out, _info = map_space_window_stack(
            rows, t_ref, PERIOD_HOURS, CX, CY, DISK_SEMI_A,
            POLE_PA_DEG, B_DEG, POLAR_EQ_RATIO,
        )

    # Sample along +x from center: just inside the disk, mid-feather-band,
    # and past the feather band.
    row_y = int(CY)
    near_limb = disk_out[row_y, int(CX + DISK_SEMI_A - 2)]
    mid_band = disk_out[row_y, int(CX + DISK_SEMI_A + 6)]
    past_band = disk_out[row_y, int(CX + DISK_SEMI_A + 13)]
    assert past_band == 0.0
    assert 0.0 <= mid_band < near_limb, (
        f"expected a gradual taper, got near_limb={near_limb:.4f} mid_band={mid_band:.4f}"
    )


if __name__ == "__main__":
    test_disk_to_map_valid_mask_is_roughly_half_the_globe()
    print("disk_to_map valid mask is roughly half the globe: OK")
    test_disk_to_map_to_disk_round_trip_no_rotation()
    print("disk_to_map_to_disk round trip (no rotation): OK")
    test_round_trip_holds_across_b_and_pole_pa_sweep()
    print("round trip holds across B/pole_pa sweep: OK")
    test_map_to_disk_never_leaks_content_from_invalid_map_cells()
    print("map_to_disk never leaks content from invalid map cells: OK")
    test_disk_to_map_zero_outside_valid_region()
    print("disk_to_map zero outside valid region: OK")
    test_map_space_window_stack_single_frame_matches_input_in_interior()
    print("map_space_window_stack single frame matches input: OK")
    test_map_space_window_stack_recovers_reference_frame_after_rotation()
    print("map_space_window_stack recovers reference frame after rotation: OK")
    test_map_space_window_stack_limb_feather_is_gradual_not_a_hard_step()
    print("map_space_window_stack limb feather is gradual: OK")
    print("\nAll checks passed.")
