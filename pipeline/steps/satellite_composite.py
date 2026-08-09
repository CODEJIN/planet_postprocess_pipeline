"""
Satellite/shadow multi-rate compositing subsystem for Step 4 de-rotation.

Split out of pipeline/steps/derotate_stack.py (2026-08-09) — everything here
is gated behind config.satellite.enabled/composite_enabled (both default
False) and is independent of the core de-rotation pipeline, which never
imports from this module in the other direction.

Provides:
  - Motion-based Gaussian/capsule blend masks and the exp9 compositing method
    (_apply_satellite_composite) — overwrites planet TIFs with Europa/shadow
    composited stacks when config.satellite.composite_enabled.
  - Satellite tracker orientation auto-detection (_detect_tracker_flip_ns) and
    plate-scale auto-calibration from observed shadow transits
    (_auto_calibrate_plate_scale).
  - Three small orchestration helpers (resolve_tracker_flip_ns,
    compute_session_disk_radius, run_plate_scale_calibration) extracted from
    logic that used to be inlined directly in derotate_stack.py's run().
"""
from __future__ import annotations

import contextlib
import io
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

from pipeline.config import PipelineConfig
from pipeline.modules import derotation, image_io
from pipeline.modules.derotation import (
    apply_shift,
    find_disk_center,
    query_horizons_np_ang,
    quality_weighted_stack,
)
from pipeline.modules.satellite_tracker import (
    SatelliteTracker,
    _load_skyfield_kernels,
    _MOON_SF_ID,
    detect_tracker_flip_ns,
)
from pipeline.modules.wavelet import sharpen

# Duplicated from derotate_stack.py rather than imported: this module is
# imported BY derotate_stack.py, so importing the list back from there would
# create a circular import. It's a 5-item list, not worth a third
# shared-constants module for.
_FILT_PREF      = ["IR", "R", "G", "B", "CH4"]
_FILT_PREF_EXT  = ["IR", "R", "G", "B", "CH4", "color"]


# ── Satellite compositing helpers (exp9 method: motion-based Gaussian blend) ──

_SATELLITE_RADII_KM: Dict[str, float] = {
    "Io":       1_821.6,
    "Europa":   1_560.8,
    "Ganymede": 2_631.2,
    "Callisto": 2_410.3,
}


# ── Satellite compositing helpers (exp9 method: motion-based Gaussian blend) ──

def _gaussian_mask(shape: Tuple[int, int], cx: float, cy: float, sigma: float) -> np.ndarray:
    H, W = shape
    ys, xs = np.ogrid[:H, :W]
    dist_sq = (xs - cx) ** 2 + (ys - cy) ** 2
    return np.exp(-dist_sq / (2.0 * sigma ** 2)).astype(np.float32)


def _capsule_gaussian_mask(
    shape: Tuple[int, int],
    traj_xy: List[Tuple[float, float]],
    sigma_perp: float,
) -> np.ndarray:
    """Capsule-shaped Gaussian: exp(-min_dist_to_polyline² / 2σ²).

    Area grows linearly with smearing length (vs quadratically for circular),
    keeping the blend region tight along the trajectory axis.
    """
    H, W = shape
    ys, xs = np.mgrid[0:H, 0:W]
    xs_f = xs.astype(np.float32)
    ys_f = ys.astype(np.float32)

    if len(traj_xy) == 1:
        dx = xs_f - traj_xy[0][0]
        dy = ys_f - traj_xy[0][1]
        return np.exp(-(dx ** 2 + dy ** 2) / (2.0 * sigma_perp ** 2)).astype(np.float32)

    min_dist_sq = np.full((H, W), np.inf, dtype=np.float32)
    for i in range(len(traj_xy) - 1):
        p0 = np.array(traj_xy[i],     dtype=np.float64)
        p1 = np.array(traj_xy[i + 1], dtype=np.float64)
        seg = p1 - p0
        seg_len_sq = float(np.dot(seg, seg))
        dx = xs_f - p0[0]
        dy = ys_f - p0[1]
        if seg_len_sq < 1e-6:
            dist_sq = dx ** 2 + dy ** 2
        else:
            t = np.clip((dx * seg[0] + dy * seg[1]) / seg_len_sq, 0.0, 1.0)
            cx_s = (p0[0] + t * seg[0]).astype(np.float32)
            cy_s = (p0[1] + t * seg[1]).astype(np.float32)
            dist_sq = (xs_f - cx_s) ** 2 + (ys_f - cy_s) ** 2
        min_dist_sq = np.minimum(min_dist_sq, dist_sq)

    return np.exp(-min_dist_sq / (2.0 * sigma_perp ** 2)).astype(np.float32)


def _compute_sigma_from_motion(
    label: str,
    positions: List,
    ref_pos,
    apparent_r_px: float,
    coverage_scale: float,
) -> float:
    """Motion-based Gaussian blend sigma.

    sigma = max(max_motion_px, apparent_radius_px) × coverage_scale

    ref_pos: canonical SatellitePos at window center time (same for all filters).
    α at the farthest streak endpoint = exp(−1/(2×coverage_scale²))  (exp9 validated).
    """
    max_motion = 0.0
    if ref_pos is not None:
        for pos in positions:
            if pos is None:
                continue
            d = float(np.hypot(pos.x_px - ref_pos.x_px, pos.y_px - ref_pos.y_px))
            max_motion = max(max_motion, d)
    effective = max(max_motion, apparent_r_px)
    sigma = effective * coverage_scale
    alpha_ep = float(np.exp(-max_motion ** 2 / (2 * sigma ** 2))) if sigma > 0 else 0.0
    print(
        f"      [σ/{label}] apparent_r={apparent_r_px:.2f}px  "
        f"max_motion={max_motion:.2f}px  σ={sigma:.2f}px  α@ep={alpha_ep:.3f}"
    )
    return sigma


def _apparent_radius_px(moon_name: str, t_ref, plate_scale: float) -> float:
    """Satellite apparent radius in pixels at t_ref (Skyfield + LTT correction)."""
    from pipeline.modules.satellite_tracker import _load_skyfield_kernels, _MOON_SF_ID
    r_km = _SATELLITE_RADII_KM.get(moon_name, 1_560.8)
    sf = _load_skyfield_kernels()
    if sf is None:
        return 3.0
    ts, eph, jup_moons = sf
    t_sf = ts.utc(t_ref.year, t_ref.month, t_ref.day,
                  t_ref.hour, t_ref.minute, t_ref.second)
    earth_km = eph["earth"].at(t_sf).position.km
    jup_km_t = eph["jupiter barycenter"].at(t_sf).position.km
    d_EJ_km  = float(np.linalg.norm(jup_km_t - earth_km))
    lt_days  = d_EJ_km / (299_792.458 * 86_400.0)
    t_emit   = ts.tt_jd(float(t_sf.tt) - lt_days)
    sf_id    = _MOON_SF_ID.get(moon_name, moon_name.lower())
    moon_km  = jup_moons[sf_id].at(t_emit).position.km
    d_earth_sat = float(np.linalg.norm(moon_km - earth_km))
    return r_km / d_earth_sat * 206_265.0 / plate_scale


def _satellite_translate_stack(
    rows: List[dict], positions: List, ref_pos,
    keep_color: bool = False,
) -> Optional[np.ndarray]:
    """Stack frames by pure translation to align satellite at ref_pos.

    ref_pos: canonical SatellitePos at window center time (same for all filters),
             so all filter stacks place the satellite at the same pixel coordinate.
    No planet warp — only the satellite/shadow region is reliably sharp.
    The planet background is smeared, but it is masked out by the Gaussian blend.

    keep_color: if True, return an (H, W, 3) stack preserving color channels
                (used for color-camera-mode TIFs).
    """
    if ref_pos is None:
        return None

    imgs: List[np.ndarray] = []
    weights: List[float] = []
    for i, row in enumerate(rows):
        pos = positions[i]
        if pos is None or not pos.on_disk:
            continue
        raw = image_io.read_tif(row["path"])
        img = raw.astype(np.float32) / 65535.0 if raw.dtype == np.uint16 else raw.astype(np.float32)
        if img.ndim == 3 and not keep_color:
            img = img.mean(axis=2)
        elif img.ndim == 2 and keep_color:
            img = np.stack([img, img, img], axis=2)

        adx, ady = row.get("align_shift_px", (0.0, 0.0))
        imgs.append(apply_shift(img, ref_pos.x_px - pos.x_px + adx, ref_pos.y_px - pos.y_px + ady))
        weights.append(float(row["norm_score"]))
    if not imgs:
        return None
    return quality_weighted_stack(imgs, weights)


def _planet_bg_estimate(
    rows: List[dict],
    positions: List,
    ref_pos,
    planet_bg: np.ndarray,
    keep_color: bool = False,
) -> Optional[np.ndarray]:
    """Quality-weighted average of planet_bg shifted by the same translate as _satellite_translate_stack.

    Estimates what the planet background looks like inside the satellite stack so
    that (sat_stack − bg_estimate) isolates the satellite signal from the background.
    Only on-disk frames (matching _satellite_translate_stack selection) are included.
    """
    if ref_pos is None:
        return None
    imgs: List[np.ndarray] = []
    weights: List[float] = []
    bg_base = planet_bg
    if bg_base.ndim == 3 and not keep_color:
        bg_base = bg_base.mean(axis=2).astype(np.float32)
    elif bg_base.ndim == 2 and keep_color:
        bg_base = np.stack([bg_base, bg_base, bg_base], axis=2).astype(np.float32)
    for i, row in enumerate(rows):
        pos = positions[i]
        if pos is None or not pos.on_disk:
            continue
        adx, ady = row.get("align_shift_px", (0.0, 0.0))
        imgs.append(apply_shift(bg_base, ref_pos.x_px - pos.x_px + adx, ref_pos.y_px - pos.y_px + ady))
        weights.append(float(row["norm_score"]))
    if not imgs:
        return None
    return quality_weighted_stack(imgs, weights)


def _compute_smearing_map(
    rows: List[dict],
    positions: List,
    ref_pos,
    sat_signal: np.ndarray,
    app_r: float,
    warp_params: Optional[dict] = None,
) -> Optional[np.ndarray]:
    """Estimate the satellite/shadow smearing baked into the planet composite.

    Uses a clean Gaussian template (depth estimated from sat_signal, shape from
    apparent radius) instead of sat_signal itself as the smearing kernel.
    This avoids amplifying the raw-vs-derotated noise present in sat_signal.

    warp_params: when provided, uses the de-rotation-warped shadow position
    for each frame (not the raw position) to place the smearing template.
    This is critical: the planet de-rotation warp displaces each frame's
    shadow by drift*(cos_pa, sin_pa) relative to its raw position, so the
    actual smearing pattern in the planet TIF is at warped positions, not
    raw positions.  Without this correction the smearing map is placed up to
    ~10 px away from the actual smear, leaving it un-subtracted and causing a
    double-shadow artifact when sat_signal is additively blended in.
    Keys: disk_cx, disk_cy, disk_r, period_hours, warp_scale, pole_pa_deg,
          polar_eq_ratio (optional, default 1.0), t_reference (datetime).

    Returns a map to subtract from planet before additive blending:
      planet_base = planet - smearing
      composite   = planet_base + alpha * sat_signal
    """
    if ref_pos is None or sat_signal is None:
        return None
    total_quality = sum(
        float(row["norm_score"])
        for i, row in enumerate(rows)
        if positions[i] is not None and positions[i].on_disk
    )
    if total_quality == 0:
        return None

    # Build a clean Gaussian template at ref_pos with sigma = app_r.
    # Depth is the mean of sat_signal within the satellite/shadow spot.
    # NOTE: apply_shift clips to [0,1], so we shift the non-negative spot_alpha
    # and multiply by depth (which may be negative for a shadow) afterward.
    shape2d = sat_signal.shape[:2]
    spot_alpha = _gaussian_mask(shape2d, ref_pos.x_px, ref_pos.y_px, app_r)
    spot_mask  = spot_alpha > np.exp(-0.5)  # pixels within 1σ of ref_pos
    sig = sat_signal.astype(np.float32)
    is_color_sig = sig.ndim == 3
    if is_color_sig:
        depth = np.array([np.mean(sig[:, :, c][spot_mask]) for c in range(sig.shape[2])],
                         dtype=np.float32)
        smearing_shape = sig.shape
    else:
        depth = float(np.mean(sig[spot_mask])) if spot_mask.any() else 0.0
        smearing_shape = shape2d

    # Pre-compute warp displacement parameters for warped-position smearing.
    _warp_active = False
    if warp_params is not None:
        try:
            _dcx       = float(warp_params["disk_cx"])
            _dcy       = float(warp_params["disk_cy"])
            _dr        = float(warp_params["disk_r"])
            _ph        = float(warp_params["period_hours"])
            _ws        = float(warp_params["warp_scale"])
            _pa        = float(warp_params["pole_pa_deg"])
            _per       = float(warp_params.get("polar_eq_ratio", 1.0))
            _tref      = warp_params["t_reference"]
            _period_sec = _ph * 3600.0
            _cos_pa    = float(np.cos(np.radians(_pa)))
            _sin_pa    = float(np.sin(np.radians(_pa)))
            _warp_r    = _dr * 1.05
            _polar_sq  = (1.0 / max(_per, 1e-3)) ** 2
            _warp_active = True
        except (KeyError, TypeError):
            pass

    # Shadows (depth < 0) are supported when sat_signal is a clean synthetic Gaussian
    # (not raw sat_signal).  Raw sat_signal had Gaussian cross-talk that inflated the
    # smearing map, but synthetic sat_signal has no such issue.
    depth_scalar = float(np.mean(depth)) if is_color_sig else depth
    if depth_scalar == 0.0:
        return None

    smearing = np.zeros(smearing_shape, dtype=np.float32)
    for i, row in enumerate(rows):
        pos = positions[i]
        if pos is None or not pos.on_disk:
            continue
        q = float(row["norm_score"]) / total_quality

        if _warp_active:
            # Compute warped position: where this frame's shadow lands in the
            # planet TIF after de-rotation warp is applied.
            # output_pos = raw_pos + drift * (cos_pa, sin_pa)
            t_frame = row["timestamp"]
            if hasattr(t_frame, "tzinfo") and t_frame.tzinfo is not None:
                t_frame = t_frame.replace(tzinfo=None)
            dt_sec      = (t_frame - _tref).total_seconds()
            delta_lam   = (dt_sec / _period_sec) * 2.0 * np.pi
            rx          = pos.x_px - _dcx
            ry          = pos.y_px - _dcy
            rx_eq       = rx * _cos_pa + ry * _sin_pa
            ry_pol      = -rx * _sin_pa + ry * _cos_pa
            depth_sq    = _warp_r ** 2 - rx_eq ** 2 - _polar_sq * ry_pol ** 2
            frame_depth = float(np.sqrt(max(0.0, depth_sq)))
            drift       = _ws * delta_lam * frame_depth
            warped_x    = pos.x_px + drift * _cos_pa
            warped_y    = pos.y_px + drift * _sin_pa
            dx          = warped_x - ref_pos.x_px
            dy          = warped_y - ref_pos.y_px
        else:
            dx = pos.x_px - ref_pos.x_px
            dy = pos.y_px - ref_pos.y_px

        shifted_alpha = apply_shift(spot_alpha, dx, dy)  # [0,1] — no clipping issue
        if is_color_sig:
            smearing += q * (shifted_alpha[:, :, np.newaxis] * depth[np.newaxis, np.newaxis, :])
        else:
            smearing += q * (shifted_alpha * depth)
    return smearing


def _blend_additive(
    planet: np.ndarray,
    sat_signal: Optional[np.ndarray],
    ref_pos,
    sigma: float,
    traj_xy: Optional[List[Tuple[float, float]]] = None,
    mask_shape: str = "circular",
) -> np.ndarray:
    """Blend background-corrected satellite signal into planet additively.

    composite = planet + alpha × sat_signal

    sat_signal = sat_stack − bg_estimate (background already subtracted).
    Because sat_signal ≈ 0 everywhere except at the satellite/shadow, a large
    sigma does NOT shift the planet disk: alpha × 0 = 0 far from the satellite.

    A per-channel DC bias in sat_signal (from imperfect bg_estimate) is corrected
    by measuring sat_signal where alpha ≈ 0 (off-satellite region) and subtracting
    that offset before blending.
    """
    if sat_signal is None or ref_pos is None or not ref_pos.on_disk:
        return planet
    if mask_shape == "capsule" and traj_xy:
        alpha = _capsule_gaussian_mask(planet.shape[:2], traj_xy, sigma)
    else:
        alpha = _gaussian_mask(planet.shape[:2], ref_pos.x_px, ref_pos.y_px, sigma)

    if planet.ndim == 3:
        alpha = alpha[:, :, np.newaxis]
    return np.clip(planet.astype(np.float32) + alpha * sat_signal.astype(np.float32), 0.0, 1.0)


def _blend_one(
    planet: np.ndarray,
    sat_stack: Optional[np.ndarray],
    ref_pos,
    sigma: float,
    traj_xy: Optional[List[Tuple[float, float]]] = None,
    mask_shape: str = "circular",
) -> np.ndarray:
    """Blend a single satellite or shadow stack into the planet image.

    mask_shape="circular": isotropic Gaussian at ref_pos with given sigma.
    mask_shape="capsule":  Gaussian along traj_xy polyline; sigma = sigma_perp.
    """
    if sat_stack is None or ref_pos is None or not ref_pos.on_disk:
        return planet
    if mask_shape == "capsule" and traj_xy:
        alpha = _capsule_gaussian_mask(planet.shape[:2], traj_xy, sigma)
    else:
        alpha = _gaussian_mask(planet.shape[:2], ref_pos.x_px, ref_pos.y_px, sigma)
    if planet.ndim == 3:
        alpha = alpha[:, :, np.newaxis]
    return np.clip((1.0 - alpha) * planet + alpha * sat_stack, 0.0, 1.0)


# ── NOT USED: Approach B — planet warp + satellite translation ─────────────────
#
# Experiment 10 tested applying the same planet de-rotation warp to each
# satellite-stack frame before translating to align the satellite.  The idea
# was that the background texture in the satellite stack would then match the
# planet stack, reducing the mismatch visible at the Gaussian blend boundary.
#
# Results (2026-05-05 Jupiter, Window 3):
#   IR  filter: max pixel difference vs pure-translation (exp9) = 0.12%
#   CH4 filter: max pixel difference vs pure-translation (exp9) = 0.67%
#   Visual: indistinguishable at any filter or zoom level.
#
# Root cause: The warp corrects only ~2.8 px per 4-minute interval at the
# satellite position.  This is far smaller than the stacking-induced background
# smear (which spans the full motion range, e.g. ~6 px for IR).  The smear is
# inherent to stacking N frames with different planet-background offsets; no
# per-frame warp correction can eliminate it without a fundamentally different
# compositing strategy (e.g., background subtraction or inpainting).
#
# Scalability: at 2× pixel scale (C14 + Barlow), both the warp correction and
# the motion range scale proportionally, so the improvement ratio stays ~24%
# and the absolute difference stays below 1.3% — still visually negligible.
#
# When to re-evaluate:
#   - If a background-subtraction or inpainting strategy is adopted, making
#     precise per-frame background texture alignment meaningful.
#   - If spectral analysis (not visual imaging) requires sub-pixel accuracy of
#     the planet background at the blend boundary.
#
# def _warp_displacement_at(x_sat, y_sat, dt_sec, cx, cy, disk_r,
#                            period_hours, warp_scale, pole_pa_deg, polar_eq_ratio):
#     """Return (dx, dy) displacement that planet warp applies at satellite pos."""
#     period_sec = period_hours * 3600.0
#     delta_lambda_rad = (dt_sec / period_sec) * 2.0 * np.pi
#     warp_radius = disk_r * 1.05
#     pole_pa_rad = np.radians(pole_pa_deg)
#     cos_pa = float(np.cos(pole_pa_rad))
#     sin_pa = float(np.sin(pole_pa_rad))
#     rx = x_sat - cx; ry = y_sat - cy
#     rx_eq  = rx * cos_pa + ry * sin_pa
#     ry_pol = -rx * sin_pa + ry * cos_pa
#     polar_scale_sq = (1.0 / max(polar_eq_ratio, 1e-3)) ** 2
#     depth_sq = warp_radius**2 - rx_eq**2 - polar_scale_sq * ry_pol**2
#     depth = float(np.sqrt(max(0.0, depth_sq)))
#     drift = warp_scale * delta_lambda_rad * depth
#     return drift * cos_pa, drift * sin_pa
#
# def _satellite_warp_translate_stack(rows, positions, ref_idx, t_ref,
#                                      disk_cx, disk_cy, disk_sr,
#                                      period_hours, warp_scale,
#                                      pole_pa_deg, polar_eq_ratio):
#     """Approach B: planet warp + additional translation to align satellite.
#
#     Step 1: apply planet de-rotation warp (same as the planet stack).
#     Step 2: compute where satellite lands after warp (analytical displacement).
#     Step 3: translate the residual difference to align satellite at ref pos.
#
#     Max pixel improvement vs pure translation (exp9): 0.12% (IR), 0.67% (CH4).
#     Visually indistinguishable — see NOT USED block above for full analysis.
#     """
#     from pipeline.modules.derotation import (
#         apply_shift, quality_weighted_stack, spherical_derotation_warp,
#     )
#     ref_pos = positions[ref_idx]
#     if ref_pos is None:
#         return None
#     imgs, weights = [], []
#     for i, row in enumerate(rows):
#         pos = positions[i]
#         if pos is None:
#             continue
#         raw = image_io.read_tif(row["path"])
#         img = raw.astype(np.float32) / 65535.0 if raw.dtype == np.uint16 else raw.astype(np.float32)
#         if img.ndim == 3:
#             img = img.mean(axis=2)
#         row_t = row["timestamp"]
#         row_t = row_t.replace(tzinfo=None) if row_t.tzinfo else row_t
#         dt_sec = (row_t - t_ref).total_seconds()
#         warped = spherical_derotation_warp(
#             img, dt_sec, disk_cx, disk_cy, disk_sr,
#             period_hours=period_hours,
#             scale=warp_scale,
#             flip_direction=False,
#             pole_pa_deg=pole_pa_deg,
#             polar_equatorial_ratio=polar_eq_ratio,
#         )
#         wdx, wdy = _warp_displacement_at(
#             pos.x_px, pos.y_px, dt_sec,
#             disk_cx, disk_cy, disk_sr,
#             period_hours, warp_scale, pole_pa_deg, polar_eq_ratio,
#         )
#         sat_x_after_warp = pos.x_px + wdx
#         sat_y_after_warp = pos.y_px + wdy
#         dx = ref_pos.x_px - sat_x_after_warp
#         dy = ref_pos.y_px - sat_y_after_warp
#         imgs.append(apply_shift(warped, dx, dy))
#         weights.append(float(row["norm_score"]))
#     if not imgs:
#         return None
#     return quality_weighted_stack(imgs, weights)
# ─────────────────────────────────────────────────────────────────────────────


def _poisson_solve_channel(
    planet_ch: np.ndarray,
    sat_ch: np.ndarray,
    interior: np.ndarray,
) -> np.ndarray:
    """Solve ∇²result = ∇²sat_ch inside `interior`, planet_ch as Dirichlet BC.

    Pure-numpy Conjugate Gradient — no scipy required.
    CG converges in at most n iterations (n = interior pixel count) and
    typically O(√n) in practice; for our small satellite blobs this is fast.
    """
    H, W = planet_ch.shape
    sat    = sat_ch.astype(np.float64)
    planet = planet_ch.astype(np.float64)

    # Keep interior 1 pixel away from image edges so every pixel has 4 valid neighbours.
    safe = np.zeros((H, W), dtype=bool)
    safe[1:H-1, 1:W-1] = True
    interior = interior & safe

    ys, xs = np.where(interior)
    n = len(ys)
    if n == 0:
        return planet_ch.copy()

    idx_map = np.full((H, W), -1, dtype=np.int32)
    idx_map[interior] = np.arange(n, dtype=np.int32)

    # Guidance: ∇²sat at each interior pixel.
    b = (4.0 * sat[ys, xs]
         - sat[ys - 1, xs] - sat[ys + 1, xs]
         - sat[ys, xs - 1] - sat[ys, xs + 1])

    # Add planet Dirichlet BC contribution from exterior neighbours.
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ny = ys + dy
        nx = xs + dx
        ext = idx_map[ny, nx] < 0
        if ext.any():
            b[ext] += planet[ny[ext], nx[ext]]

    # Operator A: discrete negative Laplacian restricted to interior pixels.
    def _apply_A(v: np.ndarray) -> np.ndarray:
        u = np.zeros((H, W), dtype=np.float64)
        u[interior] = v
        return (4.0 * u[ys, xs]
                - u[ys - 1, xs] - u[ys + 1, xs]
                - u[ys, xs - 1] - u[ys, xs + 1])

    # Conjugate Gradient (pure numpy, no preconditioning).
    x  = planet[interior].copy()
    r  = b - _apply_A(x)
    p  = r.copy()
    rr = np.dot(r, r)
    for _ in range(min(n, 500)):
        if rr < 1e-12:
            break
        Ap     = _apply_A(p)
        alpha  = rr / np.dot(p, Ap)
        x     += alpha * p
        r     -= alpha * Ap
        rr_new = np.dot(r, r)
        p      = r + (rr_new / rr) * p
        rr     = rr_new

    result = planet.copy()
    result[interior] = x
    return result


def _blend_poisson(
    planet: np.ndarray,
    sat_stack: Optional[np.ndarray],
    ref_pos,
    sigma: float,
    traj_xy: Optional[List[Tuple[float, float]]] = None,
    mask_shape: str = "circular",
) -> np.ndarray:
    """Gradient-domain Poisson blend: splice sat_stack texture into planet.

    Solves ∇²result = ∇²sat_stack inside the alpha mask (threshold 0.1) with
    planet values as Dirichlet boundary conditions at the mask edge.  This
    eliminates the DC colour cast that additive blending produces when the
    background-subtracted sat_signal has a per-filter residual offset.

    Falls back to _blend_one when scipy is unavailable or the mask is empty.
    """
    if sat_stack is None or ref_pos is None or not ref_pos.on_disk:
        return planet
    if mask_shape == "capsule" and traj_xy:
        alpha = _capsule_gaussian_mask(planet.shape[:2], traj_xy, sigma)
    else:
        alpha = _gaussian_mask(planet.shape[:2], ref_pos.x_px, ref_pos.y_px, sigma)

    interior = alpha > 0.1
    if not interior.any():
        return _blend_one(planet, sat_stack, ref_pos, sigma, traj_xy, mask_shape)

    sat = sat_stack.astype(np.float32)
    if planet.ndim == 3:
        result = np.stack(
            [_poisson_solve_channel(planet[:, :, c], sat[:, :, c], interior)
             for c in range(planet.shape[2])],
            axis=2,
        )
    else:
        result = _poisson_solve_channel(planet, sat, interior)
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def _apply_satellite_composite(
    window: dict,
    filter_results: dict,
    config: "PipelineConfig",
    tracker,
    pole_pa_deg: float,
    np_ang_deg: float,
    r_ref: float | None = None,
) -> Dict[str, dict]:
    """Apply multi-rate satellite compositing for all on-disk moons and shadows.

    Each filter uses its own disk center for tracker queries so that the
    satellite lands at the same disk-relative pixel in every filter TIF.
    Any Galilean moon body or shadow predicted to be on disk at t_center is
    composited; moons that are off disk are silently skipped.

    Returns:
        dict mapping filter name → {"cx": ..., "cy": ..., "r": ...} for each
        filter that was processed.  Used by aperture_contrast to read the
        exact disk center that was used for compositing (so it doesn't have
        to recompute from the post-composite image where bright moons can
        shift the Otsu threshold).
    """
    t_center = window["center_time"]
    t_center_naive = t_center.replace(tzinfo=None) if t_center.tzinfo else t_center
    mask_shape     = config.satellite.composite_mask_shape
    blend_mode     = config.satellite.composite_blend_mode
    coverage_scale = (
        config.satellite.composite_coverage_scale_capsule
        if mask_shape == "capsule"
        else config.satellite.composite_coverage_scale_circular
    )

    # ── Reference filter: plate_scale only ────────────────────────────────────
    ref_filt = next(
        (f for f in _FILT_PREF
         if filter_results.get(f, (None,))[0] is not None
         and filter_results[f][0].exists()),
        None,
    )
    if ref_filt is None:
        return {}
    disk_centers: Dict[str, dict] = {}
    ref_raw = image_io.read_tif(filter_results[ref_filt][0])
    ref_lum = ref_raw.astype(np.float32) / 65535.0 if ref_raw.dtype == np.uint16 else ref_raw.astype(np.float32)
    if ref_lum.ndim == 3:
        ref_lum = ref_lum.mean(axis=2)
    if r_ref is None:
        _, _, r_ref, _, _ = derotation.find_disk_center(ref_lum)
    plate_scale = tracker.get_plate_scale(r_ref, t_center_naive)

    # ── Per-filter composite ───────────────────────────────────────────────────
    for filt, (out_path, flog) in filter_results.items():
        if out_path is None or not out_path.exists():
            continue
        rows = window.get("per_filter", {}).get(filt, {}).get("included", [])
        if not rows:
            continue

        # Augment rows with disk-center alignment shifts from derotation.
        # Each source frame has its own disk center; derotation corrects this
        # wobble with align_shift_px before stacking. Without this correction,
        # translate_stack applies ephemeris-based shadow shifts to un-aligned
        # frames, so shadows from different frames land at different pixels →
        # elongated "line" in the stacked result instead of a circular spot.
        _align_map = {
            f["stem"]: f.get("align_shift_px", [0.0, 0.0])
            for f in flog.get("frames", [])
        }
        rows = [
            {**r, "align_shift_px": _align_map.get(r["stem"], [0.0, 0.0])}
            for r in rows
        ]

        print(f"    [{filt}] satellite composite…")

        planet_raw = image_io.read_tif(out_path)
        planet = (planet_raw.astype(np.float32) / 65535.0 if planet_raw.dtype == np.uint16
                  else planet_raw.astype(np.float32))
        is_color = planet.ndim == 3
        planet_lum = planet.mean(axis=2) if is_color else planet
        disk_cx, disk_cy, disk_sr, disk_sr_b, _ = derotation.find_disk_center(planet_lum)
        disk_centers[filt] = {"cx": float(disk_cx), "cy": float(disk_cy), "r": float(disk_sr)}
        polar_eq_ratio = float(disk_sr_b) / float(disk_sr) if disk_sr > 0 else 1.0
        warp_params = {
            "disk_cx":       disk_cx,
            "disk_cy":       disk_cy,
            "disk_r":        disk_sr,
            "period_hours":  config.derotation.rotation_period_hours,
            "warp_scale":    config.derotation.warp_scale,
            "pole_pa_deg":   pole_pa_deg,
            "polar_eq_ratio": polar_eq_ratio,
            "t_reference":   t_center_naive,
        }

        time_sorted = sorted(rows, key=lambda r: r["timestamp"])
        t_list = [r["timestamp"] for r in time_sorted]
        body_pos = tracker.get_positions(
            t_list, disk_cx, disk_cy, disk_sr,
            pole_pa_deg=pole_pa_deg, np_ang_deg=np_ang_deg,
        )
        shad_pos = tracker.get_shadow_positions(
            t_list, disk_cx, disk_cy, disk_sr,
            pole_pa_deg=pole_pa_deg, np_ang_deg=np_ang_deg,
        )

        # ref_pos = satellite/shadow position at exactly window["center_time"].
        # The planet stack is de-rotated to this exact moment; querying the
        # ephemeris at the same time eliminates the frame-discretisation error
        # from the old approach (closest frame timestamp, off by up to half a
        # frame interval).
        body_ref = tracker.get_positions(
            [t_center_naive], disk_cx, disk_cy, disk_sr,
            pole_pa_deg=pole_pa_deg, np_ang_deg=np_ang_deg,
        )
        shad_ref = tracker.get_shadow_positions(
            [t_center_naive], disk_cx, disk_cy, disk_sr,
            pole_pa_deg=pole_pa_deg, np_ang_deg=np_ang_deg,
        )

        composite = planet if is_color else planet_lum
        composited: List[str] = []

        # Body composites — any moon with a transit detected in the full-frame query
        for moon_name, positions in body_pos.items():
            ref = body_ref.get(moon_name, [None])[0]
            if ref is None or not ref.on_disk:
                continue
            app_r = _apparent_radius_px(moon_name, t_center_naive, plate_scale)
            if mask_shape == "capsule":
                traj_xy = [(p.x_px, p.y_px) for p in positions if p is not None and p.on_disk]
                sigma = app_r * coverage_scale
                print(f"      [σ/{moon_name}] apparent_r={app_r:.2f}px  σ_perp={sigma:.2f}px  (capsule)")
            else:
                traj_xy = None
                sigma = _compute_sigma_from_motion(moon_name, positions, ref, app_r, coverage_scale)
            stack = _satellite_translate_stack(time_sorted, positions, ref, keep_color=is_color)
            if blend_mode == "poisson":
                composite = _blend_poisson(composite, stack, ref, sigma, traj_xy=traj_xy, mask_shape=mask_shape)
            else:
                bg    = _planet_bg_estimate(time_sorted, positions, ref, composite, keep_color=is_color)
                sat_signal = (stack.astype(np.float32) - bg.astype(np.float32)) if (stack is not None and bg is not None) else stack
                smearing   = _compute_smearing_map(time_sorted, positions, ref, sat_signal, app_r, warp_params=warp_params)
                planet_base = np.clip(composite.astype(np.float32) - smearing, 0.0, 1.0) if smearing is not None else composite
                composite = _blend_additive(planet_base, sat_signal, ref, sigma, traj_xy=traj_xy, mask_shape=mask_shape)
            composited.append(f"{moon_name}(σ={sigma:.1f}px,{mask_shape[:3]},{blend_mode[:3]})")

        # Shadow composites — any shadow with a transit detected in the full-frame query
        for shad_name, positions in shad_pos.items():
            ref = shad_ref.get(shad_name, [None])[0]
            if ref is None or not ref.on_disk:
                continue
            moon_name = shad_name.replace("_shadow", "")
            app_r = _apparent_radius_px(moon_name, t_center_naive, plate_scale)
            if mask_shape == "capsule":
                traj_xy = [(p.x_px, p.y_px) for p in positions if p is not None and p.on_disk]
                sigma = app_r * coverage_scale
                print(f"      [σ/{shad_name}] apparent_r={app_r:.2f}px  σ_perp={sigma:.2f}px  (capsule)")
            else:
                traj_xy = None
                sigma = _compute_sigma_from_motion(shad_name, positions, ref, app_r, coverage_scale)
            stack = _satellite_translate_stack(time_sorted, positions, ref, keep_color=is_color)
            if blend_mode == "poisson":
                composite = _blend_poisson(composite, stack, ref, sigma, traj_xy=traj_xy, mask_shape=mask_shape)
            else:
                bg    = _planet_bg_estimate(time_sorted, positions, ref, composite, keep_color=is_color)
                sat_signal = (stack.astype(np.float32) - bg.astype(np.float32)) if (stack is not None and bg is not None) else stack
                smearing    = _compute_smearing_map(time_sorted, positions, ref, sat_signal, app_r, warp_params=warp_params)
                planet_base = np.clip(composite.astype(np.float32) - smearing, 0.0, 1.0) if smearing is not None else composite
                composite   = _blend_additive(planet_base, sat_signal, ref, sigma, traj_xy=traj_xy, mask_shape=mask_shape)
            composited.append(f"{shad_name}(σ={sigma:.1f}px,{mask_shape[:3]},{blend_mode[:3]})")

        if not composited:
            print(f"    [{filt}] no on-disk bodies/shadows — composite skipped")
            continue

        image_io.write_tif_16bit(composite, out_path)
        print(f"      → {out_path.name}  ({', '.join(composited)})")

    return disk_centers


def _detect_tracker_flip_ns(
    windows: List[dict],
    session_pole_pa: float,
    horizons_id: str = "599",
) -> Tuple[Optional[bool], float]:
    """Load frames from all windows and call detect_tracker_flip_ns().

    Aggregates frames across all windows (one preferred filter per window) to
    maximise signal, then delegates to satellite_tracker.detect_tracker_flip_ns.

    Returns (flip_ns, confidence): flip_ns=None if inconclusive or not applicable.
    """
    print("  [tracker_flip] Auto-detecting tracker N/S orientation…")

    frames: List[np.ndarray] = []
    cx_ref = cy_ref = r_ref = None

    for win in windows:
        filt = next(
            (f for f in _FILT_PREF_EXT
             if f in win.get("per_filter", {}) and win["per_filter"][f].get("included")),
            None,
        )
        if filt is None:
            continue
        for row in win["per_filter"][filt]["included"]:
            try:
                raw = image_io.read_tif(row["path"])
                lum = raw if raw.ndim == 2 else raw.mean(axis=2).astype(np.float32)
                lum = lum.astype(np.float32)
                if lum.max() > 1.5:
                    lum /= 65535.0
                if cx_ref is None:
                    cx_ref, cy_ref, r_ref, *_ = find_disk_center(lum)
                    if r_ref < 5:
                        cx_ref = None
                        continue
                frames.append(lum)
            except Exception:
                continue

    if not frames or cx_ref is None:
        print("  [tracker_flip] No usable frames — cannot auto-detect")
        return None, 0.0

    try:
        flip_ns, confidence = detect_tracker_flip_ns(
            frames=frames,
            cx=cx_ref, cy=cy_ref,
            disk_radius_px=r_ref,
            pole_pa_deg=session_pole_pa,
            horizons_id=horizons_id,
        )
        status = "INCONCLUSIVE" if flip_ns is None else f"flip_ns={flip_ns}"
        print(
            f"  [tracker_flip] → {status}  (confidence={confidence:.3f}, "
            f"n_frames={len(frames)})"
        )
        return flip_ns, confidence

    except Exception as exc:
        warnings.warn(f"  [tracker_flip] Detection failed: {exc}")
        return None, 0.0


def _auto_calibrate_plate_scale(
    scores: dict,
    tracker: "SatelliteTracker",
    session_r_ref: float,
    pole_pa_deg: float,
    np_ang_deg: float,
    *,
    crop: int = 20,
    safe_dist: float = -38.0,
    min_depth: float = 0.05,
    min_frames: int = 3,
) -> Optional[dict]:
    """2-param (cx + ps) lstsq calibration using shadow transit frames.

    Scans ALL session frames from the step-3 scores dict (window-selection-
    independent), finds frames where a shadow is on-disk and at least
    |safe_dist| px from the limb, auto-detects the shadow position via argmin,
    then fits:

        actual_x = cx_fit + pred_dx_px * k

    where pred_dx_px = predicted_shadow_x − disk_cx and k = ps_nom / ps_fit.

    Returns dict(ps_fit, cx_offset, dps_pct, n, rmse_nom, rmse_fit) or None.
    """

    _WAVELET = [200., 200., 200., 0., 0., 0.]

    # ── Collect all IR frame paths & timestamps from step-3 scores ───────────
    # Using scores (all session frames) rather than selected windows so that
    # shadow frames excluded by the de-overlap step still contribute to calibration.
    frame_info: list = []
    filt = next((f for f in _FILT_PREF if scores.get(f)), None)
    if filt is not None:
        for row in sorted(scores[filt], key=lambda r: r["timestamp"]):
            path = row.get("path")
            ts   = row.get("timestamp")
            if path and ts:
                t = ts.replace(tzinfo=None) if getattr(ts, "tzinfo", None) else ts
                frame_info.append((path, t))

    if not frame_info:
        return None

    # ── Per-frame disk_cx (find_disk_center on each frame) ───────────────────
    frame_cx: dict = {}
    for path, _ in frame_info:
        try:
            raw = image_io.read_tif(path)
            lum = raw if raw.ndim == 2 else raw.mean(axis=2).astype("float32")
            if lum.max() > 1.5:
                lum /= 65535.0
            cx, cy, *_ = derotation.find_disk_center(lum)
            frame_cx[path] = (float(cx), float(cy))
        except Exception:
            pass

    if not frame_cx:
        return None

    session_cx = float(np.median([v[0] for v in frame_cx.values()]))
    session_cy = float(np.median([v[1] for v in frame_cx.values()]))

    # ── Bulk shadow position query (suppress per-moon print spam) ─────────────
    valid_frames = [(p, t) for p, t in frame_info if p in frame_cx]
    all_times = [t for _, t in valid_frames]

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        shad_dict = tracker.get_shadow_positions(
            all_times, session_cx, session_cy, session_r_ref,
            pole_pa_deg=pole_pa_deg,
            np_ang_deg=np_ang_deg,
        )

    if not shad_dict:
        return None

    # Per-frame: pick the first on-disk shadow that's safe from limb
    transit_by_idx: dict = {}
    for shadow_key, pos_list in shad_dict.items():
        for i, pos in enumerate(pos_list):
            if i in transit_by_idx:
                continue
            if pos.on_disk and (pos.dist_px - session_r_ref) < safe_dist:
                transit_by_idx[i] = (pos, shadow_key)

    if not transit_by_idx:
        return None

    # ── argmin shadow detection + data collection ─────────────────────────────
    pred_dx_pxs: list = []
    actual_xs:   list = []
    disk_cxs:    list = []

    for i, (path, _) in enumerate(valid_frames):
        if i not in transit_by_idx:
            continue
        pos, _ = transit_by_idx[i]

        disk_cx_frame = frame_cx[path][0]
        # pos.x_px was computed with session_cx; adjust search centre for per-frame cx
        pred_x = pos.x_px + (disk_cx_frame - session_cx)
        pred_y = pos.y_px

        try:
            raw = image_io.read_tif(path)
            lum = raw if raw.ndim == 2 else raw.mean(axis=2).astype("float32")
            if lum.max() > 1.5:
                lum /= 65535.0
            lum = sharpen(lum, levels=6, amounts=_WAVELET)
            lum = np.clip(lum, 0., 1.)
            h, w = lum.shape

            x0 = max(0, int(round(pred_x)) - crop)
            x1 = min(w, int(round(pred_x)) + crop + 1)
            y0 = max(0, int(round(pred_y)) - crop)
            y1 = min(h, int(round(pred_y)) + crop + 1)
            patch = lum[y0:y1, x0:x1]
            if patch.size == 0:
                continue
            depth = float(patch.max() - patch.min())
            if depth < min_depth:
                continue

            idx = np.unravel_index(np.argmin(patch), patch.shape)
            actual_x = float(x0 + idx[1])

            # pred_dx_px is independent of per-frame cx (= dx_arcsec / ps_nom)
            pred_dx_pxs.append(float(pos.x_px - session_cx))
            actual_xs.append(actual_x)
            disk_cxs.append(disk_cx_frame)

        except Exception:
            continue

    n = len(pred_dx_pxs)
    if n < min_frames:
        return None

    pred_dx = np.array(pred_dx_pxs)
    actual  = np.array(actual_xs)
    dcxs    = np.array(disk_cxs)

    # 2-param lstsq: actual_x = alpha + k * pred_dx_px
    A = np.column_stack([np.ones(n), pred_dx])
    coef, _, _, _ = np.linalg.lstsq(A, actual, rcond=None)
    cx_fit = float(coef[0])
    k      = float(coef[1])          # = ps_nom / ps_fit

    ps_nom  = tracker._plate_scale   # nominal, already cached
    ps_fit  = ps_nom / k
    cx_offset = cx_fit - session_cx  # systematic correction to add to disk_cx

    rmse_fit = float(np.sqrt(np.mean((actual - A @ coef) ** 2)))
    rmse_nom = float(np.sqrt(np.mean((actual - (dcxs + pred_dx)) ** 2)))

    return dict(
        ps_fit=ps_fit,
        ps_nom=ps_nom,
        cx_offset=cx_offset,
        dps_pct=100.0 * (ps_fit - ps_nom) / ps_nom,
        n=n,
        rmse_nom=rmse_nom,
        rmse_fit=rmse_fit,
    )


# ── Orchestration helpers extracted from derotate_stack.py's run() ────────────
# (previously inlined directly in run() rather than as function calls; moved
# here verbatim, just wrapped, as part of the derotate_stack.py/
# satellite_composite.py split — see run()'s call sites for the exact
# call sequence and guard conditions these depend on.)

def resolve_tracker_flip_ns(
    config: PipelineConfig,
    windows: List[dict],
    session_pole_pa: float,
    derot_flip: bool,
) -> bool:
    """Resolve the satellite tracker's camera-orientation flip.

    Independent of derot_flip: tells the tracker which way is "north" in the
    image. Priority: explicit sat_cfg override -> belt-asymmetry auto-detect
    -> derot_flip.
    """
    sat_cfg = config.satellite
    if sat_cfg.flip_ns is not None:
        tracker_flip_ns = bool(sat_cfg.flip_ns)
        print(f"  [tracker] flip_ns override = {tracker_flip_ns} (from sat_cfg)")
    else:
        auto_flip, auto_conf = _detect_tracker_flip_ns(
            windows, session_pole_pa,
            horizons_id=config.derotation.horizons_id,
        )
        if auto_flip is not None:
            tracker_flip_ns = auto_flip
            print(f"  [tracker] flip_ns = {tracker_flip_ns} "
                  f"(belt-asymmetry auto-detect, confidence={auto_conf:.3f})")
        else:
            tracker_flip_ns = derot_flip
            print(f"  [tracker] flip_ns = {tracker_flip_ns} "
                  f"(fallback to derot_flip — belt detection inconclusive)")
    return tracker_flip_ns


def compute_session_disk_radius(windows: List[dict]) -> Optional[float]:
    """Session-wide median disk radius (for plate_scale stability).

    Only meaningful when a satellite tracker is active — the caller
    (run()) is expected to only call this when tracker is not None (matches
    the original inline guard), so this function doesn't need tracker itself.
    """
    _r_vals: list[float] = []
    for _win in windows:
        _filt = next(
            (f for f in _FILT_PREF
             if f in _win.get("per_filter", {})
             and _win["per_filter"][f].get("included")),
            None,
        )
        if _filt is None:
            continue
        for _row in _win["per_filter"][_filt]["included"]:
            try:
                _raw = image_io.read_tif(_row["path"])
                _lum = _raw if _raw.ndim == 2 else _raw.mean(axis=2).astype(np.float32)
                _lum = _lum.astype(np.float32)
                if _lum.max() > 1.5:
                    _lum /= 65535.0
                _, _, _r, *_ = derotation.find_disk_center(_lum)
                if _r > 5:
                    _r_vals.append(_r)
            except Exception:
                continue
    session_r_ref: Optional[float] = None
    if _r_vals:
        session_r_ref = float(np.median(_r_vals))
        print(f"  [satellite] session disk radius: median={session_r_ref:.3f}px "
              f"(n={len(_r_vals)}, range={min(_r_vals):.1f}–{max(_r_vals):.1f})")
    return session_r_ref


def run_plate_scale_calibration(
    tracker: "SatelliteTracker",
    config: PipelineConfig,
    windows: List[dict],
    results_03: dict,
    session_r_ref: float,
    session_pole_pa: float,
) -> Optional[dict]:
    """Plate-scale auto-calibration from an observed shadow transit, if present.

    Mutates tracker in place via tracker.set_plate_scale_calibration(...).
    Caller is expected to only call this when tracker is not None and
    session_r_ref is not None (matches the original inline guard in run()) —
    MUST run before any code that reads tracker.get_plate_scale(), since that
    caches its result on first use.

    Returns the calib_result dict (or None if no shadow transit was found) —
    the caller (run()) also needs this for its session-summary JSON, not just
    for the calibration side effect.
    """
    _t_mid_cal = sorted(windows, key=lambda w: w["center_time"])[len(windows) // 2]["center_time"]
    _np_ang_cal = query_horizons_np_ang(
        config.derotation.horizons_id, _t_mid_cal, config.derotation.observer_code,
    ) or 0.0
    print("  [satellite] running plate_scale auto-calibration…", flush=True)
    calib_result = _auto_calibrate_plate_scale(
        results_03.get("scores", {}), tracker, session_r_ref,
        pole_pa_deg=session_pole_pa,
        np_ang_deg=_np_ang_cal,
    )
    if calib_result is not None:
        tracker.set_plate_scale_calibration(
            calib_result["ps_fit"], calib_result["cx_offset"]
        )
        print(
            f"  [satellite] calibration: N={calib_result['n']}  "
            f"Δps={calib_result['dps_pct']:+.2f}%  "
            f"cx_offset={calib_result['cx_offset']:+.2f}px  "
            f"RMSE {calib_result['rmse_nom']:.3f}→{calib_result['rmse_fit']:.3f}px"
        )
    else:
        print("  [satellite] no shadow transit detected — plate_scale calibration skipped")
    return calib_result
