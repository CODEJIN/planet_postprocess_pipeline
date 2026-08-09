"""
Step 4 – De-rotation stacking.

For each selected time window from Step 3:
  1. Pre-scan all windows to detect session-median image-space pole PA.
  2. Auto-detect camera orientation (flip_ns) from pole PA sign.
  3. Apply spherical de-rotation warp using the correct pole PA.
  4. Sub-pixel translate-align rotated frames via phase correlation.
  5. Combine with quality-weighted mean stack (weights = Step 3 norm_scores).
  6. If config.satellite.enabled: predict Galilean moon/shadow positions via
     JPL Horizons + Skyfield, refine with CV blob detection, and log positions.
  7. If config.satellite.composite_enabled: apply multi-rate compositing (exp9
     method) — overwrite planet TIFs with Europa+shadow composited stacks.

Output (when config.save_step04 is True):
    <output_base>/step04_derotated/
        window_01/
            IR_derotated.tif
            R_derotated.tif
            ...
            derotation_log.json   ← includes satellite positions when enabled
        derotation_summary.txt
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from pipeline.config import PipelineConfig
from pipeline.modules import derotation, image_io
from pipeline.modules.derotation import (
    auto_detect_ns_flip,
    auto_detect_pole_pa,
    find_disk_center,
    pole_pa_from_disk_ellipse,
    query_horizons_np_ang,
    spherical_derotation_warp,
)
from pipeline.modules.satellite_tracker import detect_tracker_flip_ns

# Mono filters in priority order; "color" is appended for color-camera sessions.
_FILT_PREF      = ["IR", "R", "G", "B", "CH4"]
_FILT_PREF_EXT  = ["IR", "R", "G", "B", "CH4", "color"]

# Physical mean radii of Galilean moons (km) — used for apparent-size computation
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
    from pipeline.modules.derotation import apply_shift, quality_weighted_stack
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
    from pipeline.modules.derotation import apply_shift, quality_weighted_stack
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
    from pipeline.modules.derotation import apply_shift
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


def _scan_session_pole_pa(
    scores: dict,
    config: PipelineConfig,
) -> Optional[float]:
    """Return the session-median image-space pole PA from all input frames.

    Iterates every frame in the preferred filter (from step03 scores dict,
    which covers all input TIFs regardless of window selection), computes
    per-frame belt-gradient PA, and returns the median.
    """
    # Pick the first preferred filter that has any scored frames.
    filt = next((f for f in _FILT_PREF_EXT if scores.get(f)), None)
    if filt is None:
        return None

    all_rows: List[dict] = sorted(scores[filt], key=lambda r: r["timestamp"])
    print(f"  [pole_pa] Pre-scanning {len(all_rows)} frame(s) for image-space pole PA…")

    # Detect disk geometry once from the middle frame (stable across session).
    mid_row = all_rows[len(all_rows) // 2]
    try:
        mid_raw = image_io.read_tif(mid_row["path"])
        mid_lum = mid_raw if mid_raw.ndim == 2 else mid_raw.mean(axis=2).astype(np.float32)
        cx, cy, semi_a, *_ = find_disk_center(mid_lum)
        if semi_a < 5:
            raise ValueError("disk too small")
    except Exception as exc:
        warnings.warn(f"  [pole_pa] disk detection failed: {exc}")
        return None

    raw_pas: List[float] = []
    for i, row in enumerate(all_rows):
        try:
            raw = image_io.read_tif(row["path"])
            lum = raw if raw.ndim == 2 else raw.mean(axis=2).astype(np.float32)
            pa = auto_detect_pole_pa(frames=[lum], cx=cx, cy=cy, disk_radius_px=semi_a)
            print(f"    frame {i+1}/{len(all_rows)}: raw pole_pa = {pa:.1f}° via {filt} [belt_gradient]")
            raw_pas.append(pa)
        except Exception as exc:
            try:
                raw = image_io.read_tif(row["path"])
                lum = raw if raw.ndim == 2 else raw.mean(axis=2).astype(np.float32)
                pa = pole_pa_from_disk_ellipse(lum)
                if pa is not None:
                    print(f"    frame {i+1}/{len(all_rows)}: raw pole_pa = {pa:.1f}° via {filt} [disk_ellipse]")
                    raw_pas.append(pa)
            except Exception:
                pass

    if not raw_pas:
        return None

    session_pa = float(np.median(raw_pas))
    raw_str = [f"{p:.1f}" for p in raw_pas]
    print(
        f"  [pole_pa] session pole_pa = {session_pa:.1f}° "
        f"(n={len(raw_pas)}, raw: {raw_str})"
    )
    return session_pa


def _detect_session_flip_ns(
    windows: List[dict],
    config: PipelineConfig,
    session_pole_pa: float,
) -> Tuple[bool, float, float]:
    """Detect de-rotation warp direction from atmospheric feature drift.

    Returns (derot_flip, ncc_flip_false, ncc_flip_true).
    derot_flip is passed as flip_direction to spherical_derotation_warp.
    NOTE: this does NOT determine satellite-tracker orientation — use
    sat_cfg.flip_ns for that (S-up cameras should set flip_ns=True there).
    Falls back to (False, 0.0, 0.0) when detection is ambiguous.

    Strategy: collect ALL frames from all windows (sorted by time) using the
    preferred filter, then slide pairs separated by window_frames positions.
    Each pair casts one vote; majority decides derot_flip.
    """
    print("  [derot_flip] Detecting de-rotation warp direction via drift test…")

    # Collect all frames across all windows using the preferred filter.
    filt = None
    all_rows: List[dict] = []
    for preferred in _FILT_PREF_EXT:
        for win in windows:
            pf = win.get("per_filter", {})
            if preferred in pf and pf[preferred].get("included"):
                all_rows.extend(pf[preferred]["included"])
        if len(all_rows) >= 2:
            filt = preferred
            break
        all_rows = []

    if len(all_rows) < 2:
        print("  [derot_flip] No suitable frames — defaulting to flip_direction=False")
        return False, 0.0, 0.0

    all_rows.sort(key=lambda r: r["timestamp"])

    # Load all frames as luminance arrays.
    loaded_frames: List[np.ndarray] = []
    loaded_ts: List = []
    for row in all_rows:
        raw = image_io.read_tif(row["path"])
        lum = raw if raw.ndim == 2 else raw.mean(axis=2).astype(np.float32)
        loaded_frames.append(lum)
        loaded_ts.append(row["timestamp"])

    # Find disk center from the middle frame (most representative).
    mid = len(loaded_frames) // 2
    try:
        cx, cy, semi_a, semi_b, _ = find_disk_center(loaded_frames[mid])
    except Exception:
        print("  [derot_flip] Disk detection failed — defaulting to flip_direction=False")
        return False, 0.0, 0.0
    if semi_a < 5:
        print("  [derot_flip] Disk too small — defaulting to flip_direction=False")
        return False, 0.0, 0.0
    polar_eq = float(np.clip(semi_b / max(semi_a, 1.0), 0.85, 1.0))

    # Step size = window_frames - 1: within one window of W frames, the max span
    # is index 0 ↔ index W-1, so pairs are separated by W-1 positions.
    W = config.quality.window_frames - 1
    t_center = loaded_ts[mid]

    votes: List[Tuple[bool, float, float, float]] = []  # (flip, confidence, ncc_f, ncc_t)

    for i in range(len(loaded_frames) - W):
        frames_pair = [loaded_frames[i], loaded_frames[i + W]]
        dt_pair = [
            (loaded_ts[i]     - t_center).total_seconds(),
            (loaded_ts[i + W] - t_center).total_seconds(),
        ]
        dt = (loaded_ts[i + W] - loaded_ts[i]).total_seconds()

        try:
            flip, ncc_f, ncc_t = auto_detect_ns_flip(
                frames=frames_pair,
                dt_sec_list=dt_pair,
                cx=cx, cy=cy,
                disk_radius_px=semi_a,
                period_hours=config.derotation.rotation_period_hours,
                warp_scale=config.derotation.warp_scale,
                pole_pa_deg=session_pole_pa,
                polar_equatorial_ratio=polar_eq,
            )
            confidence = abs(ncc_f - ncc_t)
            votes.append((flip, confidence, ncc_f, ncc_t))
            print(
                f"  [derot_flip] pair vote [{i}→{i+W}]: flip={flip}  confidence={confidence:.5f}"
                f"  [Δt={dt:.0f}s, filter={filt}]"
            )
        except Exception as exc:
            warnings.warn(f"  [derot_flip] pair [{i}→{i+W}] failed: {exc}")

    if not votes:
        print("  [derot_flip] No valid pairs — defaulting to flip_direction=False")
        return False, 0.0, 0.0

    n_true  = sum(1 for v, *_ in votes if v)
    n_false = len(votes) - n_true
    if n_true != n_false:
        derot_flip = n_true > n_false
    else:
        derot_flip = max(votes, key=lambda x: x[1])[0]

    best = max(votes, key=lambda x: x[1])
    ncc_f, ncc_t = best[2], best[3]

    print(
        f"  [derot_flip] → flip_direction={derot_flip}  "
        f"[{n_true}×True / {n_false}×False, {len(votes)} pair(s)]"
    )
    return derot_flip, ncc_f, ncc_t


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


def _measure_derot_confidence(
    windows: List[dict],
    config: "PipelineConfig",
    session_pole_pa: float,
    flip_ns: bool,
    scale_min: float = 0.50,
    scale_max: float = 1.20,
    n_steps: int = 13,
    min_rotation_deg: float = 3.0,
) -> dict:
    """Measure de-rotation confidence via high-pass NCC sweep.

    warp_scale is a physical constant (empirically calibrated on best-seeing data,
    default 0.80) and is NOT derived from this sweep.  Instead the sweep answers:
    "given that we apply config.derotation.warp_scale, how much does the belt
    structure actually support the de-rotation?"

    Returns a dict:
        ncc_at_config_scale : NCC at config.derotation.warp_scale — primary
                              confidence metric.  Low (<0.3) means belt structure
                              is too blurry/absent for reliable de-rotation;
                              consider using a shorter window.
        estimated_peak_scale: scale where NCC peaks — diagnostic only, NOT used
                              to set warp_scale.
        best_ncc            : maximum NCC across sweep — diagnostic.
        rotation_deg        : rotation span used for measurement.
        measured            : False if measurement could not be performed.

    Forward prediction uses flip_direction = not flip_ns because de-rotation undoes
    the drift while forward-prediction replicates it.
    """
    config_scale = config.derotation.warp_scale
    fallback = {
        "ncc_at_config_scale":  0.0,
        "estimated_peak_scale": config_scale,
        "best_ncc":             0.0,
        "rotation_deg":         0.0,
        "measured":             False,
    }

    print("  [derot_conf] Measuring de-rotation confidence via NCC sweep…")

    # Select the window with the longest time span.
    best_win: Optional[Tuple[dict, str]] = None
    best_span = 0.0
    for win in windows:
        filt = next(
            (f for f in _FILT_PREF_EXT
             if f in win.get("per_filter", {}) and win["per_filter"][f].get("included")),
            None,
        )
        if filt is None:
            continue
        rows = win["per_filter"][filt]["included"]
        if len(rows) < 2:
            continue
        ts = [r["timestamp"] for r in rows]
        span = (max(ts) - min(ts)).total_seconds()
        if span > best_span:
            best_span = span
            best_win = (win, filt)

    if best_win is None:
        print("  [derot_conf] No suitable window → confidence unmeasured")
        return fallback

    win, filt = best_win
    period_sec = config.derotation.rotation_period_hours * 3600.0
    rotation_deg = best_span / period_sec * 360.0
    if rotation_deg < min_rotation_deg:
        print(
            f"  [derot_conf] Rotation {rotation_deg:.1f}° < {min_rotation_deg}° "
            "→ confidence unmeasured"
        )
        return {**fallback, "rotation_deg": rotation_deg}

    rows = sorted(win["per_filter"][filt]["included"], key=lambda r: r["timestamp"])
    try:
        raw_e = image_io.read_tif(rows[0]["path"])
        raw_l = image_io.read_tif(rows[-1]["path"])
    except Exception as exc:
        warnings.warn(f"  [derot_conf] Frame read failed: {exc}")
        return fallback

    def _lum(raw: np.ndarray) -> np.ndarray:
        img = raw if raw.ndim == 2 else raw.mean(axis=2).astype(np.float32)
        return img.astype(np.float32) / 65535.0 if img.dtype == np.uint16 else img.astype(np.float32)

    lum_e = _lum(raw_e)
    lum_l = _lum(raw_l)

    cx, cy, semi_a, semi_b, _ = find_disk_center(lum_e)
    if semi_a < 5:
        print("  [derot_conf] Disk detection failed → confidence unmeasured")
        return fallback

    polar_eq = float(np.clip(semi_b / max(semi_a, 1.0), 0.85, 1.0))
    dt = (rows[-1]["timestamp"] - rows[0]["timestamp"]).total_seconds()

    # High-pass filter (σ=30 px) removes limb darkening before NCC.
    # Without it, the smooth radial limb-darkening gradient dominates and NCC
    # decreases monotonically with scale, so scale=0 always wins.
    _HP_SIGMA = 30.0

    def _highpass(img: np.ndarray) -> np.ndarray:
        return img - cv2.GaussianBlur(img, (0, 0), _HP_SIGMA)

    h, w = lum_e.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    disk_mask = ((xx - cx) ** 2 + (yy - cy) ** 2) < (0.7 * semi_a) ** 2
    ref_px = _highpass(lum_l)[disk_mask].astype(np.float64)
    if ref_px.std() < 1e-6:
        print("  [derot_conf] Reference frame featureless → confidence unmeasured")
        return fallback

    # Forward prediction replicates the drift (opposite of de-rotation direction).
    forward_flip = not flip_ns

    # Sweep points: uniform grid + config_scale explicitly included.
    sweep_scales = sorted(set(
        [float(s) for s in np.linspace(scale_min, scale_max, n_steps)]
        + [config_scale]
    ))

    best_ncc = -1.0
    estimated_peak_scale = config_scale
    ncc_at_config_scale = 0.0
    ncc_pairs: List[Tuple[float, float]] = []

    for scale in sweep_scales:
        warped = spherical_derotation_warp(
            lum_e, dt, cx, cy, semi_a,
            period_hours=config.derotation.rotation_period_hours,
            scale=scale,
            flip_direction=forward_flip,
            pole_pa_deg=session_pole_pa,
            polar_equatorial_ratio=polar_eq,
        )
        pred_px = _highpass(warped)[disk_mask].astype(np.float64)
        ncc = float(np.corrcoef(ref_px, pred_px)[0, 1]) if pred_px.std() > 1e-6 else 0.0
        ncc_pairs.append((scale, ncc))
        if ncc > best_ncc:
            best_ncc = ncc
            estimated_peak_scale = scale
        if abs(scale - config_scale) < 1e-9:
            ncc_at_config_scale = ncc

    ncc_str = "  ".join(f"{s:.2f}:{n:.4f}" for s, n in ncc_pairs)
    print(
        f"  [derot_conf] NCC sweep ({len(sweep_scales)} pts, Δt={dt:.0f}s, "
        f"{rotation_deg:.1f}°, {filt}):\n    {ncc_str}"
    )
    print(
        f"  [derot_conf] config_scale={config_scale:.2f}  "
        f"NCC@config={ncc_at_config_scale:.4f}  "
        f"peak_scale={estimated_peak_scale:.3f}  best_NCC={best_ncc:.4f}"
    )

    if ncc_at_config_scale < 0.30:
        print(
            f"  [derot_conf] WARNING: NCC={ncc_at_config_scale:.3f} at scale={config_scale:.2f} "
            "is low — belt structure may be too blurry for reliable de-rotation. "
            "Consider using a shorter window in Step 03."
        )

    return {
        "ncc_at_config_scale":  ncc_at_config_scale,
        "estimated_peak_scale": estimated_peak_scale,
        "best_ncc":             best_ncc,
        "rotation_deg":         rotation_deg,
        "measured":             True,
    }


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
    import contextlib, io
    from pipeline.modules.wavelet import sharpen

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


def run(
    config: PipelineConfig,
    results_03: dict,
    progress_callback=None,
    cancel_event=None,
) -> Dict[str, List[Dict]]:
    """Run Step 4 de-rotation stacking.

    Args:
        config:      Pipeline configuration.
        results_03:  Output of step03_quality_assess.run().

    Returns:
        {"windows": [{window_index, center_time, outputs, log}, ...]}
    """
    windows: List[dict] = results_03.get("windows", [])
    if not windows:
        print("  [WARNING] No time windows from Step 3 — de-rotation skipped.")
        return {"windows": []}

    print(f"  Processing {len(windows)} window(s) × {len(config.filters)} filter(s)…")
    print(f"  Period: {config.derotation.rotation_period_hours}h  "
          f"|  sub-pixel alignment: enabled")

    # ── Session-level pole PA ──────────────────────────────────────────────────
    session_pole_pa = _scan_session_pole_pa(results_03.get("scores", {}), config)
    if session_pole_pa is None:
        session_pole_pa = 0.0
        print("  [WARNING] pole_pa scan failed — using 0.0°")

    # ── De-rotation warp direction (derot_flip) ───────────────────────────────
    # Determines flip_direction for spherical_derotation_warp via feature drift test.
    # For N-up AND pure NS-flip cameras this is almost always False (leftward drift).
    # sat_cfg.flip_ns is NOT used here — it controls satellite tracker only.
    derot_flip, _ncc_f, _ncc_t = _detect_session_flip_ns(windows, config, session_pole_pa)

    # ── Satellite tracker orientation (tracker_flip_ns) ───────────────────────
    # Independent of derot_flip: tells the tracker which way is "north" in the image.
    # Priority: explicit sat_cfg override → belt-asymmetry auto-detect → derot_flip.
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

    # ── warp_scale: fixed at config value (empirically calibrated from good-seeing data) ──
    # The physical rotation rate does not change with seeing; the NCC sweep result
    # is unreliable when belt structures are blurry (returns low scale instead of
    # the true ~0.80).  Confidence is measured separately and logged for diagnostics.
    warp_scale = config.derotation.warp_scale

    # ── De-rotation confidence (diagnostic, does not change warp_scale) ────────
    derot_conf = _measure_derot_confidence(windows, config, session_pole_pa, derot_flip)

    # ── SatelliteTracker ───────────────────────────────────────────────────────
    tracker = None
    if sat_cfg.enabled:
        from pipeline.modules.satellite_tracker import SatelliteTracker
        tracker = SatelliteTracker(
            jupiter_horizons_id=config.derotation.horizons_id,
            observer_code=config.derotation.observer_code,
            flip_ew=sat_cfg.flip_ew,
            flip_ns=tracker_flip_ns,
        )
        print(f"  [satellite] tracker enabled  "
              f"(tracker_flip_ns={tracker_flip_ns}, flip_ew={sat_cfg.flip_ew})")

    # ── Session-wide median disk radius (for plate_scale stability) ────────────
    session_r_ref: float | None = None
    if tracker is not None:
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
        if _r_vals:
            session_r_ref = float(np.median(_r_vals))
            print(f"  [satellite] session disk radius: median={session_r_ref:.3f}px "
                  f"(n={len(_r_vals)}, range={min(_r_vals):.1f}–{max(_r_vals):.1f})")

    # ── plate_scale auto-calibration from shadow transit (if present) ──────────
    calib_result: Optional[dict] = None
    if tracker is not None and session_r_ref is not None:
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

    # ── Output directory ───────────────────────────────────────────────────────
    out_base: Optional[Path] = None
    if config.save_step04:
        out_base = config.step_dir(4, "derotated")
        out_base.mkdir(parents=True, exist_ok=True)
        print(f"  Output → {out_base}")
    else:
        print("  save_step04=False: results not written to disk")

    # ── Process each window ────────────────────────────────────────────────────
    all_results: List[dict] = []
    _conf_ncc  = derot_conf["ncc_at_config_scale"]
    _conf_peak = derot_conf["estimated_peak_scale"]
    _conf_ok   = derot_conf["measured"]

    summary_lines: List[str] = [
        "=== Step 4 De-rotation Summary ===",
        "",
        f"  pole_pa          : {session_pole_pa:.2f}°",
        f"  warp_scale       : {warp_scale:.4f}  (config fixed)",
        (
            f"  derot_confidence : {_conf_ncc:.4f}  "
            f"(NCC@scale={warp_scale:.2f};  est. peak={_conf_peak:.3f})"
            if _conf_ok else
            f"  derot_confidence : unmeasured"
        ),
        f"  derot_flip       : {derot_flip}",
        f"  tracker_flip_ns  : {tracker_flip_ns}",
        f"  ncc_flip_false   : {_ncc_f:.4f}",
        f"  ncc_flip_true    : {_ncc_t:.4f}",
        "",
    ]

    session_log = {
        "pole_pa_deg":             session_pole_pa,
        "warp_scale":              warp_scale,
        "derot_ncc_confidence":    _conf_ncc,
        "derot_estimated_scale":   _conf_peak,
        "derot_confidence_valid":  _conf_ok,
        "derot_flip":              derot_flip,
        "tracker_flip_ns":         tracker_flip_ns,
        "ncc_flip_false":          _ncc_f,
        "ncc_flip_true":           _ncc_t,
    }
    if calib_result is not None:
        session_log["plate_scale_calibration"] = {
            "ps_fit":     calib_result["ps_fit"],
            "ps_nom":     calib_result["ps_nom"],
            "dps_pct":    calib_result["dps_pct"],
            "cx_offset":  calib_result["cx_offset"],
            "n_frames":   calib_result["n"],
            "rmse_nom":   calib_result["rmse_nom"],
            "rmse_fit":   calib_result["rmse_fit"],
        }

    n_windows = len(windows)
    for win_idx, window in enumerate(windows, start=1):
        if cancel_event is not None and cancel_event.is_set():
            print("  [CANCELLED] Stopping Step 4.", flush=True)
            break
        if progress_callback is not None:
            progress_callback(win_idx - 1, n_windows)

        t_center = window["center_time"]
        t_center_str = t_center.strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"\n  Window {win_idx}  [{t_center_str}]  "
              f"quality={window['window_quality']:.4f}  "
              f"rotation={window['rotation_degrees']:.1f}°")

        # ── NP.ang from Horizons (celestial North Pole angle) ─────────────────
        np_ang = query_horizons_np_ang(
            horizons_id=config.derotation.horizons_id,
            t_utc=t_center,
            observer_code=config.derotation.observer_code,
        )
        np_ang_val = np_ang if np_ang is not None else 0.0
        if np_ang is None:
            print("    [WARNING] NP.ang not available → using 0.0°")
        else:
            print(f"  [NP.ang = {np_ang_val:.3f}° (celestial)]")

        # pole_pa for the WARP: image-space angle from auto_detect_pole_pa()
        # pole_pa for the TRACKER: pole_pa + NP.ang = camera rotation θ_cam
        pole_pa_for_warp = session_pole_pa
        print(f"  [pole_pa = {pole_pa_for_warp:.1f}° (image-space, for warp)]")

        # Create per-window output directory
        win_out_dir: Optional[Path] = None
        if out_base is not None:
            win_out_dir = out_base / f"window_{win_idx:02d}"
            win_out_dir.mkdir(parents=True, exist_ok=True)

        # ── Satellite position prediction ──────────────────────────────────────
        sat_log: Dict = {}
        if tracker is not None:
            sat_log = {
                "np_ang_deg":       np_ang_val,
                "pole_pa_deg":      pole_pa_for_warp,
                "derot_flip":       derot_flip,
                "tracker_flip_ns":  tracker_flip_ns,
            }

        # ── De-rotate all filters ──────────────────────────────────────────────
        filter_results = derotation.derotate_window(
            window=window,
            required_filters=(
                list(window["per_filter"].keys())
                if config.camera_mode == "color"
                else config.filters
            ),
            period_hours=config.derotation.rotation_period_hours,
            warp_scale=warp_scale,
            align=True,
            normalize_brightness=config.derotation.normalize_brightness,
            min_quality_threshold=config.derotation.min_quality_threshold,
            pole_pa_deg=pole_pa_for_warp,
            color_mode=(config.camera_mode == "color"),
            flip_ns=derot_flip,
            out_dir=win_out_dir,
            weight_power=config.derotation.stack_weight_power,
        )

        # ── Satellite compositing (exp9 method) ───────────────────────────────
        if sat_cfg.composite_enabled and tracker is not None:
            print(f"  [satellite composite] Window {win_idx}…")
            disk_centers = _apply_satellite_composite(
                window=window,
                filter_results=filter_results,
                config=config,
                tracker=tracker,
                pole_pa_deg=pole_pa_for_warp,
                np_ang_deg=np_ang_val,
                r_ref=session_r_ref,
            )
            if disk_centers and sat_log:
                sat_log["disk_centers"] = disk_centers

        # ── Build log and save JSON ────────────────────────────────────────────
        log_dict = derotation.derotation_log_to_json(win_idx, window, filter_results)
        log_dict["session"] = session_log
        if sat_log:
            log_dict["satellite"] = sat_log
        if win_out_dir is not None:
            json_path = win_out_dir / "derotation_log.json"
            with open(json_path, "w") as f:
                json.dump(log_dict, f, indent=2, default=str)
            print(f"    → {json_path.name}")

        # ── Summary ───────────────────────────────────────────────────────────
        summary_lines.append(
            f"Window {win_idx}  {t_center_str}  "
            f"quality={window['window_quality']:.4f}  "
            f"rotation_span={window['rotation_degrees']:.1f}°"
        )
        summary_filters = (
            list(filter_results.keys()) if config.camera_mode == "color"
            else config.filters
        )
        for filt in summary_filters:
            if filt in filter_results:
                out_path, flog = filter_results[filt]
                n = flog.get("n_stacked", 0)
                snr = round(float(n) ** 0.5, 2)
                fname = out_path.name if out_path else "—"
                summary_lines.append(
                    f"  {filt:>4}: {fname}  ({n} frames, SNR×{snr:.2f})"
                )
            else:
                summary_lines.append(f"  {filt:>4}: not available")
        summary_lines.append("")

        outputs = {filt: res[0] for filt, res in filter_results.items()}
        logs    = {filt: res[1] for filt, res in filter_results.items()}
        all_results.append({
            "window_index": win_idx,
            "center_time":  t_center_str,
            "outputs":      outputs,
            "log":          logs,
            "satellite":    sat_log,
        })

    if progress_callback is not None:
        progress_callback(n_windows, n_windows)

    # ── Save summary ───────────────────────────────────────────────────────────
    summary_text = "\n".join(summary_lines)
    print()
    print(summary_text)
    if out_base is not None:
        txt_path = out_base / "derotation_summary.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(summary_text)
        print(f"  → {txt_path}")

    return {
        "windows":                  all_results,
        "derot_ncc_confidence":     _conf_ncc,
        "derot_confidence_measured": _conf_ok,
        "session_pole_pa":          session_pole_pa,
    }
