"""
Planetary de-rotation module.

Algorithm (per filter, per time window):
  1. Find planet disk center via ellipse fitting on each image (sub-pixel accurate)
  2. Compute CML displacement from the configured rotation period
  3. Apply spherical de-rotation warp using cv2.remap with Lanczos-4 interpolation.

     The warp direction is determined by the planet's north pole position angle
     (pole_pa_deg, queried from JPL Horizons quantity 17 "NP.ang"):

       Δx(x,y) = scale × Δλ_rad × depth(x,y) × cos(pole_pa_rad)
       Δy(x,y) = scale × Δλ_rad × depth(x,y) × sin(pole_pa_rad)
       depth(x,y) = sqrt(max(0, R² − (x−cx)² − (y−cy)²))

     pole_pa_deg = 0  →  equatorial view (drift purely horizontal, default for Jupiter)
     pole_pa_deg ≠ 0  →  tilted pole, e.g. Saturn at non-zero sub-Earth latitude

  4. Sub-pixel translate alignment via phase correlation (cv2.phaseCorrelate)
  5. Quality-weighted mean stack using Step 4 norm_scores as weights

warp_scale = 1.00 (empirically confirmed optimal for Jupiter via NCC sweep):
  Theoretical value is 1.0 (full spherical projection). NCC sweep across
  multiple datasets shows the peak consistently near 1.0; earlier interim
  values of 0.80 and 0.20 appearing in older comments/defaults are outdated.
  The NCC sweep is now used as a diagnostic confidence metric only — warp_scale
  is fixed at the config value (default 1.00) and not updated by the sweep.

Saturn notes:
  - Use rotation_period_hours=10.56 (System III)
  - pole_pa_deg is fetched from Horizons NP.ang; for low sub-Earth latitudes
    (<15°) the horizontal approximation (pole_pa=0) is acceptable.
  - Ring features do NOT co-rotate with the atmosphere; they will be slightly
    smeared in the stack (atmosphere is the primary target).
  - find_disk_center() isolates the disk from an attached ring before fitting
    (gated on the raw fit's aspect ratio — see its docstring), so semi_major/
    semi_minor/warp_radius describe the disk only, not the disk+ring blob.
  - CH4-band Saturn frames invert this: atmospheric methane absorption makes
    the globe darker than the icy rings, so no brightness-based disk/ring
    separation applies. find_disk_center() cannot locate the disk in CH4
    frames; callers should reuse geometry detected from a sibling filter
    (R/G/B/IR) in the same session instead — Saturn's apparent size/position
    barely changes across one filter cycle.

Comparison with WinJUPOS:
  - WinJUPOS: requires manual CML entry and frame selection
  - Our approach: fully automated (Step 4 quality scores drive frame selection)
  - Our approach: adds sub-pixel phase correlation alignment (WinJUPOS does not)
"""
from __future__ import annotations

import json
import math
import re
import urllib.parse
import urllib.request
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from pipeline.modules import image_io
from pipeline.modules.wavelet import sharpen as _wavelet_sharpen

# ── Constants ──────────────────────────────────────────────────────────────────

# Jupiter System II rotation period (9h 55m 41.0s)
SYSTEM_II_PERIOD_SEC: float = 9.0 * 3600 + 55 * 60 + 41.0  # 35741.0 s


# ── Bundled NP.ang lookup table ───────────────────────────────────────────────
#
# Pre-downloaded from JPL Horizons for Jupiter (599), Saturn (699), Mars (499),
# covering 2016-01-01 ~ 2036-12-31 at 1-day resolution (~473 KB JSON).
# Lookup uses linear interpolation with circular-angle wraparound handling.
# No internet access is required for these three bodies within the covered range.

_NP_ANG_TABLE_PATH = Path(__file__).parent.parent / "data" / "np_ang_table.json"

# Lazily loaded bundle: {horizons_id: {YYYY-MM-DD: float}}
_NP_ANG_BUNDLE: Optional[Dict[str, Dict[str, float]]] = None


def _load_bundle() -> Dict[str, Dict[str, float]]:
    global _NP_ANG_BUNDLE
    if _NP_ANG_BUNDLE is None:
        try:
            with open(_NP_ANG_TABLE_PATH, encoding="utf-8") as f:
                raw = json.load(f)
            _NP_ANG_BUNDLE = raw.get("planets", {})
        except Exception as exc:
            warnings.warn(f"[NP.ang] Could not load bundled table: {exc}")
            _NP_ANG_BUNDLE = {}
    return _NP_ANG_BUNDLE


def _interp_angle_deg(a0: float, a1: float, t: float) -> float:
    """Linearly interpolate between two angles in degrees, handling 360/0 wrap."""
    diff = (a1 - a0 + 540.0) % 360.0 - 180.0   # shortest arc in (-180, 180)
    return (a0 + t * diff) % 360.0


def query_horizons_np_ang(
    horizons_id: str,
    t_utc: datetime,
    observer_code: str = "500@399",
) -> Optional[float]:
    """Return the planet's north pole position angle (NP.ang) at *t_utc*.

    Lookup order:
      1. Bundled pre-downloaded table (Jupiter 599 / Saturn 699 / Mars 499,
         2016-01-01 ~ 2036-12-31) — no internet required.
      2. User-local cache (~/.astropipe/horizons_cache.json) populated by a
         previous successful online query.
      3. Live JPL Horizons query (requires internet).

    NP.ang (Horizons quantity 17): angle from celestial North to the body's
    north pole, measured eastward.  Used by :func:`spherical_derotation_warp`
    as ``pole_pa_deg``.  Returns None only if all sources fail.
    """
    # ── 1. Bundled table (primary, offline) ───────────────────────────────────
    bundle = _load_bundle()
    planet_table = bundle.get(horizons_id)
    if planet_table:
        d0 = t_utc.strftime("%Y-%m-%d")
        d1 = (t_utc + timedelta(days=1)).strftime("%Y-%m-%d")
        if d0 in planet_table:
            v0 = planet_table[d0]
            v1 = planet_table.get(d1, v0)
            frac = (t_utc.hour * 60 + t_utc.minute) / 1440.0
            result = _interp_angle_deg(v0, v1, frac)
            print(f"  [NP.ang] {d0} → {result:.3f}° (bundle, id={horizons_id})")
            return result
        # Date out of bundle range → fall through to live query

    # ── 2. User-local cache ────────────────────────────────────────────────────
    cache_path = Path.home() / ".astropipe" / "horizons_cache.json"
    cache: dict = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    date_str  = t_utc.strftime("%Y-%m-%d")
    cache_key = f"{horizons_id}:{date_str}"
    if cache_key in cache:
        val = cache[cache_key]
        print(f"  [NP.ang] {date_str} → {val:.3f}° (user cache, id={horizons_id})")
        return val

    # ── 3. Live Horizons query (fallback) ──────────────────────────────────────
    start = t_utc.strftime("%Y-%m-%d %H:%M")
    stop  = (t_utc + timedelta(minutes=2)).strftime("%Y-%m-%d %H:%M")
    params = urllib.parse.urlencode({
        "format": "text", "COMMAND": f"'{horizons_id}'",
        "OBJ_DATA": "NO", "MAKE_EPHEM": "YES", "EPHEM_TYPE": "OBSERVER",
        "CENTER": f"'{observer_code}'",
        "START_TIME": f"'{start}'", "STOP_TIME": f"'{stop}'",
        "STEP_SIZE": "1m", "QUANTITIES": "17",
    })
    url = f"https://ssd.jpl.nasa.gov/api/horizons.api?{params}"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            text = resp.read().decode("utf-8")
    except Exception as exc:
        warnings.warn(f"[NP.ang] Horizons query failed: {exc} → defaulting to 0.0°")
        return None

    soe, eoe = text.find("$$SOE"), text.find("$$EOE")
    if soe < 0 or eoe < 0:
        warnings.warn("[NP.ang] Horizons response missing $$SOE/$$EOE → 0.0°")
        return None
    data_lines = [l for l in text[soe + 5:eoe].split("\n") if l.strip()]
    if not data_lines:
        return None
    np_ang_col: Optional[int] = None
    for line in text[:soe].split("\n"):
        if "NP.ang" in line:
            np_ang_col = line.index("NP.ang"); break

    def _parse_line(dl: str, col: Optional[int]) -> Optional[float]:
        if col is not None:
            seg = dl[max(0, col - 4): col + 12]
            m = re.search(r"-?\d+\.?\d*", seg)
            if m: return float(m.group())
        m = re.search(r"(-?\d+\.\d+)", dl[25:])
        return float(m.group(1)) if m else None

    result = _parse_line(data_lines[0], np_ang_col)
    if result is None:
        warnings.warn("[NP.ang] Could not parse Horizons response → 0.0°")
        return None

    # Save to user cache for offline reuse
    try:
        cache[cache_key] = result
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except Exception:
        pass
    print(f"  [NP.ang] {date_str} → {result:.3f}° (Horizons live, id={horizons_id})")
    return result


# ── Color helpers ─────────────────────────────────────────────────────────────

def _to_luminance(image: np.ndarray) -> np.ndarray:
    """Return a (H, W) float32 luminance array from (H, W) or (H, W, 3) input.

    Uses ITU-R BT.709 coefficients for RGB→luminance conversion.
    If the input is already 2-D it is returned as-is (zero-copy).
    """
    if image.ndim == 2:
        return image
    return (
        0.2126 * image[:, :, 0]
        + 0.7152 * image[:, :, 1]
        + 0.0722 * image[:, :, 2]
    ).astype(np.float32)


# ── Pure-numpy image utilities (replaces scipy.ndimage) ───────────────────────

def _bilinear_interp(image: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Bilinear interpolation at arbitrary (ys, xs) coordinates; out-of-bounds → 0."""
    h, w = image.shape
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1, y1 = x0 + 1, y0 + 1
    fx = (xs - x0).astype(np.float64)
    fy = (ys - y0).astype(np.float64)

    def _safe(y, x):
        valid = (y >= 0) & (y < h) & (x >= 0) & (x < w)
        return np.where(valid, image[np.clip(y, 0, h-1), np.clip(x, 0, w-1)].astype(np.float64), 0.0)

    return ((1-fy)*(1-fx)*_safe(y0, x0) + (1-fy)*fx*_safe(y0, x1) +
               fy *(1-fx)*_safe(y1, x0) +    fy *fx*_safe(y1, x1))


def _gaussian_filter1d_np(x: np.ndarray, sigma: float) -> np.ndarray:
    """1D Gaussian smoothing via numpy convolution (replaces scipy.ndimage.gaussian_filter1d)."""
    radius = max(1, int(3.0 * sigma + 0.5))
    t = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (t / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(x.astype(np.float64), kernel, mode="same")


# ── Disk geometry ──────────────────────────────────────────────────────────────

def _gradient_disk_r(
    image: np.ndarray,
    cx: float,
    cy: float,
    r_rough: float,
    n_rays: int = 72,
    search_frac: tuple = (0.75, 1.30),
    n_samples: int = 100,
    smooth_sigma: float = 1.5,
    outlier_sigma: float = 2.0,
) -> float:
    """Estimate disk radius from steepest-gradient limb edge along radial rays.

    Replaces the Otsu-threshold disk_r (which underestimates the true limb by
    ~5 px due to limb darkening) with a gradient-profile measurement.  cx/cy
    are unchanged.  Falls back to r_rough if fewer than 8 valid rays are found.
    """
    h, w = image.shape[:2]
    angles  = np.linspace(0.0, 2.0 * np.pi, n_rays, endpoint=False)
    r_start = r_rough * search_frac[0]
    r_end   = r_rough * search_frac[1]
    r_vals  = np.linspace(r_start, r_end, n_samples)
    dr      = r_vals[1] - r_vals[0]

    edge_radii: list[float] = []
    for angle in angles:
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        xs = cx + r_vals * cos_a
        ys = cy + r_vals * sin_a
        if (xs < 0).any() or (xs >= w - 1).any() or \
           (ys < 0).any() or (ys >= h - 1).any():
            continue
        profile = _bilinear_interp(image, ys, xs)
        smoothed = _gaussian_filter1d_np(profile, sigma=smooth_sigma)
        grad = np.gradient(smoothed, dr)
        idx = int(np.argmin(grad))
        if idx == 0 or idx == len(grad) - 1:
            continue
        y0, y1, y2 = grad[idx - 1], grad[idx], grad[idx + 1]
        denom = 2.0 * (y2 - 2.0 * y1 + y0)
        sub = -(y2 - y0) / denom if abs(denom) > 1e-12 else 0.0
        edge_radii.append(float(r_vals[idx] + sub * dr))

    if len(edge_radii) < 8:
        return r_rough
    arr = np.array(edge_radii)
    med = float(np.median(arr))
    std = float(np.std(arr))
    keep = arr[np.abs(arr - med) < outlier_sigma * (std + 0.5)]
    if len(keep) < 6:
        keep = arr
    return float(np.median(keep))


def find_disk_center(
    image: np.ndarray,
    margin_factor: float = 0.10,
    fixed_threshold: int = 0,
    core_percentile: float = 60.0,
) -> Tuple[float, float, float, float, float]:
    """Locate planet disk via ellipse fitting.

    Args:
        image:           2-D float [0, 1] image.
        margin_factor:   Margin below Otsu threshold to include dim limb pixels.
                         Ignored when fixed_threshold > 0.
        fixed_threshold: Fixed brightness threshold (0–255). When > 0, skips
                         Otsu and uses this value directly — matches AS!4
                         _stabilization_planet_threshold=20 for consistent
                         disk detection across frames.
        core_percentile: Intensity percentile (within the loose blob) used to
                         isolate the brighter disk core from a fainter attached
                         ring — see "Disk-core isolation" below. Not meaningful
                         for filters where the disk is darker than the ring
                         (e.g. Saturn's CH4 band); such frames should get their
                         geometry from a sibling filter instead of this function.

    Returns:
        (cx, cy, semi_major, semi_minor, angle_deg) — ellipse parameters.
        Falls back to image centroid if ellipse fitting fails.
    """
    # Convert to uint8 for thresholding
    arr8 = np.clip(image * 255, 0, 255).astype(np.uint8)
    if fixed_threshold > 0:
        effective_thresh = int(fixed_threshold)
    else:
        thresh_val, _ = cv2.threshold(arr8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Apply threshold with small downward margin (include dim limb)
        effective_thresh = max(1, int(thresh_val * (1.0 - margin_factor)))
    _, binary = cv2.threshold(arr8, effective_thresh, 255, cv2.THRESH_BINARY)

    # Morphological closing to fill gaps in the disk
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        h, w = image.shape[:2]
        return float(w / 2), float(h / 2), float(min(h, w) / 4), float(min(h, w) / 4), 0.0

    # Use the largest contour
    largest = max(contours, key=cv2.contourArea)

    if len(largest) >= 5:
        # Fit ellipse (requires >= 5 points)
        (cx, cy), (ma, mi), angle = cv2.fitEllipse(largest)
        # OpenCV returns axes in (width, height) order along the rotated frame,
        # NOT guaranteed major > minor.  When angle ≈ 90° (nearly vertical ellipse,
        # e.g. Jupiter with pole_pa ≈ 8°) the first axis (ma) can be the shorter
        # polar axis and the second (mi) the longer equatorial axis.
        # Always return (semi_major, semi_minor) with semi_major >= semi_minor so
        # callers can rely on the 3rd return value being the larger (equatorial) axis.
        # When axes are swapped, rotate the returned angle by 90° so it always
        # describes the direction of the semi_major axis (in degrees, 0-180).
        if ma >= mi:
            semi_a = ma / 2
            semi_b = mi / 2
            angle_major = angle
        else:
            semi_a = mi / 2
            semi_b = ma / 2
            angle_major = (angle + 90.0) % 180.0

        # ── Disk-core isolation (ringed planets only) ─────────────────────────
        # A ringed planet's loose mask fuses the disk and its (fainter) rings
        # into one blob — Saturn's rings span ~2.2x the disk diameter — so the
        # raw fit above fits the disk+ring shape, not the disk: massively
        # overestimating the equatorial radius and understating the oblateness
        # (confirmed on real data: a naive fit gave semi_major=129px /
        # semi_minor=59px, aspect 0.46, on a frame whose actual disk was
        # ~65-70px radius). A real oblate planet's raw aspect never gets this
        # low (Jupiter ~0.94, Saturn's own disk ~0.90), so gate on aspect
        # first — this keeps Jupiter/Mars/Venus on the untouched original path
        # (any percentile re-threshold shrinks even a plain limb-darkened disk,
        # e.g. Jupiter's blob keeps only ~40% of its area at the 60th
        # percentile — that is NOT ring-stripping, just a smaller "hot core"
        # of the same disk, and would regress the non-ringed case if applied
        # unconditionally).
        raw_aspect = semi_b / semi_a if semi_a > 0 else 1.0
        if raw_aspect < 0.80:
            x, y, bw, bh = cv2.boundingRect(largest)
            roi_vals = arr8[y : y + bh, x : x + bw]
            roi_mask = binary[y : y + bh, x : x + bw] > 0
            vals = roi_vals[roi_mask]
            if vals.size > 0:
                core_thv = np.percentile(vals, core_percentile)
                core_bin = (((roi_vals >= core_thv) & roi_mask).astype(np.uint8)) * 255
                core_contours, _ = cv2.findContours(core_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if core_contours:
                    core_largest = max(core_contours, key=cv2.contourArea)
                    if len(core_largest) >= 5:
                        (cx, cy), (cma, cmi), cangle = cv2.fitEllipse(core_largest + np.array([x, y], dtype=core_largest.dtype))
                        if cma >= cmi:
                            semi_a, semi_b, angle_major = cma / 2, cmi / 2, cangle
                        else:
                            semi_a, semi_b, angle_major = cmi / 2, cma / 2, (cangle + 90.0) % 180.0

        # The core fit (when used) is itself a high-threshold underestimate of
        # the true visual disk, on top of the same limb-darkening bias that
        # motivates this refinement for the non-ringed case — so widen the
        # search window when we just isolated a ring-stripped core (raw_aspect
        # already told us which case we're in).
        search_frac = (0.6, 1.8) if raw_aspect < 0.80 else (0.75, 1.30)
        semi_a_refined = _gradient_disk_r(
            image, float(cx), float(cy), float(semi_a), search_frac=search_frac
        )
        # Preserve the fitted oblateness ratio while correcting the absolute
        # scale via the more robust gradient-edge search.
        semi_b_refined = semi_a_refined * (float(semi_b) / float(semi_a)) if semi_a > 0 else float(semi_b)
        return float(cx), float(cy), float(semi_a_refined), float(semi_b_refined), float(angle_major)
    else:
        # Fallback: centroid of bounding box
        x, y, w, h = cv2.boundingRect(largest)
        return float(x + w / 2), float(y + h / 2), float(max(w, h) / 2), float(min(w, h) / 2), 0.0


# ── Spherical de-rotation warp ────────────────────────────────────────────────

def spherical_derotation_warp(
    image: np.ndarray,
    dt_sec: float,
    cx: float,
    cy: float,
    disk_radius_px: float,
    period_hours: float = 9.9281,
    scale: float = 0.20,
    flip_direction: bool = False,
    pole_pa_deg: float = 0.0,
    polar_equatorial_ratio: float = 1.0,
) -> np.ndarray:
    """Apply spherical de-rotation warp to bring image to reference orientation.

    CML drift shifts features by an amount proportional to sphere depth:

        drift(x, y) = scale × Δλ_rad × depth(x, y)

    For an oblate spheroid (polar_equatorial_ratio < 1.0), the depth formula
    accounts for the different equatorial vs polar radii:

        depth² = R² − rx_eq² − (R/R_pole)² · ry_pol²
               = R² − rx_eq² − (1/polar_equatorial_ratio)² · ry_pol²

    where rx_eq and ry_pol are the equatorial and polar components of the
    offset from disk centre, projected using pole_pa_deg.

    For a perfect sphere (polar_equatorial_ratio=1.0) this reduces to the
    original formula: depth² = R² − rx² − ry².

    The drift direction is perpendicular to the planet's rotation axis as seen
    in the image, parameterised by *pole_pa_deg* (JPL Horizons "NP.ang"):

        Δx = drift × cos(pole_pa_rad)   [horizontal component]
        Δy = drift × sin(pole_pa_rad)   [vertical component]

    For equatorial views (Jupiter, pole_pa ≈ 0°) Δy ≈ 0 and the warp is
    purely horizontal, matching the original implementation.  For Saturn at
    non-zero sub-Earth latitude the warp is rotated accordingly.

    Args:
        image:                  2-D float [0, 1] array.
        dt_sec:                 (t_image - t_reference).total_seconds().
                                Positive = image taken AFTER reference.
        cx, cy:                 Disk center coordinates (pixels).
        disk_radius_px:         Disk semi-major axis (pixels), used as warp radius.
        period_hours:           Atmospheric rotation period in hours.
        scale:                  Empirical warp scale factor (0.20 from NCC sweep on Jupiter).
        flip_direction:         If True, negate the shift direction.
        pole_pa_deg:            North pole position angle in degrees (Horizons NP.ang).
                                0° = north up (equatorial view, horizontal drift only).
                                Positive = CCW from north (eastward tilt).
        polar_equatorial_ratio: polar_radius / equatorial_radius.
                                1.0 = perfect sphere (default).
                                ~0.935 = Jupiter (oblateness ≈ 6.5%).
                                Pass semi_minor / semi_major from find_disk_center().

    Returns:
        Warped float [0, 1] array, same shape as input.
    """
    period_sec = period_hours * 3600.0
    delta_lambda_rad = (dt_sec / period_sec) * 2.0 * np.pi

    h, w = image.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # Sphere/spheroid depth:
    # Use 5% padded radius so the sqrt singularity at r=R (slope → ∞) is
    # pushed outside the visible disk boundary, preventing limb distortion.
    warp_radius = disk_radius_px * 1.05
    rx = xx - cx
    ry = yy - cy

    # Decompose (rx, ry) into equatorial and polar frame using pole_pa_deg.
    # The equatorial drift direction is (cos(pa), sin(pa));
    # the polar axis is perpendicular: (-sin(pa), cos(pa)).
    # For Jupiter (pole_pa ≈ 0°): rx_eq ≈ rx, ry_pol ≈ ry.
    pole_pa_rad = np.radians(pole_pa_deg)
    cos_pa = float(np.cos(pole_pa_rad))
    sin_pa = float(np.sin(pole_pa_rad))
    rx_eq  = (rx * cos_pa + ry * sin_pa).astype(np.float32)   # equatorial
    ry_pol = (-rx * sin_pa + ry * cos_pa).astype(np.float32)  # polar

    # Oblate-spheroid depth formula:
    #   depth² = R² − rx_eq² − (R/R_pole)² · ry_pol²
    # For ratio=1 (sphere): depth² = R² − rx² − ry²  (identical to original)
    _polar_scale_sq = (1.0 / max(polar_equatorial_ratio, 1e-3)) ** 2
    depth_sq = warp_radius ** 2 - rx_eq ** 2 - _polar_scale_sq * ry_pol ** 2
    depth_map = np.where(depth_sq > 0.0, np.sqrt(depth_sq.clip(0)), 0.0).astype(np.float32)

    sign = -1.0 if flip_direction else 1.0
    drift = (sign * scale * delta_lambda_rad * depth_map).astype(np.float32)

    # Decompose drift into image-plane x/y using the pole position angle.
    # pole_pa = 0°  → cos=1, sin=0  → pure horizontal (Jupiter default)
    # pole_pa = 90° → cos=0, sin=1  → pure vertical
    # (cos_pa / sin_pa already computed above for the depth decomposition)
    map_x = (xx - drift * cos_pa).astype(np.float32)
    map_y = (yy - drift * sin_pa).astype(np.float32)

    # Mixed interpolation: INTER_CUBIC interior, INTER_LINEAR near the limb.
    #
    # INTER_CUBIC (and Lanczos-4) have negative side-lobe ringing (Gibbs effect)
    # at sharp high-contrast boundaries (bright disk vs. black background).
    # That ringing gets further amplified by wavelet sharpening.
    # Solution: run two remaps with the same map_x/map_y but different
    # interpolation kernels, then blend spatially:
    #   weight = 1.0  → pure CUBIC  (disk interior, detail-preserving)
    #   weight = 0.0  → pure LINEAR (disk edge and exterior, ringing-free)
    # The blend weight transitions smoothly over `_interp_feather_px` pixels
    # inside the disk edge, so no sharp pixel-domain boundary is introduced.
    src_f32 = image.astype(np.float32)
    warped_cubic = cv2.remap(
        src_f32, map_x, map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    )
    warped_linear = cv2.remap(
        src_f32, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    )

    # Weight map: distance from disk center in the output image.
    # The disk center doesn't shift during rotation, so cx/cy/disk_radius_px
    # remain valid for the output frame.
    _interp_feather_px = 12.0
    dist_from_center = np.sqrt(rx ** 2 + ry ** 2).astype(np.float32)
    w_cubic = np.clip(
        (disk_radius_px - dist_from_center) / _interp_feather_px, 0.0, 1.0
    )
    # For 3-channel (H, W, 3) images, expand weight map so broadcasting works.
    if warped_cubic.ndim == 3:
        w_cubic = w_cubic[:, :, np.newaxis]
    warped = warped_cubic * w_cubic + warped_linear * (1.0 - w_cubic)

    return np.clip(warped, 0.0, 1.0)


# ── Pole PA auto-detection ────────────────────────────────────────────────────

def auto_detect_pole_pa(
    frames: List[np.ndarray],
    cx: float,
    cy: float,
    disk_radius_px: float,
) -> float:
    """Auto-detect the image-space pole position angle from belt gradient direction.

    Builds a gradient-angle histogram over the inner disk (0.75R) using two
    complementary wavelet-sharpened views of the frames, then returns the angle
    of the dominant gradient direction converted to pole_pa convention.

    Jupiter's belts run perpendicular to the rotation axis, so the dominant
    gradient direction inside the disk directly encodes the pole position angle:

        pole_pa = 0°   → belts horizontal, North straight up
        pole_pa = +θ   → belts tilted θ CCW from horizontal (camera rotated CW)
        pole_pa = −θ   → belts tilted θ CW  from horizontal (camera rotated CCW)

    No warp, no time information, and no rotation period are required — the
    belt orientation is a static geometric property of each frame.

    Args:
        frames:        List of 2-D float [0,1] images (luminance).
        cx, cy:        Disk centre (pixels).
        disk_radius_px: Disk semi-major radius (pixels).

    Returns:
        pole_pa_deg in (-90°, +90°].
    """
    h, w = frames[0].shape[:2]
    Y, X = np.ogrid[:h, :w]
    disk_mask = ((X - cx) ** 2 + (Y - cy) ** 2) < (0.75 * disk_radius_px) ** 2

    # Gradient-angle histogram: find dominant gradient direction in the disk.
    # Jupiter's belts are perpendicular to the rotation axis, so the dominant
    # gradient direction inside the disk = pole axis direction.
    #
    # Critical: wavelet pre-sharpening must isolate belt-boundary gradients
    # and suppress smooth limb-darkening radial gradients.  Two complementary
    # scales are combined to stabilise the estimate:
    #   fine  [200,200,200,0,0,0] → 2-8 px belt-edge detail (sharp, responsive)
    #   belt  [0,0,200,200,200,0] → 8-32 px belt-width band (robust, anti-noise)
    # Including coarser scales (level 4+) amplifies the radial limb gradient and
    # biases the histogram by ~10° toward steeper angles — omit them here.
    _HIST_AMOUNTS = [
        [200.0, 200.0, 200.0, 0.0, 0.0, 0.0],  # fine-scale: belt edge detail
        [0.0, 0.0, 200.0, 200.0, 200.0, 0.0],  # belt-scale: mid-width structure
    ]
    N_BINS = 360  # 0.5° resolution after folding to [0°, 180°)
    estimates: List[float] = []

    for amounts in _HIST_AMOUNTS:
        sharp_frames = [_wavelet_sharpen(f, levels=6, amounts=amounts) for f in frames]
        accum = np.zeros(N_BINS // 2, dtype=np.float64)
        for sf in sharp_frames:
            gx = cv2.Sobel(sf, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(sf, cv2.CV_32F, 0, 1, ksize=3)
            mag = np.hypot(gx, gy)
            thresh = float(np.percentile(mag[disk_mask], 70))
            strong = disk_mask & (mag > thresh)
            # Fold gradient angles to [0°, 180°): dark→bright and bright→dark
            # across a belt represent the same physical orientation.
            angles_deg = np.degrees(np.arctan2(gy[strong], gx[strong])) % 180.0
            bin_idx = np.clip(
                (angles_deg / 180.0 * (N_BINS // 2)).astype(int), 0, N_BINS // 2 - 1
            )
            np.add.at(accum, bin_idx, mag[strong])

        sigma_bins = max(1, int(round(4.0 / (180.0 / (N_BINS // 2)))))
        kernel_size = 2 * (3 * sigma_bins) + 1
        accum_smooth = cv2.GaussianBlur(
            accum.astype(np.float32).reshape(1, -1), (kernel_size, 1), sigma_bins
        ).reshape(-1)
        best_bin = int(np.argmax(accum_smooth))
        # Bin centre → pole_pa: shift by -90 so 0° = vertical (North straight up)
        estimates.append(best_bin * (180.0 / (N_BINS // 2)) - 90.0)

    return float(np.mean(estimates))


def auto_detect_ns_flip(
    frames: List[np.ndarray],
    dt_sec_list: List[float],
    cx: float,
    cy: float,
    disk_radius_px: float,
    period_hours: float,
    warp_scale: float = 0.80,
    pole_pa_deg: float = 0.0,
    polar_equatorial_ratio: float = 1.0,
) -> Tuple[bool, float, float]:
    """Detect the de-rotation warp direction from atmospheric feature drift.

    Tests two forward-prediction directions (flip_direction=False / True) and
    picks whichever better matches f_late from f_early inside the inner disk.

    IMPORTANT — what this function detects and what it does NOT detect:
      • It determines which flip_direction to pass to spherical_derotation_warp.
        For BOTH N-up AND pure NS-flip (South-up, East-left) cameras the
        atmospheric features drift in the same image-plane direction (leftward),
        so this function returns flip_ns=False in both cases.
        flip_direction=False is the *correct* de-rotation direction for both.
      • It does NOT determine the satellite-tracker orientation (flip_ns for the
        tracker), because the tracker's NS-sign is purely about whether the camera
        image has north at the top or bottom — this is camera-orientation knowledge
        that cannot be inferred from feature drift alone for a pure NS-flip setup.
        Use sat_cfg.flip_ns (SatelliteConfig) to tell the tracker the actual
        camera orientation independently.

    Args:
        frames:                 List of float [0,1] images (2-D or 3-D).
        dt_sec_list:            Time offset (s) of each frame from a common reference.
        cx, cy:                 Disk centre (pixels).
        disk_radius_px:         Disk semi-major radius (pixels).
        period_hours:           Atmospheric rotation period (hours).
        warp_scale:             Empirical spherical warp scale (default 0.80).
        pole_pa_deg:            Image-space pole PA from auto_detect_pole_pa().
        polar_equatorial_ratio: semi_minor / semi_major from find_disk_center().

    Returns:
        (derot_flip, score_flip_false, score_flip_true)
        derot_flip=True  → use flip_direction=True in spherical_derotation_warp.
        derot_flip=False → use flip_direction=False (default, correct for most setups).
        Defaults to (False, 0.0, 0.0) when ambiguous (|Δcorr| < 0.001).
    """
    if len(frames) < 2:
        print("  [derot_flip] fewer than 2 frames — defaulting to flip_direction=False")
        return False, 0.0, 0.0

    dts = np.array(dt_sec_list, dtype=np.float64)
    i_min = int(np.argmin(dts))
    i_max = int(np.argmax(dts))
    if i_min == i_max:
        print("  [derot_flip] all frames at same time — defaulting to flip_direction=False")
        return False, 0.0, 0.0

    dt_span = float(dts[i_max] - dts[i_min])   # always positive

    def _lum(f: np.ndarray) -> np.ndarray:
        if f.ndim == 3:
            return (0.2126 * f[:, :, 0] + 0.7152 * f[:, :, 1] + 0.0722 * f[:, :, 2]).astype(np.float32)
        return f.astype(np.float32)

    _DRIFT_SHARPEN = [200, 200, 200, 0, 0, 0]

    f_early = _wavelet_sharpen(_lum(frames[i_min]), amounts=_DRIFT_SHARPEN)
    f_late  = _wavelet_sharpen(_lum(frames[i_max]), amounts=_DRIFT_SHARPEN)
    h, w = f_early.shape
    Y, X = np.ogrid[:h, :w]
    disk_mask = ((X - cx) ** 2 + (Y - cy) ** 2) < (disk_radius_px * 0.75) ** 2

    ref_pixels = f_late[disk_mask].astype(np.float64)
    ref_std = float(ref_pixels.std())

    # Forward-prediction correlation test.
    # In astronomical images East is LEFT, so prograde features drift LEFTWARD (N-up).
    #
    #   flip_direction=False → map_x = xx − drift  (content shifts RIGHTWARD in output)
    #                          Matches f_late when features actually move RIGHT → S-up
    #   flip_direction=True  → map_x = xx + drift  (content shifts LEFTWARD in output)
    #                          Matches f_late when features actually move LEFT  → N-up
    #
    # Note: this is the OPPOSITE of the de-rotation convention (where flip=False = N-up),
    # because de-rotation undoes the motion while forward-prediction replicates it.
    scores: dict = {}
    for flip_direction in (False, True):
        predicted = spherical_derotation_warp(
            f_early, +dt_span,
            cx, cy, disk_radius_px,
            period_hours=period_hours,
            scale=warp_scale,
            flip_direction=flip_direction,
            pole_pa_deg=pole_pa_deg,
            polar_equatorial_ratio=polar_equatorial_ratio,
        )
        tgt_pixels = predicted[disk_mask].astype(np.float64)
        tgt_std = float(tgt_pixels.std())
        if ref_std > 1e-6 and tgt_std > 1e-6:
            scores[flip_direction] = float(np.corrcoef(ref_pixels, tgt_pixels)[0, 1])
        else:
            scores[flip_direction] = 0.0

    # scores[True] > scores[False] → leftward drift (N-up / S-up East-left) → derot_flip=False
    # scores[False] > scores[True] → rightward drift (180°-rotated, no prism) → derot_flip=True
    delta = scores[False] - scores[True]
    derot_flip = delta > 0.0
    confidence = abs(delta)
    angle_deg = abs(dt_span) / (period_hours * 3600.0) * 360.0
    print(
        f"  [derot_flip] Δt={dt_span:.0f}s ({angle_deg:.2f}°)  "
        f"flip_False_corr={scores[False]:.5f}  flip_True_corr={scores[True]:.5f}  "
        f"→ derot_flip={derot_flip} (|Δcorr|={confidence:.5f})"
    )
    if confidence < 0.001:
        print("  [derot_flip] low confidence — defaulting to derot_flip=False")
        return False, float(scores[False]), float(scores[True])
    return derot_flip, float(scores[False]), float(scores[True])


def pole_pa_from_disk_ellipse(image: np.ndarray) -> Optional[float]:
    """Estimate pole PA from disk ellipse major-axis direction.

    For oblate planets the equatorial (major) axis = drift direction = pole_pa
    in the warp convention (pole_pa=0° → horizontal drift → North straight up).

    Works for any planet with detectable oblateness (Jupiter f≈6.5%, Saturn
    disk f≈9.8%, Uranus f≈2.3%).  Not reliable for nearly-spherical bodies
    (Mars f≈0.6%) or when the disk is too small.

    Args:
        image: 2-D or 3-D float image (luminance extracted automatically).

    Returns:
        pole_pa_deg in (-90°, +90°], or None if the disk is undetectable or
        too circular (semi_b/semi_a > 0.995, oblateness < 0.5%).
    """
    lum = _to_luminance(image) if image.ndim == 3 else image.astype(np.float32)
    _, _, semi_a, semi_b, angle_major = find_disk_center(lum)
    if semi_a < 5:
        return None
    if semi_b / max(semi_a, 1.0) > 0.995:
        return None
    # angle_major ∈ [0°, 180°) — major (equatorial) axis from horizontal (OpenCV).
    # Map to warp convention (-90°, +90°]: fold values > 90° back to negative side.
    pole_pa = float(angle_major) if float(angle_major) <= 90.0 else float(angle_major) - 180.0
    return pole_pa


# ── Sub-pixel alignment ────────────────────────────────────────────────────────

def apply_shift(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Translate *image* by (dx, dy) pixels using Bicubic interpolation.

    Uses INTER_CUBIC (not LANCZOS4) to avoid Gibbs ringing at the limb boundary,
    and BORDER_REPLICATE (not REFLECT_101) to prevent black-value intrusion at
    the image edge when frames are shifted during sub-pixel alignment.
    """
    h, w = image.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(
        image.astype(np.float32),
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return np.clip(shifted, 0.0, 1.0)


def subpixel_align(
    reference: np.ndarray,
    target: np.ndarray,
) -> Tuple[float, float]:
    """Compute sub-pixel translation shift from target to reference.

    Uses phase correlation (cv2.phaseCorrelate), accurate to ~0.1 pixel.

    Args:
        reference: 2-D float [0, 1] reference image.
        target:    2-D float [0, 1] image to align to reference.

    Returns:
        (dx, dy) — shift to apply to target (via apply_shift(target, dx, dy))
        to align it with reference.
    """
    # cv2.phaseCorrelate(reference, target) returns the forward content shift
    # of target relative to reference (i.e. target ≈ reference shifted by
    # (raw_dx, raw_dy)) — NOT the correction needed to undo that shift.
    # apply_shift()/cv2.warpAffine with M=[[1,0,dx],[0,1,dy]] moves image
    # content by (+dx,+dy), so passing the raw phaseCorrelate output straight
    # into apply_shift(target, ...) pushes target further from reference
    # instead of correcting it. Verified with a synthetic known-shift
    # round-trip test (tests/test_subpixel_align.py): un-negated MSE was
    # worse than doing no alignment at all; negated MSE recovered the
    # reference closely. Negate here so every caller's
    # apply_shift(target, dx, dy) pattern is correct.
    ref_f32 = reference.astype(np.float32)
    tgt_f32 = target.astype(np.float32)
    (raw_dx, raw_dy), _ = cv2.phaseCorrelate(ref_f32, tgt_f32)
    return -float(raw_dx), -float(raw_dy)


def limb_center_align(
    ref_cx: float,
    ref_cy: float,
    target_lum: np.ndarray,
    max_shift_px: float = 15.0,
    fixed_threshold: int = 0,
) -> Tuple[float, float]:
    """Compute sub-pixel translation shift using disk limb center alignment.

    Directly measures where the planet disk center is in *target_lum* via
    ellipse fitting and returns the shift needed to move it to the reference
    center (ref_cx, ref_cy).

    This is more robust than phaseCorrelate at the limb:
    - phaseCorrelate correlates the entire frame including noisy background;
      a 0.5 px background-noise bias shifts ALL frames together, causing limb
      smearing that wavelet amplifies into ringing.
    - Limb-center alignment directly measures the disk edge position and
      corrects only the whole-disk translation, leaving interior features
      untouched.

    Falls back to (0, 0) if ellipse fitting fails or produces an implausibly
    large shift (> max_shift_px), which signals a detection failure.

    Args:
        ref_cx, ref_cy:  Reference frame disk center (pixels).
        target_lum:      2-D float [0, 1] luminance of the warped frame.
        max_shift_px:    Clamp: shifts larger than this are treated as
                         detection failures and (0, 0) is returned instead.

    Returns:
        (dx, dy) — shift to apply to the warped frame so its disk center
        aligns with (ref_cx, ref_cy).
    """
    try:
        cx, cy, semi_a, *_ = find_disk_center(target_lum, fixed_threshold=fixed_threshold)
        if semi_a < 5:
            return 0.0, 0.0
        dx = float(ref_cx - cx)
        dy = float(ref_cy - cy)
        if abs(dx) > max_shift_px or abs(dy) > max_shift_px:
            # Likely detection failure; don't apply a wild shift
            return 0.0, 0.0
        return dx, dy
    except Exception:
        return 0.0, 0.0


# ── Visual limb radius detection ──────────────────────────────────────────────

def find_visual_limb_radius(
    image: np.ndarray,
    cx: float,
    cy: float,
    radius_estimate: float,
    n_angles: int = 36,
    threshold_frac: float = 0.05,
    search_margin: int = 30,
) -> float:
    """Find the actual visual limb radius by scanning radial brightness profiles.

    ``find_disk_center()`` returns the Otsu-threshold radius, which sits at the
    ~50% brightness point and can be 10-20 px inside the actual visible disk
    boundary.  This function scans outward from that estimate and returns the
    radius where brightness drops below *threshold_frac* × image peak — the
    true visual edge.

    Args:
        image:            2-D or 3-D float image.
        cx, cy:           Disk centre (from find_disk_center).
        radius_estimate:  Otsu radius to start scanning from.
        n_angles:         Number of equally-spaced radial directions to sample.
        threshold_frac:   Intensity fraction below which a pixel is considered
                          background (default 0.05 = 5 % of peak).
        search_margin:    How many pixels beyond radius_estimate to scan.

    Returns:
        Median of per-angle visual-edge detections (pixels).  Falls back to
        ``radius_estimate`` if detection fails.
    """
    lum = image.mean(axis=2).astype(np.float32) if image.ndim == 3 else image.astype(np.float32)
    h, w = lum.shape
    peak = float(lum.max())
    if peak < 1e-6:
        return radius_estimate
    threshold = peak * threshold_frac

    radii: List[float] = []
    for angle in np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False):
        cos_a = float(np.cos(angle))
        sin_a = float(np.sin(angle))
        found = False
        # Start a few pixels inside the Otsu estimate so we always cross the edge
        for dr in range(-5, search_margin + 1):
            r = radius_estimate + dr
            xi = int(round(cx + r * cos_a))
            yi = int(round(cy + r * sin_a))
            if 0 <= xi < w and 0 <= yi < h:
                if lum[yi, xi] < threshold:
                    radii.append(max(r - 1.0, radius_estimate))
                    found = True
                    break
        if not found:
            radii.append(radius_estimate + search_margin)

    return float(np.median(radii)) if radii else radius_estimate


# ── Disk edge feathering mask ─────────────────────────────────────────────────

def make_disk_feather_mask(
    shape: Tuple[int, int],
    cx: float,
    cy: float,
    radius: float,
    feather_px: float = 8.0,
) -> np.ndarray:
    """Create a soft disk mask that fades to 0 at the limb edge.

    The mask is 1.0 inside (radius - feather_px) and smoothly fades to 0.0
    at the geometric disk edge (radius).  Applied to each warped frame before
    stacking to prevent background zeros from bleeding into the limb average,
    which would create a darkening band that wavelet sharpening amplifies.

    Args:
        shape:      (H, W) of the image.
        cx, cy:     Disk center (pixels).
        radius:     Disk semi-major axis (pixels, from find_disk_center).
        feather_px: Width of the fade-out transition in pixels.

    Returns:
        2-D float32 [0, 1] mask of shape (H, W).
    """
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    # Linearly ramp from 1 (interior) to 0 (exterior) over feather_px pixels
    mask = np.clip((radius - dist) / feather_px, 0.0, 1.0)
    return mask.astype(np.float32)


# ── Stacking ───────────────────────────────────────────────────────────────────

def _planet_median(image: np.ndarray) -> float:
    """Return median of planet-disk pixels (central 50% of image area)."""
    lum = _to_luminance(image)    # use luminance so this works for color images
    h, w = lum.shape[:2]
    cy, cx = h // 2, w // 2
    r = int(min(h, w) * 0.25)
    roi = lum[cy - r : cy + r, cx - r : cx + r]
    fg = roi[roi > roi.mean() * 0.3]
    return float(np.median(fg)) if fg.size else float(np.median(roi))


def normalize_brightness_to_reference(
    images: List[np.ndarray],
    reference_idx: int = 0,
) -> List[np.ndarray]:
    """Scale each image so its planet-disk median matches the reference frame.

    Multiplies each image by (ref_median / frame_median).  Clips to [0, 1].
    Does NOT alter the reference frame itself.

    Args:
        images:        List of 2-D float [0, 1] arrays.
        reference_idx: Index of the frame to treat as brightness reference.
    """
    ref_med = _planet_median(images[reference_idx])
    if ref_med < 1e-6:
        return images  # degenerate frame — skip normalization
    normalized = []
    for i, img in enumerate(images):
        if i == reference_idx:
            normalized.append(img)
        else:
            frame_med = _planet_median(img)
            scale = ref_med / frame_med if frame_med > 1e-6 else 1.0
            normalized.append(np.clip(img.astype(np.float64) * scale, 0.0, 1.0).astype(np.float32))
    return normalized


def quality_weighted_stack(
    images: List[np.ndarray],
    weights: List[float],
    weight_power: float = 1.0,
) -> np.ndarray:
    """Stack images with quality weights (weighted mean).

    Args:
        images:  List of 2-D float [0, 1] arrays (all same shape).
        weights: Quality score per image (norm_score from Step 4).
                 Does not need to sum to 1 — normalised internally.
        weight_power: Exponent applied to weights before normalising
                 (weights = weights ** weight_power). 1.0 (default) is the
                 original linear behaviour. Raising it sharpens the blend
                 toward the best-scoring frame(s) at the cost of some SNR —
                 useful when per-frame quality varies a lot within one
                 window (confirmed on real Saturn data: G/B channel
                 norm_score spread 0.07-0.94 within a single window, whose
                 linear blend measurably softened fine detail relative to
                 weight_power=2-4; see session notes 2026-08-09). Default
                 1.0 preserves the exact original stack for every caller
                 that doesn't pass this explicitly.

    Returns:
        Weighted mean stack, float [0, 1].
    """
    if len(images) == 1:
        return images[0].copy()

    w_arr = np.array(weights, dtype=np.float64)
    w_arr = np.clip(w_arr, 1e-9, None)
    if weight_power != 1.0:
        w_arr = w_arr ** weight_power
    w_arr /= w_arr.sum()

    stack = np.zeros_like(images[0], dtype=np.float64)
    for img, w in zip(images, w_arr):
        stack += w * img.astype(np.float64)
    return np.clip(stack, 0.0, 1.0).astype(np.float32)


# ── Per-filter de-rotation pipeline ───────────────────────────────────────────

def derotate_filter(
    included_rows: List[dict],
    t_reference: datetime,
    period_hours: float = 9.9281,
    warp_scale: float = 1.00,
    align: bool = True,
    normalize_brightness: bool = False,
    min_quality_threshold: float = 0.0,
    pole_pa_deg: float = 0.0,
    color_mode: bool = False,
    flip_direction: bool = False,
    weight_power: float = 1.0,
) -> Tuple[np.ndarray, dict]:
    """De-rotate and stack a single filter's images for one time window.

    Uses spherical de-rotation warp (Δx ∝ depth) — NOT image rotation.

    Args:
        included_rows: List of score dicts from Step 4 (must have 'path',
                       'timestamp', 'norm_score').
        t_reference:   Window center time (reference orientation).
        period_hours:  System II period.
        warp_scale:    Spherical warp empirical scale factor (default 1.00).
        align:          If True, apply sub-pixel phase correlation alignment
                        after warp. Disable for speed testing.
        color_mode:     If True, preserve RGB channels throughout; disk detection
                        and alignment are computed on the luminance channel.
        flip_direction: If True, negate the warp drift direction (South-up camera).
                        Must match the flip_ns detected by auto_detect_ns_flip().
        weight_power:   Exponent applied to norm_score weights before
                        stacking (see quality_weighted_stack). 1.0 (default)
                        = unchanged linear blend.

    Returns:
        (stacked_image, log_dict)
        stacked_image: float [0, 1] 2-D array (mono) or (H, W, 3) (color)
        log_dict:      per-frame details for JSON logging
    """
    if not included_rows:
        raise ValueError("No images to de-rotate")

    # Quality threshold filtering
    if min_quality_threshold > 0.0:
        filtered = [r for r in included_rows if float(r["norm_score"]) >= min_quality_threshold]
        n_dropped = len(included_rows) - len(filtered)
        if n_dropped:
            print(f" [{n_dropped} frame(s) dropped by quality threshold]", end="")
        included_rows = filtered if filtered else included_rows  # 전부 탈락하면 그냥 진행

    # Sort by timestamp proximity to t_reference; first = reference frame
    sorted_rows = sorted(
        included_rows,
        key=lambda r: abs((r["timestamp"] - t_reference).total_seconds()),
    )
    reference_row = sorted_rows[0]

    # ── Shared disk centre (detect once from the reference frame) ─────────────
    # Per-frame Otsu detection gives (cx, cy) that differ by a few pixels
    # between frames → each frame gets a slightly different spherical warp →
    # misaligned limbs when stacked → wavelet amplifies the boundary mismatch
    # → asymmetric limb artifact (thin left limb, thick right limb).
    # Fix: detect the disk centre once from the reference frame and use the
    # same (ref_cx, ref_cy, ref_semi_a) for every frame in the window.
    _ref_raw = image_io.read_tif(reference_row["path"])
    if color_mode:
        if _ref_raw.ndim == 2:
            _ref_raw = np.stack([_ref_raw] * 3, axis=2)
        _ref_lum = _to_luminance(_ref_raw)
    else:
        _ref_lum = _ref_raw if _ref_raw.ndim == 2 else _ref_raw.mean(axis=2).astype(np.float32)
    ref_cx, ref_cy, ref_semi_a, ref_semi_b, _ = find_disk_center(_ref_lum)
    # Measured polar/equatorial ratio from the reference frame ellipse fit.
    # For Jupiter ~0.935; clamped to [0.85, 1.0] to guard against fitting errors.
    _polar_eq_ratio = float(np.clip(ref_semi_b / max(ref_semi_a, 1.0), 0.85, 1.0))

    warped_images: List[np.ndarray] = []
    weights: List[float] = []
    log_frames: List[dict] = []
    # Pre-warp disk center shifts: measured from raw frame before any warp is
    # applied, so the shift is purely seeing-induced wobble and is not
    # contaminated by the warp-induced brightness redistribution that biases
    # post-warp limb_center_align toward partially undoing the de-rotation.
    _pre_warp_shifts: dict = {}   # stem → (dx, dy)

    ref_img: Optional[np.ndarray] = None

    for row in included_rows:
        img = image_io.read_tif(row["path"])

        if color_mode:
            # Keep (H, W, 3); use luminance only for geometry/alignment
            if img.ndim == 2:
                # Unexpected mono TIF in color mode — replicate to 3 channels
                img = np.stack([img] * 3, axis=2)
        else:
            # Mono mode: flatten to 2-D
            if img.ndim == 3:
                img = img.mean(axis=2).astype(np.float32)

        # Measure disk center from the raw (pre-warp) frame for later alignment.
        _raw_lum = _to_luminance(img)
        try:
            _cx_i, _cy_i, _semi_i, *_ = find_disk_center(_raw_lum)
            if _semi_i >= 5:
                _dx = float(ref_cx - _cx_i)
                _dy = float(ref_cy - _cy_i)
                if abs(_dx) <= 15.0 and abs(_dy) <= 15.0:
                    _pre_warp_shifts[row["stem"]] = (_dx, _dy)
        except Exception:
            pass

        dt_sec = (row["timestamp"] - t_reference).total_seconds()
        warped = spherical_derotation_warp(
            img, dt_sec, ref_cx, ref_cy, ref_semi_a,
            period_hours=period_hours,
            scale=warp_scale,
            flip_direction=flip_direction,
            pole_pa_deg=pole_pa_deg,
            polar_equatorial_ratio=_polar_eq_ratio,
        )

        log_frames.append({
            "stem":              row["stem"],
            "timestamp":         row["timestamp"].strftime("%Y-%m-%dT%H:%M:%SZ"),
            "norm_score":        round(float(row["norm_score"]), 4),
            "dt_sec":            round(dt_sec, 2),
            "disk_center_px":    [round(ref_cx, 2), round(ref_cy, 2)],
            "disk_radius_px":    round(ref_semi_a, 2),
            "delta_lambda_deg":  round((dt_sec / (period_hours * 3600.0)) * 360.0, 4),
        })

        if row["stem"] == reference_row["stem"]:
            ref_img = warped

        warped_images.append(warped)
        weights.append(float(row["norm_score"]))

    # ── Per-frame brightness normalization ───────────────────────────────────
    if normalize_brightness and len(warped_images) > 1:
        ref_idx = next(
            i for i, fl in enumerate(log_frames)
            if fl["stem"] == reference_row["stem"]
        )
        warped_images = normalize_brightness_to_reference(warped_images, ref_idx)

    # ── Sub-pixel translation alignment (pre-warp center based) ─────────────
    # Use disk centers measured from raw (pre-warp) frames.
    # Post-warp limb_center_align is biased: the warp redistributes atmospheric
    # brightness (belts/zones move), causing find_disk_center to shift in the
    # same direction as the warp. Correcting that shift partially undoes the
    # de-rotation (empirically: align_shift ∝ warp_scale × dt, cancelling
    # ~40% of the warp). Pre-warp measurements capture only genuine
    # seeing-induced disk wobble and are not contaminated by the warp.
    # Fallback chain: pre_warp_center → limb_center (post-warp, if pre-warp
    # detection failed) → phase_correlate (if limb_center also returns zero).
    if align and ref_img is not None and len(warped_images) > 1:
        aligned_images: List[np.ndarray] = []
        ref_lum = _to_luminance(ref_img)
        for img, frame_log in zip(warped_images, log_frames):
            if frame_log["stem"] == reference_row["stem"]:
                aligned_images.append(img)
                frame_log["align_shift_px"] = [0.0, 0.0]
                frame_log["align_method"] = "reference"
            else:
                stem = frame_log["stem"]
                if stem in _pre_warp_shifts:
                    dx, dy = _pre_warp_shifts[stem]
                    method = "pre_warp_center"
                else:
                    # Pre-warp detection failed — fall back to post-warp limb center
                    img_lum = _to_luminance(img)
                    dx, dy = limb_center_align(ref_cx, ref_cy, img_lum)
                    method = "limb_center"
                    if dx == 0.0 and dy == 0.0:
                        dx, dy = subpixel_align(ref_lum, img_lum)
                        method = "phase_correlate"
                aligned = apply_shift(img, dx, dy)
                aligned_images.append(aligned)
                frame_log["align_shift_px"] = [round(dx, 3), round(dy, 3)]
                frame_log["align_method"] = method
        warped_images = aligned_images

        # ── Outlier-shift rejection ──────────────────────────────────────
        # A frame whose alignment correction is drastically larger than its
        # window-mates' usually means pre-warp registration failed silently
        # and the limb_center/phase_correlate fallback landed on an
        # imprecise large correction (a known CH4 failure mode: brightness
        # inversion breaks both find_disk_center and limb_center_align).
        # Blending such a frame in at full weight smears real detail rather
        # than improving SNR — confirmed 2026-08-09 on window6 CH4, where
        # one frame needed an ~8px correction vs 0.4-1.1px for its four
        # window-mates and produced a visible top/bottom color split.
        # N per window is typically small (3-9), so MAD alone is noisy —
        # require both a robust relative outlier test AND a hard absolute
        # floor before excluding a frame.
        shifts_px = np.array([fl["align_shift_px"] for fl in log_frames], dtype=np.float64)
        non_ref_mask = np.array([fl["align_method"] != "reference" for fl in log_frames])
        if non_ref_mask.sum() >= 2:
            non_ref_shifts = shifts_px[non_ref_mask]
            median_shift = np.median(non_ref_shifts, axis=0)
            dist = np.linalg.norm(shifts_px - median_shift, axis=1)
            mad = np.median(np.abs(dist[non_ref_mask] - np.median(dist[non_ref_mask])))
            robust_sigma = 1.4826 * mad
            _OUTLIER_ABS_PX = 5.0
            outlier = non_ref_mask & (dist > max(1.5, 4.0 * robust_sigma)) & (dist > _OUTLIER_ABS_PX)
            n_outliers = int(outlier.sum())
            if 0 < n_outliers < len(warped_images):
                kept_images: List[np.ndarray] = []
                kept_weights: List[float] = []
                for i, (img, wgt) in enumerate(zip(warped_images, weights)):
                    fl = log_frames[i]
                    if outlier[i]:
                        fl["outlier_excluded"] = True
                        fl["outlier_dist_px"] = round(float(dist[i]), 2)
                        print(
                            f"    [outlier] {fl['stem']}: shift={fl['align_shift_px']} is "
                            f"{dist[i]:.1f}px from window median {median_shift.round(2).tolist()} "
                            f"(robust_sigma={robust_sigma:.2f}) — excluded from stack"
                        )
                        continue
                    kept_images.append(img)
                    kept_weights.append(wgt)
                warped_images = kept_images
                weights = kept_weights

    stacked = quality_weighted_stack(warped_images, weights, weight_power=weight_power)

    log_dict = {
        "n_stacked":             len(warped_images),
        "reference_stem":        reference_row["stem"],
        "reference_time":        t_reference.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "period_hours":          period_hours,
        "warp_scale":            warp_scale,
        "pole_pa_deg":           pole_pa_deg,
        "flip_direction":        flip_direction,
        "weight_power":          weight_power,
        "align_enabled":         align,
        "normalize_brightness":  normalize_brightness,
        "min_quality_threshold": min_quality_threshold,
        "frames":                log_frames,
    }

    return stacked, log_dict


# ── Multi-filter de-rotation for one window ────────────────────────────────────

def derotate_window(
    window: dict,
    required_filters: List[str],
    period_hours: float = 9.9281,
    warp_scale: float = 1.00,
    align: bool = True,
    normalize_brightness: bool = False,
    min_quality_threshold: float = 0.0,
    pole_pa_deg: float = 0.0,
    color_mode: bool = False,
    flip_ns: bool = False,
    out_dir: Optional[Path] = None,
    weight_power: float = 1.0,
) -> Dict[str, Tuple[Optional[Path], dict]]:
    """De-rotate and stack all filters in a single time window.

    Args:
        window:           Window dict from Step 4 (center_time + per_filter data).
        required_filters: Filters to process.
        period_hours:     System II rotation period.
        warp_scale:       Spherical warp scale (passed through to derotate_filter).
        align:            Sub-pixel alignment between frames.
        flip_ns:          South-up camera flag from auto_detect_ns_flip(); passed
                          as flip_direction to spherical_derotation_warp().
        out_dir:          If provided, save TIF files here.
        weight_power:     Exponent applied to norm_score weights in the
                          per-filter stack (see quality_weighted_stack). 1.0
                          (default) = original linear blend, unchanged for
                          every existing caller.

    Returns:
        {filter: (output_path_or_None, log_dict)}
    """
    t_ref = window["center_time"]
    results: Dict[str, Tuple[Optional[Path], dict]] = {}

    for filt in required_filters:
        if filt not in window["per_filter"]:
            print(f"    [{filt}] Not in window — skipped")
            continue

        included = window["per_filter"][filt]["included"]
        if not included:
            print(f"    [{filt}] No included frames — skipped")
            continue

        n = len(included)
        print(f"    [{filt}] De-rotating {n} frame(s)…", end="", flush=True)

        try:
            stacked, log = derotate_filter(
                included, t_ref, period_hours,
                warp_scale=warp_scale,
                align=align,
                normalize_brightness=normalize_brightness,
                min_quality_threshold=min_quality_threshold,
                pole_pa_deg=pole_pa_deg,
                color_mode=color_mode,
                flip_direction=flip_ns,
                weight_power=weight_power,
            )
        except Exception as exc:
            print(f" ERROR: {exc}")
            results[filt] = (None, {"error": str(exc)})
            continue

        out_path: Optional[Path] = None
        if out_dir is not None:
            out_path = out_dir / f"{filt}_derotated.tif"
            if color_mode:
                image_io.write_tif_color_16bit(stacked, out_path)
            else:
                image_io.write_tif_16bit(stacked, out_path)

        snr_gain = round(float(np.sqrt(n)), 3)
        print(f" done  (SNR×{snr_gain:.2f})")
        results[filt] = (out_path, log)

    return results


# ── JSON serialisation helper ──────────────────────────────────────────────────

def derotation_log_to_json(
    window_index: int,
    window: dict,
    filter_results: Dict[str, Tuple[Optional[Path], dict]],
) -> dict:
    """Serialise de-rotation log to a JSON-compatible dict."""
    def _fmt(dt: datetime) -> str:
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    filters_log = {}
    for filt, (out_path, log) in filter_results.items():
        filters_log[filt] = {
            "output_file": str(out_path) if out_path else None,
            **log,
        }

    return {
        "window_index":     window_index,
        "center_time":      _fmt(window["center_time"]),
        "window_start":     _fmt(window["window_start"]),
        "window_end":       _fmt(window["window_end"]),
        "window_quality":   window["window_quality"],
        "rotation_degrees": window["rotation_degrees"],
        "filters":          filters_log,
    }
