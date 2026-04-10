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

warp_scale ≈ 0.80 (empirically determined for Jupiter through systematic testing):
  Theoretical value is 1.0 (full spherical projection) but effective scale is
  reduced by seeing blur, plate scale uncertainty, and oblate disk geometry.
  0.80 was found to minimize inter-filter alignment error; 0.20 caused severe
  filter-to-filter drift, confirming that near-theoretical scale is needed.
  For other planets or significantly different setups, NCC sweep re-calibration
  is recommended.

Saturn notes:
  - Use rotation_period_hours=10.56 (System III)
  - pole_pa_deg is fetched from Horizons NP.ang; for low sub-Earth latitudes
    (<15°) the horizontal approximation (pole_pa=0) is acceptable.
  - Ring features do NOT co-rotate with the atmosphere; they will be slightly
    smeared in the stack (atmosphere is the primary target).

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


# ── Disk geometry ──────────────────────────────────────────────────────────────

def find_disk_center(
    image: np.ndarray,
    margin_factor: float = 0.10,
) -> Tuple[float, float, float, float, float]:
    """Locate planet disk via ellipse fitting.

    Args:
        image:         2-D float [0, 1] image.
        margin_factor: Margin below Otsu threshold to include dim limb pixels.

    Returns:
        (cx, cy, semi_major, semi_minor, angle_deg) — ellipse parameters.
        Falls back to image centroid if ellipse fitting fails.
    """
    # Convert to uint8 for thresholding
    arr8 = np.clip(image * 255, 0, 255).astype(np.uint8)
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
        semi_a = max(ma, mi) / 2
        semi_b = min(ma, mi) / 2
        return float(cx), float(cy), float(semi_a), float(semi_b), float(angle)
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
        (dx, dy) — shift to apply to target to align with reference.
    """
    ref_f32 = reference.astype(np.float32)
    tgt_f32 = target.astype(np.float32)
    (dx, dy), _ = cv2.phaseCorrelate(ref_f32, tgt_f32)
    return float(dx), float(dy)


def limb_center_align(
    ref_cx: float,
    ref_cy: float,
    target_lum: np.ndarray,
    max_shift_px: float = 15.0,
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
        cx, cy, semi_a, *_ = find_disk_center(target_lum)
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
) -> np.ndarray:
    """Stack images with quality weights (weighted mean).

    Args:
        images:  List of 2-D float [0, 1] arrays (all same shape).
        weights: Quality score per image (norm_score from Step 4).
                 Does not need to sum to 1 — normalised internally.

    Returns:
        Weighted mean stack, float [0, 1].
    """
    if len(images) == 1:
        return images[0].copy()

    w_arr = np.array(weights, dtype=np.float64)
    w_arr = np.clip(w_arr, 1e-9, None)
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
    warp_scale: float = 0.20,
    align: bool = True,
    normalize_brightness: bool = False,
    min_quality_threshold: float = 0.0,
    pole_pa_deg: float = 0.0,
    color_mode: bool = False,
) -> Tuple[np.ndarray, dict]:
    """De-rotate and stack a single filter's images for one time window.

    Uses spherical de-rotation warp (Δx ∝ depth) — NOT image rotation.

    Args:
        included_rows: List of score dicts from Step 4 (must have 'path',
                       'timestamp', 'norm_score').
        t_reference:   Window center time (reference orientation).
        period_hours:  System II period.
        warp_scale:    Spherical warp empirical scale factor (default 0.20).
        align:         If True, apply sub-pixel phase correlation alignment
                       after warp. Disable for speed testing.
        color_mode:    If True, preserve RGB channels throughout; disk detection
                       and alignment are computed on the luminance channel.

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

        dt_sec = (row["timestamp"] - t_reference).total_seconds()
        warped = spherical_derotation_warp(
            img, dt_sec, ref_cx, ref_cy, ref_semi_a,
            period_hours=period_hours,
            scale=warp_scale,
            flip_direction=False,
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

    # ── Sub-pixel translation alignment (limb-center based) ──────────────────
    # Primary: ellipse-fit disk center alignment (limb_center_align).
    # Each warped frame's disk center is measured via ellipse fitting and
    # shifted to match the reference center (ref_cx, ref_cy).
    # This directly corrects whole-disk position wobble from atmospheric seeing
    # without being biased by background noise as phaseCorrelate can be.
    # Fallback: phaseCorrelate when limb_center_align returns (0, 0) for a
    # non-reference frame (signals detection failure).
    if align and ref_img is not None and len(warped_images) > 1:
        aligned_images: List[np.ndarray] = []
        ref_lum = _to_luminance(ref_img)
        for img, frame_log in zip(warped_images, log_frames):
            if frame_log["stem"] == reference_row["stem"]:
                aligned_images.append(img)
                frame_log["align_shift_px"] = [0.0, 0.0]
                frame_log["align_method"] = "reference"
            else:
                img_lum = _to_luminance(img)
                dx, dy = limb_center_align(ref_cx, ref_cy, img_lum)
                method = "limb_center"
                if dx == 0.0 and dy == 0.0:
                    # Detection failed — fall back to phaseCorrelate
                    dx, dy = subpixel_align(ref_lum, img_lum)
                    method = "phase_correlate"
                aligned = apply_shift(img, dx, dy)
                aligned_images.append(aligned)
                frame_log["align_shift_px"] = [round(dx, 3), round(dy, 3)]
                frame_log["align_method"] = method
        warped_images = aligned_images

    stacked = quality_weighted_stack(warped_images, weights)

    log_dict = {
        "n_stacked":             len(warped_images),
        "reference_stem":        reference_row["stem"],
        "reference_time":        t_reference.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "period_hours":          period_hours,
        "warp_scale":            warp_scale,
        "pole_pa_deg":           pole_pa_deg,
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
    warp_scale: float = 0.20,
    align: bool = True,
    normalize_brightness: bool = False,
    min_quality_threshold: float = 0.0,
    pole_pa_deg: float = 0.0,
    color_mode: bool = False,
    out_dir: Optional[Path] = None,
) -> Dict[str, Tuple[Optional[Path], dict]]:
    """De-rotate and stack all filters in a single time window.

    Args:
        window:           Window dict from Step 4 (center_time + per_filter data).
        required_filters: Filters to process.
        period_hours:     System II rotation period.
        warp_scale:       Spherical warp scale (passed through to derotate_filter).
        align:            Sub-pixel alignment between frames.
        out_dir:          If provided, save TIF files here.

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
