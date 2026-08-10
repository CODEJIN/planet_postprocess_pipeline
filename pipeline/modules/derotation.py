"""
Planetary de-rotation module.

Algorithm (per filter, per time window):
  1. Find planet disk center via ellipse fitting on each image (sub-pixel accurate)
  2. Compute CML displacement from the configured rotation period
  3. Apply spherical de-rotation warp using cv2.remap, mixing INTER_CUBIC
     (disk interior, detail-preserving) with INTER_LINEAR (limb/exterior,
     ringing-free) via a feathered blend — see spherical_derotation_warp's
     own docstring for the exact weighting; NOT Lanczos-4 despite what an
     earlier revision of this note claimed.

     The warp direction is determined by *pole_pa_deg* — an IMAGE-SPACE
     angle measured directly from pixel data (equator_pa_from_disk_ellipse() /
     auto_detect_equator_pa(), called from derotate_stack.py's
     _scan_session_pole_pa()), NEVER a raw sky-frame quantity fed in
     directly. This is worth stating unambiguously because an earlier
     revision of this exact paragraph claimed pole_pa_deg was "queried from
     JPL Horizons quantity 17 NP.ang" — that was wrong, and it independently
     misled two rounds of external code review into flagging a "Horizons
     sky-PA vs image-PA mismatch" bug that doesn't exist in the actual data
     flow. NP.ang IS queried (query_horizons_np_ang(), a real, correct,
     celestial-sky-frame angle) but only for a different purpose entirely:
     the satellite/shadow tracker's camera-to-sky rotation, computed as
     θ_cam = pole_pa_deg + NP.ang in pipeline/steps/derotate_stack.py. The
     two angles live in different frames and are never interchangeable.

       Δx(x,y) = scale × Δλ_rad × depth(x,y) × cos(pole_pa_rad)
       Δy(x,y) = scale × Δλ_rad × depth(x,y) × sin(pole_pa_rad)
       depth(x,y) = sqrt(max(0, R² − (x−cx)² − (y−cy)²))

     pole_pa_deg = 0  →  the drift axis is horizontal in image pixels
                         (default for Jupiter's typical camera orientation)
     pole_pa_deg ≠ 0  →  drift axis rotated in the image (camera roll),
                         independent of sub_observer_lat_deg (B) — see the
                         "True oblate-spheroid reprojection" section below
                         for why these are two genuinely different angles,
                         not the same "tilt" described two ways.

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
  - pole_pa_deg is measured from the image (see note above — not Horizons),
    independent of sub-observer latitude (B). For low |B| (<15°), skipping
    the 3D reprojection's B-tilt term and using this linear warp is an
    acceptable approximation regardless of what pole_pa_deg happens to be —
    forcing pole_pa_deg to 0 is a separate, unrelated simplification that
    only makes sense if the drift axis is ACTUALLY horizontal in the image.
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
from dataclasses import dataclass
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
    north pole, measured eastward, in the celestial-sky frame. NOT used as
    :func:`spherical_derotation_warp`'s ``pole_pa_deg`` (that is a separate,
    image-space angle measured directly from pixel data — see that
    function's docstring). This value is used instead by the satellite/
    shadow tracker's camera-to-sky rotation, θ_cam = pole_pa_deg + NP.ang,
    in pipeline/steps/derotate_stack.py. Returns None only if all sources
    fail.
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
            frac = (
                t_utc.hour * 3600.0 + t_utc.minute * 60.0
                + t_utc.second + t_utc.microsecond / 1_000_000.0
            ) / 86400.0
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


# ── Bundled sub-observer latitude (B) lookup table ────────────────────────────
#
# Same coverage/format as the NP.ang table above (Jupiter 599 / Saturn 699 /
# Mars 499, 2016-01-01 ~ 2036-12-31, 1-day resolution). Sub-observer latitude
# does not wrap around like a position angle, so interpolation is plain
# linear (no 360°-wraparound handling needed).

_SUB_OBS_LAT_TABLE_PATH = Path(__file__).parent.parent / "data" / "sub_observer_lat_table.json"

_SUB_OBS_LAT_BUNDLE: Optional[Dict[str, Dict[str, float]]] = None


def _load_sub_obs_lat_bundle() -> Dict[str, Dict[str, float]]:
    global _SUB_OBS_LAT_BUNDLE
    if _SUB_OBS_LAT_BUNDLE is None:
        try:
            with open(_SUB_OBS_LAT_TABLE_PATH, encoding="utf-8") as f:
                raw = json.load(f)
            _SUB_OBS_LAT_BUNDLE = raw.get("planets", {})
        except Exception as exc:
            warnings.warn(f"[ObsSub-LAT] Could not load bundled table: {exc}")
            _SUB_OBS_LAT_BUNDLE = {}
    return _SUB_OBS_LAT_BUNDLE


def query_horizons_sub_observer_lat(
    horizons_id: str,
    t_utc: datetime,
    observer_code: str = "500@399",
) -> Optional[float]:
    """Return the planet's sub-observer (planetographic) latitude B at *t_utc*.

    Lookup order mirrors :func:`query_horizons_np_ang` exactly: bundled table
    → user-local cache → live JPL Horizons query.

    Horizons quantity 14 ("Observer sub-longitude & sub-latitude") returns
    the apparent planetodetic (=planetographic) longitude/latitude of the
    disc center as seen by the observer — this is Earth's latitude on the
    target body, i.e. how far the rotation axis is tilted toward/away from
    the observer. Used by :func:`spherical_derotation_warp_3d` as
    ``sub_observer_lat_deg``. Returns None only if all sources fail.

    Two numbers are returned per line by Horizons for this quantity
    (ObsSub-LON, ObsSub-LAT, in that order) — only the second is latitude.
    """
    # ── 1. Bundled table (primary, offline) ───────────────────────────────────
    bundle = _load_sub_obs_lat_bundle()
    planet_table = bundle.get(horizons_id)
    if planet_table:
        d0 = t_utc.strftime("%Y-%m-%d")
        d1 = (t_utc + timedelta(days=1)).strftime("%Y-%m-%d")
        if d0 in planet_table:
            v0 = planet_table[d0]
            v1 = planet_table.get(d1, v0)
            frac = (
                t_utc.hour * 3600.0 + t_utc.minute * 60.0
                + t_utc.second + t_utc.microsecond / 1_000_000.0
            ) / 86400.0
            result = v0 + frac * (v1 - v0)
            print(f"  [ObsSub-LAT] {d0} → {result:.3f}° (bundle, id={horizons_id})")
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
    cache_key = f"subobslat:{horizons_id}:{date_str}"
    if cache_key in cache:
        val = cache[cache_key]
        print(f"  [ObsSub-LAT] {date_str} → {val:.3f}° (user cache, id={horizons_id})")
        return val

    # ── 3. Live Horizons query (fallback) ──────────────────────────────────────
    start = t_utc.strftime("%Y-%m-%d %H:%M")
    stop  = (t_utc + timedelta(minutes=2)).strftime("%Y-%m-%d %H:%M")
    params = urllib.parse.urlencode({
        "format": "text", "COMMAND": f"'{horizons_id}'",
        "OBJ_DATA": "NO", "MAKE_EPHEM": "YES", "EPHEM_TYPE": "OBSERVER",
        "CENTER": f"'{observer_code}'",
        "START_TIME": f"'{start}'", "STOP_TIME": f"'{stop}'",
        "STEP_SIZE": "1m", "QUANTITIES": "14",
    })
    url = f"https://ssd.jpl.nasa.gov/api/horizons.api?{params}"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            text = resp.read().decode("utf-8")
    except Exception as exc:
        warnings.warn(f"[ObsSub-LAT] Horizons query failed: {exc} → defaulting to None")
        return None

    soe, eoe = text.find("$$SOE"), text.find("$$EOE")
    if soe < 0 or eoe < 0:
        warnings.warn("[ObsSub-LAT] Horizons response missing $$SOE/$$EOE")
        return None
    data_lines = [l for l in text[soe + 5:eoe].split("\n") if l.strip()]
    if not data_lines:
        return None

    def _parse_line(dl: str) -> Optional[float]:
        # Format: "<date> <time>     <ObsSub-LON> <ObsSub-LAT>" — take the
        # LAST two numbers on the line and use the second one (latitude).
        nums = re.findall(r"-?\d+\.\d+", dl)
        return float(nums[-1]) if len(nums) >= 2 else None

    result = _parse_line(data_lines[0])
    if result is None:
        warnings.warn("[ObsSub-LAT] Could not parse Horizons response")
        return None

    # Save to user cache for offline reuse
    try:
        cache[cache_key] = result
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except Exception:
        pass
    print(f"  [ObsSub-LAT] {date_str} → {result:.3f}° (Horizons live, id={horizons_id})")
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

def _has_ring_signature(image: np.ndarray, aspect_threshold: float = 0.80) -> bool:
    """Cheap check: does the raw (uncorrected) disk blob look ring-contaminated?

    Mirrors the raw Otsu + fitEllipse step inside find_disk_center() — same
    threshold, same morphology, same raw_aspect test — without the expensive
    core-isolation/gradient refinement that follows it. Used by
    lucky_stack.score_frames_log_disk() to decide whether to restrict its
    Laplacian-variance quality mask to the disk interior (ring edges
    otherwise bias that AS!4-mirroring metric toward "the ring looks sharp"
    rather than the atmosphere). find_disk_center() itself now does the
    equivalent aspect-based gating inline (see raw_aspect < 0.80 there) and
    no longer calls this function directly — resolve_shared_shape()/
    resolve_filter_pose() decide cross-filter sharing from the confidence/
    shape_reliable fields _find_disk_center_impl() already returns instead.
    """
    arr8 = np.clip(image * 255, 0, 255).astype(np.uint8)
    thresh_val, _ = cv2.threshold(arr8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, binary = cv2.threshold(arr8, max(1, int(thresh_val * 0.90)), 255, cv2.THRESH_BINARY)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return False
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 5:
        return False
    (_, _), (ma, mi), _ = cv2.fitEllipse(largest)
    semi_a, semi_b = (ma, mi) if ma >= mi else (mi, ma)
    if semi_a <= 0:
        return False
    return (semi_b / semi_a) < aspect_threshold


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
    return_n_valid: bool = False,
):
    """Estimate disk radius from steepest-gradient limb edge along radial rays.

    Replaces the Otsu-threshold disk_r (which underestimates the true limb by
    ~5 px due to limb darkening) with a gradient-profile measurement.  cx/cy
    are unchanged.  Falls back to r_rough if fewer than 8 valid rays are found
    — callers that need to distinguish that internal failure from a genuine
    measurement (which can legitimately return a value equal to r_rough, so
    comparing the return value alone is not reliable) should pass
    return_n_valid=True to additionally get the number of rays that
    contributed to the result.

    Returns:
        float radius, or (float radius, int n_valid_rays) if return_n_valid.
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
        return (r_rough, len(edge_radii)) if return_n_valid else r_rough
    arr = np.array(edge_radii)
    med = float(np.median(arr))
    std = float(np.std(arr))
    keep = arr[np.abs(arr - med) < outlier_sigma * (std + 0.5)]
    if len(keep) < 6:
        keep = arr
    result = float(np.median(keep))
    return (result, len(edge_radii)) if return_n_valid else result


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
                         isolate the disk core from a fainter attached ring.
                         Only the bright-core hypothesis is tried (a
                         symmetric dark-core attempt was tested and does NOT
                         work on real data — see _find_disk_center_impl's
                         confidence=0.5/0.3 docstring below); when bright-core
                         isolation fails (e.g. Saturn's CH4 band, whose disk
                         is darker than its ring), this falls back to a
                         radial limb search instead of a polarity flip —
                         still filter-agnostic, just not via "try both
                         polarities".

    Returns:
        (cx, cy, semi_major, semi_minor, angle_deg) — ellipse parameters.
        Falls back to image centroid if ellipse fitting fails.
    """
    cx, cy, semi_a, semi_b, angle_major, _confidence, _shape_reliable = _find_disk_center_impl(
        image, margin_factor, fixed_threshold, core_percentile
    )
    return cx, cy, semi_a, semi_b, angle_major


def _find_disk_center_impl(
    image: np.ndarray,
    margin_factor: float = 0.10,
    fixed_threshold: int = 0,
    core_percentile: float = 60.0,
) -> Tuple[float, float, float, float, float, float, bool]:
    """Implementation for find_disk_center(), plus confidence/reliability info.

    Returns (cx, cy, semi_major, semi_minor, angle_major_deg, confidence, shape_reliable).

    confidence semantics:
        1.0  — raw_aspect >= 0.80, no ring-core isolation was needed at all
               (ringless target: Jupiter/Mars/Venus).
        (0,1]— bright-core percentile isolation ran and found a plausible
               core; value is isolated-core-pixels / raw-blob-pixels
               (smaller = more confidently shrunk down from the disk+ring
               blob to just the disk). Note this can coincidentally overlap
               the fixed 0.5/0.3 values below in rare cases — check
               shape_reliable, not confidence alone, to distinguish "real"
               percentile isolation from the radial-limb fallback paths.
        0.5  — bright-core isolation failed (disk darker than an attached
               ring, e.g. Saturn's CH4 band, or a ring band crossing in
               front of the disk splits the blob regardless of polarity —
               both confirmed on real data); fell back to a direct radial
               limb search for semi_major, and that search found >= 8 valid
               edge crossings — a genuine, gradient-confirmed measurement
               (see shape_reliable).
        0.3  — same fallback, but the radial limb search found too few valid
               edge crossings to trust on its own (confirmed common on real
               Saturn CH4 data — its limb gradient is weak/smooth enough that
               most rays' steepest drop lands at the search window's edge,
               not a clean interior minimum). semi_major is the geometric
               seed (bounding-box vertical half-extent) alone, unconfirmed by
               direct edge detection — empirically stable and physically
               plausible, but a weaker claim than the 0.5 case.
        0.0  — neither approach found a plausible disk; fell back to the raw
               (disk+ring) fit, which is known-unreliable for this frame.

    shape_reliable: False when semi_minor/angle_major are just a circular
        placeholder (the confidence=0.5/0.3 radial-limb-only paths measure
        semi_major directly but have no independent oblateness/orientation
        measurement) — callers that need shape should borrow aspect_ratio/
        angle from a sibling filter's reliable fit in this case (this is
        exactly what resolve_shared_shape/resolve_filter_pose do).

    Only find_disk_center()'s 5-tuple is part of the public API; the rest is
    for internal callers (resolve_shared_shape) that need to compare/combine
    candidates across filters.
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
        return float(w / 2), float(h / 2), float(min(h, w) / 4), float(min(h, w) / 4), 0.0, 0.0, False

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
        confidence = 1.0
        shape_reliable = True
        if raw_aspect < 0.80:
            confidence = 0.0  # overwritten below if a plausible result is found
            shape_reliable = False
            x, y, bw, bh = cv2.boundingRect(largest)
            roi_vals = arr8[y : y + bh, x : x + bw]
            roi_mask = binary[y : y + bh, x : x + bw] > 0
            vals = roi_vals[roi_mask]
            core_found = False
            if vals.size > 0:
                # Bright-core isolation: works when the disk is brighter than
                # an attached ring (the common case).
                core_thv = np.percentile(vals, core_percentile)
                core_bin = (((roi_vals >= core_thv) & roi_mask).astype(np.uint8)) * 255
                core_contours, _ = cv2.findContours(core_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if core_contours:
                    core_largest = max(core_contours, key=cv2.contourArea)
                    if len(core_largest) >= 5:
                        (ccx, ccy), (cma, cmi), cangle = cv2.fitEllipse(
                            core_largest + np.array([x, y], dtype=core_largest.dtype)
                        )
                        if cma >= cmi:
                            c_semi_a, c_semi_b, c_angle = cma / 2, cmi / 2, cangle
                        else:
                            c_semi_a, c_semi_b, c_angle = cmi / 2, cma / 2, (cangle + 90.0) % 180.0
                        c_aspect = c_semi_b / c_semi_a if c_semi_a > 0 else 0.0
                        # Plausible only if it's a real, non-degenerate shrink
                        # from the raw disk+ring blob — a real oblate disk
                        # never has aspect < 0.4, and a genuine core isolation
                        # should shrink meaningfully, not barely move.
                        if c_semi_a > 0 and c_aspect >= 0.4 and c_semi_a < 0.95 * semi_a:
                            cx, cy, semi_a, semi_b, angle_major = ccx, ccy, c_semi_a, c_semi_b, c_angle
                            confidence = int(np.count_nonzero(core_bin)) / vals.size
                            shape_reliable = True
                            core_found = True

            if not core_found:
                # Bright-core isolation failed. This happens when the disk is
                # actually DARKER than an attached ring (e.g. Saturn's CH4
                # band: methane absorption makes the globe darker than the
                # icy rings — confirmed: bright-core isolation there stayed
                # ring-sized, aspect ~0.16-0.34), or when a bright ring band
                # crosses directly in front of the disk and splits the
                # brightness-based blob into disconnected pieces, defeating
                # single-threshold segmentation at ANY polarity (confirmed
                # visually on real CH4 data — a symmetric "dark core"
                # percentile attempt there also failed, for this reason).
                # Neither cause is filter-specific, so don't special-case by
                # name: fall back to a direct radial limb search for the
                # outer edge instead of blob segmentation. Seed it from the
                # bounding box's VERTICAL extent (bh/2), not the horizontal
                # extent or the raw ellipse fit — a sideways-extending ring
                # inflates width/semi_major but rarely inflates the blob's
                # height nearly as much (confirmed on real Saturn CH4 data:
                # this seed landed at ~56px, matching the true disk, vs. the
                # raw ellipse fit's 145+px). The wide search window plus
                # _gradient_disk_r's existing median/outlier-sigma robustness
                # then lets the majority of ring-free rays outvote the
                # minority that do cross the ring.
                bx, by, bw2, bh2 = cv2.boundingRect(largest)
                # Moments-based centroid (mass-weighted centre of the actual
                # blob shape), not the bounding-box corner-midpoint — more
                # robust if the blob is asymmetric (e.g. a ring band
                # occluding only one side of the disk), though it does not
                # fully solve that case (still a single blob-wide estimate,
                # not disk-limb-specific).
                _m = cv2.moments(largest)
                if _m["m00"] > 0:
                    seed_cx, seed_cy = _m["m10"] / _m["m00"], _m["m01"] / _m["m00"]
                else:
                    seed_cx, seed_cy = bx + bw2 / 2.0, by + bh2 / 2.0
                seed_r = bh2 / 2.0
                limb_r, _n_valid_rays = _gradient_disk_r(
                    image, seed_cx, seed_cy, seed_r, search_frac=(0.5, 1.6),
                    outlier_sigma=1.5, return_n_valid=True,
                )
                # _n_valid_rays >= 8 is required for _gradient_disk_r() to
                # even attempt a real measurement (see its own docstring) —
                # without checking this explicitly, its internal too-few-rays
                # fallback (which just echoes r_rough back unchanged) would be
                # indistinguishable from a genuine successful measurement.
                # This matters in practice: real Saturn CH4 data typically
                # gets very few valid rays here (confirmed 0-6 of 72) because
                # CH4's limb gradient is weak/smooth enough that most rays'
                # steepest drop lands at the search window's edge rather than
                # a clean interior minimum — so most CH4 frames land in the
                # "unconfirmed seed" branch below, not this one.
                if _n_valid_rays >= 8 and limb_r > 5.0 and 0.5 * seed_r <= limb_r <= 1.5 * seed_r:
                    # Semi-minor/angle are unknown here — a circular
                    # placeholder (shape_reliable=False signals this to
                    # callers that can borrow a sibling filter's oblateness).
                    cx, cy, semi_a, semi_b, angle_major = seed_cx, seed_cy, limb_r, limb_r, 0.0
                    confidence = 0.5  # genuine gradient-confirmed measurement
                    shape_reliable = False
                elif seed_r > 5.0:
                    # Gradient search inconclusive — fall back to the
                    # geometric seed itself (bounding-box vertical
                    # half-extent), unconfirmed by direct edge detection.
                    # Empirically stable and physically plausible on real
                    # Saturn CH4 data (consistently ~56px across a whole
                    # session, smaller than sibling filters' ~66-68px, as
                    # expected for methane-band depth-sensing), but this is a
                    # weaker, lower-confidence claim than a real measurement,
                    # so it must not be indistinguishable from one — a caller
                    # comparing confidence across filters (resolve_shared_shape)
                    # should prefer an actual measurement when one exists.
                    cx, cy, semi_a, semi_b, angle_major = seed_cx, seed_cy, seed_r, seed_r, 0.0
                    confidence = 0.3  # unconfirmed geometric estimate
                    shape_reliable = False
                # else: neither approach worked — keep the raw (disk+ring)
                # fit; confidence stays 0.0, a known-unreliable result.

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
        return (
            float(cx), float(cy), float(semi_a_refined), float(semi_b_refined),
            float(angle_major), float(confidence), bool(shape_reliable),
        )
    else:
        # Fallback: centroid of bounding box
        x, y, w, h = cv2.boundingRect(largest)
        return float(x + w / 2), float(y + h / 2), float(max(w, h) / 2), float(min(w, h) / 2), 0.0, 0.0, False


# ── Shared disk geometry across filters (shape/pose separation) ───────────────
#
# A ringed planet's per-filter disk fits disagree slightly (different SNR/
# contrast per filter — confirmed on real Saturn data: fitted semi_b/semi_a
# ranged 0.84-0.93 across IR/R/G/B in one window) and some filters can fail
# to determine their own oblateness at all (see shape_reliable in
# _find_disk_center_impl). Sharing helps, but naively sharing (cx, cy,
# semi_a, semi_b) wholesale from one "probe" filter onto every other filter
# — as this module used to do — conflates two physically different things:
#
#   shape (aspect_ratio, equator_pa_deg): pure geometry (this instant's
#       oblate silhouette orientation), genuinely the same across every
#       filter and safe to share.
#   size (semi_major_px):  can legitimately differ per filter — e.g.
#       Saturn's CH4-band disk is measurably smaller than broadband filters
#       because deep methane absorption only lets us see the highest
#       atmospheric layer (well-known radiative-transfer effect, confirmed
#       on real data: CH4 ~56px vs. IR/R/G/B ~66-68px in the same session).
#   pose (cx, cy): this filter's own optical registration — never borrowed
#       by default, since filter-wheel decentering or differential
#       refraction between filters could otherwise silently corrupt every
#       non-probe filter's position.
#
# PlanetShape carries only the first (shape); FilterPose carries only pose.
# Each filter's own semi_major_px always comes from its own detection.
# No filter name is ever treated specially by either resolver below —
# whichever filter's own fit happens to be shape_reliable/high-confidence
# wins, so a different filter with a similar issue in the future is handled
# the same way, and CH4 itself is free to win if some future frame's own
# detection succeeds.

@dataclass
class PlanetShape:
    """Oblateness/orientation shared across filters at one instant."""
    aspect_ratio: float       # semi_minor / semi_major
    # NOTE (flagged by external review, 2026-08-10, confirmed real): this is
    # computed by resolve_shared_shape() and logged, but never actually
    # substituted into the warp's own pole_pa_deg anywhere in derotate_filter()
    # — the warp always uses derotate_stack.py's session_pole_pa (a robust
    # median over EVERY frame in the session) uniformly across all filters.
    # This is not obviously a bug to fix by simply wiring it in: equator_pa_deg
    # here comes from a SINGLE window's SINGLE reference-frame ellipse fit
    # (noisier), so substituting it for the session-wide median could easily
    # be a net regression for the common case, not an improvement — it would
    # need real validation on a ring/CH4-affected Saturn session (where the
    # aspect_ratio half of this dataclass already gets used) before touching,
    # not a blind wire-up. Left as a documented, deliberately-open question.
    equator_pa_deg: float
    # Reserved for a future true oblate-spheroid re-projection warp (see
    # project notes on WinJUPOS-style de-rotation) — unused today.
    sub_observer_lat_deg: Optional[float] = None


@dataclass
class FilterPose:
    """This filter's own disk center (pixels)."""
    center_x_px: float
    center_y_px: float
    # Only set for the "registered_to_probe" fallback (this filter's own
    # detection failed outright, confidence==0.0): the probe filter's own
    # semi_major, usable as a size estimate since this filter has none of
    # its own. None for the normal "own_detection" case (this filter's own
    # semi_a from _find_disk_center_impl is used directly instead).
    semi_major_px: Optional[float] = None


def resolve_shared_shape(
    candidate_fits: Dict[str, Tuple[float, float, float, float, float, float, bool]],
) -> Optional[Tuple[PlanetShape, str]]:
    """Pick the most confident shape-reliable disk fit among candidates.

    Args:
        candidate_fits: {filter_name: _find_disk_center_impl(...) result},
            for whichever filters/frames the caller already evaluated.

    Returns:
        (PlanetShape, source_filter_name), or None if no candidate has a
        reliable shape (mirrors the "no ring detected -> stay independent"
        behaviour this replaces).
    """
    best: Optional[Tuple[str, float, PlanetShape]] = None
    for filt, fit in candidate_fits.items():
        _cx, _cy, semi_a, semi_b, angle, confidence, shape_reliable = fit
        if not shape_reliable or semi_a <= 0:
            continue
        if best is None or confidence > best[1]:
            shape = PlanetShape(aspect_ratio=semi_b / semi_a, equator_pa_deg=angle)
            best = (filt, confidence, shape)
    if best is None:
        return None
    filt, _confidence, shape = best
    return shape, filt


_RADIUS_SHARE_REL_TOL = 0.08  # 8% — see resolve_shared_radius()


def resolve_shared_radius(
    candidate_fits: Dict[str, Tuple[float, float, float, float, float, float, bool]],
) -> Optional[float]:
    """Median semi_major_px across confident filter fits in one window.

    Found via a user-reported "ring effect" investigation (2026-08-11):
    each filter's own semi_major (used as spherical_derotation_warp()'s
    disk_radius_px) normally differs from its siblings by 1-3px on real
    Jupiter data — pure Otsu-threshold noise from each filter's own
    SNR/contrast, not a real size difference. The warp treats anything
    beyond disk_radius_px*1.05 as having no valid solution and forces it to
    background — so a 1-3px per-filter radius difference means each
    filter's de-rotated stack goes to zero at a slightly different radius.
    Composited, that shows as a colour fringe at the limb; wavelet
    sharpening (which has no way to know the "edge" it's amplifying is an
    algorithmic artifact, not the real limb) makes it worse. This is
    present already in each filter's own step04 output, before composite
    or wavelet ever run — confirmed directly against real per-window
    derotation logs (radii spanning 103.0-104.9px in one window).

    derotate_window() calls this once per window; derotate_filter() then
    only accepts the shared value for a filter whose OWN fit is already
    close to it (see _RADIUS_SHARE_REL_TOL) — a filter with a genuinely
    different apparent size (e.g. an absorption band where the visible
    "surface" sits at a different atmospheric depth) is left on its own
    measurement rather than forced to match, matching this module's
    existing filter-agnostic policy of deciding by measured agreement, not
    by filter name.

    Returns None if fewer than 2 filters have a confident (confidence>0)
    fit — nothing to reconcile.
    """
    semi_majors = [
        fit[2] for fit in candidate_fits.values()
        if fit[5] > 0.0 and fit[2] > 0
    ]
    if len(semi_majors) < 2:
        return None
    return float(np.median(semi_majors))


def resolve_filter_pose(
    fit: Tuple[float, float, float, float, float, float, bool],
    lum: Optional[np.ndarray] = None,
    probe_lum: Optional[np.ndarray] = None,
    probe_pose: Optional[FilterPose] = None,
    probe_semi_major_px: Optional[float] = None,
) -> Tuple[FilterPose, str]:
    """Return this filter's own disk-center pose.

    Args:
        fit: this filter's own _find_disk_center_impl(...) result.
        lum, probe_lum, probe_pose, probe_semi_major_px: only used for the
            fallback path — a probe filter's luminance/pose to register
            against when this filter's own detection fails outright.
            probe_semi_major_px should always be supplied by real callers
            (it sizes the ROI crop and seeds FilterPose.semi_major_px below);
            omitting it falls back to a generic, probe-frame-size-based ROI
            (NOT this filter's own semi_a, which is exactly the known-bad
            disk+ring value that got us into this fallback in the first
            place) and leaves the returned pose's semi_major_px unset.

    Returns:
        (FilterPose, method) where method is "own_detection" (the default —
        used even when shape_reliable is False, since pose is still
        trustworthy there) or "registered_to_probe" (this filter's own
        detection failed completely — confidence == 0.0 — and a probe was
        available to register against instead).
    """
    cx, cy, _semi_a, _semi_b, _angle, confidence, _shape_reliable = fit
    if confidence > 0.0 or probe_lum is None or probe_pose is None or lum is None:
        return FilterPose(cx, cy), "own_detection"

    # ROI-cropped registration against the probe frame — same pattern as
    # composite.align_channels() and derotate_filter()'s pre-warp shift
    # measurement, so a ring elsewhere in frame can't bias the correlation.
    h, w = probe_lum.shape[:2]
    if probe_semi_major_px and probe_semi_major_px > 0:
        r = probe_semi_major_px
    else:
        # No validated size available — this filter's own _semi_a here is
        # the raw disk+ring fit (confidence==0.0's documented meaning), so
        # using it would defeat the ROI crop's whole purpose. Fall back to a
        # generic, geometry-agnostic fraction of the probe frame instead.
        r = min(h, w) * 0.25
    ys = max(0, int(probe_pose.center_y_px - r)); ye = min(h, int(probe_pose.center_y_px + r))
    xs = max(0, int(probe_pose.center_x_px - r)); xe = min(w, int(probe_pose.center_x_px + r))
    if (ye - ys) > 10 and (xe - xs) > 10:
        dx, dy = subpixel_align(probe_lum[ys:ye, xs:xe], lum[ys:ye, xs:xe])
    else:
        dx, dy = subpixel_align(probe_lum, lum)
    # subpixel_align(reference, target) returns the shift to apply to TARGET
    # (via apply_shift(target, dx, dy)) to align it onto REFERENCE — i.e.
    # target's true position expressed in reference coordinates is
    # target_pos + (dx, dy) = reference_pos, so target_pos = reference_pos -
    # (dx, dy). Here reference=probe (known position), target=this filter's
    # own frame (unknown true position, what we're solving for) — so this
    # filter's true centre is probe_pose.center - (dx, dy), NOT + (dx, dy).
    return (
        FilterPose(
            probe_pose.center_x_px - dx,
            probe_pose.center_y_px - dy,
            semi_major_px=probe_semi_major_px,
        ),
        "registered_to_probe",
    )


# ── Spherical de-rotation warp ────────────────────────────────────────────────

def spherical_derotation_warp(
    image: np.ndarray,
    dt_sec: float,
    cx: float,
    cy: float,
    disk_radius_px: float,
    period_hours: float = 9.9281,
    scale: float = 1.00,
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
    in the image, parameterised by *pole_pa_deg* — an IMAGE-SPACE angle
    measured directly from pixel data (equator_pa_from_disk_ellipse() /
    auto_detect_equator_pa()), NOT a raw sky-frame quantity from Horizons. (JPL
    Horizons' NP.ang, queried via query_horizons_np_ang(), is a genuinely
    different, celestial-sky-frame angle used elsewhere — the satellite
    tracker's camera-to-sky rotation, θ_cam = pole_pa_deg + NP.ang, in
    pipeline/steps/derotate_stack.py. Do not feed NP.ang into this function
    directly; earlier revisions of this docstring incorrectly implied you
    should, which independently misled two rounds of external code review —
    see project notes.):

        Δx = drift × cos(pole_pa_rad)   [horizontal component]
        Δy = drift × sin(pole_pa_rad)   [vertical component]

    When the drift axis happens to be horizontal in the image (pole_pa ≈ 0°,
    the common case for this project's typical camera/target orientations)
    Δy ≈ 0 and the warp is purely horizontal, matching the original
    implementation. A non-zero pole_pa_deg here reflects camera roll in the
    image, NOT sub-observer latitude (B) — see spherical_derotation_warp_3d
    for the model that actually incorporates B.

    Args:
        image:                  2-D float [0, 1] array.
        dt_sec:                 (t_image - t_reference).total_seconds().
                                Positive = image taken AFTER reference.
        cx, cy:                 Disk center coordinates (pixels).
        disk_radius_px:         Disk semi-major axis (pixels), used as warp radius.
        period_hours:           Atmospheric rotation period in hours.
        scale:                  Empirical warp scale factor. 1.00 = full
                                theoretical spherical projection, confirmed
                                optimal for Jupiter via NCC sweep (see module
                                docstring). Saturn needs ~0.05-0.15 for
                                reasons not fully understood — see
                                DerotationConfig.warp_scale.
        flip_direction:         If True, negate the shift direction.
        pole_pa_deg:            Image-space equatorial/drift-axis position
                                angle in degrees — see note above, this is
                                NOT Horizons NP.ang fed in raw, and it is NOT
                                the same thing as sub-observer latitude (B):
                                pole_pa_deg=0 only means "the drift axis is
                                horizontal in this image", independent of
                                how tilted the planet's pole actually is.
                                Measured in image pixel coordinates (x right,
                                y DOWN): 0°=+x (right), +90°=+y (down).
                                Because y increases downward, a positive
                                angle sweeps CLOCKWISE as displayed on
                                screen — confirmed empirically by warping a
                                test point and reading its screen position
                                (see tests/test_reprojection.py); an earlier
                                revision of this line said "CCW", which was
                                wrong for this pixel-coordinate convention.
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


# ── True oblate-spheroid reprojection (sub-observer latitude B) ──────────────
#
# WinJUPOS-style true 3D reprojection, additive alternative to the linear
# spherical_derotation_warp() above. See project notes ("WinJUPOS-style
# de-rotation reprojection") for the full derivation and its adversarial
# verification. Summary:
#
#   Body-fixed parametrization (phi is an internal-only "parametric
#   latitude" — NOT planetographic or planetocentric; never expose/compare
#   it outside these functions):
#     x_b = Req*cos(phi)*cos(lam), y_b = Req*cos(phi)*sin(lam), z_b = Rpol*sin(phi)
#   Line of sight uses the Horizons sub-observer latitude B *directly*, no
#   planetographic->planetocentric conversion (Horizons' "ObsSub-LAT" is
#   already defined as the sub-observer point's surface-normal angle, which
#   by definition equals the line-of-sight angle).
#   Forward:  X=y_b, Y=-x_b*sinB+z_b*cosB, depth=x_b*cosB+z_b*sinB (visible iff >0),
#             then rotate (X,Y) by position angle P=pole_pa_deg using the
#             SAME rotation convention as spherical_derotation_warp's own
#             pole_pa decomposition (dx=X*cosP-Y*sinP, dy=X*sinP+Y*cosP) —
#             pole_pa_deg in this codebase is always an image-space angle,
#             measured directly from pixel data (equator_pa_from_disk_ellipse/
#             auto_detect_equator_pa), never a sky-frame Horizons quantity fed
#             in raw — so no separate "convert sky PA to image PA" step or
#             math-convention/pixel-row-down flip belongs here. An earlier
#             version of this rotation included such a flip on dy alone,
#             which silently turned the rotation into an improper one
#             (determinant -1, i.e. a reflection, not a rotation) — caught
#             via external code review + a determinant check, and confirmed
#             against real Jupiter data (the linear and 3D warps disagreed
#             on the y-direction of drift at nonzero pole_pa, by an amount
#             that grew with pole_pa and delta_lambda, and flip_pole_axis
#             had zero effect on it — exactly what an extra reflection
#             predicts, since flipping Y before a reflection doesn't undo
#             the reflection). Fixed; tests/test_reprojection.py's B=0
#             direction-match test now checks nonzero pole_pa too.
#   Inverse:  solve the resulting quadratic in z_b, picking the depth>0 root;
#             branches to a direct closed-form solve for |B|<1e-4 deg since
#             the general form divides by sin(B) (numerically unstable near
#             B=0 — confirmed empirically; the branch has <1e-6px round-trip
#             error even at B=1e-8 deg).
#
# flip_pole_axis is an escape hatch for a handedness/reflection ambiguity in
# this module's own parametrization (which of the two possible directions
# u_up=v_los×u_right was chosen, and which sign of the internal "parametric
# latitude" phi corresponds to the near vs far pole) — NOT the same thing as
# flipping the sign of B itself. Concretely, negating Y here
# (Y=-x_b*sinB+z_b*cosB -> -Y) is mathematically DIFFERENT from substituting
# B -> -B into that same formula (confirmed directly: -Y = x_b*sinB-z_b*cosB,
# whereas Y(B->-B) = x_b*sinB+z_b*cosB — the z_b term's sign differs). If you
# want to test "is B's sign wrong", pass -sub_observer_lat_deg, not
# flip_pole_axis=True; this flag is for testing an unrelated, genuinely
# free choice made during derivation that has no way to be pinned down from
# geometry alone. Resolved per-target empirically via NCC forward-prediction
# (see auto_detect_pole_axis_flip/_measure_derot_confidence), not assumed.
# This is a genuinely separate concern from the pole_pa rotation bug fixed
# above (which affected B=0 too, where flip_pole_axis has no B-related
# effect to resolve in the first place — confirmed via external code review
# 2026-08-10, which correctly pointed out the Y-negation != B-negation gap).

_SUB_OBS_LAT_SMALL_DEG = 1e-4  # below this |B|, use the direct (non-quadratic) solve


def _oblate_ortho_forward(
    phi: np.ndarray,
    lam: np.ndarray,
    sub_observer_lat_deg: float,
    pole_pa_deg: float,
    req_px: float,
    rpol_px: float,
    flip_pole_axis: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project body-fixed (phi, lam) to pixel offset (dx, dy) from disk centre.

    Returns (dx, dy, depth); depth>0 means the point is on the near
    (visible) side of the body. phi/lam/dx/dy may be scalars or arrays of
    matching shape.
    """
    B = math.radians(sub_observer_lat_deg)
    P = math.radians(pole_pa_deg)
    phi = np.asarray(phi, dtype=np.float64)
    lam = np.asarray(lam, dtype=np.float64)

    xb = req_px * np.cos(phi) * np.cos(lam)
    yb = req_px * np.cos(phi) * np.sin(lam)
    zb = rpol_px * np.sin(phi)

    sin_b, cos_b = math.sin(B), math.cos(B)
    X = yb
    Y = -xb * sin_b + zb * cos_b
    depth = xb * cos_b + zb * sin_b
    if flip_pole_axis:
        Y = -Y

    sin_p, cos_p = math.sin(P), math.cos(P)
    dx = X * cos_p - Y * sin_p
    dy = X * sin_p + Y * cos_p
    return dx, dy, depth


def _oblate_ortho_inverse(
    dx: np.ndarray,
    dy: np.ndarray,
    sub_observer_lat_deg: float,
    pole_pa_deg: float,
    req_px: float,
    rpol_px: float,
    flip_pole_axis: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unproject pixel offset (dx, dy) from disk centre to body-fixed (phi, lam).

    Returns (phi, lam, depth); phi/lam are NaN wherever no near-side
    (depth>0) solution exists. dx/dy may be scalars or arrays.
    """
    B = math.radians(sub_observer_lat_deg)
    P = math.radians(pole_pa_deg)
    dx = np.asarray(dx, dtype=np.float64)
    dy = np.asarray(dy, dtype=np.float64)

    sin_p, cos_p = math.sin(P), math.cos(P)
    X = dx * cos_p + dy * sin_p
    Y = -dx * sin_p + dy * cos_p
    if flip_pole_axis:
        Y = -Y

    sin_b, cos_b = math.sin(B), math.cos(B)
    req2 = req_px * req_px

    if abs(B) < math.radians(_SUB_OBS_LAT_SMALL_DEG):
        # B ~ 0: Y = z_b*cos_b directly (sin_b ~ 0) -> closed form, avoids
        # dividing by sin_b (unstable — see module notes above).
        zb = Y / cos_b
        xb_sq = req2 * (1.0 - (zb * zb) / (rpol_px * rpol_px)) - X * X
        xb = np.sqrt(np.clip(xb_sq, 0.0, None))
        depth = xb * cos_b + zb * sin_b
        depth = np.where(xb_sq < 0.0, -1.0, depth)  # no real solution -> invalid
    else:
        A = cos_b * cos_b + sin_b * sin_b * (req_px / rpol_px) ** 2
        Bq = -2.0 * Y * cos_b
        Cq = Y * Y + sin_b * sin_b * (X * X - req2)
        disc = Bq * Bq - 4.0 * A * Cq
        sq = np.sqrt(np.clip(disc, 0.0, None))
        z1 = (-Bq + sq) / (2.0 * A)
        z2 = (-Bq - sq) / (2.0 * A)
        x1 = (z1 * cos_b - Y) / sin_b
        x2 = (z2 * cos_b - Y) / sin_b
        depth1 = x1 * cos_b + z1 * sin_b
        depth2 = x2 * cos_b + z2 * sin_b
        valid1 = (disc >= 0.0) & (depth1 > 0.0)
        valid2 = (disc >= 0.0) & (depth2 > 0.0)
        pick1 = np.where(valid1 & valid2, depth1 >= depth2, valid1)
        zb    = np.where(pick1, z1, z2)
        xb    = np.where(pick1, x1, x2)
        depth = np.where(pick1, depth1, depth2)
        depth = np.where(valid1 | valid2, depth, -1.0)

    phi = np.arcsin(np.clip(zb / rpol_px, -1.0, 1.0))
    lam = np.arctan2(X, xb)
    invalid = depth <= 0.0
    phi = np.where(invalid, np.nan, phi)
    lam = np.where(invalid, np.nan, lam)
    return phi, lam, depth


def _reprojected_position(
    x: np.ndarray,
    y: np.ndarray,
    dt_sec: float,
    cx: float,
    cy: float,
    disk_radius_px: float,
    period_hours: float,
    sub_observer_lat_deg: float,
    pole_pa_deg: float,
    polar_equatorial_ratio_true: float,
    scale: float = 1.0,
    flip_direction: bool = False,
    flip_pole_axis: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shared core: unproject (x,y) -> shift body-fixed longitude by the
    same rotation the atmosphere undergoes over dt_sec -> reproject.

    Returns (new_x, new_y, valid) where valid is False wherever (x,y) had no
    near-side solution (point was off-disk) — new_x/new_y equal x/y there
    (no-op, matching the linear code's depth<=0 -> zero-drift fallback).
    Used both by spherical_derotation_warp_3d (vectorized, whole image) and
    by single-point callers (e.g. satellite/shadow smearing-position
    correction in satellite_composite.py) so this logic exists once.
    """
    # 5% padding, matching spherical_derotation_warp's warp_radius. This IS
    # a deliberate numerical regularization with a real geometric cost, not
    # a free improvement: it models a body 5% larger than the fitted disk,
    # a genuine (small) mismatch between the fitted apparent limb and the
    # projection surface. The tradeoff was tested directly rather than
    # assumed: removing the padding makes the INVERSE projection's
    # numerical sensitivity substantially WORSE near the limb — finite-
    # difference sensitivity of (phi, lam) to a 1px input perturbation at
    # r=0.995x the visible disk radius is ~0.11 rad/px with
    # req_px=disk_radius_px (no padding) vs ~0.03 rad/px with the 5%
    # padding kept, a ~3.7x difference — matching the earlier-flagged
    # "unprojection Jacobian diverges near the limb" concern from this
    # feature's original design phase. Given that, the small geometric
    # mismatch is judged worth paying for the numerical stability gain;
    # keeping the padding.
    req_px  = disk_radius_px * 1.05
    rpol_px = req_px * polar_equatorial_ratio_true

    dx0 = np.asarray(x, dtype=np.float64) - cx
    dy0 = np.asarray(y, dtype=np.float64) - cy
    phi, lam, depth = _oblate_ortho_inverse(
        dx0, dy0, sub_observer_lat_deg, pole_pa_deg, req_px, rpol_px,
        flip_pole_axis=flip_pole_axis,
    )

    period_sec = period_hours * 3600.0
    delta_lambda = (dt_sec / period_sec) * 2.0 * math.pi
    # NOTE: this sign is the OPPOSITE of spherical_derotation_warp's internal
    # `sign = -1 if flip_direction else 1` — that one scales depth-based
    # drift directly (a different parametrization), while this one shifts a
    # body-fixed longitude recovered by *unprojecting the output position*.
    # Empirically calibrated (see tests/test_reprojection.py's B=0 ground-
    # truth check) so that flip_direction=False here reproduces the exact
    # same raw-pixel -> de-rotated-output shift direction as the validated
    # linear warp's own flip_direction=False, not assumed from the algebra
    # alone (this project has a history of exactly this kind of sign error).
    sign = 1.0 if flip_direction else -1.0
    lam_src = lam + sign * scale * delta_lambda

    dx1, dy1, source_depth = _oblate_ortho_forward(
        phi, lam_src, sub_observer_lat_deg, pole_pa_deg, req_px, rpol_px,
        flip_pole_axis=flip_pole_axis,
    )

    # A point visible in the OUTPUT orientation is not necessarily visible
    # in the SOURCE orientation after shifting its longitude by delta_lambda
    # — for large enough rotation it can end up on the far side at the
    # source time, which orthographic projection would otherwise silently
    # fold onto the same screen position as a genuine near-side point.
    # Reject that case too (source_depth>0), not just the output-side check.
    # (Measured on real Jupiter data at this module's typical per-frame
    # rotation, ~10°: affects ~0.06% of on-disk pixels, all within ~2px of
    # the fitted limb — narrow, but real.)
    valid = np.isfinite(phi) & (source_depth > 0.0)
    # Invalid points map to a sentinel far outside any real image's coordinate
    # range, so a caller using this for cv2.remap's map_x/map_y with
    # BORDER_CONSTANT gets background there instead of silently re-sampling
    # whatever raw content happens to sit at that screen position (which is
    # not the same body location any more). Empirically, cv2.remap's cubic
    # kernel already returns a clean 0 at exactly -1.0 (one pixel out), but
    # values a further pixel or two out (e.g. -1.5) can show small nonzero
    # ringing from the interpolation kernel's negative side-lobes reaching
    # across the BORDER_CONSTANT edge — using a sentinel far from any
    # possible kernel support removes that dependence on interpolation-
    # kernel-width implementation details entirely. spherical_derotation_warp_3d
    # also re-masks the final warped output directly with `valid` as a second,
    # independent safeguard (see there). _reprojection_point_shift()
    # (single-point callers) checks `valid` before ever reading new_x/new_y,
    # so this sentinel never leaks into its (dx, dy) contract either way.
    _INVALID_MAP_COORD = -1.0e4
    new_x = np.where(valid, cx + dx1, _INVALID_MAP_COORD)
    new_y = np.where(valid, cy + dy1, _INVALID_MAP_COORD)
    return new_x, new_y, valid


def _reprojection_point_shift(
    x: float,
    y: float,
    dt_sec: float,
    cx: float,
    cy: float,
    disk_radius_px: float,
    period_hours: float,
    sub_observer_lat_deg: float,
    pole_pa_deg: float,
    polar_equatorial_ratio_true: float,
    scale: float = 1.0,
    flip_direction: bool = False,
    flip_pole_axis: bool = False,
) -> Tuple[float, float]:
    """Single-point convenience wrapper around _reprojected_position().

    Returns (dx, dy): the shift a point currently at pixel (x,y) undergoes
    under the true reprojection over dt_sec. (0.0, 0.0) if (x,y) is off-disk
    (no warp applies there) — mirrors spherical_derotation_warp's implicit
    zero-drift behaviour outside disk_radius_px.
    """
    new_x, new_y, valid = _reprojected_position(
        x, y, dt_sec, cx, cy, disk_radius_px, period_hours,
        sub_observer_lat_deg, pole_pa_deg, polar_equatorial_ratio_true,
        scale=scale, flip_direction=flip_direction, flip_pole_axis=flip_pole_axis,
    )
    if not bool(valid):
        return 0.0, 0.0
    return float(new_x) - x, float(new_y) - y


def spherical_derotation_warp_3d(
    image: np.ndarray,
    dt_sec: float,
    cx: float,
    cy: float,
    disk_radius_px: float,
    period_hours: float,
    sub_observer_lat_deg: float,
    pole_pa_deg: float,
    polar_equatorial_ratio_true: float,
    scale: float = 1.0,
    flip_direction: bool = False,
    flip_pole_axis: bool = False,
) -> np.ndarray:
    """True oblate-spheroid orthographic reprojection de-rotation warp.

    Additive alternative to spherical_derotation_warp() — NOT a drop-in
    replacement, and not called unless DerotationConfig.use_true_reprojection
    is enabled. Unlike the linear warp, this correctly incorporates the
    sub-observer latitude (see module notes above for the full derivation).

    Args mirror spherical_derotation_warp() except sub_observer_lat_deg
    (Horizons "ObsSub-LAT", quantity 14 — see query_horizons_sub_observer_lat)
    replaces nothing (pole_pa_deg is still required, same source as before)
    and polar_equatorial_ratio_true must be the planet's TRUE physical
    Rpol/Req (a per-target constant, e.g. Saturn=0.9021) — NOT the apparent
    fitted ellipse aspect ratio used by the linear warp, which is
    contaminated by B-foreshortening once B is modelled explicitly.

    Returns: warped float [0, 1] array, same shape as input.
    """
    if abs(scale - 1.0) > 1e-6:
        # Unlike the linear warp (a first-order approximation where an
        # empirical scale legitimately absorbs some of the approximation
        # error), this function computes the geometrically exact rotation
        # for the given Δt/period — there is no first-order error left for
        # `scale` to compensate for. A value other than 1.0 here (e.g.
        # Saturn's empirically-tuned 0.05-0.15 for the linear warp) is a
        # signal that something else is off — timestamps, rotation period,
        # sign/orientation convention, or the target's real atmospheric
        # motion not matching rigid-body rotation — not a normal calibration
        # knob for this warp. Surfaced once (Python's default warning filter
        # dedupes by call site) rather than silently reusing the linear
        # warp's tuned value here.
        warnings.warn(
            f"spherical_derotation_warp_3d called with scale={scale:.3f} "
            "(!= 1.0). True reprojection has no first-order-approximation "
            "error for `scale` to absorb — treat a non-1.0 value here as a "
            "geometry/timing/convention diagnostic signal, not a normal "
            "calibration parameter carried over from the linear warp.",
            stacklevel=2,
        )
    h, w = image.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)

    new_x, new_y, valid = _reprojected_position(
        xx, yy, dt_sec, cx, cy, disk_radius_px, period_hours,
        sub_observer_lat_deg, pole_pa_deg, polar_equatorial_ratio_true,
        scale=scale, flip_direction=flip_direction, flip_pole_axis=flip_pole_axis,
    )
    map_x = new_x.astype(np.float32)
    map_y = new_y.astype(np.float32)

    # Mixed interpolation: same CUBIC-interior / LINEAR-edge feather blend as
    # spherical_derotation_warp(), for visual-quality parity near the limb.
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
    _interp_feather_px = 12.0
    dist_from_center = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.float32)
    w_cubic = np.clip(
        (disk_radius_px - dist_from_center) / _interp_feather_px, 0.0, 1.0
    )
    if warped_cubic.ndim == 3:
        w_cubic = w_cubic[:, :, np.newaxis]
    warped = warped_cubic * w_cubic + warped_linear * (1.0 - w_cubic)

    # Second, independent safeguard against far-side leakage (see the
    # sentinel note in _reprojected_position): explicitly zero out pixels
    # _reprojected_position marked invalid, rather than relying solely on
    # cv2.remap's BORDER_CONSTANT behaviour at the sentinel coordinate.
    invalid = ~valid
    if np.any(invalid):
        if warped.ndim == 3:
            warped[invalid, :] = 0.0
        else:
            warped[invalid] = 0.0

    return np.clip(warped, 0.0, 1.0)


def auto_detect_pole_axis_flip(
    frames: List[np.ndarray],
    dt_sec_list: List[float],
    cx: float,
    cy: float,
    disk_radius_px: float,
    period_hours: float,
    sub_observer_lat_deg: float,
    warp_scale: float = 1.00,
    pole_pa_deg: float = 0.0,
    polar_equatorial_ratio_true: float = 1.0,
    flip_direction: bool = False,
) -> Tuple[bool, float, float]:
    """Detect flip_pole_axis for the true 3D reprojection warp, from real
    atmospheric feature drift — same forward-prediction NCC technique as
    auto_detect_ns_flip(), extended to the reprojection's own sign
    ambiguity (see spherical_derotation_warp_3d's module notes).

    flip_direction should already be resolved (e.g. via auto_detect_ns_flip)
    before calling this — this only searches the ORTHOGONAL ambiguity that
    is specific to modelling sub-observer latitude B explicitly and does
    not exist in the linear warp at all.

    Returns:
        (flip_pole_axis, score_false, score_true)
        Defaults to (False, 0.0, 0.0) when ambiguous (|Δcorr| < 0.001) or
        too few/degenerate frames — matching auto_detect_ns_flip's
        fail-safe behaviour.
    """
    if len(frames) < 2:
        print("  [flip_pole_axis] fewer than 2 frames — defaulting to False")
        return False, 0.0, 0.0

    dts = np.array(dt_sec_list, dtype=np.float64)
    i_min = int(np.argmin(dts))
    i_max = int(np.argmax(dts))
    if i_min == i_max:
        print("  [flip_pole_axis] all frames at same time — defaulting to False")
        return False, 0.0, 0.0

    dt_span = float(dts[i_max] - dts[i_min])

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

    scores: dict = {}
    for flip_pole_axis in (False, True):
        predicted = spherical_derotation_warp_3d(
            f_early, +dt_span,
            cx, cy, disk_radius_px, period_hours,
            sub_observer_lat_deg=sub_observer_lat_deg,
            pole_pa_deg=pole_pa_deg,
            polar_equatorial_ratio_true=polar_equatorial_ratio_true,
            scale=warp_scale,
            flip_direction=flip_direction,
            flip_pole_axis=flip_pole_axis,
        )
        tgt_pixels = predicted[disk_mask].astype(np.float64)
        tgt_std = float(tgt_pixels.std())
        if ref_std > 1e-6 and tgt_std > 1e-6:
            scores[flip_pole_axis] = float(np.corrcoef(ref_pixels, tgt_pixels)[0, 1])
        else:
            scores[flip_pole_axis] = 0.0

    delta = scores[True] - scores[False]
    flip_pole_axis = delta > 0.0
    confidence = abs(delta)
    angle_deg = abs(dt_span) / (period_hours * 3600.0) * 360.0
    print(
        f"  [flip_pole_axis] Δt={dt_span:.0f}s ({angle_deg:.2f}°)  "
        f"False_corr={scores[False]:.5f}  True_corr={scores[True]:.5f}  "
        f"→ flip_pole_axis={flip_pole_axis} (|Δcorr|={confidence:.5f})"
    )
    if confidence < 0.001:
        print("  [flip_pole_axis] low confidence — defaulting to False")
        return False, float(scores[False]), float(scores[True])
    return flip_pole_axis, float(scores[False]), float(scores[True])


# ── Pole PA auto-detection ────────────────────────────────────────────────────

def auto_detect_equator_pa(
    frames: List[np.ndarray],
    cx: float,
    cy: float,
    disk_radius_px: float,
) -> float:
    """Estimate the image-space equatorial/drift-axis angle from Jupiter belts.

    Builds a gradient-angle histogram over the inner disk (0.75R) using two
    complementary wavelet-sharpened views of the frames. The dominant
    belt-edge gradient direction is approximately aligned with the projected
    polar axis (belts run perpendicular to the rotation axis); this function
    rotates that gradient direction by 90° to return the equatorial/drift-
    axis angle used by the de-rotation warp (its pole_pa_deg parameter).

        result = 0°   → belts horizontal, drift axis horizontal
        result = +θ   → belts tilted θ CW from horizontal, as displayed
                        (pixel coordinates, y-down — verified empirically
                        by rotating a synthetic belted image with a known,
                        independently-defined direction and checking the
                        sign of the detected angle; see
                        tests/test_reprojection.py)
        result = −θ   → belts tilted θ CCW from horizontal, as displayed

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
    warp_scale: float = 1.00,
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
        so this function returns flip_direction=False in both cases.
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
        warp_scale:             Empirical spherical warp scale (default 1.00).
        pole_pa_deg:            Image-space pole PA from auto_detect_equator_pa().
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


def equator_pa_from_disk_ellipse(image: np.ndarray) -> Optional[float]:
    """Estimate the image-space equatorial/drift-axis angle from the disk
    ellipse major axis.

    For oblate planets the equatorial (major) axis IS the drift direction —
    this is used directly as the warp's pole_pa_deg (pole_pa_deg=0° →
    horizontal drift).

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
    at the geometric disk edge (radius). Available utility for the intended
    purpose below — NOT currently called anywhere in the pipeline (confirmed
    via repo-wide grep, 2026-08-10); an earlier version of this docstring
    claimed it was already applied, which was inaccurate.

    Intended use: applied to each warped frame before stacking, to prevent
    background zeros from bleeding into the limb average, which would
    create a darkening band that wavelet sharpening amplifies.

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

    NOTE on true-reprojection far-side pixels (external review, 2026-08-10):
    spherical_derotation_warp_3d() correctly zeroes out pixels that rotate
    to the far side by the source time (see its own docstring), but THIS
    function has no per-pixel validity mask — a zeroed far-side pixel in
    one frame is averaged in as a real 0 alongside valid frames, which
    could in principle darken the limb slightly. Deliberately not
    addressed here: measured impact on real Jupiter data (this module's
    typical ~10° per-frame rotation) is ~0.06% of on-disk pixels, all
    within ~2px of the limb — swamped by the existing CUBIC/LINEAR feather
    blend at that same location. Implementing proper validity-weighted
    stacking would mean threading per-pixel masks through every caller of
    this function, a real architecture change, for a currently-immeasurable
    benefit — left as a documented option, not implemented speculatively.
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
    shared_shape: Optional[PlanetShape] = None,
    filter_pose: Optional[FilterPose] = None,
    shared_radius_px: Optional[float] = None,
    weight_power: float = 1.0,
    use_true_reprojection: bool = False,
    sub_observer_lat_deg: float = 0.0,
    true_polar_equatorial_ratio: float = 1.0,
    flip_pole_axis: bool = False,
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
                        Must match the flip_direction detected by auto_detect_ns_flip().
        shared_shape:   Optional oblateness/orientation (aspect_ratio,
                        equator_pa_deg) to use ONLY when this filter's own
                        reference-frame detection couldn't determine shape
                        reliably (see resolve_shared_shape()).
        filter_pose:    Optional (cx, cy) to use ONLY when this filter's own
                        reference-frame detection failed outright (see
                        resolve_filter_pose()) — never used to override a
                        successful independent pose measurement.
        shared_radius_px: Optional window-wide consensus semi_major (see
                        resolve_shared_radius()), substituted for this
                        filter's own value ONLY when its own value is
                        already within _RADIUS_SHARE_REL_TOL of it — i.e.
                        this reconciles per-filter Otsu-noise (typically
                        1-3px on real Jupiter data) that would otherwise
                        each cut off the warp's valid region at a slightly
                        different radius (see resolve_shared_radius()'s
                        docstring). A filter with a genuinely different
                        apparent size (e.g. Saturn's CH4 band) stays on its
                        own measurement — this is a reconciliation of
                        near-identical measurements, not a size override.
        weight_power:   Exponent applied to norm_score weights before
                        stacking (see quality_weighted_stack). 1.0 (default)
                        = unchanged linear blend.
        use_true_reprojection: If True, use spherical_derotation_warp_3d()
                        instead of the linear spherical_derotation_warp() —
                        see DerotationConfig.use_true_reprojection. Default
                        False (linear warp, unchanged behaviour).
        sub_observer_lat_deg: Horizons "ObsSub-LAT" (quantity 14) — only
                        used when use_true_reprojection=True.
        true_polar_equatorial_ratio: Planet's TRUE physical Rpol/Req — only
                        used when use_true_reprojection=True (replaces the
                        apparent ellipse-fit ratio the linear warp uses).
        flip_pole_axis: Sign-ambiguity escape hatch for the reprojection —
                        only used when use_true_reprojection=True.

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
    _ref_fit = _find_disk_center_impl(_ref_lum)
    _rcx, _rcy, ref_semi_a, _rsemi_b, _rangle, _rconf, _rshape_ok = _ref_fit
    if _rconf <= 0.0 and filter_pose is not None:
        # This filter's own detection failed outright — use the pose
        # registered against a sibling filter's frame instead. Its own
        # ref_semi_a at this point is the raw disk+ring fit (confidence==0.0
        # means exactly that, per _find_disk_center_impl's docstring) — if
        # the caller also supplied a probe-derived size estimate (only set
        # in this exact fallback case; see resolve_filter_pose), use that
        # instead of the known-bad raw fit.
        ref_cx, ref_cy = filter_pose.center_x_px, filter_pose.center_y_px
        if filter_pose.semi_major_px is not None and filter_pose.semi_major_px > 0:
            ref_semi_a = filter_pose.semi_major_px
        _geometry_source = "pose_registered"
    else:
        ref_cx, ref_cy = _rcx, _rcy
        _geometry_source = "pose_independent"

    ref_semi_b = _rsemi_b
    if (
        shared_radius_px is not None
        and ref_semi_a > 0
        and abs(ref_semi_a - shared_radius_px) <= _RADIUS_SHARE_REL_TOL * shared_radius_px
    ):
        # This filter's own radius already agrees with the window-wide
        # consensus (resolve_shared_radius()) — snap to the exact shared
        # value so every filter's warp goes invalid beyond the SAME radius,
        # instead of each filter's own Otsu-threshold noise (typically
        # 1-3px) cutting the valid region off at a slightly different
        # radius per filter — confirmed as the direct cause of a colour
        # fringe at the limb once composited (2026-08-11 investigation).
        # Rescale semi_b to preserve this filter's own measured aspect
        # ratio — only the overall size is reconciled, not the oblateness.
        if _rsemi_b > 0:
            ref_semi_b = _rsemi_b * (shared_radius_px / ref_semi_a)
        ref_semi_a = shared_radius_px
        _geometry_source += "+radius_shared"

    if not _rshape_ok and shared_shape is not None:
        # Own detection couldn't determine oblateness (e.g. ring crossing
        # the disk) — borrow orientation/aspect (this overrides any
        # radius-share rescaling above, since shape itself is untrustworthy
        # here regardless of size agreement).
        ref_semi_b = ref_semi_a * shared_shape.aspect_ratio
        _geometry_source += "+shape_shared"
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
        # When shared_shape is set (ring detected somewhere in this window),
        # per-frame ellipse-fit disk detection is untrustworthy for at least
        # one filter in the window — most acutely CH4, whose brightness
        # inversion (or a ring band crossing the disk) can make
        # find_disk_center() silently return a plausible-looking but wrong
        # centre (no exception, so the abs(dx)<=15 sanity gate below can't
        # catch it). Content-based phase correlation against the reference
        # frame of the SAME filter doesn't require identifying "the disk" at
        # all, so it isn't fooled by which feature is brighter — use it
        # instead whenever this window has any ring involvement.
        #
        # MUST be cropped to the disk ROI first: a full-frame phaseCorrelate
        # here reintroduces the exact ring-pollution bug fixed in
        # composite.align_channels() (2026-08-09) — confirmed on real data,
        # this returned -14.19px on a Saturn R-channel frame pair (vs. the
        # true ~0.5px, measured both via find_disk_center-on-target and via
        # ROI-cropped correlation). The ring is large, elongated, and its
        # exact appearance isn't perfectly frame-to-frame stable, so it can
        # dominate an unmasked correlation regardless of which filter it is.
        _raw_lum = _to_luminance(img)
        try:
            if shared_shape is not None:
                _rh, _rw = _ref_lum.shape[:2]
                _rys = max(0, int(ref_cy - ref_semi_a)); _rye = min(_rh, int(ref_cy + ref_semi_a))
                _rxs = max(0, int(ref_cx - ref_semi_a)); _rxe = min(_rw, int(ref_cx + ref_semi_a))
                if (_rye - _rys) > 10 and (_rxe - _rxs) > 10:
                    _dx, _dy = subpixel_align(
                        _ref_lum[_rys:_rye, _rxs:_rxe], _raw_lum[_rys:_rye, _rxs:_rxe]
                    )
                else:
                    _dx, _dy = subpixel_align(_ref_lum, _raw_lum)
                if abs(_dx) <= 15.0 and abs(_dy) <= 15.0:
                    _pre_warp_shifts[row["stem"]] = (_dx, _dy)
            else:
                _cx_i, _cy_i, _semi_i, *_ = find_disk_center(_raw_lum)
                if _semi_i >= 5:
                    _dx = float(ref_cx - _cx_i)
                    _dy = float(ref_cy - _cy_i)
                    if abs(_dx) <= 15.0 and abs(_dy) <= 15.0:
                        _pre_warp_shifts[row["stem"]] = (_dx, _dy)
        except Exception:
            pass

        dt_sec = (row["timestamp"] - t_reference).total_seconds()
        if use_true_reprojection:
            warped = spherical_derotation_warp_3d(
                img, dt_sec, ref_cx, ref_cy, ref_semi_a,
                period_hours=period_hours,
                sub_observer_lat_deg=sub_observer_lat_deg,
                pole_pa_deg=pole_pa_deg,
                polar_equatorial_ratio_true=true_polar_equatorial_ratio,
                scale=warp_scale,
                flip_direction=flip_direction,
                flip_pole_axis=flip_pole_axis,
            )
        else:
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
        "geometry_source":       _geometry_source,
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
    flip_direction: bool = False,
    out_dir: Optional[Path] = None,
    weight_power: float = 1.0,
    use_true_reprojection: bool = False,
    sub_observer_lat_deg: float = 0.0,
    true_polar_equatorial_ratio: float = 1.0,
    flip_pole_axis: bool = False,
) -> Dict[str, Tuple[Optional[Path], dict]]:
    """De-rotate and stack all filters in a single time window.

    Args:
        window:           Window dict from Step 4 (center_time + per_filter data).
        required_filters: Filters to process.
        period_hours:     System II rotation period.
        warp_scale:       Spherical warp scale (passed through to derotate_filter).
        align:            Sub-pixel alignment between frames.
        flip_direction:   Atmospheric-rotation sign from auto_detect_ns_flip()
                          (renamed from flip_ns 2026-08-10 — this is the
                          de-rotation longitude-drift sign, NOT the same
                          thing as the satellite tracker's flip_ns, a
                          different, unrelated camera-orientation flag on
                          SatelliteConfig/SatelliteTracker that happened to
                          share the name). Passed straight through as
                          flip_direction to spherical_derotation_warp() /
                          spherical_derotation_warp_3d().
        out_dir:          If provided, save TIF files here.
        weight_power:     Exponent applied to norm_score weights in the
                          per-filter stack (see quality_weighted_stack). 1.0
                          (default) = original linear blend, unchanged for
                          every existing caller.
        use_true_reprojection, sub_observer_lat_deg,
        true_polar_equatorial_ratio, flip_pole_axis:
                          Passed straight through to derotate_filter() — see
                          its docstring. Default use_true_reprojection=False
                          leaves every existing caller's behaviour unchanged.

    Returns:
        {filter: (output_path_or_None, log_dict)}
    """
    t_ref = window["center_time"]
    results: Dict[str, Tuple[Optional[Path], dict]] = {}

    # ── Ring-aware shared shape/pose ───────────────────────────────────────────
    # Each filter's disk centre/radius is normally fit independently (in
    # derotate_filter, from that filter's own reference frame). For a ringed
    # planet this is unreliable: find_disk_center()'s disk-core isolation runs
    # per filter on a blob whose SNR/contrast differs by filter, so R/G/B/IR
    # can each converge on a measurably different centre/radius/oblateness for
    # the *same* instant (confirmed on real Saturn data: fitted semi_b/semi_a
    # ranged 0.84-0.93 across IR/R/G/B in one window). Some filters can also
    # fail outright (e.g. Saturn's CH4 band, or a ring band crossing directly
    # in front of the disk — see _find_disk_center_impl's fallback chain).
    #
    # Fix: fit every candidate filter's reference frame once, then let
    # resolve_shared_shape()/resolve_filter_pose() decide — by confidence, not
    # by filter name — which filter's oblateness/orientation to share with
    # filters that couldn't determine their own, and which filter (if any) to
    # register a totally-failed filter's pose against. Shape/orientation and
    # POSE are always this filter's own measurement unless its own detection
    # failed outright — see the PlanetShape/FilterPose module docstring for
    # why those are never borrowed the way the old single shared_geometry
    # tuple used to.
    # shared_shape/filter_pose stay unused for ringless targets (Jupiter,
    # Mars, Venus): every filter's own fit already has shape_reliable=True
    # and confidence=1.0. resolve_shared_radius() below is a SEPARATE, much
    # narrower reconciliation added 2026-08-11 (see its docstring) — it DOES
    # apply to ringless targets too, since it only nudges each filter's
    # semi_major toward the group's own consensus when they already roughly
    # agree, rather than borrowing a value across genuinely different fits.
    _fits: Dict[str, Tuple[float, float, float, float, float, float, bool]] = {}
    _lums: Dict[str, np.ndarray] = {}
    for cfilt in required_filters:
        c_included = window.get("per_filter", {}).get(cfilt, {}).get("included")
        if not c_included:
            continue
        # Mirror derotate_filter()'s own reference-row selection exactly (same
        # quality-threshold drop, then nearest-to-t_ref) so this is the same
        # frame that filter will actually use as its reference.
        if min_quality_threshold > 0.0:
            c_filtered = [r for r in c_included if float(r["norm_score"]) >= min_quality_threshold]
            c_included = c_filtered if c_filtered else c_included
        c_ref_row = min(c_included, key=lambda r: abs((r["timestamp"] - t_ref).total_seconds()))
        try:
            c_raw = image_io.read_tif(c_ref_row["path"])
            if color_mode:
                if c_raw.ndim == 2:
                    c_raw = np.stack([c_raw] * 3, axis=2)
                c_lum = _to_luminance(c_raw)
            else:
                c_lum = c_raw if c_raw.ndim == 2 else c_raw.mean(axis=2).astype(np.float32)
        except Exception as exc:
            print(f"    [geometry] {cfilt} reference frame read failed ({exc}) — excluded from shape/pose resolution")
            continue
        _lums[cfilt] = c_lum
        _fits[cfilt] = _find_disk_center_impl(c_lum)

    shared_shape_result = resolve_shared_shape(_fits)
    shared_shape: Optional[PlanetShape] = None
    _shape_source: Optional[str] = None
    if shared_shape_result is not None:
        shared_shape, _shape_source = shared_shape_result
        print(
            f"    [geometry] sharing disk shape from {_shape_source} "
            f"(aspect_ratio={shared_shape.aspect_ratio:.3f} equator_pa={shared_shape.equator_pa_deg:.1f}°)"
        )

    shared_radius_px = resolve_shared_radius(_fits)
    if shared_radius_px is not None:
        print(f"    [geometry] window radius consensus = {shared_radius_px:.2f}px "
              f"(filters within {_RADIUS_SHARE_REL_TOL*100:.0f}% snap to it)")

    filter_poses: Dict[str, FilterPose] = {}
    for cfilt, fit in _fits.items():
        probe_pose = filter_poses.get(_shape_source) if _shape_source else None
        if probe_pose is None and _shape_source is not None:
            _sf = _fits[_shape_source]
            probe_pose = FilterPose(_sf[0], _sf[1])
        pose, method = resolve_filter_pose(
            fit,
            lum=_lums.get(cfilt),
            probe_lum=_lums.get(_shape_source) if _shape_source else None,
            probe_pose=probe_pose,
            probe_semi_major_px=_fits[_shape_source][2] if _shape_source else None,
        )
        filter_poses[cfilt] = pose
        if method != "own_detection":
            print(f"    [geometry] {cfilt} pose {method} (own detection failed outright)")

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
                flip_direction=flip_direction,
                shared_shape=shared_shape,
                filter_pose=filter_poses.get(filt),
                shared_radius_px=shared_radius_px,
                weight_power=weight_power,
                use_true_reprojection=use_true_reprojection,
                sub_observer_lat_deg=sub_observer_lat_deg,
                true_polar_equatorial_ratio=true_polar_equatorial_ratio,
                flip_pole_axis=flip_pole_axis,
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
