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
from pipeline.modules.wavelet import coverage_to_confidence
from pipeline.modules.limb_darkening import evaluate_limb_darkening_curve, LimbDarkeningFit

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


def _subpixel_ray_edge(
    profile: np.ndarray,
    dr: float,
    smooth_sigma: float = 1.5,
    margin: Optional[int] = None,
) -> Optional[float]:
    """Steepest-gradient sub-pixel edge index along one already-sampled
    radial brightness profile (uniform sample spacing `dr`). Shared by
    `_gradient_disk_r()` and `_robust_ellipse_refit()` (2026-08-15 extract
    -- previously this exact smooth/gradient/argmin/parabolic-interpolation
    sequence was duplicated inline in `_gradient_disk_r`; factored out here
    so the two callers can't drift out of sync when one is tuned and the
    other isn't).

    `margin`: exclude this many samples at each end of the profile from the
    argmin search (default: len(profile)//20, min 3) -- guards against a
    convolution edge-padding artifact winning the global argmin instead of
    a genuine interior gradient minimum (found 2026-08-15 diagnosing why an
    early version of the independent ground-truth measurement script
    returned zero valid rays on real ring-adjacent Saturn data: the profile
    kept decreasing all the way to the sampled window's own boundary, and
    the "same"-mode convolution's edge replication made the discrete
    gradient there spuriously the most negative point in the array).

    Returns the sub-pixel sample INDEX (not a radius -- caller multiplies by
    dr and adds the corresponding r_vals[0]/r_start), or None if no genuine
    interior minimum exists (result would land in the excluded margin).
    """
    smoothed = _gaussian_filter1d_np(profile, sigma=smooth_sigma)
    grad = np.gradient(smoothed, dr)
    if margin is None:
        margin = max(3, len(grad) // 20)
    if margin <= 0:
        search = grad
    else:
        if len(grad) <= 2 * margin:
            return None
        search = grad[margin:-margin]
    idx = margin + int(np.argmin(search))
    if idx <= 0 or idx >= len(grad) - 1:
        return None
    y0, y1, y2 = grad[idx - 1], grad[idx], grad[idx + 1]
    denom = 2.0 * (y2 - 2.0 * y1 + y0)
    sub = -(y2 - y0) / denom if abs(denom) > 1e-12 else 0.0
    return idx + sub


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
        # margin=0 matches this function's own long-established convention
        # (only reject when the argmin lands EXACTLY on index 0 or the last
        # index, not a wider margin) -- preserved exactly for byte-identical
        # behavior; _robust_ellipse_refit (a new, different caller) uses
        # _subpixel_ray_edge's own default margin instead, since it targets
        # a different (narrower, closer-in) search window where the wider
        # default margin matters (see that function's docstring).
        sub_idx = _subpixel_ray_edge(profile, dr, smooth_sigma=smooth_sigma, margin=0)
        if sub_idx is None:
            continue
        edge_radii.append(float(r_vals[0] + sub_idx * dr))

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


def _ray_limb_edge(
    image: np.ndarray,
    cx: float,
    cy: float,
    semi_a: float,
    semi_b: float,
    angle_deg: float,
    theta_deg: float,
    search_frac: Tuple[float, float] = (0.90, 1.10),
    n_samples: int = 300,
    smooth_sigma: float = 1.5,
) -> Optional[float]:
    """Steepest-gradient limb radius at image-frame angle `theta_deg`,
    searched in a NARROW window around the already-fitted ellipse's own
    boundary radius in that direction. Used by `_robust_ellipse_refit()` to
    re-measure the residual between a seed ellipse (`find_disk_center()`'s
    output) and the true limb, one ray at a time.

    The narrow `search_frac` (vs. `_gradient_disk_r`'s wider 0.75-1.30) is
    deliberate: this only needs to resolve a small residual on top of an
    already-refined seed, not re-find the limb from scratch. A wide window
    risks reaching Saturn's ring re-brightening the profile past the true
    dark gap (see `_subpixel_ray_edge`'s margin-exclusion docstring for the
    related boundary-artifact issue this also avoids).

    Returns the true edge radius (distance from (cx, cy)), or None if the
    ray exits the image or `_subpixel_ray_edge` finds no genuine interior
    gradient minimum.
    """
    theta = math.radians(theta_deg)
    dx_u, dy_u = math.cos(theta), math.sin(theta)
    ang = math.radians(angle_deg)
    cos_a, sin_a = math.cos(ang), math.sin(ang)
    dxr = cos_a * dx_u + sin_a * dy_u
    dyr = -sin_a * dx_u + cos_a * dy_u
    r_ell = 1.0 / math.sqrt((dxr / semi_a) ** 2 + (dyr / semi_b) ** 2)

    r_vals = np.linspace(r_ell * search_frac[0], r_ell * search_frac[1], n_samples)
    dr = r_vals[1] - r_vals[0]
    xs = cx + r_vals * dx_u
    ys = cy + r_vals * dy_u
    h, w = image.shape[:2]
    if (xs < 0).any() or (xs >= w - 1).any() or (ys < 0).any() or (ys >= h - 1).any():
        return None
    profile = _bilinear_interp(image, ys, xs)
    sub_idx = _subpixel_ray_edge(profile, dr, smooth_sigma=smooth_sigma)
    if sub_idx is None:
        return None
    return float(r_vals[0] + sub_idx * dr)


def _robust_ellipse_refit(
    image: np.ndarray,
    cx: float,
    cy: float,
    semi_a: float,
    semi_b: float,
    angle_deg: float,
    n_rays: int = 72,
    search_frac: Tuple[float, float] = (0.90, 1.10),
    n_samples: int = 300,
    smooth_sigma: float = 1.5,
    n_iter: int = 3,
    outlier_sigma: float = 2.5,
    min_keep: int = 20,
    min_arc_span_deg: float = 180.0,
) -> Optional[Tuple[float, float, float, float, float, int]]:
    """Refine a seed ellipse (from `find_disk_center()`) against the true
    photometric limb via an iteratively-reweighted (MAD-based outlier
    rejection) refit over `n_rays` sub-pixel edge measurements.

    Added 2026-08-15 to address a measured ~0.5-0.9px ASYMMETRIC error
    between `find_disk_center()`'s ellipse fit and the true limb (root cause
    of the gray halo / white-rim wavelet artifact trade-off documented in
    `SATURN_RING_WAVELET_STATUS_2026-08-15.md`). Validated via
    `experiments/scratch_globe_fit_asymmetry_diagnosis.py` on real data:
    dramatically reduces worst-case fit-vs-true-limb residual on Jupiter
    (ringless target, 9.04px->2.26px and 7.84px->2.49px across two windows)
    by rejecting rays hijacked by local albedo features (e.g. cloud belts)
    that compete with the true limb gradient.

    **Validated for ringless targets only.** On Saturn, MAD-based rejection
    keeps ~71-72/72 rays (essentially rejects nothing) because ring-crossing
    contamination is a large *contiguous* angular arc (~40% of rays) rather
    than scattered points -- point-wise robust statistics can't distinguish
    "consensus" from "contaminated majority" in that regime. Callers should
    NOT rely on this to help Saturn; it is expected to be a near-no-op there
    (see the same status doc's "Saturn root cause" section for the full set
    of ruled-out hypotheses: ring-ray exclusion alone, hybrid, frame-count,
    quadrupole/aspect-scale).

    This is a NEW, additive function -- does not modify `find_disk_center()`,
    `_find_disk_center_impl()`, or `_gradient_disk_r()`'s behavior. Callers
    must treat a `None` return as "keep the seed ellipse unchanged"; this
    function never returns something worse than the seed by construction
    (insufficient/too-narrow-arc surviving rays -> None, not a bad fit).

    Returns (cx, cy, semi_a, semi_b, angle_deg, n_kept), or None if fewer
    than `min_keep` rays survive or the surviving rays don't span at least
    `min_arc_span_deg` of arc (an ellipse fit from a narrow arc is
    numerically ill-conditioned and not trustworthy).
    """
    thetas_deg = np.arange(0.0, 360.0, 360.0 / n_rays)
    pts = []
    kept_thetas = []
    for theta_deg in thetas_deg:
        r_true = _ray_limb_edge(
            image, cx, cy, semi_a, semi_b, angle_deg, theta_deg,
            search_frac=search_frac, n_samples=n_samples, smooth_sigma=smooth_sigma,
        )
        if r_true is None:
            continue
        theta = math.radians(theta_deg)
        pts.append((cx + r_true * math.cos(theta), cy + r_true * math.sin(theta)))
        kept_thetas.append(theta_deg)
    if len(pts) < min_keep:
        return None
    pts = np.array(pts, dtype=np.float32)
    kept_thetas = np.array(kept_thetas)

    def _arc_span_ok(thetas: np.ndarray) -> bool:
        bins = set(int(t) // 10 for t in thetas)
        return len(bins) * 10 >= min_arc_span_deg

    if not _arc_span_ok(kept_thetas):
        return None

    current = pts
    current_thetas = kept_thetas
    for _ in range(n_iter):
        if len(current) < 5:
            return None
        (fcx, fcy), (fma, fmi), fangle = cv2.fitEllipse(current)
        dx = current[:, 0] - fcx
        dy = current[:, 1] - fcy
        ang = np.radians(fangle)
        cos_a, sin_a = np.cos(ang), np.sin(ang)
        dxr = cos_a * dx + sin_a * dy
        dyr = -sin_a * dx + cos_a * dy
        semi_a_i = max(fma, fmi) / 2.0
        semi_b_i = max(1e-3, min(fma, fmi) / 2.0)
        pred_r = 1.0 / np.sqrt((dxr / semi_a_i) ** 2 + (dyr / semi_b_i) ** 2 + 1e-12)
        actual_r = np.sqrt(dx ** 2 + dy ** 2)
        resid = actual_r - pred_r
        med = np.median(resid)
        # MAD (median absolute deviation, x1.4826 for normal-equivalent
        # scale) -- far more resistant than std to a high outlier fraction;
        # see docstring above re: std "masking" itself on Saturn.
        scale = 1.4826 * np.median(np.abs(resid - med))
        keep_mask = np.abs(resid - med) < outlier_sigma * (scale + 0.3)
        if keep_mask.sum() < min_keep or keep_mask.sum() == len(current):
            current = current[keep_mask] if keep_mask.sum() >= min_keep else current
            break
        current = current[keep_mask]
        current_thetas = current_thetas[keep_mask]
        if not _arc_span_ok(current_thetas):
            return None

    if len(current) < min_keep:
        return None
    (fcx, fcy), (fma, fmi), fangle = cv2.fitEllipse(current)
    semi_a_f, semi_b_f = max(fma, fmi) / 2.0, min(fma, fmi) / 2.0
    if fma < fmi:
        fangle = (fangle + 90.0) % 180.0
    return fcx, fcy, semi_a_f, semi_b_f, fangle, len(current)


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


# ── Ring geometry (Saturn) ──────────────────────────────────────────────────────
#
# REPLACED 2026-08-11 (real Saturn data + a user-identified architecture gap):
# an earlier image-based ring-edge detector (detect_ring_geometry(), gradient-
# based radial search) was validated against real Saturn_Data and failed on
# 100% of 45 real frames — synthetic-calibrated gradient thresholds never
# fired on real seeing-blurred rings. Separately, the user pointed out the
# deeper problem this was meant to solve: the ring (a flat, non-corotating
# structure) doesn't rotate like the globe's atmosphere, and wherever the
# ring visually crosses the globe's own silhouette, spherical_derotation_
# warp() was applying the atmosphere's depth-based rotation to ring pixels —
# wrong physics, differently wrong per frame (different dt_sec), blurring
# the stack. Confirmed directly by rendering real frames
# (Saturn_Data/step04_derotated/window_01 and window_05, IR): yes, the ring
# visually crosses the globe in this session's data (B=-11.07 deg, nearly
# edge-on, confirmed via query_horizons_sub_observer_lat).
#
# This is now computed ANALYTICALLY instead of detected in the image at all.
# _oblate_ortho_forward(phi, lam, B, P, req_px, rpol_px) already projects a
# body-fixed point to screen coordinates; at phi=0 (the equatorial plane,
# where the ring lies) it reduces to a point on an ellipse with semi-major
# req_px and semi-minor req_px*sin(|B|), at position angle P. So the ring's
# inner/outer projected ellipses are just two concentric ellipses at the
# globe's own already-measured pole_pa_deg, scaled by Saturn's fixed
# physical ring/Req ratios below and sin(|B|) for the minor axis -- no
# per-pixel ray search, no gradient thresholds, no reliance on real-image
# contrast at all.

# Saturn's real physical ring-system radii (IAU/NASA fact sheet, km from
# planet centre): C-ring inner edge ~74,658 km, A-ring outer edge ~136,780 km.
# Saturn's own equatorial radius is ~60,268 km, so as a fraction of that:
_SATURN_RING_INNER_REQ = 74_658.0 / 60_268.0   # ~1.239
_SATURN_RING_OUTER_REQ = 136_780.0 / 60_268.0  # ~2.269

# Feather width (IMAGE pixels, not depth units — see the 2026-08-11 bugfix
# note inside compute_ring_occlusion_weight() explaining why depth-space
# feathering alone produced a non-uniform, sometimes near-hard edge in image
# space) for the foreground/background occlusion boundary. Same regime as
# the module's _interp_feather_px=12.0 limb-feather convention (see
# spherical_derotation_warp) so both discontinuities get comparable
# treatment scale.
_RING_DEPTH_FEATHER_PX = 12.0


def _ring_globe_overlap_ellipses(
    h: int,
    w: int,
    cx: float,
    cy: float,
    disk_semi_a: float,
    disk_semi_b: float,
    pole_pa_deg: float,
    sub_observer_lat_deg: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Screen positions where the ring's (apparent, 2D) projected footprint
    overlaps the globe's own silhouette -- shared between compute_ring_
    occlusion_weight() (legacy linear-warp depth convention) and compute_
    ring_occlusion_weight_3d() (true-reprojection depth convention), which
    differ only in how they resolve foreground/background DEPTH at a given
    overlap pixel, not in this footprint test itself (pure apparent-ellipse
    geometry, unaffected by which depth model is used downstream).

    Returns (overlap, dx, dy, xr, yr). dx/dy are RAW screen offsets from
    centre (xx-cx, yy-cy); xr/yr are the SAME offsets after undoing pole_pa
    rotation (used for the ellipse tests here and for compute_ring_
    occlusion_weight's depth_ring closed form). A caller needing _oblate_
    ortho_inverse's own depth MUST pass it dx/dy, never xr/yr -- that
    function performs its own internal pole_pa un-rotation, so feeding it
    the already-rotated xr/yr would double-rotate (invisible at pole_pa=0,
    which is exactly why this note exists -- see compute_ring_occlusion_
    weight_3d()).
    """
    sin_b = abs(math.sin(math.radians(sub_observer_lat_deg)))
    inner_ring_semi_a = disk_semi_a * _SATURN_RING_INNER_REQ
    inner_ring_semi_b = inner_ring_semi_a * sin_b
    outer_ring_semi_a = disk_semi_a * _SATURN_RING_OUTER_REQ
    outer_ring_semi_b = outer_ring_semi_a * sin_b

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    ang = math.radians(pole_pa_deg)
    cos_a, sin_a = math.cos(ang), math.sin(ang)
    dx, dy = xx - cx, yy - cy
    xr = dx * cos_a + dy * sin_a
    yr = -dx * sin_a + dy * cos_a

    in_globe = (xr / disk_semi_a) ** 2 + (yr / disk_semi_b) ** 2 <= 1.0
    in_ring_outer = (xr / outer_ring_semi_a) ** 2 + (yr / max(outer_ring_semi_b, 1e-6)) ** 2 <= 1.0
    in_ring_inner = (xr / inner_ring_semi_a) ** 2 + (yr / max(inner_ring_semi_b, 1e-6)) ** 2 <= 1.0
    in_ring_annulus = in_ring_outer & ~in_ring_inner
    overlap = in_globe & in_ring_annulus
    return overlap, dx, dy, xr, yr


def _feather_ring_foreground_boundary(
    h: int, w: int, is_foreground: np.ndarray
) -> np.ndarray:
    """Shared feathering tail for compute_ring_occlusion_weight() and
    compute_ring_occlusion_weight_3d(): feather is_foreground's boundary by
    real image-pixel distance (see compute_ring_occlusion_weight()'s
    2026-08-11 bugfix note for why depth-space feathering alone produced a
    non-uniform, sometimes near-hard edge in image space -- the same
    reasoning applies regardless of which depth model produced
    `is_foreground`).

    BUG FIXED 2026-08-15 (real-Saturn-data visual inspection, window_01/IR,
    pole_pa=-7deg, B=-11.07deg): an earlier version only ran this distance-
    transform feathering when the caller's `overlap` region contained BOTH
    foreground and background pixels, and it then re-masked the result back
    down to exactly `overlap` (zero outside it) -- i.e. TWO separate hard
    edges: (a) whenever `overlap` happened to be 100% foreground with no
    background portion at all (confirmed to occur on real data, not just a
    theoretical edge case), it skipped feathering entirely and fell back to
    a raw, pixelated boolean mask; (b) even in the normal case, the smooth
    feather field was clipped hard at the analytic ring/globe overlap
    footprint's own boundary, which -- since that footprint is capped by
    the globe's OWN silhouette (`in_globe`) -- coincides with the disk's
    true limb over most of the ring-crossing band's width. Both produced a
    genuinely hard mask-value step exactly at/near the true photometric
    limb, which wavelet sharpening amplified into a visible bright wedge
    right where the ring crosses the globe (real-data confirmed: present
    even with ring occlusion completely neutered in the WARP, i.e. this bug
    lives entirely in this mask function, not in how it's applied).

    Fixed by feathering `is_foreground` against its full-image complement
    unconditionally (well-defined and correctly near-zero far from the
    band even when `is_foreground` happens to equal the ENTIRE analytic
    overlap region), and returning that field directly instead of re-
    masking it to `overlap` -- letting the same smooth falloff carry
    ~_RING_DEPTH_FEATHER_PX px past the analytic footprint's own edge
    (harmless: outside the disk this only touches pixels the base warp's
    own identity fallback already keeps at ~zero drift; inside the disk it
    replaces exactly the discontinuity described above with a smooth ramp).
    """
    if not is_foreground.any():
        return np.zeros((h, w), dtype=np.float32)
    fg_u8 = is_foreground.astype(np.uint8)
    dist_from_fg = cv2.distanceTransform(1 - fg_u8, cv2.DIST_L2, 5)
    dist_into_fg = cv2.distanceTransform(fg_u8, cv2.DIST_L2, 5)
    # 0 right at the boundary, ramping to 1 over _RING_DEPTH_FEATHER_PX px on
    # the foreground side; ramping to 0 over the same distance on the
    # background/exterior side (dist_from_fg subtracted so it's negative
    # there, clipped to 0).
    signed_dist = np.where(fg_u8 > 0, dist_into_fg, -dist_from_fg)
    ring_exclude = np.clip(signed_dist / _RING_DEPTH_FEATHER_PX + 0.5, 0.0, 1.0)
    return ring_exclude.astype(np.float32)


def compute_ring_occlusion_weight(
    h: int,
    w: int,
    cx: float,
    cy: float,
    disk_semi_a: float,
    disk_semi_b: float,
    pole_pa_deg: float,
    sub_observer_lat_deg: float,
) -> np.ndarray:
    """Continuous [0,1] weight: how much a pixel is occluded by FOREGROUND
    ring material (1.0 = fully ring, atmosphere de-rotation must not be
    applied there; 0.0 = not foreground ring — either plain atmosphere, or
    the ring's far side hidden behind the globe's own near surface, in which
    case what's actually visible is ordinary atmosphere and de-rotation
    SHOULD apply normally).

    BUG FIXED 2026-08-11 (external review): the previous version of this
    function (compute_ring_crossing_mask, returned a bool) only tested
    whether the ring's projected 2D footprint overlapped the globe's
    silhouette — it could not distinguish the ring passing IN FRONT of the
    globe (genuinely occludes the atmosphere, must exclude) from the ring's
    far side, hidden BEHIND the globe's own near surface (occluded BY the
    globe — nothing ring-related is actually visible there, it's just
    atmosphere, and should have gotten normal de-rotation all along). That
    conflation silently killed atmosphere de-rotation over roughly half of
    the annulus/globe overlap region, exactly where the reviewer's
    foreground/background occlusion critique says it should not have.

    Analytic geometry only, no image content examined (same philosophy as
    before): the ring's inner/outer ellipses share the globe's own
    pole_pa_deg (coplanar) and are scaled by Saturn's fixed physical
    ring/Req ratios, with sin(|B|) foreshortening the minor axis (see
    _oblate_ortho_forward's phi=0 case). NEW: within that footprint overlap,
    each pixel's ring-plane point vs. the globe's own near-surface point at
    that same screen position are compared by line-of-sight depth (same
    depth convention/units as spherical_derotation_warp's own depth_map) to
    decide foreground vs. background, feathered smoothly across the
    boundary using the depth difference itself (not an arbitrary pixel
    distance) — see module docstring below for the closed-form derivation.

    Cheap early-exit to an all-zero weight (verified correct no-op) whenever
    the ring's inner edge cannot possibly reach inside the globe at this
    tilt — covers Jupiter/other non-ringed targets (any caller not actually
    observing Saturn should not pass real ring constants here at all, but
    this is a defensive backstop) and high-|B| Saturn sessions where the
    ring genuinely clears the globe.
    """
    sin_b = abs(math.sin(math.radians(sub_observer_lat_deg)))
    inner_ring_semi_a = disk_semi_a * _SATURN_RING_INNER_REQ
    inner_ring_semi_b = inner_ring_semi_a * sin_b
    if inner_ring_semi_b >= disk_semi_b:
        # Even the ring's closest (inner-edge, minor-axis) approach to the
        # centre stays outside the globe's own silhouette — no crossing
        # possible at this tilt.
        return np.zeros((h, w), dtype=np.float32)

    overlap, _dx, _dy, xr, yr = _ring_globe_overlap_ellipses(
        h, w, cx, cy, disk_semi_a, disk_semi_b, pole_pa_deg, sub_observer_lat_deg,
    )

    if not overlap.any():
        return np.zeros((h, w), dtype=np.float32)

    weight = np.zeros((h, w), dtype=np.float64)

    B_rad = math.radians(sub_observer_lat_deg)
    if abs(sub_observer_lat_deg) < _SUB_OBS_LAT_SMALL_DEG:
        # Near-exact edge-on: the ring plane projects to a degenerate line
        # (every ring point's yr -> 0 as sin(B) -> 0), so depth_ring = -yr /
        # tan(B) is numerically unstable right where it matters most. This
        # tilt is also where the whole occlusion question is most physically
        # ambiguous. Safe conservative fallback: keep the pre-existing
        # behaviour (treat the whole footprint overlap as foreground/
        # excluded) rather than risk a wrong depth comparison here.
        weight[overlap] = 1.0
        return weight.astype(np.float32)

    # Ring-plane point depth (closed form derived from _oblate_ortho_forward
    # at phi=0): for a ring point (r, lam) with body-fixed Y = -r*cos(lam)*
    # sin(B) [same pole_pa-derotated screen Y this function already computes
    # as `yr`], its LOS depth is r*cos(lam)*cos(B) = -yr / tan(B) --
    # independent of the ring radius r, since it only depends on how far
    # off the ring-crossing line (yr=0) this screen position sits. MUST stay
    # algebraically consistent with _oblate_ortho_forward — see
    # tests/test_ring_occlusion_weight.py's cross-check against it directly.
    depth_ring = -yr / math.tan(B_rad)

    # Globe near-surface depth at the same screen position — MUST match
    # spherical_derotation_warp()'s own depth_map formula exactly (same
    # warp_radius padding, same rx_eq/ry_pol decomposition) so the two are
    # directly comparable in the same units. Duplicated here (rather than
    # calling into spherical_derotation_warp) to keep that function planet-
    # agnostic — ring physics stays entirely in this Saturn-specific helper.
    # polar_equatorial_ratio isn't a parameter of this function (only shape,
    # not size/oblateness ratio, varies enough between filters to matter
    # here — see PlanetShape/resolve_shared_shape module note above), so use
    # this call's own disk_semi_b/disk_semi_a exactly as spherical_
    # derotation_warp does when given this filter's own fitted ellipse.
    warp_radius = disk_semi_a * 1.05
    polar_equatorial_ratio = disk_semi_b / max(disk_semi_a, 1.0)
    polar_scale_sq = (1.0 / max(polar_equatorial_ratio, 1e-3)) ** 2
    depth_globe_sq = warp_radius ** 2 - xr ** 2 - polar_scale_sq * yr ** 2
    depth_globe = np.sqrt(depth_globe_sq.clip(0))

    # BUG FIXED 2026-08-11 (real-data visual inspection, before this ever
    # shipped): feathering directly by depth_diff (as a first attempt) made
    # the transition's WIDTH IN IMAGE PIXELS wildly non-uniform, because
    # depth_ring is constant along a line of fixed yr while depth_globe
    # falls off quickly near the limb (large |xr|) — the boundary curve
    # bulges toward more "foreground" near the limb, and in the direction
    # perpendicular to that bulge the depth gradient can be steep enough
    # that the depth-space feather (_RING_DEPTH_FEATHER_PX) collapses to a
    # near-hard edge in actual pixel space. Rendered real Saturn stacks
    # showed a visible dark seam tracing that curved boundary — confirmed
    # by comparing directly against the pre-occlusion-fix version, which
    # did NOT have this specific artifact (it had the coarser, already-
    # documented "wrong region entirely" problem instead). Fix: decide
    # foreground/background as a boolean via the depth comparison (this
    # part of the physics is sound — see the algebra cross-check in
    # tests/test_ring_occlusion_weight.py), then feather that boolean's
    # boundary by actual image-pixel distance (same technique as the
    # earlier, now-superseded seam fix), which guarantees a spatially
    # uniform transition width regardless of how steep the underlying
    # depth gradient is at any particular point on the boundary.
    is_foreground = overlap & (depth_ring > depth_globe)
    return _feather_ring_foreground_boundary(h, w, is_foreground)


def compute_ring_occlusion_weight_3d(
    h: int,
    w: int,
    cx: float,
    cy: float,
    disk_semi_a: float,
    disk_semi_b: float,
    pole_pa_deg: float,
    sub_observer_lat_deg: float,
    polar_equatorial_ratio_true: float,
    flip_pole_axis: bool = False,
) -> np.ndarray:
    """Same continuous [0,1] foreground-ring-occlusion weight as compute_
    ring_occlusion_weight() (see its docstring for the physical picture),
    but with the globe-side depth resolved in the TRUE 3D oblate-spheroid
    reprojection's own depth convention (_oblate_ortho_inverse/_oblate_
    ortho_forward) instead of that function's linear-warp-derived sqrt
    approximation -- for use as spherical_derotation_warp_3d()'s
    ring_crossing_mask (2026-08-15, external-review-identified gap: the
    2026-08-11 ring-occlusion fix was only ever wired into the legacy
    linear warp; this session's production Saturn config actually runs
    use_true_reprojection=True, so that validated fix was silently inert).

    depth_ring (the ring-plane side of the comparison) needs NO change: the
    closed form -yr/tan(B) is proven algebraically identical to _oblate_
    ortho_forward(phi=0, ...)'s own depth (see tests/test_ring_occlusion_
    weight.py's test_matches_oblate_ortho_forward_ring_depth) -- it was
    always expressed in this same 3D convention, just reused by the linear
    warp's helper too. Only depth_globe changes here.

    Two bugs found and fixed during this feature's design review, both
    invisible at the field-default pole_pa_deg=0 / flip_pole_axis=False
    that a quick smoke test would exercise -- do not simplify this function
    to "just call compute_ring_occlusion_weight's formula with a different
    depth_globe" without re-deriving these:

      1. flip_pole_axis sign: _oblate_ortho_forward negates Y AFTER depth is
         computed from the un-negated Y (see its own docstring/code, and
         the module note on flip_pole_axis above spherical_derotation_
         warp_3d). `yr` here (from _ring_globe_overlap_ellipses) recovers
         Y_USED (post-flip), not Y_raw. For a ring point (phi=0): depth =
         -Y_raw/tan(B). flip_pole_axis=False -> yr=Y_raw -> depth=-yr/
         tan(B) (matches compute_ring_occlusion_weight's fixed formula,
         which has no flip_pole_axis concept since the linear warp doesn't
         have one). flip_pole_axis=True -> yr=-Y_raw -> depth=+yr/tan(B):
         the sign FLIPS. Verified algebraically and via
         test_matches_oblate_ortho_forward_ring_depth_3d (must use nonzero
         pole_pa -- the bug is invisible at pole_pa=0 for an unrelated
         reason, see #2).
      2. _oblate_ortho_inverse's own internal pole_pa un-rotation: it must
         be called with RAW (dx, dy) = (xx-cx, yy-cy), never the already-
         rotated (xr, yr) this function also uses for the ellipse tests and
         depth_ring -- feeding it (xr, yr) double-rotates. This is exactly
         why _ring_globe_overlap_ellipses() returns both pairs separately
         rather than just (xr, yr): at pole_pa_deg=0, dx==xr and dy==yr, so
         this mistake would be silently invisible during the most obvious
         smoke test. Real sessions use nonzero pole_pa (this file's own
         tests already use -7.0/20.0), so this is not a theoretical
         concern.
      3. _oblate_ortho_inverse marks an unresolvable (no near-side
         solution) point with a literal depth=-1.0 sentinel, not -inf/NaN.
         Comparing depth_ring directly against that raw sentinel would
         misclassify exactly the physically important near-crossing-line
         band (depth_ring in the tens of pixels close to 0, well within
         (-1.0, 0) at this module's real pixel-unit depth scale) -- the
         opposite of the intended "unresolvable globe depth -> treat
         conservatively as foreground/occluded" policy (the same
         conservative posture already used by the B~0 fallback above).
         Fixed by explicitly replacing invalid points (phi is NaN) with
         -inf before the comparison, rather than trusting the raw sentinel.
    """
    sin_b = abs(math.sin(math.radians(sub_observer_lat_deg)))
    inner_ring_semi_a = disk_semi_a * _SATURN_RING_INNER_REQ
    inner_ring_semi_b = inner_ring_semi_a * sin_b
    if inner_ring_semi_b >= disk_semi_b:
        return np.zeros((h, w), dtype=np.float32)

    overlap, dx, dy, xr, yr = _ring_globe_overlap_ellipses(
        h, w, cx, cy, disk_semi_a, disk_semi_b, pole_pa_deg, sub_observer_lat_deg,
    )

    if not overlap.any():
        return np.zeros((h, w), dtype=np.float32)

    B_rad = math.radians(sub_observer_lat_deg)
    if abs(sub_observer_lat_deg) < _SUB_OBS_LAT_SMALL_DEG:
        weight = np.zeros((h, w), dtype=np.float64)
        weight[overlap] = 1.0
        return weight.astype(np.float32)

    depth_ring = (1.0 if flip_pole_axis else -1.0) * yr / math.tan(B_rad)

    # Same req_px/rpol_px convention as _reprojected_position()/spherical_
    # derotation_warp_3d() -- MUST match so depth_globe is directly
    # comparable to depth_ring in the same units (both ultimately derive
    # from _oblate_ortho_forward/_inverse's shared parametrization).
    req_px = disk_semi_a * 1.05
    rpol_px = req_px * polar_equatorial_ratio_true
    phi_globe, _lam_globe, depth_globe_raw = _oblate_ortho_inverse(
        dx, dy, sub_observer_lat_deg, pole_pa_deg, req_px, rpol_px,
        flip_pole_axis=flip_pole_axis,
    )
    # Bug #3 fix: never compare against the raw -1.0 invalid-depth sentinel.
    depth_globe = np.where(np.isnan(phi_globe), -np.inf, depth_globe_raw)

    is_foreground = overlap & (depth_ring > depth_globe)
    return _feather_ring_foreground_boundary(h, w, is_foreground)


def compute_ring_sharpening_mask(
    h: int,
    w: int,
    cx: float,
    cy: float,
    disk_semi_a: float,
    disk_semi_b: float,
    pole_pa_deg: float,
    sub_observer_lat_deg: float,
    outer_safety_factor: float = 1.35,
) -> np.ndarray:
    """[0,1] weight for wavelet.sharpen_disk_aware's extra_weight_map: where
    Saturn's rings should receive sharpening gain, as a TRUE ANNULUS rather
    than a filled ellipse.

    2026-08-15 root cause (real-data confirmed, see project_ring_limb_
    ringing_bug memory and SATURN_RING_WAVELET_STATUS_2026-08-15.md): the
    previous mechanism (sharpen_disk_aware's extra_rx/extra_ry/extra_gap_px)
    builds a FILLED ellipse out to the ring's outer edge, with only an
    extra_gap_px-wide (~8px) ramp protecting the zone just outside the
    globe's own true boundary. But the real gap between the globe (r=1) and
    the ring's own true inner edge (r=_SATURN_RING_INNER_REQ, ~1.239) is far
    wider than 8px, and real-data profiling confirmed it carries high-SNR
    signal that is NOT ring material -- it's the globe's own PSF-blurred
    limb tail plus genuine empty gap. Applying full ring-level sharpening
    gain there amplified whatever faint gradient sits in that zone into the
    white-rim/dark-trough artifact. This function instead:

      1. Restricts the "extra" gain to the TRUE ring annulus (inner_ring_semi_a
         to outer_ring_semi_a*outer_safety_factor) -- nothing in the gap
         between the globe and the ring's real inner edge gets any gain from
         this mask at all.
      2. Where that annulus overlaps the globe's own silhouette (the ring
         visually crossing the disk), restricts further to the FRONT arc
         only (depth_ring > 0, the same closed-form depth compute_ring_
         occlusion_weight_3d already uses) -- the far/occluded ring arc must
         not receive gain there, since what's actually visible is the
         globe's own near surface, not ring. Everywhere the annulus does
         NOT overlap the globe (the vast majority of the ring), both near
         and far arcs are equally real, visible ring material and get full
         coverage -- gating by depth_ring globally (this function's own
         first, incorrect attempt) wrongly zeroed out the entire far arc's
         open-sky portion too.
      3. Feathers only the annulus's own outer boundary (reusing
         _feather_ring_foreground_boundary, the same helper already used for
         compute_ring_occlusion_weight/_3d) -- no independent inner-edge
         ramp, so no new seam is introduced at the globe/ring connection:
         wherever this mask overlaps the primary disk mask, the primary
         mask (already ~1.0 there) simply dominates via the caller's
         max-combine.

    outer_safety_factor: matches wavelet_master.py's existing _RING_MASK_
    SAFETY_FACTOR -- real multi-frame Saturn stacks show ring signal fading
    out well past the strict IAU A-ring outer-edge ratio (2.269x), almost
    certainly PSF/seeing/residual-stacking blur smearing the true edge
    outward (see wavelet_master.py's own comment on this constant). Not a
    physical claim, just over-covering being harmless (extends gain a bit
    into background) versus under-covering leaving real ring detail
    unsharpened.

    Validated 2026-08-15 on real window_01/R: eliminates the white-rim
    overshoot that extra_rx produced at the disk-ring junction while
    keeping ring band detail sharpened right up to (but not into) the
    globe's own halo -- see experiments/ringing_fix_validation/v2_fullring_*
    from that session's investigation.
    """
    sin_b = abs(math.sin(math.radians(sub_observer_lat_deg)))
    inner_ring_semi_a = disk_semi_a * _SATURN_RING_INNER_REQ
    inner_ring_semi_b = inner_ring_semi_a * sin_b
    outer_ring_semi_a = disk_semi_a * _SATURN_RING_OUTER_REQ * outer_safety_factor
    outer_ring_semi_b = max(outer_ring_semi_a * sin_b, 1e-6)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    ang = math.radians(pole_pa_deg)
    cos_a, sin_a = math.cos(ang), math.sin(ang)
    dx, dy = xx - cx, yy - cy
    xr = dx * cos_a + dy * sin_a
    yr = -dx * sin_a + dy * cos_a

    in_ring_outer = (xr / outer_ring_semi_a) ** 2 + (yr / outer_ring_semi_b) ** 2 <= 1.0
    in_ring_inner = (xr / inner_ring_semi_a) ** 2 + (yr / max(inner_ring_semi_b, 1e-6)) ** 2 <= 1.0
    in_ring_annulus = in_ring_outer & ~in_ring_inner
    in_globe = (xr / disk_semi_a) ** 2 + (yr / max(disk_semi_b, 1e-6)) ** 2 <= 1.0

    if abs(sub_observer_lat_deg) < _SUB_OBS_LAT_SMALL_DEG:
        # Ring-plane edge-on: front/back is undefined/degenerate: no globe
        # overlap to worry about resolving, so skip the depth test entirely.
        return _feather_ring_foreground_boundary(h, w, in_ring_annulus)

    B_rad = math.radians(sub_observer_lat_deg)
    depth_ring = -1.0 * yr / math.tan(B_rad)
    front_only = depth_ring > 0

    ring_footprint = in_ring_annulus & (front_only | ~in_globe)
    return _feather_ring_foreground_boundary(h, w, ring_footprint)


def _ring_annulus_mask(
    h: int,
    w: int,
    cx: float,
    cy: float,
    disk_semi_a: float,
    pole_pa_deg: float,
    sub_observer_lat_deg: float,
    feather: bool = True,
) -> np.ndarray:
    """[0,1] weight over Saturn's analytic ring annulus (inner to outer
    physical radius), with NO globe-overlap/depth distinction -- unlike
    compute_ring_sharpening_mask()/compute_ring_occlusion_weight(), which
    both care about foreground-vs-background where the ring crosses the
    globe silhouette. This is deliberately simpler: added 2026-08-16
    (project_ring_globe_layer_separation_roadmap Phase 1), originally for
    use as a phase-correlation window so subpixel_align()'s measured
    frame-to-frame shift is driven by the ring's own signal instead of the
    globe's. Whether that shift lands in front of or behind the globe at
    any given pixel is irrelevant for this purpose -- both are real ring
    material moving together.

    feather: when True (default -- suitable for compositing use, e.g. a
    future Phase 3 blend weight), the boundary is smoothly ramped via
    _feather_ring_foreground_boundary(), same convention as the other ring
    masks in this module. When False, returns the RAW hard boolean mask
    (cast to float) instead. **This matters, not just a cosmetic choice**:
    empirically (synthetic tests, tests/test_ring_occlusion_weight.py),
    multiplying phase-correlation input by the FEATHERED version measurably
    corrupts subpixel_align()'s result for some shift magnitudes (verified
    wrong by several pixels at (3,-2)px true shift, correct at (1,0)px and
    (5,5)px -- not a simple monotonic degradation) -- the smooth amplitude
    taper apparently interacts badly with phase correlation's frequency-
    domain assumptions here. The hard mask, despite its sharp edge (usually
    the thing a feather is meant to avoid), measured correctly across every
    shift tested. Callers doing registration (not compositing) MUST pass
    feather=False.

    Same physical ratios/rotation convention as the other ring-geometry
    helpers in this module (_SATURN_RING_INNER_REQ/_SATURN_RING_OUTER_REQ,
    pole_pa-aligned ellipses) -- reuses that scaling, not a new geometry
    model.
    """
    sin_b = abs(math.sin(math.radians(sub_observer_lat_deg)))
    inner_semi_a = disk_semi_a * _SATURN_RING_INNER_REQ
    inner_semi_b = inner_semi_a * sin_b
    outer_semi_a = disk_semi_a * _SATURN_RING_OUTER_REQ
    outer_semi_b = max(outer_semi_a * sin_b, 1e-6)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    ang = math.radians(pole_pa_deg)
    cos_a, sin_a = math.cos(ang), math.sin(ang)
    dx, dy = xx - cx, yy - cy
    xr = dx * cos_a + dy * sin_a
    yr = -dx * sin_a + dy * cos_a

    in_outer = (xr / outer_semi_a) ** 2 + (yr / outer_semi_b) ** 2 <= 1.0
    in_inner = (xr / inner_semi_a) ** 2 + (yr / max(inner_semi_b, 1e-6)) ** 2 <= 1.0
    in_annulus = in_outer & ~in_inner
    if not feather:
        return in_annulus.astype(np.float32)
    return _feather_ring_foreground_boundary(h, w, in_annulus)


def _raised_cosine_falloff(value: np.ndarray, inner: float, outer: float) -> np.ndarray:
    """1.0 for value<=inner, 0.0 for value>=outer, smooth (C1, zero-derivative
    at both ends) cosine interpolation between. Distinct from
    _feather_ring_foreground_boundary's LINEAR distance-transform ramp above,
    which feathers a BOOLEAN mask's pixel-distance-to-boundary, not a
    continuous scalar field's own value (r_norm, angular distance) -- no
    existing helper in this module does the latter, hence this one.
    """
    if outer <= inner:
        return (value <= inner).astype(np.float64)
    t = np.clip((value - inner) / (outer - inner), 0.0, 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * t))


def _raised_cosine_rise(value: np.ndarray, inner: float, outer: float) -> np.ndarray:
    """0.0 for value<=inner, 1.0 for value>=outer -- rising counterpart to
    _raised_cosine_falloff (same cosine shape, mirrored)."""
    return 1.0 - _raised_cosine_falloff(value, inner, outer)


def estimate_ring_scatter_leak(
    image: np.ndarray,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    pole_pa_deg: float,
    ld_fit: LimbDarkeningFit,
    ansa_half_width_deg: float = 15.0,
    ansa_feather_to_deg: float = 25.0,
    r_norm_core: Tuple[float, float] = (0.90, 0.95),
    r_norm_feather_in: float = 0.80,
    r_norm_feather_out: float = 0.98,
    blur_sigma_px: float = 2.5,
) -> np.ndarray:
    """Estimate Saturn ring optical/PSF scattered light leaking onto the
    globe near the ring ansae (phi=0/180 in the pole_pa_deg-aligned frame),
    as a per-pixel field to SUBTRACT from `image` BEFORE wavelet
    decomposition -- see project_limb_darkening_confidence_map memory's "PSF
    산란광 가설" investigation (reproduced across 9 independent same-night
    Saturn stacks, IR/R/G/B, 2026-08-17) for the validated background.

    DIFFERENT FROM EVERY OTHER RING/LIMB FIX IN THIS MODULE
    (compute_ring_sharpening_mask, _ring_annulus_mask, compute_ring_
    occlusion_weight*): those reshape sharpening GAIN or de-rotation WEIGHT
    around a frozen input. This estimates a real photometric contamination
    IN the input signal and removes it before anything downstream (gain
    maps, confidence maps, wavelet decomposition) sees the pixel values --
    it composes with, rather than competes against, all of the above. It is
    also NOT expected to fix the unrelated ellipse-fit-asymmetry ringing bug
    (project_ring_limb_ringing_bug memory, ~10 previously-tried and rejected
    mask/gain/filter-on-top-of-frozen-input approaches) -- that is a
    separate, already-diagnosed phenomenon; do not evaluate this feature
    against it.

    Method: `image - evaluate_limb_darkening_curve(r_norm, ld_fit)` is the
    residual over the disk's own symmetric Minnaert baseline (fit EXCLUDING
    ring pixels, so it is ring-contamination-free by construction). Only
    POSITIVE residual ("excess") is ever removed -- a pixel at or below the
    model's prediction is left alone. The excess is lightly Gaussian-blurred
    (removing only the smooth "glow" component, not pixel noise/real fine
    detail) and multiplied by a window that is 1.0 only in the
    r_norm~[0.90,0.95] band nearest the ansae (in the pole_pa_deg-aligned
    frame, matching how this project's ring geometry is measured throughout)
    and fades smoothly to 0.0 both radially and angularly away from there.

    RADIAL RANGE IS DELIBERATELY KEPT WELL SHORT OF r_norm=1.0 (the fitted
    disk boundary), not just short of the ring annulus at r_norm~1.239 (see
    _SATURN_RING_INNER_REQ): the Minnaert model's cos(theta)^m term goes to
    exactly 0 at r_norm=1.0 by construction (theta=90deg), so `excess` grows
    explosively as r_norm approaches/crosses 1.0 for a reason that has
    nothing to do with any real leak -- confirmed on real Saturn data
    (window_01/R, 2026-08-17): max excess in [0.90,0.95] was 0.056-0.096
    (matches the validated ~0.03-0.08 finding), but including up to
    r_norm=1.0 inflated it to 0.26, and up to 1.05 to 0.34 -- because at
    r_norm just past 1.0 near the ansa, the pixel is no longer globe at all,
    it is the ring itself (exactly the validated "ring projects outside the
    disk near the ansae" geometry) showing through, and "correcting" an
    actual ring pixel down to the globe model's ~0 prediction there is not
    the intended effect. The default r_norm_core/r_norm_feather_out values
    stay at/under 0.98, matching fit_limb_darkening_curve's own
    r_norm_fit_max=0.98 convention for excluding this same unstable region.

    SAFETY CLAMP: the returned leak is clamped to `<= excess` POINTWISE, not
    just windowed/blurred. Gaussian blur can, at any pixel, produce a value
    larger than that pixel's own raw excess (mixing in a neighbour's larger
    excess) -- without this clamp, `image - strength*leak` (strength<=1)
    could push a pixel BELOW the model's own prediction, inventing a new
    local dark trough at the correction's own boundary -- the exact failure
    mode 3 of this project's ~10 previously-rejected approaches hit (see
    project_ring_limb_ringing_bug memory), which the critical-defect bar
    (feedback_white_rim_is_critical_defect memory) does not tolerate. With
    the clamp, `image - strength*leak` is bounded
    `predicted <= corrected <= image` for every pixel, every
    strength in [0, 1] -- by construction, not by tuning.

    Args:
        image:        Float array in [0, 1], 2-D (single channel/luminance)
                       ONLY. Raises ValueError on a 3-D input -- callers with
                       a color image must pass a luminance plane and decide
                       separately how to redistribute the correction across
                       channels (not yet designed/validated -- see
                       WaveletConfig.master_ring_scatter_subtraction_enabled).
        cx, cy, rx, ry: The disk's FINAL fitted ellipse geometry (after any
                       nav-fit refinement), same convention as
                       limb_darkening._ellipse_normalized_radius.
        pole_pa_deg:  Ring/pole position angle -- ansae (phi=0/180) are
                       defined in THIS frame, matching wavelet_master.py's
                       own reorientation of the globe mask to pole_pa_deg
                       for has_rings targets (NOT the disk ellipse's own,
                       possibly mod-180-degenerate, fitted angle).
        ld_fit:       A LimbDarkeningFit already fit from this SAME image
                       (limb_darkening.fit_limb_darkening_curve), ideally
                       with ring pixels excluded from the fit. Not re-fit or
                       validated here.
        ansa_half_width_deg / ansa_feather_to_deg:
                       Angular window (degrees from the nearest ansa, 0 or
                       180, pole_pa_deg frame): full weight within
                       ansa_half_width_deg, smoothly to zero by
                       ansa_feather_to_deg.
        r_norm_core / r_norm_feather_in / r_norm_feather_out:
                       Radial window (ellipse-normalized radius): full
                       weight within r_norm_core, ramped from 0 at
                       r_norm_feather_in up to full at r_norm_core[0], back
                       to 0 at r_norm_feather_out.
        blur_sigma_px: Gaussian blur sigma (pixels), applied to the raw
                       excess field before windowing/clamping.

    Returns:
        Float32 (H, W) array, same shape as `image`, always >= 0 -- the
        amount to subtract, scaled by the caller's own `strength` in [0, 1].
        This function has no strength knob itself -- the physical estimate
        and the "how much to trust it" scaling decision stay separate, same
        as confidence_map vs. gain elsewhere in this codebase.
    """
    if image.ndim != 2:
        raise ValueError(
            f"estimate_ring_scatter_leak expects a 2-D luminance plane, got ndim={image.ndim}"
        )

    h, w = image.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    ang = math.radians(pole_pa_deg)
    cos_a, sin_a = math.cos(ang), math.sin(ang)
    dx, dy = xx - cx, yy - cy
    xr = dx * cos_a + dy * sin_a
    yr = -dx * sin_a + dy * cos_a
    r_norm = np.sqrt((xr / rx) ** 2 + (yr / ry) ** 2)
    phi_deg = np.degrees(np.arctan2(yr / ry, xr / rx))

    predicted = evaluate_limb_darkening_curve(r_norm, ld_fit)
    excess = np.maximum(0.0, image.astype(np.float64) - predicted)

    excess_smooth = cv2.GaussianBlur(
        excess.astype(np.float32), (0, 0), sigmaX=blur_sigma_px,
    ).astype(np.float64)

    d0 = np.abs(((phi_deg - 0.0 + 180.0) % 360.0) - 180.0)
    d180 = np.abs(((phi_deg - 180.0 + 180.0) % 360.0) - 180.0)
    d_ansa = np.minimum(d0, d180)
    w_phi = _raised_cosine_falloff(d_ansa, ansa_half_width_deg, ansa_feather_to_deg)

    w_r = (
        _raised_cosine_rise(r_norm, r_norm_feather_in, r_norm_core[0])
        * _raised_cosine_falloff(r_norm, r_norm_core[1], r_norm_feather_out)
    )

    leak = np.minimum(w_phi * w_r * excess_smooth, excess)
    return leak.astype(np.float32)


def _predicted_apparent_ratio(
    true_polar_equatorial_ratio: float,
    sub_observer_lat_deg: float,
) -> float:
    """Analytically predict a body-of-revolution's APPARENT (projected)
    semi-minor/semi-major ratio from its TRUE physical oblateness and the
    sub-observer latitude B -- no image, no ellipse fit, no rotation matrix.

    2026-08-16, part of the navigation-constrained limb fit
    (`_navigation_constrained_ellipse_fit`): standard result for the
    orthographic silhouette of an oblate spheroid. Because the body is a
    surface of revolution about its polar axis, the screen axis
    PERPENDICULAR to the tilt plane is always exactly `Req` regardless of B
    or sub-observer longitude (rotating the body about its own polar axis
    doesn't change this axis's projected length) -- only the axis WITHIN the
    tilt plane foreshortens, from `Rpol` at B=0 (edge-on) to `Req` at B=90
    (pole-on). Derivation (envelope of the family of latitude circles'
    projected ellipses as Z sweeps the polar axis, standard `p*Z +
    q*sqrt(c^2-Z^2)` maximization): the tilt-plane semi-axis has apparent
    length `sqrt(Rpol^2*cos(B)^2 + Req^2*sin(B)^2)`, so as a ratio to `Req`:

        apparent_ratio = sqrt(true_ratio^2 * cos(B)^2 + sin(B)^2)

    (checked against the stated limits above: B=0 -> true_ratio exactly,
    B=90 -> 1.0 exactly.)

    Deliberately NOT derived via `_oblate_ortho_forward`/`_oblate_ortho_
    inverse` (correct but would require numerically tracing the silhouette
    envelope) -- this closed form needs no rotation convention at all, so it
    carries none of the sign/handedness risk that has bitten this module's
    other B/pole_pa math before (see module docstring's `_oblate_ortho_*`
    history). Symmetric in the sign of B by construction (only |sin|/|cos|
    matter via the squares).
    """
    b_rad = math.radians(sub_observer_lat_deg)
    return math.sqrt((true_polar_equatorial_ratio ** 2) * math.cos(b_rad) ** 2 + math.sin(b_rad) ** 2)


def _ring_contaminated_theta_mask(
    thetas_deg: np.ndarray,
    cx: float,
    cy: float,
    disk_semi_a: float,
    disk_semi_b: float,
    pole_pa_deg: float,
    sub_observer_lat_deg: float,
    outer_safety_factor: float = 1.35,
) -> np.ndarray:
    """For each image-frame angle in `thetas_deg`, test whether the SEED
    ellipse's own boundary point in that direction falls inside Saturn's
    analytic ring annulus footprint -- i.e. which limb rays are expected to
    be ring-contaminated and should be excluded before fitting.

    Reuses the exact ellipse-membership tests already used by
    `_ring_globe_overlap_ellipses`/`compute_ring_sharpening_mask` (same
    `_SATURN_RING_INNER_REQ`/`_SATURN_RING_OUTER_REQ` physical ratios, same
    `outer_safety_factor` convention), evaluated at 1-D boundary points
    instead of over a full (h, w) pixel grid -- a thin wrapper, not a new
    model. The seed `disk_semi_a`/`disk_semi_b` (from the current, possibly
    ring-biased fit) only sets the SCALE of the ring geometry for exclusion
    purposes, generously padded by `outer_safety_factor` -- the same
    tolerance-to-seed-error the two functions above already rely on.

    Returns a bool array (len == len(thetas_deg)), True where the ray at
    that theta should be EXCLUDED (ring-contaminated).
    """
    sin_b = abs(math.sin(math.radians(sub_observer_lat_deg)))
    inner_ring_semi_a = disk_semi_a * _SATURN_RING_INNER_REQ
    inner_ring_semi_b = inner_ring_semi_a * sin_b
    outer_ring_semi_a = disk_semi_a * _SATURN_RING_OUTER_REQ * outer_safety_factor
    outer_ring_semi_b = max(outer_ring_semi_a * sin_b, 1e-6)

    thetas = np.radians(np.asarray(thetas_deg, dtype=np.float64))
    ang = math.radians(pole_pa_deg)
    cos_a, sin_a = math.cos(ang), math.sin(ang)

    # The globe-limb boundary point in image space at image-frame angle
    # theta is the point at distance r_ell (same closed form `_ray_limb_
    # edge` uses) from the centre along that direction; rotate it into the
    # ellipse-aligned frame (xr, yr) the same way `_ring_globe_overlap_
    # ellipses` rotates dx/dy, for the ring-membership test below.
    dxu, dyu = np.cos(thetas), np.sin(thetas)
    dxr = cos_a * dxu + sin_a * dyu
    dyr = -sin_a * dxu + cos_a * dyu
    r_ell = 1.0 / np.sqrt((dxr / disk_semi_a) ** 2 + (dyr / disk_semi_b) ** 2)
    xr = dxr * r_ell
    yr = dyr * r_ell

    in_ring_outer = (xr / outer_ring_semi_a) ** 2 + (yr / outer_ring_semi_b) ** 2 <= 1.0
    in_ring_inner = (xr / inner_ring_semi_a) ** 2 + (yr / max(inner_ring_semi_b, 1e-6)) ** 2 <= 1.0
    return in_ring_outer & ~in_ring_inner


def _fixed_shape_circle_fit(
    points_xy: np.ndarray,
    angle_deg: float,
    ratio: float,
) -> Optional[Tuple[float, float, float]]:
    """Fit (cx, cy, semi_a) from boundary points, given a FIXED orientation
    and FIXED semi_b/semi_a ratio -- i.e. the only unknowns are the centre
    and one overall scale, not a free 5-parameter conic fit.

    Method: rotate points into the ellipse-aligned frame by -angle_deg (same
    convention as `_ray_limb_edge`/`_robust_ellipse_refit`), rescale the
    rotated y-coordinate by 1/ratio -- this turns the known-oblate ellipse
    into a circle -- then solve the standard linear (Kasa) least-squares
    circle fit for (cx', cy', r) in that rescaled frame, and finally
    un-rescale/un-rotate the centre back to image coordinates.

    Returns None if fewer than 5 points are given or the linear system is
    singular/ill-conditioned (near-collinear points).
    """
    pts = np.asarray(points_xy, dtype=np.float64)
    if len(pts) < 5:
        return None
    ang = math.radians(angle_deg)
    cos_a, sin_a = math.cos(ang), math.sin(ang)
    x, y = pts[:, 0], pts[:, 1]
    xr = x * cos_a + y * sin_a
    yr = (-x * sin_a + y * cos_a) / ratio

    # Kasa circle fit: minimize sum((xr-a)^2+(yr-b)^2-r^2)^2 via the linear
    # system [2xr 2yr 1][a b c]^T = xr^2+yr^2, where c = a^2+b^2-r^2.
    A = np.column_stack([2.0 * xr, 2.0 * yr, np.ones_like(xr)])
    rhs = xr ** 2 + yr ** 2
    solution, _residuals, rank, _sv = np.linalg.lstsq(A, rhs, rcond=None)
    if rank < 3:
        return None
    a_fit, b_fit, c_fit = solution
    r2 = c_fit + a_fit ** 2 + b_fit ** 2
    if r2 <= 0:
        return None
    r_fit = math.sqrt(r2)

    # Un-rescale/un-rotate the centre back to image coordinates.
    cx_r, cy_r = a_fit, b_fit * ratio
    cx_img = cx_r * cos_a - cy_r * sin_a
    cy_img = cx_r * sin_a + cy_r * cos_a
    return float(cx_img), float(cy_img), float(r_fit)


def _navigation_constrained_ellipse_fit(
    image: np.ndarray,
    cx: float,
    cy: float,
    semi_a: float,
    semi_b: float,
    pole_pa_deg: float,
    sub_observer_lat_deg: float,
    true_polar_equatorial_ratio: float,
    n_rays: int = 72,
    search_frac: Tuple[float, float] = (0.90, 1.10),
    n_samples: int = 300,
    smooth_sigma: float = 1.5,
    outlier_sigma: float = 2.5,
    min_keep: int = 20,
    min_arc_span_deg: float = 90.0,
    ring_safety_factor: float = 1.35,
) -> Optional[Tuple[float, float, float, float, float, int]]:
    """Refine a seed ellipse against the true limb using EXTERNAL navigation
    data (Horizons B, already-known true oblateness, already-measured
    pole_pa_deg) instead of statistical outlier rejection -- the has_rings
    counterpart to `_robust_ellipse_refit` (which is validated ringless-only
    and a documented near-no-op on Saturn, see that function's docstring).

    Added 2026-08-16 after `_robust_ellipse_refit` (MAD-based) and an
    exclude-ring-rays-then-free-refit scratch experiment
    (`experiments/scratch_globe_fit_asymmetry_diagnosis.py`) both proved
    insufficient on Saturn: ring contamination is a large CONTIGUOUS ~40%
    angular arc, not scattered points, so (a) point-wise robust statistics
    can't tell "contaminated majority" from "consensus", and (b) even after
    excluding that arc, asking a free 5-parameter conic fit to recover
    orientation AND aspect ratio from the remaining ~60% is ill-conditioned
    (the excluded arc sits near the major axis, exactly where that
    information is most needed).

    This function instead fixes orientation (pole_pa_deg, already measured
    independent of any ellipse fit via `auto_detect_equator_pa()`) and
    aspect ratio (analytically predicted by `_predicted_apparent_ratio()`
    from Horizons B and the planet's TRUE physical oblateness) BEFORE
    looking at any ray data -- leaving only (cx, cy, scale) to fit from the
    ring-free ~60% of rays, a heavily over-determined, well-conditioned
    problem regardless of which contiguous arc is missing.

    Steps: (1) predict the apparent ratio; (2) sample `n_rays` image-frame
    thetas and drop ring-contaminated ones via
    `_ring_contaminated_theta_mask()`; (3) measure the true edge at each
    surviving theta via the existing `_ray_limb_edge()` (unmodified reuse,
    narrow-window gradient search around the seed); (4) one round of
    MAD-based rejection (same scale/threshold convention as
    `_robust_ellipse_refit`) to catch ordinary local-albedo outliers
    (belt/zone features), now a much smaller, better-behaved problem since
    the systematic ring block is already gone analytically; (5) fit
    `(cx, cy, semi_a)` via `_fixed_shape_circle_fit()` with the fixed
    angle/ratio.

    Same "never worse than the seed" contract as `_robust_ellipse_refit`:
    returns None (caller keeps the seed unchanged) if too few rays survive
    or the surviving arc span collapses below `min_arc_span_deg`.

    Returns (cx, cy, semi_a, semi_b, pole_pa_deg, n_kept), or None.
    """
    predicted_ratio = _predicted_apparent_ratio(true_polar_equatorial_ratio, sub_observer_lat_deg)

    thetas_deg = np.arange(0.0, 360.0, 360.0 / n_rays)
    excluded = _ring_contaminated_theta_mask(
        thetas_deg, cx, cy, semi_a, semi_b, pole_pa_deg, sub_observer_lat_deg,
        outer_safety_factor=ring_safety_factor,
    )

    pts = []
    kept_thetas = []
    for theta_deg, is_excluded in zip(thetas_deg, excluded):
        if is_excluded:
            continue
        r_true = _ray_limb_edge(
            image, cx, cy, semi_a, semi_b, pole_pa_deg, float(theta_deg),
            search_frac=search_frac, n_samples=n_samples, smooth_sigma=smooth_sigma,
        )
        if r_true is None:
            continue
        theta = math.radians(float(theta_deg))
        pts.append((cx + r_true * math.cos(theta), cy + r_true * math.sin(theta)))
        kept_thetas.append(theta_deg)

    if len(pts) < min_keep:
        return None
    pts = np.array(pts, dtype=np.float64)
    kept_thetas = np.array(kept_thetas)

    def _arc_span_ok(thetas: np.ndarray) -> bool:
        bins = set(int(t) // 10 for t in thetas)
        return len(bins) * 10 >= min_arc_span_deg

    if not _arc_span_ok(kept_thetas):
        return None

    # One round of MAD-based rejection against the seed ellipse's own
    # predicted boundary, to drop ordinary local-albedo outliers (belt/zone
    # features) -- reuses the same scale/threshold convention as
    # `_robust_ellipse_refit`, but against the FIXED seed geometry rather
    # than an iteratively refit one, since there is no free-parameter fit
    # loop here to iterate against.
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    ang = math.radians(pole_pa_deg)
    cos_a, sin_a = math.cos(ang), math.sin(ang)
    dxr = cos_a * dx + sin_a * dy
    dyr = -sin_a * dx + cos_a * dy
    pred_r = 1.0 / np.sqrt((dxr / semi_a) ** 2 + (dyr / semi_b) ** 2 + 1e-12)
    actual_r = np.sqrt(dx ** 2 + dy ** 2)
    resid = actual_r - pred_r
    med = np.median(resid)
    scale = 1.4826 * np.median(np.abs(resid - med))
    keep_mask = np.abs(resid - med) < outlier_sigma * (scale + 0.3)
    if keep_mask.sum() >= min_keep:
        pts = pts[keep_mask]
        kept_thetas = kept_thetas[keep_mask]
        if not _arc_span_ok(kept_thetas):
            return None

    if len(pts) < min_keep:
        return None

    fit = _fixed_shape_circle_fit(pts, pole_pa_deg, predicted_ratio)
    if fit is None:
        return None
    cx_fit, cy_fit, semi_a_fit = fit
    semi_b_fit = semi_a_fit * predicted_ratio
    return cx_fit, cy_fit, semi_a_fit, semi_b_fit, pole_pa_deg, len(pts)


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
    SNR/contrast, not a real size difference. disk_radius_px sets the warp's
    spatial scale (warp_radius = disk_radius_px*1.05) and the CUBIC/LINEAR
    interpolation feather boundary near the limb, so a 1-3px per-filter
    difference means each filter's de-rotation gets a slightly different
    depth/drift profile and feather transition even where both are
    otherwise "valid" — not just at whatever the current invalid-pixel
    fallback happens to be (this reasoning holds for both the linear warp's
    identity fallback and the true-reprojection warp's, see
    _reprojected_position — do not re-word this as being about an
    invalid/background cutoff, since neither warp has one as of 2026-08-11).
    Composited, the per-filter mismatch shows as a colour fringe at the
    limb; wavelet sharpening (which has no way to know the "edge" it's
    amplifying is an algorithmic artifact, not the real limb) makes it
    worse. This is present already in each filter's own step04 output,
    before composite or wavelet ever run — confirmed directly against real
    per-window derotation logs (radii spanning 103.0-104.9px in one window).

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

    POLICY NOTE (external review, 2026-08-11, deliberately not changed):
    any confidence>0.0 fit counts here, including confidence=0.3
    ("unconfirmed geometric estimate" — see _find_disk_center_impl) as well
    as 0.5/1.0. On real Jupiter data this never matters (every filter is
    confidence=1.0), but on a future ring-affected/CH4 session a 0.3 fit
    could pull the median toward an unconfirmed estimate if broadband fits
    are few. Not raising the floor to >=0.5 preemptively without real
    evidence it's a problem — derotate_filter()'s frame log now records
    own_disk_radius_px/warp_disk_radius_px/radius_shared specifically so
    this can be checked against real Saturn data when that work resumes,
    rather than guessed at now.
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
    ring_crossing_mask: Optional[np.ndarray] = None,
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
        ring_crossing_mask:     Optional continuous [0,1] float array, same
                                (h, w) as image, from compute_ring_occlusion_
                                weight(). 1.0 = pixel is occluded by
                                FOREGROUND ring material, so this function
                                scales drift toward zero there (keeping the
                                frame's own content, since ring material is
                                flat and non-corotating, not a point on the
                                rotating sphere); 0.0 = ordinary atmosphere,
                                including the ring's far side hidden behind
                                the globe's own near surface — full normal
                                drift applies. Already feathered across the
                                foreground/background boundary by real image-
                                pixel distance (see compute_ring_occlusion_
                                weight()'s docstring) — this function only
                                needs a plain multiplicative scale, no
                                separate feathering of its own. None
                                (default) preserves prior behaviour exactly
                                for every existing caller (Jupiter and any
                                caller not passing this). ALSO used
                                source-side (2026-08-11, external review):
                                an ordinary-atmosphere OUTPUT pixel's
                                shifted FETCH coordinate can independently
                                land on foreground-ring territory even when
                                the output pixel itself is correctly
                                classified as atmosphere (confirmed
                                empirically, ~1% of atmosphere pixels at
                                this session's real drift magnitudes) —
                                same category as this module's existing
                                "no valid same-kind source, fall back to
                                identity" pattern (off-disk, depth_sq<=0),
                                just checked at the fetch point instead of
                                the output point, by re-sampling this same
                                mask there (see the code just after map_x/
                                map_y are computed). Purely geometric — no
                                new image content examined, no fill/inpaint
                                (three fill-based attempts at this same
                                leak were tried and reverted first; see
                                project_derotation_ring_occlusion_fix
                                memory for why).
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
    is_atmosphere = depth_sq > 0.0
    depth_map = np.where(is_atmosphere, np.sqrt(depth_sq.clip(0)), 0.0).astype(np.float32)
    if ring_crossing_mask is not None:
        # ring_crossing_mask is a continuous [0,1] occlusion weight from
        # compute_ring_occlusion_weight() (1.0 = foreground ring, exclude;
        # 0.0 = normal atmosphere, including the ring's far side hidden
        # behind the globe's own near surface) -- already feathered across
        # the boundary by real image-pixel distance, so a plain
        # multiplicative scale-down is all that's needed here.
        depth_map = depth_map * (1.0 - ring_crossing_mask)

    sign = -1.0 if flip_direction else 1.0
    drift = (sign * scale * delta_lambda_rad * depth_map).astype(np.float32)

    # Decompose drift into image-plane x/y using the pole position angle.
    # pole_pa = 0°  → cos=1, sin=0  → pure horizontal (Jupiter default)
    # pole_pa = 90° → cos=0, sin=1  → pure vertical
    # (cos_pa / sin_pa already computed above for the depth decomposition)
    map_x = (xx - drift * cos_pa).astype(np.float32)
    map_y = (yy - drift * sin_pa).astype(np.float32)

    if ring_crossing_mask is not None:
        # BUG FIXED 2026-08-11 (external review, source-side leak): the
        # ring_crossing_mask multiply above only stops THIS pixel's own
        # drift when THIS pixel's own (output) position is foreground ring
        # -- it does nothing about an ordinary-atmosphere pixel whose
        # computed FETCH point (map_x, map_y) happens to land on
        # foreground-ring territory anyway (confirmed empirically: ~1% of
        # atmosphere-classified output pixels at this session's real drift
        # magnitudes). Same category of problem this module already solves
        # elsewhere by falling back to identity when there's no valid same-
        # kind source to sample (off-disk pixels, depth_sq<=0, the 3D
        # reprojection's far-side check in _reprojected_position) -- a
        # rotating-atmosphere formula has no business fetching from a
        # flat, non-corotating ring just because the arithmetic lands
        # there. Fix: sample ring_crossing_mask itself at the FETCH point
        # (not the output point) and blend the fetch coordinate back
        # toward identity (this pixel's own position) proportionally to
        # how much the fetch point is foreground ring -- purely
        # geometric (reuses the same already-computed, already-feathered
        # occlusion weight via interpolation), no image content examined.
        _src_ring_weight = cv2.remap(
            ring_crossing_mask, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
        )
        map_x = map_x * (1.0 - _src_ring_weight) + xx * _src_ring_weight
        map_y = map_y * (1.0 - _src_ring_weight) + yy * _src_ring_weight

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
    # CORRECTED 2026-08-11 (real-data user report): invalid points now fall
    # back to IDENTITY (new_x=x, new_y=y) — i.e. no shift at all — matching
    # spherical_derotation_warp()'s own established behaviour for its
    # analogous "no computable depth" case (there, depth_sq<=0 forces
    # depth_map=0, so drift=0 and the pixel simply keeps its original raw
    # content, whatever natural limb-darkening/PSF signal it has, rather
    # than being replaced with background).
    #
    # An earlier version of this function instead mapped invalid points to
    # a sentinel far outside the image, so cv2.remap's BORDER_CONSTANT would
    # return background (0.0) there — reasoned as "don't silently re-sample
    # whatever raw content happens to sit at that screen position, since
    # it's not the same body location any more". That reasoning wasn't
    # wrong on its own terms, but it made this warp's edge behaviour
    # fundamentally different from (and much harder-edged than) the
    # validated linear warp's: a direct radial-profile comparison on real
    # Jupiter data showed the linear warp's brightness tapers smoothly for
    # ~20px past the fitted disk radius (still ~4% of peak at +15px beyond
    # disk_radius_px, the ordinary limb-darkening/PSF tail), while this
    # function's sentinel-based version dropped from ~20% to exactly 0.0
    # within about 10px — a hard, largely unfeathered cutoff that wavelet
    # sharpening amplified into a visible ring the user correctly flagged
    # as newly-introduced masking (this whole reprojection feature, and
    # thus this behaviour, did not exist before this session). Removing a
    # separate, later-added explicit re-masking pass in
    # spherical_derotation_warp_3d did NOT fix this on its own (verified:
    # reintroducing that pass via monkeypatch through the real
    # derotate_window()/derotate_filter() pipeline produced byte-identical
    # output to removing it) — the sentinel here was always the real
    # source, regardless of that other pass's presence.
    # _reprojection_point_shift() (single-point callers) still checks
    # `valid` before using new_x/new_y at all, so this change is invisible
    # to it — it already treats invalid as "(0, 0), no shift", exactly the
    # same philosophy this now extends to the full-image warp.
    #
    # IMPORTANT CAVEAT (external review, 2026-08-11): this identity fallback
    # is a deliberate visual-quality trade-off, NOT a physically accurate
    # far-side sample. The source body point genuinely isn't observable
    # there at the source time — geometrically, background/no-data would be
    # the "correct" answer. Identity fallback instead shows whatever this
    # frame's real content happened to be at that SCREEN position (limb
    # brightness tail / seeing PSF / diffraction), which is not the same
    # body location any more. This is the right call for how this module is
    # actually used (composite + wavelet sharpening punish a hard cutoff far
    # more than they punish a few px of slightly-stale limb content), but a
    # caller that needs true per-pixel source-visibility ground truth (e.g.
    # a future ring/shadow-geometry consumer) MUST use the returned `valid`
    # mask itself, not assume new_x/new_y are a real body-surface sample
    # wherever valid is False.
    new_x = np.where(valid, cx + dx1, x)
    new_y = np.where(valid, cy + dy1, y)
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


# ── Map-space (lat/lon) projection primitives (2026-08-16, Phase A of ────────
# project_map_space_derotation_roadmap) ───────────────────────────────────────
#
# Reuses _oblate_ortho_forward/_oblate_ortho_inverse exactly as-is (no new
# geometry math) -- these two functions just wrap them in the same
# "vectorize the whole grid, sample via cv2.remap, use depth>0 for validity"
# pattern _reprojected_position()/spherical_derotation_warp_3d() already use
# for disk-pixel-space de-rotation, applied instead to disk<->map conversion.
# Callers are responsible for the 5% req_px/rpol_px padding convention (same
# split of responsibility as _oblate_ortho_forward/inverse themselves --
# _reprojected_position applies the padding once and passes req_px/rpol_px
# down; these two functions do the same).
#
# Map grid convention (fixed by these two functions together -- a caller
# must use the same n_lat/n_lon for both the _disk_to_map() call that
# produced a map and any _map_to_disk() call consuming it):
#   phi (latitude):  row 0 = -90 deg, row n_lat-1 = +90 deg, linspace
#   lam (longitude): col 0 = -180 deg, col n_lon-1 = +180-360/n_lon deg,
#                    linspace with endpoint=False (col n_lon wraps to col 0)
#
# KNOWN LIMITATION (Phase A, not yet addressed): _map_to_disk()'s cv2.remap
# uses BORDER_WRAP so longitude (the column axis) wraps correctly at the
# +-180 deg seam -- but BORDER_WRAP applies to BOTH axes, and latitude (the
# row axis) is NOT physically periodic (the poles are hard edges, not
# adjacent to each other). In practice this only matters within <1px of
# phi=+-90 deg (bilinear interpolation's kernel support reaching past
# row 0 / row n_lat-1), since a real (phi,lam) from _oblate_ortho_inverse
# always lands in [0, n_lat-1] by construction (arcsin's range) -- a real
# defect, but a narrow one; not fixed here (would need two remap calls with
# different border modes, one per axis, combined -- deferred until Phase A's
# round-trip validation shows whether it matters in practice).


def _disk_to_map(
    disk_image: np.ndarray,
    cx: float,
    cy: float,
    req_px: float,
    rpol_px: float,
    pole_pa_deg: float,
    sub_observer_lat_deg: float,
    n_lat: int = 180,
    n_lon: int = 360,
    flip_pole_axis: bool = False,
    lam_shift: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project a disk-space image into an equirectangular (phi, lam) map.

    For every map cell (phi, lam), computes the screen position that
    body-fixed point projects to (via _oblate_ortho_forward) -- with one
    twist for multi-frame combination (2026-08-16, Phase B of
    project_map_space_derotation_roadmap): `lam` in the returned map's own
    indexing is always the REQUESTED grid value (unshifted) -- lam_shift
    only affects which screen position gets SAMPLED for that cell, via
    `lam_source = lam + lam_shift` fed to _oblate_ortho_forward. This lets a
    caller treat the map as expressed in one fixed (e.g. reference-time)
    orientation while sourcing each frame's own pixels at that frame's own
    sub-observer-meridian-relative longitude -- lam_shift must be computed
    exactly the way _reprojected_position() computes its `sign*scale*
    delta_lambda` term (same sign convention; this project has a history of
    exactly this kind of sign error, do not re-derive it independently).
    Default 0.0: byte-identical to before this parameter existed (a single
    frame's own map, no time-shift).

    Returns (map_image, valid_mask): valid_mask is 1.0 where that map cell
    is visible (near side, i.e. depth>0) AND its screen position falls
    inside the source image, 0.0 elsewhere (far side / off-image) --
    map_image is 0.0 wherever valid_mask is 0.0 (not a real sample, must be
    excluded by any caller combining multiple maps, not blended in).
    """
    h, w = disk_image.shape[:2]
    phi_vals = np.linspace(-math.pi / 2.0, math.pi / 2.0, n_lat)
    lam_vals = np.linspace(-math.pi, math.pi, n_lon, endpoint=False)
    lam_grid, phi_grid = np.meshgrid(lam_vals, phi_vals)  # both (n_lat, n_lon)

    dx, dy, depth = _oblate_ortho_forward(
        phi_grid, lam_grid + lam_shift, sub_observer_lat_deg, pole_pa_deg, req_px, rpol_px,
        flip_pole_axis=flip_pole_axis,
    )
    map_x = (cx + dx).astype(np.float32)
    map_y = (cy + dy).astype(np.float32)

    src = disk_image.astype(np.float32)
    sampled = cv2.remap(
        src, map_x, map_y, interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    )
    in_bounds = (map_x >= 0) & (map_x <= w - 1) & (map_y >= 0) & (map_y <= h - 1)
    valid = (depth > 0.0) & in_bounds
    if sampled.ndim == 3:
        valid_b = valid[:, :, np.newaxis]
    else:
        valid_b = valid
    map_image = np.where(valid_b, sampled, 0.0).astype(np.float32)
    valid_mask = valid.astype(np.float32)
    return map_image, valid_mask


def _map_to_disk(
    map_image: np.ndarray,
    valid_mask: np.ndarray,
    output_shape: Tuple[int, int],
    cx: float,
    cy: float,
    req_px: float,
    rpol_px: float,
    pole_pa_deg: float,
    sub_observer_lat_deg: float,
    flip_pole_axis: bool = False,
    lam_shift: float = 0.0,
) -> np.ndarray:
    """Inverse of _disk_to_map(): render a disk-space image from a (phi, lam)
    map, at the SAME orientation (sub_observer_lat_deg/pole_pa_deg) the map's
    (phi, lam) grid is expressed in -- a caller wanting the disk view at a
    DIFFERENT reference time must shift the map's own longitude coordinate
    (or, equivalently, shift lam before the _disk_to_map() calls that built
    it) the same way _reprojected_position()'s delta_lambda shift does; this
    function itself does no time/rotation handling.

    lam_shift (radians) is added to the recovered lam before sampling
    map_image, mirroring _disk_to_map()'s own lam_shift parameter exactly
    (same sign convention) -- used by tests to synthesize "what a frame
    captured at some other time would look like" from a single reference
    map, by applying the negation of the shift map_space_window_stack()
    would use to undo it.

    For every output disk pixel, unprojects to (phi, lam) (via
    _oblate_ortho_inverse), converts to map pixel coordinates using the
    SAME linspace convention _disk_to_map() uses, and samples map_image
    there (only where valid_mask says that map cell was real). Output
    pixels with no near-side (phi,lam) solution, or whose map cell has
    valid_mask<0.5, are 0.0.
    """
    h, w = output_shape
    n_lat, n_lon = map_image.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    dx0 = xx - cx
    dy0 = yy - cy
    phi, lam, depth = _oblate_ortho_inverse(
        dx0, dy0, sub_observer_lat_deg, pole_pa_deg, req_px, rpol_px,
        flip_pole_axis=flip_pole_axis,
    )
    lam = lam + lam_shift

    # Map the recovered (phi, lam) to fractional map-pixel coordinates,
    # inverse of _disk_to_map()'s linspace(-pi/2, pi/2, n_lat) /
    # linspace(-pi, pi, n_lon, endpoint=False).
    row = (phi + math.pi / 2.0) / math.pi * (n_lat - 1)
    col = ((lam + math.pi) % (2.0 * math.pi)) / (2.0 * math.pi) * n_lon
    map_x = col.astype(np.float32)
    map_y = row.astype(np.float32)

    src = map_image.astype(np.float32)
    sampled = cv2.remap(src, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    valid_sampled = cv2.remap(
        valid_mask.astype(np.float32), map_x, map_y,
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP,
    )

    out_valid = np.isfinite(phi) & (depth > 0.0) & (valid_sampled > 0.5)
    if sampled.ndim == 3:
        out_valid_b = out_valid[:, :, np.newaxis]
    else:
        out_valid_b = out_valid
    disk_image = np.where(out_valid_b, sampled, 0.0).astype(np.float32)
    return disk_image


_MAP_SPACE_LIMB_FEATHER_PX = 12.0  # same feather-width convention as
# _interp_feather_px (spherical_derotation_warp_3d) and _RING_DEPTH_FEATHER_PX
# (ring occlusion) -- not re-derived, reused for consistency.


def map_space_window_stack(
    included_rows: List[dict],
    t_reference: datetime,
    period_hours: float,
    cx: float,
    cy: float,
    disk_radius_px: float,
    pole_pa_deg: float,
    sub_observer_lat_deg: float,
    polar_equatorial_ratio_true: float,
    flip_direction: bool = False,
    flip_pole_axis: bool = False,
    n_lat: int = 180,
    n_lon: int = 360,
) -> Tuple[np.ndarray, dict]:
    """Phase B of project_map_space_derotation_roadmap: reconstruct a
    de-rotated disk view by projecting each frame into a COMMON (phi, lam)
    map expressed at t_reference's orientation, combining with a SIMPLE
    (unweighted, per-cell) average, then reprojecting the combined map back
    to a disk view at t_reference.

    Deliberately the simplest possible map-space combination -- no quality
    weighting (Phase C), no ring exclusion (Phase D: has_rings targets
    should NOT use this yet, ring pixels would get unprojected as if they
    were globe surface and produce nonsense). This function exists to test
    the core hypothesis (does map-space combination alone produce a
    sensible result?) before adding complexity on top of it. Not wired into
    derotate_filter()/derotate_window() (Phase E) -- call directly.

    Each frame's own longitude shift relative to t_reference is computed
    EXACTLY the way _reprojected_position() computes its lam shift (same
    formula, same sign convention, scale fixed at 1.0 -- per spherical_
    derotation_warp_3d's own documented finding that the true reprojection
    has no first-order approximation error for a `scale` parameter to
    absorb, unlike the linear warp): `lam_shift = sign*delta_lambda`,
    `sign = 1.0 if flip_direction else -1.0`. This is passed to _disk_to_map()
    as `lam_shift` so lam_shift=0 exactly reproduces the reference frame's
    own map, and other frames sample their own sub-observer-meridian-
    relative longitude for the same map cell.

    Limb feathering (added after Phase A's real-data validation found a
    hard, unfeathered cutoff at the limb in the naive round-trip -- see
    project_map_space_derotation_roadmap memory): the combined disk output
    is tapered to 0 over _MAP_SPACE_LIMB_FEATHER_PX pixels approaching
    disk_radius_px, the same feather-width convention already used by
    spherical_derotation_warp_3d's cubic/linear blend and the ring-occlusion
    feathering -- this is a fixed-radius approximation to the true
    (oblate, occlusion-shaped) valid-region boundary, matching how
    spherical_derotation_warp_3d's own limb feather already uses a plain
    radial distance rather than the exact silhouette shape.

    Returns (disk_image, info): info carries `n_stacked` (frames actually
    combined) and `coverage_mean`/`coverage_min` (the combined per-map-cell
    coverage fraction, analogous to compute_frame_coverage_mask's n(x) for
    the disk-pixel-space path) for diagnostic/logging parity with the
    existing pipeline.
    """
    req_px = disk_radius_px * 1.05
    rpol_px = req_px * polar_equatorial_ratio_true
    period_sec = period_hours * 3600.0
    sign = 1.0 if flip_direction else -1.0

    accum: Optional[np.ndarray] = None
    weight_sum: Optional[np.ndarray] = None
    out_shape: Optional[Tuple[int, int]] = None
    n_stacked = 0

    for row in included_rows:
        img = image_io.read_tif(row["path"])
        # Mono only for Phase B (proving the core hypothesis) -- color/mono
        # handling parity with derotate_filter() is a Phase E concern.
        lum = img.mean(axis=2) if img.ndim == 3 else img
        if out_shape is None:
            out_shape = lum.shape[:2]

        dt_sec = (row["timestamp"] - t_reference).total_seconds()
        delta_lambda = (dt_sec / period_sec) * 2.0 * math.pi
        lam_shift = sign * delta_lambda

        map_img, valid_mask = _disk_to_map(
            lum.astype(np.float32), cx, cy, req_px, rpol_px,
            pole_pa_deg, sub_observer_lat_deg,
            n_lat=n_lat, n_lon=n_lon, flip_pole_axis=flip_pole_axis,
            lam_shift=lam_shift,
        )
        if accum is None:
            accum = np.zeros(map_img.shape, dtype=np.float64)
            weight_sum = np.zeros(valid_mask.shape, dtype=np.float64)
        accum += map_img.astype(np.float64) * valid_mask
        weight_sum += valid_mask.astype(np.float64)
        n_stacked += 1

    if accum is None or out_shape is None:
        raise ValueError("No images to stack")

    combined_map = np.where(weight_sum > 1e-9, accum / np.maximum(weight_sum, 1e-9), 0.0).astype(np.float32)
    combined_valid = (weight_sum > 1e-9).astype(np.float32)

    disk_out = _map_to_disk(
        combined_map, combined_valid, out_shape, cx, cy, req_px, rpol_px,
        pole_pa_deg, sub_observer_lat_deg, flip_pole_axis=flip_pole_axis,
    )

    h, w = out_shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    dist_from_center = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    limb_alpha = np.clip(
        (disk_radius_px + _MAP_SPACE_LIMB_FEATHER_PX - dist_from_center) / _MAP_SPACE_LIMB_FEATHER_PX,
        0.0, 1.0,
    ).astype(np.float32)
    disk_out = (disk_out * limb_alpha).astype(np.float32)

    info = {
        "n_stacked": n_stacked,
        "coverage_mean": round(float(combined_valid.mean()), 4),
        "coverage_min": round(float(combined_valid.min()), 4),
    }
    return disk_out, info


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
    ring_crossing_mask: Optional[np.ndarray] = None,
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

    ring_crossing_mask: Optional continuous [0,1] float array, same (h, w)
                        as image, from compute_ring_occlusion_weight_3d()
                        (2026-08-15 — NOT compute_ring_occlusion_weight(),
                        which is depth-calibrated for the LINEAR warp; see
                        that function's docstring for why the two aren't
                        interchangeable). 1.0 = pixel is occluded by
                        FOREGROUND ring material; 0.0 = ordinary atmosphere.
                        None (default) is a complete no-op, byte-identical
                        to every existing caller. Mechanism differs from
                        spherical_derotation_warp()'s (which damps an
                        intermediate depth_map/drift quantity before it
                        forms map_x/map_y, equivalent there to blending the
                        fetch COORDINATES toward identity only because that
                        warp's map_x is a strictly linear function of one
                        scalar depth value): this function instead blends
                        the fully-resolved PIXEL VALUES (the completed warp
                        vs. this frame's own untouched content) at the very
                        end, AFTER the cubic/linear interpolation blend —
                        found necessary by real-Saturn-data visual
                        inspection (2026-08-15): blending map_x/map_y
                        directly (mirroring the linear warp's mechanism)
                        samples a physically meaningless in-between screen
                        position wherever this warp's highly nonlinear
                        reprojection is strongly curved, producing a bright
                        seam at the globe/ring boundary and a visible break
                        where the ring exits the disk silhouette. Applies
                        the same two-step (output-side, then source-side
                        fetch-point leak) structure as spherical_derotation_
                        warp()'s ring_crossing_mask handling, just as a
                        value blend instead of a coordinate blend.

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

    # `valid` (3rd return) is intentionally unused here — see the NOTE below
    # on why this no longer re-masks the output with it directly.
    new_x, new_y, _valid = _reprojected_position(
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

    # No separate masking of `invalid` pixels here — _reprojected_position()
    # itself now falls back to identity (no shift) for them, so map_x/map_y
    # already sample this frame's own original content there (see its
    # docstring, 2026-08-11, for why: the real fix for the ring the user
    # reported was there, not an extra masking pass in this function, which
    # was tried first and did not actually change the output).

    if ring_crossing_mask is not None:
        # BUG FOUND 2026-08-15 (real-data visual inspection, before this ever
        # shipped): an earlier version of this block blended the FETCH
        # COORDINATES (map_x/map_y) toward identity by the mask -- exactly
        # mirroring spherical_derotation_warp()'s mechanism, which is
        # algebraically equivalent to damping its depth_map there ONLY
        # because that warp's map_x is a strictly LINEAR function of one
        # scalar depth value. This warp's map_x/map_y are a highly nonlinear
        # function of the full oblate-spheroid reprojection (longitude
        # shift -> reproject), so a straight-line blend between the
        # "rotated" and "identity" coordinates samples a screen position
        # that corresponds to neither -- a physically meaningless
        # in-between point wherever the reprojection is strongly curved.
        # Rendered on real Saturn data this produced a bright seam right at
        # the globe/ring boundary and a visible break where the ring exits
        # the disk silhouette -- exactly where curvature is highest. Fixed
        # by blending the fully-resolved PIXEL VALUES instead (the already-
        # completed `warped` render vs. this frame's own untouched
        # `src_f32`), a plain alpha-composite of two independently correct
        # images -- same category of blend already used elsewhere in this
        # module for an analogous "smoothly trust one interpretation over
        # another" problem (the warped_cubic/warped_linear blend just
        # above, and derotate_filter's S0/S_L stacking blend).
        ring_b = ring_crossing_mask[:, :, np.newaxis] if warped.ndim == 3 else ring_crossing_mask
        warped = warped * (1.0 - ring_b) + src_f32 * ring_b

        # Source-side leak (mirrors spherical_derotation_warp()'s fix): an
        # ordinary-atmosphere OUTPUT pixel's fetch point can still land on
        # foreground-ring territory. Resample the mask at the (unmodified)
        # fetch point and blend the VALUE further toward this frame's own
        # original content -- same value-blend principle as above, not a
        # further coordinate adjustment.
        _src_ring_weight = cv2.remap(
            ring_crossing_mask, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
        )
        _src_b = _src_ring_weight[:, :, np.newaxis] if warped.ndim == 3 else _src_ring_weight
        warped = warped * (1.0 - _src_b) + src_f32 * _src_b

    return np.clip(warped, 0.0, 1.0)


def compute_frame_coverage_mask(
    h: int,
    w: int,
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
    """Per-pixel bool: True where this frame's source content, AT ITS OWN
    CAPTURE TIME, was a genuine on-globe, rotation-valid sample (as opposed
    to identity-fallback content from a stale epoch).

    Deliberately keeps two checks SEPARATE rather than collapsing them into
    a single "valid" test, per a 2026-08-13 attempt this session that used
    _reprojected_position()'s combined `valid` directly as a per-pixel
    STACKING weight, found a real bug, and was fully reverted (zero trace
    left in this file — confirmed via `git log -S"return_valid"` on this
    module returning no hits):

      1. on_globe_domain = isfinite(phi) from _oblate_ortho_inverse — this
         is dt-INDEPENDENT: False wherever the OUTPUT position itself falls
         outside the padded oblate-spheroid model's domain (r beyond
         req_px = disk_radius_px*1.05), true for EVERY frame regardless of
         rotation. On a ringed planet this is exactly where the ring system
         (and background sky) lives — nothing to do with atmosphere
         rotation staleness at all.
      2. rotation_valid = _reprojected_position()'s own `valid` (isfinite
         (phi) & source_depth>0) — dt-DEPENDENT: the genuine "this frame's
         source had already rotated to the far side at ITS OWN capture
         time" signal.

    The reverted attempt conflated these (used the combined `valid` alone),
    which wrongly collapsed non-reference frames' weight across the ENTIRE
    ring system and background sky, not just the true rotation-invalid
    band near the limb. Fixed here by reporting True for off-domain pixels
    ("not applicable — full coverage"), so this signal is only ever
    resisted by real, dt-dependent rotation invalidity:

        return rotation_valid | ~on_globe_domain

    This time the signal feeds sharpening gain and an S0/S_L stacking
    blend (see quality_weighted_stack's docstring history and
    derotate_filter's compute_coverage_map/s0_sl_blend_enabled), not the
    per-pixel stacking weight itself — the earlier finding that this kind
    of signal barely moved the STACK's own sharpness (real rotation-
    invalid fraction, once correctly isolated, was only 0-5% in the
    affected band) does not apply to these different consumers.

    KNOWN GAP, not fixed here (2026-08-15): this function has no notion of
    Saturn's ring occlusion (see compute_ring_occlusion_weight_3d) — a pixel
    held at identity because FOREGROUND ring material occludes it (not
    because of rotation invalidity) still reports on_globe_domain=True and
    typically rotation_valid=True here, so n(x) over-reports genuine
    coverage there. Not addressed in the same change that wired ring
    occlusion into spherical_derotation_warp_3d(), since compute_coverage_map/
    s0_sl_blend_enabled and has_rings are not combined in this session's
    production config — revisit if that combination is ever enabled.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    req_px = disk_radius_px * 1.05
    rpol_px = req_px * polar_equatorial_ratio_true
    phi, _lam, _depth = _oblate_ortho_inverse(
        xx - cx, yy - cy, sub_observer_lat_deg, pole_pa_deg, req_px, rpol_px,
        flip_pole_axis=flip_pole_axis,
    )
    on_globe_domain = np.isfinite(phi)

    _new_x, _new_y, rotation_valid = _reprojected_position(
        xx, yy, dt_sec, cx, cy, disk_radius_px, period_hours,
        sub_observer_lat_deg, pole_pa_deg, polar_equatorial_ratio_true,
        scale=scale, flip_direction=flip_direction, flip_pole_axis=flip_pole_axis,
    )
    return rotation_valid | ~on_globe_domain


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
        pole_pa_deg:            Image-space equatorial/drift-axis angle from
                                auto_detect_equator_pa() (not a pole-axis PA).
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


# Safety clamp for the pre-warp scale correction below -- NOT a physical
# claim that real frame-to-frame disk radius never varies by more than 5%,
# just a generous margin above the largest real variation measured this
# session (~2.15%, Saturn window_01/R) to reject wild affine distortion from
# a bad disk-shape fit rather than trust it.
_SCALE_MIN = 0.95
_SCALE_MAX = 1.05

# Smoothing applied ONCE to the aggregated multi-frame coverage map n(x)
# (derotate_filter's compute_coverage_map), not per-frame -- summing several
# frames' hard per-pixel boolean coverage (compute_frame_coverage_mask),
# each at a slightly different dt-dependent boundary radius, already grades
# the field into up to N discrete levels without any per-frame feathering
# (deliberately NOT repeating the 2026-08-13 reverted attempt's per-frame
# signed-distance feather, which leaked ~50% weight onto invalid pixels).
# A single small blur on the final sum removes the remaining discrete-level
# steps so a per-pixel wavelet-gain multiplier or S0/S_L blend weight
# derived from it doesn't itself introduce a new sharp edge for unsharp-
# mask sharpening to ring on. 2.0px matches this module's own established
# feather-scale convention (_RING_DEPTH_FEATHER_PX, spherical_derotation_
# warp's _interp_feather_px) for the same class of problem.
_COVERAGE_SMOOTH_SIGMA_PX = 2.0


def apply_shift_and_scale(
    image: np.ndarray,
    target_cx: float,
    target_cy: float,
    ref_cx: float,
    ref_cy: float,
    scale: float,
) -> np.ndarray:
    """Map this frame's own measured disk centre/radius onto the reference
    frame's geometry: output = ref_center + scale * (input - target_center).

    Added 2026-08-11 (real-data investigation: the Cassini Division washes
    out in a multi-frame Saturn stack even though it's visible in a single
    frame). Root cause confirmed empirically: pre-warp alignment only ever
    corrected TRANSLATION between frames — each frame's own apparent disk
    RADIUS (from find_disk_center) varies by ~1-2% frame-to-frame in real
    data (seeing/focus, not a bug), which is under 1px at the disk itself
    but becomes 1.6-2.8px at the Cassini Division's radius (~2x the disk
    radius) -- more than enough to blur out a gap only a few px wide when
    several frames are averaged without correcting for it. (An earlier,
    physically incorrect diagnosis blamed uncorrected ring orbital motion
    instead -- wrong, because the Division is an axisymmetric radial
    structure, I(r,lambda)=I(r), and shifting material in longitude alone
    is a no-op on such a structure by definition; that plan was discarded
    before implementation.)

    BUG CAUGHT BEFORE SHIPPING (external review, math re-derived directly
    to confirm): scaling around ref_center and then adding the existing
    (ref_center - target_center) TRANSLATION-only shift is NOT the same
    transform as scaling around target_center and then translating to
    ref_center -- they only agree at scale=1.0. This function implements
    the latter (pivot the scale at the TARGET's own centre first), which
    is what tests/test_apply_shift_and_scale.py's center-mapping test
    verifies directly (target_center must map exactly to ref_center).

    A frame-to-frame ROTATION correction was also tried here (2026-08-13/14,
    real-data investigation: even after the scale fix above, a 3-frame
    Saturn stack's globe edge was measurably blurrier than a single
    near-reference frame put through the identical pipeline) but REVERTED
    after real-data verification: it made no measurable difference to the
    Saturn transition width/sharpness, and produced a small but consistent
    regression on Jupiter's 45-combo sharpness sweep (median delta -0.0015,
    worst-case -0.0297, CH4 systematically worse across nearly every
    window). Conclusion: the measured 3-4 degree frame-to-frame ellipse
    angle_major swing is very likely contour-fit noise, not real rotation --
    "correcting" for it just adds an extra interpolation pass that softens
    real image content without fixing anything. See project memory for the
    full investigation; do not re-add this without new evidence the angle
    measurement itself is reliable.
    """
    M = cv2.getRotationMatrix2D((float(target_cx), float(target_cy)), 0.0, float(scale))
    M[0, 2] += float(ref_cx) - float(target_cx)
    M[1, 2] += float(ref_cy) - float(target_cy)
    h, w = image.shape[:2]
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


def _laplacian_var_central(
    image: np.ndarray,
    cx: float,
    cy: float,
    semi_a: float,
    radius_frac: float = 0.55,
) -> Optional[float]:
    """Pure Laplacian-variance sharpness over the central disk interior, at
    an ALREADY-KNOWN geometry (cx, cy, semi_a) -- no disk detection here,
    so a caller that already ran find_disk_center()/_find_disk_center_impl
    for other reasons doesn't pay for it twice. See frame_sharpness_central()
    below for the full justification and real-data evidence for this metric.
    """
    h, w = image.shape[:2]
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = r < (radius_frac * semi_a)
    if mask.sum() < 25:
        return None
    lap = cv2.Laplacian(image.astype(np.float32), cv2.CV_32F, ksize=3)
    return float(np.var(lap[mask]))


def frame_sharpness_central(image: np.ndarray, radius_frac: float = 0.55) -> Optional[float]:
    """Public wrapper: runs find_disk_center() then delegates to
    _laplacian_var_central(). Returns None if disk detection fails
    (semi_a < 5px) -- callers MUST treat None as "unmeasurable", never
    coerce it to 0.0 (that reads as "worst frame" and would wrongly exclude
    an undetectable-but-otherwise-fine frame, e.g. CH4's known-unreliable
    disk fit).

    Why this metric (not quality.laplacian_var() / norm_score) -- a
    2026-08-13 real-data investigation into why the final multi-frame
    stack is measurably blurrier than the single sharpest raw frame in the
    same window (a gap present deep in the disk interior, i.e. unrelated
    to de-rotation warp coverage/edge effects, and present equally on both
    Saturn and Jupiter -- so not a target-specific or coordinate-system
    issue) found: stacking only the top-half of a window's included frames
    BY THIS METRIC raised the median (stack_sharpness / best_single_raw_
    frame_sharpness) ratio from 0.733 to 0.878 across 51 real window x
    filter combos (mixed Saturn/Jupiter). A same-frame-COUNT bottom-half
    control (worst frames instead of best) was WORSE than the unfiltered
    baseline in 5/6 spot checks -- proving the effect is about WHICH
    frames are kept, not merely averaging fewer of them (a real per-frame
    seeing/PSF heterogeneity signal, not a sample-size artifact).
    norm_score (pipeline.modules.quality.quality_metrics()/compute_scores())
    already tries to measure sharpness via a 0.5*laplacian + 0.3*tenengrad
    + 0.2*norm_variance blend, computed after a 1.2-sigma Gaussian denoise,
    over the FULL Otsu-detected disk mask (including the limb) -- but was
    separately measured this session to correlate ~-0.10 (near-zero, wrong
    sign) with real per-frame sharpness. Three differences are deliberately
    NOT carried over here, as the suspected causes: no pre-metric denoise
    (likely smooths away exactly the fine detail that distinguishes good
    frames from mediocre ones); central-radius-only mask, not the full
    disk (the limb's limb-darkening/seeing-smeared low-contrast pixels
    dilute the signal); pure Laplacian variance only, not a 3-metric blend.
    """
    cx, cy, semi_a, _semi_b, _angle = find_disk_center(image)
    if semi_a < 5:
        return None
    return _laplacian_var_central(image, cx, cy, semi_a, radius_frac)


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

    NOTE on true-reprojection far-side pixels (updated 2026-08-11, was
    stale): spherical_derotation_warp_3d() does NOT zero out far-side
    pixels — as of the 2026-08-11 fix it falls back to identity (keeps
    each frame's own original content) for source-far-side points, exactly
    like the validated linear warp already does for its analogous
    depth_sq<=0 case (see _reprojected_position's docstring for why the
    earlier zero/sentinel behaviour was reverted — it caused a visible
    ring). So this function's stacking has nothing extra to account for
    here: every frame contributes its own real (if slightly stale, right
    at the tilt-limb) content rather than an artificial 0, and a plain
    weighted mean is exactly as appropriate for that as for any other pixel.
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
    has_rings: bool = False,
    sharpness_selection_enabled: bool = False,
    sharpness_keep_fraction: float = 1.0,
    compute_coverage_map: bool = False,
    s0_sl_blend_enabled: bool = False,
    compute_ring_only_stack: bool = False,
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
        sub_observer_lat_deg: Horizons "ObsSub-LAT" (quantity 14) — used
                        when use_true_reprojection=True, AND (2026-08-11)
                        when has_rings=True, to build the ring occlusion
                        weight (see compute_ring_occlusion_weight). Unused
                        otherwise.
        true_polar_equatorial_ratio: Planet's TRUE physical Rpol/Req — only
                        used when use_true_reprojection=True (replaces the
                        apparent ellipse-fit ratio the linear warp uses).
        flip_pole_axis: Sign-ambiguity escape hatch for the reprojection —
                        only used when use_true_reprojection=True.
        has_rings:      If True, compute a ring occlusion weight from this
                        filter's own resolved geometry and
                        sub_observer_lat_deg, and scale down the warp's
                        atmosphere rotation only where FOREGROUND ring
                        material actually occludes the view. Wired into
                        WHICHEVER warp is actually selected below: the
                        legacy linear warp via compute_ring_occlusion_weight()
                        (as spherical_derotation_warp()'s ring_crossing_mask),
                        or (2026-08-15) the true 3D reprojection via
                        compute_ring_occlusion_weight_3d() (as spherical_
                        derotation_warp_3d()'s ring_crossing_mask, using the
                        same depth convention as that warp instead of the
                        linear warp's approximation). Previously this was
                        computed with the linear-warp version unconditionally
                        and never passed to spherical_derotation_warp_3d() at
                        all — silently inert for use_true_reprojection=True
                        sessions (this session's actual production Saturn
                        config: has_rings=True AND use_true_reprojection=True
                        simultaneously — the premise that "Saturn uses the
                        linear warp by default" no longer holds). Set by
                        derotate_window()/derotate_stack.py only for Saturn
                        (this uses Saturn-specific physical ring constants,
                        meaningless for other targets). Default False leaves
                        every existing caller's behaviour unchanged.
        sharpness_selection_enabled, sharpness_keep_fraction: Opt-in (see
                        frame_sharpness_central()'s docstring for the
                        2026-08-13 real-data justification). When enabled,
                        excludes the least-sharp (by raw measured Laplacian
                        variance, NOT norm_score) fraction of non-reference
                        included frames from the stack — a window-local,
                        per-filter selection, applied after outlier-shift
                        rejection. sharpness_keep_fraction=1.0 (default) or
                        sharpness_selection_enabled=False (default) is a
                        complete no-op, byte-identical to before this
                        parameter existed.
        compute_coverage_map, s0_sl_blend_enabled: Opt-in (see
                        compute_frame_coverage_mask()'s docstring). Only
                        meaningful when use_true_reprojection=True — a
                        per-pixel de-rotation coverage signal n(x) doesn't
                        exist for the linear warp, so both are silently
                        inert (no-op) otherwise. compute_coverage_map alone
                        only computes and logs n(x) (e.g. for step05's
                        coverage-aware sharpening to consume) without
                        touching the stack itself. s0_sl_blend_enabled
                        additionally blends the stack toward the reference
                        frame's own (dt=0, no warp) rendering wherever n(x)
                        is low: result = alpha(x)*stack + (1-alpha(x))*
                        ref_img, alpha(x) = coverage_to_confidence(n(x),
                        floor=0.0) — alpha->0 falls back EXACTLY to the
                        reference frame where coverage is poor, so the
                        result is never worse than the single-reference-
                        frame baseline there by construction. Both default
                        False: complete no-op, byte-identical to before
                        these parameters existed.
        compute_ring_only_stack: Opt-in (see DerotationConfig.compute_ring_
                        only_stack's docstring). Only meaningful when
                        has_rings=True; silently inert otherwise. When both
                        are True, a second per-filter stack is computed and
                        returned via log_dict["ring_only_stack"] (raw
                        ndarray, must be popped before JSON — same contract
                        as coverage_map/ring_crossing_mask). Does not affect
                        `stacked` (the atmosphere stack) at all. Default
                        False: complete no-op, byte-identical to before this
                        parameter existed.

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
    # This filter's own independent measurement, before any pose_registered/
    # radius_shared/shape_shared override below — logged per-frame (external
    # review, 2026-08-11) so a future session can check whether
    # resolve_shared_radius()'s median was actually pulled toward a
    # low-confidence outlier fit, rather than guessing.
    _own_semi_a = ref_semi_a
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

    # Ring occlusion weight (2026-08-11, real-Saturn-data + user-identified
    # architecture gap — see compute_ring_occlusion_weight's module note).
    # Only attempted when the caller explicitly says this target has rings
    # (has_rings=True, set by derotate_window()/derotate_stack.py only for
    # Saturn) — this uses Saturn-specific physical ring/Req constants, so it
    # would be meaningless (and, worse, silently wrong) applied to Jupiter or
    # any other target using this same shared function. Computed once per
    # filter here (not per frame): it depends only on this filter's own
    # resolved geometry and the window's B, not on any individual frame's
    # content.
    # See ring_crossing_mask's docstring entry in spherical_derotation_warp()
    # for the source-side fetch-point check (2026-08-11, external review)
    # that also applies this mask to prevent the inverse remap from ever
    # sampling foreground-ring content for an ordinary-atmosphere pixel.
    #
    # 2026-08-15 (external review, production-config gap): this filter's own
    # ring_crossing_mask is now computed with WHICHEVER warp is actually
    # used below -- compute_ring_occlusion_weight_3d() (matching depth
    # convention) + true_polar_equatorial_ratio/flip_pole_axis for the true
    # 3D reprojection, or the original compute_ring_occlusion_weight() for
    # the legacy linear warp. Before this, ring_crossing_mask was ALWAYS the
    # linear-warp version and was simply never passed to spherical_
    # derotation_warp_3d() at all -- silently inert for every session with
    # use_true_reprojection=True AND has_rings=True (this session's actual
    # production Saturn config, confirmed via ~/.astropipe/session.json).
    ring_crossing_mask: Optional[np.ndarray] = None
    ring_crosses_disk = False
    if has_rings:
        _rh, _rw = _ref_lum.shape[:2]
        if use_true_reprojection:
            ring_crossing_mask = compute_ring_occlusion_weight_3d(
                _rh, _rw, ref_cx, ref_cy, ref_semi_a, ref_semi_b,
                pole_pa_deg, sub_observer_lat_deg,
                polar_equatorial_ratio_true=true_polar_equatorial_ratio,
                flip_pole_axis=flip_pole_axis,
            )
        else:
            ring_crossing_mask = compute_ring_occlusion_weight(
                _rh, _rw, ref_cx, ref_cy, ref_semi_a, ref_semi_b,
                pole_pa_deg, sub_observer_lat_deg,
            )
        ring_crosses_disk = bool((ring_crossing_mask > 0.5).any())

    # Re-checked locally rather than trusted from the caller (matches this
    # function's existing style, e.g. ring_crosses_disk above is
    # independently re-derived) -- the coverage signal only exists for the
    # 3D reprojection warp (see compute_frame_coverage_mask's docstring).
    _do_coverage = compute_coverage_map and use_true_reprojection

    # Ring-only stack (2026-08-16, opt-in, Phase 1 of
    # project_ring_globe_layer_separation_roadmap) -- see DerotationConfig.
    # compute_ring_only_stack's docstring. The annulus window depends only
    # on the reference frame's own resolved geometry (same as ring_crossing_
    # mask above), so it's built once here, not per frame.
    _do_ring_only_stack = has_rings and compute_ring_only_stack
    _ring_win: Optional[np.ndarray] = None
    if _do_ring_only_stack:
        _rh3, _rw3 = _ref_lum.shape[:2]
        # feather=False: see _ring_annulus_mask's docstring -- the smoothly
        # feathered version measurably corrupts subpixel_align()'s result
        # here (empirically verified wrong by several px at some shift
        # magnitudes); the hard mask measures correctly across every shift
        # tested despite its sharp edge.
        _ring_win = _ring_annulus_mask(
            _rh3, _rw3, ref_cx, ref_cy, ref_semi_a, pole_pa_deg, sub_observer_lat_deg,
            feather=False,
        )

    warped_images: List[np.ndarray] = []
    weights: List[float] = []
    coverage_masks: List[np.ndarray] = []
    ring_only_images: List[np.ndarray] = []
    log_frames: List[dict] = []
    # Pre-warp disk center shifts: measured from raw frame before any warp is
    # applied, so the shift is purely seeing-induced wobble and is not
    # contaminated by the warp-induced brightness redistribution that biases
    # post-warp limb_center_align toward partially undoing the de-rotation.
    _pre_warp_shifts: dict = {}   # stem → (dx, dy, scale, target_cx, target_cy)

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
        # BUG FIXED 2026-08-11 (real-data investigation, user pushback on
        # weight_power as a band-aid): this used to gate on `shared_shape is
        # not None` — a WINDOW-level flag (true whenever ANY filter in this
        # window needed shape-sharing, e.g. CH4's brightness inversion) —
        # even though it decides a PER-FILTER question (should THIS filter's
        # own frames use find_disk_center or the correlation fallback for
        # pre-warp alignment). Since shared_shape is passed identically to
        # every filter's derotate_filter() call, ONE filter (CH4) needing the
        # fallback silently downgraded every OTHER filter in the same window
        # too — even ones with their own reliable ellipse fit (confidence
        # ~0.40, shape_reliable=True on real Saturn IR/R/G/B). Measured
        # impact directly: synthetic known-shift recovery on real Saturn
        # frames gives find_disk_center() error <0.09px vs subpixel_align()
        # (phase correlation) error 0.15-0.6+px on the SAME content — this
        # codebase's own ellipse-fit approach is already far more precise
        # here than frame-to-frame correlation (the same reason WinJUPOS
        # fits a model to the whole limb boundary rather than correlating
        # patches of internal texture). The correlation fallback below is
        # still exactly right for a filter whose OWN detection is actually
        # unreliable (CH4) — just no longer forced onto filters that don't
        # need it. Gate on THIS filter's own _rshape_ok (already resolved
        # above from this filter's own reference-frame fit), not the
        # window-wide shared_shape flag.
        _raw_lum = _to_luminance(img)
        _row_sharpness: Optional[float] = None
        try:
            if shared_shape is not None and not _rshape_ok:
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
                    # No independent radius measurement in this fallback
                    # path (subpixel_align only ever returns a translation),
                    # so scale correction is not attempted here — this is
                    # already the branch for filters whose own shape/size
                    # detection isn't trusted (see the block above), so
                    # trying to derive a scale factor from it would be
                    # trusting exactly the measurement already ruled out.
                    # At scale=1.0, apply_shift_and_scale(p) = ref_center +
                    # (p - target_center), so to reproduce the original pure
                    # translation-by-(dx,dy) behaviour exactly, target_center
                    # must be ref_center - (dx,dy), NOT ref_center itself
                    # (that would collapse to the identity transform).
                    _pre_warp_shifts[row["stem"]] = (_dx, _dy, 1.0, ref_cx - _dx, ref_cy - _dy)
            else:
                _cx_i, _cy_i, _semi_i, _semi_b_i, _angle_i, _conf_i, _shape_ok_i = (
                    _find_disk_center_impl(_raw_lum)
                )
                if _semi_i >= 5:
                    # Raw per-frame sharpness (2026-08-13, see
                    # frame_sharpness_central()'s docstring): reuses the
                    # geometry this branch already computed above, no
                    # second disk-detection pass. Independent of the dx/dy
                    # sanity gate below (sharpness doesn't need a valid
                    # pre-warp shift to be meaningful).
                    _row_sharpness = _laplacian_var_central(_raw_lum, _cx_i, _cy_i, _semi_i)
                    _dx = float(ref_cx - _cx_i)
                    _dy = float(ref_cy - _cy_i)
                    if abs(_dx) <= 15.0 and abs(_dy) <= 15.0:
                        # Scale correction (2026-08-11, real-data
                        # investigation — see apply_shift_and_scale's
                        # docstring for why): map this frame's own measured
                        # radius onto _own_semi_a, THIS FILTER's own raw
                        # reference-frame radius -- not ref_semi_a, which by
                        # this point may already reflect cross-filter
                        # radius_shared/shape_shared consensus adjustments
                        # (see _own_semi_a's own definition above) that have
                        # nothing to do with registering this frame against
                        # its own filter's reference frame. Gated on this
                        # frame's own fit confidence (mirrors the existing
                        # translation sanity gate below) and clamped to a
                        # safety range well outside the largest real
                        # variation measured this session (~2.15%) — this
                        # range is a defensive backstop against a bad fit
                        # producing a wild affine distortion, not a physical
                        # claim that real scale never varies more than 5%.
                        _can_scale = _conf_i > 0.0 and np.isfinite(_semi_i)
                        _scale = float(_own_semi_a) / _semi_i if _can_scale else 1.0
                        if not (_SCALE_MIN <= _scale <= _SCALE_MAX):
                            _scale = 1.0
                        _pre_warp_shifts[row["stem"]] = (_dx, _dy, _scale, _cx_i, _cy_i)
        except Exception:
            pass

        # BUG FIXED 2026-08-11 (real-data investigation, user-reported step05
        # sharpness regression vs step07): the measured pre-warp shift used to
        # be applied AFTER warping, as a rigid apply_shift() on the already-
        # warped image (see the second alignment loop below). spherical_
        # derotation_warp()/_3d() is a spatially-varying, depth-dependent
        # deformation defined relative to the ASSUMED centre (ref_cx, ref_cy)
        # — it has no way to know this raw frame's true disk centre actually
        # sits (_dx, _dy) away from that assumption, so warping an off-centre
        # frame and rigidly shifting the RESULT afterward is NOT equivalent to
        # recentring the frame first and then warping (these do not commute
        # for a non-rigid transform). Measured on real Saturn data: this left
        # a median 0.58px (up to 1.38px) residual misalignment on a ~130px
        # disk after "correction" — enough to measurably blur the stack.
        # Fix: apply the pre-warp shift to the RAW frame here, before it ever
        # reaches the warp, so the warp itself always operates on a correctly-
        # centred frame. The post-warp application below is now a no-op for
        # any stem with a pre-warp shift (kept only for the limb_center/
        # phase_correlate fallback chain, which inherently needs the warped
        # image to measure against).
        # Ring-only registration (2026-08-16, opt-in, Phase 1 of
        # project_ring_globe_layer_separation_roadmap) -- computed from the
        # ORIGINAL raw `img` (before the atmosphere pre-warp reassignment
        # right below), using a fresh subpixel_align() measurement windowed
        # to the ring annulus (_ring_win, a HARD mask -- see
        # _ring_annulus_mask's docstring for why the feathered version is
        # wrong for this purpose) instead of the globe-based (_dx, _dy)
        # above. Reuses that globe-based measurement's SCALE only (see
        # DerotationConfig.compute_ring_only_stack's docstring for why scale
        # isn't re-derived independently here).
        #
        # SANITY-CHECKED against the globe-based shift, not used blindly:
        # empirically (synthetic tests), unwindowed phase correlation on
        # this annulus mask is reliable for sub-~1px shifts but can lock
        # onto a badly wrong value at larger ones (verified wrong by
        # several px at some 3-5px true shifts, in both a real-ring-photo-
        # like noise texture and a purely elliptical mask/content match --
        # not just an unrealistic test scenario). Since the ring and globe
        # physically move together (same seeing jitter), a large disagreement
        # between the two measurements means the ring lock failed, not that
        # the ring truly moved independently -- fall back to the (_globe_dx,
        # _globe_dy) shift in that case.
        #
        # CAVEAT (found in adversarial review, not yet fixed): when this
        # frame's own globe-based pre-warp measurement failed outright
        # (_pre_warp_shifts.get(stem) is None -- low fit confidence, radius
        # too small, |dx|/|dy| > 15px, or an exception), _globe_dx/_globe_dy
        # below are a hardcoded (0.0, 0.0), NOT the atmosphere path's actual
        # eventual correction for that frame -- the atmosphere path only
        # resolves that case LATER, in a separate post-warp limb_center/
        # phase_correlate fallback loop this block runs before and has no
        # access to. So for exactly the hardest frames, ring_only_stack can
        # silently fall back to an identity transform while the atmosphere
        # stack gets a real (possibly large) correction for the same frame
        # -- the "never worse than the atmosphere path" framing above is
        # only guaranteed relative to the GLOBE'S OWN pre-warp measurement,
        # not the atmosphere path's full fallback chain. Not fixed here
        # (would need restructuring to share that later fallback value);
        # flagging honestly since this feature has not yet shown a real-data
        # benefit to justify the added complexity (see project_ring_globe_
        # layer_separation_roadmap's Phase 1 real-data validation notes).
        _RING_SHIFT_DISAGREEMENT_PX = 2.0
        if _do_ring_only_stack:
            _pw_for_ring = _pre_warp_shifts.get(row["stem"])
            _globe_dx = _pw_for_ring[0] if _pw_for_ring is not None else 0.0
            _globe_dy = _pw_for_ring[1] if _pw_for_ring is not None else 0.0
            _ring_scale = _pw_for_ring[2] if _pw_for_ring is not None else 1.0
            try:
                _ring_dx, _ring_dy = subpixel_align(_ref_lum * _ring_win, _raw_lum * _ring_win)
                _plausible = (
                    abs(_ring_dx) <= 15.0 and abs(_ring_dy) <= 15.0
                    and math.hypot(_ring_dx - _globe_dx, _ring_dy - _globe_dy) <= _RING_SHIFT_DISAGREEMENT_PX
                )
                if not _plausible:
                    _ring_dx, _ring_dy = _globe_dx, _globe_dy
            except Exception:
                _ring_dx, _ring_dy = _globe_dx, _globe_dy
            ring_only_images.append(apply_shift_and_scale(
                img, ref_cx - _ring_dx, ref_cy - _ring_dy, ref_cx, ref_cy, _ring_scale,
            ))

        if row["stem"] in _pre_warp_shifts:
            _, _, _pw_scale, _pw_target_cx, _pw_target_cy = _pre_warp_shifts[row["stem"]]
            img = apply_shift_and_scale(img, _pw_target_cx, _pw_target_cy, ref_cx, ref_cy, _pw_scale)

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
                ring_crossing_mask=ring_crossing_mask,
            )
        else:
            warped = spherical_derotation_warp(
                img, dt_sec, ref_cx, ref_cy, ref_semi_a,
                period_hours=period_hours,
                scale=warp_scale,
                flip_direction=flip_direction,
                pole_pa_deg=pole_pa_deg,
                polar_equatorial_ratio=_polar_eq_ratio,
                ring_crossing_mask=ring_crossing_mask,
            )

        _pw_entry = _pre_warp_shifts.get(row["stem"])
        log_frames.append({
            "stem":              row["stem"],
            "timestamp":         row["timestamp"].strftime("%Y-%m-%dT%H:%M:%SZ"),
            "norm_score":        round(float(row["norm_score"]), 4),
            "dt_sec":            round(dt_sec, 2),
            "disk_center_px":    [round(ref_cx, 2), round(ref_cy, 2)],
            "disk_radius_px":    round(ref_semi_a, 2),
            "own_disk_radius_px": round(_own_semi_a, 2),
            "radius_shared":     "+radius_shared" in _geometry_source,
            "delta_lambda_deg":  round((dt_sec / (period_hours * 3600.0)) * 360.0, 4),
            "pre_warp_shift_dx": round(_pw_entry[0], 3) if _pw_entry else None,
            "pre_warp_shift_dy": round(_pw_entry[1], 3) if _pw_entry else None,
            "pre_warp_scale":    round(_pw_entry[2], 6) if _pw_entry else None,
            "raw_sharpness":     round(_row_sharpness, 8) if _row_sharpness is not None else None,
        })

        if row["stem"] == reference_row["stem"]:
            ref_img = warped

        if _do_coverage:
            if row["stem"] == reference_row["stem"]:
                # dt=0 by construction: always full coverage, unconditionally.
                coverage_masks.append(np.ones(warped.shape[:2], dtype=bool))
            else:
                coverage_masks.append(compute_frame_coverage_mask(
                    warped.shape[0], warped.shape[1], dt_sec, ref_cx, ref_cy, ref_semi_a,
                    period_hours, sub_observer_lat_deg, pole_pa_deg,
                    true_polar_equatorial_ratio, scale=warp_scale,
                    flip_direction=flip_direction, flip_pole_axis=flip_pole_axis,
                ))

        warped_images.append(warped)
        weights.append(float(row["norm_score"]))

    # ── Per-frame brightness normalization ───────────────────────────────────
    if normalize_brightness and len(warped_images) > 1:
        ref_idx = next(
            i for i, fl in enumerate(log_frames)
            if fl["stem"] == reference_row["stem"]
        )
        warped_images = normalize_brightness_to_reference(warped_images, ref_idx)
        if _do_ring_only_stack and len(ring_only_images) > 1:
            ring_only_images = normalize_brightness_to_reference(ring_only_images, ref_idx)

    # ── Sub-pixel translation alignment (pre-warp center based) ─────────────
    # Disk centres measured from raw (pre-warp) frames are now applied BEFORE
    # warping (see the fix above, in the per-frame warp loop) — so for any
    # stem in _pre_warp_shifts, warped_images already reflects that
    # correction and this loop is a no-op passthrough for it (kept only to
    # populate the log with the same fields as before, for continuity with
    # the outlier-rejection step right after this and any external log
    # consumers). Post-warp limb_center_align is still biased the way the
    # comment below always said: the warp redistributes atmospheric
    # brightness (belts/zones move), causing find_disk_center to shift in the
    # same direction as the warp, so correcting that shift AFTER warping
    # partially undoes the de-rotation — this is why the fallback chain below
    # is still only used when pre-warp detection failed outright, never as a
    # general-purpose post-warp correction.
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
                    # Already applied to the raw frame before warping —
                    # nothing left to do here except log what was applied.
                    dx, dy, _, _, _ = _pre_warp_shifts[stem]
                    aligned_images.append(img)
                    frame_log["align_shift_px"] = [round(dx, 3), round(dy, 3)]
                    frame_log["align_method"] = "pre_warp_center"
                    continue
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
                kept_coverage: List[np.ndarray] = []
                kept_ring_only: List[np.ndarray] = []
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
                    if _do_coverage:
                        kept_coverage.append(coverage_masks[i])
                    if _do_ring_only_stack:
                        kept_ring_only.append(ring_only_images[i])
                warped_images = kept_images
                weights = kept_weights
                if _do_coverage:
                    coverage_masks = kept_coverage
                if _do_ring_only_stack:
                    ring_only_images = kept_ring_only

    # ── Raw-sharpness-based frame selection (opt-in, 2026-08-13) ──────────
    # See frame_sharpness_central()'s docstring for the real-data
    # justification. log_frames itself is never filtered/reordered above —
    # only warped_images/weights shrink (outlier-shift rejection). Rebuild
    # the correspondence by skipping any log entry already marked
    # outlier_excluded, in original order: both warped_images and this
    # filtered log view were built by skipping exactly those same entries,
    # in the same order, so surviving_logs[i] <-> warped_images[i] holds
    # here regardless of whether outlier rejection actually ran.
    if sharpness_selection_enabled and 0.0 < sharpness_keep_fraction < 1.0:
        surviving_logs = [fl for fl in log_frames if not fl.get("outlier_excluded")]
        candidates = [
            i for i, fl in enumerate(surviving_logs)
            if fl.get("raw_sharpness") is not None and fl["stem"] != reference_row["stem"]
        ]
        if len(candidates) >= 2:
            vals = np.array([surviving_logs[i]["raw_sharpness"] for i in candidates])
            threshold = float(np.quantile(vals, 1.0 - sharpness_keep_fraction))
            exclude_idx = {i for i in candidates if surviving_logs[i]["raw_sharpness"] < threshold}
            # Never empty the stack, and always keep at least the
            # reference frame plus one other — mirrors the outlier-
            # rejection block's own "0 < n < len(...)" guard above.
            if 0 < len(exclude_idx) < len(warped_images) - 1:
                kept = [i for i in range(len(warped_images)) if i not in exclude_idx]
                for i in exclude_idx:
                    fl = surviving_logs[i]
                    fl["sharpness_excluded"] = True
                    print(
                        f"    [sharpness] {fl['stem']}: raw_sharpness={fl['raw_sharpness']:.2e} "
                        f"below window threshold {threshold:.2e} "
                        f"(keep_fraction={sharpness_keep_fraction}) — excluded from stack"
                    )
                for i in candidates:
                    if i not in exclude_idx:
                        surviving_logs[i]["sharpness_excluded"] = False
                warped_images = [warped_images[i] for i in kept]
                weights = [weights[i] for i in kept]
                if _do_coverage:
                    coverage_masks = [coverage_masks[i] for i in kept]
                if _do_ring_only_stack:
                    ring_only_images = [ring_only_images[i] for i in kept]

    n_map: Optional[np.ndarray] = None
    if _do_coverage and coverage_masks:
        # Duplicates quality_weighted_stack's own weight-normalisation
        # formula rather than refactoring that stable, tested function --
        # keep in sync if that normalisation ever changes.
        _cov_w = np.clip(np.array(weights, dtype=np.float64), 1e-9, None)
        if weight_power != 1.0:
            _cov_w = _cov_w ** weight_power
        _cov_w /= _cov_w.sum()
        n_map = np.zeros(coverage_masks[0].shape, dtype=np.float64)
        for _cov, _wi in zip(coverage_masks, _cov_w):
            n_map += _wi * _cov.astype(np.float64)
        n_map = cv2.GaussianBlur(
            n_map.astype(np.float32), (0, 0), sigmaX=_COVERAGE_SMOOTH_SIGMA_PX
        )
        n_map = np.clip(n_map, 0.0, 1.0).astype(np.float32)

    stacked = quality_weighted_stack(warped_images, weights, weight_power=weight_power)

    ring_only_stacked: Optional[np.ndarray] = None
    if _do_ring_only_stack and ring_only_images:
        ring_only_stacked = quality_weighted_stack(ring_only_images, weights, weight_power=weight_power)

    _s0_sl_blend_applied = False
    if s0_sl_blend_enabled and use_true_reprojection and n_map is not None and ref_img is not None:
        alpha = coverage_to_confidence(n_map, floor=0.0)
        alpha_b = alpha[:, :, np.newaxis] if (stacked.ndim == 3 and alpha.ndim == 2) else alpha
        stacked = np.clip(
            alpha_b * stacked.astype(np.float64) + (1.0 - alpha_b) * ref_img.astype(np.float64),
            0.0, 1.0,
        ).astype(np.float32)
        _s0_sl_blend_applied = True

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
        "has_rings":             has_rings,
        "ring_crosses_disk":     ring_crosses_disk,
        "sub_observer_lat_deg":  sub_observer_lat_deg,
        "sharpness_selection_enabled": sharpness_selection_enabled,
        "sharpness_keep_fraction":     sharpness_keep_fraction,
        "coverage_computed":     _do_coverage,
        "s0_sl_blend_enabled":   s0_sl_blend_enabled,
        "s0_sl_blend_applied":   _s0_sl_blend_applied,
        "coverage_mean":         round(float(n_map.mean()), 4) if n_map is not None else None,
        "coverage_min":          round(float(n_map.min()), 4) if n_map is not None else None,
        # RAW ndarray -- MUST be popped by the caller (derotate_window())
        # before this dict reaches any JSON serialization path (see
        # derotate_window's coverage-map handling). Kept in this same
        # (stacked, log_dict) 2-tuple, not a 3rd return value, so existing
        # direct callers of derotate_filter() (tests, scratch scripts) keep
        # working unchanged.
        "coverage_map":          n_map,
        # RAW ndarray, same pop-before-JSON contract as coverage_map above
        # (see derotate_window's ring-mask handling). Added 2026-08-16
        # (project_ring_globe_layer_separation_roadmap Phase 0): this array
        # was already computed above whenever has_rings=True (needed by the
        # warp itself) but previously went out of scope unused once this
        # function returned -- a future ring-only stacking/compositing stage
        # needs it without recomputation, so it's persisted the same way
        # coverage_map already is. None when has_rings=False (no extra cost
        # in that case -- this is not a new computation, just retaining one
        # that already happens).
        "ring_crossing_mask":    ring_crossing_mask,
        # RAW ndarray, same pop-before-JSON contract. Added 2026-08-16
        # (project_ring_globe_layer_separation_roadmap Phase 1, opt-in via
        # compute_ring_only_stack): a second per-filter stack registered by
        # a fresh ring-annulus-windowed subpixel_align() measurement per
        # frame instead of the globe-based pre-warp shift -- see
        # DerotationConfig.compute_ring_only_stack's docstring. None unless
        # both has_rings and compute_ring_only_stack are True.
        "ring_only_stack":       ring_only_stacked,
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
    has_rings: bool = False,
    sharpness_selection_enabled: bool = False,
    sharpness_keep_fraction: float = 1.0,
    compute_coverage_map: bool = False,
    s0_sl_blend_enabled: bool = False,
    compute_ring_only_stack: bool = False,
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
        true_polar_equatorial_ratio, flip_pole_axis, has_rings,
        sharpness_selection_enabled, sharpness_keep_fraction:
                          Passed straight through to derotate_filter() — see
                          its docstring. Default use_true_reprojection=False,
                          has_rings=False leaves every existing caller's
                          behaviour unchanged (Jupiter and any caller not
                          passing has_rings=True). sharpness_selection_enabled
                          default False / sharpness_keep_fraction default 1.0
                          is likewise a complete no-op for every existing
                          caller. compute_coverage_map/s0_sl_blend_enabled:
                          passed straight through to derotate_filter() — see
                          its docstring; default False/False is a complete
                          no-op. When compute_coverage_map is True and
                          out_dir is provided, the per-pixel coverage map is
                          additionally saved as ``<filt>_coverage.tif`` next
                          to the derotated TIF, with its path recorded in
                          log_dict["coverage_map_file"] (the raw array itself
                          is never left in log_dict — see the popping logic
                          below — since that dict is later spread wholesale
                          into a JSON file elsewhere in the pipeline).
                          When has_rings is True and out_dir is provided, the
                          ring_crossing_mask that derotate_filter() already
                          computes for the warp is likewise saved as
                          ``<filt>_ring_mask.tif``, path recorded in
                          log_dict["ring_mask_file"] (2026-08-16,
                          project_ring_globe_layer_separation_roadmap Phase 0
                          — no new computation, just persisting a value that
                          already existed transiently).
        compute_ring_only_stack: Passed straight through to derotate_filter()
                          — see its docstring. When True (and has_rings=True)
                          and out_dir is provided, the second ring-registered
                          stack is saved as ``<filt>_ring_only.tif``, path
                          recorded in log_dict["ring_only_stack_file"]
                          (2026-08-16, project_ring_globe_layer_separation_
                          roadmap Phase 1). Default False: complete no-op.

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
                has_rings=has_rings,
                sharpness_selection_enabled=sharpness_selection_enabled,
                sharpness_keep_fraction=sharpness_keep_fraction,
                compute_coverage_map=compute_coverage_map,
                s0_sl_blend_enabled=s0_sl_blend_enabled,
                compute_ring_only_stack=compute_ring_only_stack,
            )
        except Exception as exc:
            print(f" ERROR: {exc}")
            results[filt] = (None, {"error": str(exc)})
            continue

        # The raw coverage-map ndarray (if any) must never survive into
        # log_dict past this point: derotation_log_to_json() later spreads
        # log_dict wholesale (**log) into a dict that gets json.dump'd, and
        # an ndarray there would break/bloat that. Save it as a companion
        # TIF (mirrors output_file's own path-not-pixels convention) and
        # keep only the path string.
        _coverage_arr = log.pop("coverage_map", None)
        if _coverage_arr is not None and out_dir is not None:
            cov_path = out_dir / f"{filt}_coverage.tif"
            image_io.write_tif_16bit(_coverage_arr, cov_path)
            log["coverage_map_file"] = str(cov_path)
        else:
            log["coverage_map_file"] = None

        # Same pop-before-JSON contract as coverage_map above. Added
        # 2026-08-16 (project_ring_globe_layer_separation_roadmap Phase 0):
        # persists the already-computed ring_crossing_mask (has_rings=True
        # targets only, None/not-written otherwise) as a companion TIF so a
        # future ring-only stacking/compositing stage can reuse it without
        # recomputing the geometry.
        _ring_mask_arr = log.pop("ring_crossing_mask", None)
        if _ring_mask_arr is not None and out_dir is not None:
            ring_mask_path = out_dir / f"{filt}_ring_mask.tif"
            image_io.write_tif_16bit(_ring_mask_arr, ring_mask_path)
            log["ring_mask_file"] = str(ring_mask_path)
        else:
            log["ring_mask_file"] = None

        # Same pop-before-JSON contract, added 2026-08-16 (Phase 1 of
        # project_ring_globe_layer_separation_roadmap, opt-in via
        # compute_ring_only_stack).
        _ring_only_arr = log.pop("ring_only_stack", None)
        if _ring_only_arr is not None and out_dir is not None:
            ring_only_path = out_dir / f"{filt}_ring_only.tif"
            if color_mode:
                image_io.write_tif_color_16bit(_ring_only_arr, ring_only_path)
            else:
                image_io.write_tif_16bit(_ring_only_arr, ring_only_path)
            log["ring_only_stack_file"] = str(ring_only_path)
        else:
            log["ring_only_stack_file"] = None

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
