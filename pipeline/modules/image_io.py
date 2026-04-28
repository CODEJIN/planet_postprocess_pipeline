"""
Image I/O and filename utilities for the planetary imaging pipeline.

Supported filename conventions (tried in order by parse_filename):

1. FireCapture / WinJUPOS (default):
       YYYY-MM-DD-HHMM_D-CAM-FILTER-TARGET_suffix.tif
       e.g. 2026-03-20-1046_1-U-IR-Jup_pipp_lapl3_ap51.tif

2. ASIAIR (planetary video mode):
       TargetName_YYYYMMDD-HHMMSS[_suffix].ser/.tif
       e.g. Jupiter_20231025-213045_lapl6_ap47.tif

3. SharpCap (time-only filename, date encoded in parent folder):
       HH_MM_SS[_suffix].ser/.tif   under  .../YYYY-MM-DD/TargetName/
       e.g. 21_50_23_lapl6_ap47_conv.tif

For patterns 2 and 3, filter is unknown (color camera assumed) and is
returned as None; group_by_filter() maps None → "color".
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Optional: tifffile gives better 16/32-bit TIF support
try:
    import tifffile
    _HAS_TIFFFILE = True
except ImportError:
    _HAS_TIFFFILE = False


# ── Filename parsing ───────────────────────────────────────────────────────────

# Pattern 1 – FireCapture / WinJUPOS: 2026-03-20-1046_1-U-IR-Jup_pipp…
_FNAME_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})"          # group 1: date
    r"-(\d{4}_\d)"                    # group 2: HHMM_D
    r"-[^-]+"                         # camera id (ignored)
    r"-([A-Za-z0-9]+)"                # group 3: filter
    r"-([A-Za-z]+)"                   # group 4: target
    r"_"
)

# Pattern 2 – ASIAIR: Jupiter_20231025-213045[_anything]
_ASIAIR_RE = re.compile(
    r"^([A-Za-z][A-Za-z0-9]*)"       # group 1: target name (word)
    r"_(\d{4})(\d{2})(\d{2})"        # groups 2-4: YYYY MM DD
    r"-(\d{2})(\d{2})(\d{2})"        # groups 5-7: HH MM SS
)

# Pattern 3 – SharpCap: HH_MM_SS[_anything]
_SHARPCAP_TIME_RE = re.compile(
    r"^(\d{2})_(\d{2})_(\d{2})(?:[_.]|$)"  # groups 1-3: HH MM SS
)

# Folder-name date pattern used to extract date from parent directories
_DATE_FOLDER_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})$")

# Planet name → WinJUPOS abbreviation
TARGET_ABBREV: Dict[str, str] = {
    "jupiter": "Jup", "saturn": "Sat", "mars":    "Mar",
    "uranus":  "Ura", "neptune": "Nep", "venus":  "Ven",
    "moon":    "Moo", "sun":     "Sun",
}


def _extract_date_from_parents(path: Path) -> Optional[str]:
    """Search the two nearest parent directories for a YYYY-MM-DD name."""
    for ancestor in (path.parent, path.parent.parent):
        if _DATE_FOLDER_RE.match(ancestor.name):
            return ancestor.name
    return None


def _mtime_date(path: Path) -> str:
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        mtime = datetime.now(tz=timezone.utc)
    return mtime.strftime("%Y-%m-%d")


def _target_from_folder(path: Path) -> Optional[str]:
    """Try to find a planet name in the two nearest parent folder names."""
    for ancestor in (path.parent, path.parent.parent):
        abbrev = TARGET_ABBREV.get(ancestor.name.lower())
        if abbrev:
            return abbrev
    return None


def _build_meta(date_str: str, hh: int, mm: int, ss: int,
                filter_name: Optional[str], target: Optional[str],
                stem: str) -> Dict:
    d = min(9, round(ss / 6))
    date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    timestamp = date + timedelta(seconds=hh * 3600 + mm * 60 + d * 6)
    return {
        "date":      date_str,
        "timestamp": timestamp,
        "filter":    filter_name,
        "target":    target,
        "stem":      stem,
    }


def parse_filename(path: Path) -> Optional[Dict]:
    """Extract date, timestamp, filter, and target from a stacked TIF/SER path.

    Tries FireCapture, ASIAIR, and SharpCap patterns in that order.
    Returns None if no pattern matches.

    For ASIAIR and SharpCap files, filter is None (color camera assumed);
    target may also be None when the folder structure doesn't encode it.
    """
    name = path.name
    stem = path.stem

    # ── Pattern 1: FireCapture / WinJUPOS ─────────────────────────────────────
    m = _FNAME_RE.match(name)
    if m:
        date_str, time_str, filter_name, target = m.groups()
        hh = int(time_str[0:2])
        mm = int(time_str[2:4])
        d  = int(time_str[5])
        date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        timestamp = date + timedelta(seconds=hh * 3600 + mm * 60 + d * 6)
        return {"date": date_str, "timestamp": timestamp,
                "filter": filter_name, "target": target, "stem": stem}

    # ── Pattern 2: ASIAIR ─────────────────────────────────────────────────────
    m = _ASIAIR_RE.match(stem)
    if m:
        target_raw = m.group(1)
        yyyy, mon, dd = m.group(2), m.group(3), m.group(4)
        hh, mm_val, ss = int(m.group(5)), int(m.group(6)), int(m.group(7))
        if 0 <= hh <= 23 and 0 <= mm_val <= 59 and 0 <= ss <= 59:
            date_str = f"{yyyy}-{mon}-{dd}"
            target = TARGET_ABBREV.get(target_raw.lower(),
                                       target_raw[:3].capitalize())
            return _build_meta(date_str, hh, mm_val, ss, None, target, stem)

    # ── Pattern 3: SharpCap ───────────────────────────────────────────────────
    m = _SHARPCAP_TIME_RE.match(stem)
    if m:
        hh, mm_val, ss = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 0 <= hh <= 23 and 0 <= mm_val <= 59 and 0 <= ss <= 59:
            date_str = _extract_date_from_parents(path) or _mtime_date(path)
            target   = _target_from_folder(path)
            return _build_meta(date_str, hh, mm_val, ss, None, target, stem)

    return None


def infer_winjupos_stem(path: Path, filter_name: str, target: str,
                        camera: str = "U") -> str:
    """Return a WinJUPOS-compatible file stem for *path*.

    If *path* already follows the FireCapture/WinJUPOS convention the original
    stem is returned unchanged.  Otherwise a new stem is constructed from the
    ASIAIR or SharpCap pattern using *filter_name* and *target* from config.
    Falls back to the original stem if no pattern is recognised.
    """
    stem = path.stem

    if _FNAME_RE.match(path.name):
        return stem

    m = _ASIAIR_RE.match(stem)
    if m:
        yyyy, mon, dd = m.group(2), m.group(3), m.group(4)
        hh, mm_val, ss = int(m.group(5)), int(m.group(6)), int(m.group(7))
        if 0 <= hh <= 23 and 0 <= mm_val <= 59 and 0 <= ss <= 59:
            d = min(9, round(ss / 6))
            return f"{yyyy}-{mon}-{dd}-{hh:02d}{mm_val:02d}_{d}-{camera}-{filter_name}-{target}"

    m = _SHARPCAP_TIME_RE.match(stem)
    if m:
        hh, mm_val, ss = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 0 <= hh <= 23 and 0 <= mm_val <= 59 and 0 <= ss <= 59:
            d = min(9, round(ss / 6))
            date_str = _extract_date_from_parents(path) or _mtime_date(path)
            return f"{date_str}-{hh:02d}{mm_val:02d}_{d}-{camera}-{filter_name}-{target}"

    return stem


def group_by_filter(
    tif_dir: Path,
    target: str = "Jup",
) -> Dict[str, List[Tuple[Path, Dict]]]:
    """Scan *tif_dir* for TIF files and group them by filter name.

    Files whose target cannot be determined (SharpCap/ASIAIR without folder
    context) are included regardless of the *target* argument.  Files whose
    filter is None (color camera) are grouped under the key "color".

    Returns:
        {filter_name: [(path, meta), ...]}  sorted by timestamp within each group.
    """
    groups: Dict[str, List[Tuple[Path, Dict]]] = {}

    for p in sorted(tif_dir.glob("*.tif")):
        meta = parse_filename(p)
        if meta is None:
            continue
        # Skip only when target is positively known and doesn't match
        if meta["target"] is not None and meta["target"] != target:
            continue
        f = meta["filter"] if meta["filter"] is not None else "color"
        meta = {**meta, "filter": f}
        groups.setdefault(f, []).append((p, meta))

    for f in groups:
        groups[f].sort(key=lambda x: x[1]["timestamp"])

    return groups


# ── Image reading ──────────────────────────────────────────────────────────────

def read_tif(path: Path) -> np.ndarray:
    """Read a TIF file and return a float32 array normalised to [0, 1].

    Supports 8-bit and 16-bit grayscale TIFs.
    Uses tifffile when available (better multi-page / float TIF support),
    falls back to OpenCV otherwise.
    """
    if _HAS_TIFFFILE:
        img = tifffile.imread(str(path))
    else:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Cannot read: {path}")

    img = img.astype(np.float32)

    # Normalise to [0, 1] based on bit depth
    if img.max() > 1.0:
        max_val = 65535.0 if img.max() > 255 else 255.0
        img /= max_val

    return img


# ── Image writing ──────────────────────────────────────────────────────────────

def write_png_16bit(image: np.ndarray, path: Path) -> None:
    """Write a float [0, 1] image as a 16-bit grayscale PNG."""
    arr = np.clip(image * 65535.0, 0, 65535).astype(np.uint16)
    cv2.imwrite(str(path), arr)


def write_png_autostretch(
    image: np.ndarray,
    path: Path,
    plow: float = 0.5,
    phigh: float = 99.5,
) -> None:
    """Write a float [0, 1] image as a 16-bit PNG with percentile auto-stretch.

    Stretching makes the preview visible for quality inspection even when the
    raw pixel values occupy only a small portion of the dynamic range.
    """
    lo, hi = np.percentile(image, [plow, phigh])
    span = hi - lo
    if span < 1e-9:
        stretched = np.zeros_like(image)
    else:
        stretched = np.clip((image - lo) / span, 0.0, 1.0)
    write_png_16bit(stretched, path)


def write_tif_16bit(image: np.ndarray, path: Path) -> None:
    """Write a float [0, 1] image as a 16-bit grayscale TIF."""
    arr = np.clip(image * 65535.0, 0, 65535).astype(np.uint16)
    if _HAS_TIFFFILE:
        tifffile.imwrite(str(path), arr)
    else:
        cv2.imwrite(str(path), arr)


def read_png(path: Path) -> np.ndarray:
    """Read a PNG (8-bit or 16-bit, grayscale or colour) and return float32 [0, 1].

    Colour images are returned as (H, W, 3) in RGB order.
    Grayscale images are returned as (H, W).
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    img = img.astype(np.float32)
    max_val = 65535.0 if img.max() > 255 else 255.0
    img /= max_val
    # OpenCV reads colour as BGR → convert to RGB
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def write_png_color_16bit(image: np.ndarray, path: Path) -> None:
    """Write a float [0, 1] (H, W, 3) RGB image as a 16-bit colour PNG."""
    arr = np.clip(image * 65535.0, 0, 65535).astype(np.uint16)
    # OpenCV expects BGR
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def write_tif_color_16bit(image: np.ndarray, path: Path) -> None:
    """Write a float [0, 1] (H, W, 3) RGB image as a 16-bit colour TIF."""
    arr = np.clip(image * 65535.0, 0, 65535).astype(np.uint16)
    if _HAS_TIFFFILE:
        # tifffile preserves channel order (RGB) natively
        tifffile.imwrite(str(path), arr)
    else:
        # OpenCV expects BGR
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), bgr)
