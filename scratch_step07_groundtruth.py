"""Ground-truth check: does the REAL production step07 (single-best-frame,
wavelet-sharpened preview) output for window_01 (or any window) actually show
a resolvable Cassini Division in the ansa -- verified directly against real
files on disk, not reconstructed.

Per task: step07_wavelet_preview/<FILTER>/<stem>_wavelet.png files already
exist on disk (confirmed: 10 real IR + 10 real R files, covering all 9
selected windows' best-norm_score frames). Load these directly.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from pipeline.modules import image_io
from pipeline.modules.derotation import find_disk_center

STEP07_DIR = Path("Saturn_Data/step07_wavelet_preview")
STEP04_DIR = Path("Saturn_Data/step04_derotated")
WINDOWS_JSON = Path("Saturn_Data/step03_quality/windows.json")
CROP_DIR = Path("scratch_investigation_crops")
CROP_DIR.mkdir(exist_ok=True)


def radial_profile(img: np.ndarray, cx, cy, semi_a, pole_pa_deg,
                    r0=1.0, r1=2.4, step_px=0.5, half_width=1.5):
    """Sample mean brightness in thin bins perpendicular to pole_pa_deg
    (i.e. along the ring-plane/ansa direction), both +x and -x sides,
    from r0*semi_a to r1*semi_a, sampled every step_px pixels. Same
    rotation convention as scratch_cassini_scale_ab.py's ansa mask
    (ang = pole_pa_deg directly gives the ansa/equatorial axis in this
    codebase's convention)."""
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    ang = np.radians(pole_pa_deg)
    cos_a, sin_a = np.cos(ang), np.sin(ang)
    dx, dy = xx - cx, yy - cy
    xr = dx * cos_a + dy * sin_a
    yr = -dx * sin_a + dy * cos_a

    r_min_px, r_max_px = r0 * semi_a, r1 * semi_a
    radii_px = np.arange(r_min_px, r_max_px + 1e-9, step_px)
    prof_pos, prof_neg = [], []
    for r in radii_px:
        band_pos = (np.abs(xr - r) < half_width) & (np.abs(yr) < half_width)
        band_neg = (np.abs(xr + r) < half_width) & (np.abs(yr) < half_width)
        prof_pos.append(float(img[band_pos].mean()) if band_pos.sum() > 0 else np.nan)
        prof_neg.append(float(img[band_neg].mean()) if band_neg.sum() > 0 else np.nan)
    return radii_px / semi_a, np.array(prof_pos), np.array(prof_neg)


def _smooth(x, win=5):
    if len(x) < win:
        return x.copy()
    kernel = np.ones(win) / win
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xp, kernel, mode="valid")[: len(x)]


def find_local_extrema(rr, prof, smooth_win=5, min_prominence_frac=0.03):
    """Smooth first (pixel-level speckle at 0.5px sampling otherwise
    produces dozens of spurious sub-1%-amplitude wiggles), then use
    scipy.signal.find_peaks with a prominence floor relative to the
    profile's own dynamic range so only real structure survives."""
    from scipy.signal import find_peaks

    valid = ~np.isnan(prof)
    if valid.sum() < 5:
        return []
    sm = _smooth(prof, smooth_win)
    rng = np.nanmax(sm) - np.nanmin(sm)
    if rng <= 0:
        return []
    min_prom = rng * min_prominence_frac

    out = []
    idx_max, prop_max = find_peaks(sm, prominence=min_prom)
    for i, p in zip(idx_max, prop_max["prominences"]):
        out.append((rr[i], prof[i], "max", p))
    idx_min, prop_min = find_peaks(-sm, prominence=min_prom)
    for i, p in zip(idx_min, prop_min["prominences"]):
        out.append((rr[i], prof[i], "min", p))
    out.sort(key=lambda t: t[0])
    return out


def save_ansa_crop(img: np.ndarray, cx, cy, semi_a, pole_pa_deg, out_path, upscale=3):
    """Rotate so ansa axis is horizontal, crop a band around the ring
    plane out to 2.4*semi_a on both sides, upscale 3x nearest, normalize
    to 99.5th percentile."""
    h, w = img.shape
    M = cv2.getRotationMatrix2D((cx, cy), pole_pa_deg, 1.0)
    rot = cv2.warpAffine(img.astype(np.float32), M, (w, h), flags=cv2.INTER_LINEAR)
    half = int(semi_a * 2.5)
    band = int(semi_a * 0.5)
    x0, x1 = int(cx - half), int(cx + half)
    y0, y1 = int(cy - band), int(cy + band)
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, w), min(y1, h)
    crop = rot[y0:y1, x0:x1]
    p995 = np.percentile(crop, 99.5)
    crop_n = np.clip(crop / max(p995, 1e-6), 0, 1)
    crop_u8 = (crop_n * 255).astype(np.uint8)
    crop_big = cv2.resize(crop_u8, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(out_path), crop_big)
    return crop_big.shape


def analyze_file(png_path: Path, pole_pa_deg: float, label: str, save_crop=False):
    img = image_io.read_png(png_path)
    lum = img if img.ndim == 2 else img.mean(axis=2)
    cx, cy, semi_a, semi_b, angle_deg = find_disk_center(lum)
    if semi_a < 5:
        print(f"{label}: disk detect FAILED")
        return None
    rr, prof_pos, prof_neg = radial_profile(lum, cx, cy, semi_a, pole_pa_deg)
    ext_pos = find_local_extrema(rr, prof_pos)
    ext_neg = find_local_extrema(rr, prof_neg)
    print(f"\n=== {label} ({png_path.name}) ===")
    print(f"  disk fit: cx={cx:.1f} cy={cy:.1f} semi_a={semi_a:.2f} semi_b={semi_b:.2f} angle={angle_deg:.1f}")
    print(f"  +x extrema (r/semi_a, val, kind, prominence): "
          f"{[(round(r,3), round(v,4), k, round(p,4)) for r,v,k,p in ext_pos]}")
    print(f"  -x extrema (r/semi_a, val, kind, prominence): "
          f"{[(round(r,3), round(v,4), k, round(p,4)) for r,v,k,p in ext_neg]}")
    if save_crop:
        shape = save_ansa_crop(lum, cx, cy, semi_a, pole_pa_deg, CROP_DIR / f"step07_groundtruth_{label}.png")
        print(f"  crop saved: {CROP_DIR / f'step07_groundtruth_{label}.png'} shape={shape}")
    return dict(cx=cx, cy=cy, semi_a=semi_a, rr=rr.tolist(), prof_pos=prof_pos.tolist(),
                prof_neg=prof_neg.tolist(), ext_pos=ext_pos, ext_neg=ext_neg)


def main():
    windows = {w["window_index"]: w for w in json.load(open(WINDOWS_JSON))["selected_windows"]}
    results = {}

    for widx in range(1, 10):
        w = windows[widx]
        wlog_path = STEP04_DIR / f"window_{widx:02d}" / "derotation_log.json"
        if not wlog_path.exists():
            continue
        wlog = json.load(open(wlog_path))
        results[widx] = {}
        for filt in ["IR", "R"]:
            pole_pa = wlog["filters"][filt]["pole_pa_deg"]
            included = w["per_filter"].get(filt, {}).get("included", [])
            if not included:
                continue
            best = max(included, key=lambda x: x["norm_score"])
            png_path = STEP07_DIR / filt / (best["stem"] + "_wavelet.png")
            if not png_path.exists():
                print(f"window_{widx:02d} {filt}: NO real step07 file for best stem {best['stem']}")
                continue
            label = f"w{widx:02d}_{filt}"
            save_crop = (widx == 1)  # only save full crops for window_01 per task step 4
            res = analyze_file(png_path, pole_pa, label, save_crop=save_crop)
            results[widx][filt] = res

    # Summary: which window shows the most convincing local minimum in the
    # r/semi_a in [1.15, 2.3] band (plausible Cassini-ish window), i.e. any
    # interior min flanked by two maxima with decent contrast.
    print("\n\n=== SUMMARY: candidate ring-gap minima per window/filter ===")
    for widx, per_filt in results.items():
        for filt, res in per_filt.items():
            if res is None:
                continue
            for side, ext in [("+x", res["ext_pos"]), ("-x", res["ext_neg"])]:
                mins = [(r, v) for r, v, k, p in ext if k == "min" and 1.1 <= r <= 2.35]
                for r, v in mins:
                    # find flanking max values for contrast estimate
                    prof = res["prof_pos"] if side == "+x" else res["prof_neg"]
                    rr = res["rr"]
                    idx = min(range(len(rr)), key=lambda i: abs(rr[i] - r))
                    left = max(prof[:idx]) if idx > 0 else np.nan
                    right = max(prof[idx+1:]) if idx < len(prof)-1 else np.nan
                    flank = np.nanmean([left, right])
                    contrast = 1 - v / flank if flank and flank > 0 else np.nan
                    print(f"  w{widx:02d} {filt} {side}: dip at r/semi_a={r:.3f} val={v:.4f} "
                          f"flank~{flank:.4f} contrast~{contrast:.3f}")


if __name__ == "__main__":
    main()
