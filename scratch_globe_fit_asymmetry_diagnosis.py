"""Diagnose the exact SHAPE of find_disk_center()'s asymmetric fit error on
real Saturn data, to distinguish two hypotheses before designing a fix:

  (A) a roughly UNIFORM CENTER OFFSET (cx,cy shifted by the ring's asymmetric
      pull on the core-isolation contour) -- predicts true-edge-minus-fit
      residuals varying as a single cos(theta - phi0) around the ellipse,
      with the SAME magnitude on opposite sides but OPPOSITE sign (outside
      on one ansa, inside on the other, by the same amount)
  (B) a genuinely non-uniform SHAPE error (e.g. one side's contour locally
      dragged out further than a pure translation would predict) -- would
      NOT fit a single cosine, needs per-direction correction

This matters: (A) is fixable by a much simpler, more targeted center
correction; (B) would need the harder, previously-attempted-and-failed
per-direction edge-model redesign (see project_saturn_ring_globe_separation
memory -- but note that failure was about detect_ring_geometry(), a
DIFFERENT function that tried to find the RING's own edge, not this).

Uses the real production find_disk_center() on the real derotate_window()
output (same object wavelet_master.py itself measures), per
feedback_ab_test_via_real_pipeline.
"""
from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from pipeline.modules import image_io
from pipeline.modules.derotation import derotate_window, find_disk_center


def _parse_ts(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")


def _hydrate(window: dict, step02_dir: Path) -> dict:
    hydrated = {"center_time": _parse_ts(window["center_time"]), "per_filter": {}}
    for filt, pf in window["per_filter"].items():
        included = []
        for item in pf["included"]:
            m = list(step02_dir.glob(f"{item['stem']}.tif"))
            if not m:
                continue
            included.append({
                "path": str(m[0]), "stem": item["stem"],
                "timestamp": _parse_ts(item["timestamp"]), "norm_score": item["norm_score"],
            })
        hydrated["per_filter"][filt] = {"included": included}
    return hydrated


def _bilinear(img, ys, xs):
    map_x = xs.astype(np.float32).reshape(1, -1)
    map_y = ys.astype(np.float32).reshape(1, -1)
    return cv2.remap(img.astype(np.float32), map_x, map_y,
                      interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).ravel()


def measure_true_edge(lum, cx, cy, rx, ry, angle_deg, theta_deg,
                       search_frac=(0.7, 1.3), n_samples=300, smooth_sigma=1.5):
    """Steepest-gradient limb radius along one ray at angle theta_deg
    (measured in the image's own x/y frame, NOT relative to the ellipse's
    own tilt) -- independent re-measurement, not reusing _gradient_disk_r,
    so this diagnosis isn't circular. Returns (r_ell, r_true) or
    (r_ell, None) if the steepest gradient lands at the search window's own
    edge (not a genuine interior minimum -- must be discarded, not reported
    as if it were real, exactly per _gradient_disk_r's own convention)."""
    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    theta = np.radians(theta_deg)
    # Ellipse-boundary radius in this direction (image frame)
    dx, dy = np.cos(theta), np.sin(theta)
    dxr = cos_a * dx + sin_a * dy
    dyr = -sin_a * dx + cos_a * dy
    r_ell = 1.0 / np.sqrt((dxr / rx) ** 2 + (dyr / ry) ** 2)

    r_vals = np.linspace(r_ell * search_frac[0], r_ell * search_frac[1], n_samples)
    dr = r_vals[1] - r_vals[0]
    xs = cx + r_vals * dx
    ys = cy + r_vals * dy
    profile = _bilinear(lum, ys, xs)
    k = max(1, int(3 * smooth_sigma))
    kernel = np.exp(-0.5 * (np.arange(-k, k + 1) / smooth_sigma) ** 2)
    kernel /= kernel.sum()
    smoothed = np.convolve(profile, kernel, mode="same")
    grad = np.gradient(smoothed, dr)
    idx = int(np.argmin(grad))
    if idx <= 0 or idx >= len(grad) - 1:
        return r_ell, None
    y0, y1, y2 = grad[idx - 1], grad[idx], grad[idx + 1]
    denom = 2.0 * (y2 - 2.0 * y1 + y0)
    sub = -(y2 - y0) / denom if abs(denom) > 1e-9 else 0.0
    return r_ell, r_vals[idx] + sub * dr


def run_case(target_dir, wi, filt, true_polar_ratio):
    print(f"\n=== {target_dir} window_{wi:02d}/{filt} ===")
    step02 = Path(f"{target_dir}/step02_lucky_stack")
    step04 = Path(f"{target_dir}/step04_derotated")
    windows_json = Path(f"{target_dir}/step03_quality/windows.json")

    data = json.load(open(windows_json))
    windows = {w["window_index"]: w for w in data["selected_windows"]}
    window = windows[wi]
    window_log = json.load(open(step04 / f"window_{wi:02d}/derotation_log.json"))
    hydrated = _hydrate(window, step02)
    flog = window_log["filters"][filt]

    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        results = derotate_window(
            hydrated, required_filters=[filt],
            period_hours=flog["period_hours"], warp_scale=flog["warp_scale"],
            align=flog["align_enabled"], normalize_brightness=flog["normalize_brightness"],
            min_quality_threshold=0.0, pole_pa_deg=flog["pole_pa_deg"],
            color_mode=False, flip_direction=flog["flip_direction"],
            weight_power=flog["weight_power"], has_rings=flog.get("has_rings", False),
            sub_observer_lat_deg=flog.get("sub_observer_lat_deg", 0.0),
            use_true_reprojection=True, true_polar_equatorial_ratio=true_polar_ratio,
            out_dir=tmp,
        )
        out_path, log = results[filt]
        assert out_path is not None, log
        img = image_io.read_tif(out_path)

    lum = img.mean(axis=2) if img.ndim == 3 else img
    cx, cy, rx, ry, angle = find_disk_center(lum)
    print(f"  fit: cx={cx:.2f} cy={cy:.2f} rx={rx:.2f} ry={ry:.2f} angle={angle:.2f}")

    thetas_all = np.arange(0, 360, 10)
    thetas = []
    residuals = []
    for theta_deg in thetas_all:
        r_fit, r_true = measure_true_edge(lum, cx, cy, rx, ry, angle, theta_deg)
        if r_true is None:
            continue
        thetas.append(theta_deg)
        residuals.append(r_true - r_fit)
    thetas = np.array(thetas)
    residuals = np.array(residuals)
    n_dropped = len(thetas_all) - len(thetas)
    if n_dropped:
        print(f"  ({n_dropped}/{len(thetas_all)} rays dropped: no interior gradient minimum found)")

    # Fit a single cosine: residual(theta) ~ A*cos(theta - phi0) + C
    theta_rad = np.radians(thetas)
    X = np.stack([np.cos(theta_rad), np.sin(theta_rad), np.ones_like(theta_rad)], axis=1)
    coef, sse_res, *_ = np.linalg.lstsq(X, residuals, rcond=None)
    fitted = X @ coef
    A = np.hypot(coef[0], coef[1])
    phi0 = np.degrees(np.arctan2(coef[1], coef[0]))
    ss_tot = np.sum((residuals - residuals.mean()) ** 2)
    ss_res = np.sum((residuals - fitted) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else float("nan")

    print(f"  residual(theta) [true-fit, px] by angle (deg): ")
    for t, r in zip(thetas, residuals):
        print(f"    {t:5.0f}: {r:+.3f}")
    print(f"  single-cosine fit: amplitude={A:.3f}px phase={phi0:.1f}deg offset={coef[2]:+.3f}px  R^2={r2:.3f}")
    print(f"  (R^2 close to 1.0 => dominated by a uniform center offset; "
          f"low R^2 => genuine non-uniform shape error)")


if __name__ == "__main__":
    run_case("Saturn_Data", 1, "R", 0.9021)
    run_case("Saturn_Data", 1, "IR", 0.9021)
