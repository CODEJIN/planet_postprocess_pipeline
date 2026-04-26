"""
Image quality assessment for planetary image stacks.

Computes sharpness/contrast metrics on the planet disk region only,
then identifies overlapping time windows across all filters for
optimal multi-filter compositing.

Metrics (all evaluated inside the planet disk mask):
  laplacian_var  — Laplacian variance: best single proxy for atmospheric
                   seeing quality.  High = sharp detail = good seeing.
  tenengrad      — Gradient energy (sum of squared Sobel gradients):
                   complementary sharpness measure, less sensitive to noise.
  norm_variance  — Variance normalised by mean: captures contrast
                   independent of absolute brightness.

Combined score = weighted sum of per-filter-normalised metrics.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from pipeline.modules import image_io


# ── Planet disk masking ────────────────────────────────────────────────────────

def planet_mask(image: np.ndarray, margin_factor: float = 0.15) -> np.ndarray:
    """Return a boolean mask isolating the planet disk.

    Uses Otsu thresholding on the 16-bit image (or normalised float)
    to separate planet from sky background.

    Args:
        image:         2-D float [0,1] or uint16 array.
        margin_factor: Fraction of the threshold added as a small margin
                       to include slightly dim limb regions.

    Returns:
        Boolean array, True = planet disk pixel.
    """
    if image.dtype != np.uint16:
        arr = np.clip(image * 65535, 0, 65535).astype(np.uint16)
    else:
        arr = image

    # Scale to 8-bit for Otsu (Otsu operates on 8-bit in OpenCV)
    arr8 = (arr >> 8).astype(np.uint8)
    thresh, _ = cv2.threshold(arr8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Apply threshold (with small downward margin to include dim limb)
    threshold_16 = max(0, int(thresh * 256 * (1.0 - margin_factor)))
    mask = arr > threshold_16

    # Keep only the largest connected component (the planet disk)
    mask_u8 = mask.astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8)
    if n_labels > 1:
        # Label 0 = background; pick the largest non-background label
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = labels == largest

    return mask


# ── Per-image quality metrics ──────────────────────────────────────────────────

def _to_float32(image: np.ndarray) -> np.ndarray:
    """Normalise any supported dtype to float32 [0, 1]."""
    if image.dtype == np.uint16:
        return image.astype(np.float32) / 65535.0
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    return image.astype(np.float32)


def laplacian_var(image: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Laplacian variance on the planet disk."""
    f32 = _to_float32(image)
    # CV_32F→CV_64F is unsupported in OpenCV ≥4.13; use CV_32F throughout.
    lap = cv2.Laplacian(f32, cv2.CV_32F)
    if mask is not None:
        return float(lap[mask].var())
    return float(lap.var())


def tenengrad(image: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Tenengrad sharpness: mean squared gradient magnitude on the planet disk."""
    f32 = _to_float32(image)
    gx = cv2.Sobel(f32, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f32, cv2.CV_32F, 0, 1, ksize=3)
    grad2 = gx**2 + gy**2
    if mask is not None:
        return float(grad2[mask].mean())
    return float(grad2.mean())


def norm_variance(image: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Normalised variance (contrast): var / mean on the planet disk."""
    if mask is not None:
        vals = image[mask]
    else:
        vals = image.flatten()
    m = float(vals.mean())
    return float(vals.var() / m) if m > 1e-9 else 0.0


def quality_metrics(
    image: np.ndarray,
    denoise_sigma: float = 1.2,
) -> Dict[str, float]:
    """Compute all quality metrics for a single image.

    Args:
        image:         Float [0, 1] 2-D array.
        denoise_sigma: Gaussian σ (pixels) applied before sharpness metrics.
                       A value of ~1 px suppresses random pixel noise without
                       blurring genuine planetary detail (which spans many px).
                       Set to 0 to disable (reverts to old noise-sensitive behaviour).

    Returns a dict with keys: 'laplacian', 'tenengrad', 'norm_variance'.

    Why denoise before sharpness?
        Laplacian / Tenengrad measure high-spatial-frequency energy.  Random
        sensor / shot noise also has high spatial frequency, so a noisy-but-blurry
        frame can outscore a genuinely sharp frame.  The Gaussian blur acts as a
        noise gate: coherent planetary detail (belt/zone edges) survives because it
        spans many pixels; random noise is killed.
    """
    if image.ndim == 3:
        image = image.mean(axis=2)

    mask = planet_mask(image)

    if denoise_sigma > 0:
        # Blur in float to avoid rounding artifacts from int conversion
        img_sharp = cv2.GaussianBlur(
            image.astype(np.float32), (0, 0), denoise_sigma
        )
    else:
        img_sharp = image

    return {
        "laplacian":     laplacian_var(img_sharp, mask),
        "tenengrad":     tenengrad(img_sharp, mask),
        "norm_variance": norm_variance(image, mask),   # contrast: no need to denoise
    }


# ── Batch scoring ──────────────────────────────────────────────────────────────

def compute_scores(
    groups: Dict[str, List[Tuple[Path, dict]]],
    lap_w: float = 0.5,
    ten_w: float = 0.3,
    nv_w:  float = 0.2,
    progress_callback=None,
    cancel_event=None,
) -> Dict[str, List[dict]]:
    """Score every TIF file in *groups*.

    Args:
        groups:  Output of :func:`image_io.group_by_filter`.
        lap_w:   Weight for Laplacian variance.
        ten_w:   Weight for Tenengrad.
        nv_w:    Weight for normalised variance.

    Returns:
        {filter_name: [{"stem", "timestamp", "path",
                        "laplacian", "tenengrad", "norm_variance",
                        "raw_score"}, ...]}
        Sorted by timestamp within each filter.  raw_score is the
        un-normalised weighted sum (used for within-filter ranking).
    """
    results: Dict[str, List[dict]] = {}
    total = sum(len(v) for v in groups.values())
    done = 0

    for filt in sorted(groups):
        if cancel_event is not None and cancel_event.is_set():
            print("  [CANCELLED] Stopping quality scoring.", flush=True)
            break
        entries = groups[filt]
        rows: List[dict] = []

        for path, meta in entries:
            if cancel_event is not None and cancel_event.is_set():
                break
            img = image_io.read_tif(path)
            m = quality_metrics(img)

            # Raw combined score (not yet normalised across the filter)
            raw = lap_w * m["laplacian"] + ten_w * m["tenengrad"] + nv_w * m["norm_variance"]

            rows.append({
                "stem":         meta["stem"],
                "timestamp":    meta["timestamp"],
                "path":         path,
                "filter":       filt,
                "laplacian":    m["laplacian"],
                "tenengrad":    m["tenengrad"],
                "norm_variance": m["norm_variance"],
                "raw_score":    raw,
            })
            done += 1
            print(f"\r  [{done:>3}/{total}] {filt:>4}: {path.name}", end="", flush=True)
            if progress_callback is not None:
                progress_callback(done, total)

        print()
        results[filt] = rows

    return results


def normalise_scores(
    scores: Dict[str, List[dict]],
) -> Dict[str, List[dict]]:
    """Add 'norm_score' (0–1) and 'rank' columns to each entry.

    Normalisation is done per filter so that cross-filter comparisons
    reflect relative quality within each filter band.
    """
    for filt, rows in scores.items():
        raw_vals = np.array([r["raw_score"] for r in rows])
        lo, hi = raw_vals.min(), raw_vals.max()
        span = hi - lo if hi > lo else 1.0
        norm_vals = (raw_vals - lo) / span   # 0 = worst, 1 = best

        # Rank: 1 = best
        order = np.argsort(-norm_vals)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(order) + 1)

        for i, row in enumerate(rows):
            row["norm_score"] = float(norm_vals[i])
            row["rank"]       = int(ranks[i])

    return scores


# ── Overlapping window selection ───────────────────────────────────────────────

def find_best_windows(
    scores: Dict[str, List[dict]],
    required_filters: Optional[List[str]] = None,
    window_minutes: float = 15.0,
    cycle_minutes: float = 4.5,
    n_windows: int = 3,
    outlier_sigma: float = 1.5,
    allow_overlap: bool = False,
) -> List[dict]:
    """Find time windows suitable for de-rotation stacking.

    Each window spans *window_minutes* and may contain multiple images per
    filter (one per filter cycle).  Outliers within each filter are removed
    by sigma-clipping before computing the aggregate quality.

    Window quality components (all bounded [0, 1]):
      quality_post  — mean norm_score of included images (higher = sharper)
      snr_factor    — sqrt(n_included / n_expected), capped at 1.0; rewards
                      more frames (SNR ∝ √n for stacking)
      stability     — 1/(1+CV) where CV=std/mean; penalises variable seeing

    Per-filter score = quality_post × snr_factor × stability
    Window quality   = geometric mean across all required filters

    Args:
        scores:           Output of :func:`normalise_scores`.
        required_filters: Filters that must all be present (default: all keys).
        window_minutes:   Duration of the de-rotation time window (minutes).
        cycle_minutes:    Duration of one complete filter cycle (minutes).
                          Used to compute expected images per window.
        n_windows:        Number of windows to return.
        outlier_sigma:    Sigma threshold below which images are excluded
                          (threshold = mean − outlier_sigma × std).
        allow_overlap:    If True windows may share time ranges.  If False
                          (default) each window center must be at least
                          *window_minutes* away from every previously
                          selected window.

    Returns:
        List of window dicts (best first), each containing:
        {
          "center_time":      datetime,
          "window_start":     datetime,
          "window_end":       datetime,
          "window_quality":   float [0, 1],
          "rotation_degrees": float,   # Jupiter longitude spanned ~0.6°/min
          "per_filter": {
            filter: {
              "n_total":        int,
              "n_included":     int,
              "n_excluded":     int,
              "quality_pre":    float,   # mean score before exclusion
              "quality_post":   float,   # mean score after exclusion
              "snr_factor":     float,   # sqrt(n_included / n_expected)
              "stability":      float,   # 1 / (1 + CV)
              "filter_quality": float,   # combined per-filter score
              "included":       [row, ...],
              "excluded":       [row, ...],
            }, ...
          },
        }
    """
    if required_filters is None:
        required_filters = sorted(scores.keys())

    # Prefer IR as anchor (most relevant channel); fall back to first filter
    anchor_filter = "IR" if "IR" in required_filters else required_filters[0]
    anchor_rows = scores.get(anchor_filter, [])
    if not anchor_rows:
        return []

    by_filter: Dict[str, List[dict]] = {
        filt: scores.get(filt, []) for filt in required_filters
    }
    half_window = timedelta(minutes=window_minutes / 2)
    # Expected images per filter per window (float; used to normalise SNR factor)
    n_expected = window_minutes / cycle_minutes

    candidates: List[dict] = []

    for anchor_row in anchor_rows:
        t_center = anchor_row["timestamp"]
        t_start = t_center - half_window
        t_end   = t_center + half_window

        per_filter: Dict[str, dict] = {}
        complete = True

        for filt in required_filters:
            in_window = [
                r for r in by_filter[filt]
                if t_start <= r["timestamp"] <= t_end
            ]
            if not in_window:
                complete = False
                break

            n_total = len(in_window)
            scores_arr = np.array([r["norm_score"] for r in in_window])
            mean_pre = float(scores_arr.mean())
            std_pre  = float(scores_arr.std()) if n_total > 1 else 0.0

            # Sigma-clipping: exclude images below (mean - k*std)
            threshold = mean_pre - outlier_sigma * std_pre
            included = [r for r in in_window if r["norm_score"] >= threshold]
            excluded = [r for r in in_window if r["norm_score"] < threshold]
            if not included:          # safety: keep best even if all clipped
                included = [max(in_window, key=lambda r: r["norm_score"])]
                excluded = []

            n_included  = len(included)
            incl_scores = np.array([r["norm_score"] for r in included])
            mean_post   = float(incl_scores.mean())
            std_post    = float(incl_scores.std()) if n_included > 1 else 0.0

            # SNR factor: relative stacking gain, capped at 1.0
            snr_factor = float(min(1.0, np.sqrt(n_included / n_expected)))

            # Seeing stability: penalise high intra-window variance
            cv        = std_post / mean_post if mean_post > 1e-9 else 1.0
            stability = 1.0 / (1.0 + cv)

            filter_quality = mean_post * snr_factor * stability

            per_filter[filt] = {
                "n_total":        n_total,
                "n_included":     n_included,
                "n_excluded":     len(excluded),
                "quality_pre":    round(mean_pre, 4),
                "quality_post":   round(mean_post, 4),
                "snr_factor":     round(snr_factor, 4),
                "stability":      round(stability, 4),
                "filter_quality": round(filter_quality, 4),
                "included":       included,
                "excluded":       excluded,
            }

        if not complete:
            continue

        # Cross-filter quality: geometric mean of per-filter scores
        fq_vals = [per_filter[f]["filter_quality"] for f in required_filters]
        if all(v > 0 for v in fq_vals):
            window_quality = float(np.prod(fq_vals) ** (1.0 / len(fq_vals)))
        else:
            window_quality = 0.0

        # Rotation span: longitude drift across all included images (~0.6°/min)
        all_times = [
            r["timestamp"]
            for filt in required_filters
            for r in per_filter[filt]["included"]
        ]
        t_min_w, t_max_w = min(all_times), max(all_times)
        rotation_deg = (t_max_w - t_min_w).total_seconds() / 60.0 * 0.6

        candidates.append({
            "center_time":      t_center,
            "window_start":     t_start,
            "window_end":       t_end,
            "window_quality":   round(window_quality, 4),
            "rotation_degrees": round(rotation_deg, 2),
            "per_filter":       per_filter,
        })

    # Sort best-first, then select top-N windows
    candidates.sort(key=lambda w: -w["window_quality"])

    if allow_overlap:
        # Overlapping allowed: just take the best N candidates
        selected = candidates[:n_windows]
    else:
        # Non-overlapping: each window center must be at least window_minutes
        # away from every already-selected window center.
        selected = []
        for win in candidates:
            t_c = win["center_time"]
            gap_ok = all(
                abs((t_c - s["center_time"]).total_seconds() / 60) >= window_minutes
                for s in selected
            )
            if gap_ok:
                selected.append(win)
            if len(selected) >= n_windows:
                break

    return selected


# ── I/O helpers ────────────────────────────────────────────────────────────────

def scores_to_csv_rows(scores: Dict[str, List[dict]]) -> List[dict]:
    """Flatten the scores dict into a list of CSV-ready dicts."""
    rows = []
    for filt, entries in scores.items():
        for e in entries:
            rows.append({
                "timestamp":    e["timestamp"].strftime("%Y-%m-%dT%H:%M:%SZ"),
                "filter":       filt,
                "stem":         e["stem"],
                "laplacian":    round(e["laplacian"], 2),
                "tenengrad":    round(e["tenengrad"], 2),
                "norm_variance": round(e["norm_variance"], 6),
                "raw_score":    round(e["raw_score"], 4),
                "norm_score":   round(e.get("norm_score", 0.0), 6),
                "rank":         e.get("rank", -1),
            })
    rows.sort(key=lambda r: (r["timestamp"], r["filter"]))
    return rows


def windows_to_json(windows: List[dict]) -> dict:
    """Serialise window list to a JSON-compatible dict."""
    def _fmt(dt: datetime) -> str:
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    out = []
    for i, win in enumerate(windows):
        pf_serial: dict = {}
        for filt, info in win["per_filter"].items():
            pf_serial[filt] = {
                "n_total":        info["n_total"],
                "n_included":     info["n_included"],
                "n_excluded":     info["n_excluded"],
                "quality_pre":    info["quality_pre"],
                "quality_post":   info["quality_post"],
                "snr_factor":     info["snr_factor"],
                "stability":      info["stability"],
                "filter_quality": info["filter_quality"],
                "included": [
                    {
                        "stem":       r["stem"],
                        "timestamp":  _fmt(r["timestamp"]),
                        "norm_score": round(r["norm_score"], 4),
                        "rank":       r["rank"],
                    }
                    for r in info["included"]
                ],
                "excluded": [
                    {
                        "stem":       r["stem"],
                        "timestamp":  _fmt(r["timestamp"]),
                        "norm_score": round(r["norm_score"], 4),
                        "rank":       r["rank"],
                    }
                    for r in info["excluded"]
                ],
            }
        out.append({
            "window_index":     i + 1,
            "center_time":      _fmt(win["center_time"]),
            "window_start":     _fmt(win["window_start"]),
            "window_end":       _fmt(win["window_end"]),
            "window_quality":   win["window_quality"],
            "rotation_degrees": win["rotation_degrees"],
            "per_filter":       pf_serial,
        })
    return {"selected_windows": out}


def windows_summary(windows: List[dict]) -> str:
    """Return a human-readable text summary of the selected windows."""
    lines = ["=== Recommended De-rotation Windows ===\n"]
    for i, win in enumerate(windows):
        lines.append(
            f"Window {i+1}  [quality: {win['window_quality']:.4f}  "
            f"rotation: {win['rotation_degrees']:.1f}°]"
        )
        lines.append(
            f"  {win['window_start'].strftime('%H:%M:%S')} – "
            f"{win['window_end'].strftime('%H:%M:%S')}  "
            f"(center: {win['center_time'].strftime('%Y-%m-%dT%H:%M:%SZ')})"
        )
        lines.append("")
        lines.append(
            f"  {'Filter':>6}  {'Total':>5}  {'Kept':>4}  {'Drop':>4}  "
            f"{'Pre-Q':>6}  {'Post-Q':>6}  {'SNR×':>5}  {'Stab':>5}  {'FQ':>6}"
        )
        lines.append("  " + "─" * 64)
        for filt in sorted(win["per_filter"]):
            info = win["per_filter"][filt]
            lines.append(
                f"  {filt:>6}  {info['n_total']:>5}  {info['n_included']:>4}  "
                f"{info['n_excluded']:>4}  {info['quality_pre']:>6.4f}  "
                f"{info['quality_post']:>6.4f}  {info['snr_factor']:>5.3f}  "
                f"{info['stability']:>5.3f}  {info['filter_quality']:>6.4f}"
            )

        # List excluded outliers for human review
        has_excluded = any(
            info["n_excluded"] > 0 for info in win["per_filter"].values()
        )
        if has_excluded:
            lines.append("")
            lines.append("  Excluded outliers:")
            for filt in sorted(win["per_filter"]):
                for r in win["per_filter"][filt]["excluded"]:
                    lines.append(
                        f"    {filt:>4}: {r['stem']}  (score={r['norm_score']:.4f})"
                    )
        lines.append("")
    return "\n".join(lines)
