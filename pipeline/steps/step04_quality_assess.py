"""
Step 4 – Automated quality assessment and optimal time-window selection.

Evaluates every stacked TIF with sharpness and contrast metrics computed
on the planet disk only, then identifies 1–3 overlapping time windows
where all required filters have simultaneously good quality.

Output (when config.save_step04 is True):
    <output_base>/step04_quality/
        quality_scores.csv      — per-file scores and per-filter rankings
        windows.json            — recommended windows (machine-readable)
        windows_summary.txt     — human-readable window summary
        <FILTER>_ranking.csv    — per-filter sorted ranking
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pipeline.config import PipelineConfig
from pipeline.modules import image_io, quality


def run(
    config: PipelineConfig,
    groups: Optional[Dict[str, List[Tuple[Path, dict]]]] = None,
    progress_callback=None,
) -> dict:
    """Run Step 4 for all TIF files in *config.input_dir*.

    Args:
        config: Pipeline configuration.
        groups: Pre-computed filter groups (from image_io.group_by_filter).
                If None, re-scanned from config.input_dir.

    Returns:
        {
          "scores":  {filter: [row_dict, ...]},        # normalised
          "windows": [window_dict, ...],               # top-N windows
          "groups":  {filter: [(path, meta), ...]},    # for downstream steps
        }
    """
    # ── Output directory ───────────────────────────────────────────────────────
    out_dir: Optional[Path] = None
    if config.save_step04:
        out_dir = config.step_dir(4, "quality")
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Output → {out_dir}")
    else:
        print("  save_step04=False: results not written to disk")

    # ── Discover files ─────────────────────────────────────────────────────────
    if groups is None:
        groups = image_io.group_by_filter(config.input_dir, config.target)
    if not groups:
        print(f"  [WARNING] No matching TIF files found in {config.input_dir}")
        return {}

    total = sum(len(v) for v in groups.values())
    print(f"  Scoring {total} files across {len(groups)} filters…")

    # ── Compute quality metrics ────────────────────────────────────────────────
    scores = quality.compute_scores(
        groups,
        lap_w=config.quality.laplacian_weight,
        ten_w=config.quality.fourier_hf_weight,     # re-used for Tenengrad
        nv_w =config.quality.norm_variance_weight,
        progress_callback=progress_callback,
    )
    scores = quality.normalise_scores(scores)

    # ── Find overlapping windows ───────────────────────────────────────────────
    # Optional: drop frames below quality threshold before window search
    min_q = config.quality.min_quality_threshold
    if min_q > 0.0:
        before = sum(len(v) for v in scores.values())
        scores = {
            filt: [r for r in rows if r.get("norm_score", 1.0) >= min_q]
            for filt, rows in scores.items()
        }
        after = sum(len(v) for v in scores.values())
        if before != after:
            print(f"  Filtered {before - after} frame(s) below threshold {min_q:.2f}")

    cycle_min  = config.quality.cycle_minutes
    window_min = config.quality.window_minutes

    overlap_note = " [겹침허용]" if config.quality.allow_overlap else ""
    print(f"\n  Searching for top de-rotation windows "
          f"(window={window_min:.1f} min, "
          f"cycle={cycle_min:.2f} min, "
          f"n={config.quality.n_windows}, "
          f"σ={config.quality.outlier_sigma}{overlap_note})…")
    windows = quality.find_best_windows(
        scores,
        required_filters=list(scores.keys()) if config.camera_mode == "color" else config.filters,
        window_minutes=window_min,
        cycle_minutes=cycle_min,
        n_windows=config.quality.n_windows,
        outlier_sigma=config.quality.outlier_sigma,
        allow_overlap=config.quality.allow_overlap,
    )
    print(f"  Found {len(windows)} de-rotation window(s)")

    # ── Save outputs ───────────────────────────────────────────────────────────
    if out_dir is not None:
        _save_csv(scores, out_dir)
        _save_per_filter_rankings(scores, out_dir)
        _save_windows(windows, out_dir)
        print(f"\n  Saved quality scores and window recommendations to {out_dir}")

    # ── Print summary to console ───────────────────────────────────────────────
    summary = quality.windows_summary(windows)
    print()
    print(summary)

    return {
        "scores":  scores,
        "windows": windows,
        "groups":  groups,
    }


# ── Private helpers ────────────────────────────────────────────────────────────

def _save_csv(scores: dict, out_dir: Path) -> None:
    """Write quality_scores.csv."""
    rows = quality.scores_to_csv_rows(scores)
    csv_path = out_dir / "quality_scores.csv"
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  → {csv_path.name}  ({len(rows)} rows)")


def _save_per_filter_rankings(scores: dict, out_dir: Path) -> None:
    """Write <FILTER>_ranking.csv for each filter, sorted best-first."""
    for filt, rows in scores.items():
        sorted_rows = sorted(rows, key=lambda r: r["rank"])
        csv_path = out_dir / f"{filt}_ranking.csv"
        fieldnames = ["rank", "norm_score", "raw_score", "laplacian",
                      "tenengrad", "norm_variance", "timestamp", "stem"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames,
                                    extrasaction="ignore")
            writer.writeheader()
            for r in sorted_rows:
                writer.writerow({
                    **r,
                    "timestamp": r["timestamp"].strftime("%Y-%m-%dT%H:%M:%SZ"),
                })
        print(f"  → {csv_path.name}  ({len(sorted_rows)} rows)")


def _save_windows(windows: list, out_dir: Path) -> None:
    """Write windows.json and windows_summary.txt."""
    # JSON
    json_path = out_dir / "windows.json"
    with open(json_path, "w") as f:
        json.dump(quality.windows_to_json(windows), f, indent=2)
    print(f"  → {json_path.name}")

    # Human-readable summary
    txt_path = out_dir / "windows_summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(quality.windows_summary(windows))
    print(f"  → {txt_path.name}")
