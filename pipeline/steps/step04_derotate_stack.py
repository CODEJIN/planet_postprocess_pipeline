"""
Step 4 – De-rotation stacking.

For each selected time window from Step 3:
  1. Rotate each included image to the window center time (System II period).
  2. Sub-pixel translate-align rotated frames via phase correlation.
  3. Combine with quality-weighted mean stack (weights = Step 3 norm_scores).
  4. Save one 16-bit TIF per filter per window.

Output (when config.save_step04 is True):
    <output_base>/step04_derotated/
        window_01/
            IR_derotated.tif
            R_derotated.tif
            G_derotated.tif
            B_derotated.tif
            CH4_derotated.tif
            derotation_log.json
        window_02/
            ...
        derotation_summary.txt
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pipeline.config import PipelineConfig
from pipeline.modules import derotation, image_io


def run(
    config: PipelineConfig,
    results_03: dict,
    progress_callback=None,
    cancel_event=None,
) -> Dict[str, List[Dict]]:
    """Run Step 4 de-rotation stacking.

    Args:
        config:      Pipeline configuration.
        results_03:  Output of step03_quality_assess.run(), containing:
                     - "windows": list of window dicts from quality.find_best_windows()
                     - "groups":  {filter: [(path, meta), ...]}

    Returns:
        {
          "windows": [
            {
              "window_index": int,
              "center_time":  str,
              "outputs": {filter: path_or_None, ...},
              "log":     {filter: log_dict, ...},
            }, ...
          ]
        }
    """
    windows: List[dict] = results_03.get("windows", [])
    if not windows:
        print("  [WARNING] No time windows from Step 3 — de-rotation skipped.")
        return {"windows": []}

    print(f"  Processing {len(windows)} window(s) × {len(config.filters)} filter(s)…")
    print(f"  Period: {config.derotation.rotation_period_hours}h  "
          f"|  warp_scale: {config.derotation.warp_scale}  "
          f"|  sub-pixel alignment: enabled")

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
    summary_lines: List[str] = ["=== Step 4 De-rotation Summary ===\n"]

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

        # ── Look up north pole position angle ─────────────────────────────────
        np_ang = derotation.query_horizons_np_ang(
            horizons_id=config.derotation.horizons_id,
            t_utc=t_center,
            observer_code=config.derotation.observer_code,
        )
        pole_pa_deg = np_ang if np_ang is not None else 0.0
        if np_ang is None:
            print(f"    [WARNING] NP.ang not available → pole_pa_deg = 0.0°")

        # Create per-window output directory
        win_out_dir: Optional[Path] = None
        if out_base is not None:
            win_out_dir = out_base / f"window_{win_idx:02d}"
            win_out_dir.mkdir(parents=True, exist_ok=True)

        # De-rotate all filters in this window
        filter_results = derotation.derotate_window(
            window=window,
            required_filters=list(window["per_filter"].keys()) if config.camera_mode == "color" else config.filters,
            period_hours=config.derotation.rotation_period_hours,
            warp_scale=config.derotation.warp_scale,
            align=True,
            normalize_brightness=config.derotation.normalize_brightness,
            min_quality_threshold=config.derotation.min_quality_threshold,
            pole_pa_deg=pole_pa_deg,
            color_mode=(config.camera_mode == "color"),
            out_dir=win_out_dir,
        )

        # Build log dict and save JSON
        log_dict = derotation.derotation_log_to_json(win_idx, window, filter_results)
        if win_out_dir is not None:
            json_path = win_out_dir / "derotation_log.json"
            with open(json_path, "w") as f:
                json.dump(log_dict, f, indent=2)
            print(f"    → {json_path.name}")

        # Summary lines for this window
        summary_lines.append(
            f"Window {win_idx}  {t_center_str}  "
            f"quality={window['window_quality']:.4f}  "
            f"rotation_span={window['rotation_degrees']:.1f}°"
        )
        summary_filters = list(filter_results.keys()) if config.camera_mode == "color" else config.filters
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
        })

    if progress_callback is not None:
        progress_callback(n_windows, n_windows)

    # ── Save summary ───────────────────────────────────────────────────────────
    summary_text = "\n".join(summary_lines)
    print()
    print(summary_text)

    if out_base is not None:
        txt_path = out_base / "derotation_summary.txt"
        with open(txt_path, "w") as f:
            f.write(summary_text)
        print(f"  → {txt_path}")

    return {"windows": all_results}
