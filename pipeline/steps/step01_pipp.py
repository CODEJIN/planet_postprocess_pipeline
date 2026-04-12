"""
Step 1 – PIPP-style SER preprocessing.

For each raw SER file in ``config.ser_input_dir``:
  1. **Frame rejection** — discard frames where the planet is:
       - partially outside the frame (clipping check)
       - deformed / non-circular (aspect-ratio check)
       - cut by a straight data-transfer artefact (straight-edge check)
       - too small (below ``pipp.min_diameter``)
       - an abnormal size relative to adjacent accepted frames
         (sliding-window median check with ``pipp.size_tolerance``)
  2. **Centre-align & crop** — crop a square ROI of ``pipp.roi_size`` pixels,
       centred on the geometric centroid of the planet disk.
  3. **Write** the accepted, cropped frames to a new SER file in
       ``<output_base>/step01_pipp/``.

Output (when ``config.save_step01`` is True):
    <output_base>/step01_pipp/
        2026-04-02-1231_5-U-IR-Jup_pipp.ser
        2026-04-02-1231_5-U-IR-Jup_pipp.txt   ← rejection stats
        ...

Return value::

    {
        "<original_stem>": {
            "output_path": Path | None,      # None when save_step01=False
            "input_frames": int,
            "accepted_frames": int,
            "rejection_rate": float,         # 0.0–1.0
        },
        ...
    }
"""
from __future__ import annotations

import multiprocessing as _mp
import threading as _threading
from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor, as_completed as _as_completed
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from pipeline.config import PipelineConfig
from pipeline.modules import planet_detect, ser_io


# ── Public entry point ────────────────────────────────────────────────────────

def run(config: PipelineConfig, progress_callback=None) -> Dict[str, Dict]:
    """Process all SER files found in ``config.ser_input_dir``."""
    ser_dir = config.ser_input_dir
    if not ser_dir.exists():
        print(f"  [WARNING] ser_input_dir does not exist: {ser_dir}")
        return {}

    ser_files: List[Path] = sorted(ser_dir.glob("*.ser"))
    if not ser_files:
        print(f"  [WARNING] No SER files found in {ser_dir}")
        return {}

    out_dir: Optional[Path] = None
    if config.save_step01:
        if config.step01_output_dir is not None:
            out_dir = Path(config.step01_output_dir)
        else:
            out_dir = config.step_dir(1, "pipp")
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Output → {out_dir}")

    n_files = len(ser_files)
    print(f"  Found {n_files} SER file(s) in {ser_dir}")

    # ── Worker count: cap at 4 to avoid disk I/O contention ──────────────────
    _cfg_workers = int(getattr(config.pipp, "n_workers", 0))
    _cpu         = _mp.cpu_count() or 1
    _step1_max   = min(4, _cfg_workers if _cfg_workers > 0 else _cpu)
    _step1_max   = max(1, _step1_max)

    results: Dict[str, Dict] = {}

    if _step1_max == 1 or n_files == 1:
        # ── Sequential path (original behaviour) ─────────────────────────────
        frame_counts: List[int] = []
        if progress_callback is not None:
            for sp in ser_files:
                try:
                    r = ser_io.SERReader(sp)
                    frame_counts.append(r.header["FrameCount"])
                    r.close()
                except Exception:
                    frame_counts.append(0)
            total_frames = sum(frame_counts)
        else:
            total_frames = 0

        frames_offset = 0
        for idx, ser_path in enumerate(ser_files):
            result = _process_one(
                ser_path, out_dir, config,
                progress_callback=progress_callback,
                frames_offset=frames_offset,
                total_frames=total_frames,
            )
            results[ser_path.stem] = result
            frames_offset += frame_counts[idx] if progress_callback else 0

    else:
        # ── Parallel path: up to _step1_max files simultaneously ─────────────
        print(f"  [Step1] Parallel mode: {_step1_max} workers", flush=True)
        _lock      = _threading.Lock()
        _completed = [0]

        def _run_one(ser_path: Path):
            result = _process_one(
                ser_path, out_dir, config,
                progress_callback=None,   # suppress per-frame progress in parallel
                frames_offset=0,
                total_frames=0,
            )
            with _lock:
                _completed[0] += 1
                done = _completed[0]
            if progress_callback is not None:
                progress_callback(done, n_files)
            return ser_path.stem, result

        with _ThreadPoolExecutor(max_workers=_step1_max) as executor:
            futs = [executor.submit(_run_one, sp) for sp in ser_files]
            for fut in _as_completed(futs):
                stem, result = fut.result()
                results[stem] = result

    total_in  = sum(r["input_frames"]    for r in results.values())
    total_out = sum(r["accepted_frames"] for r in results.values())
    rej_rate  = 1.0 - total_out / total_in if total_in else 0.0
    print(
        f"\n  Step 1 complete: {len(results)} files | "
        f"{total_in} → {total_out} frames "
        f"({rej_rate:.1%} rejected)"
    )
    return results


# ── Per-file processing ───────────────────────────────────────────────────────

def _process_one(
    ser_path: Path,
    out_dir: Optional[Path],
    config: PipelineConfig,
    progress_callback=None,
    frames_offset: int = 0,
    total_frames: int = 0,
) -> Dict:
    pipp = config.pipp
    stem = ser_path.stem
    print(f"\n  [{stem}]", end="", flush=True)

    reader = ser_io.SERReader(ser_path)
    num_frames: int = reader.header["FrameCount"]
    timestamps = reader.get_all_timestamps()
    if not timestamps:
        timestamps = [0] * num_frames

    out_path: Optional[Path] = None
    writer: Optional[ser_io.SERWriter] = None
    if out_dir is not None:
        out_path = out_dir / (stem + "_pipp.ser")
        writer = ser_io.SERWriter(out_path, reader.header, pipp.roi_size, pipp.roi_size)

    # Counters
    accepted = 0
    rejected_clip   = 0
    rejected_shape  = 0
    rejected_size   = 0
    rejected_detect = 0

    # Sliding-window size reference (tracks accepted frames only)
    w_history: List[float] = []
    h_history: List[float] = []

    try:
        for i in range(num_frames):
            raw = reader.get_frame(i)

            info = planet_detect.analyze_planet(
                raw,
                min_diameter=pipp.min_diameter,
                aspect_ratio_limit=pipp.aspect_ratio_limit,
                straight_edge_limit=pipp.straight_edge_limit,
            )

            if info is None:
                rejected_detect += 1
                continue

            curr_w = float(info["width"])
            curr_h = float(info["height"])

            # Sliding-window size check (kicks in after 20 accepted frames)
            if len(w_history) >= 20:
                ref_w = float(np.median(w_history))
                ref_h = float(np.median(h_history))
                if (
                    curr_w < ref_w * (1.0 - pipp.size_tolerance)
                    or curr_h < ref_h * (1.0 - pipp.size_tolerance)
                ):
                    rejected_size += 1
                    continue  # rejected frame does NOT update the reference

            # Frame accepted — update sliding window
            w_history.append(curr_w)
            h_history.append(curr_h)
            if len(w_history) > pipp.window_size:
                w_history.pop(0)
                h_history.pop(0)

            # Demosaic (no-op for mono), crop, write
            rgb = reader.to_rgb(raw)
            cropped = planet_detect.get_cropped_frame(
                rgb, info["centroid"], pipp.roi_size
            )
            if writer is not None:
                writer.write_frame(cropped, timestamps[i])
            accepted += 1

            if i % 500 == 0:
                print(f"\r  [{stem}]  {i}/{num_frames}", end="", flush=True)
                if progress_callback is not None and total_frames > 0:
                    progress_callback(frames_offset + i, total_frames)

    finally:
        if writer is not None:
            writer.close()
        reader.close()

    total_rejected = num_frames - accepted
    rej_rate = total_rejected / num_frames if num_frames else 0.0

    # 0-frame result: delete the output file (empty SER is useless)
    if accepted == 0 and out_path is not None and out_path.exists():
        out_path.unlink()
        out_path = None
        print(
            f"\r  [{stem}]  {num_frames} → 0 frames — "
            f"no planet detected, output deleted"
        )
    else:
        print(
            f"\r  [{stem}]  {num_frames} → {accepted} frames "
            f"(rejected: {total_rejected} = detect:{rejected_detect} "
            f"size:{rejected_size})"
        )

    # Write a small stats sidecar for inspection
    if out_dir is not None:
        _write_stats(
            out_dir / (stem + "_pipp.txt"),
            ser_path, num_frames, accepted,
            rejected_detect, rejected_size,
        )

    return {
        "output_path": out_path,
        "input_frames": num_frames,
        "accepted_frames": accepted,
        "rejection_rate": rej_rate,
    }


def _write_stats(
    txt_path: Path,
    ser_path: Path,
    total: int,
    accepted: int,
    rejected_detect: int,
    rejected_size: int,
) -> None:
    lines = [
        f"Input:            {ser_path}",
        f"Total frames:     {total}",
        f"Accepted frames:  {accepted}",
        f"Rejected (no planet / clipped / shape): {rejected_detect}",
        f"Rejected (size anomaly):                {rejected_size}",
        f"Rejection rate:   {(total - accepted) / total:.1%}" if total else "N/A",
    ]
    txt_path.write_text("\n".join(lines) + "\n")
