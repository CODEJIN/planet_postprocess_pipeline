"""
Step 2 — Lucky stacking: SER video frames → stacked TIF.

Reads PIPP-preprocessed SER files (step01 output) and runs AS!4-style lucky
stacking to produce one 16-bit TIF per SER file.

Input:
    Scans ``config.ser_input_dir`` (or ``config.step01_output_dir``) for SER
    files matching ``*_pipp.ser``.  Falls back to any ``*.ser`` files if no
    ``_pipp.ser`` are found.

Output:
    <output_base>/step02_lucky_stack/
        2026-04-07-1114_7-U-IR-Jup_pipp_lucky.tif
        2026-04-07-1114_7-U-IR-Jup_pipp_lucky.json   ← processing log
        ...

Output filename convention:
    ``<original_stem>_lucky.tif``
    The ``_pipp_lucky`` suffix (after the target group) is transparent to the
    ``parse_filename`` regex in image_io.py, so steps 03-10 consume these TIFs
    without any changes.

Return value::

    {
        "<stem>": {
            "output_path": Path | None,
            "input_frames": int,
            "stacked_frames": int,
            "rejection_rate": float,
            "disk_radius_px": float,
            "n_aps": int,
            "timing_s": dict,
        },
        ...
    }
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from pipeline.config import PipelineConfig
from pipeline.modules import image_io
from pipeline.modules.lucky_stack import lucky_stack_ser


# ── Public entry point ────────────────────────────────────────────────────────

def run(
    config: PipelineConfig,
    progress_callback=None,
) -> Dict[str, Dict]:
    """Process all SER files found in the step01 output directory.

    Args:
        config:            Full pipeline config (uses config.lucky_stack sub-config).
        progress_callback: Optional (done, total) callback for UI progress.

    Returns:
        Dict keyed by SER stem with per-file processing results.
    """
    # ── Locate SER input directory ────────────────────────────────────────────
    ser_dir: Optional[Path] = None
    ser_files: List[Path] = []

    gui_ser_dir = getattr(config, "step02_ser_dir", None)
    print(f"  [Step2] GUI SER dir: {gui_ser_dir}")

    if gui_ser_dir is not None:
        # GUI explicitly chose this directory — use it exclusively, no silent fallback.
        p = Path(gui_ser_dir)
        if not p.exists():
            print(f"  [ERROR] Step 2: SER input directory not found: {p}")
            return {}
        pipp_files = sorted(p.glob("*_pipp.ser"))
        ser_files  = pipp_files if pipp_files else sorted(p.glob("*.ser"))
        if not ser_files:
            print(f"  [ERROR] Step 2: No SER files in: {p}")
            return {}
        ser_dir = p
    else:
        # GUI left it blank → auto-detect from fallback chain.
        candidates: List[Path] = []
        if config.step01_output_dir is not None:
            candidates.append(Path(config.step01_output_dir))
        candidates.append(config.output_base_dir / "step01_pipp")
        if hasattr(config, "ser_input_dir") and config.ser_input_dir is not None:
            raw = Path(config.ser_input_dir)
            if raw != Path("."):   # skip the "." default that build_config injects
                candidates.append(raw)

        for cand in candidates:
            if cand.exists():
                pipp_files = sorted(cand.glob("*_pipp.ser"))
                if pipp_files:
                    ser_dir = cand
                    ser_files = pipp_files
                    break
                any_ser = sorted(cand.glob("*.ser"))
                if any_ser:
                    ser_dir = cand
                    ser_files = any_ser
                    break

        if ser_dir is None or not ser_files:
            print(
                "  [WARNING] Step 2: No SER files found.\n"
                "  Searched: " + ", ".join(str(c) for c in candidates)
            )
            return {}

    print(f"  [Step2] Using SER dir: {ser_dir}")

    # ── Output directory ──────────────────────────────────────────────────────
    out_dir: Optional[Path] = None
    if config.save_step02:
        if config.step02_output_dir is not None:
            out_dir = Path(config.step02_output_dir)
        else:
            out_dir = config.step_dir(2, "lucky_stack")
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Output → {out_dir}")

    print(f"  Found {len(ser_files)} SER file(s) in {ser_dir}")

    # ── Per-file processing ───────────────────────────────────────────────────
    results: Dict[str, Dict] = {}
    n_files = len(ser_files)

    for file_idx, ser_path in enumerate(ser_files):
        print(f"\n  [{file_idx+1}/{n_files}] {ser_path.name}", flush=True)

        def _file_prog(done: int, total: int) -> None:
            if progress_callback is not None:
                # Map intra-file progress into the overall file range
                overall_done = file_idx * total + done
                overall_total = n_files * total
                progress_callback(overall_done, overall_total)

        result = _process_one(ser_path, out_dir, config, progress_callback=_file_prog)
        results[ser_path.stem] = result

    # ── Summary ───────────────────────────────────────────────────────────────
    total_in = sum(r["input_frames"] for r in results.values())
    total_stacked = sum(r["stacked_frames"] for r in results.values())
    ok = sum(1 for r in results.values() if r["output_path"] is not None)
    print(
        f"\n  Step 2 complete: {ok}/{len(results)} files OK | "
        f"{total_in} raw → {total_stacked} stacked frames"
    )
    return results


# ── Per-file processing ───────────────────────────────────────────────────────

def _process_one(
    ser_path: Path,
    out_dir: Optional[Path],
    config: PipelineConfig,
    progress_callback=None,
) -> Dict:
    """Run lucky stacking on a single SER file.

    Returns a result dict regardless of success/failure.
    """
    cfg = config.lucky_stack
    stem = ser_path.stem

    try:
        stacked, log = lucky_stack_ser(ser_path, cfg, progress_callback=progress_callback)
    except Exception as exc:
        print(f"\n  ERROR processing {ser_path.name}: {exc}")
        return {
            "output_path": None,
            "input_frames": 0,
            "stacked_frames": 0,
            "rejection_rate": 1.0,
            "disk_radius_px": 0.0,
            "n_aps": 0,
            "timing_s": {},
            "error": str(exc),
        }

    # ── Save TIF + JSON log ───────────────────────────────────────────────────
    out_path: Optional[Path] = None
    if out_dir is not None:
        out_path = out_dir / (stem + "_lucky.tif")
        image_io.write_tif_16bit(stacked, out_path)

        log_path = out_dir / (stem + "_lucky.json")
        # Remove large per-frame list for compact log (keep summary only)
        log_compact = {k: v for k, v in log.items() if k != "frames"}
        log_compact["n_frames_logged"] = len(log.get("frames", []))
        log_path.write_text(json.dumps(log_compact, indent=2))

        print(f"  Saved: {out_path.name}", flush=True)

    n_in = log.get("n_frames_total", 0)
    n_stacked = log.get("n_stacked", 0)
    rej_rate = 1.0 - n_stacked / n_in if n_in else 0.0

    return {
        "output_path": out_path,
        "input_frames": n_in,
        "stacked_frames": n_stacked,
        "rejection_rate": round(rej_rate, 4),
        "disk_radius_px": log.get("disk_radius_px", 0.0),
        "n_aps": log.get("n_aps", 0),
        "timing_s": log.get("timing_s", {}),
    }
