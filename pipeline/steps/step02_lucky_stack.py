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
import multiprocessing as _mp
import threading as _threading
from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor, as_completed as _as_completed
from pathlib import Path
from typing import Dict, List, Optional

from pipeline.config import PipelineConfig
from pipeline.modules import image_io
from pipeline.modules.lucky_stack import lucky_stack_ser, compute_session_aps_from_ser


# ── Public entry point ────────────────────────────────────────────────────────

def run(
    config: PipelineConfig,
    progress_callback=None,
    cancel_event=None,
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

    # ── Session-wide AS4 AP pre-computation ──────────────────────────────────
    cfg = config.lucky_stack
    _session_aps = None
    _session_ref_cx = 0.0
    _session_ref_cy = 0.0

    if getattr(cfg, "use_as4_ap_grid", False):
        ref_ser = _pick_reference_ser(ser_files, getattr(cfg, "ap_reference_filter", ""))
        if ref_ser is not None:
            print(f"  [Step2] Session AP reference: {ref_ser.name}", flush=True)
            try:
                _session_aps, _session_ref_cx, _session_ref_cy, _ref_r = \
                    compute_session_aps_from_ser(ref_ser, cfg)
                from collections import Counter as _Ctr
                _c = _Ctr(sz for _, _, sz in _session_aps)
                print(
                    f"  [Step2] Session APs: {len(_session_aps)} pts "
                    + " ".join(f"{sz}px×{_c[sz]}" for sz in sorted(_c))
                    + f"  (ref disk cx={_session_ref_cx:.0f} cy={_session_ref_cy:.0f} r={_ref_r:.0f}px)",
                    flush=True,
                )
            except Exception as _e:
                print(f"  [Step2] WARNING: Session AP computation failed ({_e}) — falling back to per-SER grid.", flush=True)
                _session_aps = None
        else:
            print("  [Step2] WARNING: No suitable reference SER found — falling back to per-SER grid.", flush=True)

    # ── Per-file processing ───────────────────────────────────────────────────
    results: Dict[str, Dict] = {}
    n_files = len(ser_files)

    n_ser_parallel = int(getattr(cfg, "n_ser_parallel", 1))
    if n_ser_parallel <= 0:
        n_ser_parallel = max(1, _mp.cpu_count() // 4)
    n_ser_parallel = min(n_ser_parallel, n_files)

    _file_counter = [0]          # completed file count (for progress mapping)
    _counter_lock = _threading.Lock()

    def _run_one(file_idx_and_path):
        file_idx, ser_path = file_idx_and_path
        if cancel_event is not None and cancel_event.is_set():
            print(f"\n  [CANCELLED] Skipping {ser_path.name}", flush=True)
            return ser_path.stem, {"input_frames": 0, "stacked_frames": 0, "output_path": None}
        print(f"\n  [{file_idx+1}/{n_files}] {ser_path.name}", flush=True)

        def _file_prog(done: int, total: int) -> None:
            if progress_callback is not None:
                with _counter_lock:
                    completed = _file_counter[0]
                overall_done  = completed * total + done
                overall_total = n_files * total
                progress_callback(overall_done, overall_total)

        result = _process_one(
            ser_path, out_dir, config,
            progress_callback=_file_prog,
            session_aps=_session_aps,
            session_ref_cx=_session_ref_cx,
            session_ref_cy=_session_ref_cy,
            cancel_event=cancel_event,
        )
        with _counter_lock:
            _file_counter[0] += 1
        return ser_path.stem, result

    if n_ser_parallel > 1:
        print(f"  [Step2] Processing {n_files} SER files with {n_ser_parallel} parallel workers", flush=True)
        with _ThreadPoolExecutor(max_workers=n_ser_parallel) as executor:
            futs = {executor.submit(_run_one, (i, p)): p
                    for i, p in enumerate(ser_files)}
            for fut in _as_completed(futs):
                stem, result = fut.result()
                results[stem] = result
    else:
        for file_idx, ser_path in enumerate(ser_files):
            if cancel_event is not None and cancel_event.is_set():
                print("  [CANCELLED] Stopping SER processing.", flush=True)
                break
            stem, result = _run_one((file_idx, ser_path))
            results[stem] = result

    # ── Summary ───────────────────────────────────────────────────────────────
    total_in = sum(r["input_frames"] for r in results.values())
    total_stacked = sum(r["stacked_frames"] for r in results.values())
    ok = sum(1 for r in results.values() if r["output_path"] is not None)
    print(
        f"\n  Step 2 complete: {ok}/{len(results)} files OK | "
        f"{total_in} raw → {total_stacked} stacked frames"
    )
    return results


# ── Filename helpers ─────────────────────────────────────────────────────────

_FILTER_PRIORITY = ["IR", "R", "G", "B", "CH4", "color"]

_KNOWN_FILTERS = {"IR", "R", "G", "B", "CH4"}


def _extract_filter_from_stem(stem: str) -> Optional[str]:
    """Return the filter name embedded in a FireCapture-style stem, or None."""
    for flt in _KNOWN_FILTERS:
        if f"-{flt}-" in stem or f"_{flt}_" in stem:
            return flt
    return None


# ── Reference SER selection for session-wide AP sharing ──────────────────────


def _pick_reference_ser(ser_files: List[Path], forced_filter: str = "") -> Optional[Path]:
    """Return the best reference SER for session-wide AP generation.

    If forced_filter is set, pick from SERs matching that filter only.
    Otherwise use priority order: IR > R > G > B > CH4 > color.
    Among candidates with the same filter, pick the temporally central one
    (middle of sorted list) as most representative of the session.
    """
    if forced_filter:
        candidates = [f for f in ser_files if forced_filter.upper() in f.stem.upper()]
        if not candidates:
            candidates = ser_files
    else:
        candidates = None
        for flt in _FILTER_PRIORITY:
            matched = [f for f in ser_files if f"-{flt}-" in f.stem or f"_{flt}_" in f.stem
                       or f.stem.upper().endswith(f"-{flt.upper()}")
                       or f"-{flt.upper()}-" in f.stem.upper()]
            if matched:
                candidates = matched
                break
        if candidates is None:
            candidates = ser_files

    if not candidates:
        return None
    # Pick temporally central SER (sorted filenames encode timestamps)
    return sorted(candidates)[len(candidates) // 2]


# ── Per-file processing ───────────────────────────────────────────────────────

def _process_one(
    ser_path: Path,
    out_dir: Optional[Path],
    config: PipelineConfig,
    progress_callback=None,
    session_aps=None,
    session_ref_cx: float = 0.0,
    session_ref_cy: float = 0.0,
    cancel_event=None,
) -> Dict:
    """Run lucky stacking on a single SER file.

    Returns a result dict regardless of success/failure.
    """
    cfg = config.lucky_stack
    stem = ser_path.stem

    try:
        stacked, log = lucky_stack_ser(
            ser_path, cfg,
            progress_callback=progress_callback,
            session_aps=session_aps,
            session_ref_cx=session_ref_cx,
            session_ref_cy=session_ref_cy,
            cancel_event=cancel_event,
        )
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
        filter_name = ("color" if config.camera_mode == "color"
                       else _extract_filter_from_stem(stem) or "L")
        out_stem = image_io.infer_winjupos_stem(
            ser_path, filter_name=filter_name, target=config.target
        )
        out_path = out_dir / (out_stem + "_lucky.tif")
        image_io.write_tif_16bit(stacked, out_path)

        log_path = out_dir / (out_stem + "_lucky.json")
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
