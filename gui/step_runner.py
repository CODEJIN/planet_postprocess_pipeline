"""QThread-based pipeline step runner with stdout capture.

Architecture
------------
StepRunner runs in a background thread.  It redirects sys.stdout through
LogCapture so every print() in the pipeline emits a log_line signal that
the main thread appends to the LogWidget — no direct GUI calls from the
worker thread.

Usage::

    runner = StepRunner(config, steps_to_run, results_in)
    runner.log_line.connect(log_widget.append_line)
    runner.step_started.connect(sidebar.set_running)
    runner.step_finished.connect(on_step_done)
    runner.start()
"""
from __future__ import annotations

import io
import sys
import threading
import traceback
from pathlib import Path
from typing import Any

from PySide6.QtCore import QThread, Signal

from pipeline.config import PipelineConfig


# ── stdout capture ─────────────────────────────────────────────────────────────

class _LogCapture(io.TextIOBase):
    """Replaces sys.stdout to forward print() calls to a callback."""

    def __init__(self, callback):
        super().__init__()
        self._cb = callback
        self._original = sys.stdout

    def write(self, text: str) -> int:
        if text:
            self._cb(text)
        return len(text)

    def flush(self) -> None:
        pass


# ── step execution map ─────────────────────────────────────────────────────────

def _import_steps():
    """Lazy-import step modules (avoids startup cost)."""
    from pipeline.steps import (
        step01_pipp,
        step02_lucky_stack,
        step03_quality_assess,
        step04_derotate_stack,
        step05_wavelet_master,
        step06_rgb_composite,
        step07_wavelet_preview,
        step08_series_composite,
        step09_gif,
        step10_summary_grid,
    )
    return {
        "01": step01_pipp,
        "02": step02_lucky_stack,
        "03": step03_quality_assess,
        "04": step04_derotate_stack,
        "05": step05_wavelet_master,
        "06": step06_rgb_composite,
        "07": step07_wavelet_preview,
        "08": step08_series_composite,
        "09": step09_gif,
        "10": step10_summary_grid,
    }


# ── worker thread ──────────────────────────────────────────────────────────────

class StepRunner(QThread):
    """Runs one or more pipeline steps sequentially in a background thread.

    Signals
    -------
    log_line(str)                   — one line of stdout output
    step_started(step_id)           — step is beginning
    step_finished(step_id, ok, res) — step completed (ok=True) or failed
    all_done()                      — all requested steps finished
    """

    log_line      = Signal(str)
    step_started  = Signal(str)
    step_finished = Signal(str, bool, object)
    progress      = Signal(str, int, int)   # step_id, current, total
    all_done      = Signal()
    cancelled     = Signal()               # emitted after all_done when aborted

    def __init__(
        self,
        config: PipelineConfig,
        steps: list[str],
        prior_results: dict[str, Any] | None = None,
        parent=None,
    ) -> None:
        """
        Parameters
        ----------
        config        : fully-built PipelineConfig
        steps         : list of step IDs to run, e.g. ["03", "04", "05"]
        prior_results : results from steps that already ran this session
        """
        super().__init__(parent)
        self._config  = config
        self._steps   = steps
        self._results: dict[str, Any] = dict(prior_results or {})
        self._abort   = False
        self._cancel_event = threading.Event()

    def abort(self) -> None:
        """Request graceful stop; signals cancel_event for in-progress steps."""
        self._abort = True
        self._cancel_event.set()

    # ── QThread.run ────────────────────────────────────────────────────────────

    def run(self) -> None:
        capture = _LogCapture(self._emit_line)
        old_stdout = sys.stdout
        sys.stdout  = capture  # type: ignore[assignment]

        try:
            mods = _import_steps()
            for step_id in self._steps:
                if self._abort:
                    break
                self.step_started.emit(step_id)
                ok, res = self._run_one(step_id, mods)
                if ok:
                    self._results[step_id] = res
                self.step_finished.emit(step_id, ok, res)
                if not ok:
                    break   # stop chain on error
        finally:
            sys.stdout = old_stdout

        self.all_done.emit()
        if self._abort:
            self.cancelled.emit()

    # ── internal ───────────────────────────────────────────────────────────────

    def _emit_line(self, text: str) -> None:
        for line in text.splitlines(keepends=True):
            self.log_line.emit(line)

    def _make_progress_cb(self, step_id: str):
        """Return a callable(current, total) that emits the progress signal."""
        def _cb(current: int, total: int) -> None:
            self.progress.emit(step_id, current, total)
        return _cb

    def _run_one(self, step_id: str, mods: dict) -> tuple[bool, Any]:
        cfg  = self._config
        res  = self._results
        pcb  = self._make_progress_cb(step_id)

        try:
            print(f"\n{'='*60}")
            print(f"=== Step {step_id} ===")

            ce = self._cancel_event
            if step_id == "01":
                r = mods["01"].run(cfg, progress_callback=pcb, cancel_event=ce)
            elif step_id == "02":
                r = mods["02"].run(cfg, progress_callback=pcb, cancel_event=ce)
            elif step_id == "03":
                r = mods["03"].run(cfg, progress_callback=pcb, cancel_event=ce)
            elif step_id == "04":
                r = mods["04"].run(cfg, res.get("03", {}), progress_callback=pcb, cancel_event=ce)
            elif step_id == "05":
                r = mods["05"].run(cfg, res.get("04", {}), cancel_event=ce)
            elif step_id == "06":
                r = mods["06"].run(cfg, res.get("05", {}), cancel_event=ce)
            elif step_id == "07":
                r = mods["07"].run(cfg, progress_callback=pcb, cancel_event=ce)
            elif step_id == "08":
                r = mods["08"].run(cfg, res.get("07", {}), progress_callback=pcb, cancel_event=ce)
            elif step_id == "09":
                r = mods["09"].run(cfg, res.get("08", {}), progress_callback=pcb, cancel_event=ce)
            elif step_id == "10":
                r = mods["10"].run(cfg, res.get("06", {}), res.get("04", {}), res.get("05", {}), cancel_event=ce)
            else:
                print(f"  [WARN] Step {step_id} has no runner implementation.")
                r = {}

            return True, r

        except Exception:
            tb = traceback.format_exc()
            for line in tb.splitlines():
                self.log_line.emit("[ERROR] " + line + "\n")
            return False, None
