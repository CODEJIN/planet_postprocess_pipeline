"""Levels (black-point / gamma) before/after preview widget — used by Step 10.

Loads one representative PNG from ``input_dir`` (Step 7 composite output),
applies the levels adjustment used by step10_summary_grid, and shows the
original vs. adjusted image side by side.

Auto-renders on first show; re-renders (debounced 400 ms) when params change.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from gui.i18n import S

import numpy as np
from PySide6.QtCore import QObject, QThread, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

_PANEL_SIZE   = 200
_PANEL_STYLE  = "QLabel { background: #1a1a1a; border: 1px solid #444; border-radius: 4px; }"
_CAP_STYLE    = "color: #888; font-size: 10px;"
_STATUS_STYLE = "color: #666; font-size: 10px; font-style: italic;"


# ── Background worker ──────────────────────────────────────────────────────────

class _Worker(QObject):
    done  = Signal(bytes, bytes, int, int, int, str)   # orig, adjusted, h, w, ch, name
    error = Signal(str)

    def __init__(
        self,
        png_path: Path,
        black_point: float,
        white_point: float,
        gamma: float,
    ) -> None:
        super().__init__()
        self._path        = png_path
        self._black_point = black_point
        self._white_point = white_point
        self._gamma       = gamma

    @Slot()
    def run(self) -> None:
        try:
            from pipeline.modules import image_io
            orig = image_io.read_png(self._path)   # float32 [0,1]
            adj  = _apply_levels(orig, self._black_point, self._white_point, self._gamma)

            orig_u8 = _to_uint8_levels(orig)
            adj_u8  = _to_uint8_levels(adj)

            if orig_u8.ndim == 2:
                h, w, ch = *orig_u8.shape, 1
            else:
                h, w, ch = orig_u8.shape

            self.done.emit(
                bytes(np.ascontiguousarray(orig_u8)),
                bytes(np.ascontiguousarray(adj_u8)),
                h, w, ch,
                self._path.name,
            )
        except Exception as exc:
            self.error.emit(str(exc))


# ── Helpers ────────────────────────────────────────────────────────────────────

def _apply_levels(img: np.ndarray, black_point: float, white_point: float, gamma: float) -> np.ndarray:
    span = max(white_point - black_point, 1e-8)
    out  = (img.clip(black_point, white_point) - black_point) / span
    if abs(gamma - 1.0) > 1e-6:
        out = np.power(out, 1.0 / gamma)
    return out.clip(0.0, 1.0).astype(np.float32)


def _to_uint8_stretch(arr: np.ndarray) -> np.ndarray:
    """Percentile stretch for the 'original' panel so dim images are visible."""
    lo = float(np.percentile(arr, 0.5))
    hi = float(np.percentile(arr, 99.5))
    if hi <= lo:
        hi = lo + 1e-6
    return ((arr.clip(lo, hi) - lo) / (hi - lo) * 255).astype(np.uint8)


def _to_uint8_levels(arr: np.ndarray) -> np.ndarray:
    """Direct [0,1] → uint8 for the levels-adjusted panel (no extra stretch)."""
    return (arr * 255).clip(0, 255).astype(np.uint8)


def _bytes_to_pixmap(data: bytes, h: int, w: int, ch: int) -> QPixmap:
    arr = np.frombuffer(data, dtype=np.uint8).reshape((h, w) if ch == 1 else (h, w, ch))
    if ch == 1:
        qimg = QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8)
    else:
        rgb = np.ascontiguousarray(arr[:, :, :3])
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
    px = QPixmap.fromImage(qimg.copy())
    if max(w, h) > _PANEL_SIZE:
        px = px.scaled(
            _PANEL_SIZE, _PANEL_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
    return px


def _pick_png(folder: Path) -> Optional[Path]:
    """Pick a representative PNG — prefer an RGB composite."""
    if not folder.is_dir():
        return None
    pngs = sorted(folder.rglob("*.png"))
    # Prefer files named 'RGB' or 'IR-RGB' for a representative image
    for preferred in ("RGB.png", "IR-RGB.png"):
        for p in pngs:
            if p.name == preferred:
                return p
    return pngs[0] if pngs else None


def _make_img_label() -> QLabel:
    lbl = QLabel()
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lbl.setStyleSheet(_PANEL_STYLE)
    lbl.setFixedSize(_PANEL_SIZE, _PANEL_SIZE)
    lbl.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    return lbl


# ── Widget ─────────────────────────────────────────────────────────────────────

class LevelsPreviewWidget(QWidget):
    """Before / after panel showing the effect of black_point + gamma levels."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._input_dir:   Optional[Path] = None
        self._black_point: float = 0.04
        self._white_point: float = 1.0
        self._gamma:       float = 0.9

        self._running = False
        self._pending = False
        self._thread: Optional[QThread] = None
        self._worker: Optional[_Worker] = None

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._do_update)

        self._build_ui()

    # ── UI ─────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 4, 0, 0)
        root.setSpacing(4)

        self._header_lbl = QLabel(S("preview.label"))
        self._header_lbl.setStyleSheet("color: #aaa; font-size: 11px;")
        root.addWidget(self._header_lbl)

        self._status_lbl = QLabel(S("preview.status.step7"))
        self._status_lbl.setStyleSheet(_STATUS_STYLE)
        self._status_lbl.setWordWrap(True)
        root.addWidget(self._status_lbl)

        panels = QHBoxLayout()
        panels.setSpacing(8)

        self._orig_lbl = _make_img_label()
        self._adj_lbl  = _make_img_label()

        self._cap_orig_lbl = QLabel(S("preview.cap.before"))
        self._cap_adj_lbl  = QLabel(S("preview.cap.after"))
        for img_lbl, cap_lbl in (
            (self._orig_lbl, self._cap_orig_lbl),
            (self._adj_lbl,  self._cap_adj_lbl),
        ):
            col = QVBoxLayout()
            col.setSpacing(2)
            col.addWidget(img_lbl)
            cap_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            cap_lbl.setStyleSheet(_CAP_STYLE)
            col.addWidget(cap_lbl)
            panels.addLayout(col)

        root.addLayout(panels)
        root.addStretch()

    # ── Public API ──────────────────────────────────────────────────────────────

    def retranslate(self) -> None:
        self._header_lbl.setText(S("preview.label"))
        self._status_lbl.setText(S("preview.status.step7"))
        self._cap_orig_lbl.setText(S("preview.cap.before"))
        self._cap_adj_lbl.setText(S("preview.cap.after"))

    def set_input_dir(self, folder) -> None:
        if folder:
            self._input_dir = Path(str(folder))
        else:
            self._input_dir = None

        if self._input_dir is None:
            self._status_lbl.setText(S("preview.status.step7"))
        elif self.isVisible():
            self.schedule_update(100)

    def set_params(self, black_point: float, white_point: float = 1.0, gamma: float = 1.0) -> None:
        self._black_point = black_point
        self._white_point = white_point
        self._gamma       = gamma

    def schedule_update(self, delay: int = 400) -> None:
        if self._input_dir is None:
            return
        self._timer.start(delay)

    # ── Qt events ───────────────────────────────────────────────────────────────

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._input_dir is not None and not self._running:
            self.schedule_update(150)

    # ── Internal ────────────────────────────────────────────────────────────────

    def _do_update(self) -> None:
        if self._input_dir is None:
            return
        if self._running:
            self._pending = True
            return

        png = _pick_png(self._input_dir)
        if png is None:
            if not self._input_dir.is_dir():
                self._status_lbl.setText(S("preview.run_step7_first"))
            else:
                self._status_lbl.setText(S("preview.no_png", d=self._input_dir))
            return

        self._running = True
        self._pending = False
        self._status_lbl.setText(f"{S('preview.rendering')}  {png.name}")

        worker = _Worker(png, self._black_point, self._white_point, self._gamma)
        thread = QThread(self)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.done.connect(self._on_done)
        worker.error.connect(self._on_error)
        worker.done.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(thread.deleteLater)

        self._thread = thread
        self._worker = worker
        thread.start()

    def _on_done(self, orig_b: bytes, adj_b: bytes, h: int, w: int, ch: int, name: str) -> None:
        self._running = False
        self._thread  = None
        self._worker  = None

        self._orig_lbl.setPixmap(_bytes_to_pixmap(orig_b, h, w, ch))
        self._adj_lbl.setPixmap(_bytes_to_pixmap(adj_b, h, w, ch))
        self._status_lbl.setText(name)

        if self._pending:
            self._pending = False
            self.schedule_update(200)

    def _on_error(self, msg: str) -> None:
        self._running = False
        self._thread  = None
        self._worker  = None
        self._status_lbl.setText(S("preview.error", msg=msg))
        if self._pending:
            self._pending = False
