"""Wavelet before/after preview widget (shared by Step 3 and Step 6).

Loads one representative TIF from ``input_dir``, applies wavelet sharpening
in a background thread, and shows original vs. sharpened side by side.

Public API
----------
preview = WaveletPreviewWidget(sharpen_filter=0.1)
preview.set_input_dir("/path/to/tifs")
preview.set_params(amounts=[200,200,200,0,0,0])
preview.trigger_update()           # immediate
preview.schedule_update(delay=400) # debounced
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

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

_PANEL_SIZE = 200   # px per preview panel (200×200 each → ~410 px total width)


# ── Background worker ──────────────────────────────────────────────────────────

class _Worker(QObject):
    """Runs in a QThread — reads TIF and applies wavelet sharpening."""

    done  = Signal(bytes, bytes, int, int, int, str)
    error = Signal(str)

    def __init__(
        self,
        tif_path: Path,
        amounts: List[float],
        levels: int,
        power: float,
        sharpen_filter: float,
        edge_feather_factor: float = 0.0,
        auto_params: bool = False,
        color_correct: bool = False,
        denoise_amounts: Optional[List[float]] = None,
        filter_type: str = 'gaussian',
    ) -> None:
        super().__init__()
        self._path                = tif_path
        self._amounts             = amounts
        self._levels              = levels
        self._power               = power
        self._sharpen_filter      = sharpen_filter
        self._edge_feather_factor = edge_feather_factor
        self._auto_params         = auto_params
        self._color_correct       = color_correct
        self._denoise_amounts     = denoise_amounts
        self._filter_type         = filter_type

    @Slot()
    def run(self) -> None:
        try:
            from pipeline.modules import image_io, wavelet

            orig = image_io.read_tif(self._path)

            working = orig
            if self._color_correct and orig.ndim == 3:
                from pipeline.steps.step06_rgb_composite import _auto_color_correct
                working, _ = _auto_color_correct(orig)

            if self._auto_params or self._edge_feather_factor > 0.0:
                from pipeline.modules.derotation import find_disk_center
                _lum = working.mean(axis=2) if working.ndim == 3 else working
                try:
                    _cx, _cy, _rx, _ry, _angle = find_disk_center(_lum)
                    _has_disk = _rx >= 5
                except Exception:
                    _has_disk = False

                if _has_disk:
                    _angle_rad = np.radians(_angle)
                    if self._auto_params:
                        _eff, _expand = wavelet.auto_wavelet_params(
                            _lum, _cx, _cy, _rx, _ry, _angle_rad
                        )
                    else:
                        _eff    = self._edge_feather_factor
                        _expand = 0.0
                    sharp = wavelet.sharpen_disk_aware(
                        working, _cx, _cy, _rx,
                        levels=self._levels,
                        amounts=self._amounts,
                        power=self._power,
                        sharpen_filter=self._sharpen_filter,
                        edge_feather_factor=_eff,
                        ry=_ry, angle=_angle_rad,
                        expand_px=_expand,
                        denoise_amounts=self._denoise_amounts,
                        filter_type=self._filter_type,
                    )
                else:
                    sharp = wavelet.sharpen(
                        working,
                        levels=self._levels,
                        amounts=self._amounts,
                        power=self._power,
                        sharpen_filter=self._sharpen_filter,
                        denoise_amounts=self._denoise_amounts,
                        filter_type=self._filter_type,
                    )
            else:
                sharp = wavelet.sharpen(
                    working,
                    levels=self._levels,
                    amounts=self._amounts,
                    power=self._power,
                    sharpen_filter=self._sharpen_filter,
                    denoise_amounts=self._denoise_amounts,
                    filter_type=self._filter_type,
                )

            orig_u8  = _to_uint8(orig)
            sharp_u8 = _to_uint8(sharp)

            if orig_u8.ndim == 2:
                h, w = orig_u8.shape
                ch = 1
            else:
                h, w, ch = orig_u8.shape

            self.done.emit(
                bytes(np.ascontiguousarray(orig_u8)),
                bytes(np.ascontiguousarray(sharp_u8)),
                h, w, ch,
                self._path.name,
            )
        except Exception as exc:
            self.error.emit(str(exc))


# ── Helpers ────────────────────────────────────────────────────────────────────

def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Percentile stretch [0.5%, 99.5%] → uint8."""
    lo = float(np.percentile(arr, 0.5))
    hi = float(np.percentile(arr, 99.5))
    if hi <= lo:
        hi = lo + 1e-6
    stretched = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (stretched * 255).astype(np.uint8)


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


def _pick_tif(folder: Path) -> Optional[Path]:
    # Search recursively — Step 5 output lives in window_01/, window_02/, …
    tifs = sorted(folder.rglob("*.tif")) + sorted(folder.rglob("*.TIF"))
    return tifs[0] if tifs else None


# ── Widget ─────────────────────────────────────────────────────────────────────

_PANEL_STYLE  = "QLabel { background: #1a1a1a; border: 1px solid #444; border-radius: 4px; }"
_CAP_STYLE    = "color: #888; font-size: 10px;"
_STATUS_STYLE = "color: #666; font-size: 10px; font-style: italic;"


class WaveletPreviewWidget(QWidget):
    """Side-by-side original / sharpened comparison panel.

    Parameters
    ----------
    sharpen_filter : float
        Step 3 uses 0.1 (WaveSharp default); Step 6 uses 0.0.
    """

    def __init__(
        self,
        sharpen_filter: float = 0.0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._sharpen_filter         = sharpen_filter
        self._input_dir: Optional[Path] = None
        self._amounts:   List[float]    = [200.0, 200.0, 200.0, 0.0, 0.0, 0.0]
        self._levels:    int            = 6
        self._power:     float          = 1.0
        self._edge_feather_factor: float = 0.0
        self._auto_params: bool          = False
        self._color_correct: bool        = False
        self._denoise_amounts: Optional[List[float]] = None
        self._filter_type: str           = 'gaussian'

        # State flags — more reliable than thread.isRunning()
        self._running = False
        self._pending = False

        # Keep strong refs to prevent premature GC
        self._thread: Optional[QThread]  = None
        self._worker: Optional[_Worker]  = None

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._do_update)

        self._build_ui()

    # ── UI ─────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 4, 0, 0)
        root.setSpacing(4)

        # Header
        self._header_lbl = QLabel(S("preview.label"))
        self._header_lbl.setStyleSheet("color: #aaa; font-size: 11px;")
        root.addWidget(self._header_lbl)

        # Status label (shows filename while rendering, or error/hint)
        self._status_lbl = QLabel(S("preview.status.tif"))
        self._status_lbl.setStyleSheet(_STATUS_STYLE)
        self._status_lbl.setWordWrap(True)
        root.addWidget(self._status_lbl)

        # Image panels (side by side)
        panels = QHBoxLayout()
        panels.setSpacing(8)

        self._orig_lbl  = _make_img_label()
        self._sharp_lbl = _make_img_label()

        self._cap_orig_lbl  = QLabel(S("preview.cap.original"))
        self._cap_sharp_lbl = QLabel(S("preview.cap.wavelet"))
        for img_lbl, cap_lbl in (
            (self._orig_lbl,  self._cap_orig_lbl),
            (self._sharp_lbl, self._cap_sharp_lbl),
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
        self._status_lbl.setText(S("preview.status.tif"))
        self._cap_orig_lbl.setText(S("preview.cap.original"))
        self._cap_sharp_lbl.setText(S("preview.cap.wavelet"))

    def set_input_dir(self, folder) -> None:
        """Set TIF source folder.  Accepts Path, str, or None."""
        if folder:
            p = Path(str(folder))
            self._input_dir = p if p.is_dir() else None
        else:
            self._input_dir = None

        if self._input_dir is None:
            self._status_lbl.setText(S("preview.status.tif"))
        elif self.isVisible():
            # Already on screen — render immediately
            self.schedule_update(100)

    def set_params(
        self,
        amounts: List[float],
        levels: int = 6,
        power: float = 1.0,
        edge_feather_factor: float = 0.0,
        auto_params: bool = False,
        denoise_amounts: Optional[List[float]] = None,
        filter_type: str = 'gaussian',
    ) -> None:
        self._amounts             = list(amounts)
        self._levels              = levels
        self._power               = power
        self._edge_feather_factor = edge_feather_factor
        self._auto_params         = auto_params
        self._denoise_amounts     = list(denoise_amounts) if denoise_amounts is not None else None
        self._filter_type         = filter_type

    def set_color_correct(self, enabled: bool) -> None:
        self._color_correct = enabled

    def schedule_update(self, delay: int = 400) -> None:
        """Debounced update — restarts timer on every call."""
        if self._input_dir is None:
            return
        self._timer.start(delay)

    def trigger_update(self) -> None:
        """Immediate update (skips debounce)."""
        self._timer.stop()
        self._do_update()

    # ── Qt events ───────────────────────────────────────────────────────────────

    def showEvent(self, event) -> None:
        """Auto-render when the panel first becomes visible."""
        super().showEvent(event)
        if self._input_dir is not None and not self._running:
            self.schedule_update(150)

    # ── Internal ────────────────────────────────────────────────────────────────

    def _do_update(self) -> None:
        if self._input_dir is None:
            return

        # If already running, mark as pending so we retry after done
        if self._running:
            self._pending = True
            return

        tif = _pick_tif(self._input_dir)
        if tif is None:
            self._status_lbl.setText(S("preview.no_tif", d=self._input_dir))
            return

        self._running = True
        self._pending = False
        self._status_lbl.setText(f"{S('preview.rendering')}  {tif.name}")

        worker = _Worker(
            tif,
            list(self._amounts),
            self._levels,
            self._power,
            self._sharpen_filter,
            self._edge_feather_factor,
            self._auto_params,
            self._color_correct,
            denoise_amounts=list(self._denoise_amounts) if self._denoise_amounts is not None else None,
            filter_type=self._filter_type,
        )
        # QThread(self): Qt parent keeps C++ object alive so Python ref-drops
        # in _on_done/_on_error are safe even if the OS thread is still quitting.
        thread = QThread(self)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.done.connect(self._on_done)
        worker.error.connect(self._on_error)
        worker.done.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(thread.deleteLater)

        # Keep strong Python refs so they're not GC'd while running
        self._thread = thread
        self._worker = worker
        thread.start()

    def _on_done(
        self,
        orig_bytes: bytes,
        sharp_bytes: bytes,
        h: int, w: int, ch: int,
        filename: str,
    ) -> None:
        self._running = False
        self._thread  = None
        self._worker  = None

        self._orig_lbl.setPixmap(_bytes_to_pixmap(orig_bytes, h, w, ch))
        self._sharp_lbl.setPixmap(_bytes_to_pixmap(sharp_bytes, h, w, ch))
        self._status_lbl.setText(filename)

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


# ── Widget helpers ─────────────────────────────────────────────────────────────

def _make_img_label() -> QLabel:
    lbl = QLabel()
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lbl.setStyleSheet(_PANEL_STYLE)
    lbl.setFixedSize(_PANEL_SIZE, _PANEL_SIZE)
    lbl.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    return lbl
