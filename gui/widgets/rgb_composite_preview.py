"""RGB composite preview widget — used by Step 7.

Loads filter master PNGs from ``step05_wavelet_master`` (first window found),
applies the first composite spec without channel alignment (fast preview), and
shows:
  - Left : reference channel (L if LRGB, else R) as grayscale
  - Right: RGB / LRGB composite result

Auto-renders on first show; re-renders (debounced 400 ms) when params change.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from gui.i18n import S

import cv2
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


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fit(img: np.ndarray, max_px: int) -> np.ndarray:
    """Downscale img so max(h, w) <= max_px.  Never upscales."""
    h, w = img.shape[:2]
    scale = min(max_px / max(h, w), 1.0)
    if scale < 1.0:
        img = cv2.resize(
            img,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return img


def _to_pixmap(data: bytes, h: int, w: int) -> QPixmap:
    """RGB bytes → QPixmap."""
    arr  = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)
    qimg = QImage(arr.data, w, h, w * 3, QImage.Format.Format_RGB888)
    px   = QPixmap.fromImage(qimg.copy())
    if max(w, h) > _PANEL_SIZE:
        px = px.scaled(
            _PANEL_SIZE, _PANEL_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
    return px


def _make_img_label() -> QLabel:
    lbl = QLabel()
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lbl.setStyleSheet(_PANEL_STYLE)
    lbl.setFixedSize(_PANEL_SIZE, _PANEL_SIZE)
    lbl.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    return lbl


def _pick_first_window(step06_dir: Path) -> Optional[Path]:
    """Return the first window sub-directory in step05_wavelet_master/."""
    if not step06_dir.is_dir():
        return None
    dirs = sorted(p for p in step06_dir.iterdir() if p.is_dir())
    return dirs[0] if dirs else None


# ── Background worker ──────────────────────────────────────────────────────────

class _Worker(QObject):
    # left_bytes, lh, lw,  right_bytes, rh, rw,  status_text
    done  = Signal(bytes, int, int, bytes, int, int, str)
    error = Signal(str)

    def __init__(
        self,
        step06_dir: Path,
        r_filt: str,
        g_filt: str,
        b_filt: str,
        l_filt: str,
    ) -> None:
        super().__init__()
        self._step06_dir = step06_dir
        self._r = r_filt
        self._g = g_filt
        self._b = b_filt
        self._l = l_filt or None

    @Slot()
    def run(self) -> None:
        try:
            from pipeline.modules import composite as comp_mod, image_io  # comp_mod: make_rgb/make_lrgb

            win_dir = _pick_first_window(self._step06_dir)
            if win_dir is None:
                msg = (
                    "윈도우 없음: Step 5를 먼저 실행해 주세요."
                    if self._step06_dir.is_dir()
                    else "Step 5 출력 폴더를 찾을 수 없습니다."
                )
                self.error.emit(msg)
                return

            def _load(filt: str) -> np.ndarray:
                p = win_dir / f"{filt}_master.png"
                if not p.exists():
                    raise FileNotFoundError(f"{filt}_master.png 없음 ({win_dir.name})")
                img = image_io.read_png(p)          # float32 [0,1], (H,W) or (H,W,3)
                if img.ndim == 3:
                    img = img.mean(axis=2).astype(np.float32)
                return img

            r_img = _load(self._r)
            g_img = _load(self._g)
            b_img = _load(self._b)
            l_img = _load(self._l) if self._l else None

            # No extra stretch — use raw float [0,1] values from step 5 PNGs directly.
            # Step 5 masters are stored as 16-bit PNGs (float * 65535 on write,
            # /65535 on read), so their pixel values already reflect the true
            # sensor output.  Adding auto_stretch on top would be a second stretch
            # that makes the preview artificially brighter than the actual step 6
            # output that step 10 reads.  Matching step 10's _to_uint8_levels
            # approach: just multiply by 255, no additional normalisation.
            l_s = l_img   # may be None

            # Composite
            if l_s is not None:
                right_f   = comp_mod.make_lrgb(l_s, r_img, g_img, b_img)
                left_f    = l_s
                spec_desc = f"LRGB  (L={self._l}, R={self._r}, G={self._g}, B={self._b})"
            else:
                right_f   = comp_mod.make_rgb(r_img, g_img, b_img)
                left_f    = r_img
                spec_desc = f"RGB  (R={self._r}, G={self._g}, B={self._b})"

            # _to_uint8_levels: direct [0,1] → uint8, identical to step 10 rendering
            left_u8  = (left_f * 255).clip(0, 255).astype(np.uint8)
            left_rgb = cv2.cvtColor(left_u8, cv2.COLOR_GRAY2RGB)

            right_u8  = (right_f * 255).clip(0, 255).astype(np.uint8)
            right_rgb = np.ascontiguousarray(right_u8[:, :, :3])

            # Scale to panel size
            left_fit  = _fit(left_rgb,  _PANEL_SIZE)
            right_fit = _fit(right_rgb, _PANEL_SIZE)

            lh, lw = left_fit.shape[:2]
            rh, rw = right_fit.shape[:2]

            status = f"{win_dir.name}  •  {spec_desc}"

            self.done.emit(
                bytes(np.ascontiguousarray(left_fit)), lh, lw,
                bytes(np.ascontiguousarray(right_fit)), rh, rw,
                status,
            )
        except Exception as exc:
            self.error.emit(str(exc))


# ── Widget ─────────────────────────────────────────────────────────────────────

class RgbCompositePreviewWidget(QWidget):
    """Input channel (grayscale) | RGB/LRGB composite result."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._step06_dir: Optional[Path] = None
        self._r_filt: str = "R"
        self._g_filt: str = "G"
        self._b_filt: str = "B"
        self._l_filt: str = ""

        self._running = False
        self._pending = False
        self._thread: Optional[QThread] = None
        self._worker: Optional[_Worker] = None

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._do_update)

        self._build_ui()

    # ── UI ──────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 4, 0, 0)
        root.setSpacing(4)

        self._header_lbl = QLabel(S("preview.label"))
        self._header_lbl.setStyleSheet("color: #aaa; font-size: 11px;")
        root.addWidget(self._header_lbl)

        self._status_lbl = QLabel(S("preview.status.step6"))
        self._status_lbl.setStyleSheet(_STATUS_STYLE)
        self._status_lbl.setWordWrap(True)
        root.addWidget(self._status_lbl)

        panels = QHBoxLayout()
        panels.setSpacing(8)

        self._left_lbl  = _make_img_label()
        self._right_lbl = _make_img_label()

        self._cap_left_lbl  = QLabel(S("preview.cap.input_ch"))
        self._cap_right_lbl = QLabel(S("preview.cap.composite"))
        for img_lbl, cap_lbl in (
            (self._left_lbl,  self._cap_left_lbl),
            (self._right_lbl, self._cap_right_lbl),
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
        self._status_lbl.setText(S("preview.status.step6"))
        self._cap_left_lbl.setText(S("preview.cap.input_ch"))
        self._cap_right_lbl.setText(S("preview.cap.composite"))

    def set_input_dir(self, folder) -> None:
        if folder:
            self._step06_dir = Path(str(folder))
        else:
            self._step06_dir = None

        if self._step06_dir is None:
            self._status_lbl.setText(S("preview.status.step6"))
        elif self.isVisible():
            self.schedule_update(100)

    def set_spec(
        self,
        r_filter: str,
        g_filter: str,
        b_filter: str,
        l_filter: str = "",
    ) -> None:
        self._r_filt = r_filter
        self._g_filt = g_filter
        self._b_filt = b_filter
        self._l_filt = l_filter

    def schedule_update(self, delay: int = 400) -> None:
        if self._step06_dir is None:
            return
        self._timer.start(delay)

    # ── Qt events ───────────────────────────────────────────────────────────────

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._step06_dir is not None and not self._running:
            self.schedule_update(150)

    # ── Internal ────────────────────────────────────────────────────────────────

    def _do_update(self) -> None:
        if self._step06_dir is None:
            return
        if self._running:
            self._pending = True
            return

        self._running = True
        self._pending = False
        self._status_lbl.setText(S("preview.rendering"))

        worker = _Worker(
            self._step06_dir,
            self._r_filt,
            self._g_filt,
            self._b_filt,
            self._l_filt,
        )
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

    def _on_done(
        self,
        left_b: bytes,  lh: int, lw: int,
        right_b: bytes, rh: int, rw: int,
        status: str,
    ) -> None:
        self._running = False
        self._thread  = None
        self._worker  = None

        self._left_lbl.setPixmap(_to_pixmap(left_b, lh, lw))
        self._right_lbl.setPixmap(_to_pixmap(right_b, rh, rw))
        self._status_lbl.setText(status)

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
