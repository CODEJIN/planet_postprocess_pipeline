"""SER frame preview widget — used by Step 1 (SER Crop).

Loads a middle frame from one SER file found in ``input_dir``, runs
planet detection, and shows:
  - Left : raw frame scaled to fit, with detected-planet bounding box
            (cyan) and ROI crop area (green) overlaid.
  - Right: ROI-cropped result (SER Crop output for this frame).

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

def _pick_ser(folder: Path) -> Optional[Path]:
    """Return the first SER file found (recursive)."""
    if not folder.is_dir():
        return None
    sers = sorted(folder.rglob("*.ser")) + sorted(folder.rglob("*.SER"))
    return sers[0] if sers else None


def _to_rgb8(frame: np.ndarray) -> np.ndarray:
    """Convert raw frame (any dtype, mono or RGB) to uint8 RGB."""
    if frame.dtype == np.uint16:
        lo, hi = int(frame.min()), int(frame.max())
        if hi > lo:
            frame = ((frame.astype(np.float32) - lo) / (hi - lo) * 255).astype(np.uint8)
        else:
            frame = np.zeros_like(frame, dtype=np.uint8)
    elif frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    return frame


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
    """RGB bytes → QPixmap, scaled to fit _PANEL_SIZE if needed."""
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


# ── Background worker ──────────────────────────────────────────────────────────

class _Worker(QObject):
    # raw_bytes, raw_h, raw_w,  crop_bytes, crop_h, crop_w,  status_text
    done  = Signal(bytes, int, int, bytes, int, int, str)
    error = Signal(str)

    def __init__(self, ser_path: Path, roi_size: int, min_diameter: int) -> None:
        super().__init__()
        self._path     = ser_path
        self._roi_size = roi_size
        self._min_diam = min_diameter

    @Slot()
    def run(self) -> None:
        try:
            from pipeline.modules import planet_detect, ser_io

            with ser_io.SERReader(self._path) as reader:
                total   = int(reader.header["FrameCount"])
                mid_idx = total // 2
                frame   = reader.get_frame_rgb(mid_idx)   # uint8 or uint16, mono or RGB

            disp    = _to_rgb8(frame)          # uint8 RGB, full resolution
            overlay = disp.copy()
            h, w    = overlay.shape[:2]

            stem = self._path.stem
            # CH4 (methane-band) Saturn frames are darker on the disk than the
            # rings, so the usual brightness-based disk/ring shape check can't
            # apply here — skip it, same as pipeline.steps.ser_crop.
            skip_shape_check = "-CH4-" in stem or "_CH4_" in stem
            result = planet_detect.analyze_planet(
                frame, min_diameter=self._min_diam, skip_shape_check=skip_shape_check
            )

            # Coordinate bookmarks for text labels (populated below)
            bbox_pt: tuple[int, int] | None = None   # top-left of planet bbox
            roi_pt:  tuple[int, int] | None = None   # top-left of ROI box
            roi_color = (80, 255, 120)               # default: green (detected)

            if result:
                cx, cy   = result["centroid"]
                pw, ph   = result["width"], result["height"]
                diameter = max(pw, ph)
                passed   = diameter >= self._min_diam

                # Cyan rectangle — detected planet bounding box
                bx, by = int(cx - pw / 2), int(cy - ph / 2)
                cv2.rectangle(overlay, (bx, by), (bx + pw, by + ph), (0, 220, 255), 2)
                bbox_pt = (bx, by)

                # Green rectangle — ROI crop area
                half = self._roi_size // 2
                rx1, ry1 = int(cx) - half, int(cy) - half
                cv2.rectangle(
                    overlay,
                    (rx1, ry1),
                    (rx1 + self._roi_size, ry1 + self._roi_size),
                    (80, 255, 120), 2,
                )
                roi_pt = (rx1, ry1)

                verdict = S("preview.detect.pass") if passed else S("preview.detect.fail", min=self._min_diam)
                status  = S("preview.detect.found", d=diameter, verdict=verdict, i=mid_idx + 1, t=total)
                crop_raw = planet_detect.get_cropped_frame(frame, (cx, cy), self._roi_size)
            else:
                # Not detected — draw ROI box at frame centre (orange)
                cx, cy   = w / 2.0, h / 2.0
                half     = self._roi_size // 2
                rx1, ry1 = int(cx) - half, int(cy) - half
                cv2.rectangle(
                    overlay,
                    (rx1, ry1),
                    (rx1 + self._roi_size, ry1 + self._roi_size),
                    (255, 180, 0), 2,
                )
                roi_pt    = (rx1, ry1)
                roi_color = (255, 180, 0)
                status    = S("preview.detect.none", i=mid_idx + 1, t=total)
                crop_raw  = planet_detect.get_cropped_frame(frame, (cx, cy), self._roi_size)

            # Scale overlay to fit panel; draw text AFTER scaling so font size is consistent
            overlay_fit = _fit(overlay, _PANEL_SIZE)
            oh, ow      = overlay_fit.shape[:2]
            sx, sy      = ow / w, oh / h

            font  = cv2.FONT_HERSHEY_SIMPLEX
            fsc   = 0.42
            thick = 1

            if bbox_pt:
                # "Planet" label just above the cyan detection box
                tx = max(0, int(bbox_pt[0] * sx) + 2)
                ty = max(12, int(bbox_pt[1] * sy) - 3)
                cv2.putText(overlay_fit, "Planet", (tx, ty),
                            font, fsc, (0, 220, 255), thick, cv2.LINE_AA)

            if roi_pt:
                # "ROI" label inside the top-left corner of the ROI box
                tx = int(roi_pt[0] * sx) + 4
                ty = int(roi_pt[1] * sy) + 14
                cv2.putText(overlay_fit, "ROI", (tx, ty),
                            font, fsc, roi_color, thick, cv2.LINE_AA)

            crop_disp = _to_rgb8(crop_raw)

            oh, ow  = overlay_fit.shape[:2]
            crh, crw = crop_disp.shape[:2]

            self.done.emit(
                bytes(np.ascontiguousarray(overlay_fit)), oh, ow,
                bytes(np.ascontiguousarray(crop_disp)),   crh, crw,
                status,
            )
        except Exception as exc:
            self.error.emit(str(exc))


# ── Widget ─────────────────────────────────────────────────────────────────────

class SerPreviewWidget(QWidget):
    """Raw frame with detection overlay | ROI crop result."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._input_dir: Optional[Path] = None
        self._roi_size:  int = 448
        self._min_diam:  int = 50

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

        self._status_lbl = QLabel(S("preview.status.ser"))
        self._status_lbl.setStyleSheet(_STATUS_STYLE)
        self._status_lbl.setWordWrap(True)
        root.addWidget(self._status_lbl)

        panels = QHBoxLayout()
        panels.setSpacing(8)

        self._raw_lbl  = _make_img_label()
        self._crop_lbl = _make_img_label()

        self._cap_raw_lbl  = QLabel(S("preview.cap.raw_detect"))
        self._cap_crop_lbl = QLabel(S("preview.cap.roi_crop"))
        for img_lbl, cap_lbl in (
            (self._raw_lbl,  self._cap_raw_lbl),
            (self._crop_lbl, self._cap_crop_lbl),
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
        self._status_lbl.setText(S("preview.status.ser"))
        self._cap_raw_lbl.setText(S("preview.cap.raw_detect"))
        self._cap_crop_lbl.setText(S("preview.cap.roi_crop"))

    def set_input_dir(self, folder) -> None:
        if folder:
            self._input_dir = Path(str(folder))
        else:
            self._input_dir = None

        if self._input_dir is None:
            self._status_lbl.setText(S("preview.status.ser"))
        elif self.isVisible():
            self.schedule_update(100)

    def set_params(self, roi_size: int, min_diameter: int) -> None:
        self._roi_size = roi_size
        self._min_diam = min_diameter

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

        ser = _pick_ser(self._input_dir)
        if ser is None:
            if not self._input_dir.is_dir():
                self._status_lbl.setText(S("preview.status.ser"))
            else:
                self._status_lbl.setText(S("preview.no_ser", d=self._input_dir))
            return

        self._running = True
        self._pending = False
        self._status_lbl.setText(f"{S('preview.rendering')}  {ser.name}")

        worker = _Worker(ser, self._roi_size, self._min_diam)
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
        raw_b: bytes,  rh: int, rw: int,
        crop_b: bytes, crh: int, crw: int,
        status: str,
    ) -> None:
        self._running = False
        self._thread  = None
        self._worker  = None

        self._raw_lbl.setPixmap(_to_pixmap(raw_b, rh, rw))
        self._crop_lbl.setPixmap(_to_pixmap(crop_b, crh, crw))
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
