"""Lucky Stacking AP-grid preview widget — used by Step 2.

Reads a middle frame from one SER file found in ``input_dir``, detects the
planet disk via ellipse fitting, generates the AP alignment-point grid with
the current ``ap_size`` / ``ap_step`` settings, and renders an overlay:

  - Cyan ellipse  : detected disk boundary
  - Green dots    : accepted AP centres (brightness + contrast pass)
  - Blue rect     : one example AP patch (shows patch size at scale)
  - Status label  : AP count, disk radius, SER filename

Auto-renders on first show; re-renders (debounced 500 ms) when params change.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QObject, QThread, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from gui.i18n import S

_PANEL_SIZE   = 280
_PANEL_STYLE  = "QLabel { background: #1a1a1a; border: 1px solid #444; border-radius: 4px; }"
_STATUS_STYLE = "color: #666; font-size: 10px; font-style: italic;"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _pick_ser(folder: Path) -> Optional[Path]:
    """Return the first SER file found (recursive)."""
    if not folder.is_dir():
        return None
    sers = sorted(folder.rglob("*.ser")) + sorted(folder.rglob("*.SER"))
    return sers[0] if sers else None


def _to_gray_f32(frame: np.ndarray) -> np.ndarray:
    """Convert raw frame to float32 grayscale [0, 1]."""
    if frame.ndim == 3:
        gray = frame.mean(axis=2)
    else:
        gray = frame.astype(np.float64)
    hi = gray.max()
    return (gray / hi).astype(np.float32) if hi > 0 else gray.astype(np.float32)


def _to_rgb8(frame: np.ndarray) -> np.ndarray:
    """Convert raw frame (any dtype, mono or RGB) to uint8 RGB for display."""
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


def _fit_to(img: np.ndarray, max_px: int) -> Tuple[np.ndarray, float]:
    """Downscale img so max(h, w) <= max_px. Returns (scaled_img, scale_factor)."""
    h, w = img.shape[:2]
    scale = min(max_px / max(h, w), 1.0)
    if scale < 1.0:
        img = cv2.resize(
            img,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return img, scale


def _to_pixmap(data: bytes, h: int, w: int) -> QPixmap:
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


# ── Background worker ──────────────────────────────────────────────────────────

class _Worker(QObject):
    done  = Signal(bytes, int, int, str)   # img_bytes, h, w, status_text
    error = Signal(str)

    def __init__(
        self,
        ser_path: Path,
        ap_size: int,
        ap_step: int,
        ap_min_brightness: float,
        ap_min_contrast: float,
    ) -> None:
        super().__init__()
        self._path      = ser_path
        self._ap_size   = ap_size
        self._ap_step   = ap_step
        self._min_bright = ap_min_brightness
        self._min_cont  = ap_min_contrast

    @Slot()
    def run(self) -> None:
        try:
            from collections import Counter
            from pipeline.modules import ser_io
            from pipeline.modules.derotation import find_disk_center
            from pipeline.modules.lucky_stack import generate_adaptive_ap_grid, LuckyStackConfig

            with ser_io.SERReader(self._path) as reader:
                total   = int(reader.header["FrameCount"])
                mid_idx = total // 2
                frame   = reader.get_frame_rgb(mid_idx)

            # Display version (uint8 RGB)
            disp = _to_rgb8(frame)

            # Analysis version (float32 grayscale [0,1])
            gray = _to_gray_f32(frame)
            H, W = gray.shape[:2]

            # Detect disk → (cx, cy, semi_major, semi_minor, angle_deg)
            cx, cy, semi_a, semi_b, angle_deg = find_disk_center(gray)
            disk_radius = semi_a  # semi_major is the larger (equatorial) axis

            # Generate adaptive AP grid (try14+: LoG energy + cross-size NMS)
            # ap_size is the *minimum* AP size; actual sizes are auto-selected per location.
            cfg = LuckyStackConfig(
                ap_size           = self._ap_size,
                ap_min_brightness = self._min_bright,
                ap_min_contrast   = self._min_cont,
                use_adaptive_ap   = True,
            )
            aps = generate_adaptive_ap_grid(cx, cy, disk_radius, gray, cfg)
            # aps: List[Tuple[int, int, int]] → (ax, ay, ap_size)

            # ── Draw overlay ──────────────────────────────────────────────────
            overlay, scale = _fit_to(disp.copy(), _PANEL_SIZE)
            oh, ow = overlay.shape[:2]

            sx, sy = ow / W, oh / H
            s_avg  = (sx + sy) / 2

            # Disk ellipse (cyan)
            cv2.ellipse(
                overlay,
                (int(cx * sx), int(cy * sy)),
                (max(1, int(semi_a * s_avg)), max(1, int(semi_b * s_avg))),
                angle_deg,
                0, 360,
                (0, 210, 255), 2,
            )

            # Find the AP closest to disk centre to draw as a sample patch box
            sample_ap = None
            if aps:
                dists = [(ax - cx)**2 + (ay - cy)**2 for ax, ay, _ in aps]
                sample_ap = aps[int(np.argmin(dists))]

            # Example AP patch rectangle (blue, shows actual ap_size of that AP)
            if sample_ap is not None:
                sax, say, sap_sz = int(sample_ap[0] * sx), int(sample_ap[1] * sy), sample_ap[2]
                half_px = max(1, int((sap_sz / 2) * s_avg))
                cv2.rectangle(
                    overlay,
                    (sax - half_px, say - half_px),
                    (sax + half_px, say + half_px),
                    (80, 140, 255), 1,
                )

            # AP circles — outline radius = ap_size/2, color: green(small) → yellow(large)
            all_sizes = sorted(set(sz for _, _, sz in aps)) if aps else [self._ap_size]
            sz_min, sz_max = all_sizes[0], all_sizes[-1]
            for ax, ay, ap_sz in aps:
                t = (ap_sz - sz_min) / (sz_max - sz_min) if sz_max > sz_min else 0.0
                color = (int(60 + t * 180), int(240 - t * 20), int(100 - t * 40))
                px_x, px_y = int(ax * sx), int(ay * sy)
                circle_r = max(4, int((ap_sz / 2) * s_avg))
                cv2.circle(overlay, (px_x, px_y), circle_r, color, 1)   # patch extent
                cv2.circle(overlay, (px_x, px_y), 2, color, -1)          # center dot

            # Labels
            font  = cv2.FONT_HERSHEY_SIMPLEX
            fsc   = 0.38
            thick = 1
            cv2.putText(overlay, "AP (adaptive)", (4, 14), font, fsc, (60, 240, 100), thick, cv2.LINE_AA)
            cv2.putText(overlay, "Disk",          (4, 26), font, fsc, (0, 210, 255),  thick, cv2.LINE_AA)

            size_counts = Counter(sz for _, _, sz in aps)
            size_str = " ".join(f"{sz}px×{cnt}" for sz, cnt in sorted(size_counts.items()))
            status = (
                f"AP: {len(aps)}개  [{size_str}] · 반경: {disk_radius:.0f}px · "
                f"프레임 {mid_idx + 1}/{total} · {self._path.name}"
            )

            oh, ow = overlay.shape[:2]
            self.done.emit(bytes(np.ascontiguousarray(overlay)), oh, ow, status)

        except Exception as exc:
            self.error.emit(str(exc))


# ── Widget ─────────────────────────────────────────────────────────────────────

class LuckyStackPreviewWidget(QWidget):
    """Single-panel preview: SER frame with disk outline and AP grid overlay."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._input_dir: Optional[Path] = None
        self._ap_size:          int   = 64
        self._ap_step:          int   = 16
        self._ap_min_brightness: float = 0.196
        self._ap_min_contrast:   float = 0.01

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

        self._header_lbl = QLabel("AP 그리드 미리보기")
        self._header_lbl.setStyleSheet("color: #aaa; font-size: 11px;")
        root.addWidget(self._header_lbl)

        self._status_lbl = QLabel("SER 입력 폴더를 설정하면 미리보기가 활성화됩니다.")
        self._status_lbl.setStyleSheet(_STATUS_STYLE)
        self._status_lbl.setWordWrap(True)
        root.addWidget(self._status_lbl)

        self._img_lbl = QLabel()
        self._img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img_lbl.setStyleSheet(_PANEL_STYLE)
        self._img_lbl.setFixedSize(_PANEL_SIZE, _PANEL_SIZE)
        self._img_lbl.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        root.addWidget(self._img_lbl)

        self._legend_lbl = QLabel(
            '<span style="color:#3cf064">○</span> AP 크기 (소→대: 녹→황)'
            '　<span style="color:#50a0ff">■</span> 중앙 AP 패치'
            '　<span style="color:#00d2ff">○</span> 디스크 경계'
        )
        self._legend_lbl.setStyleSheet("font-size: 9px; color: #888;")
        root.addWidget(self._legend_lbl)
        root.addStretch()

    # ── Public API ──────────────────────────────────────────────────────────────

    def retranslate(self) -> None:
        self._header_lbl.setText("AP 그리드 미리보기")

    def set_input_dir(self, folder) -> None:
        if folder:
            self._input_dir = Path(str(folder))
        else:
            self._input_dir = None

        if self._input_dir is None:
            self._status_lbl.setText("SER 입력 폴더를 설정하면 미리보기가 활성화됩니다.")
        elif self.isVisible():
            self.schedule_update(150)

    def set_params(
        self,
        ap_size: int,
        ap_step: int = 16,
        ap_min_brightness: float = 0.196,
        ap_min_contrast: float = 0.01,
    ) -> None:
        changed = (
            self._ap_size           != ap_size
            or self._ap_step        != ap_step
            or self._ap_min_brightness != ap_min_brightness
            or self._ap_min_contrast   != ap_min_contrast
        )
        self._ap_size           = ap_size
        self._ap_step           = ap_step
        self._ap_min_brightness = ap_min_brightness
        self._ap_min_contrast   = ap_min_contrast
        if changed:
            self.schedule_update()

    def schedule_update(self, delay: int = 500) -> None:
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
            msg = (
                "SER 입력 폴더를 설정하면 미리보기가 활성화됩니다."
                if not self._input_dir.is_dir()
                else f"SER 파일 없음: {self._input_dir}"
            )
            self._status_lbl.setText(msg)
            return

        self._running = True
        self._pending = False
        self._status_lbl.setText(f"렌더링 중…  {ser.name}")

        worker = _Worker(
            ser,
            self._ap_size,
            self._ap_step,
            self._ap_min_brightness,
            self._ap_min_contrast,
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

    def _on_done(self, img_b: bytes, h: int, w: int, status: str) -> None:
        self._running = False
        self._thread  = None
        self._worker  = None

        self._img_lbl.setPixmap(_to_pixmap(img_b, h, w))
        self._status_lbl.setText(status)

        if self._pending:
            self._pending = False
            self.schedule_update(300)

    def _on_error(self, msg: str) -> None:
        self._running = False
        self._thread  = None
        self._worker  = None
        self._status_lbl.setText(f"오류: {msg}")
        if self._pending:
            self._pending = False
