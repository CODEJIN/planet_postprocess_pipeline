"""Step 4 — De-rotation stacking panel."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
)

from gui.i18n import S
from gui.panels.base_panel import BasePanel

_SPINBOX_STYLE = (
    "QDoubleSpinBox { background: #3c3c3c; color: #d4d4d4; border: 1px solid #555;"
    " border-radius: 3px; padding: 3px 6px; }"
    "QDoubleSpinBox:focus { border-color: #4da6ff; }"
)
_CHECK_STYLE = (
    "QCheckBox { color: #d4d4d4; }"
    "QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #666;"
    " border-radius: 2px; background: #3c3c3c; }"
    "QCheckBox::indicator:checked { background: #4da6ff; border-color: #4da6ff; }"
    "QCheckBox::indicator:unchecked { background: #2a2a2a; border-color: #555; }"
)
_READONLY_STYLE = (
    "QLineEdit { background: #2a2a2a; color: #888; border: 1px solid #3a3a3a;"
    " border-radius: 3px; padding: 3px 6px; }"
)
_BTN_STYLE = (
    "QPushButton { background: #3c3c3c; color: #d4d4d4; border: 1px solid #555;"
    " border-radius: 3px; padding: 3px 10px; font-size: 11px; }"
    "QPushButton:hover { background: #4a4a4a; border-color: #4da6ff; }"
    "QPushButton:disabled { color: #555; border-color: #444; }"
)

# Filter priority for sweep: prefer high-contrast, wide-band filters first
_FILTER_PRIORITY = ["IR", "R", "G", "B", "CH4"]


# ── Background worker ──────────────────────────────────────────────────────────

class _WarpSweepWorker(QThread):
    """Sweep warp_scale values and find the sharpest de-rotated stack.

    Signals:
        finished(best_scale, confidence, message)
            confidence: "high" | "low" | "error"
    """
    finished = Signal(float, str, str)

    def __init__(
        self,
        quality_dir: Path,
        input_dir: Path,
        filters: list[str],
        period_hours: float,
    ) -> None:
        super().__init__()
        self._quality_dir  = quality_dir
        self._input_dir    = input_dir
        self._filters      = filters
        self._period_hours = period_hours

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_ts(s: str) -> datetime:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)

    def _pick_filter(self) -> str | None:
        """Return the first usable filter from priority list, then any available."""
        # Try user's filters in priority order
        for filt in _FILTER_PRIORITY:
            if filt in self._filters:
                csv_path = self._quality_dir / f"{filt}_ranking.csv"
                if csv_path.exists():
                    return filt
        # Fall back to any CSV present in the quality dir
        for csv_path in sorted(self._quality_dir.glob("*_ranking.csv")):
            return csv_path.stem.replace("_ranking", "")
        return None

    def _load_window(self, filt: str, win_min: int = 30) -> list[dict]:
        csv_path = self._quality_dir / f"{filt}_ranking.csv"
        rows: list[dict] = []
        with open(csv_path, newline="") as f:
            for r in csv.DictReader(f):
                # Try input_dir first, then quality_dir parent
                stem = r["stem"]
                tif = self._input_dir / f"{stem}.tif"
                if not tif.exists():
                    tif = self._quality_dir.parent / f"{stem}.tif"
                if tif.exists():
                    rows.append({
                        "path":       tif,
                        "timestamp":  self._parse_ts(r["timestamp"]),
                        "norm_score": float(r["norm_score"]),
                    })
        if not rows:
            return []
        rows.sort(key=lambda r: r["norm_score"], reverse=True)
        t_ref = rows[0]["timestamp"]
        return [r for r in rows
                if abs((r["timestamp"] - t_ref).total_seconds()) <= win_min * 60]

    @staticmethod
    def _lap_sharpness(img: np.ndarray, cx: float, cy: float, radius: float) -> float:
        import cv2
        lap = cv2.Laplacian(img.astype(np.float32), cv2.CV_32F, ksize=3)
        h, w = img.shape
        yy, xx = np.mgrid[0:h, 0:w]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < (radius * 0.85) ** 2
        vals = lap[mask]
        return float(np.var(vals)) if vals.size else 0.0

    # ── main ──────────────────────────────────────────────────────────────────

    def run(self) -> None:
        try:
            self._sweep()
        except Exception as exc:
            self.finished.emit(0.80, "error", f"오류: {exc}")

    def _sweep(self) -> None:
        from pipeline.modules import image_io
        from pipeline.modules.derotation import spherical_derotation_warp, find_disk_center

        # ── pick filter ───────────────────────────────────────────────────────
        filt = self._pick_filter()
        if filt is None:
            self.finished.emit(0.80, "error",
                               "Step 3 데이터가 없습니다. Step 3을 먼저 실행하세요.")
            return

        # ── load frames ───────────────────────────────────────────────────────
        window = self._load_window(filt)
        if len(window) < 2:
            self.finished.emit(0.80, "error",
                               f"{filt} 필터 프레임이 부족합니다 (최소 2개 필요).")
            return

        dt_secs = [(r["timestamp"] - window[0]["timestamp"]).total_seconds()
                   for r in window]

        imgs: list[np.ndarray] = []
        for r in window:
            img = image_io.read_tif(r["path"])
            if img.ndim == 3:
                img = img.mean(axis=2).astype(np.float32)
            imgs.append(img)

        ref_img = imgs[0]
        cx, cy, radius, _, _ = find_disk_center(ref_img)

        # ── sweep ─────────────────────────────────────────────────────────────
        scales = np.arange(0.0, 1.55, 0.1)
        sharpness_values: list[float] = []

        for scale in scales:
            warped = [
                spherical_derotation_warp(
                    img, dt, cx, cy, radius,
                    period_hours=self._period_hours,
                    scale=float(scale),
                    flip_direction=False,
                    pole_pa_deg=0.0,
                )
                for img, dt in zip(imgs, dt_secs)
            ]
            stack = np.mean(warped, axis=0).astype(np.float32)
            sharpness_values.append(self._lap_sharpness(stack, cx, cy, radius))

        # ── evaluate result ───────────────────────────────────────────────────
        best_idx   = int(np.argmax(sharpness_values))
        best_scale = float(scales[best_idx])
        base_sharp = sharpness_values[0]
        max_sharp  = sharpness_values[best_idx]

        if base_sharp > 1e-12:
            improvement_pct = (max_sharp - base_sharp) / base_sharp * 100.0
        else:
            improvement_pct = 0.0

        if improvement_pct < 3.0:
            confidence = "low"
            msg = (f"최적값 {best_scale:.2f} ({filt} 기준) — "
                   f"개선 {improvement_pct:.1f}% (시잉 불량, 차이 미미)")
        else:
            confidence = "high"
            msg = (f"최적값 {best_scale:.2f} ({filt} 기준) — "
                   f"스택 선명도 +{improvement_pct:.1f}%")

        self.finished.emit(best_scale, confidence, msg)


# ── Panel ──────────────────────────────────────────────────────────────────────

class Step04Panel(BasePanel):
    STEP_ID   = "04"
    TITLE_KEY = "step04.title"
    DESC_KEY  = "step04.desc"
    OPTIONAL  = False

    def __init__(self, parent: QWidget | None = None) -> None:
        self._output_dir:    Path | None = None
        self._input_dir:     Path | None = None
        self._filters:       list[str]   = []
        self._period_hours:  float       = 9.9281
        self._sweep_worker:  _WarpSweepWorker | None = None
        super().__init__(parent)

    # ── BasePanel interface ───────────────────────────────────────────────────

    def build_form(self) -> None:
        form_widget = QWidget()
        form_widget.setStyleSheet("background: transparent;")
        fl = QFormLayout(form_widget)
        fl.setSpacing(10)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        # Folder display (auto-derived, read-only)
        self._input_lbl = QLineEdit()
        self._input_lbl.setReadOnly(True)
        self._input_lbl.setStyleSheet(_READONLY_STYLE)
        lbl_in = QLabel(S("step04.input_dir"))
        lbl_in.setToolTip(
            "Step 2 Lucky Stacking 출력 TIF 폴더입니다.\n"
            "cascade로 자동 설정됩니다."
        )
        fl.addRow(lbl_in, self._input_lbl)

        self._quality_lbl = QLineEdit()
        self._quality_lbl.setReadOnly(True)
        self._quality_lbl.setStyleSheet(_READONLY_STYLE)
        lbl_q = QLabel(S("step04.quality_dir"))
        lbl_q.setToolTip(
            "Step 3가 생성한 품질 점수 CSV 파일이 있는 폴더입니다.\n"
            "Step 3 완료 후 자동으로 설정됩니다.\n"
            "이 폴더가 있어야 '최적값 자동탐색' 버튼이 활성화됩니다."
        )
        fl.addRow(lbl_q, self._quality_lbl)

        self._output_lbl = QLineEdit()
        self._output_lbl.setReadOnly(True)
        self._output_lbl.setStyleSheet(_READONLY_STYLE)
        lbl_out = QLabel(S("step04.output_dir"))
        lbl_out.setToolTip(
            "De-rotation 스태킹된 TIF 마스터 이미지가 저장될 폴더입니다.\n"
            "자동으로 설정됩니다."
        )
        fl.addRow(lbl_out, self._output_lbl)

        # Warp scale
        _tip_warp = (
            "De-rotation 워프의 강도 계수입니다.\n"
            "\n"
            "【원리】\n"
            "행성은 구체이므로 같은 자전각도라도 원반 중심부(지구 방향)\n"
            "는 많이 이동하고, 가장자리(시선 수직 방향)는 거의 안 움직입니다.\n"
            "이 깊이(depth)에 비례한 위치별 보정이 구면 워프입니다.\n"
            "  drift(x,y) = warp_scale × Δλ × depth(x,y)\n"
            "\n"
            "【값의 의미】\n"
            "0.0 = 보정 없음 (모든 픽셀에 동일한 이동, WinJUPOS 방식)\n"
            "1.0 = 이론적 완전 구체 보정\n"
            "0.8 = 실험 최적값 (목성, 일반 시잉 조건)\n"
            "\n"
            "【조절이 의미 있는 경우】\n"
            "· 시잉이 매우 좋은 날 → 1.0~1.2 시도\n"
            "· 합성 후 동·서 limb 근처에서 필터 간 색수차가 심하면\n"
            "  0.6~0.9 범위에서 줄여보세요\n"
            "· 시잉이 나쁜 날은 어떤 값이든 결과 차이가 거의 없습니다"
        )
        self._warp_scale = QDoubleSpinBox()
        self._warp_scale.setStyleSheet(_SPINBOX_STYLE)
        self._warp_scale.setRange(0.0, 2.0)
        self._warp_scale.setDecimals(2)
        self._warp_scale.setSingleStep(0.01)
        self._warp_scale.setValue(0.80)
        self._warp_scale.setToolTip(_tip_warp)
        lbl_warp = QLabel(S("step04.warp_scale"))
        lbl_warp.setToolTip(_tip_warp)
        fl.addRow(lbl_warp, self._warp_scale)

        # ── Auto-sweep button row ─────────────────────────────────────────────
        sweep_widget = QWidget()
        sweep_widget.setStyleSheet("background: transparent;")
        sweep_layout = QHBoxLayout(sweep_widget)
        sweep_layout.setContentsMargins(0, 0, 0, 0)
        sweep_layout.setSpacing(8)

        self._sweep_btn = QPushButton(S("step04.sweep_btn"))
        self._sweep_btn.setStyleSheet(_BTN_STYLE)
        self._sweep_btn.setFixedWidth(130)
        self._sweep_btn.setToolTip(
            "현재 Step 3 데이터를 바탕으로 스택 선명도가 최대가 되는\n"
            "warp_scale 값을 자동으로 탐색합니다.\n"
            "약 2~4초 소요. Step 3이 완료되어 있어야 합니다."
        )
        self._sweep_btn.clicked.connect(self._on_sweep_clicked)

        self._sweep_result = QLabel("")
        self._sweep_result.setStyleSheet("QLabel { color: #888; font-size: 11px; }")
        self._sweep_result.setWordWrap(True)

        sweep_layout.addWidget(self._sweep_btn)
        sweep_layout.addWidget(self._sweep_result, 1)

        fl.addRow("", sweep_widget)

        # Min quality threshold
        _tip_mq = (
            "이 값 이하의 품질 점수를 가진 프레임은 스태킹에서 제외됩니다.\n"
            "0.0 = 모든 프레임 포함. 나쁜 날씨 시 0.3~0.5로 올려보세요."
        )
        self._min_quality = QDoubleSpinBox()
        self._min_quality.setStyleSheet(_SPINBOX_STYLE)
        self._min_quality.setRange(0.0, 1.0)
        self._min_quality.setDecimals(2)
        self._min_quality.setSingleStep(0.05)
        self._min_quality.setValue(0.05)
        self._min_quality.setToolTip(_tip_mq)
        lbl_mq = QLabel(S("step04.min_quality"))
        lbl_mq.setToolTip(_tip_mq)
        fl.addRow(lbl_mq, self._min_quality)

        # Normalize brightness
        _tip_norm = (
            "스태킹 전 각 프레임의 밝기를 정규화합니다.\n"
            "시잉 변화로 인한 밝기 차이가 큰 경우 활성화하세요."
        )
        self._normalize = QCheckBox()
        self._normalize.setStyleSheet(_CHECK_STYLE)
        self._normalize.setChecked(False)
        self._normalize.setToolTip(_tip_norm)
        lbl_norm = QLabel(S("step04.normalize"))
        lbl_norm.setToolTip(_tip_norm)
        fl.addRow(lbl_norm, self._normalize)

        idx = self._form_layout.count() - 1
        self._form_layout.insertWidget(idx, form_widget)

    def get_config_updates(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "warp_scale":            self._warp_scale.value(),
            "min_quality_threshold": self._min_quality.value(),
            "normalize_brightness":  self._normalize.isChecked(),
        }
        out_text = self._output_lbl.text().strip()
        if out_text:
            result["output_dir"] = str(Path(out_text).parent)
        return result

    def load_session(self, data: dict[str, Any]) -> None:
        inp = data.get("input_dir", "")
        out = data.get("output_dir", "")
        if inp:
            self._input_lbl.setText(inp)
            self._input_dir = Path(inp)
        if out:
            p = Path(out)
            if hasattr(self, "_quality_lbl"):
                self._quality_lbl.setText(str(p / "step03_quality"))
            self._output_lbl.setText(str(p / "step04_derotated"))
            self._output_dir = p

        # Store for sweep worker
        filters_str = data.get("filters", "IR,R,G,B,CH4")
        self._filters = [f.strip() for f in filters_str.split(",") if f.strip()]
        self._period_hours = float(data.get("rotation_period", 9.9281))

        self._warp_scale.setValue(float(data.get("warp_scale", 0.80)))
        self._min_quality.setValue(float(data.get("min_quality_threshold", 0.05)))
        self._normalize.setChecked(bool(data.get("normalize_brightness", False)))

        self._update_sweep_btn_state()

    def output_paths(self) -> list[Path]:
        if self._output_dir is None:
            return []
        step_dir = self._output_dir / "step04_derotated"
        if not step_dir.exists():
            return []
        paths = sorted(step_dir.rglob("*.tif"))
        return paths[:8]

    def set_output_dir(self, path: Path | str) -> None:
        self._output_dir = Path(path) if path else None

    # ── Sweep helpers ─────────────────────────────────────────────────────────

    def _quality_dir(self) -> Path | None:
        if self._output_dir is None:
            return None
        q = self._output_dir / "step03_quality"
        return q if q.exists() else None

    def _update_sweep_btn_state(self) -> None:
        if not hasattr(self, "_sweep_btn"):
            return
        enabled = self._quality_dir() is not None
        self._sweep_btn.setEnabled(enabled)
        if not enabled:
            self._sweep_result.setStyleSheet("QLabel { color: #666; font-size: 11px; }")
            self._sweep_result.setText(S("step04.sweep_wait"))

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_sweep_clicked(self) -> None:
        qdir = self._quality_dir()
        if qdir is None or self._input_dir is None:
            return

        self._sweep_btn.setEnabled(False)
        self._sweep_btn.setText(S("step04.sweeping"))
        self._sweep_result.setStyleSheet("QLabel { color: #888; font-size: 11px; }")
        self._sweep_result.setText("")

        self._sweep_worker = _WarpSweepWorker(
            quality_dir=qdir,
            input_dir=self._input_dir,
            filters=self._filters,
            period_hours=self._period_hours,
        )
        self._sweep_worker.finished.connect(self._on_sweep_finished)
        self._sweep_worker.start()

    def _on_sweep_finished(self, best_scale: float, confidence: str, msg: str) -> None:
        self._sweep_btn.setEnabled(True)
        self._sweep_btn.setText(S("step04.sweep_btn"))

        if confidence == "error":
            color = "#e06c75"   # red
        elif confidence == "low":
            color = "#e5c07b"   # orange/yellow
            self._warp_scale.setValue(best_scale)
        else:
            color = "#98c379"   # green
            self._warp_scale.setValue(best_scale)

        self._sweep_result.setStyleSheet(
            f"QLabel {{ color: {color}; font-size: 11px; }}"
        )
        self._sweep_result.setText(msg)
