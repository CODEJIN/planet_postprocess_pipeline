"""Step 10 — Summary grid panel."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from gui.i18n import S
from gui.panels.base_panel import BasePanel
from gui.widgets.levels_preview import LevelsPreviewWidget

_SPINBOX_STYLE = (
    "QDoubleSpinBox { background: #3c3c3c; color: #d4d4d4; border: 1px solid #555;"
    " border-radius: 3px; padding: 3px 6px; }"
    "QDoubleSpinBox:focus { border-color: #4da6ff; }"
)
_INT_SPINBOX_STYLE = (
    "QSpinBox { background: #3c3c3c; color: #d4d4d4; border: 1px solid #555;"
    " border-radius: 3px; padding: 3px 6px; }"
    "QSpinBox:focus { border-color: #4da6ff; }"
)
_READONLY_STYLE = (
    "QLineEdit { background: #2a2a2a; color: #888; border: 1px solid #3a3a3a;"
    " border-radius: 3px; padding: 3px 6px; }"
)


class Step10Panel(BasePanel):
    STEP_ID   = "10"
    TITLE_KEY = "step10.title"
    DESC_KEY  = "step10.desc"
    OPTIONAL  = True
    HAS_NEXT  = False

    def __init__(self, parent: QWidget | None = None) -> None:
        self._output_dir: Path | None = None
        super().__init__(parent)

    def build_form(self) -> None:
        # ── Horizontal split: controls (left) | preview (right) ────────────
        main_widget = QWidget()
        main_widget.setStyleSheet("background: transparent;")
        main_hlayout = QHBoxLayout(main_widget)
        main_hlayout.setSpacing(16)
        main_hlayout.setContentsMargins(0, 0, 0, 0)

        # ── Left: controls ──────────────────────────────────────────────────
        left_widget = QWidget()
        left_widget.setStyleSheet("background: transparent;")
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(8)
        left_layout.setContentsMargins(0, 0, 0, 0)

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
        lbl_in = QLabel(S("step10.input_dir"))
        lbl_in.setToolTip("Step 7 RGB 합성 결과 폴더 (자동 설정)")
        fl.addRow(lbl_in, self._input_lbl)

        self._output_lbl = QLineEdit()
        self._output_lbl.setReadOnly(True)
        self._output_lbl.setStyleSheet(_READONLY_STYLE)
        lbl_out = QLabel(S("step10.output_dir"))
        lbl_out.setToolTip("요약 그리드 PNG가 저장될 폴더 (자동 설정)")
        fl.addRow(lbl_out, self._output_lbl)

        _tip_bp = (
            "이 값 이하의 픽셀을 순수 검정으로 처리합니다.\n"
            "배경 노이즈를 억제하고 행성 배경을 깔끔하게 만듭니다. (0.0~0.5)"
        )
        self._black_point = QDoubleSpinBox()
        self._black_point.setStyleSheet(_SPINBOX_STYLE)
        self._black_point.setRange(0.0, 0.5)
        self._black_point.setDecimals(2)
        self._black_point.setSingleStep(0.01)
        self._black_point.setValue(0.04)
        self._black_point.setToolTip(_tip_bp)
        lbl_bp = QLabel(S("step10.black_point"))
        lbl_bp.setToolTip(_tip_bp)
        fl.addRow(lbl_bp, self._black_point)

        _tip_gamma = (
            "감마 보정값입니다.\n"
            "1.0 = 선형 (보정 없음)  /  < 1.0 = 밝아짐  /  > 1.0 = 어두워짐\n"
            "행성 표면 디테일 강조를 위해 0.8~1.0 범위를 권장합니다."
        )
        self._gamma = QDoubleSpinBox()
        self._gamma.setStyleSheet(_SPINBOX_STYLE)
        self._gamma.setRange(0.1, 3.0)
        self._gamma.setDecimals(2)
        self._gamma.setSingleStep(0.05)
        self._gamma.setValue(0.9)
        self._gamma.setToolTip(_tip_gamma)
        lbl_gamma = QLabel(S("step10.gamma"))
        lbl_gamma.setToolTip(_tip_gamma)
        fl.addRow(lbl_gamma, self._gamma)

        _tip_cell = "요약 그리드에서 각 이미지 셀의 크기(px)입니다. (100~1024px 범위)"
        self._cell_size = QSpinBox()
        self._cell_size.setStyleSheet(_INT_SPINBOX_STYLE)
        self._cell_size.setRange(100, 1024)
        self._cell_size.setSingleStep(50)
        self._cell_size.setValue(300)
        self._cell_size.setToolTip(_tip_cell)
        lbl_cell = QLabel(S("step10.cell_size"))
        lbl_cell.setToolTip(_tip_cell)
        fl.addRow(lbl_cell, self._cell_size)

        left_layout.addWidget(form_widget)
        left_layout.addStretch()
        main_hlayout.addWidget(left_widget, 1)

        # ── Right: preview ──────────────────────────────────────────────────
        self._preview = LevelsPreviewWidget(parent=self)
        main_hlayout.addWidget(self._preview, 0)

        idx = self._form_layout.count() - 1
        self._form_layout.insertWidget(idx, main_widget)

        # Connect param changes → debounced preview update
        self._black_point.valueChanged.connect(self._on_params_changed)
        self._gamma.valueChanged.connect(self._on_params_changed)

    def get_config_updates(self) -> dict[str, Any]:
        return {
            "black_point":  self._black_point.value(),
            "gamma":        self._gamma.value(),
            "cell_size_px": self._cell_size.value(),
        }

    def load_session(self, data: dict[str, Any]) -> None:
        out = data.get("output_dir", "")
        if out:
            p = Path(out)
            self._input_lbl.setText(str(p / "step07_rgb_composite"))
            self._output_lbl.setText(str(p / "step10_summary_grid"))
            if hasattr(self, "_preview"):
                self._preview.set_input_dir(p / "step07_rgb_composite")
        self._black_point.setValue(float(data.get("black_point", 0.04)))
        self._gamma.setValue(float(data.get("gamma", 0.9)))
        self._cell_size.setValue(int(data.get("cell_size_px", 300)))
        if hasattr(self, "_preview"):
            self._preview.set_params(
                black_point=float(data.get("black_point", 0.04)),
                gamma=float(data.get("gamma", 0.9)),
            )

    def output_paths(self) -> list[Path]:
        if self._output_dir is None:
            return []
        step_dir = self._output_dir / "step10_summary_grid"
        if step_dir.exists():
            return sorted(step_dir.glob("*.png"))
        return []

    def set_output_dir(self, path: Path | str) -> None:
        self._output_dir = Path(path) if path else None
        step07_dir = self._output_dir / "step07_rgb_composite" if self._output_dir else None
        if hasattr(self, "_preview"):
            self._preview.set_input_dir(step07_dir)

    # ── Qt events ────────────────────────────────────────────────────────────

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if hasattr(self, "_preview"):
            self._preview.schedule_update(150)

    def refresh_after_run(self) -> None:
        super().refresh_after_run()
        if hasattr(self, "_preview"):
            self._preview.schedule_update(500)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_params_changed(self) -> None:
        if not hasattr(self, "_preview"):
            return
        self._preview.set_params(
            black_point=self._black_point.value(),
            gamma=self._gamma.value(),
        )
        self._preview.schedule_update()
