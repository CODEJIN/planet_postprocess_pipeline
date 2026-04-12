"""Step 6 — Wavelet master sharpening panel."""
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
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from gui.i18n import S
from gui.panels.base_panel import BasePanel
from gui.widgets.wavelet_preview import WaveletPreviewWidget

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
_SLIDER_STYLE = (
    "QSlider::groove:horizontal { background: #444; height: 4px; border-radius: 2px; }"
    "QSlider::handle:horizontal { background: #4da6ff; width: 14px; height: 14px;"
    " margin: -5px 0; border-radius: 7px; }"
    "QSlider::sub-page:horizontal { background: #2d6a9f; border-radius: 2px; }"
)

_WAVELET_DEFAULTS = [200.0, 200.0, 200.0, 0.0, 0.0, 0.0]
_MAX_AMOUNT = 500.0


def _make_wavelet_row(
    level: int,
    default: float,
    on_change=None,
) -> tuple[QHBoxLayout, QDoubleSpinBox]:
    """Return (layout, spinbox) for one wavelet level.

    ``on_change`` fires exactly once per user action (slider drag or spin type).
    """
    row = QHBoxLayout()
    row.setSpacing(6)

    lbl = QLabel(f"L{level}")
    lbl.setFixedWidth(22)
    lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    lbl.setStyleSheet("color: #d4d4d4; font-size: 12px;")
    row.addWidget(lbl)

    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setRange(0, int(_MAX_AMOUNT))
    slider.setValue(int(default))
    slider.setStyleSheet(_SLIDER_STYLE)
    row.addWidget(slider, 1)

    spin = QDoubleSpinBox()
    spin.setStyleSheet(_SPINBOX_STYLE)
    spin.setRange(0.0, _MAX_AMOUNT)
    spin.setDecimals(1)
    spin.setSingleStep(10.0)
    spin.setValue(default)
    spin.setFixedWidth(72)
    row.addWidget(spin)

    def _slider_to_spin(v: int) -> None:
        spin.blockSignals(True)
        spin.setValue(float(v))
        spin.blockSignals(False)

    def _spin_to_slider(v: float) -> None:
        slider.blockSignals(True)
        slider.setValue(int(v))
        slider.blockSignals(False)

    slider.valueChanged.connect(_slider_to_spin)
    spin.valueChanged.connect(_spin_to_slider)

    if on_change is not None:
        slider.valueChanged.connect(lambda _: on_change())
        spin.valueChanged.connect(lambda _: on_change())

    return row, spin


class Step06Panel(BasePanel):
    STEP_ID   = "06"
    TITLE_KEY = "step06.title"
    DESC_KEY  = "step06.desc"
    OPTIONAL  = False

    def __init__(self, parent: QWidget | None = None) -> None:
        self._output_dir: Path | None = None
        super().__init__(parent)

    # ── BasePanel interface ───────────────────────────────────────────────────

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

        # Folder display (read-only)
        folder_widget = QWidget()
        folder_widget.setStyleSheet("background: transparent;")
        fl = QFormLayout(folder_widget)
        fl.setSpacing(8)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self._input_lbl = QLineEdit()
        self._input_lbl.setReadOnly(True)
        self._input_lbl.setStyleSheet(_READONLY_STYLE)
        lbl_in = QLabel(S("step06.input_dir"))
        lbl_in.setToolTip("Step 5 De-rotation 결과물이 있는 폴더 (자동 설정)")
        fl.addRow(lbl_in, self._input_lbl)

        self._output_lbl = QLineEdit()
        self._output_lbl.setReadOnly(True)
        self._output_lbl.setStyleSheet(_READONLY_STYLE)
        lbl_out = QLabel(S("step06.output_dir"))
        lbl_out.setToolTip("웨이블릿 마스터 이미지가 저장될 폴더 (자동 설정)")
        fl.addRow(lbl_out, self._output_lbl)
        left_layout.addWidget(folder_widget)

        # Section label
        amounts_label = QLabel(S("step06.amounts"))
        amounts_label.setStyleSheet("color: #aaa; font-size: 11px;")
        amounts_label.setToolTip(
            "스태킹된 마스터 이미지에 적용할 웨이블릿 선명화 강도입니다.\n"
            "마스터 이미지는 노이즈가 훨씬 낮으므로 더 강하게 적용해도 안전합니다."
        )
        left_layout.addWidget(amounts_label)

        # Wavelet slider rows — on_change triggers debounced preview update
        wavelet_widget = QWidget()
        wavelet_widget.setStyleSheet("background: transparent;")
        wav_layout = QVBoxLayout(wavelet_widget)
        wav_layout.setSpacing(6)
        wav_layout.setContentsMargins(0, 0, 0, 0)

        self._wavelet_spins: list[QDoubleSpinBox] = []
        defaults = list(_WAVELET_DEFAULTS)
        for i in range(0, len(defaults), 2):
            pair = QHBoxLayout()
            pair.setSpacing(12)
            for j in range(2):
                if i + j < len(defaults):
                    row_layout, spin = _make_wavelet_row(
                        i + j + 1, defaults[i + j], on_change=self._on_params_changed
                    )
                    pair.addLayout(row_layout)
                    self._wavelet_spins.append(spin)
            wav_layout.addLayout(pair)
        left_layout.addWidget(wavelet_widget)

        left_layout.addStretch()

        main_hlayout.addWidget(left_widget, 1)

        # ── Right: preview ──────────────────────────────────────────────────
        self._preview = WaveletPreviewWidget(sharpen_filter=0.0, parent=self)
        main_hlayout.addWidget(self._preview, 0)

        idx = self._form_layout.count() - 1
        self._form_layout.insertWidget(idx, main_widget)

    def retranslate(self) -> None:
        self._preview.retranslate()

    def get_config_updates(self) -> dict[str, Any]:
        return {
            "master_amounts": [s.value() for s in self._wavelet_spins],
        }

    def load_session(self, data: dict[str, Any]) -> None:
        out = data.get("output_dir", "")
        if out:
            p = Path(out)
            self._input_lbl.setText(str(p / "step05_derotated"))
            self._output_lbl.setText(str(p / "step06_wavelet_master"))
            if hasattr(self, "_preview"):
                self._preview.set_input_dir(p / "step05_derotated")
        amounts = data.get("master_amounts", _WAVELET_DEFAULTS)
        for spin, val in zip(self._wavelet_spins, amounts):
            spin.setValue(float(val))
        if hasattr(self, "_preview"):
            self._preview.set_params(
                amounts=amounts, levels=6, power=1.0,
                auto_params=True,
            )

    def output_paths(self) -> list[Path]:
        if self._output_dir is None:
            return []
        step_dir = self._output_dir / "step06_wavelet_master"
        if not step_dir.exists():
            return []
        return sorted(step_dir.rglob("*.png"))

    def set_output_dir(self, path: Path | str) -> None:
        self._output_dir = Path(path) if path else None
        step05_dir = self._output_dir / "step05_derotated" if self._output_dir else None
        if hasattr(self, "_preview"):
            self._preview.set_input_dir(step05_dir)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_params_changed(self) -> None:
        if not hasattr(self, "_preview") or not hasattr(self, "_wavelet_spins"):
            return
        self._preview.set_params(
            amounts=[s.value() for s in self._wavelet_spins],
            levels=6,
            power=1.0,
            auto_params=True,
        )
        self._preview.schedule_update()
