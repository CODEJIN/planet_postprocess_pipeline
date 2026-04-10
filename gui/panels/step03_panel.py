"""Step 3 — Wavelet preview sharpening panel."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QDoubleSpinBox,
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
_INPUT_STYLE = (
    "QLineEdit { background: #3c3c3c; color: #d4d4d4; border: 1px solid #555;"
    " border-radius: 3px; padding: 3px 6px; }"
    "QLineEdit:focus { border-color: #4da6ff; }"
)
_INPUT_EMPTY_STYLE = (
    "QLineEdit { background: #3c2020; color: #d4d4d4; border: 1px solid #884444;"
    " border-radius: 3px; padding: 3px 6px; }"
    "QLineEdit:focus { border-color: #ff6666; }"
)
_BTN_BROWSE = (
    "QPushButton { background: #3c3c3c; color: #aaa; border: 1px solid #555;"
    " border-radius: 3px; padding: 3px 8px; }"
    "QPushButton:hover { background: #4a4a4a; color: #d4d4d4; }"
)
_SLIDER_STYLE = (
    "QSlider::groove:horizontal { background: #444; height: 4px; border-radius: 2px; }"
    "QSlider::handle:horizontal { background: #4da6ff; width: 14px; height: 14px;"
    " margin: -5px 0; border-radius: 7px; }"
    "QSlider::sub-page:horizontal { background: #2d6a9f; border-radius: 2px; }"
)

_WAVELET_DEFAULTS = [200.0, 200.0, 200.0, 0.0, 0.0, 0.0]
_MAX_AMOUNT = 500.0


def _dir_row(parent: QWidget, line_edit: QLineEdit) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setSpacing(4)
    row.addWidget(line_edit)
    btn = QPushButton(S("btn.browse"))
    btn.setFixedWidth(70)
    btn.setStyleSheet(_BTN_BROWSE)

    def _browse():
        current = line_edit.text().strip()
        folder = QFileDialog.getExistingDirectory(
            parent, "폴더 선택", current or str(Path.home())
        )
        if folder:
            line_edit.setText(folder)
            line_edit.editingFinished.emit()

    btn.clicked.connect(_browse)
    row.addWidget(btn)
    return row


def _make_wavelet_row(
    parent: QWidget,
    level: int,
    default: float,
    on_change=None,
) -> tuple[QHBoxLayout, QDoubleSpinBox]:
    """Return (layout, spinbox) for one wavelet level.

    ``on_change`` is called (no args) whenever the slider OR spinbox changes.
    Both slider and spinbox connections are wired so each user action fires
    exactly one ``on_change`` call regardless of which control was used.
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

    # Sync slider ↔ spinbox (with signal blocking to avoid re-entrancy)
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

    # Auto-update callbacks:
    # - slider drag  → slider.valueChanged fires, spin is silenced → one on_change
    # - spin typed   → spin.valueChanged fires, slider is silenced → one on_change
    if on_change is not None:
        slider.valueChanged.connect(lambda _: on_change())
        spin.valueChanged.connect(lambda _: on_change())

    return row, spin


class Step03Panel(BasePanel):
    STEP_ID   = "03"
    TITLE_KEY = "step03.title"
    DESC_KEY  = "step03.desc"
    OPTIONAL  = False

    # Emitted when input/output dirs change so downstream panels can refresh.
    dirs_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        self._output_dir: Path | None = None
        self._output_manually_edited = False
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

        # Folder section
        folder_widget = QWidget()
        folder_widget.setStyleSheet("background: transparent;")
        fl = QFormLayout(folder_widget)
        fl.setSpacing(10)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        # AS!4 TIF input dir
        self._input_dir = QLineEdit()
        self._input_dir.setStyleSheet(_INPUT_EMPTY_STYLE)
        self._input_dir.setPlaceholderText(S("step03.input_dir.placeholder"))
        self._input_dir.textChanged.connect(self._on_input_changed)
        self._input_dir.editingFinished.connect(self._on_input_editing_finished)
        self._input_dir.editingFinished.connect(self.dirs_changed)
        lbl_in = QLabel(S("step03.input_dir"))
        lbl_in.setToolTip(
            "AutoStakkert 4가 TIF 파일을 저장한 폴더를 지정합니다.\n"
            "이 폴더 안의 모든 TIF 파일을 웨이블릿 처리합니다."
        )
        fl.addRow(lbl_in, _dir_row(self, self._input_dir))

        # Step3 output dir
        self._output_step3 = QLineEdit()
        self._output_step3.setStyleSheet(_INPUT_STYLE)
        self._output_step3.setPlaceholderText("자동 설정됩니다")
        self._output_step3.textEdited.connect(self._on_output_manually_edited)
        self._output_step3.editingFinished.connect(self.dirs_changed)
        lbl_out3 = QLabel(S("step03.output_dir"))
        lbl_out3.setToolTip(
            "웨이블릿 미리보기 PNG가 저장될 폴더입니다.\n"
            "입력 폴더 선택 시 자동으로 설정됩니다."
        )
        fl.addRow(lbl_out3, _dir_row(self, self._output_step3))
        left_layout.addWidget(folder_widget)

        # Wavelet amounts label
        amounts_label = QLabel(S("step03.amounts"))
        amounts_label.setStyleSheet("color: #aaa; font-size: 11px;")
        amounts_label.setToolTip(
            "L1~L6은 웨이블릿 분해 레이어의 선명화 강도입니다.\n"
            "L1 = 가장 미세한 디테일, L6 = 가장 큰 구조.\n"
            "행성 촬영에는 L1~L3만 활성화하는 것을 권장합니다."
        )
        left_layout.addWidget(amounts_label)

        # Slider rows — on_change triggers debounced preview update
        wavelet_widget = QWidget()
        wavelet_widget.setStyleSheet("background: transparent;")
        wav_layout = QVBoxLayout(wavelet_widget)
        wav_layout.setSpacing(6)
        wav_layout.setContentsMargins(0, 0, 0, 0)

        self._wavelet_spins: list[QDoubleSpinBox] = []
        for i, default in enumerate(_WAVELET_DEFAULTS):
            row_layout, spin = _make_wavelet_row(
                self, i + 1, default, on_change=self._on_params_changed
            )
            wav_layout.addLayout(row_layout)
            self._wavelet_spins.append(spin)
        left_layout.addWidget(wavelet_widget)

        # Border taper
        taper_widget = QWidget()
        taper_widget.setStyleSheet("background: transparent;")
        taper_form = QFormLayout(taper_widget)
        taper_form.setContentsMargins(0, 4, 0, 0)
        taper_form.setSpacing(8)
        taper_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self._border_taper = QSpinBox()
        self._border_taper.setStyleSheet(_INT_SPINBOX_STYLE)
        self._border_taper.setRange(0, 100)
        self._border_taper.setSingleStep(5)
        self._border_taper.setValue(0)
        lbl_taper = QLabel(S("step03.border_taper"))
        lbl_taper.setToolTip(
            "이미지 가장자리에 부드러운 경계를 적용합니다.\n"
            "0 = 비활성 (권장). 링잉 아티팩트가 심할 때만 사용하세요."
        )
        taper_form.addRow(lbl_taper, self._border_taper)
        left_layout.addWidget(taper_widget)
        left_layout.addStretch()

        main_hlayout.addWidget(left_widget, 1)

        # ── Right: preview ──────────────────────────────────────────────────
        self._preview = WaveletPreviewWidget(sharpen_filter=0.1, parent=self)
        main_hlayout.addWidget(self._preview, 0)

        idx = self._form_layout.count() - 1
        self._form_layout.insertWidget(idx, main_widget)

    def get_config_updates(self) -> dict[str, Any]:
        step3_out = self._output_step3.text().strip()
        output_base = str(Path(step3_out).parent) if step3_out else ""
        return {
            "input_dir":         self._input_dir.text().strip(),
            "output_dir":        output_base,
            "step03_output_dir": step3_out,
            "preview_amounts":   [s.value() for s in self._wavelet_spins],
            "border_taper_px":   self._border_taper.value(),
        }

    def load_session(self, data: dict[str, Any]) -> None:
        input_dir = data.get("input_dir", "")
        step3_out = data.get("step03_output_dir", "")

        self._input_dir.blockSignals(True)
        self._input_dir.setText(input_dir)
        self._input_dir.blockSignals(False)
        self._update_input_style(input_dir)

        if step3_out:
            self._output_manually_edited = True
            self._output_step3.setText(step3_out)
            self._output_dir = Path(step3_out).parent
        elif input_dir:
            self._auto_set_output(input_dir)

        amounts = data.get("preview_amounts", _WAVELET_DEFAULTS)
        for spin, val in zip(self._wavelet_spins, amounts):
            spin.setValue(float(val))
        self._border_taper.setValue(int(data.get("border_taper_px", 0)))

        # Sync preview
        if input_dir:
            self._preview.set_input_dir(input_dir)
            self._preview.set_params(amounts=amounts, levels=6, power=1.0)

    def output_paths(self) -> list[Path]:
        step3_out = self._output_step3.text().strip() if hasattr(self, "_output_step3") else ""
        if step3_out:
            p = Path(step3_out)
            if p.exists():
                return sorted(p.glob("*.png"))
        if self._output_dir is None:
            return []
        step_dir = self._output_dir / "step03_wavelet_preview"
        if not step_dir.exists():
            return []
        return sorted(step_dir.glob("*.png"))

    def set_output_dir(self, path: Path | str) -> None:
        if not self._output_manually_edited and not (
            hasattr(self, "_output_step3") and self._output_step3.text().strip()
        ):
            self._output_dir = Path(path) if path else None

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_input_changed(self, text: str) -> None:
        self._update_input_style(text)
        if hasattr(self, "_preview"):
            self._preview.set_input_dir(text.strip() or None)

    def _on_input_editing_finished(self) -> None:
        """Auto-set step03 output to a sub-folder of the TIF directory on focus-out.

        Always updates — changing the input directory drives the output path.
        """
        self._auto_set_output(self._input_dir.text())

    def _on_output_manually_edited(self, _text: str) -> None:
        self._output_manually_edited = True

    def _on_params_changed(self) -> None:
        if not hasattr(self, "_preview") or not hasattr(self, "_wavelet_spins"):
            return
        self._preview.set_params(
            amounts=[s.value() for s in self._wavelet_spins],
            levels=6,
            power=1.0,
        )
        self._preview.schedule_update()

    def _update_input_style(self, text: str) -> None:
        if text.strip():
            self._input_dir.setStyleSheet(_INPUT_STYLE)
        else:
            self._input_dir.setStyleSheet(_INPUT_EMPTY_STYLE)

    def _auto_set_output(self, input_text: str) -> None:
        t = input_text.strip()
        if not t:
            return
        p = Path(t)
        derived = str(p / "step03_wavelet_preview")
        self._output_step3.setText(derived)
        self._output_dir = p
