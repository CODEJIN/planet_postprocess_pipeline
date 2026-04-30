"""Step 7 — Wavelet preview sharpening panel."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QStackedWidget,
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
_INPUT_STYLE = (
    "QLineEdit { background: #3c3c3c; color: #d4d4d4; border: 1px solid #555;"
    " border-radius: 3px; padding: 3px 6px; }"
    "QLineEdit:focus { border-color: #4da6ff; }"
)
_INPUT_EMPTY_STYLE = (
    "QLineEdit { background: #2a2a2a; color: #888; border: 1px solid #3a3a3a;"
    " border-radius: 3px; padding: 3px 6px; }"
)
_READONLY_STYLE = (
    "QLineEdit { background: #2a2a2a; color: #888; border: 1px solid #3a3a3a;"
    " border-radius: 3px; padding: 3px 6px; }"
)
_BTN_STYLE = (
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
_SLIDER_STYLE_DENOISE = (
    "QSlider::groove:horizontal { background: #444; height: 4px; border-radius: 2px; }"
    "QSlider::handle:horizontal { background: #4aad80; width: 14px; height: 14px;"
    " margin: -5px 0; border-radius: 7px; }"
    "QSlider::sub-page:horizontal { background: #1e6b47; border-radius: 2px; }"
)
_CHECKBOX_STYLE = (
    "QCheckBox { color: #d4d4d4; font-size: 12px; }"
    "QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #555;"
    " border-radius: 3px; background: #3c3c3c; }"
    "QCheckBox::indicator:checked { background: #4da6ff; border-color: #4da6ff; }"
)

_WAVELET_DEFAULTS = [200.0, 200.0, 200.0, 0.0, 0.0, 0.0]
_DENOISE_DEFAULTS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
_MAX_AMOUNT  = 500.0
_MAX_DENOISE = 3.0


def _make_combined_row(
    level: int,
    sharpen_default: float,
    denoise_default: float,
    on_change=None,
) -> tuple[QHBoxLayout, QDoubleSpinBox, QDoubleSpinBox]:
    row = QHBoxLayout()
    row.setSpacing(4)

    lbl = QLabel(f"L{level}")
    lbl.setFixedWidth(22)
    lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    lbl.setStyleSheet("color: #d4d4d4; font-size: 12px;")
    row.addWidget(lbl)

    sharp_n = int(_MAX_AMOUNT)
    sharp_slider = QSlider(Qt.Orientation.Horizontal)
    sharp_slider.setRange(0, sharp_n)
    sharp_slider.setValue(round(sharpen_default / _MAX_AMOUNT * sharp_n))
    sharp_slider.setStyleSheet(_SLIDER_STYLE)
    row.addWidget(sharp_slider, 3)

    sharp_spin = QDoubleSpinBox()
    sharp_spin.setStyleSheet(_SPINBOX_STYLE)
    sharp_spin.setRange(0.0, _MAX_AMOUNT)
    sharp_spin.setDecimals(1)
    sharp_spin.setSingleStep(10.0)
    sharp_spin.setValue(sharpen_default)
    sharp_spin.setFixedWidth(68)
    row.addWidget(sharp_spin)

    sep = QFrame()
    sep.setFrameShape(QFrame.Shape.VLine)
    sep.setFrameShadow(QFrame.Shadow.Sunken)
    sep.setStyleSheet("color: #555;")
    sep.setFixedWidth(8)
    row.addWidget(sep)

    dn_n = 300
    dn_slider = QSlider(Qt.Orientation.Horizontal)
    dn_slider.setRange(0, dn_n)
    dn_slider.setValue(round(denoise_default / _MAX_DENOISE * dn_n))
    dn_slider.setStyleSheet(_SLIDER_STYLE_DENOISE)
    row.addWidget(dn_slider, 1)

    dn_spin = QDoubleSpinBox()
    dn_spin.setStyleSheet(_SPINBOX_STYLE)
    dn_spin.setRange(0.0, _MAX_DENOISE)
    dn_spin.setDecimals(2)
    dn_spin.setSingleStep(0.05)
    dn_spin.setValue(denoise_default)
    dn_spin.setFixedWidth(58)
    row.addWidget(dn_spin)

    def _sharp_slider_to_spin(v): sharp_spin.blockSignals(True); sharp_spin.setValue(v / sharp_n * _MAX_AMOUNT); sharp_spin.blockSignals(False)
    def _sharp_spin_to_slider(v): sharp_slider.blockSignals(True); sharp_slider.setValue(round(v / _MAX_AMOUNT * sharp_n)); sharp_slider.blockSignals(False)
    def _dn_slider_to_spin(v): dn_spin.blockSignals(True); dn_spin.setValue(v / dn_n * _MAX_DENOISE); dn_spin.blockSignals(False)
    def _dn_spin_to_slider(v): dn_slider.blockSignals(True); dn_slider.setValue(round(v / _MAX_DENOISE * dn_n)); dn_slider.blockSignals(False)

    sharp_slider.valueChanged.connect(_sharp_slider_to_spin)
    sharp_spin.valueChanged.connect(_sharp_spin_to_slider)
    dn_slider.valueChanged.connect(_dn_slider_to_spin)
    dn_spin.valueChanged.connect(_dn_spin_to_slider)

    if on_change is not None:
        sharp_slider.valueChanged.connect(lambda _: on_change())
        sharp_spin.valueChanged.connect(lambda _: on_change())
        dn_slider.valueChanged.connect(lambda _: on_change())
        dn_spin.valueChanged.connect(lambda _: on_change())

    return row, sharp_spin, dn_spin


def _section_label_text() -> str:
    return (
        f'<span style="color:#4da6ff">{S("step05.amounts")}</span>'
        ' <span style="color:#666">|</span> '
        f'<span style="color:#4aad80">{S("step05.denoise")}</span>'
    )


def _make_dir_row(line_edit: QLineEdit, browse_cb) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setSpacing(4)
    row.addWidget(line_edit)
    btn = QPushButton(S("btn.browse"))
    btn.setFixedWidth(70)
    btn.setStyleSheet(_BTN_STYLE)
    btn.clicked.connect(browse_cb)
    row.addWidget(btn)
    return row


def _build_combined_sliders(parent_layout: QVBoxLayout, on_change) -> tuple[list, list]:
    """Add 6 combined sharpen+denoise rows; return (sharp_spins, dn_spins)."""
    sharp_spins: list[QDoubleSpinBox] = []
    dn_spins: list[QDoubleSpinBox] = []
    for i, (sh, dn) in enumerate(zip(_WAVELET_DEFAULTS, _DENOISE_DEFAULTS)):
        row_layout, sharp_spin, dn_spin = _make_combined_row(
            i + 1, sh, dn, on_change=on_change
        )
        parent_layout.addLayout(row_layout)
        sharp_spins.append(sharp_spin)
        dn_spins.append(dn_spin)
    return sharp_spins, dn_spins


# ── Mono sub-widget ────────────────────────────────────────────────────────────

class _Step07MonoWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._output_dir: Path | None = None
        self._wavelet_spins: list[QDoubleSpinBox] = []
        self._denoise_spins: list[QDoubleSpinBox] = []
        self._build_ui()

    def _build_ui(self) -> None:
        main_hlayout = QHBoxLayout(self)
        main_hlayout.setSpacing(16)
        main_hlayout.setContentsMargins(0, 0, 0, 0)

        left_widget = QWidget()
        left_widget.setStyleSheet("background: transparent;")
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(8)
        left_layout.setContentsMargins(0, 0, 0, 0)

        folder_widget = QWidget()
        folder_widget.setStyleSheet("background: transparent;")
        fl = QFormLayout(folder_widget)
        fl.setSpacing(8)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self._input_lbl = QLineEdit()
        self._input_lbl.setStyleSheet(_INPUT_EMPTY_STYLE)
        self._input_lbl.setPlaceholderText(S("step07.input_dir.placeholder"))
        self._input_lbl.textChanged.connect(self._on_input_changed)
        lbl_in = QLabel(S("step07.input_dir"))
        lbl_in.setToolTip(S("step07.input_dir.tooltip"))
        fl.addRow(lbl_in, _make_dir_row(self._input_lbl, self._browse_input))

        self._output_lbl = QLineEdit()
        self._output_lbl.setReadOnly(True)
        self._output_lbl.setStyleSheet(_READONLY_STYLE)
        lbl_out = QLabel(S("step07.output_dir"))
        lbl_out.setToolTip(S("step07.output_dir.tooltip"))
        fl.addRow(lbl_out, self._output_lbl)
        left_layout.addWidget(folder_widget)

        section_lbl = QLabel(_section_label_text())
        section_lbl.setStyleSheet("font-size: 11px;")
        left_layout.addWidget(section_lbl)

        self._wavelet_spins, self._denoise_spins = _build_combined_sliders(
            left_layout, self._on_params_changed
        )
        left_layout.addStretch()

        main_hlayout.addWidget(left_widget, 1)

        self._preview = WaveletPreviewWidget(sharpen_filter=0.1, parent=self)
        main_hlayout.addWidget(self._preview, 0)

    def retranslate(self) -> None:
        self._preview.retranslate()

    def load_session(self, data: dict[str, Any]) -> None:
        inp = data.get("input_dir", "")
        out = data.get("output_dir", "")

        self._input_lbl.blockSignals(True)
        self._input_lbl.setText(inp)
        self._input_lbl.blockSignals(False)
        self._update_input_style(inp)

        if inp:
            self._preview.set_input_dir(inp)
        if out:
            p = Path(out)
            self._output_lbl.setText(str(p / "step07_wavelet_preview"))
            self._output_dir = p

        amounts = data.get("preview_amounts", _WAVELET_DEFAULTS)
        for spin, val in zip(self._wavelet_spins, amounts):
            spin.setValue(float(val))

        denoise = data.get("preview_denoise_amounts", _DENOISE_DEFAULTS)
        for spin, val in zip(self._denoise_spins, denoise):
            spin.setValue(float(val))

        self._preview.set_params(
            amounts=amounts, levels=6, power=1.0,
            denoise_amounts=denoise, filter_type="gaussian",
        )

    def get_config_updates(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "preview_amounts":         [s.value() for s in self._wavelet_spins],
            "preview_denoise_amounts": [s.value() for s in self._denoise_spins],
        }
        inp_text = self._input_lbl.text().strip()
        if inp_text:
            result["input_dir"] = inp_text
        out_text = self._output_lbl.text().strip()
        if out_text:
            result["output_dir"] = str(Path(out_text).parent)
        return result

    def validate(self, config: dict, batch_mode: bool = False) -> list:
        from gui.validation import ValidationIssue, count_files
        issues = []
        if not batch_mode:
            input_dir = config.get("input_dir", "").strip()
            if not input_dir or not count_files(input_dir, "*.tif", "*.TIF"):
                issues.append(ValidationIssue("error", S("validate.no_tif_lucky")))
        return issues

    def output_paths(self) -> list[Path]:
        if self._output_dir is None:
            return []
        step_dir = self._output_dir / "step07_wavelet_preview"
        if not step_dir.exists():
            return []
        return sorted(step_dir.glob("*.png"))

    def set_output_dir(self, path: Path | str | None) -> None:
        self._output_dir = Path(path) if path else None

    def _browse_input(self) -> None:
        current = self._input_lbl.text().strip()
        folder = QFileDialog.getExistingDirectory(
            self, S("dialog.folder_select"), current or str(Path.home())
        )
        if folder:
            self._input_lbl.setText(folder)

    def _update_input_style(self, text: str) -> None:
        self._input_lbl.setStyleSheet(
            _INPUT_STYLE if text.strip() else _INPUT_EMPTY_STYLE
        )

    def _on_input_changed(self, text: str) -> None:
        self._update_input_style(text)
        self._preview.set_input_dir(text.strip())
        inp = text.strip()
        if inp:
            p = Path(inp)
            out = p.parent / "step07_wavelet_preview"
            self._output_lbl.setText(str(out))
            self._output_dir = p.parent
        else:
            self._output_lbl.setText("")
            self._output_dir = None

    def _on_params_changed(self) -> None:
        if not hasattr(self, "_wavelet_spins"):
            return
        self._preview.set_params(
            amounts=[s.value() for s in self._wavelet_spins],
            levels=6,
            power=1.0,
            denoise_amounts=[s.value() for s in self._denoise_spins],
            filter_type="gaussian",
        )
        self._preview.schedule_update()


# ── Color sub-widget ───────────────────────────────────────────────────────────

class _Step07ColorWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._output_dir: Path | None = None
        self._wavelet_spins: list[QDoubleSpinBox] = []
        self._denoise_spins: list[QDoubleSpinBox] = []
        self._build_ui()

    def _build_ui(self) -> None:
        main_hlayout = QHBoxLayout(self)
        main_hlayout.setSpacing(16)
        main_hlayout.setContentsMargins(0, 0, 0, 0)

        left_widget = QWidget()
        left_widget.setStyleSheet("background: transparent;")
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(8)
        left_layout.setContentsMargins(0, 0, 0, 0)

        folder_widget = QWidget()
        folder_widget.setStyleSheet("background: transparent;")
        fl = QFormLayout(folder_widget)
        fl.setSpacing(8)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self._input_lbl = QLineEdit()
        self._input_lbl.setStyleSheet(_INPUT_EMPTY_STYLE)
        self._input_lbl.setPlaceholderText(S("step07.input_dir.placeholder"))
        self._input_lbl.textChanged.connect(self._on_input_changed)
        lbl_in = QLabel(S("step07.input_dir"))
        lbl_in.setToolTip(S("step07.input_dir.tooltip"))
        fl.addRow(lbl_in, _make_dir_row(self._input_lbl, self._browse_input))

        self._output_lbl = QLineEdit()
        self._output_lbl.setReadOnly(True)
        self._output_lbl.setStyleSheet(_READONLY_STYLE)
        lbl_out = QLabel(S("step07.output_dir"))
        lbl_out.setToolTip(S("step07.output_dir.tooltip"))
        fl.addRow(lbl_out, self._output_lbl)
        left_layout.addWidget(folder_widget)

        self._chk_color_correct = QCheckBox(S("step07.color_correct"))
        self._chk_color_correct.setStyleSheet(_CHECKBOX_STYLE)
        self._chk_color_correct.setToolTip(S("step07.color_correct.tooltip"))
        self._chk_color_correct.setChecked(True)
        self._chk_color_correct.toggled.connect(self._on_color_correct_toggled)
        left_layout.addWidget(self._chk_color_correct)

        section_lbl = QLabel(_section_label_text())
        section_lbl.setStyleSheet("font-size: 11px;")
        left_layout.addWidget(section_lbl)

        self._wavelet_spins, self._denoise_spins = _build_combined_sliders(
            left_layout, self._on_params_changed
        )
        left_layout.addStretch()

        main_hlayout.addWidget(left_widget, 1)

        self._preview = WaveletPreviewWidget(sharpen_filter=0.1, parent=self)
        self._preview.set_color_correct(True)
        main_hlayout.addWidget(self._preview, 0)

    def retranslate(self) -> None:
        self._preview.retranslate()
        self._chk_color_correct.setText(S("step07.color_correct"))
        self._chk_color_correct.setToolTip(S("step07.color_correct.tooltip"))

    def load_session(self, data: dict[str, Any]) -> None:
        inp = data.get("input_dir", "")
        out = data.get("output_dir", "")
        color_correct = bool(data.get("wavelet_color_correct", True))

        self._chk_color_correct.blockSignals(True)
        self._chk_color_correct.setChecked(color_correct)
        self._chk_color_correct.blockSignals(False)
        self._preview.set_color_correct(color_correct)

        self._input_lbl.blockSignals(True)
        self._input_lbl.setText(inp)
        self._input_lbl.blockSignals(False)
        self._update_input_style(inp)

        if inp:
            self._preview.set_input_dir(inp)
        if out:
            p = Path(out)
            self._output_lbl.setText(str(p / "step07_wavelet_preview"))
            self._output_dir = p

        amounts = data.get("preview_amounts", _WAVELET_DEFAULTS)
        for spin, val in zip(self._wavelet_spins, amounts):
            spin.setValue(float(val))

        denoise = data.get("preview_denoise_amounts", _DENOISE_DEFAULTS)
        for spin, val in zip(self._denoise_spins, denoise):
            spin.setValue(float(val))

        self._preview.set_params(
            amounts=amounts, levels=6, power=1.0,
            denoise_amounts=denoise, filter_type="gaussian",
        )

    def get_config_updates(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "preview_amounts":         [s.value() for s in self._wavelet_spins],
            "preview_denoise_amounts": [s.value() for s in self._denoise_spins],
            "wavelet_color_correct":   self._chk_color_correct.isChecked(),
        }
        inp_text = self._input_lbl.text().strip()
        if inp_text:
            result["input_dir"] = inp_text
        out_text = self._output_lbl.text().strip()
        if out_text:
            result["output_dir"] = str(Path(out_text).parent)
        return result

    def validate(self, config: dict, batch_mode: bool = False) -> list:
        from gui.validation import ValidationIssue, count_files
        issues = []
        if not batch_mode:
            input_dir = config.get("input_dir", "").strip()
            if not input_dir or not count_files(input_dir, "*.tif", "*.TIF"):
                issues.append(ValidationIssue("error", S("validate.no_tif_lucky")))
        return issues

    def output_paths(self) -> list[Path]:
        if self._output_dir is None:
            return []
        step_dir = self._output_dir / "step07_wavelet_preview"
        if not step_dir.exists():
            return []
        return sorted(step_dir.glob("*.png"))

    def set_output_dir(self, path: Path | str | None) -> None:
        self._output_dir = Path(path) if path else None

    def _browse_input(self) -> None:
        current = self._input_lbl.text().strip()
        folder = QFileDialog.getExistingDirectory(
            self, S("dialog.folder_select"), current or str(Path.home())
        )
        if folder:
            self._input_lbl.setText(folder)

    def _update_input_style(self, text: str) -> None:
        self._input_lbl.setStyleSheet(
            _INPUT_STYLE if text.strip() else _INPUT_EMPTY_STYLE
        )

    def _on_input_changed(self, text: str) -> None:
        self._update_input_style(text)
        self._preview.set_input_dir(text.strip())
        inp = text.strip()
        if inp:
            p = Path(inp)
            out = p.parent / "step07_wavelet_preview"
            self._output_lbl.setText(str(out))
            self._output_dir = p.parent
        else:
            self._output_lbl.setText("")
            self._output_dir = None

    def _on_color_correct_toggled(self, checked: bool) -> None:
        self._preview.set_color_correct(checked)
        self._preview.schedule_update()

    def _on_params_changed(self) -> None:
        if not hasattr(self, "_wavelet_spins"):
            return
        self._preview.set_params(
            amounts=[s.value() for s in self._wavelet_spins],
            levels=6,
            power=1.0,
            denoise_amounts=[s.value() for s in self._denoise_spins],
            filter_type="gaussian",
        )
        self._preview.schedule_update()


# ── Wrapper panel ──────────────────────────────────────────────────────────────

class Step07Panel(BasePanel):
    STEP_ID   = "07"
    TITLE_KEY = "step07.title"
    DESC_KEY  = "step07.desc"
    OPTIONAL  = True

    def __init__(self, parent: QWidget | None = None) -> None:
        self._output_dir: Path | None = None
        self._is_color: bool = False
        super().__init__(parent)

    def build_form(self) -> None:
        self._sub_stack   = QStackedWidget()
        self._mono_widget = _Step07MonoWidget()
        self._color_widget = _Step07ColorWidget()
        self._sub_stack.addWidget(self._mono_widget)    # 0
        self._sub_stack.addWidget(self._color_widget)   # 1

        idx = self._form_layout.count() - 1
        self._form_layout.insertWidget(idx, self._sub_stack)

    def retranslate(self) -> None:
        self._mono_widget.retranslate()
        self._color_widget.retranslate()

    def get_config_updates(self) -> dict[str, Any]:
        if self._is_color:
            return self._color_widget.get_config_updates()
        return self._mono_widget.get_config_updates()

    def load_session(self, data: dict[str, Any]) -> None:
        self._is_color = data.get("camera_mode", "mono") == "color"
        self._sub_stack.setCurrentIndex(1 if self._is_color else 0)
        if self._is_color:
            self._color_widget.load_session(data)
        else:
            self._mono_widget.load_session(data)

    def validate(self, config: dict, batch_mode: bool = False) -> list:
        if self._is_color:
            return self._color_widget.validate(config, batch_mode)
        return self._mono_widget.validate(config, batch_mode)

    def output_paths(self) -> list[Path]:
        if self._is_color:
            return self._color_widget.output_paths()
        return self._mono_widget.output_paths()

    def set_output_dir(self, path: Path | str) -> None:
        self._output_dir = Path(path) if path else None
        self._mono_widget.set_output_dir(self._output_dir)
        self._color_widget.set_output_dir(self._output_dir)
