"""Step 5 — Wavelet master sharpening panel."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSlider,
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
_SLIDER_STYLE_DENOISE = (
    "QSlider::groove:horizontal { background: #444; height: 4px; border-radius: 2px; }"
    "QSlider::handle:horizontal { background: #4aad80; width: 14px; height: 14px;"
    " margin: -5px 0; border-radius: 7px; }"
    "QSlider::sub-page:horizontal { background: #1e6b47; border-radius: 2px; }"
)

_WAVELET_DEFAULTS = [200.0, 200.0, 200.0, 0.0, 0.0, 0.0]
_DENOISE_DEFAULTS = [0.15, 0.15, 0.15, 0.0, 0.0, 0.0]
_MAX_AMOUNT  = 500.0
_MAX_DENOISE = 3.0


def _make_wavelet_row(
    level: int,
    default: float,
    max_val: float = _MAX_AMOUNT,
    step: float = 10.0,
    decimals: int = 1,
    spin_width: int = 72,
    slider_steps: int = 0,
    on_change=None,
) -> tuple[QHBoxLayout, QDoubleSpinBox]:
    """Return (layout, spinbox) for one wavelet level slider row."""
    row = QHBoxLayout()
    row.setSpacing(6)

    lbl = QLabel(f"L{level}")
    lbl.setFixedWidth(22)
    lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    lbl.setStyleSheet("color: #d4d4d4; font-size: 12px;")
    row.addWidget(lbl)

    n_steps = slider_steps if slider_steps > 0 else int(max_val)
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setRange(0, n_steps)
    slider.setValue(round(default / max_val * n_steps))
    slider.setStyleSheet(_SLIDER_STYLE)
    row.addWidget(slider, 1)

    spin = QDoubleSpinBox()
    spin.setStyleSheet(_SPINBOX_STYLE)
    spin.setRange(0.0, max_val)
    spin.setDecimals(decimals)
    spin.setSingleStep(step)
    spin.setValue(default)
    spin.setFixedWidth(spin_width)
    row.addWidget(spin)

    def _slider_to_spin(v: int) -> None:
        spin.blockSignals(True)
        spin.setValue(v / n_steps * max_val)
        spin.blockSignals(False)

    def _spin_to_slider(v: float) -> None:
        slider.blockSignals(True)
        slider.setValue(round(v / max_val * n_steps))
        slider.blockSignals(False)

    slider.valueChanged.connect(_slider_to_spin)
    spin.valueChanged.connect(_spin_to_slider)

    if on_change is not None:
        slider.valueChanged.connect(lambda _: on_change())
        spin.valueChanged.connect(lambda _: on_change())

    return row, spin


def _make_combined_row(
    level: int,
    sharpen_default: float,
    denoise_default: float,
    on_change=None,
) -> tuple[QHBoxLayout, QDoubleSpinBox, QDoubleSpinBox]:
    """One combined row: level label | sharpen (3/4) | separator | denoise (1/4)."""
    row = QHBoxLayout()
    row.setSpacing(4)

    lbl = QLabel(f"L{level}")
    lbl.setFixedWidth(22)
    lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    lbl.setStyleSheet("color: #d4d4d4; font-size: 12px;")
    row.addWidget(lbl)

    # ── Sharpen slider + spinbox ────────────────────────────────────────────
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

    # ── Visual separator ────────────────────────────────────────────────────
    sep = QFrame()
    sep.setFrameShape(QFrame.Shape.VLine)
    sep.setFrameShadow(QFrame.Shadow.Sunken)
    sep.setStyleSheet("color: #555;")
    sep.setFixedWidth(8)
    row.addWidget(sep)

    # ── Denoise slider + spinbox ────────────────────────────────────────────
    dn_n = 300  # 0–3.0 in 0.01 px increments
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

    # ── Bidirectional sync ──────────────────────────────────────────────────
    def _sharp_slider_to_spin(v: int) -> None:
        sharp_spin.blockSignals(True)
        sharp_spin.setValue(v / sharp_n * _MAX_AMOUNT)
        sharp_spin.blockSignals(False)

    def _sharp_spin_to_slider(v: float) -> None:
        sharp_slider.blockSignals(True)
        sharp_slider.setValue(round(v / _MAX_AMOUNT * sharp_n))
        sharp_slider.blockSignals(False)

    def _dn_slider_to_spin(v: int) -> None:
        dn_spin.blockSignals(True)
        dn_spin.setValue(v / dn_n * _MAX_DENOISE)
        dn_spin.blockSignals(False)

    def _dn_spin_to_slider(v: float) -> None:
        dn_slider.blockSignals(True)
        dn_slider.setValue(round(v / _MAX_DENOISE * dn_n))
        dn_slider.blockSignals(False)

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


class Step05Panel(BasePanel):
    STEP_ID   = "05"
    TITLE_KEY = "step05.title"
    DESC_KEY  = "step05.desc"
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
        lbl_in = QLabel(S("step05.input_dir"))
        lbl_in.setToolTip(S("step05.input_dir.tooltip"))
        fl.addRow(lbl_in, self._input_lbl)

        self._output_lbl = QLineEdit()
        self._output_lbl.setReadOnly(True)
        self._output_lbl.setStyleSheet(_READONLY_STYLE)
        lbl_out = QLabel(S("step05.output_dir"))
        lbl_out.setToolTip(S("step05.output_dir.tooltip"))
        fl.addRow(lbl_out, self._output_lbl)
        left_layout.addWidget(folder_widget)

        # ── Combined sharpen + denoise section ─────────────────────────────
        self._section_label = QLabel()
        self._section_label.setStyleSheet("font-size: 11px;")
        self._section_label.setToolTip(S("step05.amounts.tooltip"))
        self._section_label.setText(self._section_label_text())
        left_layout.addWidget(self._section_label)

        layers_widget = QWidget()
        layers_widget.setStyleSheet("background: transparent;")
        layers_layout = QVBoxLayout(layers_widget)
        layers_layout.setSpacing(4)
        layers_layout.setContentsMargins(0, 0, 0, 0)

        self._wavelet_spins: list[QDoubleSpinBox] = []
        self._denoise_spins: list[QDoubleSpinBox] = []
        for i, (sh_def, dn_def) in enumerate(zip(_WAVELET_DEFAULTS, _DENOISE_DEFAULTS)):
            row_layout, sharp_spin, dn_spin = _make_combined_row(
                i + 1, sh_def, dn_def, on_change=self._on_params_changed,
            )
            layers_layout.addLayout(row_layout)
            self._wavelet_spins.append(sharp_spin)
            self._denoise_spins.append(dn_spin)
        left_layout.addWidget(layers_widget)

        left_layout.addStretch()

        main_hlayout.addWidget(left_widget, 1)

        # ── Right: preview ──────────────────────────────────────────────────
        self._preview = WaveletPreviewWidget(sharpen_filter=0.0, parent=self)
        main_hlayout.addWidget(self._preview, 0)

        idx = self._form_layout.count() - 1
        self._form_layout.insertWidget(idx, main_widget)

    def _section_label_text(self) -> str:
        return (
            f'<span style="color:#4da6ff">{S("step05.amounts")}</span>'
            ' <span style="color:#666">|</span> '
            f'<span style="color:#4aad80">{S("step05.denoise")}</span>'
        )

    def retranslate(self) -> None:
        self._preview.retranslate()
        self._section_label.setText(self._section_label_text())
        self._section_label.setToolTip(S("step05.amounts.tooltip"))

    def get_config_updates(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "master_amounts":         [s.value() for s in self._wavelet_spins],
            "master_denoise_amounts": [s.value() for s in self._denoise_spins],
            "master_filter_type":     "gaussian",
        }
        out_text = self._output_lbl.text().strip()
        if out_text:
            result["output_dir"] = str(Path(out_text).parent)
        return result

    def validate(self, config: dict, batch_mode: bool = False) -> list:
        from gui.validation import ValidationIssue, count_files
        issues = []
        if not batch_mode:
            out_base = config.get("output_dir", "").strip()
            input_path = str(Path(out_base) / "step04_derotated") if out_base else ""
            if not count_files(input_path, "**/*.tif", "**/*.TIF"):
                issues.append(ValidationIssue("error", S("validate.no_derotation_tif")))
        return issues

    def load_session(self, data: dict[str, Any]) -> None:
        out = data.get("output_dir", "")
        if out:
            p = Path(out)
            self._input_lbl.setText(str(p / "step04_derotated"))
            self._output_lbl.setText(str(p / "step05_wavelet_master"))
            if hasattr(self, "_preview"):
                self._preview.set_input_dir(p / "step04_derotated")

        amounts = data.get("master_amounts", _WAVELET_DEFAULTS)
        for spin, val in zip(self._wavelet_spins, amounts):
            spin.setValue(float(val))

        denoise = data.get("master_denoise_amounts", _DENOISE_DEFAULTS)
        for spin, val in zip(self._denoise_spins, denoise):
            spin.setValue(float(val))

        if hasattr(self, "_preview"):
            self._preview.set_params(
                amounts=amounts, levels=6, power=1.0,
                auto_params=False,
                denoise_amounts=denoise,
                filter_type="gaussian",
            )

    def output_paths(self) -> list[Path]:
        if self._output_dir is None:
            return []
        step_dir = self._output_dir / "step05_wavelet_master"
        if not step_dir.exists():
            return []
        return sorted(step_dir.rglob("*.png"))

    def set_output_dir(self, path: Path | str) -> None:
        self._output_dir = Path(path) if path else None
        step05_dir = self._output_dir / "step04_derotated" if self._output_dir else None
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
            auto_params=False,
            denoise_amounts=[s.value() for s in self._denoise_spins],
            filter_type="gaussian",
        )
        self._preview.schedule_update()
