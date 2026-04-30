"""Step 8 — Time-series composite panel.

Architecture: Step08Panel wraps a QStackedWidget with two sub-widgets:
  - _Step08MonoWidget  : mono camera multi-filter compositing (unchanged)
  - _Step08ColorWidget : color camera per-frame stacking settings

load_session() detects camera_mode and switches the internal stack accordingly.
main_window.py needs no changes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from gui.i18n import S
from gui.panels.base_panel import BasePanel
from gui.panels.step05_panel import _make_combined_row
from gui.panels.step06_panel import (
    _SpecRow,
    _BTN_ADD,
    _BTN_SMALL,
    _COMBO_STYLE,
    _NAME_STYLE,
)

_INFO_STYLE = (
    "QLabel { background: #2a2a2a; color: #aaa; border: 1px solid #444;"
    " border-radius: 4px; padding: 10px; font-size: 11px; }"
)
_READONLY_STYLE = (
    "QLineEdit { background: #2a2a2a; color: #888; border: 1px solid #3a3a3a;"
    " border-radius: 3px; padding: 3px 6px; }"
)
_SPINBOX_STYLE = (
    "QSpinBox { background: #3c3c3c; color: #d4d4d4; border: 1px solid #555;"
    " border-radius: 3px; padding: 3px 6px; }"
    "QSpinBox:focus { border-color: #4da6ff; }"
)
_DBLSPIN_STYLE = (
    "QDoubleSpinBox { background: #3c3c3c; color: #d4d4d4; border: 1px solid #555;"
    " border-radius: 3px; padding: 3px 6px; }"
    "QDoubleSpinBox:focus { border-color: #4da6ff; }"
)

_SERIES_WAVELET_DEFAULTS = [200.0, 200.0, 200.0, 0.0, 0.0, 0.0]
_SERIES_DENOISE_DEFAULTS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Default composite specs for step 8 (mirrors step 7 defaults)
_DEFAULT_SERIES_SPECS = [
    {"name": "RGB",      "R": "R",   "G": "G", "B": "B",  "L": ""},
    {"name": "IR-RGB",   "R": "R",   "G": "G", "B": "B",  "L": "IR"},
    {"name": "CH4-G-IR", "R": "CH4", "G": "G", "B": "IR", "L": ""},
]

_SECTION_STYLE = "color: #4da6ff; font-size: 11px; font-weight: bold;"


# ── Mono sub-widget ────────────────────────────────────────────────────────────

class _Step08MonoWidget(QWidget):
    """Mono camera Step 8: multi-filter cycle compositing with independent composite specs."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._output_dir: Path | None = None
        self._spec_rows: list[_SpecRow] = []
        self._available_filters: list[str] = ["R", "G", "B", "IR", "CH4", "UV"]
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        # ── Folder display ────────────────────────────────────────────────────
        folder_widget = QWidget()
        folder_widget.setStyleSheet("background: transparent;")
        fl = QFormLayout(folder_widget)
        fl.setSpacing(8)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self._input_lbl = QLineEdit()
        self._input_lbl.setReadOnly(True)
        self._input_lbl.setStyleSheet(_READONLY_STYLE)
        lbl_in = QLabel(S("step08.input_dir"))
        lbl_in.setToolTip(S("step08.input_dir.tooltip"))
        fl.addRow(lbl_in, self._input_lbl)

        self._output_lbl = QLineEdit()
        self._output_lbl.setReadOnly(True)
        self._output_lbl.setStyleSheet(_READONLY_STYLE)
        lbl_out = QLabel(S("step08.output_dir"))
        lbl_out.setToolTip(S("step08.output_dir.tooltip"))
        fl.addRow(lbl_out, self._output_lbl)

        root.addWidget(folder_widget)

        # ── Options ───────────────────────────────────────────────────────────
        opt_widget = QWidget()
        opt_widget.setStyleSheet("background: transparent;")
        opt_fl = QFormLayout(opt_widget)
        opt_fl.setSpacing(10)
        opt_fl.setContentsMargins(0, 0, 0, 0)
        opt_fl.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self._global_normalize = QCheckBox()
        self._global_normalize.setChecked(True)
        self._global_normalize.setToolTip(S("step08.global_normalize.tooltip"))
        lbl_norm = QLabel(S("step08.global_normalize"))
        lbl_norm.setToolTip(S("step08.global_normalize.tooltip"))
        opt_fl.addRow(lbl_norm, self._global_normalize)

        self._series_scale = QDoubleSpinBox()
        self._series_scale.setStyleSheet(_DBLSPIN_STYLE)
        self._series_scale.setRange(0.1, 1.0)
        self._series_scale.setDecimals(2)
        self._series_scale.setSingleStep(0.05)
        self._series_scale.setValue(1.00)
        self._series_scale.setToolTip(S("step08.series_scale.tooltip"))
        lbl_scale = QLabel(S("step08.series_scale"))
        lbl_scale.setToolTip(S("step08.series_scale.tooltip"))
        opt_fl.addRow(lbl_scale, self._series_scale)

        self._stack_window_n = QSpinBox()
        self._stack_window_n.setStyleSheet(_SPINBOX_STYLE)
        self._stack_window_n.setRange(1, 9)
        self._stack_window_n.setSingleStep(2)
        self._stack_window_n.setValue(3)
        self._stack_window_n.setToolTip(S("step08.stack_window_n.tooltip"))
        lbl_window = QLabel(S("step08.stack_window_n"))
        lbl_window.setToolTip(S("step08.stack_window_n.tooltip"))
        opt_fl.addRow(lbl_window, self._stack_window_n)

        self._series_cycle_seconds = QSpinBox()
        self._series_cycle_seconds.setStyleSheet(_SPINBOX_STYLE)
        self._series_cycle_seconds.setRange(10, 600)
        self._series_cycle_seconds.setSingleStep(15)
        self._series_cycle_seconds.setValue(225)
        self._series_cycle_seconds.setToolTip(S("step08.cycle_seconds.tooltip"))
        lbl_cyc8 = QLabel(S("step08.cycle_seconds"))
        lbl_cyc8.setToolTip(S("step08.cycle_seconds.tooltip"))
        opt_fl.addRow(lbl_cyc8, self._series_cycle_seconds)

        self._stack_min_quality = QDoubleSpinBox()
        self._stack_min_quality.setStyleSheet(_DBLSPIN_STYLE)
        self._stack_min_quality.setRange(0.0, 0.9)
        self._stack_min_quality.setDecimals(2)
        self._stack_min_quality.setSingleStep(0.05)
        self._stack_min_quality.setValue(0.05)
        self._stack_min_quality.setToolTip(S("step08.stack_min_quality.tooltip"))
        lbl_minq = QLabel(S("step08.stack_min_quality"))
        lbl_minq.setToolTip(S("step08.stack_min_quality.tooltip"))
        opt_fl.addRow(lbl_minq, self._stack_min_quality)

        self._save_mono_frames = QCheckBox()
        self._save_mono_frames.setChecked(False)
        self._save_mono_frames.setToolTip(S("step08.save_mono_frames.tooltip"))
        lbl_mono = QLabel(S("step08.save_mono_frames"))
        lbl_mono.setToolTip(S("step08.save_mono_frames.tooltip"))
        opt_fl.addRow(lbl_mono, self._save_mono_frames)

        root.addWidget(opt_widget)

        # ── Composite specs (series-specific, independent from Step 7) ────────
        spec_header = QWidget()
        spec_header.setStyleSheet("background: transparent;")
        hdr_layout = QHBoxLayout(spec_header)
        hdr_layout.setContentsMargins(0, 8, 0, 4)
        hdr_layout.setSpacing(0)
        hdr_lbl = QLabel(S("step08.series_spec_header"))
        hdr_lbl.setStyleSheet("color: #aaa; font-size: 11px; font-weight: bold;")
        hdr_lbl.setToolTip(S("step08.series_spec_header.tooltip"))
        hdr_layout.addWidget(hdr_lbl)
        hdr_layout.addStretch()
        self._btn_add_spec = QPushButton(S("btn.add_spec"))
        self._btn_add_spec.setStyleSheet(_BTN_ADD)
        self._btn_add_spec.clicked.connect(self._add_spec_row)
        hdr_layout.addWidget(self._btn_add_spec)
        root.addWidget(spec_header)

        self._spec_container = QWidget()
        self._spec_container.setStyleSheet("background: #252525; border-radius: 4px;")
        self._spec_vbox = QVBoxLayout(self._spec_container)
        self._spec_vbox.setContentsMargins(6, 6, 6, 6)
        self._spec_vbox.setSpacing(2)

        col_hdr = QWidget()
        col_hdr.setStyleSheet("background: transparent;")
        col_hdr_layout = QHBoxLayout(col_hdr)
        col_hdr_layout.setContentsMargins(0, 0, 0, 2)
        col_hdr_layout.setSpacing(4)
        for txt, w in [(S("spec.col.name"), 90), (S("spec.col.r_channel"), 84), (S("spec.col.g_channel"), 84), (S("spec.col.b_channel"), 84), (S("spec.col.l_channel"), 84), ("", 24)]:
            lbl = QLabel(txt)
            lbl.setFixedWidth(w)
            lbl.setStyleSheet("color: #666; font-size: 10px;")
            col_hdr_layout.addWidget(lbl)
        self._spec_vbox.addWidget(col_hdr)

        scroll = QScrollArea()
        scroll.setWidget(self._spec_container)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(scroll.Shape.NoFrame)
        scroll.setMaximumHeight(160)
        scroll.setStyleSheet("QScrollArea { background: #252525; border-radius: 4px; }")
        scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        root.addWidget(scroll)

        for spec in _DEFAULT_SERIES_SPECS:
            self._add_spec_row(spec)

        # ── Wavelet sharpening (series) ───────────────────────────────────────
        wav_section = QWidget()
        wav_section.setStyleSheet("background: transparent;")
        wav_vl = QVBoxLayout(wav_section)
        wav_vl.setSpacing(6)
        wav_vl.setContentsMargins(0, 4, 0, 0)

        section_lbl = QLabel(
            f'<span style="color:#4da6ff">{S("step05.amounts")}</span>'
            ' <span style="color:#666">|</span> '
            f'<span style="color:#4aad80">{S("step05.denoise")}</span>'
        )
        section_lbl.setStyleSheet("font-size: 11px;")
        section_lbl.setToolTip(S("step08.series_amounts.tooltip"))
        wav_vl.addWidget(section_lbl)

        self._series_wavelet_spins: list[QDoubleSpinBox] = []
        self._series_denoise_spins: list[QDoubleSpinBox] = []
        for i, (sh, dn) in enumerate(zip(_SERIES_WAVELET_DEFAULTS, _SERIES_DENOISE_DEFAULTS)):
            row_layout, sharp_spin, dn_spin = _make_combined_row(i + 1, sh, dn)
            wav_vl.addLayout(row_layout)
            self._series_wavelet_spins.append(sharp_spin)
            self._series_denoise_spins.append(dn_spin)

        root.addWidget(wav_section)

    def get_config_updates(self) -> dict[str, Any]:
        series_specs = [r.to_dict() for r in self._spec_rows if r.to_dict()["name"]]
        return {
            "global_filter_normalize":      self._global_normalize.isChecked(),
            "series_scale":                 self._series_scale.value(),
            "stack_window_n":               self._stack_window_n.value(),
            "series_cycle_seconds":         self._series_cycle_seconds.value(),
            "stack_min_quality":            self._stack_min_quality.value(),
            "save_mono_frames":             self._save_mono_frames.isChecked(),
            "series_amounts":               [s.value() for s in self._series_wavelet_spins],
            "series_denoise_amounts":       [s.value() for s in self._series_denoise_spins],
            "series_composite_specs":       series_specs,
        }

    def validate(self, config: dict, batch_mode: bool = False) -> list:
        from gui.validation import ValidationIssue, count_files, filter_files_in_dir
        issues = []
        specs = config.get("series_composite_specs") or config.get("composite_specs") or []
        if not specs:
            issues.append(ValidationIssue("error", S("validate.no_series_specs")))
        if not batch_mode:
            input_dir = config.get("input_dir", "").strip()
            if not input_dir or not count_files(input_dir, "*.tif", "*.TIF"):
                issues.append(ValidationIssue("error", S("validate.no_tif_input")))
            elif specs:
                needed_filters: set[str] = set()
                for spec in specs:
                    for role in ("R", "G", "B", "L"):
                        fname = spec.get(role, "")
                        if fname:
                            needed_filters.add(fname)
                stack_n = int(config.get("stack_window_n", 3))
                for filt in sorted(needed_filters):
                    n = filter_files_in_dir(input_dir, filt)
                    if not n:
                        issues.append(ValidationIssue(
                            "error", S("validate.no_filter_tif", f=filt, d=input_dir),
                        ))
                    elif stack_n > n:
                        issues.append(ValidationIssue(
                            "error", S("validate.stack_window_too_large", f=filt, n=stack_n, count=n),
                        ))
        return issues

    def load_session(self, data: dict[str, Any]) -> None:
        out = data.get("output_dir", "")
        input_dir = data.get("input_dir", "")
        if input_dir:
            self._input_lbl.setText(input_dir)
        elif out:
            self._input_lbl.setText(str(Path(out) / "step02_lucky_stack"))
        if out:
            p = Path(out)
            self._output_lbl.setText(str(p / "step08_series"))
            self._output_dir = p
        self._global_normalize.setChecked(bool(data.get("global_filter_normalize", True)))
        self._series_scale.setValue(float(data.get("series_scale", 1.00)))
        self._series_cycle_seconds.setValue(int(data.get("series_cycle_seconds",
                                                          data.get("cycle_seconds", 225))))
        self._stack_window_n.setValue(int(data.get("stack_window_n", 3)))
        self._stack_min_quality.setValue(float(data.get("stack_min_quality", 0.05)))
        self._save_mono_frames.setChecked(bool(data.get("save_mono_frames", False)))
        amounts = data.get("series_amounts", _SERIES_WAVELET_DEFAULTS)
        for spin, val in zip(self._series_wavelet_spins, amounts):
            spin.setValue(float(val))

        denoise = data.get("series_denoise_amounts", _SERIES_DENOISE_DEFAULTS)
        for spin, val in zip(self._series_denoise_spins, denoise):
            spin.setValue(float(val))

        # Update available filter options from session
        raw = data.get("filters", "")
        if raw:
            filters = [f.strip() for f in raw.split(",") if f.strip()]
            mono_filters = [f for f in filters if f not in ("COLOR", "RGB")]
            if mono_filters:
                self._refresh_filter_options(mono_filters)

        # Load series-specific composite specs (fall back to step07 specs if not set)
        saved_specs = data.get("series_composite_specs") or data.get("composite_specs")
        if saved_specs:
            for row in list(self._spec_rows):
                row.setParent(None)
                row.deleteLater()
            self._spec_rows.clear()
            for spec in saved_specs:
                self._add_spec_row(spec)

    def output_paths(self) -> list[Path]:
        if self._output_dir is None:
            return []
        step_dir = self._output_dir / "step08_series"
        if step_dir.exists():
            return sorted(step_dir.glob("*.png"))
        return []

    def set_output_dir(self, path: Path | str) -> None:
        self._output_dir = Path(path) if path else None

    def _add_spec_row(self, spec: dict[str, str] | None = None) -> None:
        row = _SpecRow(self._available_filters, spec, parent=self._spec_container)
        row.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        row._btn_del.clicked.connect(lambda _, r=row: self._remove_spec_row(r))
        row._radio.setVisible(False)
        row._radio.setMaximumWidth(0)
        self._spec_rows.append(row)
        self._spec_vbox.addWidget(row)

    def _remove_spec_row(self, row: _SpecRow) -> None:
        if row in self._spec_rows:
            self._spec_rows.remove(row)
        row.setParent(None)
        row.deleteLater()

    def _refresh_filter_options(self, filters: list[str]) -> None:
        self._available_filters = filters
        for row in self._spec_rows:
            row.update_filters(filters)


# ── Color sub-widget ───────────────────────────────────────────────────────────

class _Step08ColorWidget(QWidget):
    """Color camera Step 8: sliding-window stacking of a single color stream.

    Controls shown differ from mono:
      - No 'Global filter normalize' (only one color channel — no cross-filter
        normalisation is needed).
      - No 'Save mono filter GIFs' (no separate filter channels to save).
      - 'Capture interval' (촬영 간격) replaces 'Filter cycle' — the concept
        changes from a 5-filter rotation to a single-image cadence.
      - Stack window N upper limit raised to 99 (color cameras can capture
        much faster, so larger windows are practical).
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._output_dir: Path | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        # ── Folder display ────────────────────────────────────────────────────
        folder_widget = QWidget()
        folder_widget.setStyleSheet("background: transparent;")
        fl = QFormLayout(folder_widget)
        fl.setSpacing(8)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self._input_lbl = QLineEdit()
        self._input_lbl.setReadOnly(True)
        self._input_lbl.setStyleSheet(_READONLY_STYLE)
        lbl_in = QLabel(S("step08.input_dir"))
        lbl_in.setToolTip(S("step08.input_dir.tooltip"))
        fl.addRow(lbl_in, self._input_lbl)

        self._output_lbl = QLineEdit()
        self._output_lbl.setReadOnly(True)
        self._output_lbl.setStyleSheet(_READONLY_STYLE)
        lbl_out = QLabel(S("step08.output_dir"))
        lbl_out.setToolTip(S("step08.output_dir.tooltip"))
        fl.addRow(lbl_out, self._output_lbl)

        root.addWidget(folder_widget)

        # ── Options ───────────────────────────────────────────────────────────
        opt_widget = QWidget()
        opt_widget.setStyleSheet("background: transparent;")
        opt_fl = QFormLayout(opt_widget)
        opt_fl.setSpacing(10)
        opt_fl.setContentsMargins(0, 0, 0, 0)
        opt_fl.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self._series_scale = QDoubleSpinBox()
        self._series_scale.setStyleSheet(_DBLSPIN_STYLE)
        self._series_scale.setRange(0.1, 1.0)
        self._series_scale.setDecimals(2)
        self._series_scale.setSingleStep(0.05)
        self._series_scale.setValue(1.00)
        self._series_scale.setToolTip(S("step08.series_scale.tooltip"))
        lbl_scale = QLabel(S("step08.series_scale"))
        lbl_scale.setToolTip(S("step08.series_scale.tooltip"))
        opt_fl.addRow(lbl_scale, self._series_scale)

        self._series_cycle_seconds = QSpinBox()
        self._series_cycle_seconds.setStyleSheet(_SPINBOX_STYLE)
        self._series_cycle_seconds.setRange(5, 300)
        self._series_cycle_seconds.setSingleStep(5)
        self._series_cycle_seconds.setValue(30)
        self._series_cycle_seconds.setToolTip(S("step08.cycle_seconds_color.tooltip"))
        lbl_cyc = QLabel(S("step08.cycle_seconds_color"))
        lbl_cyc.setToolTip(S("step08.cycle_seconds_color.tooltip"))
        opt_fl.addRow(lbl_cyc, self._series_cycle_seconds)

        self._stack_window_n = QSpinBox()
        self._stack_window_n.setStyleSheet(_SPINBOX_STYLE)
        self._stack_window_n.setRange(1, 99)
        self._stack_window_n.setSingleStep(2)
        self._stack_window_n.setValue(5)
        self._stack_window_n.setToolTip(S("step08.stack_window_n.tooltip"))
        lbl_window = QLabel(S("step08.stack_window_n"))
        lbl_window.setToolTip(S("step08.stack_window_n.tooltip"))
        opt_fl.addRow(lbl_window, self._stack_window_n)

        self._stack_min_quality = QDoubleSpinBox()
        self._stack_min_quality.setStyleSheet(_DBLSPIN_STYLE)
        self._stack_min_quality.setRange(0.0, 0.9)
        self._stack_min_quality.setDecimals(2)
        self._stack_min_quality.setSingleStep(0.05)
        self._stack_min_quality.setValue(0.05)
        self._stack_min_quality.setToolTip(S("step08.stack_min_quality.tooltip"))
        lbl_minq = QLabel(S("step08.stack_min_quality"))
        lbl_minq.setToolTip(S("step08.stack_min_quality.tooltip"))
        opt_fl.addRow(lbl_minq, self._stack_min_quality)

        root.addWidget(opt_widget)

        # ── Wavelet sharpening (series) ───────────────────────────────────────
        wav_section = QWidget()
        wav_section.setStyleSheet("background: transparent;")
        wav_vl = QVBoxLayout(wav_section)
        wav_vl.setSpacing(6)
        wav_vl.setContentsMargins(0, 4, 0, 0)

        section_lbl = QLabel(
            f'<span style="color:#4da6ff">{S("step05.amounts")}</span>'
            ' <span style="color:#666">|</span> '
            f'<span style="color:#4aad80">{S("step05.denoise")}</span>'
        )
        section_lbl.setStyleSheet("font-size: 11px;")
        section_lbl.setToolTip(S("step08.series_amounts.tooltip"))
        wav_vl.addWidget(section_lbl)

        self._series_wavelet_spins: list[QDoubleSpinBox] = []
        self._series_denoise_spins: list[QDoubleSpinBox] = []
        for i, (sh, dn) in enumerate(zip(_SERIES_WAVELET_DEFAULTS, _SERIES_DENOISE_DEFAULTS)):
            row_layout, sharp_spin, dn_spin = _make_combined_row(i + 1, sh, dn)
            wav_vl.addLayout(row_layout)
            self._series_wavelet_spins.append(sharp_spin)
            self._series_denoise_spins.append(dn_spin)

        root.addWidget(wav_section)

    def get_config_updates(self) -> dict[str, Any]:
        return {
            "series_scale":          self._series_scale.value(),
            "series_cycle_seconds":  self._series_cycle_seconds.value(),
            "stack_window_n":        self._stack_window_n.value(),
            "stack_min_quality":     self._stack_min_quality.value(),
            "series_amounts":        [s.value() for s in self._series_wavelet_spins],
            "series_denoise_amounts": [s.value() for s in self._series_denoise_spins],
        }

    def load_session(self, data: dict[str, Any]) -> None:
        inp = data.get("input_dir", "")
        if inp:
            self._input_lbl.setText(inp)
        out = data.get("output_dir", "")
        if out:
            p = Path(out)
            self._output_lbl.setText(str(p / "step08_series"))
            self._output_dir = p
        self._series_scale.setValue(float(data.get("series_scale", 1.00)))
        self._series_cycle_seconds.setValue(int(data.get("series_cycle_seconds", 30)))
        self._stack_window_n.setValue(int(data.get("stack_window_n", 5)))
        self._stack_min_quality.setValue(float(data.get("stack_min_quality", 0.05)))
        amounts = data.get("series_amounts", _SERIES_WAVELET_DEFAULTS)
        for spin, val in zip(self._series_wavelet_spins, amounts):
            spin.setValue(float(val))
        denoise = data.get("series_denoise_amounts", _SERIES_DENOISE_DEFAULTS)
        for spin, val in zip(self._series_denoise_spins, denoise):
            spin.setValue(float(val))

    def output_paths(self) -> list[Path]:
        if self._output_dir is None:
            return []
        step_dir = self._output_dir / "step08_series"
        if step_dir.exists():
            return sorted(step_dir.glob("*.png"))
        return []

    def set_output_dir(self, path: Path | str) -> None:
        self._output_dir = Path(path) if path else None


# ── Panel wrapper ──────────────────────────────────────────────────────────────

class Step08Panel(BasePanel):
    STEP_ID   = "08"
    TITLE_KEY = "step08.title"
    DESC_KEY  = "step08.desc"
    OPTIONAL  = True

    def __init__(self, parent: QWidget | None = None) -> None:
        self._output_dir: Path | None = None
        self._is_color: bool = False
        super().__init__(parent)

    def build_form(self) -> None:
        self._sub_stack    = QStackedWidget()
        self._mono_widget  = _Step08MonoWidget()
        self._color_widget = _Step08ColorWidget()
        self._sub_stack.addWidget(self._mono_widget)   # index 0
        self._sub_stack.addWidget(self._color_widget)  # index 1

        idx = self._form_layout.count() - 1
        self._form_layout.insertWidget(idx, self._sub_stack)

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

    def output_paths(self) -> list[Path]:
        if self._is_color:
            return self._color_widget.output_paths()
        return self._mono_widget.output_paths()

    def validate(self, config: dict, batch_mode: bool = False) -> list:
        if self._is_color:
            return []  # color mode: series_composite_specs 불필요
        return self._mono_widget.validate(config, batch_mode)

    def set_output_dir(self, path: Path | str) -> None:
        self._output_dir = Path(path) if path else None
        self._mono_widget.set_output_dir(self._output_dir)
        self._color_widget.set_output_dir(self._output_dir)
