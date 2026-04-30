"""Step 6 — RGB composite (mono) / Color correction (color camera) panel.

Architecture: Step06Panel is a wrapper (BasePanel) that contains a QStackedWidget
with two sub-widgets:
  - _Step06MonoWidget   : mono camera composite spec UI (filter → R/G/B/L mapping)
  - _Step06ColorWidget  : color camera auto-correction preview UI

Color camera mode:
  Per-window automatic white balance + CA correction runs entirely inside the
  pipeline (_auto_color_correct in step06_rgb_composite.py).  The GUI panel
  shows a before/after preview using the same algorithm so the user can verify
  the correction quality before running the full pipeline.

load_session() detects camera_mode and switches the internal stack accordingly.
main_window.py needs no changes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
from PySide6.QtCore import QObject, QThread, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from gui.i18n import S
from gui.panels.base_panel import BasePanel
from gui.widgets.rgb_composite_preview import RgbCompositePreviewWidget

# ── Shared styles ──────────────────────────────────────────────────────────────

_SPINBOX_STYLE = (
    "QDoubleSpinBox { background: #3c3c3c; color: #d4d4d4; border: 1px solid #555;"
    " border-radius: 3px; padding: 3px 6px; }"
    "QDoubleSpinBox:focus { border-color: #4da6ff; }"
)
_READONLY_STYLE = (
    "QLineEdit { background: #2a2a2a; color: #888; border: 1px solid #3a3a3a;"
    " border-radius: 3px; padding: 3px 6px; }"
)
_COMBO_STYLE = (
    "QComboBox { background: #3c3c3c; color: #d4d4d4; border: 1px solid #555;"
    " border-radius: 3px; padding: 2px 6px; }"
    "QComboBox::drop-down { border: none; width: 20px; }"
    "QComboBox QAbstractItemView { background: #3c3c3c; color: #d4d4d4;"
    " selection-background-color: #4da6ff; }"
)
_NAME_STYLE = (
    "QLineEdit { background: #3c3c3c; color: #d4d4d4; border: 1px solid #555;"
    " border-radius: 3px; padding: 3px 6px; }"
    "QLineEdit:focus { border-color: #4da6ff; }"
)
_BTN_SMALL = (
    "QPushButton { background: #3c3c3c; color: #aaa; border: 1px solid #555;"
    " border-radius: 3px; padding: 2px 8px; font-size: 11px; }"
    "QPushButton:hover { background: #4a4a4a; color: #d4d4d4; }"
)
_BTN_ADD = (
    "QPushButton { background: #2d6a4f; color: white; border: none;"
    " border-radius: 3px; padding: 3px 10px; font-size: 11px; }"
    "QPushButton:hover { background: #40916c; }"
)
_BTN_REFRESH = (
    "QPushButton { background: #1e4a7a; color: #7ec8f0; border: 1px solid #2d6aaa;"
    " border-radius: 4px; padding: 5px 14px; font-size: 11px; }"
    "QPushButton:hover { background: #2558a0; color: #b0d8f8; }"
    "QPushButton:disabled { background: #2a2a2a; color: #555; border-color: #3a3a3a; }"
)
_CHK_STYLE = (
    "QCheckBox { color: #d4d4d4; font-size: 11px; spacing: 5px; }"
    "QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #555;"
    " border-radius: 2px; background: #3c3c3c; }"
    "QCheckBox::indicator:checked { background: #4da6ff; border-color: #4da6ff; }"
)
_SECTION_STYLE = "color: #4da6ff; font-size: 11px; font-weight: bold;"
_INFO_STYLE    = "color: #888; font-size: 10px;"
_STATUS_STYLE  = "color: #666; font-size: 10px; font-style: italic;"
_PANEL_STYLE   = "QLabel { background: #1a1a1a; border: 1px solid #444; border-radius: 4px; }"
_CAP_STYLE     = "color: #888; font-size: 10px;"

_PANEL_SIZE = 200   # px per preview panel

# Default composite specs shown for mono mode
_DEFAULT_SPECS = [
    {"name": "RGB",      "R": "R",   "G": "G", "B": "B",  "L": ""},
    {"name": "IR-RGB",   "R": "R",   "G": "G", "B": "B",  "L": "IR"},
    {"name": "CH4-G-IR", "R": "CH4", "G": "G", "B": "IR", "L": ""},
]


# ── Mono sub-widget helpers ────────────────────────────────────────────────────

class _SpecRow(QWidget):
    """A single composite spec row: name + R/G/B/L dropdowns + delete button."""

    def __init__(
        self,
        filters: list[str],
        spec: dict[str, str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._filters = filters

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 1, 0, 1)
        row.setSpacing(4)

        self._radio = QRadioButton()
        self._radio.setFixedWidth(20)
        self._radio.setToolTip(S("spec.tooltip.preview"))
        row.addWidget(self._radio)

        self._name = QLineEdit()
        self._name.setStyleSheet(_NAME_STYLE)
        self._name.setFixedWidth(90)
        self._name.setPlaceholderText(S("spec.col.name"))
        row.addWidget(self._name)

        self._combos: dict[str, QComboBox] = {}
        for ch_label, ch_key in [("R", "R"), ("G", "G"), ("B", "B"), ("L", "L")]:
            lbl = QLabel(ch_label + ":")
            lbl.setStyleSheet("color: #aaa; font-size: 11px;")
            lbl.setFixedWidth(14)
            row.addWidget(lbl)

            cb = QComboBox()
            cb.setStyleSheet(_COMBO_STYLE)
            cb.setFixedWidth(70)
            options = (["──"] if ch_key == "L" else []) + filters
            for opt in options:
                cb.addItem(opt)
            self._combos[ch_key] = cb
            row.addWidget(cb)

        self._btn_del = QPushButton("✕")
        self._btn_del.setStyleSheet(_BTN_SMALL)
        self._btn_del.setFixedWidth(24)
        self._btn_del.setToolTip(S("spec.tooltip.delete"))
        row.addWidget(self._btn_del)

        if spec:
            self._load(spec)

    def _load(self, spec: dict[str, str]) -> None:
        self._name.setText(spec.get("name", ""))
        for ch in ("R", "G", "B", "L"):
            val = spec.get(ch, "")
            cb = self._combos[ch]
            idx = cb.findText(val if val else "──")
            if idx >= 0:
                cb.setCurrentIndex(idx)

    def update_filters(self, filters: list[str]) -> None:
        self._filters = filters
        for ch, cb in self._combos.items():
            current = cb.currentText()
            cb.blockSignals(True)
            cb.clear()
            options = (["──"] if ch == "L" else []) + filters
            for opt in options:
                cb.addItem(opt)
            idx = cb.findText(current)
            # max(0, idx) would silently fall back to index-0 ("IR") when the
            # previous value is not found in the new filter list.  Instead keep
            # the previous text if it exists; only fall back to 0 as last resort.
            cb.setCurrentIndex(idx if idx >= 0 else 0)
            cb.blockSignals(False)

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self._name.text().strip(),
            "R": self._combos["R"].currentText(),
            "G": self._combos["G"].currentText(),
            "B": self._combos["B"].currentText(),
            "L": "" if self._combos["L"].currentText() == "──" else self._combos["L"].currentText(),
        }


# ── Mono sub-widget ────────────────────────────────────────────────────────────

class _Step06MonoWidget(QWidget):
    """Mono composite spec UI: filter-to-channel mapping table + preview."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._output_dir: Path | None = None
        self._spec_rows: list[_SpecRow] = []
        self._available_filters: list[str] = ["R", "G", "B", "IR", "CH4", "UV"]
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

        form_widget = QWidget()
        form_widget.setStyleSheet("background: transparent;")
        fl = QFormLayout(form_widget)
        fl.setSpacing(10)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self._input_lbl = QLineEdit()
        self._input_lbl.setReadOnly(True)
        self._input_lbl.setStyleSheet(_READONLY_STYLE)
        self._input_lbl.setToolTip(S("step06.input_dir.tooltip"))
        lbl_in = QLabel(S("step06.input_dir"))
        fl.addRow(lbl_in, self._input_lbl)

        self._output_lbl = QLineEdit()
        self._output_lbl.setReadOnly(True)
        self._output_lbl.setStyleSheet(_READONLY_STYLE)
        lbl_out = QLabel(S("step06.output_dir"))
        fl.addRow(lbl_out, self._output_lbl)

        _tip_shift = S("step06.max_shift.tooltip")
        self._max_shift = QDoubleSpinBox()
        self._max_shift.setStyleSheet(_SPINBOX_STYLE)
        self._max_shift.setRange(0.0, 100.0)
        self._max_shift.setDecimals(1)
        self._max_shift.setSingleStep(1.0)
        self._max_shift.setValue(15.0)
        self._max_shift.setToolTip(_tip_shift)
        lbl_shift = QLabel(S("step06.max_shift"))
        lbl_shift.setToolTip(_tip_shift)
        fl.addRow(lbl_shift, self._max_shift)

        left_layout.addWidget(form_widget)

        spec_header = QWidget()
        spec_header.setStyleSheet("background: transparent;")
        hdr_layout = QHBoxLayout(spec_header)
        hdr_layout.setContentsMargins(0, 8, 0, 4)
        hdr_layout.setSpacing(0)
        hdr_lbl = QLabel(S("step06.spec_header"))
        hdr_lbl.setStyleSheet("color: #aaa; font-size: 11px; font-weight: bold;")
        hdr_layout.addWidget(hdr_lbl)
        hdr_layout.addStretch()
        self._btn_add_spec = QPushButton(S("btn.add_spec"))
        self._btn_add_spec.setStyleSheet(_BTN_ADD)
        self._btn_add_spec.clicked.connect(self._add_spec_row)
        hdr_layout.addWidget(self._btn_add_spec)
        left_layout.addWidget(spec_header)

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
        for txt, w in [("👁", 20), (S("spec.col.name"), 90), (S("spec.col.r_channel"), 84), (S("spec.col.g_channel"), 84), (S("spec.col.b_channel"), 84), (S("spec.col.l_channel"), 84), ("", 24)]:
            lbl = QLabel(txt)
            lbl.setFixedWidth(w)
            lbl.setStyleSheet("color: #666; font-size: 10px;")
            col_hdr_layout.addWidget(lbl)
        self._spec_vbox.addWidget(col_hdr)

        scroll = QScrollArea()
        scroll.setWidget(self._spec_container)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(scroll.Shape.NoFrame)
        scroll.setMaximumHeight(200)
        scroll.setStyleSheet("QScrollArea { background: #252525; border-radius: 4px; }")
        scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        left_layout.addWidget(scroll)

        chk_row = QHBoxLayout()
        chk_row.setContentsMargins(0, 6, 0, 0)
        chk_row.setSpacing(16)
        self._chk_global_normalize = QCheckBox(S("step06.global_normalize"))
        self._chk_stretch  = QCheckBox(S("step06.stretch_enabled"))
        self._chk_saturate = QCheckBox(S("step06.saturation_boost"))
        self._chk_global_normalize.setStyleSheet(_CHK_STYLE)
        self._chk_stretch.setStyleSheet(_CHK_STYLE)
        self._chk_saturate.setStyleSheet(_CHK_STYLE)
        self._chk_global_normalize.setChecked(True)
        self._chk_global_normalize.setToolTip(S("step06.global_normalize.tooltip"))
        self._chk_stretch.setChecked(False)
        self._chk_saturate.setChecked(True)
        self._chk_global_normalize.toggled.connect(self._on_params_changed)
        self._chk_stretch.toggled.connect(self._on_params_changed)
        self._chk_saturate.toggled.connect(self._on_params_changed)
        chk_row.addWidget(self._chk_global_normalize)
        chk_row.addWidget(self._chk_stretch)
        chk_row.addWidget(self._chk_saturate)
        chk_row.addStretch()
        left_layout.addLayout(chk_row)

        left_layout.addStretch()
        main_hlayout.addWidget(left_widget, 1)

        self._spec_radio_group = QButtonGroup(self)
        self._spec_radio_group.buttonClicked.connect(self._on_params_changed)

        for spec in _DEFAULT_SPECS:
            self._add_spec_row(spec)

        self._preview = RgbCompositePreviewWidget(parent=self)
        main_hlayout.addWidget(self._preview, 0)

        for row in self._spec_rows:
            for cb in row._combos.values():
                cb.currentIndexChanged.connect(self._on_params_changed)

    def retranslate(self) -> None:
        self._preview.retranslate()
        self._chk_global_normalize.setText(S("step06.global_normalize"))
        self._chk_global_normalize.setToolTip(S("step06.global_normalize.tooltip"))
        self._chk_stretch.setText(S("step06.stretch_enabled"))
        self._chk_saturate.setText(S("step06.saturation_boost"))

    def load_session(self, data: dict[str, Any]) -> None:
        out = data.get("output_dir", "")
        if out:
            p = Path(out)
            self._input_lbl.setText(str(p / "step05_wavelet_master"))
            self._output_lbl.setText(str(p / "step06_rgb_composite"))
            if hasattr(self, "_preview"):
                self._preview.set_input_dir(p / "step05_wavelet_master")
        self._max_shift.setValue(float(data.get("max_shift_px", 15.0)))
        self._chk_global_normalize.setChecked(bool(data.get("global_normalize", True)))
        self._chk_stretch.setChecked(bool(data.get("stretch_enabled", False)))
        self._chk_saturate.setChecked(bool(data.get("saturation_boost", True)))

        raw = data.get("filters", "")
        if raw:
            filters = [f.strip() for f in raw.split(",") if f.strip()]
            mono_filters = [f for f in filters if f not in ("COLOR", "RGB")]
            if mono_filters:
                self._refresh_filter_options(mono_filters)

        saved_specs = data.get("composite_specs")
        if saved_specs:
            for row in list(self._spec_rows):
                if hasattr(self, "_spec_radio_group"):
                    self._spec_radio_group.removeButton(row._radio)
                row.setParent(None)
                row.deleteLater()
            self._spec_rows.clear()
            for spec in saved_specs:
                self._add_spec_row(spec)

        self._sync_preview_spec()

    def get_config_updates(self) -> dict[str, Any]:
        specs = [r.to_dict() for r in self._spec_rows if r.to_dict()["name"]]
        result: dict[str, Any] = {
            "max_shift_px":     self._max_shift.value(),
            "composite_specs":  specs,
            "global_normalize": self._chk_global_normalize.isChecked(),
            "stretch_enabled":  self._chk_stretch.isChecked(),
            "saturation_boost": self._chk_saturate.isChecked(),
        }
        out_text = self._output_lbl.text().strip()
        if out_text:
            result["output_dir"] = str(Path(out_text).parent)
        return result

    def validate(self, config: dict, batch_mode: bool = False) -> list:
        from gui.validation import ValidationIssue, count_files
        issues = []
        specs = config.get("composite_specs") or []
        if not specs:
            issues.append(ValidationIssue("error", S("validate.no_composite_specs")))
        if not batch_mode:
            out_base = config.get("output_dir", "").strip()
            input_path = str(Path(out_base) / "step05_wavelet_master") if out_base else ""
            if not count_files(input_path, "**/*.png", "**/*.PNG"):
                issues.append(ValidationIssue("error", S("validate.no_wavelet_png")))
            elif specs and input_path:
                for spec in specs:
                    for role in ("R", "G", "B", "L"):
                        fname = spec.get(role, "")
                        if fname and not count_files(input_path, f"**/*{fname}*.png", f"**/*{fname}*.PNG"):
                            issues.append(ValidationIssue(
                                "error",
                                S("validate.no_filter_png", f=fname, d=input_path),
                            ))
        return issues

    def output_paths(self) -> list[Path]:
        if self._output_dir is None:
            return []
        step_dir = self._output_dir / "step06_rgb_composite"
        if not step_dir.exists():
            return []
        return sorted(step_dir.rglob("*.png"))

    def set_output_dir(self, path: Path | str | None) -> None:
        self._output_dir = Path(path) if path else None
        step06_dir = self._output_dir / "step05_wavelet_master" if self._output_dir else None
        if hasattr(self, "_preview"):
            self._preview.set_input_dir(step06_dir)

    def on_show(self) -> None:
        if hasattr(self, "_preview"):
            self._preview.schedule_update(150)

    def refresh_after_run(self) -> None:
        if hasattr(self, "_preview"):
            self._preview.schedule_update(500)

    def _add_spec_row(self, spec: dict[str, str] | None = None) -> None:
        row = _SpecRow(self._available_filters, spec, parent=self._spec_container)
        row.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        row._btn_del.clicked.connect(lambda _, r=row: self._remove_spec_row(r))
        self._spec_rows.append(row)
        self._spec_vbox.addWidget(row)
        if hasattr(self, "_spec_radio_group"):
            self._spec_radio_group.addButton(row._radio)
            if len(self._spec_rows) == 1:
                row._radio.setChecked(True)
        if hasattr(self, "_preview"):
            for cb in row._combos.values():
                cb.currentIndexChanged.connect(self._on_params_changed)

    def _remove_spec_row(self, row: _SpecRow) -> None:
        was_checked = row._radio.isChecked()
        if hasattr(self, "_spec_radio_group"):
            self._spec_radio_group.removeButton(row._radio)
        if row in self._spec_rows:
            self._spec_rows.remove(row)
        row.setParent(None)
        row.deleteLater()
        if was_checked and self._spec_rows:
            self._spec_rows[0]._radio.setChecked(True)
            self._on_params_changed()

    def _refresh_filter_options(self, filters: list[str]) -> None:
        self._available_filters = filters
        for row in self._spec_rows:
            row.update_filters(filters)

    def _on_params_changed(self) -> None:
        if not hasattr(self, "_preview"):
            return
        self._preview.set_stretch_enabled(self._chk_stretch.isChecked())
        self._preview.set_saturate_enabled(self._chk_saturate.isChecked())
        self._sync_preview_spec()
        self._preview.schedule_update()

    def _sync_preview_spec(self) -> None:
        if not hasattr(self, "_preview") or not self._spec_rows:
            return
        active = next(
            (r for r in self._spec_rows if r._radio.isChecked()),
            self._spec_rows[0],
        )
        spec = active.to_dict()
        self._preview.set_spec(
            r_filter=spec.get("R", "R"),
            g_filter=spec.get("G", "G"),
            b_filter=spec.get("B", "B"),
            l_filter=spec.get("L", ""),
        )


# ══════════════════════════════════════════════════════════════════════════════
# Color correction sub-widget
# ══════════════════════════════════════════════════════════════════════════════

# ── Image helpers ──────────────────────────────────────────────────────────────

def _find_color_png(step06_dir: Path) -> Optional[Path]:
    for pattern in ("COLOR_master.png", "RGB_master.png"):
        hits = sorted(step06_dir.rglob(pattern))
        if hits:
            return hits[0]
    hits = sorted(step06_dir.rglob("*.png"))
    return hits[0] if hits else None


def _img_to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    lo = float(np.percentile(img, 0.0))
    hi = float(np.percentile(img, 99.0))
    if hi <= lo:
        hi = lo + 1e-6
    scale = (255 * 0.8) / (hi - lo)
    return np.clip((img - lo) * scale, 0, 255).astype(np.uint8)


def _numpy_to_pixmap(arr_u8: np.ndarray, size: int = _PANEL_SIZE) -> QPixmap:
    h, w = arr_u8.shape[:2]
    from PySide6.QtGui import QImage
    rgb = np.ascontiguousarray(arr_u8[:, :, :3])
    qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
    px = QPixmap.fromImage(qimg.copy())
    if max(w, h) > size:
        px = px.scaled(
            size, size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
    return px


def _make_img_label_cc() -> QLabel:
    lbl = QLabel()
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lbl.setStyleSheet(_PANEL_STYLE)
    lbl.setFixedSize(_PANEL_SIZE, _PANEL_SIZE)
    lbl.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    return lbl


# ── Preview worker ─────────────────────────────────────────────────────────────

class _CCPreviewWorker(QObject):
    """Background: load Step 6 PNG → auto_color_correct → emit before/after + params."""

    # orig_bytes, corr_bytes, h, w, r_gain, b_gain, r_sx, r_sy, b_sx, b_sy
    done  = Signal(bytes, bytes, int, int, float, float, float, float, float, float)
    error = Signal(str)

    def __init__(self, png_path: Path, stretch: bool = True, saturate: bool = True) -> None:
        super().__init__()
        self._path     = png_path
        self._stretch  = stretch
        self._saturate = saturate

    @Slot()
    def run(self) -> None:
        try:
            from pipeline.modules import image_io, composite as comp_mod
            from pipeline.steps.step06_rgb_composite import _auto_color_correct

            orig = image_io.read_png(self._path)
            if orig.ndim == 2:
                orig = np.stack([orig] * 3, axis=2)

            corrected, params = _auto_color_correct(orig)

            if self._stretch:
                lo = float(np.percentile(corrected, 0.0))
                hi = float(np.percentile(corrected, 99.0))
                if hi > lo:
                    corrected = np.clip((corrected - lo) * (0.8 / (hi - lo)), 0.0, 1.0).astype(np.float32)
            if self._saturate:
                corrected = comp_mod.auto_saturate(corrected)

            orig_u8 = (orig * 255).clip(0, 255).astype(np.uint8)
            corr_u8 = (corrected * 255).clip(0, 255).astype(np.uint8)
            h, w = orig_u8.shape[:2]

            self.done.emit(
                bytes(np.ascontiguousarray(orig_u8)),
                bytes(np.ascontiguousarray(corr_u8)),
                h, w,
                params["r_gain"], params["b_gain"],
                params["r_shift_x"], params["r_shift_y"],
                params["b_shift_x"], params["b_shift_y"],
            )
        except Exception as exc:
            self.error.emit(str(exc))


# ── Graph ──────────────────────────────────────────────────────────────────────

class _CCGraphWidget(QWidget):
    """QPainter graph: left = R/G/B gain bars, right = CA shift arrows.

    Updated only after the preview worker completes (auto-calc results).
    """

    _H = 150

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(self._H)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumWidth(200)

        self._r_gain  = 1.0
        self._b_gain  = 1.0
        self._r_sx    = 0.0
        self._r_sy    = 0.0
        self._b_sx    = 0.0
        self._b_sy    = 0.0
        self._has_data = False

    def update_data(
        self,
        r_gain: float, b_gain: float,
        r_sx: float, r_sy: float,
        b_sx: float, b_sy: float,
    ) -> None:
        self._r_gain = r_gain
        self._b_gain = b_gain
        self._r_sx = r_sx
        self._r_sy = r_sy
        self._b_sx = b_sx
        self._b_sy = b_sy
        self._has_data = True
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        W = self.width()
        H = self._H

        painter.fillRect(0, 0, W, H, QColor("#1a1a1a"))
        painter.setPen(QPen(QColor("#444"), 1))
        painter.drawRect(0, 0, W - 1, H - 1)

        if not self._has_data:
            painter.setPen(QColor("#555"))
            painter.setFont(QFont("Arial", 9))
            painter.drawText(
                0, 0, W, H,
                Qt.AlignmentFlag.AlignCenter,
                S("step06.canvas.no_data"),
            )
            return

        mid = W // 2
        self._draw_gain_bars(painter, 8, 6, mid - 16, H - 12)
        self._draw_shift_diagram(painter, mid + 4, 6, W - mid - 12, H - 12)

    def _draw_gain_bars(self, painter: QPainter, ax: int, ay: int, aw: int, ah: int) -> None:
        painter.setPen(QColor("#888"))
        painter.setFont(QFont("Arial", 8))
        painter.drawText(ax, ay, aw, 14, Qt.AlignmentFlag.AlignLeft, S("step06.canvas.gain"))

        chart_y = ay + 16
        chart_h = ah - 36
        mx = max(self._r_gain, 1.0, self._b_gain, 1e-6)

        bars = [
            (self._r_gain, QColor("#e05555"), "R"),
            (1.0,          QColor("#55c055"), "G"),
            (self._b_gain, QColor("#5588e0"), "B"),
        ]
        bar_w = (aw - 4 * 4) // 3

        for i, (val, clr, lbl) in enumerate(bars):
            bar_h = max(1, int((val / mx) * chart_h))
            bx = ax + 4 + i * (bar_w + 4)
            by = chart_y + chart_h - bar_h
            painter.fillRect(bx, by, bar_w, bar_h, clr)

            painter.setPen(QColor("#ccc"))
            painter.setFont(QFont("Arial", 7))
            painter.drawText(bx, chart_y + chart_h + 2, bar_w, 12,
                             Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, lbl)
            painter.drawText(bx, chart_y + chart_h + 12, bar_w, 12,
                             Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                             f"{val:.2f}")

        # G reference line
        g_line_y = chart_y + chart_h - int((1.0 / mx) * chart_h)
        painter.setPen(QPen(QColor("#4da6ff"), 1, Qt.PenStyle.DashLine))
        painter.drawLine(ax + 2, g_line_y, ax + aw - 2, g_line_y)

    def _draw_shift_diagram(self, painter: QPainter, ax: int, ay: int, aw: int, ah: int) -> None:
        painter.setPen(QColor("#888"))
        painter.setFont(QFont("Arial", 8))
        painter.drawText(ax, ay, aw, 14, Qt.AlignmentFlag.AlignLeft, S("step06.canvas.shift"))

        cx = ax + aw // 2
        cy = ay + 16 + (ah - 36) // 2
        r  = min(aw, ah - 36) // 3

        # G reference circle
        painter.setPen(QPen(QColor("#55c055"), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(cx - r, cy - r, r * 2, r * 2)
        painter.setPen(QColor("#55c055"))
        painter.setFont(QFont("Arial", 8, QFont.Weight.Bold))
        painter.drawText(cx - 4, cy + 4, "G")

        scale = min(r * 0.8 / 3.0, 12.0)

        def _arrow(dx: float, dy: float, clr: QColor, label: str) -> None:
            tx = int(cx + dx * scale)
            ty = int(cy + dy * scale)
            painter.setPen(QPen(clr, 2))
            painter.drawLine(cx, cy, tx, ty)
            painter.setBrush(clr)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(tx - 4, ty - 4, 8, 8)
            painter.setPen(clr)
            painter.setFont(QFont("Arial", 7))
            painter.drawText(tx + (6 if dx >= 0 else -14), ty + (4 if dy >= 0 else -4), label)

        # Arrows show displacement of R/B relative to G (opposite of correction)
        _arrow(-self._r_sx, -self._r_sy, QColor("#e05555"), "R")
        _arrow(-self._b_sx, -self._b_sy, QColor("#5588e0"), "B")

        # Numeric values
        painter.setPen(QColor("#888"))
        painter.setFont(QFont("Arial", 7))
        painter.drawText(
            ax, ay + ah - 20, aw, 20,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"R({self._r_sx:+.2f},{self._r_sy:+.2f})  B({self._b_sx:+.2f},{self._b_sy:+.2f})",
        )


# ── Color sub-widget ───────────────────────────────────────────────────────────

class _Step06ColorWidget(QWidget):
    """Color camera Step 7 panel.

    Shows a before/after preview using the same auto WB + CA algorithm that
    the pipeline will apply per window.  No manual parameter entry — the
    pipeline handles each window independently.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._output_dir: Path | None = None
        self._step06_dir: Path | None = None

        self._running = False
        self._pending = False
        self._thread: Optional[QThread] = None
        self._worker: Optional[_CCPreviewWorker] = None

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._do_preview_update)

        self._build_ui()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setSpacing(12)
        root.setContentsMargins(0, 0, 0, 0)

        # ── Left: form controls ────────────────────────────────────────────
        left = QVBoxLayout()
        left.setSpacing(10)

        folder_form = QWidget()
        folder_form.setStyleSheet("background: transparent;")
        fl = QFormLayout(folder_form)
        fl.setSpacing(6)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self._input_lbl = QLineEdit()
        self._input_lbl.setReadOnly(True)
        self._input_lbl.setStyleSheet(_READONLY_STYLE)
        fl.addRow(QLabel(S("step06.input_dir")), self._input_lbl)

        self._output_lbl = QLineEdit()
        self._output_lbl.setReadOnly(True)
        self._output_lbl.setStyleSheet(_READONLY_STYLE)
        fl.addRow(QLabel(S("step06.output_dir")), self._output_lbl)

        left.addWidget(folder_form)

        info = QLabel(S("step06.cc.info"))
        info.setStyleSheet(_INFO_STYLE)
        info.setWordWrap(True)
        left.addWidget(info)

        btn_row = QHBoxLayout()
        self._btn_refresh = QPushButton(S("step06.cc.btn_refresh"))
        self._btn_refresh.setStyleSheet(_BTN_REFRESH)
        self._btn_refresh.setToolTip(S("step06.cc.btn_refresh.tooltip"))
        self._btn_refresh.clicked.connect(lambda: self.schedule_update(0))
        btn_row.addWidget(self._btn_refresh)
        btn_row.addStretch()
        left.addLayout(btn_row)

        chk_row = QHBoxLayout()
        chk_row.setContentsMargins(0, 4, 0, 0)
        chk_row.setSpacing(16)
        self._chk_global_normalize = QCheckBox(S("step06.global_normalize"))
        self._chk_stretch  = QCheckBox(S("step06.stretch_enabled"))
        self._chk_saturate = QCheckBox(S("step06.saturation_boost"))
        self._chk_global_normalize.setStyleSheet(_CHK_STYLE)
        self._chk_stretch.setStyleSheet(_CHK_STYLE)
        self._chk_saturate.setStyleSheet(_CHK_STYLE)
        self._chk_global_normalize.setChecked(True)
        self._chk_global_normalize.setToolTip(S("step06.global_normalize.tooltip"))
        self._chk_stretch.setChecked(False)
        self._chk_saturate.setChecked(True)
        self._chk_stretch.toggled.connect(lambda: self.schedule_update(0))
        self._chk_saturate.toggled.connect(lambda: self.schedule_update(0))
        chk_row.addWidget(self._chk_global_normalize)
        chk_row.addWidget(self._chk_stretch)
        chk_row.addWidget(self._chk_saturate)
        chk_row.addStretch()
        left.addLayout(chk_row)

        left.addStretch()
        root.addLayout(left, 1)

        # ── Right: preview (status + before/after images + graph) ─────────
        right = QVBoxLayout()
        right.setSpacing(4)
        right.setContentsMargins(0, 0, 0, 0)

        self._status_lbl = QLabel(S("step06.cc.status_init"))
        self._status_lbl.setStyleSheet(_STATUS_STYLE)
        self._status_lbl.setWordWrap(True)
        right.addWidget(self._status_lbl)

        panels_row = QHBoxLayout()
        panels_row.setSpacing(8)
        panels_row.setContentsMargins(0, 0, 0, 0)

        self._before_lbl = _make_img_label_cc()
        self._after_lbl  = _make_img_label_cc()

        self._cap_before_lbl = QLabel(S("step06.cc.cap_before"))
        self._cap_after_lbl  = QLabel(S("step06.cc.cap_after"))
        for img_lbl, cap_lbl in (
            (self._before_lbl, self._cap_before_lbl),
            (self._after_lbl,  self._cap_after_lbl),
        ):
            col = QVBoxLayout()
            col.setSpacing(2)
            col.addWidget(img_lbl)
            cap_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            cap_lbl.setStyleSheet(_CAP_STYLE)
            col.addWidget(cap_lbl)
            panels_row.addLayout(col)

        right.addLayout(panels_row)

        self._graph = _CCGraphWidget()
        right.addWidget(self._graph)

        right.addStretch()
        root.addLayout(right, 0)

    # ── Public interface ──────────────────────────────────────────────────────

    def retranslate(self) -> None:
        self._btn_refresh.setText(S("step06.cc.btn_refresh"))
        self._btn_refresh.setToolTip(S("step06.cc.btn_refresh.tooltip"))
        self._status_lbl.setText(S("step06.cc.status_init"))
        self._cap_before_lbl.setText(S("step06.cc.cap_before"))
        self._cap_after_lbl.setText(S("step06.cc.cap_after"))
        self._chk_global_normalize.setText(S("step06.global_normalize"))
        self._chk_global_normalize.setToolTip(S("step06.global_normalize.tooltip"))
        self._chk_stretch.setText(S("step06.stretch_enabled"))
        self._chk_saturate.setText(S("step06.saturation_boost"))

    def load_session(self, data: dict[str, Any]) -> None:
        out = data.get("output_dir", "")
        if out:
            p = Path(out)
            self._input_lbl.setText(str(p / "step05_wavelet_master"))
            self._output_lbl.setText(str(p / "step06_rgb_composite"))
            self._step06_dir = p / "step05_wavelet_master"
        self._chk_global_normalize.setChecked(bool(data.get("global_normalize", True)))
        self._chk_stretch.setChecked(bool(data.get("stretch_enabled", False)))
        self._chk_saturate.setChecked(bool(data.get("saturation_boost", True)))

    def get_config_updates(self) -> dict[str, Any]:
        return {
            "global_normalize": self._chk_global_normalize.isChecked(),
            "stretch_enabled":  self._chk_stretch.isChecked(),
            "saturation_boost": self._chk_saturate.isChecked(),
        }

    def output_paths(self) -> list[Path]:
        if self._output_dir is None:
            return []
        step_dir = self._output_dir / "step06_rgb_composite"
        if not step_dir.exists():
            return []
        return sorted(step_dir.rglob("*.png"))

    def set_output_dir(self, path: Path | str | None) -> None:
        self._output_dir = Path(path) if path else None
        if self._output_dir:
            self._step06_dir = self._output_dir / "step05_wavelet_master"
        else:
            self._step06_dir = None

    def on_show(self) -> None:
        self.schedule_update(150)

    def refresh_after_run(self) -> None:
        self.schedule_update(500)

    def schedule_update(self, delay: int = 400) -> None:
        if self._step06_dir is None:
            return
        self._timer.start(delay)

    # ── Qt events ─────────────────────────────────────────────────────────────

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._step06_dir is not None and not self._running:
            self.schedule_update(150)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _do_preview_update(self) -> None:
        if self._step06_dir is None:
            return
        if self._running:
            self._pending = True
            return

        png = _find_color_png(self._step06_dir)
        if png is None:
            self._status_lbl.setText(S("preview.no_png_short", d=self._step06_dir))
            return

        self._running = True
        self._pending = False
        self._btn_refresh.setEnabled(False)
        self._status_lbl.setText(S("preview.auto_calc", f=png.name))

        worker = _CCPreviewWorker(png, stretch=self._chk_stretch.isChecked(), saturate=self._chk_saturate.isChecked())
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

    @Slot(bytes, bytes, int, int, float, float, float, float, float, float)
    def _on_done(
        self,
        orig_bytes: bytes,
        corr_bytes: bytes,
        h: int, w: int,
        r_gain: float, b_gain: float,
        r_sx: float, r_sy: float,
        b_sx: float, b_sy: float,
    ) -> None:
        self._running = False
        self._thread  = None
        self._worker  = None
        self._btn_refresh.setEnabled(True)

        orig_u8 = np.frombuffer(orig_bytes, dtype=np.uint8).reshape(h, w, 3)
        corr_u8 = np.frombuffer(corr_bytes, dtype=np.uint8).reshape(h, w, 3)

        self._before_lbl.setPixmap(_numpy_to_pixmap(orig_u8))
        self._after_lbl.setPixmap(_numpy_to_pixmap(corr_u8))

        self._status_lbl.setText(
            S("step06.cc.done",
              wh=f"{w}×{h}", rg=r_gain, bg=b_gain,
              rsx=r_sx, rsy=r_sy, bsx=b_sx, bsy=b_sy)
        )
        self._graph.update_data(r_gain, b_gain, r_sx, r_sy, b_sx, b_sy)

        if self._pending:
            self._pending = False
            self.schedule_update(200)

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        self._running = False
        self._thread  = None
        self._worker  = None
        self._btn_refresh.setEnabled(True)
        self._status_lbl.setText(S("preview.error", msg=msg))
        if self._pending:
            self._pending = False


# ── Wrapper panel ──────────────────────────────────────────────────────────────

class Step06Panel(BasePanel):
    STEP_ID   = "06"
    TITLE_KEY = "step06.title"
    DESC_KEY  = "step06.desc"
    OPTIONAL  = False

    def __init__(self, parent: QWidget | None = None) -> None:
        self._output_dir: Path | None = None
        self._is_color: bool = False
        super().__init__(parent)

    def build_form(self) -> None:
        self._sub_stack = QStackedWidget()
        self._mono_widget  = _Step06MonoWidget()
        self._color_widget = _Step06ColorWidget()
        self._sub_stack.addWidget(self._mono_widget)   # 0
        self._sub_stack.addWidget(self._color_widget)  # 1

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

    def retranslate(self) -> None:
        self._mono_widget.retranslate()
        self._color_widget.retranslate()

    def output_paths(self) -> list[Path]:
        if self._is_color:
            return self._color_widget.output_paths()
        return self._mono_widget.output_paths()

    def set_output_dir(self, path: Path | str) -> None:
        self._output_dir = Path(path) if path else None
        if not self._is_color:
            self._mono_widget.set_output_dir(self._output_dir)
        self._color_widget.set_output_dir(self._output_dir)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._is_color:
            self._color_widget.on_show()
        else:
            self._mono_widget.on_show()

    def validate(self, config: dict, batch_mode: bool = False) -> list:
        if self._is_color:
            return []  # color mode: composite_specs 불필요, auto-correction 전용
        return self._mono_widget.validate(config, batch_mode)

    def refresh_after_run(self) -> None:
        super().refresh_after_run()
        if self._is_color:
            self._color_widget.refresh_after_run()
        else:
            self._mono_widget.refresh_after_run()
