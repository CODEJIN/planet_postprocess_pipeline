"""Global settings panel — not a step panel, does NOT extend BasePanel."""
from __future__ import annotations

import multiprocessing as _mp
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from gui.i18n import S

# Planet preset data: name → (target, horizons_id, rotation_period_hours)
_PLANET_PRESETS: dict[str, tuple[str, str, float]] = {
    "Jupiter": ("Jup", "599",   9.9281),
    "Saturn":  ("Sat", "699",  10.56),
    "Mars":    ("Mar", "499",  24.6229),
    "Uranus":  ("Ura", "799",  17.24),
    "Neptune": ("Nep", "899",  16.11),
    "Mercury": ("Mer", "199", 1407.6),
    "Venus":   ("Ven", "299", 5832.5),
    "Custom":  ("",    "",      9.9281),
}

_PANEL_BG   = "#252526"
_TEXT_COLOR = "#d4d4d4"
_INPUT_STYLE = (
    "QLineEdit { background: #3c3c3c; color: #d4d4d4; border: 1px solid #555;"
    " border-radius: 3px; padding: 3px 6px; }"
    "QLineEdit:focus { border-color: #4da6ff; }"
)
_SPINBOX_STYLE = (
    "QDoubleSpinBox { background: #3c3c3c; color: #d4d4d4; border: 1px solid #555;"
    " border-radius: 3px; padding: 3px 6px; }"
    "QDoubleSpinBox:focus { border-color: #4da6ff; }"
)
_COMBO_STYLE = (
    "QComboBox { background: #3c3c3c; color: #d4d4d4; border: 1px solid #555;"
    " border-radius: 3px; padding: 3px 6px; }"
    "QComboBox::drop-down { border: none; }"
    "QComboBox QAbstractItemView { background: #3c3c3c; color: #d4d4d4; }"
)
_BTN_SAVE = (
    "QPushButton { background: #2d6a4f; color: white; border-radius: 5px;"
    " font-weight: bold; padding: 6px 20px; }"
    "QPushButton:hover { background: #40916c; }"
)
_BTN_RESET = (
    "QPushButton { background: #7f1d1d; color: white; border-radius: 5px;"
    " font-weight: bold; padding: 6px 20px; }"
    "QPushButton:hover { background: #b91c1c; }"
)



class SettingsPanel(QWidget):
    """Global settings panel shown at the top of the left sidebar flow."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._mono_filters_backup = "IR,R,G,B,CH4"
        self.setStyleSheet(f"background: {_PANEL_BG};")
        self._build_ui()

    # ── Construction ──────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Header ─────────────────────────────────────────────────────────
        header_widget = QWidget()
        header_widget.setStyleSheet(f"background: {_PANEL_BG};")
        header_layout = QVBoxLayout(header_widget)
        header_layout.setContentsMargins(16, 14, 16, 8)
        header_layout.setSpacing(4)

        title = QLabel(S("settings.title"))
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setStyleSheet(f"color: #e8e8e8;")
        header_layout.addWidget(title)

        desc = QLabel(S("settings.desc"))
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #999; font-size: 11px;")
        header_layout.addWidget(desc)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #444;")
        header_layout.addWidget(line)

        root.addWidget(header_widget)

        # ── Scrollable form area ────────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        scroll.setStyleSheet("QScrollArea { background: transparent; }")

        form_container = QWidget()
        form_container.setStyleSheet(f"background: {_PANEL_BG};")
        fl = QFormLayout(form_container)
        fl.setContentsMargins(16, 12, 16, 12)
        fl.setSpacing(10)
        fl.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._form = fl

        def _lbl(text: str, tip: str) -> QLabel:
            l = QLabel(text)
            l.setToolTip(tip)
            return l

        # Planet preset
        self._planet_combo = QComboBox()
        self._planet_combo.setStyleSheet(_COMBO_STYLE)
        self._planet_combo.setToolTip(S("settings.planet.tooltip"))
        for pname in _PLANET_PRESETS:
            self._planet_combo.addItem(
                S(f"settings.planet.{pname.lower()}") if pname != "Custom" else S("settings.planet.custom"),
                pname,
            )
        self._planet_combo.currentIndexChanged.connect(self._on_planet_changed)
        fl.addRow(_lbl(S("settings.planet"), S("settings.planet.tooltip")), self._planet_combo)

        # Target
        self._target = QLineEdit()
        self._target.setStyleSheet(_INPUT_STYLE)
        self._target.setPlaceholderText("Jup")
        self._target.setToolTip(S("settings.target.tooltip"))
        fl.addRow(_lbl(S("settings.target"), S("settings.target.tooltip")), self._target)

        # Horizons ID
        self._horizons_id = QLineEdit()
        self._horizons_id.setStyleSheet(_INPUT_STYLE)
        self._horizons_id.setPlaceholderText("599")
        self._horizons_id.setToolTip(S("settings.horizons_id.tooltip"))
        fl.addRow(_lbl(S("settings.horizons_id"), S("settings.horizons_id.tooltip")), self._horizons_id)

        # Rotation period
        self._rotation_period = QDoubleSpinBox()
        self._rotation_period.setStyleSheet(_SPINBOX_STYLE)
        self._rotation_period.setRange(0.1, 6000.0)
        self._rotation_period.setDecimals(4)
        self._rotation_period.setSingleStep(0.01)
        self._rotation_period.setValue(9.9281)
        self._rotation_period.setToolTip(S("settings.rotation_period.tooltip"))
        fl.addRow(_lbl(S("settings.rotation_period"), S("settings.rotation_period.tooltip")), self._rotation_period)

        # Camera mode (above filters so the user sees why filters is disabled)
        mode_widget = QWidget()
        mode_widget.setStyleSheet("background: transparent;")
        mode_layout = QHBoxLayout(mode_widget)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(16)
        self._radio_mono  = QRadioButton(S("settings.camera.mono"))
        self._radio_color = QRadioButton(S("settings.camera.color"))
        self._radio_mono.setStyleSheet(f"color: {_TEXT_COLOR};")
        self._radio_color.setStyleSheet(f"color: {_TEXT_COLOR};")
        self._radio_mono.setChecked(True)
        self._radio_mono.setToolTip(S("settings.camera_mode.tooltip"))
        self._radio_color.setToolTip(S("settings.camera_mode.tooltip"))
        self._camera_group = QButtonGroup(self)
        self._camera_group.addButton(self._radio_mono,  0)
        self._camera_group.addButton(self._radio_color, 1)
        mode_layout.addWidget(self._radio_mono)
        mode_layout.addWidget(self._radio_color)
        mode_layout.addStretch()
        fl.addRow(_lbl(S("settings.camera_mode"), S("settings.camera_mode.tooltip")), mode_widget)
        # Connect AFTER adding to layout so _on_camera_changed can access self._filters
        self._radio_mono.toggled.connect(self._on_camera_changed)

        # Filters (below camera mode; auto-managed when color is selected)
        self._filters = QLineEdit()
        self._filters.setStyleSheet(_INPUT_STYLE)
        self._filters.setPlaceholderText("IR,R,G,B,CH4")
        self._filters.setToolTip(S("settings.filters.tooltip"))
        self._filters_lbl = _lbl(S("settings.filters"), S("settings.filters.tooltip"))
        fl.addRow(self._filters_lbl, self._filters)

        # Language
        self._lang_combo = QComboBox()
        self._lang_combo.setStyleSheet(_COMBO_STYLE)
        self._lang_combo.setToolTip(S("settings.language.tooltip"))
        self._lang_combo.addItem("한국어", "ko")
        self._lang_combo.addItem("English", "en")
        fl.addRow(_lbl(S("settings.language"), S("settings.language.tooltip")), self._lang_combo)

        # ── Performance ───────────────────────────────────────────────────────
        _sep2 = QFrame()
        _sep2.setFrameShape(QFrame.Shape.HLine)
        _sep2.setStyleSheet("color: #444;")
        fl.addRow(_sep2)

        _cpu_n = _mp.cpu_count() or 1
        _tip_wk = S("settings.max_workers.tooltip").format(n=_cpu_n)
        self._max_workers = QSpinBox()
        self._max_workers.setStyleSheet(
            "QSpinBox { background: #3c3c3c; color: #d4d4d4; border: 1px solid #555;"
            " border-radius: 3px; padding: 3px 6px; }"
            "QSpinBox:focus { border-color: #4da6ff; }"
        )
        self._max_workers.setRange(0, _cpu_n)
        self._max_workers.setValue(0)
        self._max_workers.setFixedWidth(100)
        self._max_workers.setSpecialValueText(f"자동 ({_cpu_n}코어)")
        self._max_workers.setToolTip(_tip_wk)
        wk_row = QHBoxLayout()
        wk_row.setSpacing(4)
        wk_row.addWidget(self._max_workers)
        wk_row.addStretch()
        fl.addRow(_lbl(S("settings.max_workers"), _tip_wk), wk_row)

        scroll.setWidget(form_container)
        root.addWidget(scroll, 1)

        # ── Save / Reset buttons ─────────────────────────────────────────────
        btn_widget = QWidget()
        btn_widget.setStyleSheet(f"background: {_PANEL_BG}; border-top: 1px solid #444;")
        btn_layout = QHBoxLayout(btn_widget)
        btn_layout.setContentsMargins(16, 8, 16, 12)
        self._btn_reset = QPushButton(S("btn.reset_session"))
        self._btn_reset.setStyleSheet(_BTN_RESET)
        self._btn_reset.setFixedHeight(34)
        self._btn_reset.setToolTip(S("btn.reset_session.tooltip"))
        self._btn_save = QPushButton(S("settings.save"))
        self._btn_save.setStyleSheet(_BTN_SAVE)
        self._btn_save.setFixedHeight(34)
        btn_layout.addStretch()
        btn_layout.addWidget(self._btn_reset)
        btn_layout.addSpacing(8)
        btn_layout.addWidget(self._btn_save)
        root.addWidget(btn_widget)

        # Apply initial preset
        self._on_planet_changed(0)

    # ── Camera mode slot ──────────────────────────────────────────────────────

    def _on_camera_changed(self, mono_checked: bool) -> None:
        """Toggle the filters field when the user switches camera mode."""
        if mono_checked:
            self._filters.setEnabled(True)
            self._filters.setStyleSheet(_INPUT_STYLE)
            self._filters_lbl.setEnabled(True)
            if self._filters.text().strip() == "COLOR":
                self._filters.setText(self._mono_filters_backup)
        else:
            current = self._filters.text().strip()
            if current != "COLOR":
                self._mono_filters_backup = current
            self._filters.setText("COLOR")
            self._filters.setEnabled(False)
            self._filters_lbl.setEnabled(False)
            self._filters.setStyleSheet(
                "QLineEdit { background: #2a2a2a; color: #666; border: 1px solid #3a3a3a;"
                " border-radius: 3px; padding: 3px 6px; }"
            )

    # ── Planet preset slot ────────────────────────────────────────────────────

    def _on_planet_changed(self, index: int) -> None:
        pname = self._planet_combo.itemData(index)
        if pname not in _PLANET_PRESETS:
            return
        target, horizons_id, period = _PLANET_PRESETS[pname]
        if pname != "Custom":
            self._target.setText(target)
            self._horizons_id.setText(horizons_id)
            self._rotation_period.setValue(period)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_session_values(self) -> dict[str, Any]:
        """Return a dict suitable for merging into session data."""
        camera_mode = "mono" if self._radio_mono.isChecked() else "color"
        planet_idx  = self._planet_combo.currentIndex()
        planet      = self._planet_combo.itemData(planet_idx) or "Jupiter"
        lang_idx    = self._lang_combo.currentIndex()
        language    = self._lang_combo.itemData(lang_idx) or "ko"
        return {
            "planet":             planet,
            "target":             self._target.text().strip(),
            "horizons_id":        self._horizons_id.text().strip(),
            "rotation_period":    self._rotation_period.value(),
            "filters":            self._filters.text().strip(),
            "camera_mode":        camera_mode,
            "language":           language,
            "global_max_workers": self._max_workers.value(),
        }

    def load_session(self, data: dict[str, Any]) -> None:
        """Populate controls from *data* (session dict)."""
        planet = data.get("planet", "Jupiter")
        for i in range(self._planet_combo.count()):
            if self._planet_combo.itemData(i) == planet:
                self._planet_combo.setCurrentIndex(i)
                break

        # Set target/horizons/period AFTER combo so preset doesn't overwrite
        self._target.setText(data.get("target", "Jup"))
        self._horizons_id.setText(data.get("horizons_id", "599"))
        self._rotation_period.setValue(float(data.get("rotation_period", 9.9281)))
        self._filters.setText(data.get("filters", "IR,R,G,B,CH4"))

        camera_mode = data.get("camera_mode", "mono")
        if camera_mode == "color":
            self._radio_color.setChecked(True)
        else:
            self._radio_mono.setChecked(True)

        lang = data.get("language", "ko")
        for i in range(self._lang_combo.count()):
            if self._lang_combo.itemData(i) == lang:
                self._lang_combo.setCurrentIndex(i)
                break

        self._max_workers.setValue(int(data.get("global_max_workers", 0)))

    def retranslate(self) -> None:
        """Update widget texts after a runtime language change."""
        self._btn_reset.setText(S("btn.reset_session"))
        self._btn_reset.setToolTip(S("btn.reset_session.tooltip"))
