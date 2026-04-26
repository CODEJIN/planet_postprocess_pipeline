"""Global settings panel — not a step panel, does NOT extend BasePanel."""
from __future__ import annotations

import multiprocessing as _mp
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from gui.i18n import S
from gui import profile_manager

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
_BTN_SMALL_SAVE = (
    "QPushButton { background: #2d6a4f; color: white; border-radius: 4px;"
    " font-size: 11px; padding: 3px 10px; border: none; }"
    "QPushButton:hover { background: #40916c; }"
    "QPushButton:disabled { background: #2a3a33; color: #666; border: none; }"
)
_BTN_SMALL = (
    "QPushButton { background: #3c3c3c; color: #d4d4d4; border: 1px solid #555;"
    " border-radius: 4px; font-size: 11px; padding: 3px 10px; }"
    "QPushButton:hover { background: #4a4a4a; }"
    "QPushButton:disabled { color: #555; background: #2d2d2d; border-color: #444; }"
)
_BTN_SMALL_DELETE = (
    "QPushButton { background: #5a2020; color: #d4d4d4; border-radius: 4px;"
    " font-size: 11px; padding: 3px 10px; border: none; }"
    "QPushButton:hover { background: #7f1d1d; }"
    "QPushButton:disabled { background: #2d2020; color: #666; border: none; }"
)



class SettingsPanel(QWidget):
    """Global settings panel shown at the top of the left sidebar flow."""

    # Emitted when user selects a profile from the dropdown (empty string = Unsaved)
    profile_selected          = Signal(str)
    # Emitted when user clicks the profile Save button
    profile_save_requested    = Signal()
    # Emitted when user names and confirms a new profile
    profile_save_as_requested = Signal(str)
    # Emitted when user confirms profile deletion
    profile_delete_requested  = Signal()

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

        # ── Profile section ─────────────────────────────────────────────────
        profile_widget = QWidget()
        profile_widget.setStyleSheet(
            f"background: {_PANEL_BG}; border-bottom: 1px solid #3a3a3a;"
        )
        profile_layout = QVBoxLayout(profile_widget)
        profile_layout.setContentsMargins(16, 10, 16, 10)
        profile_layout.setSpacing(6)

        profile_title = QLabel(S("profile.section_title"))
        profile_title.setStyleSheet("color: #888; font-size: 11px; font-weight: bold;")
        profile_layout.addWidget(profile_title)

        combo_row = QHBoxLayout()
        combo_row.setSpacing(6)

        self._profile_combo = QComboBox()
        self._profile_combo.setStyleSheet(_COMBO_STYLE)
        self._profile_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        combo_row.addWidget(self._profile_combo)

        self._btn_profile_save = QPushButton(S("profile.save"))
        self._btn_profile_save.setStyleSheet(_BTN_SMALL_SAVE)
        self._btn_profile_save.setFixedHeight(28)
        self._btn_profile_save.setToolTip(S("profile.save.tooltip"))
        combo_row.addWidget(self._btn_profile_save)

        self._btn_profile_save_as = QPushButton(S("profile.save_as"))
        self._btn_profile_save_as.setStyleSheet(_BTN_SMALL)
        self._btn_profile_save_as.setFixedHeight(28)
        self._btn_profile_save_as.setToolTip(S("profile.save_as.tooltip"))
        combo_row.addWidget(self._btn_profile_save_as)

        self._btn_profile_delete = QPushButton(S("profile.delete"))
        self._btn_profile_delete.setStyleSheet(_BTN_SMALL_DELETE)
        self._btn_profile_delete.setFixedHeight(28)
        self._btn_profile_delete.setToolTip(S("profile.delete.tooltip"))
        combo_row.addWidget(self._btn_profile_delete)

        profile_layout.addLayout(combo_row)

        self._profile_meta_lbl = QLabel("")
        self._profile_meta_lbl.setStyleSheet("color: #666; font-size: 10px;")
        profile_layout.addWidget(self._profile_meta_lbl)

        root.addWidget(profile_widget)

        # Connect profile buttons
        self._profile_combo.currentIndexChanged.connect(self._on_profile_combo_changed)
        self._btn_profile_save.clicked.connect(self.profile_save_requested)
        self._btn_profile_save_as.clicked.connect(self._on_profile_save_as_clicked)
        self._btn_profile_delete.clicked.connect(self._on_profile_delete_clicked)

        # Initialise combo with empty list; main_window calls refresh_profile_list()
        self._profile_combo.addItem(S("profile.unsaved"), "")
        self._btn_profile_save.setEnabled(False)
        self._btn_profile_delete.setEnabled(False)

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
        self._lang_combo.addItem(S("lang.korean"), "ko")
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
        self._max_workers.setSpecialValueText(S("settings.max_workers.auto", n=_cpu_n))
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

    # ── Profile slots ─────────────────────────────────────────────────────────

    def _on_profile_combo_changed(self, index: int) -> None:
        name = self._profile_combo.itemData(index) or ""
        self._btn_profile_save.setEnabled(bool(name))
        self._btn_profile_delete.setEnabled(bool(name))
        self._update_profile_meta(name)
        self.profile_selected.emit(name)

    def _on_profile_save_as_clicked(self) -> None:
        name, ok = QInputDialog.getText(
            self,
            S("profile.save_as.dialog_title"),
            S("profile.save_as.dialog_prompt"),
        )
        if ok and name.strip():
            self.profile_save_as_requested.emit(name.strip())

    def _on_profile_delete_clicked(self) -> None:
        name = self._profile_combo.currentData() or ""
        if not name:
            return
        reply = QMessageBox.question(
            self,
            S("profile.delete.confirm_title"),
            S("profile.delete.confirm_msg", name=name),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.profile_delete_requested.emit()

    def _update_profile_meta(self, name: str) -> None:
        if not name:
            self._profile_meta_lbl.setText("")
            return
        meta = profile_manager.profile_meta(name)
        planet = meta.get("planet", "")
        camera = meta.get("camera_mode", "mono")
        updated = meta.get("updated_at", "")[:16].replace("T", " ")
        camera_str = (
            S("settings.camera.mono") if camera == "mono" else S("settings.camera.color")
        )
        self._profile_meta_lbl.setText(
            f"{planet}  ·  {camera_str}  ·  {updated}" if planet else updated
        )

    # ── Profile public API ────────────────────────────────────────────────────

    def refresh_profile_list(self, names: list[str], active: str | None) -> None:
        """Repopulate the profile combo. Called from main_window after any profile change."""
        self._profile_combo.blockSignals(True)
        self._profile_combo.clear()
        self._profile_combo.addItem(S("profile.unsaved"), "")
        for n in names:
            self._profile_combo.addItem(n, n)

        idx = 0
        if active:
            for i in range(self._profile_combo.count()):
                if self._profile_combo.itemData(i) == active:
                    idx = i
                    break
        self._profile_combo.setCurrentIndex(idx)
        self._profile_combo.blockSignals(False)

        current_name = self._profile_combo.itemData(idx) or ""
        self._btn_profile_save.setEnabled(bool(current_name))
        self._btn_profile_delete.setEnabled(bool(current_name))
        self._update_profile_meta(current_name)

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
