"""Main application window for the planetary imaging pipeline GUI."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from gui import session
from gui import i18n
from gui import profile_manager
from gui.i18n import S
from gui.panels.settings_panel import SettingsPanel
from gui.panels.welcome_panel import WelcomePanel
from gui.panels.step01_panel import Step01Panel
from gui.panels.step02_panel import Step02Panel
from gui.panels.step03_panel import Step03Panel
from gui.panels.step04_panel import Step04Panel
from gui.panels.step05_panel import Step05Panel
from gui.panels.step06_panel import Step06Panel
from gui.panels.step07_panel import Step07Panel
from gui.panels.step08_panel import Step08Panel
from gui.panels.step09_panel import Step09Panel
from gui.panels.step10_panel import Step10Panel
from gui.step_runner import StepRunner
from gui.widgets.log_widget import LogWidget
from gui.widgets.step_item import StepItem
from pipeline.config import (
    PipelineConfig,
    LuckyStackConfig,
    PippConfig,
    WaveletConfig,
    QualityConfig,
    DerotationConfig,
    CompositeConfig,
    CompositeSpec,
    GifConfig,
    SummaryGridConfig,
)

# ── Dark theme stylesheet ──────────────────────────────────────────────────────

_ICONS_DIR = Path(__file__).parent / "icons"

# __CHECK_SVG__ is replaced at runtime with the actual path to check.svg
_DARK_STYLE_TEMPLATE = """
QMainWindow, QWidget {
    background-color: #1e1e1e;
    color: #d4d4d4;
}
QMenuBar {
    background-color: #2d2d2d;
    color: #d4d4d4;
    border-bottom: 1px solid #444;
}
QMenuBar::item:selected {
    background-color: #3c3c3c;
}
QMenu {
    background-color: #2d2d2d;
    color: #d4d4d4;
    border: 1px solid #555;
}
QMenu::item:selected {
    background-color: #3c3c3c;
}
QSplitter::handle {
    background-color: #444;
}
QScrollBar:vertical {
    background: #2d2d2d;
    width: 8px;
}
QScrollBar::handle:vertical {
    background: #555;
    border-radius: 4px;
    min-height: 20px;
}
QStatusBar {
    background: #252526;
    color: #888;
    border-top: 1px solid #444;
}
QPushButton {
    background: #3c3c3c;
    color: #d4d4d4;
    border: 1px solid #555;
    border-radius: 4px;
    padding: 4px 10px;
}
QPushButton:hover {
    background: #4a4a4a;
}
QPushButton:disabled {
    color: #666;
    background: #2d2d2d;
    border-color: #444;
}
QCheckBox {
    spacing: 6px;
    color: #d4d4d4;
}
QCheckBox::indicator {
    width: 14px;
    height: 14px;
    border: 2px solid #555;
    border-radius: 3px;
    background: #2a2a2a;
}
QCheckBox::indicator:hover {
    border-color: #888;
    background: #333;
}
QCheckBox::indicator:checked {
    border: 2px solid #4da6ff;
    background: #1d3a5a;
    image: url("__CHECK_SVG__");
}
QCheckBox::indicator:checked:hover {
    border-color: #66b8ff;
}
QRadioButton {
    spacing: 6px;
    color: #d4d4d4;
}
QRadioButton::indicator {
    width: 14px;
    height: 14px;
    border-radius: 7px;
    border: 2px solid #555;
    background: #2a2a2a;
}
QRadioButton::indicator:hover {
    border-color: #888;
    background: #333;
}
QRadioButton::indicator:checked {
    border: 2px solid #4da6ff;
    background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5,
        stop:0 #4da6ff, stop:0.38 #4da6ff, stop:0.42 #1d3a5a, stop:1 #1d3a5a);
}
QRadioButton::indicator:checked:hover {
    border-color: #66b8ff;
}
"""


def _build_dark_style() -> str:
    check_svg = (_ICONS_DIR / "check.svg").as_posix()
    return _DARK_STYLE_TEMPLATE.replace("__CHECK_SVG__", check_svg)


DARK_STYLE = _build_dark_style()

# ── Step definitions ───────────────────────────────────────────────────────────

_STEP_DEFS = [
    # (step_id, i18n_key, optional)
    ("01", "sidebar.step01", True),
    ("02", "sidebar.step02", True),
    ("03", "sidebar.step03", False),
    ("04", "sidebar.step04", False),
    ("05", "sidebar.step05", False),
    ("06", "sidebar.step06", False),
    ("07", "sidebar.step07", True),
    # separator before optional final steps
    ("08", "sidebar.step08", True),
    ("09", "sidebar.step09", True),
    ("10", "sidebar.step10", True),
]

# Which step IDs get a separator _before_ them in the sidebar
_SEPARATOR_BEFORE = {"03", "07"}


class MainWindow(QMainWindow):
    """Top-level application window."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(S("app.title"))
        self.resize(1440, 1010)
        self.setStyleSheet(DARK_STYLE)

        self._session_data: dict[str, Any] = {}
        self._results:      dict[str, Any] = {}
        self._runner: StepRunner | None = None
        self._step_items:   dict[str, StepItem]  = {}
        self._enabled_steps: dict[str, bool]      = {}

        self._build_ui()
        self._build_menubar()
        self.load_session()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ── Left sidebar ────────────────────────────────────────────────────
        sidebar = QWidget()
        sidebar.setFixedWidth(250)
        sidebar.setStyleSheet("background: #252526; border-right: 1px solid #3c3c3c;")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        # App title
        title_widget = QWidget()
        title_widget.setStyleSheet("background: #1e1e1e; border-bottom: 1px solid #3c3c3c;")
        title_layout = QVBoxLayout(title_widget)
        title_layout.setContentsMargins(10, 10, 10, 10)
        self._app_title_lbl = QLabel(S("app.title"))
        self._app_title_lbl.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self._app_title_lbl.setStyleSheet("color: #e8e8e8;")
        self._app_title_lbl.setWordWrap(True)
        title_layout.addWidget(self._app_title_lbl)
        sidebar_layout.addWidget(title_widget)

        # Home entry
        home_item = QWidget()
        home_item.setStyleSheet(
            "QWidget { background: transparent; padding: 2px; }"
            "QWidget:hover { background: #2a2a2a; }"
        )
        home_item_layout = QHBoxLayout(home_item)
        home_item_layout.setContentsMargins(8, 6, 8, 6)
        home_icon = QLabel("⌂")
        home_icon.setFixedWidth(18)
        home_icon.setStyleSheet("color: #888;")
        home_item_layout.addWidget(home_icon)
        self._home_lbl = QLabel(S("app.home"))
        self._home_lbl.setStyleSheet("color: #ccc;")
        home_item_layout.addWidget(self._home_lbl)
        home_item_layout.addStretch()
        home_item.setCursor(Qt.CursorShape.PointingHandCursor)
        home_item.mousePressEvent = lambda _e: self._show_panel("welcome")
        sidebar_layout.addWidget(home_item)

        # Settings entry
        settings_item = QWidget()
        settings_item.setStyleSheet(
            "QWidget { background: transparent; padding: 2px; }"
            "QWidget:hover { background: #2a2a2a; }"
        )
        settings_item_layout = QHBoxLayout(settings_item)
        settings_item_layout.setContentsMargins(8, 6, 8, 6)
        settings_icon = QLabel("⚙")
        settings_icon.setFixedWidth(18)
        settings_icon.setStyleSheet("color: #888;")
        settings_item_layout.addWidget(settings_icon)
        self._settings_lbl = QLabel(S("app.settings"))
        self._settings_lbl.setStyleSheet("color: #ccc;")
        settings_item_layout.addWidget(self._settings_lbl)
        settings_item_layout.addStretch()
        settings_item.setCursor(Qt.CursorShape.PointingHandCursor)
        settings_item.mousePressEvent = lambda _e: self._show_panel("settings")
        sidebar_layout.addWidget(settings_item)

        # Step list (scrollable)
        step_scroll = QScrollArea()
        step_scroll.setWidgetResizable(True)
        step_scroll.setFrameShape(QFrame.Shape.NoFrame)
        step_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        step_scroll.setStyleSheet("QScrollArea { background: transparent; }")

        step_list_widget = QWidget()
        step_list_widget.setStyleSheet("background: transparent;")
        step_list_layout = QVBoxLayout(step_list_widget)
        step_list_layout.setContentsMargins(0, 4, 0, 4)
        step_list_layout.setSpacing(1)

        session_data = session.load()
        enabled_steps = session_data.get("enabled_steps", {})

        for step_id, label, optional in _STEP_DEFS:
            if step_id in _SEPARATOR_BEFORE:
                sep = QFrame()
                sep.setFrameShape(QFrame.Shape.HLine)
                sep.setStyleSheet("color: #444; margin: 4px 8px;")
                step_list_layout.addWidget(sep)

            # Steps 02, 08, 09 default to enabled even though they are optional
            _default_on = {"02", "08", "09"}
            enabled = enabled_steps.get(step_id, True if step_id in _default_on else not optional)
            item = StepItem(step_id, S(label), optional=optional, enabled=enabled)
            item.clicked.connect(self._on_step_clicked)
            item.toggled.connect(self._on_step_toggled)
            self._step_items[step_id]   = item
            self._enabled_steps[step_id] = enabled
            step_list_layout.addWidget(item)

        step_list_layout.addStretch()
        step_scroll.setWidget(step_list_widget)
        sidebar_layout.addWidget(step_scroll, 1)

        # Run All button
        run_all_widget = QWidget()
        run_all_widget.setStyleSheet("background: #1e1e1e; border-top: 1px solid #3c3c3c;")
        run_all_layout = QVBoxLayout(run_all_widget)
        run_all_layout.setContentsMargins(8, 8, 8, 8)
        self._btn_run_all = QPushButton(S("app.run_all"))
        self._btn_run_all.setFixedHeight(34)
        self._btn_run_all.setToolTip(S("app.run_all.tooltip"))
        self._btn_run_all.setStyleSheet(
            "QPushButton { background: #2d6a4f; color: white; border-radius: 5px;"
            " font-weight: bold; border: none; }"
            "QPushButton:hover { background: #40916c; }"
            "QPushButton:disabled { background: #333; color: #666; border: none; }"
        )
        self._btn_run_all.clicked.connect(self._on_run_all)
        run_all_layout.addWidget(self._btn_run_all)
        sidebar_layout.addWidget(run_all_widget)

        main_layout.addWidget(sidebar)

        # ── Right area ──────────────────────────────────────────────────────
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_splitter.setStyleSheet("QSplitter::handle { background: #444; height: 3px; }")

        # Panel stack
        self._stack = QStackedWidget()
        self._stack.setStyleSheet("background: #1e1e1e;")

        # Welcome panel (index 0 — shown on startup)
        self._welcome_panel = WelcomePanel()
        self._welcome_panel.go_settings.connect(lambda: self._show_panel("settings"))
        self._welcome_panel.go_resume.connect(self._on_welcome_resume)
        self._stack.addWidget(self._welcome_panel)

        # Settings panel (index 1)
        self._settings_panel = SettingsPanel()
        self._connect_settings_signals()
        self._stack.addWidget(self._settings_panel)
        self._panel_index: dict[str, int] = {"welcome": 0, "settings": 1}

        # Step panels
        self._step_panels: dict[str, QWidget] = {}
        panel_classes = {
            "01": Step01Panel,
            "02": Step02Panel,
            "03": Step03Panel,
            "04": Step04Panel,
            "05": Step05Panel,
            "06": Step06Panel,
            "07": Step07Panel,
            "08": Step08Panel,
            "09": Step09Panel,
            "10": Step10Panel,
        }

        for step_id, cls in panel_classes.items():
            panel = cls()
            self._step_panels[step_id] = panel
            idx = self._stack.addWidget(panel)
            self._panel_index[step_id] = idx

            # Connect run_requested for BasePanel subclasses
            if hasattr(panel, "run_requested"):
                panel.run_requested.connect(self._on_run_step)

            # Connect next button for BasePanel subclasses
            if hasattr(panel, "_btn_next"):
                panel._btn_next.clicked.connect(
                    lambda _checked, sid=step_id: self._advance_to_next(sid)
                )

        # Step 02 output folder changes → auto-link to step 03 input
        self._step_panels["02"].dirs_changed.connect(self._on_step02_dirs_changed)

        # Step 01 output folder changes → refresh downstream path labels
        self._step_panels["01"].dirs_changed.connect(self._on_step01_dirs_changed)

        # Step 03 (quality) TIF input change → refresh step 03 itself and step 04+
        self._step_panels["03"].dirs_changed.connect(self._on_step03_dirs_changed)

        right_splitter.addWidget(self._stack)

        # Log widget
        self._log_widget = LogWidget()
        self._log_widget.setMinimumHeight(120)
        right_splitter.addWidget(self._log_widget)
        right_splitter.setSizes([600, 200])

        main_layout.addWidget(right_splitter, 1)

        # ── Status bar ──────────────────────────────────────────────────────
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_label = QLabel(S("app.status.ready"))
        self._status_bar.addWidget(self._status_label)
        self._output_dir_label = QLabel("")
        self._output_dir_label.setStyleSheet("color: #666;")
        self._status_bar.addPermanentWidget(self._output_dir_label)

    def _build_menubar(self) -> None:
        self.menuBar().setVisible(False)

    # ── Panel rebuild (called on language change) ─────────────────────────────

    def _rebuild_step_panels(self) -> None:
        """Remove all step panels and recreate them so every S() label picks up
        the newly loaded language.  Session data is already saved at this point."""
        panel_classes = {
            "01": Step01Panel, "02": Step02Panel, "03": Step03Panel,
            "04": Step04Panel, "05": Step05Panel, "06": Step06Panel,
            "07": Step07Panel, "08": Step08Panel, "09": Step09Panel,
            "10": Step10Panel,
        }

        # Remove old panels from the stack
        for panel in self._step_panels.values():
            self._stack.removeWidget(panel)
            panel.deleteLater()
        self._step_panels.clear()
        # Keep welcome (0) and settings (1) panels; rebuild step panel_index from scratch
        self._panel_index = {"welcome": 0, "settings": 1}

        # Recreate and reconnect
        for step_id, cls in panel_classes.items():
            panel = cls()
            self._step_panels[step_id] = panel
            idx = self._stack.addWidget(panel)
            self._panel_index[step_id] = idx

            if hasattr(panel, "run_requested"):
                panel.run_requested.connect(self._on_run_step)
            if hasattr(panel, "_btn_next"):
                panel._btn_next.clicked.connect(
                    lambda _checked, sid=step_id: self._advance_to_next(sid)
                )

        self._step_panels["02"].dirs_changed.connect(self._on_step02_dirs_changed)
        self._step_panels["01"].dirs_changed.connect(self._on_step01_dirs_changed)
        self._step_panels["03"].dirs_changed.connect(self._on_step03_dirs_changed)

        # Restore session data into all new panels from in-memory data
        # (do NOT call load_session() here — that would re-read the disk and
        # overwrite unsaved changes, e.g. a language switch before save_session)
        self._apply_session_data()

    # ── Settings panel signal wiring ──────────────────────────────────────────

    def _connect_settings_signals(self) -> None:
        """Connect all signals from the (possibly rebuilt) settings panel."""
        self._settings_panel._btn_save.clicked.connect(self._on_settings_saved)
        self._settings_panel._btn_reset.clicked.connect(self._reset_session)
        self._settings_panel.profile_selected.connect(self._on_profile_selected)
        self._settings_panel.profile_save_requested.connect(self._on_profile_save)
        self._settings_panel.profile_save_as_requested.connect(self._on_profile_save_as)
        self._settings_panel.profile_delete_requested.connect(self._on_profile_delete)

    # ── Session management ────────────────────────────────────────────────────

    def _reset_session(self) -> None:
        """Prompt the user, then wipe the session file and reload defaults."""
        from PySide6.QtWidgets import QMessageBox
        reply = QMessageBox.question(
            self,
            S("btn.reset_session"),
            S("msg.reset_session.confirm"),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self._session_data = session.reset()
        self.load_session()
        QMessageBox.information(self, S("btn.reset_session"), S("msg.reset_session.done"))

    def load_session(self) -> None:
        """Load session from disk and apply to all panels."""
        self._session_data = session.load()
        self._apply_session_data()
        self._refresh_profile_ui()

    def _apply_session_data(self) -> None:
        """Apply self._session_data to all panels without reading from disk.

        Called by load_session() after a disk read, and also by
        _rebuild_step_panels() so that an in-progress language switch does not
        trigger a second disk read that would overwrite the unsaved language
        change.
        """
        self._welcome_panel.load_session(self._session_data)
        self._settings_panel.load_session(self._session_data)

        # Apply enabled step states — iterate ALL steps (not just saved ones)
        # so that visual state and checkbox are fully reset after a session reset.
        enabled_steps = self._session_data.get("enabled_steps", {})
        _default_on = {"02", "08", "09"}
        for step_id, _label, optional in _STEP_DEFS:
            enabled = enabled_steps.get(
                step_id,
                True if step_id in _default_on else not optional,
            )
            self._enabled_steps[step_id] = enabled
            item = self._step_items.get(step_id)
            if item:
                item.set_enabled_visual(enabled)
                if item._check is not None:
                    item._check.blockSignals(True)
                    item._check.setChecked(enabled)
                    item._check.blockSignals(False)

        # Enforce Step 09 dependency on Step 08 (GIF needs series output)
        if not self._enabled_steps.get("08", True):
            item09 = self._step_items.get("09")
            if item09:
                item09.set_checkbox_enabled(False)
                item09.set_enabled_visual(False)
                self._enabled_steps["09"] = False

        # Load step panels that use load_session() to populate derived folder fields
        for sid in ("01", "02", "03", "04", "05", "06", "07", "08", "09", "10"):
            panel = self._step_panels.get(sid)
            if panel and hasattr(panel, "load_session"):
                panel.load_session(self._session_data)

        # Enforce Step 01→02 lock: if Step 01 is enabled, lock Step 02 checkbox
        if self._enabled_steps.get("01", False):
            item02 = self._step_items.get("02")
            if item02:
                item02.set_checkbox_locked(True)

        # Update batch run button label based on starting step
        self._update_run_all_button()

        # Update status bar
        output_dir = self._session_data.get("output_dir", "")
        self._output_dir_label.setText(S("label.output", d=output_dir) if output_dir else "")

    def save_session(self) -> None:
        """Collect all panel values and persist to disk."""
        data = self._session_data.copy()
        data.update(self._settings_panel.get_session_values())

        # Gather enabled steps
        data["enabled_steps"] = dict(self._enabled_steps)

        # Gather config updates from each step panel
        for step_id, panel in self._step_panels.items():
            if hasattr(panel, "get_config_updates"):
                data.update(panel.get_config_updates())

        self._session_data = data
        session.save(data)

        # Auto-save to active profile whenever session is saved
        active = data.get("active_profile")
        if active:
            profile_manager.save_profile(active, data)

    # ── Profile management ────────────────────────────────────────────────────

    def _refresh_profile_ui(self) -> None:
        """Sync the profile combo in settings panel with disk state."""
        active = self._session_data.get("active_profile")
        self._settings_panel.refresh_profile_list(
            profile_manager.list_profiles(), active
        )

    def _on_profile_selected(self, name: str) -> None:
        """User picked a profile from the dropdown (empty string = Unsaved)."""
        if not name:
            self._session_data["active_profile"] = None
            session.save(self._session_data)
            return
        data = profile_manager.load_profile(name)
        for k, v in data.items():
            self._session_data[k] = v
        self._session_data["active_profile"] = name
        self._apply_session_data()
        session.save(self._session_data)

    def _on_profile_save(self) -> None:
        """Save current settings to the active profile."""
        name = self._session_data.get("active_profile")
        if not name:
            return
        self.save_session()
        self._refresh_profile_ui()

    def _on_profile_save_as(self, name: str) -> None:
        """Save current settings as a new named profile and switch to it."""
        self.save_session()
        profile_manager.save_profile(name, self._session_data)
        self._session_data["active_profile"] = name
        session.save(self._session_data)
        self._refresh_profile_ui()

    def _on_profile_delete(self) -> None:
        """Delete the active profile and reset to Unsaved state."""
        name = self._session_data.get("active_profile")
        if not name:
            return
        profile_manager.delete_profile(name)
        self._session_data["active_profile"] = None
        session.save(self._session_data)
        self._refresh_profile_ui()

    # ── Navigation ────────────────────────────────────────────────────────────

    def _show_panel(self, key: str) -> None:
        idx = self._panel_index.get(key)
        if idx is not None:
            self._stack.setCurrentIndex(idx)

        # Update sidebar selection
        for sid, item in self._step_items.items():
            item.set_selected(sid == key)

    def _on_step_clicked(self, step_id: str) -> None:
        self._show_panel(step_id)

    def _on_step_toggled(self, step_id: str, enabled: bool) -> None:
        self._enabled_steps[step_id] = enabled
        if step_id in self._step_items:
            self._step_items[step_id].set_enabled_visual(enabled)

        # Step 01 (PIPP) checked → force Step 02 ON and lock its checkbox
        if step_id == "01":
            item02 = self._step_items.get("02")
            if item02 and item02._check is not None:
                if enabled:
                    item02._check.blockSignals(True)
                    item02._check.setChecked(True)
                    item02._check.blockSignals(False)
                    item02.set_enabled_visual(True)
                    item02.set_checkbox_locked(True)
                    self._enabled_steps["02"] = True
                else:
                    item02.set_checkbox_locked(False)
            self._update_run_all_button()
            return

        if step_id == "02":
            self._update_run_all_button()
            return

        # Step 09 (GIF) requires Step 08 (series) output — enforce dependency
        if step_id == "08":
            item09 = self._step_items.get("09")
            if item09:
                if not enabled:
                    # Step 08 disabled → uncheck Step 09 (checkbox stays clickable)
                    item09.set_checkbox_enabled(False)
                    item09.set_enabled_visual(False)
                    self._enabled_steps["09"] = False
                else:
                    # Step 08 re-enabled → restore Step 09 to checked/enabled
                    if item09._check is not None:
                        item09._check.blockSignals(True)
                        item09._check.setChecked(True)
                        item09._check.blockSignals(False)
                    item09.set_enabled_visual(True)
                    self._enabled_steps["09"] = True
        elif step_id == "09" and enabled:
            # Step 09 enabled → cascade: also enable Step 08 if it was off
            if not self._enabled_steps.get("08", False):
                item08 = self._step_items.get("08")
                if item08 is not None and item08._check is not None:
                    item08._check.setChecked(True)  # triggers _on_step_toggled("08", True)

    def _update_run_all_button(self) -> None:
        """Update batch run button label based on the current starting step."""
        if self._enabled_steps.get("01", False):
            key = "app.run_all.from1"
        elif self._enabled_steps.get("02", True):
            key = "app.run_all.from2"
        else:
            key = "app.run_all.from3"
        self._btn_run_all.setText(S(key))

    def _on_welcome_resume(self) -> None:
        """Navigate to the first enabled step panel."""
        for step_id, _, optional in _STEP_DEFS:
            if not optional or self._enabled_steps.get(step_id, False):
                self._show_panel(step_id)
                return
        # Fallback: show step 03 (always mandatory)
        self._show_panel("03")

    def _advance_to_next(self, current_step_id: str) -> None:
        """Show the panel after the given step in the sidebar order."""
        ids = [s for s, _l, _o in _STEP_DEFS]
        try:
            current_idx = ids.index(current_step_id)
        except ValueError:
            return
        if current_idx + 1 < len(ids):
            self._show_panel(ids[current_idx + 1])

    # ── Step execution ────────────────────────────────────────────────────────

    def _on_step02_dirs_changed(self) -> None:
        """Auto-link Step 2 SER/output dirs to session and all downstream steps."""
        updates = self._step_panels["02"].get_config_updates()
        step02_ser = updates.get("step02_ser_dir", "")
        step02_out = updates.get("step02_output_dir", "")
        if step02_ser:
            self._session_data["step02_ser_dir"] = step02_ser
        if step02_out:
            self._session_data["step02_output_dir"] = step02_out
            self._session_data["input_dir"] = step02_out
            # All step outputs are siblings of step02 output under the same base dir
            new_output_dir = str(Path(step02_out).parent)
            self._session_data["output_dir"] = new_output_dir
            self._output_dir_label.setText(S("label.output", d=new_output_dir))
        # Cascade to all downstream steps regardless of whether step02_out changed
        for sid in ("03", "04", "05", "06", "07", "08", "09", "10"):
            dep = self._step_panels.get(sid)
            if dep and hasattr(dep, "load_session"):
                dep.load_session(self._session_data)

    def _on_step01_dirs_changed(self) -> None:
        """Refresh output_dir and all dependent panels when Step 1 output changes."""
        updates = self._step_panels["01"].get_config_updates()
        new_output_dir = updates.get("output_dir", "")
        if new_output_dir:
            self._session_data["ser_input_dir"]     = updates.get("ser_input_dir", "")
            self._session_data["output_dir"]        = new_output_dir
            self._session_data["step01_output_dir"] = updates.get("step01_output_dir", "")
            # Clear stale downstream saved paths so panels re-derive from the new base.
            self._session_data["step02_ser_dir"]    = ""
            self._session_data["step02_output_dir"] = ""
            # NOTE: do NOT clear input_dir here — it is set below after step02 derives
            # its output, so that step03+ can see the correct TIF input path.
            self._output_dir_label.setText(S("label.output", d=new_output_dir))

            # 1) Load step02 first so it re-derives its output from the new step01 result.
            panel02 = self._step_panels.get("02")
            if panel02 and hasattr(panel02, "load_session"):
                panel02.load_session(self._session_data)
                # Propagate step02's freshly derived output into session as input_dir
                # so that step03 (and beyond) see the correct TIF source path.
                upd02 = panel02.get_config_updates()
                step02_out = upd02.get("step02_output_dir", "")
                self._session_data["step02_output_dir"] = step02_out
                self._session_data["input_dir"]         = step02_out

            # 2) Load step03-10 with updated session (input_dir now reflects step02 output).
            for sid in ("03", "04", "05", "06", "07", "08", "09", "10"):
                dep = self._step_panels.get(sid)
                if dep and hasattr(dep, "load_session"):
                    dep.load_session(self._session_data)

    def _on_step03_dirs_changed(self) -> None:
        """Propagate Step 3 TIF input change to step 03 itself and all downstream steps."""
        updates = self._step_panels["03"].get_config_updates()
        inp = updates.get("input_dir", "")
        if inp:
            self._session_data["input_dir"] = inp
            # Derive output_dir as parent of input dir (all step outputs are siblings).
            new_output_dir = str(Path(inp).parent)
            self._session_data["output_dir"] = new_output_dir
            self._output_dir_label.setText(S("label.output", d=new_output_dir))
        # Reload step 03 itself so its output folder display updates.
        panel03 = self._step_panels.get("03")
        if panel03 and hasattr(panel03, "load_session"):
            panel03.load_session(self._session_data)
        # Cascade to all downstream steps.
        for sid in ("04", "05", "06", "07", "08", "09", "10"):
            dep = self._step_panels.get(sid)
            if dep and hasattr(dep, "load_session"):
                dep.load_session(self._session_data)

    def _on_settings_saved(self) -> None:
        # 1. Apply new settings values to session data first (camera_mode etc.)
        old_lang = self._session_data.get("language", "en")
        data = self._session_data.copy()
        data.update(self._settings_panel.get_session_values())
        data["enabled_steps"] = dict(self._enabled_steps)
        self._session_data = data

        # Reload i18n and refresh UI if language changed
        new_lang = data.get("language", "en")
        if new_lang != old_lang:
            i18n.load(new_lang)

            # Rebuild settings panel with new language
            self._stack.removeWidget(self._settings_panel)
            self._settings_panel.deleteLater()
            self._settings_panel = SettingsPanel()
            self._connect_settings_signals()
            self._stack.insertWidget(1, self._settings_panel)

            # Update window title and static sidebar labels
            self.setWindowTitle(S("app.title"))
            self._app_title_lbl.setText(S("app.title"))
            self._home_lbl.setText(S("app.home"))
            self._settings_lbl.setText(S("app.settings"))

            # Update log widget buttons
            self._log_widget.retranslate()

            # Update welcome panel
            self._welcome_panel.retranslate()

            # Update sidebar step labels
            for step_id, key, _optional in _STEP_DEFS:
                item = self._step_items.get(step_id)
                if item:
                    item.set_label(S(key))

            # Rebuild step panels (applies _session_data in-memory, not from disk)
            self._rebuild_step_panels()
            self._refresh_profile_ui()

        # 2. Refresh panels with the updated session (updates _is_color in step06/07/08)
        for sid in ("01", "03", "04", "05", "06", "07", "08", "09", "10"):
            panel = self._step_panels.get(sid)
            if panel and hasattr(panel, "load_session"):
                panel.load_session(self._session_data)

        # 3. Now collect all panel values (panels are in the correct camera mode)
        self.save_session()

        output_dir = self._session_data.get("output_dir", "")
        self._output_dir_label.setText(S("label.output", d=output_dir) if output_dir else "")
        QMessageBox.information(self, S("msg.dialog.settings"), S("msg.settings_saved"))

    def _on_run_step(self, step_id: str) -> None:
        """Run a single step, with pre-flight validation."""
        if self._runner and self._runner.isRunning():
            return
        panel = self._step_panels.get(step_id)
        if panel and hasattr(panel, "validate"):
            d = self._session_data
            issues = panel.validate(d)
            errors   = [i for i in issues if i.severity == "error"]
            warnings = [i for i in issues if i.severity == "warning"]
            if errors:
                msg = "\n".join(f"⛔ {e.message}" for e in errors)
                if warnings:
                    msg += "\n\n" + "\n".join(f"⚠ {w.message}" for w in warnings)
                QMessageBox.critical(self, S("msg.dialog.run_blocked"), msg)
                return
            if warnings:
                msg = "\n".join(f"⚠ {w.message}" for w in warnings)
                ret = QMessageBox.warning(
                    self, S("msg.dialog.warning"), msg + "\n\n" + S("msg.run_confirm_warn"),
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
                )
                if ret != QMessageBox.Yes:
                    return
        config = self.build_config()
        self._runner = StepRunner(config, [step_id], self._results, parent=self)
        self._connect_runner(self._runner)
        self._runner.start()

    def _on_run_all(self) -> None:
        """Batch run: determine starting step from checkbox state, validate, confirm, then run."""
        if self._runner and self._runner.isRunning():
            return

        self.save_session()
        d = self._session_data

        # 1. Determine starting step
        if self._enabled_steps.get("01", False):
            start_from = "01"
        elif self._enabled_steps.get("02", True):
            start_from = "02"
        else:
            start_from = "03"

        # 2. Validate input files exist
        ok, err_title, err_msg = self._validate_batch_input(start_from, d)
        if not ok:
            QMessageBox.warning(self, err_title, err_msg)
            return

        # 3. Build ordered step list
        steps = self._build_batch_steps(start_from)
        if not steps:
            return

        # 4. Pre-flight validation for all planned steps
        issues: dict[str, list] = {}
        for sid in steps:
            panel = self._step_panels.get(sid)
            if panel and hasattr(panel, "validate"):
                result = panel.validate(d, batch_mode=True)
                if result:
                    issues[sid] = result

        # 5. Show graphical confirmation dialog (with validation results)
        if not self._show_batch_confirm(steps, start_from, d, issues=issues):
            return

        # 5. Reset status icons and launch runner
        for sid in steps:
            if sid in self._step_items:
                self._step_items[sid].set_status("idle")
            panel = self._step_panels.get(sid)
            if panel and hasattr(panel, "set_status"):
                panel.set_status("idle")

        config = self.build_config()
        self._runner = StepRunner(config, steps, self._results, parent=self)
        self._connect_runner(self._runner)
        self._runner.start()
        self._btn_run_all.setEnabled(False)
        self._status_label.setText(S("app.status.running"))

    def _validate_batch_input(self, start_from: str, d: dict) -> tuple[bool, str, str]:
        """Check that the starting input folder has the expected files."""
        if start_from == "01":
            path_str = d.get("ser_input_dir", "").strip()
            label = S("batch.label.step1_ser")
            patterns = ("*.ser", "*.SER")
        elif start_from == "02":
            path_str = (d.get("step02_ser_dir", "") or d.get("ser_input_dir", "")).strip()
            label = S("batch.label.step2_ser")
            patterns = ("*.ser", "*.SER")
        else:
            path_str = d.get("input_dir", "").strip()
            label = S("batch.label.step3_tif")
            patterns = ("*.tif", "*.TIF")

        if not path_str:
            return False, S("batch.no_folder.title"), S("batch.no_folder.msg", label=label)
        p = Path(path_str)
        files = [f for pat in patterns for f in p.glob(pat)]
        if not p.exists() or not files:
            return False, S("batch.no_files.title"), S("batch.no_files.msg", label=label, path=path_str)
        return True, "", ""

    def _build_batch_steps(self, start_from: str) -> list[str]:
        """Build the ordered list of steps to execute for batch run."""
        all_ids = [sid for sid, _, _ in _STEP_DEFS]
        start_idx = all_ids.index(start_from)
        result = []
        for i, (sid, _, optional) in enumerate(_STEP_DEFS):
            if i < start_idx:
                continue
            if optional and not self._enabled_steps.get(sid, False):
                continue
            result.append(sid)
        return result

    def _show_batch_confirm(
        self,
        steps: list[str],
        start_from: str,
        d: dict,
        issues: dict[str, list] | None = None,
    ) -> bool:
        """Show a graphical pipeline confirmation dialog."""
        from gui.widgets.batch_confirm_dialog import BatchConfirmDialog

        # Input summary with file count
        if start_from in ("01", "02"):
            inp_path = (
                d.get("ser_input_dir", "")
                if start_from == "01"
                else (d.get("step02_ser_dir", "") or d.get("ser_input_dir", ""))
            )
            p = Path(inp_path)
            n = len([f for pat in ("*.ser", "*.SER") for f in p.glob(pat)])
            inp_summary = S("batch.input.ser", path=inp_path, n=n)
        else:
            inp_path = d.get("input_dir", "")
            p = Path(inp_path)
            n = len([f for pat in ("*.tif", "*.TIF") for f in p.glob(pat)])
            inp_summary = S("batch.input.tif", path=inp_path, n=n)

        out_base = d.get("output_dir", "")

        _out_names = {
            "03": "step03_quality",
            "04": "step04_derotated",
            "05": "step05_wavelet_master",
            "06": "step06_rgb_composite",
            "07": "step07_wavelet_preview",
            "08": "step08_series_composite",
            "09": "step09_gif",
            "10": "step10_summary_grid",
        }

        def _out(step_id: str) -> str:
            if step_id == "01":
                return d.get("step01_output_dir", "")
            if step_id == "02":
                return d.get("step02_output_dir", "")
            if out_base and step_id in _out_names:
                return f"{out_base}/{_out_names[step_id]}"
            return ""

        output_paths = {sid: _out(sid) for sid, _, _ in _STEP_DEFS}

        dlg = BatchConfirmDialog(
            parent=self,
            steps=steps,
            all_defs=_STEP_DEFS,
            start_from=start_from,
            output_paths=output_paths,
            input_summary=inp_summary,
            issues=issues,
        )
        return dlg.exec() == dlg.DialogCode.Accepted

    def _connect_runner(self, runner: StepRunner) -> None:
        runner.log_line.connect(self._log_widget.append_line)
        runner.step_started.connect(self._on_step_started)
        runner.step_finished.connect(self._on_step_finished)
        runner.progress.connect(self._on_step_progress)
        runner.all_done.connect(self._on_all_done)
        runner.cancelled.connect(self._on_cancelled)
        # Connect stop buttons of all panels to this runner
        for panel in self._step_panels.values():
            if hasattr(panel, "stop_requested"):
                panel.stop_requested.connect(runner.abort)

    def _on_step_progress(self, step_id: str, current: int, total: int) -> None:
        panel = self._step_panels.get(step_id)
        if panel and hasattr(panel, "set_progress"):
            panel.set_progress(current, total)

    def _on_step_started(self, step_id: str) -> None:
        if step_id in self._step_items:
            self._step_items[step_id].set_status("running")
        panel = self._step_panels.get(step_id)
        if panel and hasattr(panel, "set_status"):
            panel.set_status("running")
        if panel and hasattr(panel, "set_running"):
            panel.set_running(True)
        # Auto-navigate to the currently running step panel
        self._show_panel(step_id)

    def _on_step_finished(self, step_id: str, ok: bool, results: Any) -> None:
        status = "success" if ok else "error"
        if step_id in self._step_items:
            self._step_items[step_id].set_status(status)
        panel = self._step_panels.get(step_id)
        if panel and hasattr(panel, "set_status"):
            panel.set_status(status)
        if panel and hasattr(panel, "set_running"):
            panel.set_running(False)
        # Keep step_status in sync for session persistence
        self._session_data.setdefault("step_status", {})[step_id] = status
        if ok:
            if results is not None:
                self._results[step_id] = results
            # After step 02 completes, link output dir to all downstream steps.
            if step_id == "02":
                panel02 = self._step_panels.get("02")
                if panel02 and hasattr(panel02, "get_config_updates"):
                    upd = panel02.get_config_updates()
                    ser = upd.get("step02_ser_dir", "")
                    out = upd.get("step02_output_dir", "")
                    if ser:
                        self._session_data["step02_ser_dir"] = ser
                    if out:
                        self._session_data["step02_output_dir"] = out
                        self._session_data["input_dir"] = out
                        new_output_dir = str(Path(out).parent)
                        self._session_data["output_dir"] = new_output_dir
                        self._output_dir_label.setText(S("label.output", d=new_output_dir))
                for sid in ("03", "04", "05", "06", "07", "08", "09", "10"):
                    dep = self._step_panels.get(sid)
                    if dep and hasattr(dep, "load_session"):
                        dep.load_session(self._session_data)
            # After step 03 completes (quality) → refresh step 04 sweep button
            if step_id == "03":
                dep = self._step_panels.get("04")
                if dep and hasattr(dep, "load_session"):
                    dep.load_session(self._session_data)
            # After step 04 completes (derotation) → refresh step 05 wavelet master
            if step_id == "04":
                dep = self._step_panels.get("05")
                if dep and hasattr(dep, "load_session"):
                    dep.load_session(self._session_data)
            # After step 05 completes (wavelet master) → refresh step 06 rgb composite
            if step_id == "05":
                dep = self._step_panels.get("06")
                if dep and hasattr(dep, "load_session"):
                    dep.load_session(self._session_data)
            # After step 06 completes (RGB composite) → refresh step 10 summary grid
            if step_id == "06":
                dep = self._step_panels.get("10")
                if dep and hasattr(dep, "load_session"):
                    dep.load_session(self._session_data)
            # After step 07 completes (wavelet preview) → refresh step 08 paths
            if step_id == "07":
                dep = self._step_panels.get("08")
                if dep and hasattr(dep, "load_session"):
                    dep.load_session(self._session_data)
            if panel and hasattr(panel, "refresh_after_run"):
                panel.refresh_after_run()

    def _on_all_done(self) -> None:
        self._btn_run_all.setEnabled(True)
        self._status_label.setText(S("app.status.ready"))
        # Disconnect stop buttons from finished runner to avoid stale connections
        if self._runner:
            for panel in self._step_panels.values():
                if hasattr(panel, "stop_requested"):
                    try:
                        panel.stop_requested.disconnect(self._runner.abort)
                    except RuntimeError:
                        pass

    def _on_cancelled(self) -> None:
        """Runner finished after a stop request — confirm to all visible panels."""
        for panel in self._step_panels.values():
            if hasattr(panel, "on_cancelled") and hasattr(panel, "_cancelling"):
                if panel._cancelling:
                    panel.on_cancelled()

    # ── Config builder ────────────────────────────────────────────────────────

    def build_config(self) -> PipelineConfig:
        """Build a PipelineConfig from the current session data."""
        self.save_session()
        d = self._session_data

        ser_dir    = Path(d.get("ser_input_dir", "") or ".")
        input_dir  = Path(d.get("input_dir", "") or ".")
        output_dir = Path(d.get("output_dir", "") or ".")
        step01_out_raw = d.get("step01_output_dir", "")
        step01_out = Path(step01_out_raw) if step01_out_raw else None
        step02_ser_raw = d.get("step02_ser_dir", "")
        step02_ser = Path(step02_ser_raw) if step02_ser_raw else None
        step02_out_raw = d.get("step02_output_dir", "")
        step02_out = Path(step02_out_raw) if step02_out_raw else None

        # Global core limit: 0 = auto.  Step 1 caps at 4; Step 2 uses the full value.
        _global_workers = int(d.get("global_max_workers", 0))

        lucky_stack = LuckyStackConfig(
            top_percent           = float(d.get("lucky_top_percent", 0.25)),
            ap_size               = int(d.get("lucky_ap_size", 64)),
            n_iterations          = int(d.get("lucky_n_iterations", 2)),
            n_workers             = _global_workers,   # Step 2 uses all available cores
            n_ser_parallel        = int(d.get("lucky_n_ser_parallel", 1)),
            use_tps               = bool(d.get("lucky_use_tps", False)),
            use_as4_ap_grid       = bool(d.get("lucky_use_as4_ap_grid", False)),
            use_ncc               = bool(d.get("lucky_use_ncc", False)),
            per_ap_selection      = bool(d.get("lucky_per_ap_selection", False)),
            fourier_quality_power = float(d.get("lucky_fourier_power", 1.0)),
        )

        pipp = PippConfig(
            roi_size     = int(d.get("roi_size", 448)),
            min_diameter = int(d.get("min_diameter", 50)),
            n_workers    = _global_workers,   # step01_pipp.py caps at 4 internally
        )

        wavelet = WaveletConfig(
            preview_amounts = list(d.get("preview_amounts", [200.0, 200.0, 200.0, 0.0, 0.0, 0.0])),
            master_amounts  = list(d.get("master_amounts",  [200.0, 200.0, 200.0, 0.0, 0.0, 0.0])),
            series_amounts  = list(d.get("series_amounts",  [200.0, 200.0, 200.0, 0.0, 0.0, 0.0])),
            border_taper_px = int(d.get("border_taper_px", 0)),
            auto_params     = True,
        )

        # window_frames: number of filter cycles (= time-series frames) per window.
        # Old sessions may store window_cycles or window_seconds; convert on load.
        cycle_sec = int(d.get("cycle_seconds", 270))
        if "window_frames" in d:
            window_frames = int(d["window_frames"])
        elif "window_cycles" in d:
            window_frames = int(d["window_cycles"])
        else:
            window_sec    = int(d.get("window_seconds", 900))
            window_frames = max(1, round(window_sec / cycle_sec))
        quality = QualityConfig(
            window_frames         = window_frames,
            cycle_minutes         = cycle_sec  / 60.0,
            n_windows             = int(d.get("n_windows", 1)),
            allow_overlap         = bool(d.get("allow_overlap", False)),
            min_quality_threshold = float(d.get("min_quality_threshold_03", 0.0)),
        )

        derotation = DerotationConfig(
            rotation_period_hours = float(d.get("rotation_period_hours",
                                               d.get("rotation_period", 9.9281))),
            horizons_id           = str(d.get("horizons_id", "599")),
            warp_scale            = float(d.get("warp_scale", 0.80)),
            normalize_brightness  = bool(d.get("normalize_brightness", False)),
            min_quality_threshold = float(d.get("min_quality_threshold", 0.3)),
        )

        def _parse_specs(raw: list | None):
            """Convert a list of spec dicts to CompositeSpec objects, or None."""
            if not raw:
                return None
            return [
                CompositeSpec(
                    name = s.get("name", "RGB"),
                    R    = s.get("R", "R"),
                    G    = s.get("G", "G"),
                    B    = s.get("B", "B"),
                    L    = s.get("L") or None,
                )
                for s in raw if s.get("name")
            ] or None

        # Build CompositeSpec lists from session data
        specs        = _parse_specs(d.get("composite_specs"))         # step06 specs
        series_specs = _parse_specs(d.get("series_composite_specs"))  # step08 series specs

        # series_cycle_seconds: step8-specific; fall back to step4's cycle_seconds
        _series_cyc = float(d.get("series_cycle_seconds",
                                   d.get("cycle_seconds", 270)))
        composite = CompositeConfig(
            max_shift_px            = float(d.get("max_shift_px", 8.0)),
            global_filter_normalize = bool(d.get("global_filter_normalize", True)),
            series_scale            = float(d.get("series_scale", 1.00)),
            cycle_seconds           = _series_cyc,
            stack_window_n          = int(d.get("stack_window_n", 5)),
            stack_min_quality       = float(d.get("stack_min_quality", 0.05)),
            save_mono_frames        = bool(d.get("save_mono_frames", False)),
            **({"specs": specs} if specs else {}),
            **({"series_specs": series_specs} if series_specs else {}),
        )

        gif = GifConfig(
            fps           = float(d.get("fps", 6.0)),
            resize_factor = float(d.get("resize_factor", 1.0)),
        )

        # Derive composite column names from the step06 composite_specs so the
        # summary grid always reflects what step06 actually produced.
        raw_specs = d.get("composite_specs")
        if raw_specs:
            grid_composites = [s.get("name", "") for s in raw_specs if s.get("name")]
        else:
            grid_composites = ["RGB", "IR-RGB", "CH4-G-IR"]

        grid = SummaryGridConfig(
            black_point  = float(d.get("black_point", 0.04)),
            gamma        = float(d.get("gamma", 0.8)),
            cell_size_px = int(d.get("cell_size_px", 300)),
            composites   = grid_composites,
        )

        filters_raw = d.get("filters", "IR,R,G,B,CH4")
        if isinstance(filters_raw, str):
            filters = [f.strip() for f in filters_raw.split(",") if f.strip()]
        else:
            filters = list(filters_raw)

        camera_mode = d.get("camera_mode", "mono")
        # For color camera, default filter list is a single "COLOR" entry
        if camera_mode == "color" and filters == ["IR", "R", "G", "B", "CH4"]:
            filters = ["COLOR"]
        # For mono camera, restore full filter list if session still has "COLOR"
        elif camera_mode == "mono" and filters == ["COLOR"]:
            filters = ["IR", "R", "G", "B", "CH4"]

        return PipelineConfig(
            ser_input_dir     = ser_dir,
            input_dir         = input_dir,
            output_base_dir   = output_dir,
            step01_output_dir = step01_out,
            step02_ser_dir    = step02_ser,
            step02_output_dir = step02_out,
            pipp            = pipp,
            lucky_stack     = lucky_stack,
            wavelet         = wavelet,
            quality         = quality,
            derotation      = derotation,
            composite        = composite,
            gif              = gif,
            grid            = grid,
            target          = str(d.get("target", "Jup")),
            filters         = filters,
            camera_mode     = camera_mode,
        )

    # ── Misc ──────────────────────────────────────────────────────────────────

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            S("app.about.title"),
            f"{S('app.title')}\n\n{S('app.about.body')}",
        )

    def closeEvent(self, event) -> None:  # noqa: N802
        self.save_session()
        super().closeEvent(event)
