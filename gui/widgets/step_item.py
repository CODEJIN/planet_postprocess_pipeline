"""Sidebar step card widget."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (QCheckBox, QHBoxLayout, QLabel, QSizePolicy,
                                QWidget)

from gui.i18n import S

# Status → (icon, hex colour)
STATUS_STYLE: dict[str, tuple[str, str]] = {
    "idle":    ("○",  "#888888"),
    "running": ("⟳",  "#4da6ff"),
    "success": ("✓",  "#5cad5c"),
    "error":   ("✗",  "#e05c5c"),
    "skipped": ("—",  "#555555"),
    "waiting": ("⚠",  "#e08c30"),
}


class StepItem(QWidget):
    """One row in the sidebar: [status icon] [step number + name] [optional toggle]."""

    clicked    = Signal(str)   # step_id
    toggled    = Signal(str, bool)  # step_id, enabled

    def __init__(
        self,
        step_id: str,
        label: str,
        optional: bool = False,
        enabled: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.step_id  = step_id
        self._optional = optional
        self._selected = False

        self._build_ui(label, optional, enabled)
        self._update_style()

    # ── Construction ──────────────────────────────────────────────────────────

    def _build_ui(self, label: str, optional: bool, enabled: bool) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(6)

        # Status icon
        self._icon = QLabel("○")
        self._icon.setFont(QFont("Arial", 11))
        self._icon.setFixedWidth(18)
        self._icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._icon)

        # Step label — stretch=1 so it consumes all space left over after the
        # fixed icon and the small checkbox; Ignored policy lets it shrink to 0
        # if the sidebar is too narrow rather than pushing the checkbox off-screen.
        self._label = QLabel(label)
        self._label.setFont(QFont("Arial", 10))
        self._label.setSizePolicy(QSizePolicy.Policy.Ignored,
                                  QSizePolicy.Policy.Preferred)
        layout.addWidget(self._label, stretch=1)

        # Optional toggle checkbox — fixed width so it never competes with the
        # label for stretch space.
        if optional:
            self._check = QCheckBox()
            self._check.setChecked(enabled)
            self._check.setFixedWidth(20)
            self._check.setToolTip(S("sidebar.step.optional.tooltip"))
            self._check.stateChanged.connect(
                lambda s: self.toggled.emit(
                    self.step_id, s == Qt.CheckState.Checked.value
                )
            )
            layout.addWidget(self._check)
        else:
            self._check = None  # type: ignore[assignment]

        self.setCursor(Qt.CursorShape.PointingHandCursor)

    # ── Interaction ───────────────────────────────────────────────────────────

    def mousePressEvent(self, event):  # noqa: N802
        self.clicked.emit(self.step_id)
        super().mousePressEvent(event)

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self._update_style()

    def set_status(self, status: str) -> None:
        icon, colour = STATUS_STYLE.get(status, ("○", "#888888"))
        self._icon.setText(icon)
        self._icon.setStyleSheet(f"color: {colour};")

    def set_label(self, text: str) -> None:
        self._label.setText(text)

    def set_enabled_visual(self, enabled: bool) -> None:
        """Dim the row when the step is disabled (optional + unchecked)."""
        opacity = "1.0" if enabled else "0.4"
        self._label.setStyleSheet(f"opacity: {opacity}; color: {'#cccccc' if enabled else '#666666'};")

    def set_checkbox_enabled(self, enabled: bool) -> None:
        """Uncheck the optional toggle checkbox when a required upstream step is disabled.

        The checkbox remains interactive (not disabled) so the user can click it
        and trigger a cascade that enables the upstream step automatically.
        """
        if self._check is None:
            return
        if not enabled:
            self._check.blockSignals(True)
            self._check.setChecked(False)
            self._check.blockSignals(False)

    def set_checkbox_locked(self, locked: bool) -> None:
        """Prevent or allow user from toggling the checkbox (grays it out visually)."""
        if self._check is not None:
            self._check.setEnabled(not locked)

    # ── Style ─────────────────────────────────────────────────────────────────

    def _update_style(self) -> None:
        if self._selected:
            self.setStyleSheet("StepItem { background: #2d4a6b; border-radius: 4px; }")
        else:
            self.setStyleSheet("StepItem { background: transparent; }"
                               "StepItem:hover { background: #2a2a2a; border-radius: 4px; }")
