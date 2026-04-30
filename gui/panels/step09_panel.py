"""Step 9 — Animated GIF panel."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QWidget,
)

from gui.i18n import S
from gui.panels.base_panel import BasePanel

_SPINBOX_STYLE = (
    "QDoubleSpinBox { background: #3c3c3c; color: #d4d4d4; border: 1px solid #555;"
    " border-radius: 3px; padding: 3px 6px; }"
    "QDoubleSpinBox:focus { border-color: #4da6ff; }"
)
_READONLY_STYLE = (
    "QLineEdit { background: #2a2a2a; color: #888; border: 1px solid #3a3a3a;"
    " border-radius: 3px; padding: 3px 6px; }"
)


class Step09Panel(BasePanel):
    STEP_ID   = "09"
    TITLE_KEY = "step09.title"
    DESC_KEY  = "step09.desc"
    OPTIONAL  = True

    def __init__(self, parent: QWidget | None = None) -> None:
        self._output_dir: Path | None = None
        super().__init__(parent)

    def build_form(self) -> None:
        form_widget = QWidget()
        form_widget.setStyleSheet("background: transparent;")
        fl = QFormLayout(form_widget)
        fl.setSpacing(10)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        # Folder display (auto-derived, read-only)
        self._input_lbl = QLineEdit()
        self._input_lbl.setReadOnly(True)
        self._input_lbl.setStyleSheet(_READONLY_STYLE)
        lbl_in = QLabel(S("step09.input_dir"))
        lbl_in.setToolTip(S("step09.input_dir.tooltip"))
        fl.addRow(lbl_in, self._input_lbl)

        self._output_lbl = QLineEdit()
        self._output_lbl.setReadOnly(True)
        self._output_lbl.setStyleSheet(_READONLY_STYLE)
        lbl_out = QLabel(S("step09.output_dir"))
        lbl_out.setToolTip(S("step09.output_dir.tooltip"))
        fl.addRow(lbl_out, self._output_lbl)

        self._fps = QDoubleSpinBox()
        self._fps.setStyleSheet(_SPINBOX_STYLE)
        self._fps.setRange(1.0, 30.0)
        self._fps.setDecimals(1)
        self._fps.setSingleStep(0.5)
        self._fps.setValue(6.0)
        self._fps.setToolTip(S("step09.fps.tooltip"))
        lbl_fps = QLabel(S("step09.fps"))
        lbl_fps.setToolTip(S("step09.fps.tooltip"))
        fl.addRow(lbl_fps, self._fps)

        self._resize_factor = QDoubleSpinBox()
        self._resize_factor.setStyleSheet(_SPINBOX_STYLE)
        self._resize_factor.setRange(0.1, 2.0)
        self._resize_factor.setDecimals(1)
        self._resize_factor.setSingleStep(0.1)
        self._resize_factor.setValue(1.0)
        self._resize_factor.setToolTip(S("step09.resize.tooltip"))
        lbl_resize = QLabel(S("step09.resize"))
        lbl_resize.setToolTip(S("step09.resize.tooltip"))
        fl.addRow(lbl_resize, self._resize_factor)

        idx = self._form_layout.count() - 1
        self._form_layout.insertWidget(idx, form_widget)

    def get_config_updates(self) -> dict[str, Any]:
        return {
            "fps":           self._fps.value(),
            "resize_factor": self._resize_factor.value(),
        }

    def load_session(self, data: dict[str, Any]) -> None:
        out = data.get("output_dir", "")
        if out:
            p = Path(out)
            self._input_lbl.setText(str(p / "step08_series"))
            self._output_lbl.setText(str(p / "step09_gif"))
            self._output_dir = p
        self._fps.setValue(float(data.get("fps", 6.0)))
        self._resize_factor.setValue(float(data.get("resize_factor", 1.0)))

    def validate(self, config: dict, batch_mode: bool = False) -> list:
        from gui.validation import ValidationIssue, count_files
        issues = []
        if not batch_mode:
            out_base = config.get("output_dir", "").strip()
            input_path = str(Path(out_base) / "step08_series") if out_base else ""
            if not count_files(input_path, "**/*.png", "**/*.PNG", "**/*.tif", "**/*.TIF"):
                issues.append(ValidationIssue("error", S("validate.no_series_result")))
        return issues

    def output_paths(self) -> list[Path]:
        if self._output_dir is None:
            return []
        step_dir = self._output_dir / "step09_gif"
        if step_dir.exists():
            return sorted(step_dir.glob("*.gif"))
        return []

    def set_output_dir(self, path: Path | str) -> None:
        self._output_dir = Path(path) if path else None
