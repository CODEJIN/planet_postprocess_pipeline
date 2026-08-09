"""Step 3 — Quality assessment panel."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QWidget,
)

from gui.i18n import S
from gui.panels.base_panel import BasePanel
from gui.panels.step_status_widget import FolderStatusDot

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
_READONLY_STYLE = (
    "QLineEdit { background: #2a2a2a; color: #888; border: 1px solid #3a3a3a;"
    " border-radius: 3px; padding: 3px 6px; }"
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


def _dir_row(parent: QWidget, line_edit: QLineEdit) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setSpacing(4)
    row.addWidget(line_edit)
    btn = QPushButton(S("btn.browse"))
    btn.setFixedWidth(70)
    btn.setStyleSheet(_BTN_BROWSE)

    def _browse() -> None:
        current = line_edit.text().strip()
        folder = QFileDialog.getExistingDirectory(
            parent, S("dialog.folder_select"), current or str(Path.home())
        )
        if folder:
            line_edit.setText(folder)
            line_edit.editingFinished.emit()

    btn.clicked.connect(_browse)
    row.addWidget(btn)
    return row


class QualityPanel(BasePanel):
    STEP_ID   = "03"
    TITLE_KEY = "step03.title"
    DESC_KEY  = "step03.desc"
    OPTIONAL  = False

    # Emitted on editingFinished so downstream panels (05+) refresh immediately.
    dirs_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        self._output_dir: Path | None = None
        super().__init__(parent)

    # ── BasePanel interface ───────────────────────────────────────────────────

    def build_form(self) -> None:
        form_widget = QWidget()
        form_widget.setStyleSheet("background: transparent;")
        fl = QFormLayout(form_widget)
        fl.setSpacing(10)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        # TIF input dir (editable — user can override cascade value)
        self._input_lbl = QLineEdit()
        self._input_lbl.setStyleSheet(_INPUT_EMPTY_STYLE)
        self._input_lbl.setPlaceholderText(S("step03.input_dir.placeholder"))
        self._input_lbl.textChanged.connect(self._on_input_changed)
        self._input_lbl.editingFinished.connect(self.dirs_changed)
        lbl_in = QLabel(S("step03.input_dir"))
        lbl_in.setToolTip(S("step03.input_dir.tooltip"))
        self._input_dot = FolderStatusDot()
        in_row = _dir_row(self, self._input_lbl)
        in_row.insertWidget(0, self._input_dot)
        fl.addRow(lbl_in, in_row)

        self._output_lbl = QLineEdit()
        self._output_lbl.setReadOnly(True)
        self._output_lbl.setStyleSheet(_READONLY_STYLE)
        lbl_out = QLabel(S("step03.output_dir"))
        lbl_out.setToolTip(S("step03.output_dir.tooltip"))
        fl.addRow(lbl_out, self._output_lbl)

        # Window frames
        self._window_frames = QSpinBox()
        self._window_frames.setStyleSheet(_INT_SPINBOX_STYLE)
        self._window_frames.setRange(1, 20)
        self._window_frames.setSingleStep(1)
        self._window_frames.setValue(3)
        self._window_frames.setToolTip(S("step03.window_frames.tooltip"))
        lbl_win = QLabel(S("step03.window_frames"))
        lbl_win.setToolTip(S("step03.window_frames.tooltip"))
        fl.addRow(lbl_win, self._window_frames)

        # Cycle seconds
        self._cycle_seconds = QSpinBox()
        self._cycle_seconds.setStyleSheet(_INT_SPINBOX_STYLE)
        self._cycle_seconds.setRange(10, 600)
        self._cycle_seconds.setSingleStep(15)
        self._cycle_seconds.setValue(225)
        self._cycle_seconds.setToolTip(S("step03.cycle_seconds.tooltip"))
        self._lbl_cyc = QLabel(S("step03.cycle_seconds"))
        self._lbl_cyc.setToolTip(S("step03.cycle_seconds.tooltip"))
        fl.addRow(self._lbl_cyc, self._cycle_seconds)

        # Min quality threshold
        self._min_quality = QDoubleSpinBox()
        self._min_quality.setStyleSheet(_SPINBOX_STYLE)
        self._min_quality.setRange(0.0, 1.0)
        self._min_quality.setDecimals(2)
        self._min_quality.setSingleStep(0.05)
        self._min_quality.setValue(0.05)
        self._min_quality.setToolTip(S("step03.min_quality.tooltip"))
        lbl_mq = QLabel(S("step03.min_quality"))
        lbl_mq.setToolTip(S("step03.min_quality.tooltip"))
        fl.addRow(lbl_mq, self._min_quality)

        idx = self._form_layout.count() - 1
        self._form_layout.insertWidget(idx, form_widget)

    def get_config_updates(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "input_dir":                self._input_lbl.text().strip(),
            "window_frames":            self._window_frames.value(),
            "cycle_seconds":                  self._cycle_seconds.value(),
            "quality_min_quality_threshold":  self._min_quality.value(),
        }
        out_text = self._output_lbl.text().strip()
        if out_text:
            result["output_dir"] = str(Path(out_text).parent)
        return result

    def validate(self, config: dict, batch_mode: bool = False) -> list:
        from gui.validation import ValidationIssue, count_files
        issues = []
        input_dir = config.get("input_dir", "").strip()
        if not batch_mode:
            if not input_dir:
                issues.append(ValidationIssue("error", S("validate.no_tif_dir")))
                return issues
            n_tif = count_files(input_dir, "*.tif", "*.TIF")
            if not n_tif:
                issues.append(ValidationIssue("error", S("validate.no_tif_files", d=input_dir)))
                return issues
            window_frames = int(config.get("window_frames", 3))
            if window_frames > n_tif:
                issues.append(ValidationIssue(
                    "error",
                    S("validate.files_insufficient",
                      wf=window_frames, nw=1, req=window_frames, n=n_tif),
                ))
        threshold = float(config.get("quality_min_quality_threshold",
                                      config.get("min_quality_threshold_03", 0.05)))
        if threshold > 0.5:
            issues.append(ValidationIssue(
                "warning",
                S("validate.high_threshold", t=threshold),
            ))
        return issues

    def load_session(self, data: dict[str, Any]) -> None:
        inp = data.get("input_dir", "")
        out = data.get("output_dir", "")
        self._input_lbl.blockSignals(True)
        self._input_lbl.setText(inp)
        self._input_lbl.blockSignals(False)
        self._update_input_style(inp)
        if hasattr(self, "_input_dot"):
            self._input_dot.check(inp, ["*.tif", "*.TIF"])
        if out:
            self._output_lbl.setText(str(Path(out) / "step03_quality"))

        is_color = data.get("camera_mode", "mono") == "color"
        if is_color:
            self._lbl_cyc.setText(S("step03.cycle_seconds_color"))
            self._cycle_seconds.setToolTip(S("step03.cycle_seconds_color.tooltip"))
            self._lbl_cyc.setToolTip(S("step03.cycle_seconds_color.tooltip"))
            if not data.get("cycle_seconds"):
                self._cycle_seconds.setValue(45)
        else:
            self._lbl_cyc.setText(S("step03.cycle_seconds"))
            self._cycle_seconds.setToolTip(S("step03.cycle_seconds.tooltip"))
            self._lbl_cyc.setToolTip(S("step03.cycle_seconds.tooltip"))
        # New key: window_frames. Old keys: window_cycles, window_seconds (convert on load).
        if "window_frames" in data:
            self._window_frames.setValue(int(data["window_frames"]))
        elif "window_cycles" in data:
            self._window_frames.setValue(int(data["window_cycles"]))
        else:
            old_ws = int(data.get("window_seconds", 900))
            old_cs = int(data.get("cycle_seconds", 270))
            self._window_frames.setValue(max(1, round(old_ws / old_cs)))
        self._cycle_seconds.setValue(int(data.get("cycle_seconds", 225)))
        # Support old keys (min_quality_threshold_03/_04, top_fraction) for
        # sessions saved before the quality_min_quality_threshold rename
        # (2026-08-09) — disambiguates from DerotationConfig's own,
        # unrelated min_quality_threshold session key.
        mq = data.get("quality_min_quality_threshold",
                      data.get("min_quality_threshold_03",
                               data.get("min_quality_threshold_04",
                                        data.get("top_fraction", 0.05))))
        self._min_quality.setValue(float(mq))

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_input_changed(self, text: str) -> None:
        self._update_input_style(text)
        if hasattr(self, "_input_dot"):
            self._input_dot.check(text.strip(), ["*.tif", "*.TIF"])

    def _update_input_style(self, text: str) -> None:
        if text.strip():
            self._input_lbl.setStyleSheet(_INPUT_STYLE)
        else:
            self._input_lbl.setStyleSheet(_INPUT_EMPTY_STYLE)

    def output_paths(self) -> list[Path]:
        if self._output_dir is None:
            return []
        step_dir = self._output_dir / "step03_quality"
        if not step_dir.exists():
            return []
        return sorted(step_dir.glob("*.csv"))

    def set_output_dir(self, path: Path | str) -> None:
        self._output_dir = Path(path) if path else None
