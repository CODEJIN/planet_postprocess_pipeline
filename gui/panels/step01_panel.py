"""Step 1 — PIPP preprocessing panel."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from gui.i18n import S
from gui.panels.base_panel import BasePanel
from gui.widgets.ser_preview import SerPreviewWidget

_SPINBOX_STYLE = (
    "QSpinBox { background: #3c3c3c; color: #d4d4d4; border: 1px solid #555;"
    " border-radius: 3px; padding: 3px 6px; }"
    "QSpinBox:focus { border-color: #4da6ff; }"
)
_INPUT_STYLE = (
    "QLineEdit { background: #3c3c3c; color: #d4d4d4; border: 1px solid #555;"
    " border-radius: 3px; padding: 3px 6px; }"
    "QLineEdit:focus { border-color: #4da6ff; }"
)
_BTN_BROWSE = (
    "QPushButton { background: #3c3c3c; color: #aaa; border: 1px solid #555;"
    " border-radius: 3px; padding: 3px 8px; }"
    "QPushButton:hover { background: #4a4a4a; color: #d4d4d4; }"
)
_LABEL_STYLE = "color: #d4d4d4; font-size: 12px;"


def _dir_row(parent: QWidget, line_edit: QLineEdit) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setSpacing(4)
    row.addWidget(line_edit)
    btn = QPushButton(S("btn.browse"))
    btn.setFixedWidth(70)
    btn.setStyleSheet(_BTN_BROWSE)

    def _browse():
        current = line_edit.text().strip()
        folder = QFileDialog.getExistingDirectory(
            parent, "폴더 선택", current or str(Path.home())
        )
        if folder:
            line_edit.setText(folder)
            line_edit.editingFinished.emit()

    btn.clicked.connect(_browse)
    row.addWidget(btn)
    return row


class Step01Panel(BasePanel):
    STEP_ID   = "01"
    TITLE_KEY = "step01.title"
    DESC_KEY  = "step01.desc"
    OPTIONAL  = True

    # Emitted when output dir changes (editingFinished) so downstream panels
    # can refresh their auto-derived path labels immediately.
    dirs_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        self._output_dir: Path | None = None
        self._output_manually_edited = False
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

        form_widget = QWidget()
        form_widget.setStyleSheet("background: transparent;")
        fl = QFormLayout(form_widget)
        fl.setSpacing(10)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        # SER input directory
        self._ser_dir = QLineEdit()
        self._ser_dir.setStyleSheet(_INPUT_STYLE)
        self._ser_dir.setPlaceholderText(S("step01.ser_dir.placeholder"))
        self._ser_dir.setToolTip(S("step01.ser_dir.tooltip"))
        self._ser_dir.textChanged.connect(self._on_ser_dir_changed)
        self._ser_dir.editingFinished.connect(self._auto_set_step1_output)
        lbl_ser = QLabel(S("step01.ser_dir"))
        lbl_ser.setToolTip(S("step01.ser_dir.tooltip"))
        fl.addRow(lbl_ser, _dir_row(self, self._ser_dir))

        # Output directory
        self._output_step1 = QLineEdit()
        self._output_step1.setStyleSheet(_INPUT_STYLE)
        self._output_step1.setPlaceholderText("자동 설정됩니다")
        self._output_step1.textEdited.connect(self._on_output_manually_edited)
        self._output_step1.editingFinished.connect(self.dirs_changed)
        lbl_out = QLabel(S("step01.output_dir"))
        lbl_out.setToolTip(S("step01.output_dir.tooltip"))
        fl.addRow(lbl_out, _dir_row(self, self._output_step1))

        # ROI size
        self._roi_size = QSpinBox()
        self._roi_size.setStyleSheet(_SPINBOX_STYLE)
        self._roi_size.setRange(64, 1024)
        self._roi_size.setSingleStep(16)
        self._roi_size.setValue(448)
        self._roi_size.setToolTip(S("step01.roi_size.tooltip"))
        self._roi_size.valueChanged.connect(self._on_params_changed)
        lbl_roi = QLabel(S("step01.roi_size"))
        lbl_roi.setToolTip(S("step01.roi_size.tooltip"))
        fl.addRow(lbl_roi, self._roi_size)

        # Min diameter
        self._min_diameter = QSpinBox()
        self._min_diameter.setStyleSheet(_SPINBOX_STYLE)
        self._min_diameter.setRange(10, 500)
        self._min_diameter.setSingleStep(5)
        self._min_diameter.setValue(50)
        self._min_diameter.setToolTip(S("step01.min_diameter.tooltip"))
        self._min_diameter.valueChanged.connect(self._on_params_changed)
        lbl_diam = QLabel(S("step01.min_diameter"))
        lbl_diam.setToolTip(S("step01.min_diameter.tooltip"))
        fl.addRow(lbl_diam, self._min_diameter)

        left_layout.addWidget(form_widget)
        left_layout.addStretch()
        main_hlayout.addWidget(left_widget, 1)

        # ── Right: preview ──────────────────────────────────────────────────
        self._preview = SerPreviewWidget(parent=self)
        main_hlayout.addWidget(self._preview, 0)

        idx = self._form_layout.count() - 1
        self._form_layout.insertWidget(idx, main_widget)

    def retranslate(self) -> None:
        self._preview.retranslate()

    def get_config_updates(self) -> dict[str, Any]:
        step1_out = self._output_step1.text().strip()
        result: dict[str, Any] = {
            "ser_input_dir":     self._ser_dir.text().strip(),
            "step01_output_dir": step1_out,
            "roi_size":          self._roi_size.value(),
            "min_diameter":      self._min_diameter.value(),
        }
        # Only propagate output_dir when Step 1 is actually configured; an empty
        # value would overwrite the output_dir set by the Step 2/3 cascade.
        if step1_out:
            result["output_dir"] = str(Path(step1_out).parent)
        return result

    def load_session(self, data: dict[str, Any]) -> None:
        ser_dir = data.get("ser_input_dir", "")

        self._ser_dir.blockSignals(True)
        self._ser_dir.setText(ser_dir)
        self._ser_dir.blockSignals(False)

        self._roi_size.blockSignals(True)
        self._roi_size.setValue(int(data.get("roi_size", 448)))
        self._roi_size.blockSignals(False)

        self._min_diameter.blockSignals(True)
        self._min_diameter.setValue(int(data.get("min_diameter", 50)))
        self._min_diameter.blockSignals(False)

        step1_out = data.get("step01_output_dir", "")
        if step1_out:
            self._output_manually_edited = True
            self._output_step1.setText(step1_out)
            self._output_dir = Path(step1_out).parent
        else:
            out = data.get("output_dir", "")
            if out and not self._output_manually_edited:
                self._output_step1.setText(str(Path(out) / "step01_pipp"))
                self._output_dir = Path(out)

        # Sync preview
        if hasattr(self, "_preview"):
            self._preview.set_params(
                roi_size=int(data.get("roi_size", 448)),
                min_diameter=int(data.get("min_diameter", 50)),
            )
            self._preview.set_input_dir(ser_dir or None)

    def output_paths(self) -> list[Path]:
        step1_out = self._output_step1.text().strip() if hasattr(self, "_output_step1") else ""
        if step1_out:
            p = Path(step1_out)
            if p.exists():
                return sorted(p.glob("*.ser"))
        if self._output_dir is None:
            return []
        step_dir = self._output_dir / "step01_pipp"
        if not step_dir.exists():
            return []
        return sorted(step_dir.glob("*.ser"))

    def set_output_dir(self, path: Path | str) -> None:
        if not self._output_manually_edited:
            self._output_dir = Path(path) if path else None

    # ── Qt events ─────────────────────────────────────────────────────────────

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if hasattr(self, "_preview"):
            self._preview.schedule_update(150)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_ser_dir_changed(self, text: str) -> None:
        if hasattr(self, "_preview"):
            self._preview.set_input_dir(text.strip() or None)

    def _auto_set_step1_output(self) -> None:
        """Auto-set step01 output to a sub-folder of the SER directory on focus-out.

        Always updates regardless of _output_manually_edited — changing the input
        directory is an explicit user action that should drive the output path.
        """
        t = self._ser_dir.text().strip()
        if not t:
            return
        derived = str(Path(t) / "step01_pipp")
        self._output_step1.setText(derived)
        self._output_dir = Path(t)
        self.dirs_changed.emit()

    def _on_params_changed(self) -> None:
        if not hasattr(self, "_preview"):
            return
        self._preview.set_params(
            roi_size=self._roi_size.value(),
            min_diameter=self._min_diameter.value(),
        )
        self._preview.schedule_update()

    def _on_output_manually_edited(self, _text: str) -> None:
        self._output_manually_edited = True
