"""Step 2 — Lucky Stacking panel (SER → TIF).

Runs the Python lucky_stack pipeline directly (replaces manual AS!4 step).
Inherits BasePanel for consistent Run / Next / progress-bar UI.
Emits ``dirs_changed`` after a successful run so Step 3's input folder
is auto-populated.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
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
from gui.widgets.lucky_stack_preview import LuckyStackPreviewWidget

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
_SPINBOX_STYLE = (
    "QSpinBox, QDoubleSpinBox { background: #3c3c3c; color: #d4d4d4; border: 1px solid #555;"
    " border-radius: 3px; padding: 3px 6px; }"
    "QSpinBox:focus, QDoubleSpinBox:focus { border-color: #4da6ff; }"
)


def _dir_row(parent: QWidget, line_edit: QLineEdit) -> QHBoxLayout:
    """Return (layout) containing a line-edit and a Browse button."""
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


class Step02Panel(BasePanel):
    """Lucky Stacking panel: selects best SER frames and stacks to TIF."""

    STEP_ID   = "02"
    TITLE_KEY = "step02.title"
    DESC_KEY  = "step02.desc"
    OPTIONAL  = True

    # Emitted after output dir changes so Step 3 can pick it up as input.
    dirs_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        self._output_manually_edited = False
        super().__init__(parent)

    # ── BasePanel interface ───────────────────────────────────────────────────

    def build_form(self) -> None:
        # ── Horizontal split: controls (left) | preview (right) ─────────────
        main_widget = QWidget()
        main_widget.setStyleSheet("background: transparent;")
        main_hlayout = QHBoxLayout(main_widget)
        main_hlayout.setSpacing(16)
        main_hlayout.setContentsMargins(0, 0, 0, 0)

        # ── Left: controls ───────────────────────────────────────────────────
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

        # ── SER input directory ───────────────────────────────────────────────
        self._ser_dir = QLineEdit()
        self._ser_dir.setStyleSheet(_INPUT_EMPTY_STYLE)
        self._ser_dir.setPlaceholderText(S("step02.ser_dir.placeholder"))
        self._ser_dir.setToolTip(S("step02.ser_dir.tooltip"))
        self._ser_dir.textChanged.connect(self._on_ser_dir_changed)
        self._ser_dir.editingFinished.connect(self._on_ser_editing_finished)
        lbl_ser = QLabel(S("step02.ser_dir"))
        lbl_ser.setToolTip(S("step02.ser_dir.tooltip"))
        fl.addRow(lbl_ser, _dir_row(self, self._ser_dir))

        # ── Step02 output directory ───────────────────────────────────────────
        self._output_step2 = QLineEdit()
        self._output_step2.setStyleSheet(_INPUT_STYLE)
        self._output_step2.setPlaceholderText(S("placeholder.auto_set"))
        self._output_step2.setToolTip(S("step02.output_dir.tooltip"))
        self._output_step2.textEdited.connect(self._on_output_manually_edited)
        self._output_step2.editingFinished.connect(self.dirs_changed)
        lbl_out = QLabel(S("step02.output_dir"))
        lbl_out.setToolTip(S("step02.output_dir.tooltip"))
        fl.addRow(lbl_out, _dir_row(self, self._output_step2))

        # ── top_percent (displayed as %) ──────────────────────────────────────
        self._top_percent = QSpinBox()
        self._top_percent.setStyleSheet(_SPINBOX_STYLE)
        self._top_percent.setRange(5, 100)
        self._top_percent.setSingleStep(5)
        self._top_percent.setSuffix(" %")
        self._top_percent.setValue(25)
        self._top_percent.setFixedWidth(80)
        lbl_top = QLabel(S("step02.top_percent"))
        lbl_top.setToolTip(S("step02.top_percent.tooltip"))
        top_row = QHBoxLayout()
        top_row.setSpacing(4)
        top_row.addWidget(self._top_percent)
        top_row.addStretch()
        fl.addRow(lbl_top, top_row)

        # ── ap_size ───────────────────────────────────────────────────────────
        self._ap_size = QSpinBox()
        self._ap_size.setStyleSheet(_SPINBOX_STYLE)
        self._ap_size.setRange(32, 128)
        self._ap_size.setSingleStep(32)
        self._ap_size.setValue(64)
        self._ap_size.setFixedWidth(80)
        self._ap_size.valueChanged.connect(self._on_ap_params_changed)
        lbl_ap = QLabel(S("step02.ap_size"))
        lbl_ap.setToolTip(S("step02.ap_size.tooltip"))
        ap_row = QHBoxLayout()
        ap_row.setSpacing(4)
        ap_row.addWidget(self._ap_size)
        ap_row.addStretch()
        fl.addRow(lbl_ap, ap_row)

        # ── n_iterations ──────────────────────────────────────────────────────
        self._n_iterations = QSpinBox()
        self._n_iterations.setStyleSheet(_SPINBOX_STYLE)
        self._n_iterations.setRange(1, 2)
        self._n_iterations.setValue(1)
        self._n_iterations.setFixedWidth(80)
        lbl_iter = QLabel(S("step02.n_iterations"))
        lbl_iter.setToolTip(S("step02.n_iterations.tooltip"))
        iter_row = QHBoxLayout()
        iter_row.setSpacing(4)
        iter_row.addWidget(self._n_iterations)
        iter_row.addStretch()
        fl.addRow(lbl_iter, iter_row)

        # ── warp_method ───────────────────────────────────────────────────────
        _chk_style = ("QCheckBox { color: #d4d4d4; }"
                      "QCheckBox::indicator:checked"
                      " { background: #4da6ff; border-color: #4da6ff; }"
                      "QCheckBox::indicator:unchecked"
                      " { background: #2a2a2a; border-color: #555; }")
        _combo_style = ("QComboBox { background: #2a2a2a; color: #d4d4d4;"
                        " border: 1px solid #555; border-radius: 3px; padding: 2px 6px; }"
                        "QComboBox::drop-down { border: none; }"
                        "QComboBox QAbstractItemView { background: #2a2a2a; color: #d4d4d4;"
                        " selection-background-color: #3a3a5a; }")
        self._warp_method = QComboBox()
        self._warp_method.setStyleSheet(_combo_style)
        self._warp_method.addItem(S("step02.warp_method.gaussian_kr"), userData="gaussian_kr")
        self._warp_method.addItem(S("step02.warp_method.tps"),         userData="tps")
        self._warp_method.setFixedWidth(140)
        lbl_warp = QLabel(S("step02.warp_method"))
        lbl_warp.setToolTip(S("step02.warp_method.tooltip"))
        self._warp_method.setToolTip(S("step02.warp_method.tooltip"))
        warp_row = QHBoxLayout()
        warp_row.setSpacing(4)
        warp_row.addWidget(self._warp_method)
        warp_row.addStretch()
        fl.addRow(lbl_warp, warp_row)

        # ── fourier_quality_power ─────────────────────────────────────────────
        self._fourier_power = QDoubleSpinBox()
        self._fourier_power.setStyleSheet(_SPINBOX_STYLE)
        self._fourier_power.setRange(0.5, 3.0)
        self._fourier_power.setSingleStep(0.5)
        self._fourier_power.setDecimals(1)
        self._fourier_power.setValue(1.0)
        self._fourier_power.setFixedWidth(80)
        lbl_fourier = QLabel(S("step02.fourier_quality_power"))
        lbl_fourier.setToolTip(S("step02.fourier_quality_power.tooltip"))
        self._fourier_power.setToolTip(S("step02.fourier_quality_power.tooltip"))
        fourier_row = QHBoxLayout()
        fourier_row.setSpacing(4)
        fourier_row.addWidget(self._fourier_power)
        fourier_row.addStretch()
        fl.addRow(lbl_fourier, fourier_row)

        # ── n_ser_parallel ────────────────────────────────────────────────────
        self._n_ser_parallel = QSpinBox()
        self._n_ser_parallel.setStyleSheet(_SPINBOX_STYLE)
        self._n_ser_parallel.setRange(0, 32)
        self._n_ser_parallel.setValue(1)
        self._n_ser_parallel.setFixedWidth(80)
        lbl_ser_par = QLabel(S("step02.n_ser_parallel"))
        lbl_ser_par.setToolTip(S("step02.n_ser_parallel.tooltip"))
        self._n_ser_parallel.setToolTip(S("step02.n_ser_parallel.tooltip"))
        ser_par_row = QHBoxLayout()
        ser_par_row.setSpacing(4)
        ser_par_row.addWidget(self._n_ser_parallel)
        ser_par_row.addStretch()
        fl.addRow(lbl_ser_par, ser_par_row)

        # ── use_as4_ap_grid ───────────────────────────────────────────────────
        self._use_as4_ap_grid = QCheckBox()
        self._use_as4_ap_grid.setChecked(False)
        self._use_as4_ap_grid.setStyleSheet(_chk_style)
        self._use_as4_ap_grid.toggled.connect(self._on_as4_grid_toggled)
        lbl_as4 = QLabel(S("step02.use_as4_ap_grid"))
        lbl_as4.setToolTip(S("step02.use_as4_ap_grid.tooltip"))
        self._use_as4_ap_grid.setToolTip(S("step02.use_as4_ap_grid.tooltip"))
        as4_row = QHBoxLayout()
        as4_row.setSpacing(4)
        as4_row.addWidget(self._use_as4_ap_grid)
        as4_row.addStretch()
        fl.addRow(lbl_as4, as4_row)

        # ── use_ncc ───────────────────────────────────────────────────────────
        self._use_ncc = QCheckBox()
        self._use_ncc.setChecked(False)
        self._use_ncc.setStyleSheet(_chk_style)
        lbl_ncc = QLabel(S("step02.use_ncc"))
        lbl_ncc.setToolTip(S("step02.use_ncc.tooltip"))
        self._use_ncc.setToolTip(S("step02.use_ncc.tooltip"))
        ncc_row = QHBoxLayout()
        ncc_row.setSpacing(4)
        ncc_row.addWidget(self._use_ncc)
        ncc_row.addStretch()
        fl.addRow(lbl_ncc, ncc_row)

        # ── per_ap_selection ──────────────────────────────────────────────────
        self._per_ap_selection = QCheckBox()
        self._per_ap_selection.setChecked(False)
        self._per_ap_selection.setStyleSheet(_chk_style)
        lbl_pas = QLabel(S("step02.per_ap_selection"))
        lbl_pas.setToolTip(S("step02.per_ap_selection.tooltip"))
        self._per_ap_selection.setToolTip(S("step02.per_ap_selection.tooltip"))
        pas_row = QHBoxLayout()
        pas_row.setSpacing(4)
        pas_row.addWidget(self._per_ap_selection)
        pas_row.addStretch()
        fl.addRow(lbl_pas, pas_row)

        left_layout.addWidget(form_widget)
        left_layout.addStretch()
        main_hlayout.addWidget(left_widget, 1)

        # ── Right: AP grid preview ────────────────────────────────────────────
        self._preview = LuckyStackPreviewWidget(parent=self)
        main_hlayout.addWidget(self._preview, 0)

        idx = self._form_layout.count() - 1
        self._form_layout.insertWidget(idx, main_widget)

    def get_config_updates(self) -> dict[str, Any]:
        return {
            "step02_ser_dir":         self._ser_dir.text().strip(),
            "step02_output_dir":      self._output_step2.text().strip(),
            "lucky_top_percent":      self._top_percent.value() / 100.0,
            "lucky_ap_size":          self._ap_size.value(),
            "lucky_n_iterations":     self._n_iterations.value(),
            "lucky_use_tps":          self._warp_method.currentData() == "tps",
            "lucky_use_as4_ap_grid":  self._use_as4_ap_grid.isChecked(),
            "lucky_use_ncc":          self._use_ncc.isChecked(),
            "lucky_per_ap_selection": self._per_ap_selection.isChecked(),
            "lucky_fourier_power":    self._fourier_power.value(),
            "lucky_n_ser_parallel":   self._n_ser_parallel.value(),
        }

    def load_session(self, data: dict[str, Any]) -> None:
        # SER input dir — prefer step02's own choice, then step01 output, then raw SER dir
        ser_value = (
            data.get("step02_ser_dir", "")
            or data.get("step01_output_dir", "")
            or data.get("ser_input_dir", "")
        )

        self._ser_dir.blockSignals(True)
        self._ser_dir.setText(ser_value)
        self._ser_dir.blockSignals(False)
        self._update_ser_style(ser_value)

        # Step02 output dir.
        # Reset the flag on every load so upstream cascade changes always trigger
        # re-derivation.  The flag is set True only by direct user UI interaction
        # (textEdited signal) — not by programmatic session restores.
        self._output_manually_edited = False
        step02_out = data.get("step02_output_dir", "")
        if step02_out:
            self._output_step2.setText(step02_out)
        elif ser_value:
            self._derive_output(ser_value)

        # Parameters
        top_pct = float(data.get("lucky_top_percent", 0.25))
        self._top_percent.setValue(int(round(top_pct * 100)))
        self._ap_size.setValue(int(data.get("lucky_ap_size", 64)))
        self._n_iterations.setValue(int(data.get("lucky_n_iterations", 1)))
        _use_tps = bool(data.get("lucky_use_tps", False))
        self._warp_method.setCurrentIndex(1 if _use_tps else 0)
        self._use_as4_ap_grid.setChecked(bool(data.get("lucky_use_as4_ap_grid", False)))
        self._use_ncc.setChecked(bool(data.get("lucky_use_ncc", False)))
        self._per_ap_selection.setChecked(bool(data.get("lucky_per_ap_selection", False)))
        self._fourier_power.setValue(float(data.get("lucky_fourier_power", 1.0)))
        self._n_ser_parallel.setValue(int(data.get("lucky_n_ser_parallel", 1)))

        # Sync preview
        if hasattr(self, "_preview"):
            self._preview.set_input_dir(ser_value or None)
            _ap_sz = int(data.get("lucky_ap_size", 64))
            self._preview.set_params(ap_size=_ap_sz, ap_step=_ap_sz // 2)
            self._preview.set_use_as4_ap_grid(bool(data.get("lucky_use_as4_ap_grid", False)))

    def validate(self, config: dict, batch_mode: bool = False) -> list:
        from gui.validation import ValidationIssue, count_files
        from gui.i18n import S
        issues = []
        ser_dir = (config.get("step02_ser_dir", "") or config.get("ser_input_dir", "")).strip()
        if not ser_dir:
            issues.append(ValidationIssue("error", S("validate.no_ser_dir")))
        elif not batch_mode and not count_files(ser_dir, "*.ser", "*.SER"):
            issues.append(ValidationIssue("error", S("validate.no_ser_files", d=ser_dir)))
        return issues

    def refresh_after_run(self) -> None:
        """After a successful run, emit dirs_changed so Step 3 picks up the output."""
        self.dirs_changed.emit()

    # ── Qt events ────────────────────────────────────────────────────────────

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if hasattr(self, "_preview"):
            self._preview.schedule_update(150)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_ser_dir_changed(self, text: str) -> None:
        self._update_ser_style(text)
        if hasattr(self, "_preview"):
            self._preview.set_input_dir(text.strip() or None)

    def _on_ap_params_changed(self) -> None:
        if not hasattr(self, "_preview") or not hasattr(self, "_ap_size"):
            return
        _ap_sz = self._ap_size.value()
        self._preview.set_params(ap_size=_ap_sz, ap_step=_ap_sz // 2)

    def _on_as4_grid_toggled(self, checked: bool) -> None:
        if hasattr(self, "_preview"):
            self._preview.set_use_as4_ap_grid(checked)

    def _on_ser_editing_finished(self) -> None:
        """On focus-out / Enter: update output dir and emit dirs_changed."""
        t = self._ser_dir.text().strip()
        if t:
            self._derive_output(t)
        self.dirs_changed.emit()

    def _on_output_manually_edited(self, _text: str) -> None:
        self._output_manually_edited = True

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _update_ser_style(self, text: str) -> None:
        if text.strip():
            self._ser_dir.setStyleSheet(_INPUT_STYLE)
        else:
            self._ser_dir.setStyleSheet(_INPUT_EMPTY_STYLE)

    def _derive_output(self, ser_dir: str) -> None:
        """Auto-set output to <parent of ser_dir>/step02_lucky_stack (unless manually edited)."""
        if self._output_manually_edited:
            return
        p = Path(ser_dir)
        derived = str(p.parent / "step02_lucky_stack")
        self._output_step2.setText(derived)
