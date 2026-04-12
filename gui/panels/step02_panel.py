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
    "QSpinBox { background: #3c3c3c; color: #d4d4d4; border: 1px solid #555;"
    " border-radius: 3px; padding: 3px 6px; }"
    "QSpinBox:focus { border-color: #4da6ff; }"
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
            parent, "폴더 선택", current or str(Path.home())
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
    OPTIONAL  = False

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
        self._ser_dir.setToolTip(
            "PIPP 처리된 SER 파일이 있는 폴더입니다.\n"
            "Step 1을 실행했다면 자동으로 설정됩니다."
        )
        self._ser_dir.textChanged.connect(self._on_ser_dir_changed)
        self._ser_dir.editingFinished.connect(self._on_ser_editing_finished)
        lbl_ser = QLabel(S("step02.ser_dir"))
        fl.addRow(lbl_ser, _dir_row(self, self._ser_dir))

        # ── Step02 output directory ───────────────────────────────────────────
        self._output_step2 = QLineEdit()
        self._output_step2.setStyleSheet(_INPUT_STYLE)
        self._output_step2.setPlaceholderText("자동 설정됩니다")
        self._output_step2.setToolTip(
            "스태킹된 TIF 파일이 저장될 폴더입니다.\n"
            "출력 폴더는 Step 3의 입력으로 자동 연결됩니다."
        )
        self._output_step2.textEdited.connect(self._on_output_manually_edited)
        self._output_step2.editingFinished.connect(self.dirs_changed)
        lbl_out = QLabel(S("step02.output_dir"))
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
        lbl_top.setToolTip(
            "품질 점수 기준 상위 N%의 프레임만 스태킹에 사용합니다.\n"
            "낮을수록 더 선명하지만 노이즈가 많아집니다 (AS!4 기본값: 25%)."
        )
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
        lbl_ap.setToolTip(
            "로컬 정렬에 사용할 패치(AP) 크기입니다 (픽셀).\n"
            "64px = AS!4 기본값. 32px = 더 세밀, 더 느림."
        )
        ap_row = QHBoxLayout()
        ap_row.setSpacing(4)
        ap_row.addWidget(self._ap_size)
        ap_row.addStretch()
        fl.addRow(lbl_ap, ap_row)

        # ── n_iterations ──────────────────────────────────────────────────────
        self._n_iterations = QSpinBox()
        self._n_iterations.setStyleSheet(_SPINBOX_STYLE)
        self._n_iterations.setRange(1, 3)
        self._n_iterations.setValue(2)
        self._n_iterations.setFixedWidth(80)
        lbl_iter = QLabel(S("step02.n_iterations"))
        lbl_iter.setToolTip(
            "스태킹 반복 횟수입니다.\n"
            "2회 = 첫 스택을 reference로 재사용 → AP 정밀도 향상 (권장).\n"
            "1회 = 빠르지만 정확도 낮음."
        )
        iter_row = QHBoxLayout()
        iter_row.setSpacing(4)
        iter_row.addWidget(self._n_iterations)
        iter_row.addStretch()
        fl.addRow(lbl_iter, iter_row)

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
            "step02_ser_dir":     self._ser_dir.text().strip(),
            "step02_output_dir":  self._output_step2.text().strip(),
            "lucky_top_percent":  self._top_percent.value() / 100.0,
            "lucky_ap_size":      self._ap_size.value(),
            "lucky_n_iterations": self._n_iterations.value(),
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
        self._n_iterations.setValue(int(data.get("lucky_n_iterations", 2)))

        # Sync preview
        if hasattr(self, "_preview"):
            self._preview.set_input_dir(ser_value or None)
            self._preview.set_params(ap_size=int(data.get("lucky_ap_size", 64)))

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
        self._preview.set_params(ap_size=self._ap_size.value())

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
