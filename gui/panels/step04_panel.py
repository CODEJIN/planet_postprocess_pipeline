"""Step 4 — Quality assessment panel."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QWidget,
)

from gui.i18n import S
from gui.panels.base_panel import BasePanel

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
_CHECK_STYLE = (
    "QCheckBox { color: #d4d4d4; }"
    "QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #666;"
    " border-radius: 2px; background: #3c3c3c; }"
    "QCheckBox::indicator:checked { background: #4da6ff; border-color: #4da6ff; }"
    "QCheckBox::indicator:unchecked { background: #2a2a2a; border-color: #555; }"
)
_READONLY_STYLE = (
    "QLineEdit { background: #2a2a2a; color: #888; border: 1px solid #3a3a3a;"
    " border-radius: 3px; padding: 3px 6px; }"
)


class Step04Panel(BasePanel):
    STEP_ID   = "04"
    TITLE_KEY = "step04.title"
    DESC_KEY  = "step04.desc"
    OPTIONAL  = False

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

        # Folder display (auto-derived, read-only)
        self._input_lbl = QLineEdit()
        self._input_lbl.setReadOnly(True)
        self._input_lbl.setStyleSheet(_READONLY_STYLE)
        lbl_in = QLabel(S("step04.input_dir"))
        lbl_in.setToolTip(
            "Step 4가 품질을 평가할 TIF 파일 폴더입니다.\n"
            "Step 3의 입력 폴더와 동일한 AS!4 출력 폴더를 사용합니다."
        )
        fl.addRow(lbl_in, self._input_lbl)

        self._output_lbl = QLineEdit()
        self._output_lbl.setReadOnly(True)
        self._output_lbl.setStyleSheet(_READONLY_STYLE)
        lbl_out = QLabel(S("step04.output_dir"))
        lbl_out.setToolTip(
            "품질 점수 CSV, 윈도우 추천 JSON이 저장될 폴더입니다.\n"
            "자동으로 설정됩니다."
        )
        fl.addRow(lbl_out, self._output_lbl)

        # Window frames
        _tip_win = (
            "De-rotation 윈도우의 프레임 수입니다.\n"
            "1 프레임 = 필터 한 사이클(IR→R→G→B→CH4 한 바퀴) = Step 8의 프레임 단위와 동일\n"
            "실제 윈도우 길이 = 프레임 수 × 필터 사이클 시간\n"
            "예: 3프레임 × 225초 = 675초(약 11분)\n\n"
            "이 구간에서 모든 필터의 품질 점수가 동시에 높은 윈도우를 찾습니다.\n"
            "목성 권장: 2~4프레임 / 화성·토성: 3~6프레임"
        )
        self._window_frames = QSpinBox()
        self._window_frames.setStyleSheet(_INT_SPINBOX_STYLE)
        self._window_frames.setRange(1, 20)
        self._window_frames.setSingleStep(1)
        self._window_frames.setValue(3)
        self._window_frames.setToolTip(_tip_win)
        lbl_win = QLabel(S("step04.window_frames"))
        lbl_win.setToolTip(_tip_win)
        fl.addRow(lbl_win, self._window_frames)

        # Cycle seconds
        _tip_cyc = (
            "필터 한 사이클(IR→R→G→B→CH4→IR)에 걸리는 시간(초)입니다.\n"
            "실제 촬영 패턴에 맞춰 입력하세요.\n"
            "예: 45초 × 5필터 = 225초\n\n"
            "이 값은 Step 4의 de-rotation 윈도우 길이 계산에만 사용됩니다.\n"
            "Step 8의 사이클 시간은 Step 8 패널에서 별도로 설정합니다."
        )
        self._cycle_seconds = QSpinBox()
        self._cycle_seconds.setStyleSheet(_INT_SPINBOX_STYLE)
        self._cycle_seconds.setRange(10, 600)
        self._cycle_seconds.setSingleStep(15)
        self._cycle_seconds.setValue(225)
        self._cycle_seconds.setToolTip(_tip_cyc)
        self._lbl_cyc = QLabel(S("step04.cycle_seconds"))
        self._lbl_cyc.setToolTip(_tip_cyc)
        fl.addRow(self._lbl_cyc, self._cycle_seconds)

        # n_windows
        _tip_nwin = (
            "찾을 최적 윈도우 개수입니다.\n"
            "1 = 가장 좋은 윈도우 하나만 사용 (기본, Step 5 스태킹)\n"
            "2~3 = 여러 윈도우를 찾아 Step 8 시계열 합성에 활용\n"
            "단독 스태킹이 목적이면 1로 두세요."
        )
        self._n_windows = QSpinBox()
        self._n_windows.setStyleSheet(_INT_SPINBOX_STYLE)
        self._n_windows.setRange(1, 10)
        self._n_windows.setSingleStep(1)
        self._n_windows.setValue(1)
        self._n_windows.setToolTip(_tip_nwin)
        lbl_nwin = QLabel(S("step04.n_windows"))
        lbl_nwin.setToolTip(_tip_nwin)
        fl.addRow(lbl_nwin, self._n_windows)

        # Allow overlap checkbox
        _tip_overlap = (
            "체크 시: 여러 윈도우가 시간 범위를 겹칠 수 있습니다.\n"
            "체크 해제 시: 각 윈도우는 서로 겹치지 않는 시간대에서 선택됩니다 (기본).\n\n"
            "윈도우 개수 ≥ 2일 때만 의미가 있습니다.\n"
            "비활성 상태(기본)에서는 시간적으로 고르게 분산된 윈도우를 찾습니다."
        )
        self._allow_overlap = QCheckBox()
        self._allow_overlap.setStyleSheet(_CHECK_STYLE)
        self._allow_overlap.setChecked(False)
        self._allow_overlap.setToolTip(_tip_overlap)
        lbl_overlap = QLabel(S("step04.allow_overlap"))
        lbl_overlap.setToolTip(_tip_overlap)
        fl.addRow(lbl_overlap, self._allow_overlap)

        # Min quality threshold
        _tip_mq = (
            "이 품질 점수 미만인 프레임을 윈도우 탐색에서 제외합니다.\n"
            "0.0 = 전체 프레임 사용 (기본, 비활성)\n"
            "0.2~0.3 = 명백히 나쁜 프레임(구름, 흔들림) 제거\n"
            "너무 높게 설정하면 프레임 수가 부족해질 수 있습니다.\n"
            "※ 스태킹 비율이 아닙니다 — Step 5에서 별도로 설정합니다."
        )
        self._min_quality = QDoubleSpinBox()
        self._min_quality.setStyleSheet(_SPINBOX_STYLE)
        self._min_quality.setRange(0.0, 1.0)
        self._min_quality.setDecimals(2)
        self._min_quality.setSingleStep(0.05)
        self._min_quality.setValue(0.05)
        self._min_quality.setToolTip(_tip_mq)
        lbl_mq = QLabel(S("step04.min_quality"))
        lbl_mq.setToolTip(_tip_mq)
        fl.addRow(lbl_mq, self._min_quality)

        idx = self._form_layout.count() - 1
        self._form_layout.insertWidget(idx, form_widget)

    def get_config_updates(self) -> dict[str, Any]:
        return {
            "window_frames":            self._window_frames.value(),
            "cycle_seconds":            self._cycle_seconds.value(),
            "n_windows":                self._n_windows.value(),
            "allow_overlap":            self._allow_overlap.isChecked(),
            "min_quality_threshold_04": self._min_quality.value(),
        }

    def load_session(self, data: dict[str, Any]) -> None:
        inp = data.get("input_dir", "")
        out = data.get("output_dir", "")
        if inp:
            self._input_lbl.setText(inp)
        if out:
            self._output_lbl.setText(str(Path(out) / "step04_quality"))

        is_color = data.get("camera_mode", "mono") == "color"
        if is_color:
            self._lbl_cyc.setText(S("step04.cycle_seconds_color"))
            self._cycle_seconds.setToolTip(
                "RGB 프레임 한 장 촬영에 걸리는 시간(초)입니다.\n"
                "윈도우 길이 = 프레임 수 × 이 값\n"
                "예: 프레임 수 3 × 45초 = 135초(약 2.25분)"
            )
            if not data.get("cycle_seconds"):
                self._cycle_seconds.setValue(45)
        else:
            self._lbl_cyc.setText(S("step04.cycle_seconds"))
            self._cycle_seconds.setToolTip(
                "필터 한 사이클(IR→R→G→B→CH4→IR)에 걸리는 시간(초)입니다.\n"
                "실제 촬영 패턴에 맞춰 입력하세요.\n"
                "예: 45초 × 5필터 = 225초\n\n"
                "이 값은 Step 4의 de-rotation 윈도우 길이 계산에만 사용됩니다.\n"
                "Step 8의 사이클 시간은 Step 8 패널에서 별도로 설정합니다."
            )
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
        self._n_windows.setValue(int(data.get("n_windows", 1)))
        self._allow_overlap.setChecked(bool(data.get("allow_overlap", False)))
        # Support both old key (top_fraction) and new key
        mq = data.get("min_quality_threshold_04",
                      data.get("top_fraction", 0.05))
        self._min_quality.setValue(float(mq))

    def output_paths(self) -> list[Path]:
        if self._output_dir is None:
            return []
        step_dir = self._output_dir / "step04_quality"
        if not step_dir.exists():
            return []
        return sorted(step_dir.glob("*.csv"))

    def set_output_dir(self, path: Path | str) -> None:
        self._output_dir = Path(path) if path else None
