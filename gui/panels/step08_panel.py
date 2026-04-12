"""Step 8 — Time-series composite panel.

Architecture: Step08Panel wraps a QStackedWidget with two sub-widgets:
  - _Step08MonoWidget  : mono camera multi-filter compositing (unchanged)
  - _Step08ColorWidget : color camera per-frame stacking settings

load_session() detects camera_mode and switches the internal stack accordingly.
main_window.py needs no changes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from gui.i18n import S
from gui.panels.base_panel import BasePanel
from gui.panels.step06_panel import _make_wavelet_row
from gui.panels.step07_panel import (
    _SpecRow,
    _BTN_ADD,
    _BTN_SMALL,
    _COMBO_STYLE,
    _NAME_STYLE,
)

_INFO_STYLE = (
    "QLabel { background: #2a2a2a; color: #aaa; border: 1px solid #444;"
    " border-radius: 4px; padding: 10px; font-size: 11px; }"
)
_READONLY_STYLE = (
    "QLineEdit { background: #2a2a2a; color: #888; border: 1px solid #3a3a3a;"
    " border-radius: 3px; padding: 3px 6px; }"
)
_SPINBOX_STYLE = (
    "QSpinBox { background: #3c3c3c; color: #d4d4d4; border: 1px solid #555;"
    " border-radius: 3px; padding: 3px 6px; }"
    "QSpinBox:focus { border-color: #4da6ff; }"
)
_DBLSPIN_STYLE = (
    "QDoubleSpinBox { background: #3c3c3c; color: #d4d4d4; border: 1px solid #555;"
    " border-radius: 3px; padding: 3px 6px; }"
    "QDoubleSpinBox:focus { border-color: #4da6ff; }"
)

_SERIES_WAVELET_DEFAULTS = [200.0, 200.0, 200.0, 0.0, 0.0, 0.0]

# Default composite specs for step 8 (mirrors step 7 defaults)
_DEFAULT_SERIES_SPECS = [
    {"name": "RGB",      "R": "R",   "G": "G", "B": "B",  "L": ""},
    {"name": "IR-RGB",   "R": "R",   "G": "G", "B": "B",  "L": "IR"},
    {"name": "CH4-G-IR", "R": "CH4", "G": "G", "B": "IR", "L": ""},
]

_SECTION_STYLE = "color: #4da6ff; font-size: 11px; font-weight: bold;"


# ── Mono sub-widget ────────────────────────────────────────────────────────────

class _Step08MonoWidget(QWidget):
    """Mono camera Step 8: multi-filter cycle compositing with independent composite specs."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._output_dir: Path | None = None
        self._spec_rows: list[_SpecRow] = []
        self._available_filters: list[str] = ["R", "G", "B", "IR", "CH4", "UV"]
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        # ── Folder display ────────────────────────────────────────────────────
        folder_widget = QWidget()
        folder_widget.setStyleSheet("background: transparent;")
        fl = QFormLayout(folder_widget)
        fl.setSpacing(8)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self._input_lbl = QLineEdit()
        self._input_lbl.setReadOnly(True)
        self._input_lbl.setStyleSheet(_READONLY_STYLE)
        lbl_in = QLabel(S("step08.input_dir"))
        lbl_in.setToolTip("Step 3 wavelet 미리보기 PNG 폴더 (자동 설정)")
        fl.addRow(lbl_in, self._input_lbl)

        self._output_lbl = QLineEdit()
        self._output_lbl.setReadOnly(True)
        self._output_lbl.setStyleSheet(_READONLY_STYLE)
        lbl_out = QLabel(S("step08.output_dir"))
        lbl_out.setToolTip("시계열 RGB 합성 PNG가 저장될 폴더 (자동 설정)")
        fl.addRow(lbl_out, self._output_lbl)

        root.addWidget(folder_widget)

        # ── Options ───────────────────────────────────────────────────────────
        opt_widget = QWidget()
        opt_widget.setStyleSheet("background: transparent;")
        opt_fl = QFormLayout(opt_widget)
        opt_fl.setSpacing(10)
        opt_fl.setContentsMargins(0, 0, 0, 0)
        opt_fl.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        _tip_norm = (
            "모든 프레임에 걸쳐 각 필터의 밝기 범위를 통일합니다.\n"
            "1패스: 전 프레임 PNG를 읽어 필터별 0.5~99.5 백분위 범위를 계산\n"
            "2패스: 해당 범위로 각 프레임을 정규화한 뒤 합성\n\n"
            "Step 9 GIF의 프레임 간 색상 이질감을 크게 줄여줍니다.\n"
            "활성화 시 Step 9의 '글로벌 채널 스트레치' 비활성화를 권장합니다."
        )
        self._global_normalize = QCheckBox()
        self._global_normalize.setChecked(True)
        self._global_normalize.setToolTip(_tip_norm)
        lbl_norm = QLabel(S("step08.global_normalize"))
        lbl_norm.setToolTip(_tip_norm)
        opt_fl.addRow(lbl_norm, self._global_normalize)

        _tip_scale = (
            "합성 결과에 곱하는 밝기 배율입니다.\n"
            "1.0 = 변경 없음 (기본값)\n"
            "0.80 = 전체를 80%로 어둡게\n"
            "분포 형태를 유지한 채 밝기만 낮춥니다."
        )
        self._series_scale = QDoubleSpinBox()
        self._series_scale.setStyleSheet(_DBLSPIN_STYLE)
        self._series_scale.setRange(0.1, 1.0)
        self._series_scale.setDecimals(2)
        self._series_scale.setSingleStep(0.05)
        self._series_scale.setValue(1.00)
        self._series_scale.setToolTip(_tip_scale)
        lbl_scale = QLabel(S("step08.series_scale"))
        lbl_scale.setToolTip(_tip_scale)
        opt_fl.addRow(lbl_scale, self._series_scale)

        _tip_window = (
            "슬라이딩 윈도우 스태킹 프레임 수입니다.\n"
            "1 = 단일 프레임 (현재 동작과 동일)\n"
            "3 = 앞뒤 1개씩 포함 → SNR √3 향상\n"
            "5 = 앞뒤 2개씩 포함 → SNR √5 향상\n"
            "목성 기준 권장 상한: 5 (약 20분 분량)\n"
            "홀수값 권장 (중심 프레임이 기준 시각)"
        )
        self._stack_window_n = QSpinBox()
        self._stack_window_n.setStyleSheet(_SPINBOX_STYLE)
        self._stack_window_n.setRange(1, 9)
        self._stack_window_n.setSingleStep(2)
        self._stack_window_n.setValue(3)
        self._stack_window_n.setToolTip(_tip_window)
        lbl_window = QLabel(S("step08.stack_window_n"))
        lbl_window.setToolTip(_tip_window)
        opt_fl.addRow(lbl_window, self._stack_window_n)

        _tip_cyc8 = (
            "필터 한 사이클(IR→R→G→B→CH4→IR)에 걸리는 시간(초)입니다.\n"
            "이 값으로 raw TIF 파일을 시계열 프레임 세트로 그룹핑합니다.\n"
            "예: 필터당 45초 × 5필터 = 225초\n\n"
            "Step 4의 필터 사이클과 독립적으로 설정할 수 있습니다.\n"
            "촬영 패턴에 맞게 입력하세요."
        )
        self._series_cycle_seconds = QSpinBox()
        self._series_cycle_seconds.setStyleSheet(_SPINBOX_STYLE)
        self._series_cycle_seconds.setRange(10, 600)
        self._series_cycle_seconds.setSingleStep(15)
        self._series_cycle_seconds.setValue(225)
        self._series_cycle_seconds.setToolTip(_tip_cyc8)
        lbl_cyc8 = QLabel(S("step08.cycle_seconds"))
        lbl_cyc8.setToolTip(_tip_cyc8)
        opt_fl.addRow(lbl_cyc8, self._series_cycle_seconds)

        _tip_minq = (
            "프레임 품질 필터입니다. (0.0 = 필터 없음)\n"
            "각 프레임의 선명도(Laplacian 분산)를 필터별 최대값으로 정규화한 뒤,\n"
            "이 값 미만인 프레임은 score² 가중치로 강하게 하향 적용됩니다.\n"
            "(완전 제외가 아닌 소프트 다운 가중 — 최소 기여 0.05 보장)\n"
            "권장: 0.05~0.3 (0 = 가중치 균등, 윈도우=1일 때 무관)\n"
            "너무 높으면 유효 프레임 수가 부족해질 수 있습니다."
        )
        self._stack_min_quality = QDoubleSpinBox()
        self._stack_min_quality.setStyleSheet(_DBLSPIN_STYLE)
        self._stack_min_quality.setRange(0.0, 0.9)
        self._stack_min_quality.setDecimals(2)
        self._stack_min_quality.setSingleStep(0.05)
        self._stack_min_quality.setValue(0.05)
        self._stack_min_quality.setToolTip(_tip_minq)
        lbl_minq = QLabel(S("step08.stack_min_quality"))
        lbl_minq.setToolTip(_tip_minq)
        opt_fl.addRow(lbl_minq, self._stack_min_quality)

        _tip_mono = (
            "각 필터의 흑백(모노) 프레임을 컬러 합성과 함께 저장합니다.\n"
            "Step 9에서 필터별 흑백 GIF/APNG도 자동으로 생성됩니다.\n"
            "예: IR_mono_animation.gif, R_mono_animation.gif ...\n\n"
            "필터별 독립적인 시계열 변화를 확인하거나\n"
            "각 채널의 품질을 개별적으로 모니터링할 때 유용합니다."
        )
        self._save_mono_frames = QCheckBox()
        self._save_mono_frames.setChecked(False)
        self._save_mono_frames.setToolTip(_tip_mono)
        lbl_mono = QLabel(S("step08.save_mono_frames"))
        lbl_mono.setToolTip(_tip_mono)
        opt_fl.addRow(lbl_mono, self._save_mono_frames)

        root.addWidget(opt_widget)

        # ── Composite specs (series-specific, independent from Step 7) ────────
        spec_header = QWidget()
        spec_header.setStyleSheet("background: transparent;")
        hdr_layout = QHBoxLayout(spec_header)
        hdr_layout.setContentsMargins(0, 8, 0, 4)
        hdr_layout.setSpacing(0)
        hdr_lbl = QLabel(S("step08.series_spec_header"))
        hdr_lbl.setStyleSheet("color: #aaa; font-size: 11px; font-weight: bold;")
        hdr_lbl.setToolTip(
            "Step 7과 별도로 시계열 합성 채널을 지정합니다.\n"
            "Step 7 합성 설정에 영향을 주지 않습니다."
        )
        hdr_layout.addWidget(hdr_lbl)
        hdr_layout.addStretch()
        self._btn_add_spec = QPushButton(S("btn.add_spec"))
        self._btn_add_spec.setStyleSheet(_BTN_ADD)
        self._btn_add_spec.clicked.connect(self._add_spec_row)
        hdr_layout.addWidget(self._btn_add_spec)
        root.addWidget(spec_header)

        self._spec_container = QWidget()
        self._spec_container.setStyleSheet("background: #252525; border-radius: 4px;")
        self._spec_vbox = QVBoxLayout(self._spec_container)
        self._spec_vbox.setContentsMargins(6, 6, 6, 6)
        self._spec_vbox.setSpacing(2)

        col_hdr = QWidget()
        col_hdr.setStyleSheet("background: transparent;")
        col_hdr_layout = QHBoxLayout(col_hdr)
        col_hdr_layout.setContentsMargins(0, 0, 0, 2)
        col_hdr_layout.setSpacing(4)
        for txt, w in [("", 20), (S("spec.col.name"), 90), (S("spec.col.r_channel"), 84), (S("spec.col.g_channel"), 84), (S("spec.col.b_channel"), 84), (S("spec.col.l_channel"), 84), ("", 24)]:
            lbl = QLabel(txt)
            lbl.setFixedWidth(w)
            lbl.setStyleSheet("color: #666; font-size: 10px;")
            col_hdr_layout.addWidget(lbl)
        self._spec_vbox.addWidget(col_hdr)

        scroll = QScrollArea()
        scroll.setWidget(self._spec_container)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(scroll.Shape.NoFrame)
        scroll.setMaximumHeight(160)
        scroll.setStyleSheet("QScrollArea { background: #252525; border-radius: 4px; }")
        scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        root.addWidget(scroll)

        self._spec_radio_group = QButtonGroup(self)
        for spec in _DEFAULT_SERIES_SPECS:
            self._add_spec_row(spec)

        # ── Wavelet sharpening (series) ───────────────────────────────────────
        wav_section = QWidget()
        wav_section.setStyleSheet("background: transparent;")
        wav_vl = QVBoxLayout(wav_section)
        wav_vl.setSpacing(6)
        wav_vl.setContentsMargins(0, 4, 0, 0)

        amounts_label = QLabel(S("step08.series_amounts"))
        amounts_label.setStyleSheet("color: #aaa; font-size: 11px;")
        amounts_label.setToolTip(
            "시계열 합성 각 프레임에 적용할 웨이블릿 선명화 강도입니다.\n"
            "Step 6 (마스터)와 독립적으로 조정할 수 있습니다.\n"
            "L1이 세밀한 디테일, L6으로 갈수록 넓은 구조를 담당합니다."
        )
        wav_vl.addWidget(amounts_label)

        self._series_wavelet_spins: list[QDoubleSpinBox] = []
        defaults8m = list(_SERIES_WAVELET_DEFAULTS)
        for i in range(0, len(defaults8m), 2):
            pair = QHBoxLayout()
            pair.setSpacing(12)
            for j in range(2):
                if i + j < len(defaults8m):
                    row_layout, spin = _make_wavelet_row(i + j + 1, defaults8m[i + j])
                    pair.addLayout(row_layout)
                    self._series_wavelet_spins.append(spin)
            wav_vl.addLayout(pair)

        root.addWidget(wav_section)

    def get_config_updates(self) -> dict[str, Any]:
        series_specs = [r.to_dict() for r in self._spec_rows if r.to_dict()["name"]]
        return {
            "global_filter_normalize":      self._global_normalize.isChecked(),
            "series_scale":                 self._series_scale.value(),
            "stack_window_n":               self._stack_window_n.value(),
            "series_cycle_seconds":         self._series_cycle_seconds.value(),
            "stack_min_quality":            self._stack_min_quality.value(),
            "save_mono_frames":             self._save_mono_frames.isChecked(),
            "series_amounts":               [s.value() for s in self._series_wavelet_spins],
            "series_composite_specs":       series_specs,
        }

    def load_session(self, data: dict[str, Any]) -> None:
        out = data.get("output_dir", "")
        input_dir = data.get("input_dir", "")
        if input_dir:
            self._input_lbl.setText(input_dir)
        elif out:
            self._input_lbl.setText(str(Path(out) / "step02_lucky_stack"))
        if out:
            p = Path(out)
            self._output_lbl.setText(str(p / "step08_series"))
            self._output_dir = p
        self._global_normalize.setChecked(bool(data.get("global_filter_normalize", True)))
        self._series_scale.setValue(float(data.get("series_scale", 1.00)))
        self._series_cycle_seconds.setValue(int(data.get("series_cycle_seconds",
                                                          data.get("cycle_seconds", 225))))
        self._stack_window_n.setValue(int(data.get("stack_window_n", 3)))
        self._stack_min_quality.setValue(float(data.get("stack_min_quality", 0.05)))
        self._save_mono_frames.setChecked(bool(data.get("save_mono_frames", False)))
        amounts = data.get("series_amounts", _SERIES_WAVELET_DEFAULTS)
        for spin, val in zip(self._series_wavelet_spins, amounts):
            spin.setValue(float(val))

        # Update available filter options from session
        raw = data.get("filters", "")
        if raw:
            filters = [f.strip() for f in raw.split(",") if f.strip()]
            mono_filters = [f for f in filters if f not in ("COLOR", "RGB")]
            if mono_filters:
                self._refresh_filter_options(mono_filters)

        # Load series-specific composite specs (fall back to step07 specs if not set)
        saved_specs = data.get("series_composite_specs") or data.get("composite_specs")
        if saved_specs:
            for row in list(self._spec_rows):
                if hasattr(self, "_spec_radio_group"):
                    self._spec_radio_group.removeButton(row._radio)
                row.setParent(None)
                row.deleteLater()
            self._spec_rows.clear()
            for spec in saved_specs:
                self._add_spec_row(spec)

    def output_paths(self) -> list[Path]:
        if self._output_dir is None:
            return []
        step_dir = self._output_dir / "step08_series"
        if step_dir.exists():
            return sorted(step_dir.glob("*.png"))
        return []

    def set_output_dir(self, path: Path | str) -> None:
        self._output_dir = Path(path) if path else None

    def _add_spec_row(self, spec: dict[str, str] | None = None) -> None:
        row = _SpecRow(self._available_filters, spec, parent=self._spec_container)
        row.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        row._btn_del.clicked.connect(lambda _, r=row: self._remove_spec_row(r))
        self._spec_rows.append(row)
        self._spec_vbox.addWidget(row)
        if hasattr(self, "_spec_radio_group"):
            self._spec_radio_group.addButton(row._radio)
            if len(self._spec_rows) == 1:
                row._radio.setChecked(True)

    def _remove_spec_row(self, row: _SpecRow) -> None:
        if hasattr(self, "_spec_radio_group"):
            self._spec_radio_group.removeButton(row._radio)
        if row in self._spec_rows:
            self._spec_rows.remove(row)
        row.setParent(None)
        row.deleteLater()

    def _refresh_filter_options(self, filters: list[str]) -> None:
        self._available_filters = filters
        for row in self._spec_rows:
            row.update_filters(filters)


# ── Color sub-widget ───────────────────────────────────────────────────────────

class _Step08ColorWidget(QWidget):
    """Color camera Step 8: sliding-window stacking of a single color stream.

    Controls shown differ from mono:
      - No 'Global filter normalize' (only one color channel — no cross-filter
        normalisation is needed).
      - No 'Save mono filter GIFs' (no separate filter channels to save).
      - 'Capture interval' (촬영 간격) replaces 'Filter cycle' — the concept
        changes from a 5-filter rotation to a single-image cadence.
      - Stack window N upper limit raised to 99 (color cameras can capture
        much faster, so larger windows are practical).
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._output_dir: Path | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        # ── Folder display ────────────────────────────────────────────────────
        folder_widget = QWidget()
        folder_widget.setStyleSheet("background: transparent;")
        fl = QFormLayout(folder_widget)
        fl.setSpacing(8)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self._input_lbl = QLineEdit()
        self._input_lbl.setReadOnly(True)
        self._input_lbl.setStyleSheet(_READONLY_STYLE)
        lbl_in = QLabel(S("step08.input_dir"))
        lbl_in.setToolTip("입력 TIF 폴더 (자동 설정)")
        fl.addRow(lbl_in, self._input_lbl)

        self._output_lbl = QLineEdit()
        self._output_lbl.setReadOnly(True)
        self._output_lbl.setStyleSheet(_READONLY_STYLE)
        lbl_out = QLabel(S("step08.output_dir"))
        lbl_out.setToolTip("시계열 컬러 PNG가 저장될 폴더 (자동 설정)")
        fl.addRow(lbl_out, self._output_lbl)

        root.addWidget(folder_widget)

        # ── Options ───────────────────────────────────────────────────────────
        opt_widget = QWidget()
        opt_widget.setStyleSheet("background: transparent;")
        opt_fl = QFormLayout(opt_widget)
        opt_fl.setSpacing(10)
        opt_fl.setContentsMargins(0, 0, 0, 0)
        opt_fl.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        _tip_scale = (
            "합성 결과에 곱하는 밝기 배율입니다.\n"
            "1.0 = 변경 없음 (기본값)\n"
            "0.80 = 전체를 80%로 어둡게\n"
            "분포 형태를 유지한 채 밝기만 낮춥니다."
        )
        self._series_scale = QDoubleSpinBox()
        self._series_scale.setStyleSheet(_DBLSPIN_STYLE)
        self._series_scale.setRange(0.1, 1.0)
        self._series_scale.setDecimals(2)
        self._series_scale.setSingleStep(0.05)
        self._series_scale.setValue(1.00)
        self._series_scale.setToolTip(_tip_scale)
        lbl_scale = QLabel(S("step08.series_scale"))
        lbl_scale.setToolTip(_tip_scale)
        opt_fl.addRow(lbl_scale, self._series_scale)

        _tip_cyc = (
            "연속 색상 촬영 간격(초)입니다.\n"
            "이 값으로 TIF 파일을 시계열 프레임으로 그룹핑합니다.\n"
            "예: 30초마다 1장 촬영 → 30 입력\n\n"
            "모노 카메라의 필터 사이클과는 다른 개념입니다.\n"
            "실제 촬영 간격보다 약간 크게 설정하면 안전합니다."
        )
        self._series_cycle_seconds = QSpinBox()
        self._series_cycle_seconds.setStyleSheet(_SPINBOX_STYLE)
        self._series_cycle_seconds.setRange(5, 300)
        self._series_cycle_seconds.setSingleStep(5)
        self._series_cycle_seconds.setValue(30)
        self._series_cycle_seconds.setToolTip(_tip_cyc)
        lbl_cyc = QLabel(S("step08.cycle_seconds_color"))
        lbl_cyc.setToolTip(_tip_cyc)
        opt_fl.addRow(lbl_cyc, self._series_cycle_seconds)

        _tip_window = (
            "슬라이딩 윈도우 스태킹 프레임 수입니다.\n"
            "1 = 단일 프레임\n"
            "5 = 앞뒤 2개씩 포함 → SNR √5 향상\n"
            "컬러 카메라는 촬영 속도가 빠르므로 더 큰 값을 사용할 수 있습니다.\n"
            "행성 자전량이 허용하는 범위 내에서 최대한 높게 설정하세요.\n"
            "홀수값 권장 (중심 프레임이 기준 시각)"
        )
        self._stack_window_n = QSpinBox()
        self._stack_window_n.setStyleSheet(_SPINBOX_STYLE)
        self._stack_window_n.setRange(1, 99)
        self._stack_window_n.setSingleStep(2)
        self._stack_window_n.setValue(5)
        self._stack_window_n.setToolTip(_tip_window)
        lbl_window = QLabel(S("step08.stack_window_n"))
        lbl_window.setToolTip(_tip_window)
        opt_fl.addRow(lbl_window, self._stack_window_n)

        _tip_minq = (
            "프레임 품질 필터입니다. (0.0 = 필터 없음)\n"
            "선명도(Laplacian 분산) 기준으로 품질이 낮은 프레임의 기여를 낮춥니다.\n"
            "권장: 0.05~0.3"
        )
        self._stack_min_quality = QDoubleSpinBox()
        self._stack_min_quality.setStyleSheet(_DBLSPIN_STYLE)
        self._stack_min_quality.setRange(0.0, 0.9)
        self._stack_min_quality.setDecimals(2)
        self._stack_min_quality.setSingleStep(0.05)
        self._stack_min_quality.setValue(0.05)
        self._stack_min_quality.setToolTip(_tip_minq)
        lbl_minq = QLabel(S("step08.stack_min_quality"))
        lbl_minq.setToolTip(_tip_minq)
        opt_fl.addRow(lbl_minq, self._stack_min_quality)

        root.addWidget(opt_widget)

        # ── Wavelet sharpening (series) ───────────────────────────────────────
        wav_section = QWidget()
        wav_section.setStyleSheet("background: transparent;")
        wav_vl = QVBoxLayout(wav_section)
        wav_vl.setSpacing(6)
        wav_vl.setContentsMargins(0, 4, 0, 0)

        amounts_label = QLabel(S("step08.series_amounts"))
        amounts_label.setStyleSheet("color: #aaa; font-size: 11px;")
        amounts_label.setToolTip(
            "시계열 각 프레임의 스태킹 후 적용할 웨이블릿 선명화 강도입니다.\n"
            "L1이 세밀한 디테일, L6으로 갈수록 넓은 구조를 담당합니다."
        )
        wav_vl.addWidget(amounts_label)

        self._series_wavelet_spins: list[QDoubleSpinBox] = []
        defaults8c = list(_SERIES_WAVELET_DEFAULTS)
        for i in range(0, len(defaults8c), 2):
            pair = QHBoxLayout()
            pair.setSpacing(12)
            for j in range(2):
                if i + j < len(defaults8c):
                    row_layout, spin = _make_wavelet_row(i + j + 1, defaults8c[i + j])
                    pair.addLayout(row_layout)
                    self._series_wavelet_spins.append(spin)
            wav_vl.addLayout(pair)

        root.addWidget(wav_section)

    def get_config_updates(self) -> dict[str, Any]:
        return {
            "series_scale":         self._series_scale.value(),
            "series_cycle_seconds": self._series_cycle_seconds.value(),
            "stack_window_n":       self._stack_window_n.value(),
            "stack_min_quality":    self._stack_min_quality.value(),
            "series_amounts":       [s.value() for s in self._series_wavelet_spins],
        }

    def load_session(self, data: dict[str, Any]) -> None:
        inp = data.get("input_dir", "")
        if inp:
            self._input_lbl.setText(inp)
        out = data.get("output_dir", "")
        if out:
            p = Path(out)
            self._output_lbl.setText(str(p / "step08_series"))
            self._output_dir = p
        self._series_scale.setValue(float(data.get("series_scale", 1.00)))
        self._series_cycle_seconds.setValue(int(data.get("series_cycle_seconds", 30)))
        self._stack_window_n.setValue(int(data.get("stack_window_n", 5)))
        self._stack_min_quality.setValue(float(data.get("stack_min_quality", 0.05)))
        amounts = data.get("series_amounts", _SERIES_WAVELET_DEFAULTS)
        for spin, val in zip(self._series_wavelet_spins, amounts):
            spin.setValue(float(val))

    def output_paths(self) -> list[Path]:
        if self._output_dir is None:
            return []
        step_dir = self._output_dir / "step08_series"
        if step_dir.exists():
            return sorted(step_dir.glob("*.png"))
        return []

    def set_output_dir(self, path: Path | str) -> None:
        self._output_dir = Path(path) if path else None


# ── Panel wrapper ──────────────────────────────────────────────────────────────

class Step08Panel(BasePanel):
    STEP_ID   = "08"
    TITLE_KEY = "step08.title"
    DESC_KEY  = "step08.desc"
    OPTIONAL  = True

    def __init__(self, parent: QWidget | None = None) -> None:
        self._output_dir: Path | None = None
        self._is_color: bool = False
        super().__init__(parent)

    def build_form(self) -> None:
        self._sub_stack    = QStackedWidget()
        self._mono_widget  = _Step08MonoWidget()
        self._color_widget = _Step08ColorWidget()
        self._sub_stack.addWidget(self._mono_widget)   # index 0
        self._sub_stack.addWidget(self._color_widget)  # index 1

        idx = self._form_layout.count() - 1
        self._form_layout.insertWidget(idx, self._sub_stack)

    def get_config_updates(self) -> dict[str, Any]:
        if self._is_color:
            return self._color_widget.get_config_updates()
        return self._mono_widget.get_config_updates()

    def load_session(self, data: dict[str, Any]) -> None:
        self._is_color = data.get("camera_mode", "mono") == "color"
        self._sub_stack.setCurrentIndex(1 if self._is_color else 0)
        if self._is_color:
            self._color_widget.load_session(data)
        else:
            self._mono_widget.load_session(data)

    def output_paths(self) -> list[Path]:
        if self._is_color:
            return self._color_widget.output_paths()
        return self._mono_widget.output_paths()

    def set_output_dir(self, path: Path | str) -> None:
        self._output_dir = Path(path) if path else None
        self._mono_widget.set_output_dir(self._output_dir)
        self._color_widget.set_output_dir(self._output_dir)
