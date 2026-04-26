"""Welcome / home panel — shown on startup instead of Settings."""
from __future__ import annotations

import multiprocessing
import subprocess
import sys
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPainter
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


from gui.i18n import S

# Logo path — works both in development and PyInstaller frozen builds
_HERE = Path(__file__).parent
_LOGO_PATH = str(
    (Path(sys._MEIPASS) / "gui" / "icons" / "logo_planetflow.svg")
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")
    else _HERE.parent / "icons" / "logo_planetflow.svg"
)

def _collect_system_info() -> dict[str, str]:
    """Collect CPU / RAM / GPU info once at panel creation."""
    info: dict[str, str] = {}

    # CPU
    info["cpu"] = f"{multiprocessing.cpu_count()} cores"

    # RAM — read /proc/meminfo on Linux; fall back silently
    try:
        meminfo: dict[str, int] = {}
        with open("/proc/meminfo") as fh:
            for line in fh:
                k, v = line.split(":")
                meminfo[k.strip()] = int(v.strip().split()[0])  # kB
        total = meminfo.get("MemTotal", 0) / 1024 / 1024
        avail = meminfo.get("MemAvailable", 0) / 1024 / 1024
        info["ram"] = f"{avail:.1f} GB / {total:.1f} GB"
    except Exception:
        info["ram"] = ""

    # GPU — try nvidia-smi (non-blocking, 3 s timeout)
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        if r.returncode == 0 and r.stdout.strip():
            gpus = []
            for line in r.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    name, mem_mb = parts[0], int(parts[1])
                    gpus.append(f"{name} ({mem_mb // 1024} GB)")
            info["gpu"] = "  |  ".join(gpus) if gpus else ""
        else:
            info["gpu"] = ""
    except Exception:
        info["gpu"] = ""

    return info

class _SvgWidget(QWidget):
    """Transparent SVG renderer — no background fill, so the card surface shows through."""

    def __init__(self, path: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._renderer = QSvgRenderer(path, self)
        self.setAutoFillBackground(False)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._renderer.render(p)
        p.end()


_BTN_PRIMARY = (
    "QPushButton { background: #2d5a1b; color: #b8f5a0; border: 1px solid #4a9030;"
    " border-radius: 5px; padding: 8px 20px; font-size: 13px; font-weight: bold; }"
    "QPushButton:hover { background: #3a7a25; }"
    "QPushButton:pressed { background: #1e3d12; }"
)
_BTN_SECONDARY = (
    "QPushButton { background: #3c3c3c; color: #bbb; border: 1px solid #555;"
    " border-radius: 5px; padding: 8px 20px; font-size: 13px; }"
    "QPushButton:hover { background: #4a4a4a; color: #e0e0e0; }"
)


class WelcomePanel(QWidget):
    """Landing screen displayed on app startup."""

    go_settings = Signal()
    go_resume   = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet("background: #1e1e1e;")
        self._sysinfo = _collect_system_info()
        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.setContentsMargins(0, 0, 0, 0)

        # Centre card
        card = QWidget()
        card.setFixedWidth(560)
        card.setStyleSheet(
            "background: #252526; border: 1px solid #3c3c3c; border-radius: 8px;"
        )
        card_layout = QVBoxLayout(card)
        card_layout.setSpacing(16)
        card_layout.setContentsMargins(40, 28, 40, 28)

        # PlanetFlow logo (SVG, 4:1 aspect → 480×120)
        logo = _SvgWidget(_LOGO_PATH)
        logo.setFixedSize(480, 120)
        card_layout.addWidget(logo, alignment=Qt.AlignmentFlag.AlignCenter)

        # Subtitle
        self._subtitle_lbl = QLabel(S("welcome.subtitle"))
        self._subtitle_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._subtitle_lbl.setStyleSheet(
            "color: #666; font-size: 11px; background: transparent; border: none;"
        )
        card_layout.addWidget(self._subtitle_lbl)

        card_layout.addWidget(_sep())

        # Info rows
        info_widget = QWidget()
        info_widget.setStyleSheet("background: transparent;")
        info_layout = QGridLayout(info_widget)
        info_layout.setContentsMargins(0, 4, 0, 4)
        info_layout.setHorizontalSpacing(16)
        info_layout.setVerticalSpacing(10)
        info_layout.setColumnStretch(1, 1)

        def _key_lbl(text: str) -> QLabel:
            l = QLabel(text)
            l.setStyleSheet(
                "color: #666; font-size: 11px; background: transparent; border: none;"
            )
            l.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            return l

        def _val_lbl() -> QLabel:
            l = QLabel("—")
            l.setStyleSheet(
                "color: #ccc; font-size: 11px; background: transparent; border: none;"
            )
            l.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            l.setWordWrap(True)
            return l

        none_str = S("welcome.info.none")

        self._key_profile = _key_lbl(S("welcome.info.profile"))
        self._val_profile  = _val_lbl()
        info_layout.addWidget(self._key_profile, 0, 0)
        info_layout.addWidget(self._val_profile,  0, 1)

        self._key_cpu = _key_lbl(S("welcome.info.cpu"))
        val_cpu = _val_lbl()
        val_cpu.setText(self._sysinfo.get("cpu") or none_str)
        info_layout.addWidget(self._key_cpu, 1, 0)
        info_layout.addWidget(val_cpu,       1, 1)

        self._key_ram = _key_lbl(S("welcome.info.ram"))
        val_ram = _val_lbl()
        val_ram.setText(self._sysinfo.get("ram") or none_str)
        info_layout.addWidget(self._key_ram, 2, 0)
        info_layout.addWidget(val_ram,       2, 1)

        self._key_gpu = _key_lbl(S("welcome.info.gpu"))
        val_gpu = _val_lbl()
        val_gpu.setText(self._sysinfo.get("gpu") or none_str)
        info_layout.addWidget(self._key_gpu, 3, 0)
        info_layout.addWidget(val_gpu,       3, 1)

        card_layout.addWidget(info_widget)

        card_layout.addWidget(_sep())

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)
        btn_row.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._btn_settings = QPushButton("⚙  " + S("app.settings"))
        self._btn_settings.setFixedSize(170, 40)
        self._btn_settings.setStyleSheet(_BTN_SECONDARY)
        self._btn_settings.clicked.connect(self.go_settings)
        btn_row.addWidget(self._btn_settings)

        self._btn_resume = QPushButton("▶  " + S("welcome.btn_resume"))
        self._btn_resume.setFixedSize(170, 40)
        self._btn_resume.setStyleSheet(_BTN_PRIMARY)
        self._btn_resume.clicked.connect(self.go_resume)
        btn_row.addWidget(self._btn_resume)

        card_layout.addLayout(btn_row)

        outer.addWidget(card, alignment=Qt.AlignmentFlag.AlignCenter)

    # ── Public API ────────────────────────────────────────────────────────────

    def load_session(self, data: dict[str, Any]) -> None:
        """Update profile row from session data (system info is static)."""
        profile = data.get("active_profile") or S("welcome.info.none")
        self._val_profile.setText(profile)

    def retranslate(self) -> None:
        """Re-apply i18n strings after language change."""
        self._subtitle_lbl.setText(S("welcome.subtitle"))
        self._key_profile.setText(S("welcome.info.profile"))
        self._key_cpu.setText(S("welcome.info.cpu"))
        self._key_ram.setText(S("welcome.info.ram"))
        self._key_gpu.setText(S("welcome.info.gpu"))
        self._btn_settings.setText("⚙  " + S("app.settings"))
        self._btn_resume.setText("▶  " + S("welcome.btn_resume"))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sep() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet("color: #3c3c3c; background: #3c3c3c; border: none;")
    f.setFixedHeight(1)
    return f


