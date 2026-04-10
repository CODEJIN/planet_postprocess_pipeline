# -*- mode: python ; coding: utf-8 -*-
"""
AstroPipeline PyInstaller spec — 플랫폼 공통 (Linux / Windows)

사용법:
  Linux   : pyinstaller --clean astro_pipeline.spec
  Windows : pyinstaller --clean astro_pipeline.spec

크기 최적화:
  - scipy    : wavelet.py가 numpy로 대체 → 완전 제거 (OpenBLAS ~200MB 제거)
  - skimage / astropy / astroquery / imageio : 코드 미사용 → 제외
  - torch / transformers / triton 등 ML 패키지 : 강제 제외 (환경 오염 차단)
  - PySide6  : WebEngine·3D·Multimedia 등 제외 (정규식 Qt6? 로 바이너리+바인딩 동시 처리)
  - strip=True (Linux) : ELF 디버그 심볼 제거
"""
import os
import re
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_data_files

block_cipher = None
IS_WINDOWS = sys.platform == "win32"

# ── Linux: libexpat 명시적 번들 ──────────────────────────────────────────────
_extra_binaries = []
if not IS_WINDOWS:
    _libexpat = Path(sys.prefix) / "lib" / "libexpat.so.1"
    if _libexpat.exists():
        _extra_binaries.append((str(_libexpat), "."))

# ── PySide6: 불필요한 Qt 모듈 필터링 ─────────────────────────────────────────
#
# 버그 수정: 이전 버전의 패턴 Qt6(...) 은 Qt 공유 라이브러리(libQt6WebEngine*.so)는
# 잡지만, PySide6 Python 바인딩 파일(QtWebEngineCore.abi3.so, 이름에 '6' 없음)은
# 놓쳤습니다. Qt6? 로 '6'을 선택적으로 만들어 두 케이스를 모두 처리합니다.
#
# 파일명 예시:
#   libQt6WebEngineCore.so.6   → Qt6WebEngine 매칭 ✓ (기존도 됨)
#   QtWebEngineCore.abi3.so    → Qt?WebEngine  매칭 ✓ (수정 후)
#   QtWebEngineProcess         → Qt?WebEngine  매칭 ✓ (수정 후)
#   qtwebengine_resources.pak  → qtwebengine   매칭 ✓ (경로 검사)
_QT_EXCLUDE_RE = re.compile(
    r"Qt6?(?:"                              # Qt6 or Qt (no '6') — covers both naming schemes
    r"WebEngine|WebChannel|WebSockets|WebView"
    r"|3D"
    r"|Multimedia(?!Widgets$)|MultimediaWidgets"
    r"|Quick|Qml|LabsQml"
    r"|Charts|DataVisualization"
    r"|Bluetooth|Positioning|Location|Sensors"
    r"|SerialBus|SerialPort"
    r"|RemoteObjects|Scxml|StateMachine"
    r"|SpatialAudio|VirtualKeyboard"
    r"|Pdf(?!Widgets$)|PdfWidgets"
    r"|OpenGL(?!Widgets$)|OpenGLWidgets"
    r"|PrintSupport|Concurrent|Help|Xml"
    r"|Test"
    r")",
    re.IGNORECASE,
)

# 경로 기반 추가 제외 (파일명에 Qt 접두사 없는 WebEngine 리소스 등)
_PATH_EXCLUDE_FRAGMENTS = [
    "qtwebengine",
    "webengine_locales",
    "qtmultimedia",
    "qtquick",
    "/qml/",
    "/Qt3D",
]

pyside6_datas, pyside6_binaries, pyside6_hiddenimports = collect_all("PySide6")


def _keep(path: str) -> bool:
    """True → 번들에 포함, False → 제외."""
    norm = path.replace("\\", "/").lower()
    # 경로 전체에서 제외 패턴 검사
    if any(frag in norm for frag in _PATH_EXCLUDE_FRAGMENTS):
        return False
    # 파일 이름에서 Qt 모듈 패턴 검사
    return not _QT_EXCLUDE_RE.search(os.path.basename(path))


pyside6_binaries = [(s, d) for s, d in pyside6_binaries if _keep(s)]
pyside6_datas    = [(s, d) for s, d in pyside6_datas    if _keep(s)]
pyside6_hiddenimports = [
    m for m in pyside6_hiddenimports
    if not _QT_EXCLUDE_RE.search(m)
]

# ── Analysis ──────────────────────────────────────────────────────────────────
a = Analysis(
    ["gui/main.py"],
    pathex=["."],
    binaries=pyside6_binaries + _extra_binaries,
    datas=[
        ("gui/icons",     "gui/icons"),
        ("gui/i18n",      "gui/i18n"),
        ("pipeline/data", "pipeline/data"),
        *pyside6_datas,
    ],
    hiddenimports=[
        "cv2",
        "tifffile",
        "PIL",
        "PIL.Image",
        "PIL.ImageFont",
        "PIL.ImageDraw",
        "PySide6.QtGui",          # ensures qico imageformat plugin is collected
        *pyside6_hiddenimports,
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=["build_runtime_hook.py"],
    excludes=[
        # ── 표준 불필요 패키지 ──────────────────────────────────────────────
        "tkinter",
        "matplotlib",
        "notebook",
        "jupyter",
        "IPython",
        "sphinx",
        # ── 미사용 과학/이미지 라이브러리 ──────────────────────────────────
        "scipy",
        "skimage",
        "sklearn",
        "astropy",
        "astroquery",
        "imageio",
        "imageio_ffmpeg",
        # ── ML / AI 패키지 (환경에 설치돼 있어도 포함 금지) ────────────────
        # 이 항목들이 없으면 PyInstaller가 환경의 torch/transformers 등을 발견해
        # onefile 크기가 수 GB 이상으로 폭증합니다.
        "torch",
        "torchvision",
        "torchaudio",
        "transformers",
        "triton",
        "pyarrow",
        "pandas",
        "numba",
        "librosa",
        "hydra",
        "omegaconf",
        "accelerate",
        "datasets",
        "tokenizers",
        "huggingface_hub",
        "safetensors",
        "xformers",
        "diffusers",
        "peft",
        "bitsandbytes",
        "lightning",
        "pytorch_lightning",
        "tensorflow",
        "keras",
        "jax",
        "flax",
        "optax",
        # ── 기타 무거운 패키지 ──────────────────────────────────────────────
        "sqlalchemy",
        "django",
        "flask",
        "fastapi",
        "uvicorn",
        "aiohttp",
        # ── PySide6 대형 모듈 (Python import 단계 차단) ─────────────────────
        "PySide6.QtWebEngineCore",
        "PySide6.QtWebEngineQuick",
        "PySide6.QtWebEngineWidgets",
        "PySide6.Qt3DCore",
        "PySide6.Qt3DRender",
        "PySide6.Qt3DExtras",
        "PySide6.Qt3DAnimation",
        "PySide6.Qt3DInput",
        "PySide6.QtMultimedia",
        "PySide6.QtMultimediaWidgets",
        "PySide6.QtQuick",
        "PySide6.QtQml",
        "PySide6.QtCharts",
        "PySide6.QtDataVisualization",
        "PySide6.QtBluetooth",
        "PySide6.QtPositioning",
        "PySide6.QtSensors",
        "PySide6.QtSerialBus",
        "PySide6.QtSerialPort",
        "PySide6.QtRemoteObjects",
        "PySide6.QtScxml",
        "PySide6.QtStateMachine",
        "PySide6.QtTest",
        "PySide6.QtPdf",
        "PySide6.QtPdfWidgets",
        "PySide6.QtOpenGL",
        "PySide6.QtOpenGLWidgets",
        "PySide6.QtPrintSupport",
        "PySide6.QtConcurrent",
        "PySide6.QtHelp",
        "PySide6.QtXml",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="AstroPipeline",
    debug=False,
    bootloader_ignore_signals=False,
    strip=not IS_WINDOWS,   # Linux: ELF 디버그 심볼 제거 (크기 ~20% 절감)
    upx=False,              # Qt 바이너리 UPX 충돌 위험 → 비활성화
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="gui/icons/app_icon.ico" if IS_WINDOWS else None,
)
