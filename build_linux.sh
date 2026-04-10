#!/usr/bin/env bash
# ============================================================
#  AstroPipeline — Linux 빌드 스크립트
#  결과물: dist/AstroPipeline  (단일 실행 파일)
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

APP_NAME="AstroPipeline"
DIST_DIR="dist"
BUILD_DIR="build"

echo "======================================================"
echo "  AstroPipeline Linux Build"
echo "======================================================"

# ── 1. 의존성 확인 ────────────────────────────────────────────
echo ""
echo "[1/4] 빌드 의존성 확인..."

if ! python3 -c "import PyInstaller" 2>/dev/null; then
    echo "  PyInstaller가 없습니다. 설치 중..."
    pip install pyinstaller pyinstaller-hooks-contrib
else
    echo "  PyInstaller OK"
fi

if ! python3 -c "import _pyinstaller_hooks_contrib" 2>/dev/null; then
    echo "  pyinstaller-hooks-contrib 설치 중..."
    pip install pyinstaller-hooks-contrib
fi

# ── 2. 이전 빌드 정리 ─────────────────────────────────────────
echo ""
echo "[2/4] 이전 빌드 디렉터리 정리..."
rm -rf "$BUILD_DIR" "$DIST_DIR"
echo "  완료"

# ── 3. 빌드 ──────────────────────────────────────────────────
echo ""
echo "[3/4] PyInstaller 빌드 시작..."
echo "  (첫 빌드 시 PySide6 수집에 수 분 소요될 수 있습니다)"
echo ""

python3 -m PyInstaller --clean astro_pipeline.spec

# ── 4. 결과 확인 ──────────────────────────────────────────────
echo ""
echo "[4/4] 빌드 결과 확인..."

EXE_PATH="$DIST_DIR/$APP_NAME"
if [ -f "$EXE_PATH" ]; then
    SIZE=$(du -sh "$EXE_PATH" | cut -f1)
    SIZE_BYTES=$(stat -c%s "$EXE_PATH")
    echo ""
    echo "======================================================"
    echo "  빌드 성공!"
    echo "  경로: $EXE_PATH"
    echo "  크기: $SIZE (${SIZE_BYTES} bytes)"
    echo "======================================================"
    echo ""
    echo "실행 방법:"
    echo "  ./$EXE_PATH"
    echo ""
    echo "참고: 첫 실행 시 /tmp 에 파일을 압축 해제하므로 5~10초 소요됩니다."
else
    echo ""
    echo "ERROR: 빌드 실패 - $EXE_PATH 가 생성되지 않았습니다."
    echo "위의 오류 메시지를 확인하세요."
    exit 1
fi
