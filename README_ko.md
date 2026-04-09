# AstroPipeline

**행성 촬영 후처리 파이프라인 (GUI 포함)**

AstroPipeline은 SER 동영상 파일로 촬영된 행성 관측 데이터(목성, 토성, 화성)를 처리하는 데스크톱 애플리케이션입니다. 원시 프레임 정렬부터 웨이블릿 샤프닝, 자전 보정 스태킹, 다채널 합성, 애니메이션 GIF 출력까지 전체 후처리 워크플로를 단일 PySide6 GUI에서 자동화합니다.

> English version: [README.md](README.md)

---

## 주요 기능

- **10단계 파이프라인** — 각 단계별 활성화/비활성화 제어
- **프레임 품질 평가** — Laplacian 선명도 + 퓨리에 고주파 지표 결합
- **행성 자전 보정(De-rotation)** — JPL Horizons 역표(astroquery) 기반
- **웨이블릿 샤프닝** — à trous 알고리즘, WaveSharp 호환 0–200 스케일
- **다채널 합성** — RGB, IR-RGB, CH4-G-IR
- **시계열 애니메이션** — 디스크 중심 정렬 포함 GIF 출력
- **서머리 컨택 시트** — 전체 윈도우 × 합성 조합을 하나의 이미지로
- **한국어 / 영어 UI** — 런타임 전환 지원
- **독립 실행 파일** — PyInstaller로 단일 바이너리 배포 (Python 설치 불필요)

---

## 파이프라인 단계

| 단계 | 이름 | 설명 |
|------|------|------|
| 01 | PIPP 전처리 | 잘림/변형 프레임 제거, 행성 중심 정렬, 정사각형 ROI 크롭 |
| 02 | AutoStakkert! 4 | *(외부)* Step 01 출력물을 AS!4로 직접 스태킹 |
| 03 | 웨이블릿 미리보기 | 모든 TIF 스택에 웨이블릿 샤프닝 적용 후 필터별 PNG 출력 |
| 04 | 품질 평가 | 각 TIF 점수화; 전체 필터에서 동시에 품질이 높은 최적 윈도우 탐색 |
| 05 | 자전 보정 스태킹 | 윈도우별 프레임 회전·정렬 후 품질 가중 평균 스택 |
| 06 | 웨이블릿 마스터 | 자전 보정 스택에 최종 웨이블릿 샤프닝 적용 |
| 07 | RGB 합성 | 윈도우별 다채널 합성 (RGB, IR-RGB, CH4-G-IR) |
| 08 | 시계열 합성 | 전체 관측 세션에 걸쳐 슬라이딩 윈도우 스택 합성 |
| 09 | 애니메이션 GIF | 시계열 프레임 조합 → GIF 출력 |
| 10 | 서머리 그리드 | 전체 윈도우 × 합성 조합 컨택 시트 |

---

## 필요 환경

- Python 3.10 이상
- 아래 패키지 (`pip install -r requirements.txt`):

```
numpy
scipy
opencv-python
tifffile
Pillow
imageio[ffmpeg]
astropy
astroquery
scikit-image
PySide6
```

---

## 설치

```bash
git clone https://github.com/CODEJIN/AstroPipeline.git
cd AstroPipeline
pip install -r requirements.txt
```

---

## 실행 (소스에서)

### GUI (권장)

```bash
python gui/main.py
```

### CLI

`main.py` 상단의 `PipelineConfig`에서 경로와 파라미터를 설정한 뒤:

```bash
python main.py
```

---

## 독립 실행 파일 빌드

빌드 결과물을 배포할 때는 Python 설치 불필요.

### Linux

```bash
./build_linux.sh
# 출력: dist/AstroPipeline
```

### Windows

```bat
build_windows.bat
:: 출력: dist\AstroPipeline.exe
```

두 스크립트는 공통 PyInstaller 스펙 파일(`astro_pipeline.spec`)을 사용하며, 모든 과학 라이브러리 의존성을 자동으로 수집합니다.

> **주의:** PyInstaller는 크로스 컴파일이 불가합니다. 배포 대상 OS에서 직접 빌드하세요.
> 첫 실행 시 `/tmp`(Linux) 또는 `%TEMP%`(Windows)에 압축을 해제하므로 5–15초가 소요됩니다. 이후 실행은 빠릅니다.

---

## 출력 구조

```
<출력 폴더>/
├── step01_pipp/              # 크롭된 SER + 프레임 제거 통계
├── step03_wavelet_preview/   # 필터별 PNG 미리보기 (IR/R/G/B/CH4)
├── step04_quality/           # 품질 CSV, 윈도우 JSON, 필터별 순위
├── step05_derotated/         # 윈도우별 자전 보정 16-bit TIF
├── step06_wavelet_master/    # 윈도우별 마스터 샤프닝 PNG
├── step07_rgb_composite/     # RGB/IR-RGB/CH4-G-IR 합성 이미지
├── step08_series/            # 시계열 합성 프레임
├── step09_gif/               # 애니메이션 GIF
└── step10_summary_grid/      # 최종 컨택 시트 PNG
```

---

## 워크플로

```
SER 파일
  └─► Step 01 (PIPP 크롭)
        └─► [AS!4 외부 스태킹]
              └─► Step 03 (웨이블릿 미리보기)
                    └─► Step 04 (품질 평가)
                          ├─► Step 05 (자전 보정 스택)
                          │     └─► Step 06 (웨이블릿 마스터)
                          │           └─► Step 07 (RGB 합성)
                          │                 └─► Step 10 (서머리 그리드)
                          └─► Step 08 (시계열 합성)
                                └─► Step 09 (애니메이션 GIF)
```

---

## 일반적인 사용 순서

1. 행성 카메라(예: Firecapture)로 SER 파일 촬영
2. **Step 01** 실행 — 불량 프레임 제거 및 행성 ROI 크롭
3. **AutoStakkert! 4** 로 외부 스태킹
4. **Step 03–04** 실행 — 미리보기 확인 및 품질 평가
5. **Step 05–07** 실행 — 자전 보정 최종 합성
6. *(선택)* **Step 08–10** 실행 — 시계열 애니메이션 및 서머리 출력

---