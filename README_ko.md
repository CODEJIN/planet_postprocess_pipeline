# PlanetFlow

**행성 촬영 후처리 파이프라인 (GUI 포함)**

PlanetFlow는 SER 동영상 파일로 촬영된 행성 관측 데이터를 처리하는 데스크톱 애플리케이션입니다. 원시 프레임 정렬부터 웨이블릿 샤프닝, 자전 보정 스태킹, 다채널 합성, 애니메이션 GIF 출력까지 전체 후처리 워크플로를 단일 PySide6 GUI에서 자동화합니다.

**모노크롬 카메라** (필터 휠, 다중 필터 SER)와 **컬러 카메라** (단일 베이어 센서, 연속 촬영) 모두 지원합니다.

> English version: [README.md](README.md)

---

## 주요 기능

- **10단계 파이프라인** — 각 단계별 활성화/비활성화 제어
- **이중 카메라 모드** — 모노크롬 필터 휠 워크플로 및 컬러 카메라 워크플로 지원
- **프레임 품질 평가** — Laplacian 선명도 점수 기반
- **행성 자전 보정(De-rotation)** — JPL Horizons 역표(astroquery) 기반, Warp Scale 자동탐색 기능 포함
- **웨이블릿 샤프닝** — à trous 알고리즘, WaveSharp 호환 0–500 스케일, 림브 페더 제어
- **유연한 다채널 합성** — 사용자 정의 RGB/LRGB 설정 (RGB, IR-RGB, CH4-G-IR 및 커스텀 조합)
- **독립 시계열 합성** — Step 08이 Step 06과 별도의 합성 설정을 가짐
- **자동 화이트밸런스 + 색수차 보정** — 컬러 카메라 모드용 (Step 06 & 08)
- **시계열 애니메이션** — 슬라이딩 윈도우 품질 가중 스태킹 + 애니메이션 GIF 출력
- **서머리 컨택 시트** — 합성 이미지만으로 구성된 그리드를 항상 생성하고, 모노 세션+Step 05 데이터가 있으면 필터 이미지를 포함한 2존 그리드도 추가 출력; 필터 통계 및 채널 정렬 테이블을 담은 윈도우별 Analytic View 선택 출력 가능
- **실시간 미리보기 위젯** — 웨이블릿(Step 05 & 07), RGB 합성(Step 06), 레벨(Step 10), 컬러 보정(Step 06 컬러), AP 그리드(Step 02)
- **한국어 / 영어 UI** — 런타임 전환 지원
- **독립 실행 파일** — PyInstaller로 단일 바이너리 배포 (Python 설치 불필요)
- **Lucky Stacking (Step 02)** — Fourier 도메인 품질 가중 스태킹 (Mackay 2013) — 주파수별 `|FFT|^power` 가중치 + Gaussian 롤오프, AS!4 호환 PDS AP 그리드, SER 레벨 + 프레임 레벨 멀티코어 병렬 처리
- **다중 파일명 형식 지원** — FireCapture/WinJUPOS (`YYYY-MM-DD-HHMM_D-CAM-FILTER-TARGET`), ASIAIR (`Target_YYYYMMDD-HHMMSS`), SharpCap (`HH_MM_SS`, 날짜/타겟 폴더 기반) 자동 인식
- **프로필 관리** — 이름 있는 세션 프로필 — 설정 패널에서 망원경/세션 설정을 저장·불러오기·삭제 가능
- **16비트 SER 지원** — 16비트 파이프라인 전구간 지원, 정확한 센터링 및 컬러 디베이어 처리
- **시작 화면 시스템 정보** — 활성 프로필, CPU 코어, RAM, GPU 정보 표시
- **파이프라인 강제 종료** — 모든 단계 패널에 중단 버튼 제공, 모든 스레드가 실제로 종료된 시점을 버튼으로 확인 가능

---

## 파이프라인 단계

| 단계 | 이름 | 설명 |
|------|------|------|
| 01 | PIPP 전처리 | 잘림/변형 프레임 제거, 행성 중심 정렬, 정사각형 ROI 크롭 (선택) |
| 02 | Lucky Stacking | AP별 독립 패치 스태킹 (NCC 정렬 + 광역 Gaussian 블렌딩) + AS!4 호환 AP 그리드 + σ-clip (선택) |
| 03 | 품질 평가 | 각 TIF 점수화; 프레임 수 단위로 최적 시간 윈도우 탐색 |
| 04 | 자전 보정 스태킹 | 구면 워프 자전 보정 + 품질 가중 평균 스택; Warp Scale 자동탐색 |
| 05 | 웨이블릿 마스터 | 자전 보정 스택에 최종 웨이블릿 샤프닝 + 림브 페더 적용 |
| 06 | RGB 합성 | 사용자 정의 다채널 합성; 컬러 모드는 자동 WB+색수차 보정 |
| 07 | 웨이블릿 미리보기 | 개별 TIF 스택에 웨이블릿 샤프닝 적용 후 필터별 PNG 출력 (선택) |
| 08 | 시계열 합성 | 독립적인 합성 설정으로 슬라이딩 윈도우 스택 + 글로벌 정규화 (선택) |
| 09 | 애니메이션 GIF | 시계열 프레임 조합 → GIF 출력 (선택) |
| 10 | 서머리 그리드 | 합성 전용 그리드(항상) + 필터 포함 2존 그리드(모노 선택); 윈도우별 Analytic View; 블랙 포인트 + 감마 레벨 (선택) |

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
git clone https://github.com/<your-username>/PlanetFlow.git
cd PlanetFlow
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
# 출력: dist/PlanetFlow
```

### Windows

```bat
build_windows.bat
:: 출력: dist\PlanetFlow.exe
```

두 스크립트는 공통 PyInstaller 스펙 파일(`astro_pipeline.spec`)을 사용하며, 모든 과학 라이브러리 의존성을 자동으로 수집합니다.

> **주의:** PyInstaller는 크로스 컴파일이 불가합니다. 배포 대상 OS에서 직접 빌드하세요.
> 첫 실행 시 `/tmp`(Linux) 또는 `%TEMP%`(Windows)에 압축을 해제하므로 5–15초가 소요됩니다. 이후 실행은 빠릅니다.

---

## 출력 구조

```
<출력 폴더>/
├── step03_quality/           # 품질 CSV, 윈도우 JSON, 필터별 순위
├── step04_derotated/         # 윈도우별 자전 보정 16-bit TIF
├── step05_wavelet_master/    # 윈도우별 마스터 샤프닝 PNG
├── step06_rgb_composite/     # 윈도우별 RGB/IR-RGB/CH4-G-IR 합성 이미지
├── step07_wavelet_preview/   # 필터별 PNG 미리보기 (IR/R/G/B/CH4)
├── step08_series/            # 시계열 합성 프레임
├── step09_gif/               # 애니메이션 GIF
└── step10_summary_grid/      # 최종 요약 그리드 PNG
    ├── summary_grid_simple.png   # 합성 이미지만 (항상 생성)
    ├── summary_grid.png          # 합성 + 단색 필터 (모노 카메라, Step 05 데이터 있을 때)
    └── analytic/                 # 윈도우별 Analytic View (선택, 모노 전용)
```

> Step 01 (PIPP)과 Step 02 (Lucky Stacking)은 별도로 지정한 출력 폴더를 사용하며 `<출력 폴더>` 아래에 포함되지 않습니다.

---

## 워크플로

```
SER 파일
  └─► Step 01 (PIPP 크롭, 선택)
        └─► Step 02 (Lucky Stacking, 선택)
              └─► Step 03 (품질 평가)
                    └─► Step 04 (자전 보정 스택)
                          └─► Step 05 (웨이블릿 마스터)
                                └─► Step 06 (RGB 합성)
                                      └─► Step 10 (서머리 그리드, 선택)
              └─► Step 07 (웨이블릿 미리보기, 선택)
                    └─► Step 08 (시계열 합성, 선택)
                          └─► Step 09 (애니메이션 GIF, 선택)
```

---

## 일반적인 사용 순서

1. 행성 카메라(예: Firecapture)로 SER 파일 촬영
2. *(선택)* **Step 01** 실행 — 불량 프레임 제거 및 행성 ROI 크롭
3. *(선택)* **Step 02** (Lucky Stacking) 실행 — 상위 프레임 선별 및 TIF 스태킹
4. **Step 03–06** 실행 — 품질 평가, 자전 보정 스태킹, 웨이블릿 선명화, RGB 합성
5. *(선택)* **Step 07** (웨이블릿 미리보기) 및 **Step 08–09** 실행 — 시계열 애니메이션
6. *(선택)* **Step 10** 실행 — 서머리 컨택 시트 생성

또는 **▶ 일괄 실행** 버튼을 사용하면 활성화된 단계를 시작 지점(Step 1, 2, 3 중 선택)부터 입력 검증과 확인 다이얼로그를 거쳐 자동으로 실행합니다.