# PlanetFlow — 알고리즘 테크니컬 가이드

---

## 목차

1. [개요](#1-개요)
2. [Step 01 — PIPP 전처리](#2-step-01--pipp-전처리)
3. [Step 02 — Lucky Stacking](#3-step-02--lucky-stacking)
4. [Step 03 — 품질 평가 및 윈도우 탐지](#4-step-03--품질-평가-및-윈도우-탐지)
5. [Step 04 — De-rotation 스태킹](#5-step-04--de-rotation-스태킹)
6. [Step 05 / 07 — 웨이블릿 선명화](#6-step-05--07--웨이블릿-선명화)
7. [Step 06 / 08 — RGB 합성](#7-step-06--08--rgb-합성)
8. [Step 09 — 애니메이션 GIF](#8-step-09--애니메이션-gif)
9. [Step 10 — 요약 그리드](#9-step-10--요약-그리드)
10. [공통 모듈: 디스크 감지](#10-공통-모듈-디스크-감지-find_disk_center)
11. [공통 모듈: 서브픽셀 정렬](#11-공통-모듈-서브픽셀-정렬)

---

## 1. 개요

이 문서는 GUI의 각 파라미터가 내부 알고리즘에서 **실제로 무슨 일을 하는지** 설명합니다. 파라미터 사용법은 `guide_ko.md`를 참조하고, 이 문서는 파라미터의 의미와 수학적 원리를 이해하고자 할 때 활용하세요.

### 소스 파일 구조

```
pipeline/
├── modules/
│   ├── planet_detect.py    # Step 01: 행성 감지 및 검증
│   ├── lucky_stack.py      # Step 02: Lucky Stacking 핵심 알고리즘
│   ├── quality.py          # Step 03: 화질 평가 및 윈도우 선택
│   ├── derotation.py       # Step 04: 자전 보정 워프 및 스태킹
│   ├── wavelet.py          # Step 05/07: À trous 웨이블릿 선명화
│   └── composite.py        # Step 06/08: RGB/LRGB 합성
└── config.py               # 전역 설정 (dataclass 기반)
```

---

## 2. Step 01 — PIPP 전처리

**소스**: `pipeline/modules/planet_detect.py`, `pipeline/steps/step01_pipp.py`

```
입력 프레임
    │
    ▼
8비트 그레이스케일 변환
    │
    ▼
GaussianBlur(5×5)  ← 노이즈 억제
    │
    ▼
Triangle 임계값 (Zack 1977)
    │
    ▼
최대 연결 컴포넌트 (8-연결) 추출
    │
    ▼
4단계 검증 → 실패 시 프레임 거부
    │
    ▼
바운딩박스 중심으로 정사각형 ROI 크롭
```

### GUI 파라미터 → 내부 동작

| GUI 파라미터 | 기본값 | 내부 동작 |
|---|---|---|
| **ROI 크기 (px)** | 448 | `get_cropped_frame()`의 출력 정사각형 크기. `round()`로 정수 변환하여 0.5픽셀 체계적 편향 방지. 범위 밖 픽셀은 0(검정)으로 채움 |
| **최소 원반 지름 (px)** | 50 | 4단계 검증의 마지막 기준. `max(bw, bh) < min_diameter`이면 프레임 거부 |

### 내부 고정값

| 파라미터 | 값 | 역할 |
|---|---|---|
| `padding` | 10 px | 경계 검사: 행성 바운딩박스가 이미지 엣지에서 이만큼 이상 떨어져야 함 |
| `aspect_ratio_limit` | 0.2 | 종횡비 검사: `min(w,h)/max(w,h) ≥ 1 − 0.2 = 0.8` 이어야 통과 |
| `straight_edge_limit` | 0.5 | 직선 엣지 검사: 바운딩박스 4변 중 50% 이상 점등된 변이 있으면 거부 |
| GaussianBlur 커널 | 5×5 | 임계값 전 노이즈 억제 |

### Triangle 자동 임계값

OpenCV `THRESH_TRIANGLE` 플래그로 구현합니다. 히스토그램의 최고 피크에서 가장 먼 최솟값을 임계값으로 결정합니다. 노출값이나 행성 크기에 관계없이 안정적으로 동작합니다.

### 바운딩박스 중심 크롭

목성은 벨트·대적점으로 밝기가 불균일합니다. 밝기 가중 무게중심은 밝은 구조물 쪽으로 편향이 생기므로, PIPP와 동일하게 **바운딩박스 중심** `(x + w/2, y + h/2)`을 사용합니다.

---

## 3. Step 02 — Lucky Stacking

**소스**: `pipeline/modules/lucky_stack.py`

```
SER 입력 파일
    │
    ▼
[1단계] 프레임 품질 평가 (log_disk 지표)
    │
    ▼
상위 top_percent% 프레임 선별 → selected_indices
    │
    ▼
[2단계] 기준 프레임(Reference) 구성
    품질 분포 75퍼센타일 부근 프레임
    → 전역 NCC 정렬 → 평균 스택 (안정적이고 대표적인 기준)
    │
    ▼
[3단계] AP 격자 생성 (균일 격자 또는 Greedy PDS 3 레이어)
    │
    ▼
[4단계] 프레임별 전역 정렬
    림브 중심 타원 피팅 → bicubic 서브픽셀 워프 (INTER_CUBIC)
    │
    ▼
[5단계] Fourier 도메인 품질 가중 스태킹
    각 프레임 n에 대해 F_n = FFT(정렬된 프레임 n)
    누적: S(f) += |F_n(f)|^power × F_n(f)
    가중치: W(f) += |F_n(f)|^power
    스택 스펙트럼: S(f) / W(f)
    │
    ▼
[6단계] 주파수 도메인 Gaussian 롤오프 필터
    Gaussian(σ_f = 0.20, 정규화 주파수 기준) → IFFT → 스택 이미지
    │
    ▼
n_iterations = 2 이면 결과를 기준 프레임으로 → [3단계] 재반복
    │
    ▼
출력 TIF
```

### GUI 파라미터 → 내부 동작

| GUI 파라미터 | 기본값 | 내부 동작 |
|---|---|---|
| **상위 프레임 비율 (%)** | 25 | `top_percent = 0.25`. 품질 점수 기준 상위 N% 프레임만 스태킹에 사용. `n_select = max(min_frames, round(n_frames × top_percent))` |
| **AP 크기 (px)** | 64 | AP 격자 기준 크기 s. PDS 사용 시 Layer 1=s, Layer 2=round(s×1.5/8)×8, Layer 3=s×3. AP 스텝 기본값은 AP크기 ÷ 2 (ap_step=0 = 자동). |
| **반복 횟수** | 1 | `n_iterations`. 2 설정 시 1회 스택 결과를 2회의 기준 프레임으로 사용 → 기준 프레임 SNR 향상 → 정렬 정밀도 향상 |
| **Warp 방식** | Gaussian KR | 워프 필드 보간 방식. **Gaussian KR** (기본): C∞ 연속 Nadaraya-Watson 커널 회귀. **TPS** (Thin Plate Spline): 날카로운 국소 보정, AS!4 삼각분할과 유사하지만 속도 느림, 디스크 가장자리에서 외삽 불안정 가능. |
| **Fourier Quality Power** | 1.0 | `w_n(f) = │FFT_n(f)│^power`. 주파수별 누적의 핵심 가중치. 값이 높을수록 선명한 프레임이 고주파수에서 더 큰 영향력을 가짐. 1.0=선형(기본), 1.5–2.0=적극적. |
| **SER 병렬 처리** | 1 | 동시 처리 SER 파일 수. 0=자동(CPU 코어 수÷4). 총 스레드 예산 = n_workers 고정. 각 SER는 `n_workers ÷ N_SER`개 프레임 레벨 스레드 할당. SER당 약 950 MB RAM. |
| **AS!4 AP 그리드** | Off | Off=균일 격자 (간격=AP크기÷2). On=Greedy PDS 3레이어: 디스크 중심부 조밀, 림브 방향으로 성기게 배치 |

### 내부 고정값

| 파라미터 | 값 | 역할 |
|---|---|---|
| `score_metric` | `"log_disk"` | 프레임 품질 평가 방식. AS!4의 *lapl3* 지표와 동일한 원리. `"local_gradient"`, `"laplacian"`도 config에서 선택 가능 |
| `reference_midpoint_percentage` | 75 | 기준 프레임을 품질 분포 75퍼센타일 부근으로 중심 설정 (최상위 아님). "안정적으로 좋은" 프레임으로 구성하면 위상 상관 추정 품질이 향상됨 (AS!4 기본값과 동일). |
| `reference_n_frames` | 50 | 기준 프레임 구성에 사용할 프레임 수 (midpoint_percentage 기준으로 중심) |
| `score_step` | 2 | 매 2번째 프레임만 실제 계산, 나머지는 선형 보간 |
| `ap_confidence_threshold` | 0.15 | 위상 상관 신뢰도가 이 값 미만이면 해당 AP 폐기 |
| `ap_sigma_factor` | 0.7 | 가우시안 KR의 σ = ap_step × 0.7. σ ≥ ap_step/√2 조건을 충족하여 C∞ 연속 워프 필드 보장 |
| `remap_interpolation` | `INTER_CUBIC` | 전역 워프에 사용할 cv2.remap 보간 방식 (bicubic; LINEAR보다 선명, 후처리 블러 불필요) |
| `fourier_rolloff_sigma` | 0.20 | Gaussian 롤오프 σ (정규화 주파수 단위: 0=DC, 0.5=나이퀴스트). 행성 디테일 블러 없이 잔류 고주파 노이즈 억제. |

### AP 크기와 격자 배치

**균일 격자** (AS!4 AP 그리드 Off): AP 간격 = AP크기 ÷ 2. 디스크 내부를 균일하게 커버합니다.

**Greedy PDS** (AS!4 AP 그리드 On): AP 크기 s를 기준으로 3개의 독립 레이어를 래스터 스캔 방식으로 생성합니다.

| 레이어 | AP 크기 (s=64 기준) | 최소 AP 간 거리 |
|--------|---------------------|-----------------|
| Layer 1 | 64px | `round(64 × 35/64)` = 35px |
| Layer 2 | 96px (`round(64×1.5/8)×8`) | `round(96 × 35/64)` = 52px |
| Layer 3 | 192px (`64×3`) | `round(192 × 35/64)` = 105px |

각 AP의 수용 조건: ① 디스크 내부, ② 패치 평균 밝기 ≥ 0.196 (50/255), ③ 기존 AP와의 최소 거리 이상. 적분 이미지(Integral Image)로 O(1) 패치 평균을 계산합니다.

### 프레임 품질 평가 방식 (score_metric)

config에서 선택하며 기본값은 `"log_disk"`입니다.

**`"log_disk"`** (기본): AS!4 *lapl3* 지표에 대응합니다. Gaussian 블러 후 Laplacian 분산을 임계값 이상의 픽셀에서 계산합니다. AS!4 프레임 점수와 Spearman 상관계수 0.74 (sigma=3.0, threshold=0.25).
```
mask = (frame / max) > 0.25
score = var(Laplacian(GaussianBlur(frame, σ=3.0)))  on mask
```

**`"local_gradient"`**: 각 AP 패치에서 최대 Sobel 그레디언트 계산. 최대값 사용 이유: 변동계수(CV≈6%)가 평균(CV≈1.4%)보다 4배 높아 나쁜 시잉에서도 프레임 간 변별력이 좋음.
```
patch_score = max(gx² + gy²)  over ap_size × ap_size
frame_score = mean(patch_score) over all APs
```

**`"laplacian"`**: 디스크 내부 80% 영역에서 Laplacian 분산. 림브 경계(시잉과 관계없이 항상 큰 그레디언트)를 제외하여 대기 투명도만 측정합니다.
```
mask = dist_from_center ≤ disk_radius × 0.80
score = var(Laplacian(frame / 255))  on mask
```

### Fourier 도메인 품질 가중 스태킹

기본 스태킹 알고리즘은 공간 주파수별 프레임 품질 가중 평균을 주파수 도메인에서 수행합니다 (Mackay 2013, arXiv:1303.5108):

```
각 전역 정렬 프레임 n에 대해:
    F_n(f) = FFT(정렬된 프레임 n)
    가중치  = |F_n(f)|^power          주파수별 가중치

스택 스펙트럼:
    S(f) = Σ_n [weight_n(f) × F_n(f)] / Σ_n weight_n(f)

Gaussian 롤오프:
    G(f) = exp(−f² / (2σ_f²))        σ_f = 0.20 (정규화 주파수)
    S_filtered(f) = S(f) × G(f)

출력:
    stack = real(IFFT(S_filtered))
```

**Fourier 도메인 가중치를 사용하는 이유**: 단순 평균은 모든 프레임을 모든 주파수에서 동등하게 처리합니다. 일부 프레임이 고주파(행성의 미세 디테일)에서만 더 선명하다면, 그 기여가 희석됩니다. Fourier 가중치는 각 주파수에서 가장 선명한 프레임이 가장 많이 기여하도록 하여 단순 평균보다 세밀한 스케일에서 더 많은 에너지를 가진 스택을 만듭니다.

**Gaussian 롤오프 근거**: 모든 스태킹 방법은 고주파 노이즈(보간 앨리어싱, 카메라 읽기 노이즈)를 어느 정도 누적합니다. 롤오프는 신호 대 노이즈가 낮아지는 약 0.2×나이퀴스트 이상 주파수를 억제합니다. L1 웨이블릿 선명화(×200)가 행성 디테일을 회복할 수 있도록 조정됩니다.

### 국소 워프 추정 및 가우시안 KR

AP별 Hann 윈도잉 위상 상관 + QSF(2차 곡면 피팅) 서브픽셀 정밀화로 전역 워프 맵의 시프트를 추정합니다. 신뢰할 수 있는 AP의 시프트를 가우시안 커널 회귀(Nadaraya-Watson)로 전해상도 워프 필드로 보간합니다:

```
sigma = ap_step × ap_sigma_factor    (기본: 32 × 0.7 = 22.4px)

smooth_wx = GaussianBlur(shift_x × confidence, ksize, sigma)
smooth_w  = GaussianBlur(confidence, ksize, sigma)
map_dx    = smooth_wx / smooth_w     (커버리지 ≥ 최대의 5% 영역)
```

**Delaunay 대신 가우시안 KR을 사용하는 이유**: Delaunay 선형 보간은 삼각형 경계에서 그레디언트 불연속(C⁰)이 발생합니다. 수천 장 스태킹 후 이 메시 패턴이 누적되고, 웨이블릿 선명화(×200)가 가시적인 격자 아티팩트로 증폭합니다. 가우시안 KR은 C∞ 연속 필드를 생성합니다.

---

## 4. Step 03 — 품질 평가 및 윈도우 탐지

**소스**: `pipeline/modules/quality.py`

```
Step 02 TIF 파일 목록
    │
    ▼
각 TIF별:
    Otsu 임계값 → 디스크 마스크 추출
    GaussianBlur(σ=1.2) → 디노이즈
    Laplacian 분산 (×0.5) + Tenengrad (×0.3) + 정규화 분산 (×0.2)
    → 복합 raw_score
    │
    ▼
필터별 min-max 정규화 → norm_score ∈ [0, 1]
    │
    ▼
후보 윈도우 × 필터별:
    σ-클리핑 (1.5σ) → 아웃라이어 제거
    quality_post × snr_factor × stability → filter_quality
    │
    ▼
필터 간 기하 평균 → window_quality
    │
    ▼
비겹침 조건으로 상위 N개 윈도우 선택
    │
    ▼
windows.json / *_ranking.csv 출력
```

### GUI 파라미터 → 내부 동작

| GUI 파라미터 | 기본값 | 내부 동작 |
|---|---|---|
| **윈도우 (프레임 수)** | 3 | 탐지 윈도우의 길이를 필터 사이클 수로 지정. 실제 윈도우 시간 = 프레임 수 × 필터 사이클(초). `n_expected = window_frames`로 snr_factor 계산에 사용 |
| **필터 사이클 (초)** | 225 | 1 필터 사이클(IR→R→G→B→CH4→IR) 소요 시간. `n_expected = window_minutes / cycle_minutes`에서 기대 프레임 수 계산에만 사용. Step 8의 사이클 시간과 독립 |
| **윈도우 개수** | 1 | 탐지할 최적 윈도우 수. Step 04는 윈도우 1개 사용, Step 08 시계열은 여러 개 사용 |
| **윈도우 겹침 허용** | Off | Off 시: 각 윈도우 중심이 이미 선택된 모든 윈도우로부터 ≥ window_minutes 떨어져야 함 |
| **최소 품질 임계값** | 0.05 | `norm_score < threshold`인 프레임을 윈도우 품질 계산에서 제외. 0.0이면 모든 프레임 포함 |

### 내부 고정값

| 파라미터 | 값 | 역할 |
|---|---|---|
| Laplacian 가중치 | 0.5 | 복합 점수에서 Laplacian 분산의 비중 |
| Tenengrad 가중치 | 0.3 | 복합 점수에서 Tenengrad(Sobel 제곱합)의 비중 |
| 정규화 분산 가중치 | 0.2 | 복합 점수에서 `var/mean`의 비중 |
| 디노이즈 σ | 1.2 px | 선명도 계산 전 가우시안 블러. 노이지한 프레임이 고점수를 받는 현상 방지 |
| σ-클리핑 기준 | 1.5σ | 윈도우 내 이상치 프레임 제거 기준 |

### 윈도우 품질 계산

각 후보 윈도우에 대해 필터별로 품질을 계산한 뒤 기하 평균합니다:

```
# 필터별
quality_post = mean(norm_score of included)
snr_factor   = min(1.0, √(n_included / n_expected))
stability    = 1 / (1 + CV)          CV = std/mean

filter_quality = quality_post × snr_factor × stability

# 윈도우 전체
window_quality = (∏_f  filter_quality_f) ^ (1 / num_filters)
```

**기하 평균 사용 이유**: 한 필터가 매우 나쁘면 전체 품질이 대폭 떨어집니다. 모든 필터가 최소 기준을 충족해야 좋은 합성 이미지를 만들 수 있기 때문입니다.

---

## 5. Step 04 — De-rotation 스태킹

**소스**: `pipeline/modules/derotation.py`

```
Step 02 TIF + windows.json
    │
    ▼
기준 프레임에서 디스크 감지 (윈도우 전체 공유)
    Otsu → Closing(7×7) → fitEllipse → (cx, cy, semi_a, semi_b, angle)
    │
    ▼
NP.ang 조회 (번들 테이블 → 사용자 캐시 → 라이브 Horizons API)
    │
    ▼
윈도우 내 각 프레임:
    ├─ 촬영 시각 Δt → 경도 변위 Δλ_rad
    ├─ 납작한 구형 깊이 계산 → 픽셀별 drift
    ├─ remap (CUBIC 내부 / LINEAR 림브, 12px 코사인 페더)
    └─ 서브픽셀 정렬 (림브 중심 → 폴백: 위상 상관)
    │
    ▼
품질 가중 누적 → 마스터 TIF 출력
```

### GUI 파라미터 → 내부 동작

| GUI 파라미터 | 기본값 | 내부 동작 |
|---|---|---|
| **Warp Scale** | 0.80 | 구면 워프 강도 배율. `drift = warp_scale × Δλ_rad × depth(x,y)`. 이론값 1.0이지만 대기 블러·플레이트 스케일 불확실성으로 0.80이 실험적 최적값. 자동탐색 버튼으로 Laplacian 분산이 최대가 되는 값을 탐색 |
| **최소 품질 임계값** | 0.05 | `norm_score < threshold`인 프레임은 스태킹 누적에서 제외 |
| **밝기 정규화** | Off | 스태킹 전 각 프레임의 밝기를 기준 프레임 밝기에 맞춰 정규화. 프레임 간 밝기 변동이 클 때 사용 |

### 내부 고정값

| 파라미터 | 값 | 역할 |
|---|---|---|
| `polar_equatorial_ratio` | 0.935 (목성) | 납작한 구형의 극/적도 반경 비. 깊이 공식의 `polar_scale = 1 / ratio` |
| R (구형 반경) | `disk_radius × 1.05` | 5% 패딩: 림브에서 `√(R²−r²)`의 특이점 회피 |
| `_interp_feather_px` | 12.0 px | CUBIC/LINEAR 보간 전환 구간. 림브 안쪽 12px에서 코사인 페이드 |
| `margin_factor` | 0.10 | Otsu 임계값을 10% 낮춰 어두운 림브 픽셀 포함 |

### 구면 De-rotation 워프 공식

행성 표면의 경도 변위 Δλ에 의한 픽셀 이동량은 구체 깊이에 비례합니다:

```
Δλ_rad = (dt_sec / period_sec) × 2π

# 극 방향각(pole_pa_deg=NP.ang)으로 좌표 분해
rx_eq  = (x−cx)×cos(pa) + (y−cy)×sin(pa)   (적도 방향)
ry_pol = -(x−cx)×sin(pa) + (y−cy)×cos(pa)  (극 방향)

# 납작한 구형 깊이
depth² = R² − rx_eq² − polar_scale² × ry_pol²
depth  = sqrt(max(0, depth²))

drift  = warp_scale × Δλ_rad × depth

map_x  = x − drift × cos(pole_pa_rad)
map_y  = y − drift × sin(pole_pa_rad)
```

### 공유 디스크 중심의 중요성

프레임마다 독립적으로 디스크를 감지하면 (cx, cy)가 몇 픽셀씩 달라져 각 프레임에 약간 다른 구면 워프가 적용됩니다. 스태킹 후 림브 경계가 어긋나고 웨이블릿 선명화가 이를 비대칭 림브 아티팩트로 증폭합니다. 그래서 기준 프레임 하나에서만 감지하고 전체 윈도우에 동일한 값을 적용합니다.

### NP.ang 조회 우선순위

1. **번들 테이블** (오프라인): `pipeline/data/np_ang_table.json` — 목성(599), 토성(699), 화성(499)의 2016~2036년 데이터. 날짜 내 선형 보간 (360°/0° 래핑 포함).
2. **사용자 캐시**: `~/.astropipe/horizons_cache.json` — 이전 온라인 조회 결과.
3. **라이브 Horizons API**: 번들 범위 밖 또는 Custom 행성.

---

## 6. Step 05 / 07 — 웨이블릿 선명화

**소스**: `pipeline/modules/wavelet.py`

Step 05 (마스터 선명화)와 Step 07 (미리보기)은 동일한 알고리즘 사용. 파라미터와 대상 이미지만 다릅니다.

```
입력 TIF
    │
    ▼
디스크 감지 → (cx, cy, rx, ry, angle)
    │
    ▼
auto_wavelet_params:
    expand_px = sqrt(rx·ry) × 0.0505
    eff = 림브 내측 밝기 그레디언트 너비 중앙값 / 2
    │
    ▼
타원 외부 사전 채움 (림브→배경 불연속 제거 → 링잉 방지)
    │
    ▼
À Trous B3 웨이블릿 분해 (6레벨)
    → [detail_0 (~2px), …, detail_5 (~64px), residual]
    │
    ▼
각 레벨 i:
    σ_noise = MAD(detail_i) / 0.6745
    gain_i  = (amount_i/200)^power × MAX_GAIN[i]
    weight_i = 코사인 S-커브 타원 마스크 (feather = 2^i × eff)
    contrib_i = soft_threshold(detail_i, gain_i × σ_noise) × gain_i × weight_i
    │
    ▼
재구성: original + Σ contrib_i → PNG 출력
```

### GUI 파라미터 → 내부 동작

| GUI 파라미터 | 기본값 | 내부 동작 |
|---|---|---|
| **L1 (0–500)** | 200 | ~2픽셀 스케일 웨이블릿 계수 증폭. `gain = (200/200)^1.0 × 29.15 = 29.15`. 픽셀 수준 최고 해상도 디테일 |
| **L2 (0–500)** | 200 | ~4픽셀 스케일. `gain = (200/200)^1.0 × 9.48 = 9.48`. 세밀한 구조(벨트, 줄무늬) |
| **L3 (0–500)** | 200 | ~8픽셀 스케일. `MAX_GAIN[2] = 0.0` → 현재 비활성화 상태. 중간 규모 구조 |
| **L4 (0–500)** | 0 | ~16픽셀 스케일. `MAX_GAIN[3] = 0.0`. 대규모 명암 대비 (노이즈 증폭 위험) |
| **L5, L6 (0–500)** | 0 | ~32/64픽셀 스케일. `MAX_GAIN[4,5] = 0.0`. 사용 비권장 |

> **amount의 의미**: `gain_i = (amount/200)^power × MAX_GAIN[i]`. amount=200 → gain=MAX_GAIN. amount=400 → gain=2×MAX_GAIN. amount=100 → gain=0.5×MAX_GAIN.

### 내부 고정값

| 파라미터 | 값 | 역할 |
|---|---|---|
| `MAX_GAINS` | [29.15, 9.48, 0, 0, 0, 0] | WaveSharp 기준 출력과 OLS 회귀로 결정한 레벨별 최대 이득 |
| `sharpen_filter` | 0.1 | 소프트 임계값 강도. `thr = 0.1 × σ_noise`. 작은 노이즈 계수를 억제 |
| `power` | 1.0 | `gain = (amount/200)^power × MAX_GAIN`. 1.0=선형 |
| `edge_feather_factor` | 자동 | 림브 페더 너비 계수. auto_wavelet_params()가 이미지에서 자동 측정 |
| `expand_px` | 자동 | Otsu 경계를 바깥으로 밀어 실제 림브에서 페더 시작. `sqrt(rx×ry) × 0.0505` |

### À Trous B3-스플라인 웨이블릿 분해

"À trous"(구멍이 있는)는 다운샘플링 없이 탭 사이에 0을 삽입하여 스케일을 확장하는 Undecimated 웨이블릿입니다.

```
_B3 = [1, 4, 6, 4, 1] / 16   (5탭 분리 가능 커널)

레벨 i에서 탭 간격 = 2^i:
  smoothed_i = B3_i ⊗ image_i    (reflect 패딩)
  detail_i   = image_i − smoothed_i
  image_{i+1} = smoothed_i
```

### 디스크 인식 엣지 페더링

웨이블릿 이득을 행성 디스크 내부에만 적용하고 림브에서 코사인 S-커브로 페이드아웃합니다. 레벨이 높을수록 더 넓은 페더를 사용합니다:

```
feather_L = 2^L × edge_feather_factor
t = clip(dist_from_boundary / feather_L, 0, 1)
weight_L = 0.5 × (1 − cos(π × t))
```

**타원 외부 사전 채움**: 웨이블릿 분해 전 디스크 외부를 가장 가까운 림브 픽셀 값으로 채웁니다. B3 커널이 배경 0값을 읽어 발생하는 밝은 림브 링 아티팩트를 방지합니다.

---

## 7. Step 06 / 08 — RGB 합성

**소스**: `pipeline/modules/composite.py`

```
필터별 PNG (R, G, B, [IR, L, …])
    │
    ▼
채널 자동 스트레치 (joint / independent / none)
    │
    ▼
고정 기준 채널 선택: L > IR > R > G > B
    비기준 채널 → 위상 상관 → apply_shift
    │
    ▼
np.stack([R, G, B])
    │
    ├─ [RGB 모드] 그대로 진행
    └─ [LRGB 모드] RGB→Lab, Lab_L 교체, Lab→RGB
    │
    ▼
디스크 감지 → Lab 변환 → a/b 채널 코사인 페이드 (0.89r ~ 1.04r)
    │
    ▼
RGB PNG 출력
```

### Step 06 GUI 파라미터 → 내부 동작 (모노 모드)

| GUI 파라미터 | 기본값 | 내부 동작 |
|---|---|---|
| **최대 채널 이동량 (px)** | 15.0 | 위상 상관으로 계산한 채널 간 시프트가 이 값을 초과하면 정렬을 적용하지 않음. 대기 분산이 심한 날에는 높여야 함 |
| **합성 스펙 (R/G/B/L 채널)** | RGB, IR-RGB, CH4-G-IR | 각 합성 이미지의 필터→채널 매핑 정의. L 채널 지정 시 LRGB 합성 모드 |

### Step 08 GUI 파라미터 → 내부 동작 (모노 모드)

| GUI 파라미터 | 기본값 | 내부 동작 |
|---|---|---|
| **글로벌 필터 정규화** | On | 전체 시계열에 걸쳐 각 필터의 밝기 범위를 통일. Step 9 GIF 프레임 간 색상 이질감 감소 |
| **밝기 배율** | 1.00 | `composite × series_scale`. 1.0=변경 없음 |
| **윈도우 (프레임 수)** | 3 | 슬라이딩 윈도우 크기. 홀수 권장. 3=앞뒤 1개씩. SNR 향상 ∝ √N |
| **필터 사이클 (초)** | 225 | Step 07 PNG를 시계열 프레임 세트로 그룹핑할 때 사용. Step 3의 사이클 시간과 독립 |
| **최소 품질 필터** | 0.05 | 품질 낮은 프레임의 기여를 완전 제외 없이 가중치 감소 |
| **필터별 모노 GIF** | Off | On 시 각 필터의 흑백 프레임도 저장 → Step 9에서 필터별 흑백 GIF 생성 |
| **L1–L6 (시계열)** | [200, 200, 200, 0, 0, 0] | 각 시계열 프레임에 독립적으로 적용하는 웨이블릿 선명화. Step 5와 완전히 별개 설정 |
| **합성 스펙** | RGB, IR-RGB, CH4-G-IR | Step 6과 독립적인 시계열 전용 채널 매핑 |

### 내부 고정값

| 파라미터 | 값 | 역할 |
|---|---|---|
| 기준 채널 우선순위 | L > IR > R > G > B | 동적 선택 시 프레임마다 기준이 달라지는 문제 방지를 위해 고정 |
| `desat_start` | `disk_radius × 0.89` | 림브 채도 감소 시작 반경 |
| `desat_width` | `disk_radius × 0.15` | 코사인 페이드 구간 (1.04×r에서 완료) |
| stretch 기본값 | `"none"` | 자동 스트레치 없음. `"joint"` = R/G/B 통합 lo/hi, `"independent"` = 채널별 |

### LRGB 합성

L 채널이 지정되면 Lab 색공간에서 휘도(L)를 외부 채널로 교체합니다:

```
Lab = cv2.cvtColor(rgb, COLOR_RGB2Lab)
Lab[:,:,0] = lrgb_weight × (L_external × 100) + (1−w) × Lab[:,:,0]
result = cv2.cvtColor(Lab, COLOR_Lab2RGB)
```

IR-RGB 합성 시: IR 채널의 높은 해상도가 밝기 디테일을 살리고, R/G/B가 자연스러운 색상을 부여합니다.

### 사후 림브 채도 감소

파장에 따른 림브 암화 차이(G 디스크가 B보다 약 1.5픽셀 크게 보임)로 인한 색상 프린지를 제거합니다. Lab 색공간에서 a/b 채널(채도)만 억제하며 L 채널(밝기)은 보존합니다.

---

## 8. Step 09 — 애니메이션 GIF

**소스**: `pipeline/steps/step09_gif.py`

```
Step 08 시계열 합성 PNG (타임스탬프 기준 정렬)
    │
    ▼
scale_factor로 bilinear 리샘플링
    │
    ▼
Pillow ImageSequence 조립
    frame_duration = 1000 / fps  [ms]
    │
    ▼
GIF 출력 (loop=0, 무한 반복)
```

### GUI 파라미터 → 내부 동작

| GUI 파라미터 | 기본값 | 내부 동작 |
|---|---|---|
| **FPS** | 6.0 | `frame_duration = round(1000 / fps)` [ms]. Pillow `save(duration=...)` 인수로 전달 |
| **크기 배율** | 1.0 | `new_size = (round(w × factor), round(h × factor))`. Pillow BILINEAR 리샘플링 |

---

## 9. Step 10 — 요약 그리드

**소스**: `pipeline/steps/step10_summary_grid.py`

```
Step 06 RGB 합성 PNG 목록
    │
    ▼
각 이미지:
    블랙 포인트 보정: pixel = clip((p − bp) / (1 − bp), 0, 1)
    감마 보정:        pixel = pixel ^ (1 / gamma)
    cell_size로 리샘플링
    │
    ▼
그리드 배치 → 단일 요약 PNG 출력
```

### GUI 파라미터 → 내부 동작

| GUI 파라미터 | 기본값 | 내부 동작 |
|---|---|---|
| **블랙 포인트** | 0.04 | `pixel = clip((p − 0.04) / (1 − 0.04), 0, 1)`. 배경 노이즈를 순수 검정으로 밀어냄. 0.02~0.08 권장 |
| **감마 (Gamma)** | 0.9 | `pixel = pixel ^ (1/0.9) ≈ pixel ^ 1.11`. <1.0=밝아짐(기본 0.9는 행성을 약간 밝게), >1.0=어두워짐, 1.0=변화 없음 |
| **셀 크기 (px)** | 300 | 그리드 내 각 합성 이미지를 이 크기로 리샘플링. 전체 그리드 크기는 합성 수에 따라 결정됨 |

---

## 10. 공통 모듈: 디스크 감지 (find_disk_center)

**소스**: `pipeline/modules/derotation.py`

여러 Step(04, 05, 06, 08)에서 공통으로 사용합니다.

```
1. arr8 = clip(image × 255, 0, 255).uint8
2. Otsu 임계값 → effective_thresh = Otsu × (1 − 0.10)
   (margin_factor=0.10: 어두운 림브 픽셀 포함을 위해 임계값을 낮춤)
3. 형태학적 Closing (7×7 타원형 커널) → 디스크 내 작은 갭 메움
4. 최대 윤곽선 추출 (≥5점이면 타원 피팅):
   (cx, cy), (ma, mi), angle = cv2.fitEllipse(largest_contour)
5. 반환: (cx, cy, semi_major, semi_minor, angle_deg)
   (항상 semi_major ≥ semi_minor 보장; 필요 시 angle + 90°)
```

---

## 11. 공통 모듈: 서브픽셀 정렬

**소스**: `pipeline/modules/derotation.py`

### apply_shift

```python
M = np.float32([[1, 0, dx], [0, 1, dy]])
cv2.warpAffine(image, M, (w, h), flags=INTER_CUBIC, borderMode=BORDER_REPLICATE)
```

- **INTER_CUBIC**: 이중 삼차 보간 (디테일 보존)
- **BORDER_REPLICATE**: 엣지 픽셀 복사 (BORDER_CONSTANT=0 사용 시 흑색이 림브로 침투)

### subpixel_align (위상 상관)

```python
(dx, dy), _ = cv2.phaseCorrelate(ref_f32, tgt_f32)
```

주파수 도메인 교차 상관으로 ~0.1픽셀 정밀도의 이동량을 추정합니다.

---

*References: Starck & Murtagh (2006), Bijaoui (1991), Mackay (2013 arXiv:1303.5108), Zack (1977)*
