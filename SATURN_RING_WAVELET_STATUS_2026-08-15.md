# Saturn ring occlusion / wavelet 아티팩트 — 현재 상황 (2026-08-15)

## 배경

외부 리뷰어가 `pipeline/modules/derotation.py`를 리뷰하면서 "Saturn 3D reprojection path엔
ring occlusion mask가 안 걸린다"고 지적했으나 "Saturn은 기본적으로 linear warp를 쓰니
out of scope"라고 스스로 다운그레이드함. 사용자가 "WinJUPOS라는 이미 검증된 답이 있는데
불가능하다는 건 말이 안 된다"고 반박했고, 실제로 확인해보니 사용자 말이 맞았음.

## ✅ 완료 & 커밋됨 (919d73d)

### 1. Ring occlusion을 3D reprojection warp에 배선

- **문제**: `compute_ring_occlusion_weight()`의 ring occlusion fix(2026-08-11, fab7a80,
  실측 검증 완료)가 legacy linear warp(`spherical_derotation_warp`)에만 연결되어 있었음.
  이 세션의 실제 프로덕션 Saturn 설정(`~/.astropipe/session.json`)은
  `use_true_reprojection=True` AND `has_rings=True`를 동시에 쓰는데,
  `spherical_derotation_warp_3d()`엔 ring occlusion 파라미터 자체가 없어서
  **검증된 fix가 실제로는 조용히 무효화**돼 있었음.
- **해결**: `compute_ring_occlusion_weight_3d()` 신규 추가(같은 ring-depth 공식 재사용,
  globe-depth만 `_oblate_ortho_inverse` 기반으로 교체) + `spherical_derotation_warp_3d()`에
  `ring_crossing_mask` 파라미터 배선(값 블렌딩 방식 — 좌표 블렌딩은 3D의 비선형 reprojection에서
  물리적으로 무의미한 중간 지점을 샘플링해 사용자가 지적한 대로 아티팩트를 만듦, 그래서 값
  블렌딩으로 전환).
- design review에서 pole_pa=0일 때는 안 보이는 실제 버그 3개(flip_pole_axis 부호,
  `_oblate_ortho_inverse` 이중회전, invalid depth sentinel) 발견·수정, 회귀 테스트 추가.
- 실측 검증(derotate_window→wavelet_master.run 정식 경로, window_01 IR/R): has_rings
  on/off 차이가 이제 실제로 존재하고, disk 근처에 국소적으로 집중됨.

### 2. `_feather_ring_foreground_boundary`의 hard-edge 버그 발견·수정 (linear/3D 공통)

- **문제**: overlap 영역이 100% foreground로만 분류되고 background가 하나도 없으면
  feathering을 통째로 스킵하고 raw hard binary mask를 쓰는 분기가 있었음. window_01/IR의
  실측 geometry(pole_pa=-7°, B=-11.07°)가 정확히 이 조건이었음.
- 이 hard edge가 disk의 진짜 limb과 구조적으로 겹쳐서(overlap 자체가 `in_globe`으로
  캡핑되므로) wavelet sharpening이 밝은 쐐기 + 고리 끊김으로 증폭시킴.
- **이건 오늘 만든 버그가 아니라 2026-08-11 원래 구현부터 있던 버그** — 그때 검증 대상에
  window1/IR이 없어서 이 정확한(100%-foreground) degenerate case를 우연히 만난 적이
  없었던 것으로 보임.
- **수정**: feathering을 조건 없이 항상 적용, `overlap`으로 다시 hard-mask하지 않고
  경계 바깥 ~12px까지 자연스럽게 halo로 번지도록 함. 두 경로(linear/3D)가 이 헬퍼를
  공유하므로 한 번에 다 고쳐짐 — 실측으로 linear warp 쪽도 확인함.
- 회귀 테스트 추가(`test_feather_smooth_even_with_no_background_within_overlap` 등).

### 3. Coverage-aware sharpening + S0/S_L blend + edge extension (오늘 세션 이전 작업, 같이 커밋)

opt-in, 기본 off, 이번 세션 앞부분에서 이미 구현·검증된 상태로 발견해 같은 커밋에 포함.

**테스트**: 78개 전체 통과. 실측 crop으로 쐐기/끊김 사라짐, 국소성(disk 근처에만 변화 집중)
확인 완료.

## ✅ 해결됨 — wavelet의 `extra_rx` 경계 아티팩트 (아래 "최종 해결" 참고)

이 섹션은 원인 조사 과정 기록. 최종 해결 방법은 문서 하단 "최종 해결" 섹션 참고.

사용자가 전체 프레임(크롭 아닌)을 보고 4가지 지적:
1. 고리형 타원 밖 몇 px은 sharpening된 것 같지만 행성 상/하단은 흐림
2. 행성 disk 외곽이 흐리게 표시되며 disk가 이중으로 보임
3. 행성 우측 외곽에 흰색으로 강조된 ring effect
4. 2·3번이 고리보다 앞에 있는 것처럼 보여 전경 고리가 잘린 착시

### 원인 조사 (monkeypatch로 단계별 격리, 전부 실측 확인)

- 제 ring occlusion 값을 0으로 무력화해도 증상 그대로 → 오늘 고친 ring occlusion 코드가
  원인 아님.
- `has_rings=True`/`False`의 step04 스택이 byte-identical함을 직접 증명 → 원인은
  wavelet 단계.
- `wavelet_master.py`의 `extra_rx`(ring 영역까지 sharpening 확장하는 별도 서브시스템,
  has_rings=True일 때만 활성)를 끄거나 primary ellipse 각도를 바꿔봐도 증상 그대로.
- **결정적 테스트**: has_rings=False 스택 + 순수 `sharpen_disk_aware`(extra_rx 없음) =
  깨끗함. has_rings=True 스택(제 occlusion 값이 미세하게 반영된) + 같은 순수
  sharpening = 증상 나타남 → 원인은 `extra_rx` 서브시스템 자체.
- **wavelet 적용 전(raw) 밝기 프로파일을 직접 찍어봄** — 문제 지점은 완전히 매끄러운
  단조 감소 곡선, 계단/꺾임 전혀 없음. 즉 이건 실제 이미지 콘텐츠가 아니라 **gain map
  자체의 형태가 만드는 아티팩트**.
- Jupiter(고리 없음, `extra_rx` 로직 자체가 안 걸림)는 이 문제가 전혀 없음 — `extra_rx`가
  필요조건임을 재확인.

### 시도한 수정 2개 — 둘 다 실측에서 효과 없음

1. **Gain floor 추가**: primary ellipse feather(disk 안쪽에서 0으로 떨어짐)와 extra
   ellipse ramp(disk 바깥쪽에서 0에서 올라옴) 사이에 생기는 "V자 dip"(거의 0까지
   떨어졌다 다시 올라옴)이 원인이라 보고, 그 dip에 바닥값(0.3, 기존
   `master_coverage_confidence_floor`와 동일한 원칙) 적용. 수치상 dip은 0.019→0.23으로
   얕아짐. **시각적으로는 변화 없음.**
2. **min/max를 미분 가능한 부드러운 조합으로 교체**: `np.minimum`/`np.maximum`은 두
   입력이 교차하는 지점에서 미분 불연속(kink)을 만듦 — 값 자체는 안 끊겨도 기울기가
   꺾여서 Mach band처럼 보일 수 있음. `a*b`(soft AND), `a+b-a*b`(soft OR)로 교체.
   **이것도 시각적으로 변화 없음.**

### 현재 판단

두 수정 다 원리적으로는 타당했지만 실측에서 문제를 해결하지 못함 — **이 아티팩트가
`extra_rx`의 세부 구현 버그가 아니라, 이미 `project_ring_limb_ringing_bug` 메모리에
기록된 것과 같은 근본 원인(타원 피팅이 진짜 photometric limb보다 0.5~0.9px 어긋남)일
가능성이 높다고 판단**. 그 메모리에는 이미 3번의 시도(coverage-aware gain 감소,
flat-fill edge extension, gradient-aware edge extension)가 전부 "다른 형태의 부작용으로
대체"되며 실패했다고 기록되어 있음. 오늘 시도한 것도 결국 "경계 근처 gain을 조작"하는
같은 계열이라 같은 벽에 부딪힌 것으로 보임.

차이점: 기존엔 이 링잉이 검은 배경 쪽 limb에서 주로 보고됐는데, 오늘 발견한 건 **고리
쪽 limb에서도 같은 근본 원인이 (고리가 배경보다 밝아서) 더 두드러져 보인다**는 것 —
새로운 관찰이지만 새로운 버그는 아닐 가능성.

### 두 수정 시도 (floor / smooth-blend) — 둘 다 되돌림

`pipeline/modules/wavelet.py`에 시도했던 gain floor + smooth-blend 수정은 두 번째 외부
리뷰 권고에 따라 **`git checkout --`으로 완전히 되돌림**. 이유: 실측 효과가 없었고,
그 상태로 남겨두면 다음 세션에서 "이미 시도했다"는 걸 코드 diff만 보고 알기 어려움 —
문서(이 파일 + 메모리)에 기록하는 것으로 충분.

## 최종 해결 — 리뷰어의 A/B/C/D 실험 제안을 그대로 실행해서 확정

두 번째 외부 리뷰가 핵심을 짚었음: "mask를 더 고치기 전에, **완전히 동일한 frozen
step04 배열**에 대해 (A) sharpening 없음, (B) 전역 uniform wavelet(mask 없음), (C)
primary disk mask만, (D) primary + extra_rx를 비교해서 원인을 먼저 확정하라."

### 실행 결과 (실측)

| Case | 방식 | 결과 |
|---|---|---|
| A | sharpening 없음 | 깨끗함 (당연) |
| **B** | **전역 uniform `wavelet.sharpen()`, mask 없음** | **밝은 선 나타남** |
| **C** | **`sharpen_disk_aware()` primary ellipse만, `extra_rx=None`** | **깨끗함** |
| D | 현재 프로덕션 (primary + `extra_rx`) | 밝은 선 나타남 |

### 메커니즘 결론

primary disk mask 자체가 disk 경계에서 gain을 0으로 페더링하는 게 **B에서 드러나는
이 아티팩트를 원래 숨기고 있던 것**이었음. `extra_rx`는 정확히 그 숨겨진 영역에 다시
gain을 밀어넣어서(고리까지 sharpening을 확장하려고) B와 같은 아티팩트를 재노출시킴.
즉 `extra_rx`의 세부 구현 버그가 아니라 **"primary mask가 의도적으로 숨긴 걸 다시
드러내는" 구조적 문제** — 그래서 gain floor나 smooth-blend 같은 `extra_rx` 내부 수정
시도가 둘 다 효과 없었던 것도 설명됨.

### 적용한 수정

`extra_rx` 로직 자체는 삭제하지 않고, 새 opt-in 설정 `WaveletConfig.master_ring_
extension_enabled`(기본 **False**)로 감쌈:

- `pipeline/config.py`: `master_ring_extension_enabled: bool = False` 추가, A/B/C/D
  실험 결과와 메커니즘을 상세히 문서화.
- `pipeline/steps/wavelet_master.py`: `extra_rx`/`extra_ry`/`extra_gap_px` 계산을
  `has_rings=True AND master_ring_extension_enabled` 조건으로 게이팅. **단, primary
  ellipse 각도를 `pole_pa_deg`로 맞추는 부분(별개의, 이미 검증된 fix)은 `has_rings`만
  으로 유지** — A/B/C/D 실험이 이 각도를 모든 케이스에 동일하게 썼고 C가 깨끗했으므로
  각도 자체는 문제가 아님을 확인했기 때문.

### 실측 검증

- `pipeline/modules/wavelet.py`의 두 수정(floor, smooth-blend)은 `git checkout --`으로
  완전히 되돌림 — 리뷰어 권고대로 baseline을 깨끗하게 유지.
- 실제 `wavelet_master.run()` 프로덕션 경로로 `master_ring_extension_enabled=False`
  (기본값) vs `True`(명시적) 비교: **기본값은 깨끗함(case C와 일치), 명시적으로 켜면
  기존 버그 그대로 재현** — 플래그가 정확히 동작함을 확인.
- window_01 IR/R 둘 다, 확대(4x/6x) crop 및 전체 프레임 기준 확인 완료.
- 전체 테스트(78개) 통과. `extra_rx=None`인 기존 호출자(Jupiter 등)는 완전 무관.

### 트레이드오프

기본값(off)에서는 globe는 완전히 sharpening되지만, 고리/Cassini Division 자체는
이전처럼 확장 sharpening을 받지 않음(이 서브시스템이 존재하기 전 상태와 동일) — 링잉
아티팩트를 없애는 대신 받아들인 보수적 트레이드오프. 나중에 이 아티팩트를 재현하지
않는 다른 mask 설계를 찾으면 재활성화 가능.

## 관련 메모리

- [[project_ring_occlusion_3d_reprojection_gap]] — 오늘 커밋된 전체 수정 기록(3D 배선,
  hard-edge 버그, extra_rx 최종 해결)
- [[project_ring_limb_ringing_bug]] — 3번의 선행 실패 시도 기록 + 오늘 발견의 관계
- [[project_derotation_ring_occlusion_fix]] — 2026-08-11 원래 ring occlusion fix
- [[feedback_ab_test_via_real_pipeline]] — 오늘 모든 검증에 적용한 원칙

## 이번 세션 목표였던 "진짜 근본 원인" 조사 — 타원 피팅 자체를 고쳐봄 (2026-08-15, 이어서)

사용자가 이 세션의 목표를 "gray halo(디스크 외곽 회색 링)와 white-rim(오버슈트) 둘 다
없어지는 진짜 해결책"으로 명시적으로 재지정 — 위의 `extra_rx` 해결은 고리 쪽 재노출
경로만 막았을 뿐, `find_disk_center()`의 타원 피팅이 진짜 photometric limb과
0.5~0.9px 비대칭으로 어긋나 있다는 근본 원인 자체는 그대로였음.

### 조사 결과 요약

- **Jupiter(고리 없음)도 고리와 무관한 비슷한 이상치성 오차**를 보임 → "Saturn 전용
  고리 오염" 가설 기각, robust(MAD 기반 outlier 제거) 타원 재피팅으로 피봇.
- **Jupiter: 검증된 실제 개선.** 72개 방사형 레이의 서브픽셀 gradient 측정 +
  반복적(iteratively-reweighted) MAD 기반 outlier 제거로 재피팅한 결과, 실측
  worst-case 피팅오차가 9.04px→2.26px, 7.84px→2.49px로 대폭 감소(진단 스크립트 기준;
  프로덕션 코드 자체 측정으로는 60~62% 감소, 아래 참고).
- **Saturn: 안 풀림.** 같은 robust refit이 Saturn에서는 ~40%에 달하는 ray가 고리에
  오염되어 있고 그게 "흩어진 점"이 아니라 "연속된 각도 구간"이라 MAD 같은 point-wise
  통계로는 오염된 다수를 정상으로 오인함(오염 비율이 너무 높고 연속적). ring-ray
  사전제외/hybrid/프레임 수/quadrupole(장단축 스케일 오차) 등 추가로 시도한 4가지
  전부 효과 없거나 불충분(R 필터는 조금 개선, IR은 거의 무변화) — Saturn 원인은
  여전히 미해결.
- **사용자 최종 결정**: "Jupiter용으로 구현하고 Saturn은 문서화만."

### 구현 (프로덕션, 커밋 전 — 이 세션 마지막 작업)

- `pipeline/modules/derotation.py`: `_gradient_disk_r`의 서브픽셀 gradient 측정 로직을
  `_subpixel_ray_edge()`로 추출(byte-identical 리팩터, 78개 기존 테스트로 확인) 후,
  이를 재사용하는 신규 `_ray_limb_edge()` + `_robust_ellipse_refit()` 추가. `find_disk_
  center()`/`_find_disk_center_impl()`/`_gradient_disk_r()` 자체는 무변경(등록/추적
  경로에 영향 없음 — Phase 0 스코프 경계 그대로 준수).
- `pipeline/config.py`: `WaveletConfig.master_limb_fit_refinement_enabled`(기본
  **False**) 추가.
- `pipeline/steps/wavelet_master.py`: `find_disk_center()` 결과를 얻은 직후,
  플래그가 켜져 있고 **`has_rings=False`일 때만** `_robust_ellipse_refit()`로
  재피팅(Saturn은 검증 안 됐으므로 아예 호출 자체를 안 함 — 조건문 게이팅, 근사치
  fallback이 아니라 완전 스킵).
- 신규 테스트 7개(`tests/test_limb_fit_refinement.py`): 합성 데이터 기준 정확한 기하
  복원, 국소 오염(소수 ray) 상황에서도 정확한 복원 유지, ray 부족시 None 반환(안전한
  폴백), config 기본값 False, 그리고 **실제 `wavelet_master.run()`을 monkeypatch spy로
  감시**해 (a) 플래그 꺼짐 시 전혀 호출 안 됨, (b) `has_rings=True`면 플래그가 켜져도
  호출 안 됨, (c) `has_rings=False`+플래그 켜짐이면 실제로 호출됨을 확인. 전체
  85개(기존 78 + 신규 7) 테스트 통과.

### ⚠️ 실측 검증에서 발견한 새로운 트레이드오프 — "고쳤다"고 선언할 수 없음

`feedback_ab_test_via_real_pipeline` 원칙대로 실제 `derotate_window()` →
`wavelet_master.run()` 정식 경로로 Jupiter window_03 IR/R 검증:

1. **정량 개선은 진짜임**: 프로덕션 코드 자체로 측정해도 max|residual| 9.04px→3.42px
   (+62%), 7.84px→3.13px(+60%) — 계획서의 "≥30% 개선"이라는 노이즈-바닥 기준을 크게
   상회.
2. **Saturn은 완전히 무영향**: 실제 `wavelet_master.run()`을 flag off/on으로 두 번
   돌려서 출력 PNG가 byte-identical함을 직접 증명(`has_rings=True` 게이팅이 정확히
   작동).
3. **⚠️ 그런데 Jupiter 전체 프레임을 확대해서 보니, 이전엔 없던 얇고 밝은 오버슈트
   윤곽선이 disk limb 전체를 따라 새로 나타남.** flag off는 완전히 매끄러운 단조
   감소 프로파일(오버슈트 없음, 원래의 "회색 halo"만 있음) — flag on은 같은 위치에서
   주변 대비 뚜렷한 밝은 스파이크가 생기고, 이게 한 지점의 우연이 아니라 여러 행
   (dy=-15~+15)에 걸쳐 limb을 따라 연속적으로 나타남(진짜 아치형 아티팩트).
4. **원인 추정**: 피팅이 진짜 limb에 훨씬 더 가까워지면서, primary mask의 "경계에서
   gain=0으로 페더링"하는 지점이 이제 실제 가파른 photometric gradient 바로 위에
   놓이게 됨 — 예전엔 피팅 오차(최대 9px) 덕분에 gain=0 지점이 진짜 gradient에서
   떨어져 있어서(halo의 원인이자 동시에 오버슈트 회피 수단) 우연히 보호되고 있었던
   것. `edge_feather_factor`를 2→3→4로 넓혀보면 오버슈트 진폭이 4134→2048→1196
   (16비트 기준)으로 줄지만 완전히 없어지지 않고, eff=4에서는 halo스러운 부드러움이
   다시 스며들기 시작함 — **같은 트레이드오프 곡선 위에서 재앙커링된 것일 뿐, 곡선
   자체를 벗어난 게 아님.**
5. 시각 비교/스윕 이미지: `experiments/limb_fit_validation/` (특히
   `scratch_limb_fit_jupiter_limb_zoom.png`, `scratch_limb_fit_feather_sweep.png`).

**결론**: `_robust_ellipse_refit()`는 실제로 정확하고 검증된 피팅 개선이며(합성+실측
둘 다), Saturn에 전혀 영향 없이 안전하게 게이팅되어 있음 — 코드/테스트는 그대로
유지할 가치가 있음. 하지만 **"halo와 white-rim을 동시에 없앤다"는 이번 세션 원래
목표는 아직 달성되지 못함** — 이 피팅 수정 하나만으로는 같은 gain-vs-gradient
트레이드오프를 벗어나지 못하고, 얇아진 형태로 재현됨. 기본값 **False로 유지**하고
사용자에게 있는 그대로 보고. 후속으로 고려할 수 있는 방향(미착수): (a) gain=0 지점을
피팅된 경계보다 살짝 바깥에 의도적으로 앵커링(피팅은 정확하게 하되 마스크는 여전히
약간의 여유를 둠), (b) 이 오버슈트 자체를 억제하는 국소 게인 제한 로직(3번 실패
이력 있음, `project_ring_limb_ringing_bug` 참고), (c) 현 수준(halo 유지) 수용.

## ⚠️→❌ 후속 시도 5~9: "b) 오버슈트 자체를 억제" 방향을 5가지 방식으로 시도 — 전부 실패, 정직하게 기록

사용자가 "이런 흰색 윤곽선은 퀄리티에 치명적이라서 절대로 나오면 안돼. 지금 토성에서
고리가 끊어져보이는 건 하나는 헤일로고 하나는 이런 윤곽선"이라고 명확히 못박음 —
white-rim은 halo와 대등한 "트레이드오프"가 아니라 그 자체로 출시 불가급 결함이라는
원칙 확정 ([[feedback_white_rim_is_critical_defect]] 메모리 참고). 이에 따라 위
(b) 방향(오버슈트 자체를 억제하는 로직)을 본격적으로 시도.

### 시도 5: 출력값 클램핑 (local min/max clamp) — 합성 테스트에서도, 실측 하이브리드
### 테스트에서도 실패

아이디어: 샤프닝 후 각 픽셀을 "원본(샤프닝 전) 이미지의 국소 min~max 범위" 밖으로
못 나가게 자르는 것. `pipeline/modules/wavelet.py`에 `_local_min_max()`(cv2.erode/
dilate 기반 국소 min/max) + `overshoot_clamp_radius_px` 파라미터를
`sharpen()`/`sharpen_disk_aware()`/`sharpen_color_disk_aware()`/`sharpen_color()`
4개 함수 전부에 배선(기본값 0.0=완전 무효과). `WaveletConfig.master_overshoot_
clamp_radius_px`(기본 0.0) 추가.

**실패 발견 1(합성 테스트)**: 하드 엣지에서는 오버슈트를 완벽히 0으로 제거하지만,
디스크 안쪽의 부드러운 벨트 모양 텍스처(진짜 디테일)에도 반경 1px만 줘도 샤프닝
효과의 92~98%가 사라짐. 원인: 이 프로젝트의 실제 게인 테이블
(`_MAX_GAINS=[29.15, 9.48, 0, 0, 0, 0]`)이 가장 미세한 2개 레벨에만 집중돼있는데,
"국소 대비를 높인다"는 샤프닝의 정의 자체가 "그 픽셀을 바로 옆 이웃보다 튀게
만든다"는 뜻이라, 그 스케일과 겹치는 작은 반경의 클램프는 진짜 디테일 강조까지
거의 다 눌러버림. 최초 테스트는 손으로 재구현한 gain 테이블을 써서 부정확했으나(아래
"교훈" 참고), 실제 `sharpen()`으로 재검증해도 결론은 동일.

**실패 발견 2(실측 하이브리드 테스트, 병렬 조사 워크플로우)**: bilateral 필터(아래
시도 6)와 결합해서 "관대한 sigma_color + 작은 클램프로 뒷정리"를 시도했으나, 클램프
반경 1px만 있어도 sigma_color 값(0.08~0.40 전부 시도)과 무관하게 boost가 3~6%로
붕괴 — 클램프가 항상 지배적이라 아무것도 회복시키지 못함.

**상태**: 코드/테스트(`tests/test_overshoot_clamp.py`)는 유지(기본 off, 안전) —
`master_ring_extension_enabled`처럼 "구현·테스트됐지만 비권장" 패턴. config
docstring에 "!! DO NOT ENABLE !!" 명시.

### 시도 6~9: 웹 리서치 기반 대안 — bilateral(기존 옵션), guided filter, 하이브리드,
### local-gradient-gating

사용자 지시("웹에서 좋은 방법 찾아봐")로 리서치 에이전트를 띄워 halo/ringing 없는
샤프닝의 표준 기법을 조사(Guided Filter, Local Laplacian Filters, Edge-Avoiding
Wavelets, WLS 등 확인 — 전부 "선형 다중스케일 분해가 edge를 가로질러 섞이는 게
원인, edge-aware 분해로 바꾸는 게 정답"이라는 진단에 동의하는 문헌).

**시도 6 (bilateral, 실측 검증)**: `decompose()`에 이미 있던(이번 문제엔 한 번도
적용 안 됐던) `filter_type='bilateral'` 옵션을 실제 `derotate_window()` →
`wavelet_master.run()` 정식 경로로 검증(병렬 조사 워크플로우, 실제 profile 설정
사용, 스크립트: `experiments/ringing_fix_validation/`):
- **Jupiter window_03/IR (오늘 발견한 흰 윤곽선)**: 오버슈트 진폭 42~45% 감소 —
  **완전히 없어지지 않음**, 확대하면 여전히 얇은 밝은 선이 보임.
- **Saturn window_01/R (원래 있던 비대칭 링잉, `project_ring_limb_ringing_bug`)**:
  오른쪽 ansa 오버슈트 79% 감소 — 두 케이스 중 훨씬 큰 개선.
- **그러나 두 케이스 모두 디스크 내부의 진짜 선명도(Laplacian variance)가 78%
  감소** — 벨트 무늬/Cassini Division이 육안으로도 뚜렷하게 뭉개짐. 이 프로젝트의
  존재 목적 자체가 행성 표면/고리 디테일 보존이라, 이 트레이드오프는 **받아들일 수
  없음** — bilateral을 기본값으로 바꾸는 건 기각.

**시도 7 (guided filter, He/Sun/Tang)**: 직접 구현(`cv2.boxFilter` 기반, scipy
불필요) 후 `sharpen()`에 monkeypatch로 대입해 여러 eps 조합 테스트 — **bilateral의
(오버슈트=0, 디테일유지=26.4%) 지점을 두 축 모두에서 이기는 조합을 찾지 못함**.
2차 평활화 단계(계수 a,b 자체를 다시 boxFilter)를 제거하는 ablation도 시도했으나
동일하게 실패(민감도를 낮추면 오버슈트는 줄지만 유지율이 bilateral보다 더 나쁨).

**시도 8 (bilateral + 클램프 하이브리드)**: 시도 5 참고, 실패.

**시도 9 (local-gradient/local-range 기반 detail gating, Local Laplacian
아이디어의 단순화 버전)**: 레벨별 detail 계수를 국소 gradient 크기로 게이팅(gradient
큰 곳=edge로 간주해 게인 억제) — **모든 threshold 설정에서 오버슈트가 정확히 0.30000
(gaussian과 동일)에 고정됨**, 개선 전혀 없음. 원인 규명: 이 프로젝트의 calibrated
게인이 워낙 커서(29.15, 9.48), edge 픽셀에 남은 아주 작은 detail 누출(~0.13,
gain 적용 전)만으로도 [0,1] clip 상한을 넘겨버림 — bilateral/guided filter처럼
**커널 자체가 edge를 가로질러 섞이지 않게(구조적으로 zero cross-edge diffusion)**
만드는 방식과 달리, 이미 계산된(오염된) 계수를 사후에 감쇠시키는 방식은 이 정도로
과격한 게인 앞에서는 통하지 않음.

### ✅ 유일하게 실측으로 남은 순(純) 개선: bilateral의 sigma_color 레벨별 튜닝

시도 6~9 전부 "이 문제를 해결"하진 못했지만, 그 과정에서 **부작용 없는 작은
개선**을 하나 발견: `_bilateral_smooth`의 `sigma_color`가 기존엔 모든 레벨에 고정
0.08이었는데, 레벨별로 다르게(미세 레벨엔 낮게, 거친 레벨엔 높게) 주는 grid search를
돌린 결과 **같은 오버슈트=0(정확히 동일)을 유지하면서 디테일 유지율이 26.4%→28.7%로
개선**되는 지점(`sigma_fine=0.10, sigma_coarse=0.12`)을 발견 — `filter_type=
'gaussian'`(기본값, 모든 기존 호출자)은 전혀 영향 없고, `'bilateral'`을 실제로
선택하는 경우에만 순수하게 더 나은 값. 이 하나는 실제로 적용함
(`pipeline/modules/wavelet.py`의 `_bilateral_smooth` 기본값 변경, 78개 테스트
전부 통과 확인).

### 최종 결론 (2026-08-15, 이 세션 전체에 대해)

**halo와 white-rim을 동시에 없앤다는 원래 목표는 이번 세션에서 달성하지 못함.**
Saturn/Jupiter 양쪽에서, 정확한 fitting(로버스트 리핏)이나 edge-aware 분해(bilateral/
guided filter)나 사후 클램핑이나 gradient 기반 게이팅이나 — 시도한 9가지 방향
전부 "오버슈트를 없애려면 진짜 디테일을 심각하게 희생해야 한다"는 같은 벽에
부딪힘. 유일하게 실측으로 검증된 순수 이득은 `_bilateral_smooth`의 sigma_color
재튜닝(작지만 부작용 없음, 이미 적용). 근본 원인은 이 프로젝트의 wavelet 게인
calibration이 최고 두 레벨에 매우 강하게 집중돼 있어서(29.15, 9.48), "edge에서는
게인을 죽이고 real detail에서는 살린다"는 구분이 근본적으로 어려운 것으로 보임 —
다음에 이 문제를 다시 본다면 게인 calibration 자체를 재검토하거나(레벨당 게인을
낮추고 레벨 수를 늘리는 등), Edge-Avoiding Wavelets(Fattal 2009, Photoshop
"Protect Detail"의 실제 알고리즘 — wavelet 변환 자체를 content-adaptive하게
만드는 더 근본적이지만 구현 비용이 큰 방법)를 검토할 가치가 있음. 지금은 halo만
있고 white-rim은 없는 현재 상태(모든 신규 플래그 기본 off) 유지가 유일하게
안전한 선택.

**교훈**: (1) 합성 테스트를 만들 때 실제 코드의 calibration 상수(`_MAX_GAINS` 등)를
반드시 실제로 import해서 쓸 것 — 손으로 근사한 값은 결론을 왜곡시킴(이번에 실제로
발생, 병렬 워크플로우가 재검증해서 잡아냄). (2) "문헌에서 표준으로 통하는 기법"도
이 코드베이스의 특정 calibration(극단적으로 미세 레벨 집중)에서는 기대만큼 안 통할
수 있음 — 항상 실측으로 확인. (3) 9번의 실패에도 불구하고 매번 정직하게 "안 됐다"고
기록한 덕에 최소한 다음 세션이 같은 9가지를 또 시도하지 않을 수 있음.

---

## 후속 (같은 날 저녁, cc69371/0d1abf7): `extra_rx` → true annulus mask 교체

외부 리뷰가 `extra_rx`의 실체는 "ring annulus"가 아니라 "채워진 outer ellipse"라고
재지적(globe 진짜 경계 ~ ring 진짜 inner edge 사이의 넓은 gap이 ring이 아니라 globe
자신의 PSF limb tail/darkening tail인데, 여기에도 ring 수준 gain이 걸려 white-rim/
dark-trough를 만든다는 논리). 코드 확인 결과 지적이 정확함 —
`_make_disk_weight_ellipse`는 이름과 달리 내부 전체를 채우는 함수였음.

**구현**: `compute_ring_sharpening_mask()`(`derotation.py`) 신규 추가 — 진짜
annulus(`in_ring_outer & ~in_ring_inner`), globe와 겹치는 부분만 `depth_ring>0`
(전경) 게이팅, 그 외 영역은 전체 커버. `wavelet.sharpen_disk_aware()`에
`extra_weight_map`(사전 계산된 (H,W) 배열, 매 레벨 동일하게 max-결합) 파라미터
추가해 배선 — `extra_rx` 자체는 다른 호출자 호환을 위해 유지, Saturn은
`extra_weight_map` 경로로 전환.

**실측(rigorous, same-stretch diff heatmap)**: 구조적으로는 맞는 수정(ring-globe
접합부에서 이음매 없음, heatmap상 고리 선이 더 잘 이어짐)이지만, **이 window의
지배적인 링잉 아티팩트에 대한 실질 영향은 제한적** — old vs new 전체 |diff| 1029
vs 1041, disk 밖 |diff| 622.8 vs 621.6, max |diff| 소수점까지 동일. 처음에
"개선됐다"고 성급히 판단했다가(percentile 개별 스트레치 비교 탓) 사용자 지적으로
철회·재검증함. 결론: 이 fix는 정확하지만, 아래에서 재확인되듯 **주요 아티팩트의
진짜 원인은 이 mask가 아니라 원래 있던 ellipse-fit 비대칭 링잉**이었음. 기본값
`master_ring_extension_enabled=False` 유지(annulus 방식이 구현됐지만 기본 on으로
전환할 근거는 아직 없음).

---

## 2026-08-16: 독립 진단으로 재확인 — "이건 새 globe-limb 버그가 아니라 이미 문서화된 ansa fitting 버그다"

외부 리뷰가 "radial 1D monotonicity(dip-then-rise) 위반을 극지방 sector부터
검증해보자"고 제안. 실제 frozen window_01/R 스택(현재 프로덕션 기본값, 모든 신규
플래그 off)에서 극지방(ring-free, ±70~95°)과 ansa 인접(ring-adjacent, -15~15°/
165~195°) 두 그룹의 radial 프로파일을 비교:

```
극지방(ring-free):      max|sharpened-raw| ≈ 0.0000 (사실상 전무)
ansa 인접(ring-adjacent): max|sharpened-raw| ≈ 0.03~0.09 (뚜렷)
```

sign-reversal(딥-후-회복) 자체는 raw/sharpened 어느 쪽에서도 검출 안 됨 — 즉
"wavelet이 새 극값을 만들었다"는 좁은 의미의 증거는 아니고, "ansa 방향에서만
sharpening 변화량이 집중된다"는 공간적 증거임.

**이 결과는 이 문서 이전 절인 `project_ring_limb_ringing_bug`(2026-08-13 최초
발견)의 결론과 정확히 일치함**: ring 오염이 `find_disk_center()`의 타원 피팅을
ansa 방향(우측 +0.86px, 좌측 -0.53px)으로만 비대칭하게 끌어당기고, 극지방은 ring
오염이 없어 피팅이 정확함. 즉 오늘 아침의 "새 발견"은 **9가지 시도 전부 실패로
끝난, 어제 이미 완전히 조사된 문제의 독립적 재확인**이었음 — 새로운 globe-limb
버그가 아님.

**결론 및 정책**:
- 이 아티팩트는 generic de-rotation 버그도, generic wavelet 버그도 아니고,
  **ring-contaminated ellipse fit(ansa 방향)** × **최상위 2개 레벨에 극단적으로
  집중된 wavelet gain**의 결합 결과로 재확정.
- gain mask / smooth blend / clamp / bilateral / guided filter / gradient
  gating 계열은 **9종 모두 실패**로 이미 소진됨 — 같은 계열의 10번째 변형(오늘
  제안됐던 polar-sector monotonic limiter, coefficient-attribution 재분해,
  ring-adjacent ablation 포함)은 반복 조사가 될 가능성이 커서 보류.
- 이 실패 계열을 다시 열 가치가 있으려면 **같은 이미지의 mask/gain 변형이 아니라
  질적으로 다른 정보원**이 필요함 — 예: 외부 ephemeris 기반 독립 navigation, ring
  오염이 적은 다른 필터/프레임의 globe limb 참조, 또는 `project_saturn_ring_globe_
  separation`/`project_map_space_lucky_imaging_idea`에 기록된 globe/ring 물리적
  레이어 분리(둘 다 별도 프로젝트 규모, 미착수).
- 현재 안전한 프로덕션 상태(모든 신규 플래그 기본 off: `master_ring_extension_
  enabled`, `master_limb_fit_refinement_enabled`, `master_overshoot_clamp_
  radius_px=0.0`, `master_coverage_aware_sharpening` 등)를 "알려진 한계"로
  유지 — globe interior/ring annulus는 강한 wavelet을 유지하되, ansa 근방의
  옅은 halo는 감수하고, 이를 더 강한 white-rim/dark-trough/이중 limb로 바꾸는
  거래는 하지 않음([[feedback_white_rim_is_critical_defect]]).

---

## 2026-08-16 (새 세션): "질적으로 다른 정보원" 시도 — navigation/ephemeris 기반
## 타원 피팅 재설계. 구현·테스트 완료, 실측에서 R 채널에 새 결함 발견 → 기본 off 유지

위 2026-08-16 절의 정책("재개 조건: mask/gain 변형이 아니라 질적으로 다른 정보원 —
외부 ephemeris 기반 독립 navigation, ring 오염 적은 필터/프레임의 limb 참조, 또는
globe/ring 물리적 레이어 분리")에 따라, 사용자가 이번 세션에서 첫 번째 옵션(독립
navigation/ephemeris 기반 재설계)을 지정. 코드베이스 서베이(3-agent 병렬 조사)로
핵심을 발견: 이 프로젝트는 이미 (a) `pole_pa_deg`를 `auto_detect_equator_pa()`(벨트
그래디언트 기반, 타원 피팅과 무관)로 독립 측정하고, (b) Horizons에서 sub-observer
latitude B를 조회하며, (c) 행성의 진짜 물리적 편평률(`true_polar_equatorial_ratio`,
Saturn=0.9021)을 이미 알고 있음 — 이 세 스칼라를 표준 oblate-spheroid 겉보기 형상
공식(apparent_ratio = sqrt(true_ratio²·cos²B + sin²B), 회전체의 대칭축 성질로 유도,
`_oblate_ortho_forward`의 실측 수치 envelope와 오차<1e-3으로 교차검증 완료)에
대입하면 이미지를 전혀 보지 않고 겉보기 종횡비를 정확히 예측할 수 있음.

### 설계: 방향+종횡비 고정 → (cx,cy,scale) 3-param 피팅

기존 실패한 "ring-ray 제외 후 자유 5-param 재피팅"(`experiments/scratch_globe_fit_
asymmetry_diagnosis.py`)의 문제는, 제외된 ~40% 연속 호(ansa 부근)가 하필 장축
스케일을 결정하는 데 가장 중요한 정보를 담고 있어서, 나머지 ~60%만으로 방향+종횡비
+중심+스케일을 전부 자유롭게 재피팅하는 게 조건이 나쁘다는 것. 새 설계는 방향
(pole_pa_deg, 독립측정)과 종횡비(위 공식으로 예측, 이미지 무관)를 미리 고정하고
고리 오염 섹터를 제외한 나머지 ray로 (cx, cy, scale) 3개만 피팅 — 훨씬 과결정된
문제.

**구현** (`pipeline/modules/derotation.py`): `_predicted_apparent_ratio()`(닫힌 공식,
회전 행렬 없음), `_ring_contaminated_theta_mask()`(기존 `_ring_globe_overlap_
ellipses`/`compute_ring_sharpening_mask`와 동일한 고리 멤버십 판정을 seed 타원의
경계점에 적용), `_fixed_shape_circle_fit()`(고정 각도/비율로 좌표 회전+리스케일 후
Kåsa 원 피팅), `_navigation_constrained_ellipse_fit()`(위 3개 조합 + MAD 이상치 제거
1회, `_robust_ellipse_refit`과 동일한 "seed보다 나쁘게 만들지 않는다" 계약).
`WaveletConfig.master_navigation_limb_fit_enabled`(기본 **False**) 신설,
`has_rings=True`일 때만 적용(has_rings=False 전용인 기존 `master_limb_fit_
refinement_enabled`과 상호 배타적). 신규 테스트 16개(합성 기하 복원,
`_oblate_ortho_forward` 실측 envelope 교차검증, 게이팅 3케이스) 전부 통과, 기존
106개 전체 테스트 회귀 없음.

### 실측 검증 (실제 derotate_window()→wavelet_master.run() 정식 경로, window_01
### IR/R, 실제 Horizons B=-11.07°/pole_pa=-7~4°)

**진단적으로 흥미로운 확인**: seed(`find_disk_center`)의 종횡비(IR: 0.844, R: 0.879)가
예측된 겉보기 비율(0.906)보다 눈에 띄게 더 찌그러져 있고, 링과 무관한 각도(35°~135°
등)에서 seed 대비 실측 limb이 일관되게 +3~+5px 바깥에 있음(반면 장축 근처에서는
거의 0) — 즉 seed 자체가 이미 고리 오염으로 종횡비가 눌려있다는 정황이 실측으로도
확인됨, 이 방향의 진단 자체는 타당했음.

**그러나 이 피팅을 실제로 sharpening mask 경계로 사용한 결과**:
- `_navigation_constrained_ellipse_fit`는 seed보다 유의미하게 큰 반지름을 산출
  (IR: rx 66.2→67.0, ry 55.8→60.7 / R: rx 68.2→71.4, ry 60.9→64.7).
- 정량 비교(극지방 vs ansa 인접 sharpening delta, 2026-08-16 앞 절과 동일 방법론):
  개선/악화가 혼재하고 뚜렷한 개선 없음 — IR polar mean +10%(악화), ansa-right mean
  +1%, ansa-left mean +18%(악화); R polar mean +38%(악화), ansa-right mean +19%
  (악화), ansa-left만 -7%(소폭 개선). 전반적으로 "개선"이라 부를 근거 부족.
- **육안 확인(6x/10x crop)에서 R 채널 우측 ansa에 새로운 결함 발견**: OFF에는 없던
  뚜렷한 어두운 톱니 모양 세로선이 globe-ring 접합부에 ON에서 새로 나타남
  (`experiments/navigation_limb_fit_validation/R_right_ansa_TIGHT2_10x.png`).
  IR 채널은 육안상 OFF/ON 거의 구분 안 됨(뚜렷한 신규 결함 없음). 원인 미확정이나,
  반지름이 커지면서 disk mask feather 경계가 이전과 다른 지점(globe true edge와
  ring true inner edge 사이 gap, `compute_ring_sharpening_mask` 문서가 지적한
  "고리 아님, globe PSF tail" 영역)과 새로 겹치며 seam이 생긴 것으로 추정 — 미검증.

**결론**: 이 세션의 진단(seed 종횡비가 고리 오염으로 눌려있다는 것)은 실측으로 뒷받침
되지만, 그 진단을 교정한 결과를 곧바로 sharpening mask에 쓰는 건 R 채널에서 새로운
가시적 결함(어두운 선)을 만듦 — feedback_white_rim_is_critical_defect의 원칙
("어떤 형태든 새 가시적 윤곽선/선은 트레이드오프가 아니라 결함")을 white-rim이
아니라 dark-line에도 동일하게 적용해 기본값 False 유지, 사용 비권장으로 판정.

### 후속 확인: "고리에도 wavelet을 걸면 없어지지 않나?" — 아니었음

사용자 가설(ring extension도 같이 켜면 새 dark-line이 없어질 것)을 실측으로 확인:
`master_navigation_limb_fit_enabled` × `master_ring_extension_enabled` 2×2 조합
전부 실제 파이프라인으로 렌더링(스크립트 하단 참고). 결과:
- **A(둘 다 off, baseline)**: 결함 없음.
- **B(nav만 on)**: 앞서 발견한 어두운 노치.
- **C(ring extension만 on, nav off)**: B와는 **다른** 결함 — 우측 하단에 고리 재질이
  갑자기 샤프닝 gain을 받기 시작하는 밝은 대각선 띠 + 접합부 crease.
- **D(둘 다 on)**: B의 어두운 노치가 **그대로 남고**, 거기에 C의 밝은 띠까지 겹쳐
  나타남 — 두 아티팩트가 상쇄되지 않고 공존.

즉 "고리도 sharpening 받아야 한다"는 방향으로는 이 특정 노치가 해결되지 않음 —
원인은 고리 자체의 gain 유무가 아니라 **확대된 disk 반지름 자체가 primary disk
mask의 feather 경계 위치를 진짜 limb 구조(Cassini Division 인근)를 가로지르는
곳으로 옮겨놓는 것**으로 추정(미검증). 1x 전체 프레임에서는 A/B/C/D 넷 다 육안상
거의 구분 안 됨(`experiments/navigation_limb_fit_validation/combo_{A_baseline,
B_nav_only,C_ring_only,D_both}_FULL_1x.png`) — 즉 이 아티팩트들은 서브픽셀~수픽셀
규모이나, 이 프로젝트 기준(새 윤곽선은 미세해도 결함)으로는 여전히 채택 불가.

이 문서에 기록된 9번의 mask/gain 계열 실패, 그리고 "질적으로 다른 정보원"이라는
새로운 시도 축(및 그 축 위에서의 ring-extension 조합까지) 포함해 **총 10번째
시도도 halo/white-rim/ring-cut 착시를 동시에 해결하는 데 실패**. 코드/테스트는
정확하고 안전(has_rings=False에 완전 무영향, 회귀 없음)하므로 유지할 가치는
있음 — 사용자가 이 트랙을 여기서 종료하고 다른 접근(레이어 분리 등, 아래 참고)으로
전환하기로 결정.

**검증 스크립트**: `experiments/scratch_navigation_limb_fit_validation.py`
(byte-identical 게이팅 확인, 정량 섹터 비교, 시각 crop/heatmap 생성 전부 포함).
2×2 콤보 실험은 별도 인라인 스크립트로 실행(파일로 저장 안 됨) — 산출물은
`experiments/navigation_limb_fit_validation/combo_*.png`(각 조합별 `_FULL_1x.png`
전체 프레임 + `_crop14x.png` 확대본).
