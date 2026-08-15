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
