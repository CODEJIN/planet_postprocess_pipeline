## 결론

**WinJUPOS를 더 깊게 참고하는 방향은 맞습니다.**  
다만 지금 필요한 것은 `warp_scale`, feather 폭, quality weight 같은 **튜닝을 더 하는 것**이 아니라, WinJUPOS식의 핵심 전제인 **“정확한 navigation 모델을 먼저 만들고, 물리적으로 서로 다른 표면을 분리한다”**는 구조를 반영하는 것입니다.

지금 코드에서 성과가 잘 안 난 이유도 상당 부분 설명됩니다. 현재의 `ring_crossing_mask`는 좋은 문제 인식에서 출발했지만, **한 장의 mask로는 해결할 수 없는 가림(occlusion) 문제를 처리하려 하기 때문**입니다.

---

## 현재 구현에서 가장 큰 물리적 문제

### 1. `ring_crossing_mask`가 **앞고리와 뒷고리**를 구분하지 않습니다

현재 마스크는 사실상 다음 영역을 만듭니다.

```text
globe ellipse ∩ projected ring annulus
```

그리고 해당 영역 전체에서 대기 de-rotation drift를 0으로 feather 합니다.

하지만 이 교집합에는 서로 완전히 다른 두 경우가 들어 있습니다.

| 영역 | 실제로 보이는 것 | 올바른 처리 |
|---|---|---|
| **전경 고리**가 globe 앞을 지나는 부분 | 고리 | globe de-rotation을 적용하면 안 됨 |
| **배경 고리**가 globe 뒤에 가려지는 부분 | 대기/글로브 | **반드시 globe de-rotation을 적용해야 함** |

즉, 현재 로직은 배경 고리가 원래부터 안 보이는 영역에서도 대기 rotation을 끕니다.

```python
in_globe & in_ring_annulus
```

이 조건은 **“고리의 투영 footprint가 글로브와 겹치는가”**만 판단합니다.  
하지만 필요한 것은:

```text
고리가 globe보다 카메라에 가까운가?
```

입니다.

따라서 현재 결과는 다음이 동시에 발생할 수 있습니다.

- 전경 고리가 지나가는 부분은 덜 망가질 수 있음
- 그러나 **뒷고리에 가려진 대기 영역까지 de-rotation이 중단됨**
- 매 프레임마다 `dt_sec`가 다르므로, 경계 주변의 미세한 대기 디테일이 서로 다른 위치에 남음
- feather는 hard seam을 줄일 뿐, **잘못 멈춘 대기 이동 자체를 복구하지 못함**
- 결과적으로 atmospheric detail은 약해지고 ring crossing 근방은 흐려짐

이것은 “효과가 거의 없다”는 체감과 정확히 맞는 구조적 한계입니다.

---

## 2. Feathered zero-drift는 seam을 부드럽게 만들지만, 실제로는 혼합 blur입니다

현재 핵심 처리:

```python
depth_map = depth_map * _ring_weight
```

은 ring 경계에서 대기 drift를 연속적으로 줄입니다.

그렇지만 이 경계에서는 결국 아래가 섞입니다.

```text
정상 de-rotated atmosphere
      ↓ feather zone
부분적으로 de-rotated atmosphere
      ↓
원래 위치에 남은 ring / raw atmospheric content
```

이것은 수치적으로 연속적일 뿐, 광학적으로는 **서로 다른 표면의 움직임을 섞는 것**입니다.

특히 Saturn처럼:

- 대기는 System III 기준으로 이동하고,
- 고리는 Keplerian 운동을 하지만 짧은 구간에서는 거의 image-stationary처럼 보이며,
- foreground ring은 globe 위에 얹혀 있고,
- background ring은 globe에 가려지고,
- ring shadow도 따로 존재하는

대상에서는, 단일 warp 내부에서 drift를 제어하는 방식이 본질적으로 불리합니다.

---

## 3. 현재 코드는 이미 “dual-stack architecture” 직전까지 왔습니다

지금 구현에는 이미 다음 요소가 있습니다.

- globe pose 및 radius
- oblate globe shape
- `sub_observer_lat_deg`
- analytic ring annulus
- filter 간 shape/pose resolver
- raw-frame pre-warp alignment
- atmosphere-only de-rotation
- ring overlap 판단의 필요성

따라서 다음 단계는 새 튜닝 파라미터를 추가하는 것이 아니라, **합성 구조를 바꾸는 것**입니다.

---

# WinJUPOS에서 실제로 참고할 핵심

공개 문서만으로 WinJUPOS의 내부 소스 수준 알고리즘을 완전히 검증할 수는 없지만, 실제 workflow에서 분명한 점은 다음입니다.

## WinJUPOS의 진짜 강점은 “warp”보다 navigation입니다

WinJUPOS는 단순히 이미지에 수평 이동을 적용하는 도구가 아닙니다.

1. 촬영 시각과 관측 위치를 바탕으로 ephemeris를 계산
2. 행성의 apparent geometry를 계산
   - 중심
   - 크기
   - pole orientation
   - \(D_e\): 지구에서 본 위도
   - \(D_s\): 태양에서 본 위도
   - CM / System III longitude
3. 측정된 이미지에 **3D wireframe**을 정확히 맞춤
4. globe surface의 픽셀을 body-fixed 좌표로 역투영
5. 공통 기준시각으로 longitude를 이동
6. 재투영 및 합성

Saturn에서 사용자가 wireframe을 직접 맞추는 이유도 바로 여기에 있습니다.

> Saturn은 ring edge가 globe limb, atmospheric belt, Cassini Division, seeing blur와 섞이므로, 자동 threshold/ellipse만으로는 navigation을 신뢰하기 어렵다.

현재 pipeline은 이 단계에서 자동화를 많이 달성했지만, Saturn에 대해서는 **“측정 결과의 물리적 일관성 검증”**이 아직 부족합니다.

---

# 우선 고쳐야 할 것: 3-layer geometry

토성은 다음 레이어로 분리해야 합니다.

```text
1. Background ring
2. Globe atmosphere
3. Foreground ring
```

합성 순서는 반드시:

```text
background ring → de-rotated globe → foreground ring
```

이어야 합니다.

## 필요한 기하 마스크

각 pixel에서 최소한 다음 mask가 필요합니다.

```python
globe_mask
ring_annulus_mask
foreground_ring_mask
background_ring_mask
```

관계는 다음입니다.

```text
foreground_ring_mask = ring_annulus ∩ globe ∩ (ring depth < visible globe depth)
background_ring_mask = ring_annulus ∩ globe ∩ (ring depth >= visible globe depth)
```

여기서 camera 방향으로의 depth 비교가 핵심입니다.

| 경우 | 화면상 위치 | 최종 표시 내용 |
|---|---|---|
| ring만 존재 | globe 밖 | ring |
| background ring + globe 겹침 | globe 내부 | globe atmosphere |
| foreground ring + globe 겹침 | globe 내부 | ring |
| globe만 존재 | globe 내부 | de-rotated globe |

현재 `compute_ring_crossing_mask()`는 위 표에서 foreground/background를 합쳐 버리고 있습니다.

---

## 개념적 의사코드

```python
# 1. 원본을 globe와 ring 후보 레이어로 분리
raw_globe = image.copy()
raw_ring = image.copy()

# 2. globe 영역만 atmosphere 모델로 de-rotate
derot_globe = spherical_derotation_warp(
    raw_globe,
    dt_sec=dt_sec,
    ...,
)

# 3. 고리는 de-rotate하지 않는다.
#    단, raw-frame registration으로 중심 흔들림은 이미 보정한다.
registered_ring = registered_raw_image

# 4. 각 프레임에서 visibility order에 따라 합성
frame_model = registered_ring.copy()

# 배경 고리가 globe 앞에 표시되지 않도록 globe로 덮는다.
frame_model[globe_mask] = derot_globe[globe_mask]

# 전경 고리만 globe 위에 다시 얹는다.
frame_model[foreground_ring_mask] = registered_ring[foreground_ring_mask]
```

실제 구현에서는 hard assignment 대신 alpha/PSF-aware feather가 필요하지만, **feather는 최종 합성 경계에서만** 써야 합니다. 대기 warp의 drift 자체를 줄이는 용도로 쓰면 안 됩니다.

---

# 권장 구현 순서

## Phase 0 — 지금 코드에서 즉시 수정할 항목

### A. System III 회전 주기 수정

현재 주석과 설정의:

```text
10.56 h
```

는 Saturn System III가 아닙니다.

통상 사용하는 Saturn System III 기준값은:

```text
10 h 39 m 22.4 s
= 10.6562 h
```

입니다.

| 값 | 시간 | 판단 |
|---:|---:|---|
| `10.56 h` | 10 h 33 m 36 s | System III와 다름 |
| `10.6562 h` | 10 h 39 m 22.4 s | 일반적인 System III 기준 |

짧은 window에서는 큰 차이가 아닐 수 있지만, 이미 Saturn에서 수 px급 품질을 다루는 상황이라면 **틀린 기준 주기를 유지할 이유가 없습니다.**

---

### B. `ring_crossing_mask`는 잠시 비활성화하는 편이 낫습니다

현재 방식은 개선이 확인되지 않았다면, 다음 조건에서는 사용하지 않는 편이 안전합니다.

```python
ring_crossing_mask=None
```

또는 설정으로 명시적으로 끄십시오.

```python
has_rings=False
```

이유는 단순합니다.

- 현재 방식은 foreground ring 보호 효과가 있을 수 있음
- 하지만 background-ring-occluded atmosphere도 같이 멈춤
- 따라서 globe detail에 손해를 줄 가능성이 높음
- “문제가 있는 legacy 동작”보다 “물리적으로 불완전한 새 보정”이 더 나쁠 수 있음

이건 기능을 버리는 것이 아니라, **정확한 visibility-aware layer composite가 구현될 때까지 실험 기능으로 격리하는 것**입니다.

---

## Phase 1 — analytic navigation 검증 도구를 먼저 만들기

새 warp를 만들기 전에 각 reference frame에 아래 geometry overlay를 저장하거나 표시해야 합니다.

```text
- fitted globe limb
- predicted outer A-ring edge
- predicted inner C-ring edge
- foreground ring segment
- background ring segment
- Saturn limb
- ring-plane / globe intersection line
```

이 검증 없이는 이후 결과가 나빠도 원인을 구분할 수 없습니다.

| 이상 현상 | 가능한 원인 |
|---|---|
| ring ellipse가 실제 ring edge와 다름 | radius / center / PA / B / plate scale 오류 |
| globe limb가 어긋남 | disk fit 또는 apparent oblateness 오류 |
| foreground 측이 반대 | depth sign 또는 pole handedness 오류 |
| ring shadow 위치가 반대 | Sun geometry 또는 longitude convention 오류 |
| CH4만 크게 어긋남 | sibling pose transfer 또는 chromatic registration 오류 |

현재 코드는 geometry log는 잘 남기지만, **navigation의 시각적 residual을 정량화하는 단계가 빠져 있습니다.**

---

## Phase 2 — foreground/background ring mask를 계산

`compute_ring_crossing_mask()`를 다음처럼 대체하는 방향이 맞습니다.

```python
@dataclass
class SaturnLayerMasks:
    globe: np.ndarray
    ring_annulus: np.ndarray
    foreground_ring_on_globe: np.ndarray
    background_ring_behind_globe: np.ndarray
```

핵심은 각 pixel에서:

1. 해당 pixel이 projected ring annulus에 속하는지 확인
2. globe visible surface의 line-of-sight depth 계산
3. ring plane point의 line-of-sight depth 계산
4. depth가 더 가까운 레이어를 선택

입니다.

단, ring annulus의 같은 pixel에 대해 3D ring plane상의 point는 보통 하나로 정해지므로, analytic solution으로 충분합니다.

---

## Phase 3 — atmosphere stack과 ring stack 분리

### Globe stack

- globe mask 내부
- foreground ring 영역은 제외
- background ring은 이미 globe에 가려지므로 제외할 필요 없음
- `spherical_derotation_warp_3d()` 사용 가능
- Saturn에서는 linear warp보다 3D reprojection이 더 타당함
- 다만 scale은 **`1.0`**으로 고정해야 함

### Ring stack

- raw-frame translation alignment만 적용
- atmosphere de-rotation 금지
- ring material의 Keplerian motion은 단시간 window에서 보통 작은 2차 효과
- Cassini Division과 ansae 보존을 목표로 함

### 최종 composite

```text
background ring stack
→ derotated globe stack
→ foreground ring stack
```

이 구조가 되어야 foreground ring을 지키면서도, 그 뒤에 가려진 대기까지 정상적으로 de-rotate할 수 있습니다.

---

# 3D reprojection과 Saturn에 대한 판단

현재 `spherical_derotation_warp_3d()` 자체는 상당히 잘 설계되어 있습니다. 특히:

- \(B\)를 명시적으로 반영
- oblate spheroid 사용
- far-side identity fallback
- pole axis flip 탐색
- B≈0 수치 안정성 분기
- linear warp와 방향 일치 테스트

등은 좋습니다.

하지만 Saturn에 바로 켜는 것은 아직 권하지 않습니다.

```python
use_true_reprojection=True
has_rings=True
```

조합에서는 현재:

- 3D globe warp에는 ring visibility handling이 없음
- linear path에만 `ring_crossing_mask`가 연결됨
- ring의 foreground/background depth ordering도 없음

따라서 3D warp의 정확도가 좋아도, 합성 레이어가 잘못되면 최종 이미지는 좋아지지 않습니다.

**먼저 layer separation을 구현하고, 그 다음 globe layer에만 3D reprojection을 적용하는 순서**가 맞습니다.

---

# 현실적인 우선순위

## 하지 말아야 할 것

- `warp_scale`을 0.05, 0.10, 0.20 식으로 다시 맞추기
- feather width를 8, 12, 16, 24 px로 계속 sweep하기
- `weight_power`를 크게 올려 흐린 frame을 숨기기
- `ring_crossing_mask`의 임계값을 더 세밀하게 조정하기
- CH4를 필터명으로 특별 취급하는 예외를 늘리기

이들은 구조적 문제가 남은 상태에서 결과를 약간 다르게 보이게 할 뿐입니다.

---

## 해야 할 것

1. **Saturn System III를 `10.6562 h`로 정정**
2. **현 `ring_crossing_mask` 실사용을 중단하거나 experimental 처리**
3. **globe/ring wireframe overlay와 residual diagnostic 구현**
4. **전경/배경 ring visibility mask 구현**
5. **background ring → globe → foreground ring 3-layer composite**
6. 이후에만 globe layer 대상으로:
   - linear warp 대 3D reprojection 비교
   - NCC 비교
   - limb/ring-crossing 주변 ROI 비교

---

## 최종 판단

WinJUPOS를 더 조사하는 것은 유익하지만, 핵심 교훈은 특정 비공개 보간 공식이나 마법의 `warp_scale` 값이 아닙니다.

> **Saturn은 하나의 회전하는 표면이 아니다.**
>
> 따라서 하나의 de-rotation warp와 하나의 stack으로 처리하면, 고리와 대기 중 어느 하나는 필연적으로 틀어진다.

현재 구현은 이 사실을 정확히 발견했고, `compute_ring_crossing_mask()`까지 도달했습니다. 하지만 그 mask가 **가림 순서를 모르는 2D footprint mask**인 것이 마지막 구조적 장애물입니다.

다음 개선은 tuning이 아니라 **visibility-aware 3-layer compositing**이어야 합니다.