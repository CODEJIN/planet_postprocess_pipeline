# PlanetFlow — Algorithm Technical Guide

---

## Table of Contents

1. [Overview](#1-overview)
2. [Step 01 — PIPP Pre-processing](#2-step-01--pipp-pre-processing)
3. [Step 02 — Lucky Stacking](#3-step-02--lucky-stacking)
4. [Step 03 — Quality Assessment & Window Detection](#4-step-03--quality-assessment--window-detection)
5. [Step 04 — De-rotation Stacking](#5-step-04--de-rotation-stacking)
6. [Step 05 / 07 — Wavelet Sharpening](#6-step-05--07--wavelet-sharpening)
7. [Step 06 / 08 — RGB Compositing](#7-step-06--08--rgb-compositing)
8. [Step 09 — Animated GIF](#8-step-09--animated-gif)
9. [Step 10 — Summary Grid](#9-step-10--summary-grid)
10. [Common Module: Disk Detection](#10-common-module-disk-detection-find_disk_center)
11. [Common Module: Sub-pixel Alignment](#11-common-module-sub-pixel-alignment)

---

## 1. Overview

This document explains **what each GUI parameter actually does inside the algorithm**. For usage instructions, see `guide_en.md`. Use this document when you want to understand the meaning of parameters and the mathematical principles behind them.

### Source File Structure

```
pipeline/
├── modules/
│   ├── planet_detect.py    # Step 01: Planet detection and validation
│   ├── lucky_stack.py      # Step 02: Lucky Stacking core algorithm
│   ├── quality.py          # Step 03: Image quality assessment and window selection
│   ├── derotation.py       # Step 04: De-rotation warp and stacking
│   ├── wavelet.py          # Step 05/07: À trous wavelet sharpening
│   └── composite.py        # Step 06/08: RGB/LRGB compositing
└── config.py               # Global configuration (dataclass-based)
```

---

## 2. Step 01 — PIPP Pre-processing

**Source**: `pipeline/modules/planet_detect.py`, `pipeline/steps/step01_pipp.py`

```
Input frame
    │
    ▼
Convert to 8-bit grayscale
    │
    ▼
GaussianBlur(5×5)  ← Noise suppression
    │
    ▼
Triangle threshold (Zack 1977)
    │
    ▼
Largest connected component (8-connectivity) extraction
    │
    ▼
4-stage validation → reject frame on failure
    │
    ▼
Square ROI crop at bounding box center
```

### GUI Parameters → Internal Behavior

| GUI Parameter | Default | Internal Behavior |
|---|---|---|
| **ROI Size (px)** | 448 | Output square size of `get_cropped_frame()`. Uses `round()` for integer conversion — prevents the 0.5-pixel systematic bias that `int()` would introduce. Pixels outside image boundaries are filled with 0 (black) |
| **Min Diameter (px)** | 50 | Final criterion of the 4-stage validation. Frame is rejected if `max(bw, bh) < min_diameter` |

### Internal Fixed Values

| Parameter | Value | Role |
|---|---|---|
| `padding` | 10 px | Boundary check: planet bounding box must be at least this far from image edges |
| `aspect_ratio_limit` | 0.2 | Aspect ratio check: `min(w,h)/max(w,h) ≥ 1 − 0.2 = 0.8` required to pass |
| `straight_edge_limit` | 0.5 | Straight-edge check: frame rejected if any of the 4 bounding box sides is ≥ 50% lit |
| GaussianBlur kernel | 5×5 | Noise suppression before thresholding |

### Triangle Auto-threshold

Implemented with OpenCV's `THRESH_TRIANGLE` flag. Determines the threshold by finding the minimum point most distant from the histogram's highest peak. Operates stably across a wide exposure range regardless of aperture settings or planet size.

### Bounding Box Center Crop

Jupiter has non-uniform brightness due to belts and the Great Red Spot. A brightness-weighted centroid would introduce systematic bias toward bright structures. Following PIPP's approach, **bounding box center** `(x + w/2, y + h/2)` is used to prevent this bias.

---

## 3. Step 02 — Lucky Stacking

**Source**: `pipeline/modules/lucky_stack.py`

```
SER input file
    │
    ▼
[Phase 1] Frame quality scoring (log_disk metric)
    │
    ▼
Select top top_percent% frames → selected_indices
    │
    ▼
[Phase 2] Reference frame construction
    Frames near 75th-percentile quality rank
    → global NCC alignment → mean stack (stable, representative reference)
    │
    ▼
[Phase 3] AP grid generation (uniform grid or Greedy PDS 3 layers)
    │
    ▼
[Phase 4] Global alignment per frame
    Limb-centre ellipse fitting → bicubic sub-pixel warp (INTER_CUBIC)
    │
    ▼
[Phase 5] Fourier-domain quality-weighted stacking
    For each frame n, compute F_n = FFT(aligned_frame_n)
    Accumulate: S(f) += |F_n(f)|^power × F_n(f)
    Weight:     W(f) += |F_n(f)|^power
    Stacked spectrum: S(f) / W(f)
    │
    ▼
[Phase 6] Gaussian rolloff filter in frequency domain
    Gaussian(σ_f = 0.20 in normalised freq) applied to output spectrum
    → IFFT → stacked image
    │
    ▼
If n_iterations = 2: use result as reference → repeat from [Phase 3]
    │
    ▼
Output TIF
```

### GUI Parameters → Internal Behavior

| GUI Parameter | Default | Internal Behavior |
|---|---|---|
| **Top Frame Percent (%)** | 25 | `top_percent = 0.25`. Only the top N% of frames by quality score are used. `n_select = max(min_frames, round(n_frames × top_percent))` |
| **AP Size (px)** | 64 | Base size s for the AP grid. With PDS: Layer 1=s, Layer 2=round(s×1.5/8)×8, Layer 3=s×3. AP step defaults to ap_size ÷ 2 (ap_step=0 = auto). |
| **N Iterations** | 1 | `n_iterations`. When set to 2, the 1st-pass stack is used as the reference frame for the 2nd pass → higher reference SNR → improved alignment precision |
| **Warp Method** | Gaussian KR | Warp field interpolation. **Gaussian KR** (default): C∞-continuous Nadaraya-Watson kernel regression. **TPS** (Thin Plate Spline): sharper local transitions similar to AS!4 triangulation, but slower and may extrapolate unstably at disc edges. |
| **Fourier Quality Power** | 1.0 | `w_n(f) = │FFT_n(f)│^power`. The primary per-frequency accumulation weight. Higher values give sharper frames more influence at high spatial frequencies. 1.0=linear (default), 1.5–2.0=more aggressive. |
| **SER Parallel** | 1 | Number of SER files processed simultaneously. 0=auto (CPU cores÷4). Total thread budget = n_workers fixed. Each SER gets `n_workers ÷ N_SER` frame-level threads. ~950 MB RAM per SER. |
| **AS!4 AP Grid** | Off | Off=uniform grid (spacing=AP size÷2). On=Greedy PDS 3-layer: dense at disk center, sparse toward limb |

### Internal Fixed Values

| Parameter | Value | Role |
|---|---|---|
| `score_metric` | `"log_disk"` | Frame quality scoring method. Default matches AS!4's *lapl3* metric. Also available: `"local_gradient"`, `"laplacian"` (changeable in config) |
| `reference_midpoint_percentage` | 75 | Reference frames are centred at the 75th-percentile quality rank (not the top). "Solidly good" frames make a more representative reference than rare lucky outliers (AS!4 default). |
| `reference_n_frames` | 50 | Number of frames used to construct the reference frame (centred at midpoint_percentage). |
| `score_step` | 2 | Only every 2nd frame is scored; the rest are estimated by linear interpolation |
| `ap_confidence_threshold` | 0.15 | APs with phase correlation confidence below this value are discarded |
| `ap_sigma_factor` | 0.7 | Gaussian KR σ = ap_step × 0.7. Satisfies σ ≥ ap_step/√2 to guarantee C∞ continuous warp field |
| `remap_interpolation` | `INTER_CUBIC` | cv2.remap interpolation mode for the global warp (bicubic; sharper than LINEAR, no post-stack blur needed) |
| `fourier_rolloff_sigma` | 0.20 | Gaussian rolloff sigma in normalised frequency units (0=DC, 0.5=Nyquist). Suppresses residual high-frequency noise without blurring real planetary detail. |

### AP Size and Grid Placement

**Uniform grid** (AS!4 AP Grid Off): AP spacing = AP size ÷ 2. Covers the disk interior uniformly.

**Greedy PDS** (AS!4 AP Grid On): Generates 3 independent layers via raster scan, with base size s.

| Layer | AP size (s=64 baseline) | Minimum AP spacing |
|-------|-------------------------|--------------------|
| Layer 1 | 64px | `round(64 × 35/64)` = 35px |
| Layer 2 | 96px (`round(64×1.5/8)×8`) | `round(96 × 35/64)` = 52px |
| Layer 3 | 192px (`64×3`) | `round(192 × 35/64)` = 105px |

Acceptance criteria per AP: ① inside disk, ② mean patch brightness ≥ 0.196 (50/255), ③ minimum distance from existing APs satisfied. Integral Image is used for O(1) patch mean computation.

### Frame Quality Scoring Modes (score_metric)

Selected in config; default is `"log_disk"`.

**`"log_disk"`** (default): Matches AS!4's *lapl3* metric. Laplacian variance computed after Gaussian blur, on pixels brighter than a threshold. Spearman correlation 0.74 vs AS!4 frame rankings (sigma=3.0, threshold=0.25).
```
mask = (frame / max) > 0.25
score = var(Laplacian(GaussianBlur(frame, σ=3.0)))  on mask
```

**`"local_gradient"`**: Maximum Sobel gradient in each AP patch. Maximum is used because its coefficient of variation (CV≈6%) is 4× higher than the mean (CV≈1.4%), giving much better inter-frame discriminability in poor seeing.
```
patch_score = max(gx² + gy²)  over ap_size × ap_size
frame_score = mean(patch_score) over all APs
```

**`"laplacian"`**: Laplacian variance on the inner 80% of the disk. Excludes the limb boundary (always large gradients regardless of seeing) to measure only atmospheric transparency.
```
mask = dist_from_center ≤ disk_radius × 0.80
score = var(Laplacian(frame / 255))  on mask
```

### Fourier-Domain Quality-Weighted Stacking

The primary stacking algorithm uses frequency-domain accumulation with per-frame quality weights at each spatial frequency (Mackay 2013, arXiv:1303.5108):

```
For each globally-aligned frame n:
    F_n(f) = FFT(aligned_frame_n)
    weight  = |F_n(f)|^power          per-frequency weight

Stacked spectrum:
    S(f) = Σ_n [weight_n(f) × F_n(f)] / Σ_n weight_n(f)

Gaussian rolloff:
    G(f) = exp(−f² / (2σ_f²))        σ_f = 0.20 (normalised freq)
    S_filtered(f) = S(f) × G(f)

Output:
    stack = real(IFFT(S_filtered))
```

**Why Fourier-domain weighting**: A simple mean of aligned frames weights every frame equally at every frequency. If some frames are sharper than average only at high spatial frequencies (fine planetary detail), their contribution is diluted. Fourier weighting ensures that the sharpest frame at each frequency contributes most — equivalent to optimal linear combination in the frequency domain. The result has more power at fine scales than a simple mean while maintaining natural colour and brightness.

**Gaussian rolloff rationale**: All stacking methods accumulate some high-frequency noise (interpolation aliasing, camera read noise). The rolloff suppresses frequencies beyond roughly 0.2×Nyquist where signal-to-noise drops. It is tuned so that L1 wavelet sharpening (×200) can recover fine detail without amplifying noise residuals.

### Local Warp Estimation and Gaussian KR

Per-AP Hann-windowed phase correlation with QSF (quadratic surface fitting) sub-pixel refinement estimates shifts for the global warp map. Trusted AP shifts are interpolated into a full-resolution warp field using Gaussian Kernel Regression (Nadaraya-Watson):

```
sigma = ap_step × ap_sigma_factor    (default: 32 × 0.7 = 22.4px)

smooth_wx = GaussianBlur(shift_x × confidence, ksize, sigma)
smooth_w  = GaussianBlur(confidence, ksize, sigma)
map_dx    = smooth_wx / smooth_w     (only where coverage ≥ 5% of maximum)
```

**Why Gaussian KR instead of Delaunay**: Delaunay linear interpolation produces C⁰-continuous fields (gradient discontinuities at triangle edges). These mesh patterns accumulate after stacking thousands of frames, and wavelet sharpening (×200) amplifies them into visible grid artifacts. Gaussian KR produces C∞-continuous fields.

---

## 4. Step 03 — Quality Assessment & Window Detection

**Source**: `pipeline/modules/quality.py`

```
Step 02 TIF file list
    │
    ▼
Per TIF image:
    Otsu threshold → disk mask extraction
    GaussianBlur(σ=1.2) → denoise
    Laplacian variance (×0.5) + Tenengrad (×0.3) + Normalized variance (×0.2)
    → composite raw_score
    │
    ▼
Per-filter min-max normalization → norm_score ∈ [0, 1]
    │
    ▼
Per candidate window × per filter:
    σ-clipping (1.5σ) → remove outliers
    quality_post × snr_factor × stability → filter_quality
    │
    ▼
Geometric mean across filters → window_quality
    │
    ▼
Select top N windows under non-overlap constraint
    │
    ▼
Output: windows.json / *_ranking.csv
```

### GUI Parameters → Internal Behavior

| GUI Parameter | Default | Internal Behavior |
|---|---|---|
| **Window (frames)** | 3 | Window length in filter cycle counts. Actual window time = frames × cycle seconds. Used as `n_expected = window_frames` for snr_factor calculation |
| **Cycle Seconds** | 225 | Duration of one filter cycle (IR→R→G→B→CH4→IR). Used only to compute expected frame count `n_expected = window_minutes / cycle_minutes`. Independent from Step 8's cycle seconds |
| **N Windows** | 1 | Number of optimal windows to detect. Step 04 uses 1 window; Step 08 time-series uses multiple |
| **Allow Overlap** | Off | Off: each window center must be ≥ window_minutes away from all already-selected windows |
| **Min Quality Threshold** | 0.05 | Frames with `norm_score < threshold` are excluded from window quality calculation. 0.0 includes all frames |

### Internal Fixed Values

| Parameter | Value | Role |
|---|---|---|
| Laplacian weight | 0.5 | Fraction of Laplacian variance in the composite score |
| Tenengrad weight | 0.3 | Fraction of Tenengrad (sum of squared Sobel) in the composite score |
| Normalized variance weight | 0.2 | Fraction of `var/mean` in the composite score |
| Denoise σ | 1.2 px | Gaussian blur before sharpness metrics. Prevents noisy-but-blurry frames from scoring high |
| σ-clipping threshold | 1.5σ | Outlier frame removal criterion within each window |

### Window Quality Calculation

For each candidate window, filter-level quality is computed then combined via geometric mean:

```
# Per filter
quality_post = mean(norm_score of included)
snr_factor   = min(1.0, √(n_included / n_expected))
stability    = 1 / (1 + CV)          CV = std/mean

filter_quality = quality_post × snr_factor × stability

# Across all filters
window_quality = (∏_f  filter_quality_f) ^ (1 / num_filters)
```

**Why geometric mean**: If any one filter is very poor, the overall window quality drops substantially. All filters must meet a minimum standard to produce a good composite image.

---

## 5. Step 04 — De-rotation Stacking

**Source**: `pipeline/modules/derotation.py`

```
Step 02 TIF + windows.json
    │
    ▼
Disk detection from reference frame (shared across entire window)
    Otsu → Closing(7×7) → fitEllipse → (cx, cy, semi_a, semi_b, angle)
    │
    ▼
NP.ang lookup (bundled table → user cache → live Horizons API)
    │
    ▼
Per frame in window:
    ├─ Observation time Δt → longitude displacement Δλ_rad
    ├─ Oblate spheroid depth calculation → per-pixel drift
    ├─ remap (CUBIC interior / LINEAR limb, 12px cosine feather)
    └─ Sub-pixel alignment (limb center → fallback: phase correlation)
    │
    ▼
Quality-weighted accumulation → master TIF output
```

### GUI Parameters → Internal Behavior

| GUI Parameter | Default | Internal Behavior |
|---|---|---|
| **Warp Scale** | 0.80 | Spherical warp intensity multiplier. `drift = warp_scale × Δλ_rad × depth(x,y)`. Theoretical value is 1.0, but 0.80 is the experimentally optimal value accounting for atmospheric blur and plate scale uncertainty. The auto-search button finds the value maximizing Laplacian variance |
| **Min Quality Threshold** | 0.05 | Frames with `norm_score < threshold` are excluded from stacking accumulation |
| **Normalize Brightness** | Off | Normalizes each frame's brightness to match the reference frame before stacking. Use when inter-frame brightness variation is large |

### Internal Fixed Values

| Parameter | Value | Role |
|---|---|---|
| `polar_equatorial_ratio` | 0.935 (Jupiter) | Polar/equatorial radius ratio of the oblate spheroid. `polar_scale = 1 / ratio` in the depth formula |
| R (sphere radius) | `disk_radius × 1.05` | 5% padding: avoids the `√(R²−r²)` singularity at the limb |
| `_interp_feather_px` | 12.0 px | CUBIC/LINEAR interpolation transition zone. Cosine fade over the inner 12px from the limb |
| `margin_factor` | 0.10 | Lowers Otsu threshold by 10% to include dark limb pixels |

### Spherical De-rotation Warp Formula

Pixel displacement due to longitude change Δλ is proportional to the sphere's depth at that point:

```
Δλ_rad = (dt_sec / period_sec) × 2π

# Decompose along pole position angle (pole_pa_deg = NP.ang)
rx_eq  = (x−cx)×cos(pa) + (y−cy)×sin(pa)   (equatorial direction)
ry_pol = -(x−cx)×sin(pa) + (y−cy)×cos(pa)  (polar direction)

# Oblate spheroid depth
depth² = R² − rx_eq² − polar_scale² × ry_pol²
depth  = sqrt(max(0, depth²))

drift  = warp_scale × Δλ_rad × depth

map_x  = x − drift × cos(pole_pa_rad)
map_y  = y − drift × sin(pole_pa_rad)
```

### Why a Shared Disk Center Matters

Detecting the disk independently per frame causes (cx, cy) to vary by a few pixels, applying slightly different spherical warps to each frame. After stacking, the limb boundaries misalign, and wavelet sharpening amplifies this into asymmetric limb artifacts. Therefore, detection is performed once from a single reference frame and the same values are applied to the entire window.

### NP.ang Lookup Priority

1. **Bundled table** (offline): `pipeline/data/np_ang_table.json` — Jupiter (599), Saturn (699), Mars (499) data for 2016–2036. Linear interpolation with 360°/0° wrap handling.
2. **User cache**: `~/.astropipe/horizons_cache.json` — cached results from previous online lookups.
3. **Live Horizons API**: Used when outside bundled range or for Custom planets.

---

## 6. Step 05 / 07 — Wavelet Sharpening

**Source**: `pipeline/modules/wavelet.py`

Step 05 (master sharpening) and Step 07 (preview) use the same algorithm. Only the parameters and target image differ.

```
Input TIF
    │
    ▼
Disk detection → (cx, cy, rx, ry, angle)
    │
    ▼
auto_wavelet_params:
    expand_px = sqrt(rx·ry) × 0.0505
    eff = median limb-inward brightness gradient width / 2
    │
    ▼
Pre-fill outside ellipse (remove limb→background discontinuity → prevent ringing)
    │
    ▼
À Trous B3 wavelet decomposition (6 levels)
    → [detail_0 (~2px), …, detail_5 (~64px), residual]
    │
    ▼
Per level i:
    σ_noise = MAD(detail_i) / 0.6745
    gain_i  = (amount_i/200)^power × MAX_GAIN[i]
    weight_i = cosine S-curve elliptical mask (feather = 2^i × eff)
    contrib_i = soft_threshold(detail_i, gain_i × σ_noise) × gain_i × weight_i
    │
    ▼
Reconstruct: original + Σ contrib_i → PNG output
```

### GUI Parameters → Internal Behavior

| GUI Parameter | Default | Internal Behavior |
|---|---|---|
| **L1 (0–500)** | 200 | Amplifies ~2-pixel scale wavelet coefficients. `gain = (200/200)^1.0 × 29.15 = 29.15`. Finest pixel-level resolution detail |
| **L2 (0–500)** | 200 | ~4-pixel scale. `gain = (200/200)^1.0 × 9.48 = 9.48`. Fine structures (belts, bands) |
| **L3 (0–500)** | 200 | ~8-pixel scale. `MAX_GAIN[2] = 0.0` → currently inactive. Mid-scale structures |
| **L4 (0–500)** | 0 | ~16-pixel scale. `MAX_GAIN[3] = 0.0`. Large-scale contrast (noise amplification risk) |
| **L5, L6 (0–500)** | 0 | ~32/64-pixel scale. `MAX_GAIN[4,5] = 0.0`. Not recommended |

> **What amount means**: `gain_i = (amount/200)^power × MAX_GAIN[i]`. amount=200 → gain=MAX_GAIN. amount=400 → gain=2×MAX_GAIN. amount=100 → gain=0.5×MAX_GAIN.

### Internal Fixed Values

| Parameter | Value | Role |
|---|---|---|
| `MAX_GAINS` | [29.15, 9.48, 0, 0, 0, 0] | Per-level maximum gain determined by OLS regression against WaveSharp reference output |
| `sharpen_filter` | 0.1 | Soft threshold strength. `thr = 0.1 × σ_noise`. Suppresses small noise-level coefficients |
| `power` | 1.0 | `gain = (amount/200)^power × MAX_GAIN`. 1.0=linear |
| `edge_feather_factor` | auto | Limb feather width factor. auto_wavelet_params() measures automatically from the image |
| `expand_px` | auto | Pushes Otsu boundary outward so feathering starts at the true limb. `sqrt(rx×ry) × 0.0505` |

### À Trous B3-Spline Wavelet Decomposition

"À trous" (with holes) is an undecimated wavelet that scales without downsampling by inserting zeros between taps.

```
_B3 = [1, 4, 6, 4, 1] / 16   (5-tap separable kernel)

At level i, tap spacing = 2^i:
  smoothed_i = B3_i ⊗ image_i    (reflect padding)
  detail_i   = image_i − smoothed_i
  image_{i+1} = smoothed_i
```

### Disk-Aware Edge Feathering

Wavelet gains are applied only inside the planet disk, fading out toward the limb via a cosine S-curve. Higher levels use a wider feather:

```
feather_L = 2^L × edge_feather_factor
t = clip(dist_from_boundary / feather_L, 0, 1)
weight_L = 0.5 × (1 − cos(π × t))
```

**Outside-ellipse pre-fill**: Before wavelet decomposition, pixels outside the disk are filled with the nearest limb pixel value. Prevents the bright limb ring artifact caused when the B3 kernel reads background 0-values.

---

## 7. Step 06 / 08 — RGB Compositing

**Source**: `pipeline/modules/composite.py`

```
Per-filter PNGs (R, G, B, [IR, L, …])
    │
    ▼
Channel auto-stretch (joint / independent / none)
    │
    ▼
Fixed reference channel selection: L > IR > R > G > B
    Non-reference channels → phase correlation → apply_shift
    │
    ▼
np.stack([R, G, B])
    │
    ├─ [RGB mode] proceed as-is
    └─ [LRGB mode] RGB→Lab, replace Lab_L, Lab→RGB
    │
    ▼
Disk detection → Lab conversion → cosine fade on a/b channels (0.89r ~ 1.04r)
    │
    ▼
RGB PNG output
```

### Step 06 GUI Parameters → Internal Behavior (Mono mode)

| GUI Parameter | Default | Internal Behavior |
|---|---|---|
| **Max Channel Shift (px)** | 15.0 | If the phase-correlation-computed inter-channel shift exceeds this value, alignment is not applied (prevents runaway misalignment). Raise to 20–30 on nights with strong atmospheric dispersion |
| **Composite Specs (R/G/B/L channels)** | RGB, IR-RGB, CH4-G-IR | Defines the filter-to-channel mapping for each composite image. Specifying an L channel activates LRGB compositing mode |

### Step 08 GUI Parameters → Internal Behavior (Mono mode)

| GUI Parameter | Default | Internal Behavior |
|---|---|---|
| **Global Filter Normalize** | On | Unifies the brightness range of each filter across the entire time series. Reduces color inconsistency between frames in Step 9 GIF |
| **Brightness Scale** | 1.00 | `composite × series_scale`. 1.0=no change |
| **Window (frames)** | 3 | Sliding window size. Odd numbers recommended. 3=1 frame on each side. SNR improvement ∝ √N |
| **Cycle Seconds** | 225 | Used for grouping Step 07 PNGs into time-series frame sets. Independent from Step 3's cycle seconds |
| **Min Quality Filter** | 0.05 | Soft reduction of low-quality frame contributions (weight reduction, not complete exclusion) |
| **Mono Frames per Filter** | Off | On: saves per-filter grayscale frames → Step 9 also generates per-filter grayscale GIFs |
| **L1–L6 (series)** | [200, 200, 200, 0, 0, 0] | Wavelet sharpening applied independently to each time-series frame. Completely separate from Step 5 settings |
| **Composite Specs** | RGB, IR-RGB, CH4-G-IR | Time-series-specific channel mapping, independent from Step 6 |

### Internal Fixed Values

| Parameter | Value | Role |
|---|---|---|
| Reference channel priority | L > IR > R > G > B | Fixed to prevent the composite planet position from shifting due to dynamic reference selection |
| `desat_start` | `disk_radius × 0.89` | Limb desaturation start radius |
| `desat_width` | `disk_radius × 0.15` | Cosine fade zone (complete at 1.04×r) |
| stretch default | `"none"` | No auto-stretch. `"joint"` = unified R/G/B lo/hi, `"independent"` = per-channel |

### LRGB Compositing

When an L channel is specified, the luminance (L) is replaced with the external channel in Lab color space:

```
Lab = cv2.cvtColor(rgb, COLOR_RGB2Lab)
Lab[:,:,0] = lrgb_weight × (L_external × 100) + (1−w) × Lab[:,:,0]
result = cv2.cvtColor(Lab, COLOR_Lab2RGB)
```

For IR-RGB compositing: the IR channel's higher resolution provides luminance detail, while R/G/B contribute natural color.

### Post-composite Limb Desaturation

Removes color fringing caused by wavelength-dependent limb darkening differences (the G disk appears ~1.5 pixels larger than B). Suppresses only the a/b (chroma) channels in Lab color space while preserving the L channel (brightness).

---

## 8. Step 09 — Animated GIF

**Source**: `pipeline/steps/step09_gif.py`

```
Step 08 time-series composite PNGs (sorted by timestamp)
    │
    ▼
Bilinear resampling by scale_factor
    │
    ▼
Pillow ImageSequence assembly
    frame_duration = 1000 / fps  [ms]
    │
    ▼
GIF output (loop=0, infinite repeat)
```

### GUI Parameters → Internal Behavior

| GUI Parameter | Default | Internal Behavior |
|---|---|---|
| **FPS** | 6.0 | `frame_duration = round(1000 / fps)` [ms]. Passed as the `duration` argument to Pillow's `save()` |
| **Resize Factor** | 1.0 | `new_size = (round(w × factor), round(h × factor))`. Pillow BILINEAR resampling |

---

## 9. Step 10 — Summary Grid

**Source**: `pipeline/steps/step10_summary_grid.py`

Step 10 always produces `summary_grid_simple.png` and, when in mono mode with Step 05 output present, additionally produces `summary_grid.png` (two-zone) and optionally per-window analytic PNGs in `analytic/`.

### Output Files

| File | Contents | Condition |
|------|----------|-----------|
| `summary_grid_simple.png` | Composites only (all windows × all composites) | Always |
| `summary_grid.png` | Composites (left zone) + Step 05 filter images (right zone), same cell size, vertical divider | Mono mode + Step 05 data exists |
| `analytic/window_XX_analytic.png` | Per-window detailed view (see below) | `save_analytic=True`, mono mode |

### Simple Grid (`summary_grid_simple.png`)

```
Step 06 composite PNGs (all windows)
    │
    ▼
Per image:
    Black point: pixel = clip((p − bp) / (1 − bp), 0, 1)
    Gamma:       pixel = pixel ^ (1 / gamma)
    Resample to cell_size
    │
    ▼
Grid layout (rows = windows, columns = composites) → PNG
```

### Two-Zone Grid (`summary_grid.png`)

```
Left zone: Step 06 composites (cell_size × cell_size each)
Right zone: Step 05 filter PNGs (same cell_size)
    │
    ▼
Vertical divider between zones
Column labels above each image
Row (window) time labels on the left
    │
    ▼
summary_grid.png
```

Both zones use `cell_px` = the configured **Cell Size** value. The filter zone width = `n_filters × cell_px + gaps`.

### Analytic View (`analytic/window_XX_analytic.png`)

One PNG per time window. Layout (top to bottom):

```
[Header: window time range]

[Filter images row]           ← Step 05 PNGs, cell_size each
[Filter stats block]          ← Frames / Q.Post / Stab. / Stacked per filter, column-aligned to images above

─────────────────────────────────── (divider)

[Composite images row]        ← Step 06 PNGs, cell_size each
[Align table]                 ← rows = filter names; columns = composites
                                 cell = "[role] shift" or "[role] ref" or "—"
[Sat row]                     ← saturation boost per composite

─────────────────────────────────── (separator)

[Global params line]          ← Win.Q / Rot / Wvl / bp / γ
```

#### Filter Stats Block

Drawn above the divider, x-aligned to filter image columns. No duplicate filter name header (filter names already appear as image labels above).

| Row | Value |
|-----|-------|
| **Frames** | `n_used / n_total` (frames passing quality threshold / all frames in window) |
| **Q.Post** | Mean quality score of retained frames after σ-clipping, 0–1 |
| **Stab.** | `1 / (1 + CV)` where `CV = std/mean` of per-frame quality scores |
| **Stacked** | Final frame count actually summed in the lucky stack |

#### Align Table

- **Rows** = filter names (IR, R, G, B, CH4 …) derived from `CompositeSpec` fields across all composites
- **Columns** = composite names (RGB, IR-RGB, …)
- **Cell value** = `[role] shift` where `role` ∈ {L, R, G, B} and `shift` = `(Δx, Δy)` from `composite_log.json`; `ref` if the filter was the reference channel; `—` if the filter is not used in that composite
- **Alignment keys** in `composite_log.json` are filter names (IR/R/G/B/CH4), not channel roles

#### Canvas Height Pre-calculation

Height is computed before `Image.new()` using a 1×1 probe draw:

```python
canvas_h = (pad + header_h
            + filter_lbl_h + filter_px   # filter images
            + fstats_h                   # filter stats (above divider)
            + section_gap                # divider
            + comp_lbl_h + comp_px       # composite images (name only label)
            + apar_h                     # align table + separator + global params
            + pad)
```

`label_margin` (width of widest row label + 12 px) is added to `canvas_w` to prevent row labels from overflowing the left edge.

### GUI Parameters → Internal Behavior

| GUI Parameter | Default | Internal Behavior |
|---|---|---|
| **Black Point** | 0.04 | `pixel = clip((p − 0.04) / (1 − 0.04), 0, 1)`. Pushes background noise to pure black. 0.02–0.08 recommended |
| **Gamma** | 0.9 | `pixel = pixel ^ (1/0.9) ≈ pixel ^ 1.11`. <1.0=brighter (0.9 default slightly brightens the planet), >1.0=darker, 1.0=no change |
| **Cell Size (px)** | 300 | Each composite and filter image in the grid is resampled to this size. Both zones use the same cell size in the two-zone grid |
| **Save Analytic View** | False | When True, generates `analytic/window_XX_analytic.png` for each time window. Mono mode only |

---

## 10. Common Module: Disk Detection (find_disk_center)

**Source**: `pipeline/modules/derotation.py`

Used in common across multiple Steps (04, 05, 06, 08).

```
1. arr8 = clip(image × 255, 0, 255).uint8
2. Compute Otsu threshold → effective_thresh = Otsu × (1 − 0.10)
   (margin_factor=0.10: lowers threshold slightly to include dark limb pixels)
3. Morphological Closing (7×7 elliptical kernel) → fills small gaps inside disk
4. Largest contour extraction (ellipse fitting if ≥5 points):
   (cx, cy), (ma, mi), angle = cv2.fitEllipse(largest_contour)
5. Return: (cx, cy, semi_major, semi_minor, angle_deg)
   (always guarantees semi_major ≥ semi_minor; adds 90° to angle if needed)
```

---

## 11. Common Module: Sub-pixel Alignment

**Source**: `pipeline/modules/derotation.py`

### apply_shift

```python
M = np.float32([[1, 0, dx], [0, 1, dy]])
cv2.warpAffine(image, M, (w, h), flags=INTER_CUBIC, borderMode=BORDER_REPLICATE)
```

- **INTER_CUBIC**: Bicubic interpolation (detail preservation)
- **BORDER_REPLICATE**: Replicates edge pixels (BORDER_CONSTANT=0 would bleed black into the limb region)

### subpixel_align (phase correlation)

```python
(dx, dy), _ = cv2.phaseCorrelate(ref_f32, tgt_f32)
```

Estimates translation with ~0.1-pixel precision via frequency-domain cross-correlation.

---

*References: Starck & Murtagh (2006), Bijaoui (1991), Mackay (2013 arXiv:1303.5108), Zack (1977)*
