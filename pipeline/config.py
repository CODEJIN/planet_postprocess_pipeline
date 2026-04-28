"""
Pipeline configuration.

Edit this file (or instantiate PipelineConfig in main.py) to control:
  - Input/output paths
  - Which step results to save (save_stepXX flags)
  - Processing parameters per step
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# ── Step 7 & 5: Wavelet sharpening ────────────────────────────────────────────

@dataclass
class WaveletConfig:
    """WaveSharp-compatible à trous B3-spline wavelet sharpening parameters.

    amounts[i] = sharpening amount for layer i+1 (same 0–200 scale as WaveSharp).
    Layer 1 = finest scale (~2 px), Layer 6 = coarsest.

    power:          WaveSharp 'power function' exponent (1.0 = linear).
    sharpen_filter: WaveSharp 'sharpen filter' — soft-threshold coefficient
                    per layer.  0.0 = no noise gate (matches WaveSharp default).
    """
    levels: int = 6

    # Step 7 – all three active layers at maximum (matches WaveSharp reference)
    preview_amounts: List[float] = field(
        default_factory=lambda: [200.0, 200.0, 200.0, 0.0, 0.0, 0.0]
    )
    preview_power: float = 1.0
    preview_sharpen_filter: float = 0.1   # WaveSharp default (MAD-based soft threshold)

    # Step 5 – final master output (best-quality stack per window)
    master_amounts: List[float] = field(
        default_factory=lambda: [200.0, 200.0, 200.0, 0.0, 0.0, 0.0]
    )
    master_power: float = 1.0
    master_sharpen_filter: float = 0.0

    # Step 8 – time-series animation frames (independent from Step 5)
    # Defaults match master_amounts so existing behaviour is unchanged.
    # Tune separately if the animation needs gentler/stronger sharpening.
    series_amounts: List[float] = field(
        default_factory=lambda: [200.0, 200.0, 200.0, 0.0, 0.0, 0.0]
    )
    series_power: float = 1.0
    series_sharpen_filter: float = 0.0

    # Rectangular border taper before wavelet sharpening (Step 7 and Step 5).
    # Cosine-fades the outermost border_taper_px pixels on all 4 sides to 0,
    # removing de-rotation stacking boundary gradients before wavelet can
    # amplify them.  The taper boundary lies in the near-zero background
    # region, so it does not create a new wavelet-amplifiable edge.
    # 0 = disabled.  For 280×280 images (background ~44 px), 30 is safe.
    border_taper_px: int = 0

    # Disk-edge feathering factor for sharpen_disk_aware (Steps 6 & 8).
    # Per-level feather width = 2^L × edge_feather_factor pixels.
    # With pre-fill + disk_expand_px active, kernel contamination is eliminated
    # and the feather only suppresses the de-rotation coverage gradient.
    # 2.0 is typically sufficient when disk_expand_px is set correctly.
    edge_feather_factor: float = 2.0

    # Same as edge_feather_factor but applied only to Step 8 time-series frames.
    series_edge_feather_factor: float = 2.0

    # Extra pixels to expand the disk mask boundary beyond what find_disk_center
    # detects.  find_disk_center uses Otsu thresholding on the contour, which
    # lands inside the true visual limb due to limb darkening.  Expanding by
    # 5–8 px shifts the feather zone to start at/beyond the actual limb so that
    # disk interior pixels near the real edge get full wavelet gain.
    # 0 = disabled (mask starts exactly at detected contour).
    # Ignored when auto_params=True (value estimated per-image from data).
    disk_expand_px: float = 0.0

    # When True, edge_feather_factor and disk_expand_px are estimated
    # automatically from each de-rotation stack image before sharpening.
    # The manual values above are ignored; auto-estimated values are printed.
    # Uses wavelet.auto_wavelet_params() — see that function for details.
    auto_params: bool = False


# ── Step 3: Quality assessment ─────────────────────────────────────────────────

@dataclass
class QualityConfig:
    """Image quality scoring weights and thresholds."""
    laplacian_weight: float = 0.5        # Laplacian variance (sharpness)
    fourier_hf_weight: float = 0.3       # High-frequency Fourier power
    norm_variance_weight: float = 0.2    # Normalized variance (contrast)
    # Frames with norm_score below this threshold are excluded before window search.
    # 0.0 = use all frames (disabled). Recommended: 0.2–0.3 to drop obviously bad frames.
    min_quality_threshold: float = 0.05

    # ── De-rotation window parameters ─────────────────────────────────────────
    # window_frames: number of filter cycles (= time-series frames) that form
    # one de-rotation window.  Actual duration = window_frames × cycle_minutes.
    window_frames: int = 3               # Number of frames (filter cycles) per window
    cycle_minutes: float = 3.75          # One filter cycle = 225 s (IR→R→G→B→CH4)
    outlier_sigma: float = 1.5           # Sigma threshold for outlier exclusion
    n_windows: int = 1                   # Number of windows to find
    # When True windows may overlap in time; when False each window must be at
    # least window_minutes away from every other selected window (non-overlapping).
    allow_overlap: bool = False
    #   Jupiter rotates ~0.6°/min; 13.5 min = ~8° rotation (practical limit ~20°)

    @property
    def window_minutes(self) -> float:
        """Derived: window duration in minutes (window_frames × cycle_minutes)."""
        return self.window_frames * self.cycle_minutes


# ── Step 4 / 8 / 9: De-rotation ───────────────────────────────────────────────

@dataclass
class DerotationConfig:
    """Planetary de-rotation parameters.

    Rotation period reference by planet:
        Jupiter  9.9281 h  (System II, atmospheric)
        Saturn  10.5600 h  (System III, radio/atmospheric)
        Mars    24.6229 h
        Neptune 16.1100 h
        Uranus  17.2400 h

    Horizons body IDs:
        Jupiter 599 | Saturn 699 | Mars 499 | Venus 299
        Uranus  799 | Neptune 899
    """
    # Atmospheric rotation period in hours
    rotation_period_hours: float = 9.9281
    # JPL Horizons body ID used to query the pole position angle (NP_ang).
    # Change this when switching targets (e.g. "699" for Saturn).
    horizons_id: str = "599"
    observer_code: str = "500@399"   # JPL Horizons geocentric observer

    # Spherical warp scale factor (empirically determined ~0.80 for 260320).
    # Theoretical value is 1.0; values < 1.0 apply the remainder as a rigid
    # horizontal shift per-frame before stacking.  Lower values increase east/
    # west limb blurring; 0.80 was found optimal for the 260320 Jupiter dataset.
    warp_scale: float = 0.80

    # Per-frame brightness normalization before stacking.
    # Rescales each frame so its planet-disk median matches the reference frame.
    # Useful when transparency drops across a window cause luminance mismatch.
    # WARNING: normalization discards real brightness information — use only
    # when blotchy artifacts are visible. Default False (preserve raw values).
    normalize_brightness: bool = False

    # Frames with norm_score below this threshold are excluded from stacking.
    # 0.0 = include all frames (disabled). Recommended: 0.05–0.1.
    min_quality_threshold: float = 0.05


# ── Step 8: RGB / LRGB compositing ────────────────────────────────────────────

@dataclass
class CompositeSpec:
    """Defines one composite output: which filter maps to which channel.

    Set L to a filter name to enable LRGB compositing (L replaces the
    luminance channel in Lab space).  Leave L as None for plain RGB.

    Example presets:
        CompositeSpec("RGB",      R="R",   G="G", B="B")
        CompositeSpec("IR-RGB",   R="R",   G="G", B="B",  L="IR")
        CompositeSpec("CH4-G-IR", R="CH4", G="G", B="IR")
    """
    name: str
    R: str
    G: str
    B: str
    L: Optional[str] = None          # luminance filter (None = no LRGB)
    lrgb_weight: float = 1.0         # 1.0 = pure L luminance, 0.0 = keep RGB L
    align_ref: Optional[str] = None  # alignment reference channel (None = auto: highest signal)


@dataclass
class CompositeConfig:
    """Configuration for Step 8 RGB/LRGB compositing."""
    specs: List[CompositeSpec] = field(default_factory=lambda: [
        CompositeSpec("RGB",      R="R",   G="G", B="B"),
        CompositeSpec("IR-RGB",   R="R",   G="G", B="B",  L="IR"),
        CompositeSpec("CH4-G-IR", R="CH4", G="G", B="IR"),
    ])
    align_channels: bool = True      # phase-correlation alignment between channels
    max_shift_px: float = 15.0       # max allowed alignment shift; larger → ignored (0 = no clamp)
    # Colour-channel stretch mode (applied to R, G, B before compositing):
    #   "joint"       – same lo/hi computed from all colour channels combined;
    #                   preserves natural colour ratios (recommended, matches GIMP)
    #   "independent" – each channel stretched to its own full range (over-bright)
    #   "none"        – no pre-stretch; use native pixel values (matches raw GIMP compose)
    color_stretch_mode: str = "none"
    stretch_plow: float = 0.1        # percentile low  (used in joint / independent mode)
    stretch_phigh: float = 99.9      # percentile high (used in joint / independent mode)

    # Output brightness scale applied to every Step 8 series composite.
    # Simple multiplication: comp *= series_scale.  1.0 = no change.
    # 0.80 makes the result slightly darker while preserving the pixel
    # distribution (no clipping or stretching of any channel).
    series_scale: float = 0.80

    # Global per-filter normalisation across ALL series frames (Step 8).
    # When True a lightweight first pass reads all Step 7 PNGs, computes the
    # 0.5th–99.5th percentile lo/hi for every filter across every frame, and
    # applies that single mapping before compositing.  This ensures that the
    # same filter has the same brightness range in every frame, eliminating
    # frame-to-frame colour shifts caused by varying atmospheric transparency.
    # Recommended: True when producing animated GIFs (Step 9).
    global_filter_normalize: bool = True

    # Duration of one complete filter cycle in Step 8 (seconds).
    # Used to group raw TIF frames into per-cycle sets before compositing.
    # Typical value: 270 s (45 s × 5 filters + overhead).
    # Kept separate from QualityConfig.cycle_minutes (Step 3) so the two
    # steps can be tuned independently.
    cycle_seconds: float = 225.0

    # Sliding-window stacking (Step 8).
    # stack_window_n: number of consecutive filter cycles to stack per output
    #   frame.  1 = single-frame mode (current behaviour).  Odd values keep the
    #   centre frame as the reference time.  Recommended: 1–5.
    # stack_min_quality: normalised quality threshold [0, 1].  Frames whose
    #   Laplacian-variance score (computed from the Step 7 wavelet PNG) is below
    #   this fraction of the per-filter maximum are excluded from the stack.
    #   0.0 = accept all frames.
    stack_window_n: int = 3
    stack_min_quality: float = 0.05

    # Save per-filter monochrome frames alongside the composites (Step 8).
    # When True each filter's de-rotated grayscale image is saved as
    # {filter}_mono.png in every frame directory, and Step 9 will also
    # produce {filter}_animation.gif / .apng for each filter.
    save_mono_frames: bool = False

    # Series-specific composite specs (Step 8).  When set, these override
    # `specs` for Step 8 time-series compositing, allowing different channel
    # mappings from the Step 6 master composites.  None = use `specs`.
    series_specs: Optional[List[CompositeSpec]] = None



# ── Step 10: Summary contact sheet ────────────────────────────────────────────

@dataclass
class SummaryGridConfig:
    """Configuration for Step 11 summary contact sheet.

    Produces a grid PNG with time on the rows and composite type on the columns,
    matching the reference layout (e.g. RGB / IR-RGB / CH4-G-IR across the top,
    times down the left).

    black_point / white_point / gamma:
        Levels adjustment applied to each cell to deepen background blacks and
        add visual depth to the planet.  black_point=0.04 clips anything below
        ~4% to pure black, which removes faint background gradients.
    """
    composites: List[str] = field(
        default_factory=lambda: ['RGB', 'IR-RGB', 'CH4-G-IR']
    )
    cell_size_px: int = 300        # resize each cell to this square (0 = native)
    gap_px: int = 6                # gap between cells (pixels, black)
    left_margin_px: int = 55      # left margin for time labels
    bottom_margin_px: int = 30    # bottom margin for composite type labels
    top_margin_px: int = 44       # top margin for title bar
    black_point: float = 0.04     # clip below this value (darkens background)
    white_point: float = 1.0      # clip above this value
    gamma: float = 0.9            # gamma correction (1.0 = linear/no change)
    font_size: int = 20           # label font size in pixels
    title_font_size: int = 24     # title font size in pixels (0 = no title)
    time_format: str = "%H%M"     # strftime format for row labels (e.g. "1233")


# ── Step 09: Animated GIF ─────────────────────────────────────────────────────

@dataclass
class GifConfig:
    """Parameters for Step 9 animated GIF output."""
    fps: float = 6.0              # playback speed (frames per second)
    loop: int = 0                 # 0 = infinite loop
    resize_factor: float = 1.0   # downscale factor for smaller file (1.0 = no resize)


# ── Step 2: Lucky stacking ────────────────────────────────────────────────────

@dataclass
class LuckyStackConfig:
    """AS!4-style lucky stacking from SER video frames.

    Frame selection:
      top_percent      — use the best top_percent of frames by Laplacian score.
                         0.15 = top 15 %. Raise for smoother stacks (more noise);
                         lower for sharper stacks (fewer frames, noisier).
      min_frames       — minimum number of frames to stack regardless of top_percent.
      reference_n_frames — number of top frames averaged to build the reference image.

    AP (Alignment Point) grid:
      ap_size          — patch size for local cross-correlation (pixels, power of 2).
      ap_step          — AP centre spacing; smaller = denser grid, slower processing.
      ap_search_range  — maximum allowed local shift per AP (pixels).
      ap_min_contrast  — minimum RMS contrast of a reference patch to use that AP.
                         Low-contrast patches (uniform limb, sky) give noisy shifts.
      ap_confidence_threshold — minimum phaseCorrelate peak height to accept a shift.
                         Values below this fall back to global-shift only for that AP.
      ap_sigma_factor  — Gaussian KR smoothing sigma = ap_step × ap_sigma_factor.
                         Must be ≥ 1/√2 ≈ 0.71 to guarantee C∞-smooth warp field
                         (prevents triangle-edge gradient artifacts). Higher values
                         smooth out noisy AP shifts at the cost of spatial resolution.
                         Typical range: 0.7 – 1.5.

    Stacking:
      quality_weight_power — quality score exponent for weighted stacking.
                         2.0 = best frames contribute quadratically more weight.
                         1.0 = linear quality weighting.  0 = equal weights.

    Intra-video de-rotation:
      intra_video_derotate — EXPERIMENTAL/RISKY. Applies spherical_derotation_warp()
                         to compensate for planetary rotation within the ~90-second
                         video window before AP warp. Requires SER timestamps.
                         Default False: the AP warp already absorbs the ~0.9°
                         rotation at no risk of warp-composition artifacts.
    """
    # Frame selection
    # Matches AS!4 default (25%) for fair comparison.
    top_percent: float = 0.25
    reference_n_frames: int = 50     # top frames mean-stacked as initial reference
    # reference_midpoint_percentage: percentile of quality distribution to centre the
    # reference window on (0 = top frames, 75 = 75th-percentile frames).
    # AS!4 uses reference_midpoint_percentage=75: frames that are "solidly good"
    # (not lucky outliers) make a more representative reference for phase correlation.
    # try49 confirmed: midpoint=75 gives highest correlation with AS!4 output (0.9844).
    reference_midpoint_percentage: int = 75
    # reference_percent: use this fraction of total frames for the reference stack
    # (0.0 = use reference_n_frames instead). e.g. 0.5 = top 50% of all frames.
    # AS!4 uses reference_num_frames=5394 on a 10341-frame IR file ≈ 52%.
    # try49 baseline: 0.0 (reference_n_frames=50 with midpoint=75 gave best correlation).
    reference_percent: float = 0.0
    min_frames: int = 20
    # Quality scoring metric for frame selection.
    # "laplacian"      — Laplacian variance (legacy, CV ~1.4% on poor seeing).
    # "gradient"       — Tenengrad global (mean squared Sobel, ksize=3).
    # "local_gradient" — Local Tenengrad at AP patch positions (default).
    #                    Matches AS!4 quality_type=Gradient+local; CV ~4-6%.
    #                    Generates AP grid from middle frame in Phase 0.5 so
    #                    no AS3 file is required. Strongly recommended for
    #                    poor-seeing data where global scorers fail to
    #                    discriminate frames (260415 analysis: CV 1.4%→4.4%).
    # "log_disk"       — Laplacian of Gaussian variance on planet disk.
    #                    Matches AS!4 "lapl3" quality metric (LoG sigma=3).
    #                    Spearman corr with AS!4 scores: 0.74 vs 0.006 (local_gradient).
    #                    log_disk_sigma / log_disk_threshold 파라미터로 조정.
    # try84 confirmed: log_disk visually better detail than try80 (local_gradient)
    score_metric: str = "log_disk"

    # LoG disk scoring 파라미터 (score_metric="log_disk" 시 사용)
    log_disk_sigma: float = 3.0      # GaussianBlur sigma before Laplacian
    log_disk_threshold: float = 0.25 # disk mask brightness threshold (normalized)
    # Sobel kernel size for local gradient quality scoring (score_frames_local).
    # ksize=3: standard, fastest. ksize=5 or 7: more noise-robust (wider kernel).
    # Matches AS!4 quality_gradient_noise_robust=3 when set to 5 or 7.
    # Default: 3 (backward-compatible).
    quality_gradient_ksize: int = 3

    # AP grid — matched to AS!4 (AP Size=64, Min Bright=50/255≈0.196)
    # AS!4 uses 64px APs: Jupiter's belt/zone features span 20-50 px, so 64px
    # patches contain entire features → more stable phase correlation than 32px.
    ap_size: int = 64
    ap_step: int = 0                 # 0 = auto (ap_size // 2). 명시적 값 지정 시 그대로 사용.
    ap_search_range: int = 20
    ap_min_contrast: float = 0.01    # minimum patch RMS contrast (reject uniform sky)
    ap_min_brightness: float = 0.196 # minimum patch mean brightness (≈ AS!4 Min Bright 50/255)
    # Sweep result: conf=0.15 optimal; lower thresholds accept noisy shifts that hurt.
    ap_confidence_threshold: float = 0.15
    # σ = ap_step × 0.7 = 11.2 px. Minimum for C∞ continuity: ap_step/√2 = 11.3 px.
    # Marginally below theoretical minimum but empirically optimal (wider sigma over-smooths).
    ap_sigma_factor: float = 0.9     # σ = ap_step × 0.9 = 14.4px (April-11 code optimal)

    # Adaptive AP grid (try14+: LoG scale detection + dynamic AP sizes + wide KR)
    # When True, replaces the uniform 64px grid with a local-scale-aware sparse AP
    # set (8-11 APs at mixed sizes 64–128px) selected by LoG energy + cross-size NMS.
    # Max AP size scales with disk_radius (max = disk_radius × 1.28 rounded to 8px),
    # so larger telescopes automatically get proportionally larger AP patches.
    # ap_kr_sigma: Gaussian KR smoothing sigma; 64px covers sparse AP gaps across
    #   a ~200px disk (vs legacy 14.4px which was sized for ~122 dense APs).
    # ap_candidate_step: dense candidate search step before NMS (pixels).
    # try49 베이스라인: uniform 64px grid (adaptive는 6~9개만 생성, AS!4 수준 안됨)
    use_adaptive_ap: bool = False
    # Multi-scale AP grid matching AS!4 double_ap_grid: 64px + 96px + 192px layers.
    # Overrides use_adaptive_ap when True (implies use_adaptive_ap=False path +
    # multi-size triples → adaptive warp map KR).
    use_double_ap_grid: bool = False
    # Minimum-sufficient-size multi-scale AP grid (try64).
    # Candidate grid at ap_size//2 spacing; for each position tries ap_size, ap_size*2, ...
    # up to disk_radius — uses the smallest size meeting ap_min_contrast.
    # Overrides use_adaptive_ap and use_double_ap_grid.
    use_multiscale_ap: bool = False
    ap_kr_sigma: float = 64.0        # KR sigma for adaptive warp maps (px)
    ap_candidate_step: int = 8       # candidate grid search step (px)

    # Stacking
    quality_weight_power: float = 3.0    # raised 2.0→3.0: stronger suppression of marginal frames

    # Sigma-clipping: extra pass after n_iterations stacking that rejects
    # outlier pixels per-frame.  Uses the final stacked result as reference:
    # re-warps all frames, computes per-pixel mean/std, then discards pixels
    # where |pixel − mean| > sigma_clip_kappa × std before nanmean.
    # Visually confirmed sharper than plain stacking (260407 overnight test).
    # Memory cost: one additional (N, H, W) float32 array (~800 MB for 25%).
    sigma_clip: bool = False         # enable sigma-clipping post-pass
    sigma_clip_kappa: float = 2.0    # k-sigma threshold (2.0 = overnight test value)

    # Iterative refinement: use the first-pass stack as reference for a second pass.
    # The stacked result has ~√N better SNR than a single frame, so AP shifts on the
    # second pass are much more accurate, yielding a sharper final stack.
    # Sweep result: n_iterations=2 → ratio=1.056 vs AS!4 (31 s).
    #               n_iterations=3 → ratio=1.099 but slightly noisier (45 s).
    # 1 = single pass (fast); 2 = one refinement pass (recommended).
    n_iterations: int = 1

    # Parallelism: number of CPU workers for the frame stacking loop.
    # 0 = auto (all logical cores); 1 = single-threaded (no fork overhead).
    n_workers: int = 0

    # SER-level parallelism: number of SER files to process simultaneously.
    # Each SER uses (n_workers // n_ser_parallel) frame workers internally.
    # 0 = auto (cpu_count // 4); 1 = sequential (default, safe for low-RAM systems).
    n_ser_parallel: int = 1

    # Post-stack sub-pixel smoothing.
    # GaussianBlur(sigma) applied to the final stacked image BEFORE wavelet sharpening.
    # Suppresses interpolation aliasing from INTER_LINEAR remap that concentrates at
    # wavelet level-1 (1-2px) and is amplified 29× by the sharpening step.
    # σ=0.9: CH4 noise 5.6×→1.1× vs AS!4, L2 (2-4px real detail) 87% preserved.
    # 0.0 = disabled. try44 confirmed: removing blur gives 2.1× Laplacian variance.
    # Default changed to 0.0 (try44/49 baseline).
    stack_blur_sigma: float = 0.0

    # cv2.remap interpolation mode for the combined global+local warp.
    # INTER_LINEAR (1): bilinear — fast, introduces ~0.5px blur.
    # INTER_CUBIC  (2): bicubic — sharper, recommended when stack_blur_sigma=0.
    # INTER_LANCZOS4 (8): sharpest, highest quality but slower.
    # Default changed to INTER_CUBIC (try44 confirmed: 1.36× Laplacian variance vs LINEAR).
    remap_interpolation: int = 2  # cv2.INTER_CUBIC

    # stabilization_planet_threshold: fixed brightness threshold (0–255) for planet
    # disk detection in limb_center_align(). 0 = use Otsu adaptive threshold (default).
    # AS!4 uses _stabilization_planet_threshold=20 (≈7.8% of full scale) for consistent
    # disk edge detection across frames regardless of background brightness variation.
    stabilization_planet_threshold: int = 0

    # ── try53: QSF sub-pixel peak refinement ──────────────────────────────────
    # When True, replaces cv2.phaseCorrelate with manual FFT cross-correlation +
    # 2D quadratic surface fitting (QSF) on the 3×3 neighborhood of the
    # correlation peak for sub-pixel accuracy. AS!4 uses QSF internally.
    # try58 confirmed: per_ap_selection + QSF = 46.7% Laplacian (best result).
    # Default: True (try58 baseline).
    use_qsf: bool = True

    # ── try54: CoG (Centre-of-Gravity) global alignment ───────────────────────
    # When True, replaces limb_center_align() ellipse fitting with image-moments
    # brightness-weighted centroid (cv2.moments) for global per-frame stabilization.
    # AS!4 calls this "Planet CoG" stabilization. Potentially more robust than
    # ellipse fitting when the limb is partially clipped or poorly defined.
    # Default: False (use limb_center_align ellipse fitting).
    cog_align: bool = False

    # ── try55: Noise Robust pre-scoring blur ───────────────────────────────────
    # GaussianBlur applied ONLY before gradient quality computation in
    # score_frames_local(). Does NOT affect the stacked image.
    # 0 = disabled. 1 = σ0.5px, 2 = σ1.0px, 3 = σ1.5px (matches AS!4 NR=3).
    # AS!4 uses this before quality gradient computation to reduce noise impact
    # on frame scoring without blurring the stack itself.
    # Default: 0 (disabled — try55 baseline is σ0).
    quality_noise_robust: int = 0

    # ── try56: Per-AP independent frame selection ──────────────────────────────
    # When True, computes per-AP quality scores for each frame during stacking,
    # builds a 2D quality weight map via Gaussian KR, and accumulates with
    # spatially varying weights. Different image regions are accumulated with
    # different per-region quality weights, matching AS!4's independent AP frame
    # selection. Requires local_gradient scoring to generate per-AP scores.
    # Runs in both sequential and parallel paths (n_workers > 1 supported).
    # try56 confirmed: 45.8% Laplacian. try58 (+QSF): 46.7%.
    # Routes to _spatial_per_ap_quality_stack: full KR-warped frames with
    # spatially-varying quality weights computed from per-AP Sobel on the
    # globally-aligned frame. No patch boundaries → wavelet-safe.
    per_ap_selection: bool = True
    # Exponent applied to per-AP Sobel scores for spatial quality weighting.
    # 3–4 provides sharp local selectivity without hard frame-subset cutoffs.
    per_ap_quality_power: float = 3.0

    # ── try68: True per-AP independent stacking ───────────────────────────────
    # For each AP, independently select the best sub-frames by LOCAL quality at
    # that AP position, then stack only those patches. Different APs use different
    # frame subsets — the core of true lucky imaging.
    # per_ap_stack_sub_percent: fraction of globally-selected frames used per AP
    #   (0.5 = top 50% of pre-selected pool → effectively top 12.5% globally)
    use_per_ap_stack: bool = False
    per_ap_stack_sub_percent: float = 0.5

    # ── try69: Fourier-domain quality-weighted stacking (Mackay 2013) ──────────
    # use_fourier_quality: weight each frame's contribution per spatial frequency
    #   by |FFT(frame)|^fourier_quality_power — frames sharper at freq f get more
    #   weight at that frequency. No spatial patch boundaries.
    use_fourier_quality: bool = True   # try69 BEST: 66.4% (was 46.7% with per_ap_selection)
    fourier_quality_power: float = 1.0

    # ── try73: Fourier spectral SNR masking ───────────────────────────────────
    # After weighted averaging, suppress frequencies where frames disagree.
    # snr(f) = mean(|F_n(f)|) / std(|F_n(f)|).  mask = tanh(snr / threshold).
    # threshold=1.0: SNR=1→76% pass, SNR=0.5→46% pass. Requires n_workers=1.
    fourier_snr_mask: bool = False
    fourier_snr_threshold: float = 1.0

    # ── try74: Fourier high-frequency rolloff ────────────────────────────────
    # Gaussian low-pass applied to output spectrum. sigma_f in normalized freq
    # units (0=DC, 0.5=Nyquist). 0.0 = disabled.
    # try80 confirmed best: sigma=0.20 (Noise%=1.0%, Detail%=11.6%, Pearson=0.9956)
    fourier_rolloff_sigma: float = 0.20

    # ── try75: Fourier noise floor subtraction ───────────────────────────────
    # Estimate per-frequency noise floor from bottom-25% quality frames.
    # Subtract before computing quality weights: max(|F| - floor, 0)^power.
    # Prevents noisy frames from gaining weight at noise-dominated frequencies.
    # Requires n_workers=1.
    fourier_noise_floor: bool = False

    # ── try57: Patch blending (PSS style) ─────────────────────────────────────
    # When True, replaces the KR warp field + single cv2.remap with per-AP patch
    # accumulation: for each AP and frame, extract the warped patch and accumulate
    # with a Gaussian window mask. The final image is the normalized per-pixel sum
    # of all AP contributions. Matches PlanetarySystemStacker's patch blending,
    # which avoids the KR interpolation artifacts at sparse AP boundaries.
    # Default: False (use KR warp field + single remap).
    use_patch_blend: bool = False

    # ── AS!4 greedy PDS AP grid (session-wide) ────────────────────────────────
    # When True, generates APs via greedy Poisson Disk Sampling (raster scan)
    # matching AS!4's exact placement algorithm (reverse-engineered, 96-100% match).
    # Three independent layers: s, round(s×1.5/8)×8, s×3.
    # min_dist per layer = round(ap_size × 35/64).
    # In session-wide mode (step02): APs are generated once from the reference SER
    # and shared across all SERs (with per-SER disk offset correction).
    use_as4_ap_grid: bool = False
    # Reference filter for session-wide AP generation.
    # Priority when empty (auto): IR > R > G > B > CH4 > color.
    # Set explicitly (e.g. "IR") to force a specific filter.
    ap_reference_filter: str = ""

    # try62: scikit-image DFT upsampling for sub-pixel AP shift (0.1px precision).
    # Uses phase_cross_correlation(upsample_factor=10) instead of cv2.phaseCorrelate
    # for the per-AP local shift estimation step.  Confidence gating still uses
    # phaseCorrelate (more reliable for rejection), then refines with DFT upsampling.
    # PSS SubpixelRegistration mode equivalent.
    # Default: False (use cv2.phaseCorrelate or QSF).
    use_pcc_upsample: bool = False

    # NCC (Normalized Cross-Correlation) AP shift estimation.
    # Replaces phase correlation with spatial-domain NCC computed via FFT:
    #   cc = IFFT(conj(FFT(ref-μ)) · FFT(frm-μ)) / (N·σ_ref·σ_frm)
    # Peak of cc ∈ [0,1] (true correlation coefficient, lower scale than
    # phaseCorrelate). Hann windowing + QSF sub-pixel refinement always applied.
    # Default: False.
    use_ncc: bool = True

    # NCC-specific confidence threshold override.
    # -1.0 (default) = use ap_confidence_threshold unchanged.
    # Set explicitly if NCC needs a different threshold than phase correlation.
    ncc_confidence_threshold: float = -1.0

    # try63: Thin Plate Spline warp interpolation instead of Gaussian KR.
    # TPS passes EXACTLY through each reliable AP's measured shift (no dilution),
    # whereas KR smooths/averages them with a Gaussian kernel (sigma=ap_kr_sigma).
    # Coverage mask (same Gaussian density as KR) prevents wild TPS extrapolation
    # in the sky / border regions outside the AP convex hull.
    # tps_smoothing=0.0 → exact interpolation; >0 → regularized (outlier robust).
    # Default: False (use KR).
    use_tps: bool = False
    tps_smoothing: float = 0.0

    # Debayer Bayer-pattern SER frames to RGB after stacking.
    # When True and the SER ColorID is a Bayer pattern (8–11), the stacked
    # single-channel result is demosaiced to (H,W,3) RGB before saving.
    # Required for color camera pipelines (steps 05/06/07/08 expect RGB input).
    # Default True; set False only if you want the raw Bayer stack for inspection.
    debayer: bool = True

    # Experimental — see docstring
    intra_video_derotate: bool = False

    def __post_init__(self) -> None:
        if self.ap_step <= 0:
            self.ap_step = self.ap_size // 2


# ── Step 1: PIPP preprocessing ────────────────────────────────────────────────

@dataclass
class PippConfig:
    """Frame rejection and ROI crop parameters (PIPP-style preprocessing).

    Applies to raw SER files before any stacking or sharpening.

    roi_size:            Output frame width/height in pixels (square crop).
    min_diameter:        Minimum planet diameter to accept a frame (pixels).
    size_tolerance:      Relative tolerance vs. sliding-window median (e.g. 0.05 = 5%).
    window_size:         Number of accepted frames used as size reference.
    aspect_ratio_limit:  Max deviation from 1:1 aspect ratio (0.2 = 20%).
    straight_edge_limit: Fraction of a bounding-box edge that may be lit before the
                         frame is considered clipped by a straight edge (0.5 = 50%).
    """
    roi_size: int = 448
    min_diameter: int = 50
    size_tolerance: float = 0.05
    window_size: int = 100
    aspect_ratio_limit: float = 0.2
    straight_edge_limit: float = 0.5
    # Parallel file processing: number of SER files processed simultaneously.
    # Capped at 4 in step01_pipp.py regardless of this value (I/O contention).
    # 0 = auto (min(4, cpu_count)); 1 = sequential.
    n_workers: int = 0


# ── Top-level pipeline config ─────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    # ── Paths ─────────────────────────────────────────────────────────────────
    ser_input_dir: Path = field(default_factory=lambda: Path("/data/astro_test/260402"))
    input_dir: Path = field(default_factory=lambda: Path("/data/astro_test/AS_P25"))
    output_base_dir: Path = field(default_factory=lambda: Path("/data/astro_test/output"))
    # When set, step01_pipp writes here instead of output_base_dir/step01_pipp/
    step01_output_dir: Optional[Path] = None
    # When set, step02 reads SER files from here (GUI panel choice, highest priority)
    step02_ser_dir: Optional[Path] = None
    # When set, step02_lucky_stack writes here instead of output_base_dir/step02_lucky_stack/
    step02_output_dir: Optional[Path] = None
    # When set, step07 writes here instead of output_base_dir/step07_wavelet_preview/
    step07_output_dir: Optional[Path] = None

    # ── Step save flags ────────────────────────────────────────────────────────
    save_step01: bool = True   # PIPP-processed SER files
    save_step02: bool = True   # Lucky-stacked TIF files
    save_step03: bool = True   # Quality scores CSV + ranked file list
    save_step04: bool = True   # De-rotated master TIFs per filter
    save_step05: bool = True   # Wavelet-sharpened master PNGs
    save_step06: bool = True   # RGB / IR-RGB / CH4-G-IR composites
    save_step07: bool = True   # Wavelet preview PNGs
    save_step08: bool = True   # RGB composites per time-series set
    save_step09: bool = True   # Animated GIF
    save_step10: bool = True   # Summary contact sheet

    # ── Sub-configs ────────────────────────────────────────────────────────────
    pipp: PippConfig = field(default_factory=PippConfig)
    lucky_stack: LuckyStackConfig = field(default_factory=LuckyStackConfig)
    wavelet: WaveletConfig = field(default_factory=WaveletConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    derotation: DerotationConfig = field(default_factory=DerotationConfig)
    composite: CompositeConfig = field(default_factory=CompositeConfig)

    gif: GifConfig = field(default_factory=GifConfig)
    grid: SummaryGridConfig = field(default_factory=SummaryGridConfig)

    # ── Camera mode ────────────────────────────────────────────────────────────
    # "mono"  : separate mono captures per filter (default — IR/R/G/B/CH4)
    # "color" : single color (Bayer) camera; one RGB stream, no filter separation.
    #           Steps 04–07 sharpen/derotate the single COLOR channel in Lab space.
    #           Step 08 is a colour pass-through (no compositing needed).
    #           Step 11 shows a single-column grid.
    camera_mode: str = "mono"

    # Apply auto WB + CA correction in Step 7 (color camera only)
    wavelet_color_correct: bool = True

    # ── Observation metadata ───────────────────────────────────────────────────
    target: str = "Jup"
    filters: List[str] = field(
        default_factory=lambda: ["IR", "R", "G", "B", "CH4"]
    )

    # ── Helpers ────────────────────────────────────────────────────────────────

    def step_dir(self, step_num: int, name: str) -> Path:
        """Return the output directory Path for a step (does NOT create it)."""
        return self.output_base_dir / f"step{step_num:02d}_{name}"
