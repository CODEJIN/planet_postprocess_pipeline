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
    max_shift_px: float = 5.0        # max allowed alignment shift; larger → ignored (0 = no clamp)
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
    stack_min_quality: float = 0.0

    # Save per-filter monochrome frames alongside the composites (Step 8).
    # When True each filter's de-rotated grayscale image is saved as
    # {filter}_mono.png in every frame directory, and Step 9 will also
    # produce {filter}_animation.gif / .apng for each filter.
    save_mono_frames: bool = False

    # Series-specific composite specs (Step 8).  When set, these override
    # `specs` for Step 8 time-series compositing, allowing different channel
    # mappings from the Step 6 master composites.  None = use `specs`.
    series_specs: Optional[List[CompositeSpec]] = None



# ── Step 11: Summary contact sheet ────────────────────────────────────────────

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
    gamma: float = 1.0            # gamma correction (1.0 = linear/no change)
    font_size: int = 20           # label font size in pixels
    title_font_size: int = 24     # title font size in pixels (0 = no title)
    time_format: str = "%H%M"     # strftime format for row labels (e.g. "1233")


# ── Step 10: Animated GIF ─────────────────────────────────────────────────────

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
    min_frames: int = 20

    # AP grid — matched to AS!4 (AP Size=64, Min Bright=50/255≈0.196)
    # AS!4 uses 64px APs: Jupiter's belt/zone features span 20-50 px, so 64px
    # patches contain entire features → more stable phase correlation than 32px.
    ap_size: int = 64
    ap_step: int = 16                # 16 px step → ~127 APs; 4:1 size/step ratio
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
    use_adaptive_ap: bool = True
    ap_kr_sigma: float = 64.0        # KR sigma for adaptive warp maps (px)
    ap_candidate_step: int = 8       # candidate grid search step (px)

    # Stacking
    quality_weight_power: float = 3.0    # raised 2.0→3.0: stronger suppression of marginal frames

    # Sigma-clipping: 2-pass stacking that rejects outlier pixels per-frame.
    # Pass 1 accumulates normally and computes per-pixel mean/std.
    # Pass 2 re-warps each frame (from cached shifts) and rejects pixels where
    # |pixel − mean| > sigma_clip_kappa × std before re-accumulating.
    # This removes cosmic-ray hits, satellite trails, and seeing spikes that
    # fall in the same sky position across multiple frames.
    # 3.0 = conservative (keeps ~99.7% of good pixels under Gaussian noise).
    # 0.0 = disabled (use pass-1 result only; saves processing time).
    sigma_clip_kappa: float = 0.0    # disabled: sigma-clipping hurts lucky stacking
    # (seeing variation dominates per-pixel variance, causing good high-contrast
    #  frames to be clipped as "outliers" vs the blurry pass-1 mean)

    # Iterative refinement: use the first-pass stack as reference for a second pass.
    # The stacked result has ~√N better SNR than a single frame, so AP shifts on the
    # second pass are much more accurate, yielding a sharper final stack.
    # Sweep result: n_iterations=2 → ratio=1.056 vs AS!4 (31 s).
    #               n_iterations=3 → ratio=1.099 but slightly noisier (45 s).
    # 1 = single pass (fast); 2 = one refinement pass (recommended).
    n_iterations: int = 2

    # Parallelism: number of CPU workers for the frame stacking loop.
    # 0 = auto (all logical cores); 1 = single-threaded (no fork overhead).
    n_workers: int = 0

    # Post-stack sub-pixel smoothing.
    # GaussianBlur(sigma) applied to the final stacked image BEFORE wavelet sharpening.
    # Suppresses interpolation aliasing from INTER_LINEAR remap that concentrates at
    # wavelet level-1 (1-2px) and is amplified 29× by the sharpening step.
    # σ=0.9: CH4 noise 5.6×→1.1× vs AS!4, L2 (2-4px real detail) 87% preserved.
    # 0.0 = disabled (legacy behaviour, pre-try05).
    stack_blur_sigma: float = 0.9

    # Experimental — see docstring
    intra_video_derotate: bool = False


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

    # ── Observation metadata ───────────────────────────────────────────────────
    target: str = "Jup"
    filters: List[str] = field(
        default_factory=lambda: ["IR", "R", "G", "B", "CH4"]
    )

    # ── Helpers ────────────────────────────────────────────────────────────────

    def step_dir(self, step_num: int, name: str) -> Path:
        """Return the output directory Path for a step (does NOT create it)."""
        return self.output_base_dir / f"step{step_num:02d}_{name}"
