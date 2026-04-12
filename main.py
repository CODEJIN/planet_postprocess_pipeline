"""
Planetary imaging post-processing pipeline – main entry point.

Run from the project root:
    python main.py

Edit the PipelineConfig block below to customise paths, step save flags,
and processing parameters.
"""
from __future__ import annotations

from pathlib import Path

from pipeline.config import (
    CompositeConfig,
    CompositeSpec,
    DerotationConfig,
    GifConfig,
    PipelineConfig,
    PippConfig,
    QualityConfig,
    SummaryGridConfig,
    WaveletConfig,
)
from pipeline.steps import step01_pipp
from pipeline.steps import step02_lucky_stack
from pipeline.steps import step03_wavelet_sharpen
from pipeline.steps import step04_quality_assess
from pipeline.steps import step05_derotate_stack
from pipeline.steps import step06_wavelet_master
from pipeline.steps import step08_rgb_composite
from pipeline.steps import step09_series_composite
from pipeline.steps import step10_gif
from pipeline.steps import step11_summary_grid


def main() -> None:
    # ── Configuration ──────────────────────────────────────────────────────────
    # Edit here to control the pipeline.
    config = PipelineConfig(
        # ── Paths ──────────────────────────────────────────────────────────────
        ser_input_dir=Path("/data/astro_test/260402"),          # raw SER files (Step 1)
        input_dir=Path("/data/astro_test/260402_output/Step02_as!4"),  # stacked TIFs (Step 3+)
        output_base_dir=Path("/data/astro_test/260402_output"),

        # ── Step save flags ────────────────────────────────────────────────────
        save_step01=True,    # PIPP-processed SER files (Step 1)
        save_step03=True,    # Wavelet preview PNGs  (Step 3)
        save_step04=True,    # Quality score CSV     (Step 4, not yet implemented)
        save_step05=True,    # De-rotated TIFs       (Step 5, not yet implemented)
        save_step06=True,    # Master wavelet PNGs   (Step 6, not yet implemented)
        save_step08=True,    # RGB composites        (Step 8, not yet implemented)
        save_step09=True,    # Series composites     (Step 9, not yet implemented)
        save_step10=True,    # Animated GIF          (Step 10)
        save_step11=True,    # Summary contact sheet (Step 11)

        # ── Wavelet parameters (WaveSharp-compatible, 0–200 scale) ────────────
        wavelet=WaveletConfig(
            levels=6,
            # Step 3: layers 1,2,3 all at 200 — replicates WaveSharp reference
            # (sharpen_filter=0.1, power=1.0, amount=200 on each layer)
            preview_amounts=[200.0, 200.0, 200.0, 0.0, 0.0, 0.0],
            preview_power=1.0,
            preview_sharpen_filter=0.1,   # WaveSharp default noise gate
            # Step 6: same as preview (stacking improves SNR, noise is not the limit)
            master_amounts=[200.0, 200.0, 200.0, 0.0, 0.0, 0.0],
            master_power=1.0,
            master_sharpen_filter=0.0,
            # Rectangular border taper: eliminates stacking boundary gradients
            # before wavelet amplifies them. 30px is safe for 280×280 images
            # (background margin is ~44px; taper stays entirely in background).
            border_taper_px=30,
        ),

        # ── PIPP parameters (Step 1) ───────────────────────────────────────────
        pipp=PippConfig(
            roi_size=448,         # output crop size in pixels (square)
            min_diameter=50,      # minimum planet diameter to accept a frame
            size_tolerance=0.05,  # 5% tolerance vs. sliding-window median
            window_size=100,      # frames in sliding-window size reference
        ),

        # ── Quality assessment parameters ──────────────────────────────────────
        quality=QualityConfig(
            laplacian_weight=0.5,
            fourier_hf_weight=0.3,
            norm_variance_weight=0.2,
            top_fraction=0.3,        # Keep top 30% as "good quality"
        ),

        # ── De-rotation parameters ─────────────────────────────────────────────
        derotation=DerotationConfig(
            rotation_period_hours=9.9281,   # Jupiter System II
            horizons_id="599",               # Jupiter (599=Jup, 699=Sat, 499=Mars)
            observer_code="500@399",         # Geocentric (JPL Horizons)
            use_horizons=True,
            # normalize_brightness: rescale each frame's disk median to the reference
            # before stacking. Fixes luminance-drop artifacts (e.g. Window 2 B band).
            normalize_brightness=True,
            # min_quality_threshold: drop frames below this norm_score before stack.
            # 0.3 keeps only the better half of marginal windows.
            min_quality_threshold=0.3,
        ),

        # ── Animated GIF ──────────────────────────────────────────────────────
        gif=GifConfig(
            fps=6.0,
            loop=0,
            stretch_plow=0.5,
            stretch_phigh=99.5,
            resize_factor=1.0,
        ),

        # ── RGB / LRGB compositing ─────────────────────────────────────────────
        composite=CompositeConfig(
            specs=[
                CompositeSpec("RGB",      R="R",   G="G", B="B"),
                CompositeSpec("IR-RGB",   R="R",   G="G", B="B",  L="IR"),
                CompositeSpec("CH4-G-IR", R="CH4", G="G", B="IR", align_ref="IR"),
            ],
            align_channels=True,
            # Reduced from 15 → 8 px to prevent noise-driven CH4 misalignment
            max_shift_px=8.0,
            stretch_plow=0.1,
            stretch_phigh=99.9,
        ),

        # ── Summary contact sheet (Step 11) ───────────────────────────────────
        grid=SummaryGridConfig(
            composites=["RGB", "IR-RGB", "CH4-G-IR"],
            cell_size_px=300,      # resize each composite to 300×300
            gap_px=6,
            left_margin_px=40,
            bottom_margin_px=30,
            black_point=0.04,      # clip below 4% → deepens background blacks
            white_point=1.0,
            gamma=0.8,
            font_size=20,
            time_format="%H%M",    # e.g. "1233" for 12:33 UTC
        ),

        # ── Observation metadata ───────────────────────────────────────────────
        target="Jup",
        filters=["IR", "R", "G", "B", "CH4"],
    )

    # ── Step 1: PIPP preprocessing (frame reject + crop) ─────────────────────
    print("\n=== Step 1: PIPP Preprocessing ===")
    results_01 = step01_pipp.run(config)

    # ── Step 2: Lucky stacking (SER → TIF, AS!4-style local AP warp) ─────────
    print("\n=== Step 2: Lucky Stacking ===")
    results_02 = step02_lucky_stack.run(config)
    # After step 2, use its output as input for step 3+ (wavelet / derotation).
    if results_02 and config.save_step02:
        config.input_dir = config.step_dir(2, "lucky_stack")

    # ── Step 3: Wavelet sharpening preview ────────────────────────────────────
    print("\n=== Step 3: Wavelet Sharpening (Preview) ===")
    results_03 = step03_wavelet_sharpen.run(config)

    # ── Step 4: Quality assessment ─────────────────────────────────────────────
    print("\n=== Step 4: Quality Assessment ===")
    results_04 = step04_quality_assess.run(config)  # re-reads original TIFs
    #
    # ── [HUMAN CHECKPOINT] ────────────────────────────────────────────────────
    # Review output/step04_quality/windows_summary.txt
    # Confirm or override the automatically selected windows before Step 5.
    # To override: edit windows.json or pass custom window indices to Step 5.

    # ── Step 5: De-rotation stacking ──────────────────────────────────────────
    print("\n=== Step 5: De-rotation Stacking ===")
    results_05 = step05_derotate_stack.run(config, results_04)

    # ── Step 6: Wavelet sharpening (master) ───────────────────────────────────
    print("\n=== Step 6: Wavelet Sharpening (Master) ===")
    results_06 = step06_wavelet_master.run(config, results_05)

    # ── Step 8: RGB compositing (master) ──────────────────────────────────────
    print("\n=== Step 8: RGB Compositing (Master) ===")
    results_08 = step08_rgb_composite.run(config, results_06)

    # ── Step 9: Time-series compositing ───────────────────────────────────────
    print("\n=== Step 9: Time-series RGB Compositing ===")
    results_09 = step09_series_composite.run(config, results_03)

    # ── Step 10: Animated GIF ─────────────────────────────────────────────────
    print("\n=== Step 10: Animated GIF ===")
    step10_gif.run(config, results_09)

    # ── Step 11: Summary contact sheet ────────────────────────────────────────
    print("\n=== Step 11: Summary Contact Sheet ===")
    step11_summary_grid.run(config, results_08, results_05)

    print("\n=== Pipeline finished ===")


if __name__ == "__main__":
    main()
