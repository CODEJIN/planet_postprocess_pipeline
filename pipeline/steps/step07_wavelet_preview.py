"""
Step 7 – Wavelet sharpening preview.

Applies strong à trous wavelet sharpening to every stacked TIF in the input
directory and writes auto-stretched PNG previews organised by filter.

These over-sharpened PNGs are used for visual quality inspection (Step 3):
exaggerated sharpening makes atmospheric artefacts and focus differences
easier to spot at a glance.

Output (when config.save_step07 is True):
    <output_base_dir>/step07_wavelet_preview/
        IR/  <stem>_wavelet.png
        R/   …
        G/   …
        B/   …
        CH4/ …
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from pipeline.config import PipelineConfig
from pipeline.modules import image_io, wavelet
from pipeline.steps.step06_rgb_composite import _auto_color_correct



# Type alias: per-filter list of (output_path_or_None, metadata_dict)
StepResult = Dict[str, List[Tuple[Optional[Path], dict]]]


def run(config: PipelineConfig, progress_callback=None, cancel_event=None) -> StepResult:
    """Run Step 7 for all TIF files in *config.input_dir*.

    Args:
        config: Pipeline configuration.

    Returns:
        ``{filter_name: [(png_path, meta), ...]}``

        *png_path* is the written PNG file path when ``config.save_step07``
        is True, otherwise None.
    """
    # ── Resolve output directory ───────────────────────────────────────────────
    out_dir: Optional[Path]
    if config.save_step07:
        if config.step07_output_dir is not None:
            out_dir = Path(config.step07_output_dir)
        else:
            out_dir = config.step_dir(7, "wavelet_preview")
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Output → {out_dir}")
    else:
        out_dir = None
        print("  save_step07=False: results will not be written to disk")

    # ── Discover and group input files ─────────────────────────────────────────
    groups = image_io.group_by_filter(config.input_dir, config.target)
    if not groups:
        print(f"  [WARNING] No matching TIF files found in {config.input_dir}")
        return {}

    total = sum(len(v) for v in groups.values())
    print(f"  Found {total} TIF files across {len(groups)} filters: "
          f"{sorted(groups)}")

    # ── Process each filter ────────────────────────────────────────────────────
    results: StepResult = {}
    done = 0

    for filter_name in sorted(groups):
        if cancel_event is not None and cancel_event.is_set():
            print("  [CANCELLED] Stopping Step 7.", flush=True)
            break
        entries = groups[filter_name]
        results[filter_name] = []

        # Per-filter sub-directory
        filter_out_dir: Optional[Path] = None
        if out_dir is not None:
            filter_out_dir = out_dir / filter_name
            filter_out_dir.mkdir(exist_ok=True)

        for tif_path, meta in entries:
            # Read → (taper) → sharpen → (optionally) write
            img = image_io.read_tif(tif_path)

            color_mode = config.camera_mode == "color"

            # Color camera: auto WB + CA correction before sharpening (optional)
            if color_mode and img.ndim == 3 and config.wavelet_color_correct:
                img, _ = _auto_color_correct(img)

            # Border taper: cosine-fade outermost pixels before wavelet to
            # prevent stacking boundary gradients from being amplified.
            # For color images, compute taper widths from luminance.
            if config.wavelet.border_taper_px > 0:
                taper_src = img.mean(axis=2) if img.ndim == 3 else img
                t, b, l, r = wavelet.safe_taper_widths(taper_src, config.wavelet.border_taper_px)
                img = wavelet.border_taper(img, top=t, bottom=b, left=l, right=r)

            # Plain (non-disk-aware) wavelet: sharpen_disk_aware's circular feather
            # mask cuts into Jupiter's oblate equatorial bulge, over-blurring the
            # left/right disk regions.  The stacking TIF already has a smooth limb
            # fade, so plain sharpen does not produce limb ringing.
            if color_mode:
                sharpened = wavelet.sharpen_color(
                    img,
                    levels=config.wavelet.levels,
                    amounts=config.wavelet.preview_amounts,
                    power=config.wavelet.preview_power,
                    sharpen_filter=config.wavelet.preview_sharpen_filter,
                )
            else:
                sharpened = wavelet.sharpen(
                    img,
                    levels=config.wavelet.levels,
                    amounts=config.wavelet.preview_amounts,
                    power=config.wavelet.preview_power,
                    sharpen_filter=config.wavelet.preview_sharpen_filter,
                )

            out_path: Optional[Path] = None
            if filter_out_dir is not None:
                out_path = filter_out_dir / (meta["stem"] + "_wavelet.png")
                # Raw 16-bit output — preserves original histogram shape.
                # No auto-stretch: mean brightness is unchanged by sharpening.
                if color_mode:
                    image_io.write_png_color_16bit(sharpened, out_path)
                else:
                    image_io.write_png_16bit(sharpened, out_path)

            results[filter_name].append((out_path, meta))

            done += 1
            print(
                f"\r  [{done:>3}/{total}] {filter_name:>4}: {tif_path.name}",
                end="",
                flush=True,
            )
            if progress_callback is not None:
                progress_callback(done, total)

        print()   # newline after each filter

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n  Step 7 complete: {done} files processed")
    if out_dir is not None:
        per_filter = ", ".join(
            f"{f}×{len(v)}" for f, v in sorted(results.items())
        )
        print(f"  Saved: {per_filter}")

    return results
