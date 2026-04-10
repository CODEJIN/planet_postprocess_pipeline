"""
Step 6 – Wavelet sharpening (master).

Applies final wavelet sharpening to the de-rotated master TIFs produced by
Step 5.  Uses gentler parameters than the Step 3 preview (master_amounts
vs preview_amounts) to avoid over-sharpening the already high-SNR stacks.

One PNG is written per filter per time window.  These are the direct inputs
to Step 8 (RGB compositing).

Output (when config.save_step06 is True):
    <output_base>/step06_wavelet_master/
        window_01/
            IR_master.png
            R_master.png
            G_master.png
            B_master.png
            CH4_master.png
        window_02/
            …
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pipeline.config import PipelineConfig
from pipeline.modules import image_io, wavelet
from pipeline.modules.derotation import find_disk_center, find_visual_limb_radius


def run(
    config: PipelineConfig,
    results_05: dict,
) -> Dict[str, List[Tuple[Optional[Path], str]]]:
    """Run Step 6 for all windows produced by Step 5.

    Args:
        config:      Pipeline configuration.
        results_05:  Output of step05_derotate_stack.run(), containing:
                     ``{"windows": [{"window_index", "center_time",
                                     "outputs": {filter: Path|None}, ...}, ...]}``

    Returns:
        ``{window_label: [(png_path, filter_name), ...]}``
        *png_path* is None when ``config.save_step06`` is False.
    """
    windows: List[dict] = results_05.get("windows", [])
    if not windows:
        print("  [WARNING] No Step 5 windows — Step 6 skipped.")
        return {}

    # ── Output directory ───────────────────────────────────────────────────────
    out_base: Optional[Path] = None
    if config.save_step06:
        out_base = config.step_dir(6, "wavelet_master")
        out_base.mkdir(parents=True, exist_ok=True)
        print(f"  Output → {out_base}")
    else:
        print("  save_step06=False: results not written to disk")

    print(f"  Wavelet amounts: {config.wavelet.master_amounts}  "
          f"power={config.wavelet.master_power}  "
          f"sharpen_filter={config.wavelet.master_sharpen_filter}")

    results: Dict[str, List[Tuple[Optional[Path], str]]] = {}
    total_written = 0

    for win in windows:
        win_idx = win["window_index"]
        win_label = f"window_{win_idx:02d}"
        t_str = win["center_time"]
        outputs: Dict[str, Optional[Path]] = win.get("outputs", {})

        print(f"\n  {win_label}  [{t_str}]")

        # Per-window output directory
        win_out_dir: Optional[Path] = None
        if out_base is not None:
            win_out_dir = out_base / win_label
            win_out_dir.mkdir(exist_ok=True)

        win_results: List[Tuple[Optional[Path], str]] = []

        # For color mode the outputs key is the actual filter name from the
        # file ("RGB"), not config.filters ("COLOR").  Use the actual keys.
        iter_filters = list(outputs.keys()) if config.camera_mode == "color" else config.filters
        for filt in iter_filters:
            tif_path = outputs.get(filt)
            if tif_path is None or not tif_path.exists():
                print(f"    [{filt}] No input TIF — skipped")
                win_results.append((None, filt))
                continue

            img = image_io.read_tif(tif_path)
            color_mode = config.camera_mode == "color"

            # Detect disk center and Otsu semi-major radius.
            # find_disk_center() now reliably returns semi_major >= semi_minor,
            # so _sr is the equatorial (larger) radius regardless of ellipse angle.
            #
            # WHY use Otsu_r (not visual_r) for sharpen_disk_aware:
            # visual_r ≈ 122px (5% brightness threshold) is the physical disk edge,
            # but the wavelet weight at the PERCEIVED edge (~Otsu_r ≈ 103px) is
            # still 1.0 when radius=visual_r — full wavelet right where stacking
            # artifacts create an irregular limb gradient → ringing.
            # Using Otsu_r places the fade zone at the perceived edge:
            #   - Level 0 (finest): fade over 2px near Otsu_r → suppresses ringing
            #   - Level 1: 4px, Level 2: 8px — only active levels (amounts>0)
            #   - Levels 3-5: amounts=0, weight irrelevant
            # Maximum blurred zone: 8px at the very edge — acceptable.
            _lum = img.mean(axis=2) if img.ndim == 3 else img
            try:
                _cx, _cy, _sr, _, _ = find_disk_center(_lum)
                _has_disk = _sr >= 5
            except Exception:
                _has_disk = False

            # Border taper: cosine-fade outermost pixels before wavelet to
            # prevent de-rotation stacking boundary gradients from being amplified.
            # Widths are clamped per-side to the actual background margin so
            # the taper never touches the planet disk even if off-centre.
            if config.wavelet.border_taper_px > 0:
                taper_src = img.mean(axis=2) if img.ndim == 3 else img
                t, b, l, r = wavelet.safe_taper_widths(taper_src, config.wavelet.border_taper_px)
                img = wavelet.border_taper(img, top=t, bottom=b, left=l, right=r)

            # Disk-aware sharpening: each wavelet level L fades its contribution
            # over feather_L = 2^L × factor pixels near the Otsu disk edge.
            # The fade is applied to wavelet DETAIL coefficients, not the image
            # itself — no circular mask boundary is ever added to pixel data.
            if _has_disk:
                if color_mode:
                    sharpened = wavelet.sharpen_color_disk_aware(
                        img, _cx, _cy, _sr,
                        levels=config.wavelet.levels,
                        amounts=config.wavelet.master_amounts,
                        power=config.wavelet.master_power,
                        sharpen_filter=config.wavelet.master_sharpen_filter,
                        edge_feather_factor=config.wavelet.edge_feather_factor,
                    )
                else:
                    sharpened = wavelet.sharpen_disk_aware(
                        img, _cx, _cy, _sr,
                        levels=config.wavelet.levels,
                        amounts=config.wavelet.master_amounts,
                        power=config.wavelet.master_power,
                        sharpen_filter=config.wavelet.master_sharpen_filter,
                        edge_feather_factor=config.wavelet.edge_feather_factor,
                    )
                print(f"    [{filt}] Otsu_r={_sr:.1f}px  (disk-aware, Otsu boundary)")
            else:
                if color_mode:
                    sharpened = wavelet.sharpen_color(
                        img,
                        levels=config.wavelet.levels,
                        amounts=config.wavelet.master_amounts,
                        power=config.wavelet.master_power,
                        sharpen_filter=config.wavelet.master_sharpen_filter,
                    )
                else:
                    sharpened = wavelet.sharpen(
                        img,
                        levels=config.wavelet.levels,
                        amounts=config.wavelet.master_amounts,
                        power=config.wavelet.master_power,
                        sharpen_filter=config.wavelet.master_sharpen_filter,
                    )

            out_path: Optional[Path] = None
            if win_out_dir is not None:
                out_path = win_out_dir / f"{filt}_master.png"
                if color_mode:
                    image_io.write_png_color_16bit(sharpened, out_path)
                else:
                    image_io.write_png_16bit(sharpened, out_path)
                total_written += 1

            win_results.append((out_path, filt))
            status = f"→ {out_path.name}" if out_path else "(not saved)"
            print(f"    [{filt}] {status}")

        results[win_label] = win_results

    print(f"\n  Step 6 complete: {total_written} master PNGs written")
    return results
