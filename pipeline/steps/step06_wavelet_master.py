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

import numpy as np

from pipeline.config import PipelineConfig
from pipeline.modules import image_io, wavelet
from pipeline.modules.derotation import find_disk_center




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

            # Border taper: cosine-fade outermost pixels before wavelet to
            # prevent de-rotation stacking boundary gradients from being amplified.
            # Widths are clamped per-side to the actual background margin so
            # the taper never touches the planet disk even if off-centre.
            if config.wavelet.border_taper_px > 0:
                taper_src = img.mean(axis=2) if img.ndim == 3 else img
                t, b, l, r = wavelet.safe_taper_widths(taper_src, config.wavelet.border_taper_px)
                img = wavelet.border_taper(img, top=t, bottom=b, left=l, right=r)

            # Elliptical disk-aware sharpening: feather zone follows Jupiter's
            # actual oblate ellipse (semi-major=equatorial, semi-minor=polar),
            # preventing over-blur at the equatorial limb while still suppressing
            # ringing from de-rotation coverage gradients at the disk boundary.
            _lum = img.mean(axis=2) if img.ndim == 3 else img
            try:
                _cx, _cy, _rx, _ry, _angle = find_disk_center(_lum)
                _has_disk = _rx >= 5
            except Exception:
                _has_disk = False

            if _has_disk:
                # find_disk_center returns angle in degrees; convert to radians
                _angle_rad = np.radians(_angle)

                # Auto-estimate eff and expand_px from image data if requested
                if config.wavelet.auto_params:
                    _lum_auto = img.mean(axis=2) if img.ndim == 3 else img
                    _use_eff, _use_expand = wavelet.auto_wavelet_params(
                        _lum_auto, _cx, _cy, _rx, _ry, _angle_rad
                    )
                    print(f"    [{filt}] auto params: eff={_use_eff} "
                          f"expand_px={_use_expand}")
                else:
                    _use_eff    = config.wavelet.edge_feather_factor
                    _use_expand = config.wavelet.disk_expand_px

                if color_mode:
                    sharpened = wavelet.sharpen_color_disk_aware(
                        img, _cx, _cy, _rx,
                        levels=config.wavelet.levels,
                        amounts=config.wavelet.master_amounts,
                        power=config.wavelet.master_power,
                        sharpen_filter=config.wavelet.master_sharpen_filter,
                        edge_feather_factor=_use_eff,
                        ry=_ry, angle=_angle_rad,
                        expand_px=_use_expand,
                    )
                else:
                    sharpened = wavelet.sharpen_disk_aware(
                        img, _cx, _cy, _rx,
                        levels=config.wavelet.levels,
                        amounts=config.wavelet.master_amounts,
                        power=config.wavelet.master_power,
                        sharpen_filter=config.wavelet.master_sharpen_filter,
                        edge_feather_factor=_use_eff,
                        ry=_ry, angle=_angle_rad,
                        expand_px=_use_expand,
                    )
                print(f"    [{filt}] ellipse rx={_rx:.1f} ry={_ry:.1f} angle={_angle:.1f}°")
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
