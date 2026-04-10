"""
Step 7 – RGB / LRGB compositing (master).

For each time window produced by Step 5/6, builds one composite image per
CompositeSpec defined in config.composite.specs.  Channels are auto-stretched
independently and then aligned to the reference channel before compositing.

Supported composite modes:
  - Plain RGB     (L=None in CompositeSpec)
  - LRGB          (L=filter_name; luminance replaces L in Lab colour space)
  - False colour  (any filter → any channel, no luminance blending)

Default composites (configurable in PipelineConfig):
  • RGB       → R, G, B
  • IR-RGB    → L=IR, R, G, B   (best sharpness: IR carries fine detail)
  • CH4-G-IR  → R=CH4, G=G, B=IR   (methane false colour)

Color camera mode:
  Per-window automatic white balance + chromatic aberration correction.
  Gains are computed from G-channel-relative disk medians; CA shift via
  phase correlation (cv2.phaseCorrelate, falls back gracefully if unavailable).

Output (when config.save_step07 is True):
    <output_base>/step07_rgb_composite/
        window_01/
            RGB_composite.png          (mono)
            IR-RGB_composite.png       (mono)
            CH4-G-IR_composite.png     (mono)
            composite_log.json         (mono)
            COLOR_composite.png        (color)
        window_02/
            …
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from pipeline.config import PipelineConfig
from pipeline.modules import composite, image_io
from pipeline.modules.derotation import apply_shift


# ── Color camera helpers ───────────────────────────────────────────────────────

def _auto_color_correct(
    img: np.ndarray,
) -> Tuple[np.ndarray, dict]:
    """Per-image automatic white balance + chromatic aberration correction.

    Algorithm:
        1. Detect planet disk with find_disk_center(); fall back to full frame.
        2. White balance: r_gain = G_median / R_median on disk pixels.
        3. CA shift: phaseCorrelate(G, R) and phaseCorrelate(G, B) on disk ROI.
           phaseCorrelate(G, R) returns (dx, dy) such that R displaced by (dx,dy)
           from G.  To realign: apply_shift(R, -dx, -dy).
        4. Apply gain then sub-pixel shift to R and B.

    Args:
        img: float32 [0, 1] array, shape (H, W, 3).

    Returns:
        (corrected_float32, params_dict) where params_dict keys:
            r_gain, b_gain,
            r_shift_x, r_shift_y, b_shift_x, b_shift_y
    """
    from pipeline.modules.derotation import find_disk_center

    if img.ndim == 2:
        img = np.stack([img] * 3, axis=2)

    r = img[:, :, 0].astype(np.float32)
    g = img[:, :, 1].astype(np.float32)
    b = img[:, :, 2].astype(np.float32)

    # ── Disk mask ──────────────────────────────────────────────────────────────
    try:
        cx, cy, sr, _, _ = find_disk_center(g)
        H, W = g.shape
        yy, xx = np.ogrid[:H, :W]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= sr ** 2
        disk_ok = bool(mask.sum() > 100)
    except Exception:
        mask = np.ones(g.shape, dtype=bool)
        cx = cy = sr = 0.0
        disk_ok = False

    # ── White balance from disk medians ────────────────────────────────────────
    r_med = float(np.median(r[mask]))
    g_med = float(np.median(g[mask]))
    b_med = float(np.median(b[mask]))

    r_gain = float(np.clip(g_med / r_med, 0.5, 3.0)) if r_med > 1e-6 else 1.0
    b_gain = float(np.clip(g_med / b_med, 0.5, 3.0)) if b_med > 1e-6 else 1.0

    # ── CA shift via phase correlation ─────────────────────────────────────────
    r_sx = r_sy = b_sx = b_sy = 0.0
    try:
        import cv2

        if disk_ok and sr >= 10:
            ys = int(max(0, cy - sr))
            ye = int(min(g.shape[0], cy + sr))
            xs = int(max(0, cx - sr))
            xe = int(min(g.shape[1], cx + sr))
            g64 = g[ys:ye, xs:xe].astype(np.float64)
            r64 = r[ys:ye, xs:xe].astype(np.float64)
            b64 = b[ys:ye, xs:xe].astype(np.float64)
        else:
            g64 = g.astype(np.float64)
            r64 = r.astype(np.float64)
            b64 = b.astype(np.float64)

        # phaseCorrelate(G, R) = (dx, dy): R is displaced (dx,dy) from G.
        # To align R → G: apply_shift(R, -dx, -dy).
        (dx_r, dy_r), _ = cv2.phaseCorrelate(g64, r64)
        (dx_b, dy_b), _ = cv2.phaseCorrelate(g64, b64)

        r_sx = float(np.clip(-dx_r, -20.0, 20.0))
        r_sy = float(np.clip(-dy_r, -20.0, 20.0))
        b_sx = float(np.clip(-dx_b, -20.0, 20.0))
        b_sy = float(np.clip(-dy_b, -20.0, 20.0))
    except ImportError:
        pass   # cv2 not available — CA shift stays 0.0

    # ── Apply correction ───────────────────────────────────────────────────────
    out = img.astype(np.float64)
    out[:, :, 0] *= r_gain
    out[:, :, 2] *= b_gain
    out = np.clip(out, 0.0, 1.0).astype(np.float32)

    if r_sx != 0.0 or r_sy != 0.0:
        out[:, :, 0] = apply_shift(out[:, :, 0], r_sx, r_sy)
    if b_sx != 0.0 or b_sy != 0.0:
        out[:, :, 2] = apply_shift(out[:, :, 2], b_sx, b_sy)

    params = {
        "r_gain":    r_gain,
        "b_gain":    b_gain,
        "r_shift_x": r_sx,
        "r_shift_y": r_sy,
        "b_shift_x": b_sx,
        "b_shift_y": b_sy,
    }
    return out, params


def _color_passthrough(
    config: PipelineConfig,
    results_06: Dict[str, List[Tuple[Optional[Path], str]]],
) -> Dict[str, List[Tuple[Optional[Path], str]]]:
    """Color-camera Step 7: per-window automatic WB + CA correction.

    Runs _auto_color_correct() independently for each time window so that
    varying atmospheric conditions across the observation session are handled
    without any fixed global correction values.
    """
    out_base: Optional[Path] = None
    if config.save_step07:
        out_base = config.step_dir(7, "rgb_composite")
        out_base.mkdir(parents=True, exist_ok=True)
        print(f"  Output → {out_base}")
    else:
        print("  save_step07=False: color results kept at Step 6 paths")

    all_results: Dict[str, List[Tuple[Optional[Path], str]]] = {}

    for win_label, entries in sorted(results_06.items()):
        src_path: Optional[Path] = None
        for p, _ in entries:
            if p is not None and p.exists():
                src_path = p
                break

        if src_path is None:
            print(f"  [{win_label}] No Step 6 output found — skipped")
            all_results[win_label] = [(None, "COLOR")]
            continue

        out_path: Optional[Path] = src_path

        if out_base is not None:
            win_out_dir = out_base / win_label
            win_out_dir.mkdir(exist_ok=True)
            out_path = win_out_dir / "COLOR_composite.png"

            img = image_io.read_png(src_path)
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=2)

            corrected, params = _auto_color_correct(img)
            image_io.write_png_color_16bit(corrected, out_path)

            print(
                f"  [{win_label}] COLOR → {out_path.name}  "
                f"R×{params['r_gain']:.3f} B×{params['b_gain']:.3f}  "
                f"R_shift=({params['r_shift_x']:+.2f},{params['r_shift_y']:+.2f})  "
                f"B_shift=({params['b_shift_x']:+.2f},{params['b_shift_y']:+.2f})"
            )
        else:
            print(f"  [{win_label}] COLOR → {out_path.name if out_path else '(not saved)'}")

        all_results[win_label] = [(out_path, "COLOR")]

    return all_results


def run(
    config: PipelineConfig,
    results_06: Dict[str, List[Tuple[Optional[Path], str]]],
) -> Dict[str, List[Tuple[Optional[Path], str]]]:
    """Run Step 7 for all windows produced by Step 6.

    Args:
        config:      Pipeline configuration (composite specs in config.composite).
        results_06:  Output of step06_wavelet_master.run():
                     ``{window_label: [(png_path, filter_name), ...]}``

    Returns:
        ``{window_label: [(composite_path_or_None, composite_name), ...]}``
    """
    # Color camera: auto WB + CA correction per window — no compositing
    if config.camera_mode == "color":
        print("  Color camera mode: auto white balance + CA correction per window")
        return _color_passthrough(config, results_06)

    if not results_06:
        print("  [WARNING] No Step 6 results — Step 7 skipped.")
        return {}

    # ── Output directory ───────────────────────────────────────────────────────
    out_base: Optional[Path] = None
    if config.save_step07:
        out_base = config.step_dir(7, "rgb_composite")
        out_base.mkdir(parents=True, exist_ok=True)
        print(f"  Output → {out_base}")
    else:
        print("  save_step07=False: results not written to disk")

    specs = config.composite.specs
    align = config.composite.align_channels
    plow  = config.composite.stretch_plow
    phigh = config.composite.stretch_phigh

    print(f"  Composites: {[s.name for s in specs]}")
    print(f"  Channel alignment: {'enabled' if align else 'disabled'}  "
          f"  Stretch: [{plow}%, {phigh}%]")

    all_results: Dict[str, List[Tuple[Optional[Path], str]]] = {}
    total_written = 0

    for win_label, filter_entries in sorted(results_06.items()):
        filter_paths: Dict[str, Optional[Path]] = {
            filt: path for path, filt in filter_entries
        }

        print(f"\n  {win_label}")

        win_out_dir: Optional[Path] = None
        if out_base is not None:
            win_out_dir = out_base / win_label
            win_out_dir.mkdir(exist_ok=True)

        win_results: List[Tuple[Optional[Path], str]] = []
        win_log: dict = {"composites": {}}

        for spec in specs:
            required = {spec.R, spec.G, spec.B}
            if spec.L is not None:
                required.add(spec.L)

            unavailable = {
                f for f in required
                if filter_paths.get(f) is None or not filter_paths[f].exists()
            }
            if unavailable:
                print(f"    [{spec.name}] Missing filters {unavailable} — skipped")
                win_results.append((None, spec.name))
                continue

            filter_images = {
                f: image_io.read_png(filter_paths[f])
                for f in required
            }
            for f in list(filter_images.keys()):
                img = filter_images[f]
                if img.ndim == 3:
                    filter_images[f] = img.mean(axis=2).astype("float32")

            try:
                comp_img, log = composite.compose(
                    spec, filter_images,
                    align=align,
                    max_shift_px=config.composite.max_shift_px,
                    color_stretch_mode=config.composite.color_stretch_mode,
                    stretch_plow=plow,
                    stretch_phigh=phigh,
                )
            except Exception as exc:
                print(f"    [{spec.name}] ERROR: {exc}")
                win_results.append((None, spec.name))
                continue

            out_path: Optional[Path] = None
            if win_out_dir is not None:
                out_path = win_out_dir / f"{spec.name}_composite.png"
                image_io.write_png_color_16bit(comp_img, out_path)
                total_written += 1

            win_log["composites"][spec.name] = log
            win_results.append((out_path, spec.name))
            status = f"→ {out_path.name}" if out_path else "(not saved)"
            print(f"    [{spec.name}] {status}")

        if win_out_dir is not None:
            log_path = win_out_dir / "composite_log.json"
            with open(log_path, "w") as f:
                json.dump(win_log, f, indent=2)

        all_results[win_label] = win_results

    print(f"\n  Step 7 complete: {total_written} composite PNGs written")
    return all_results
