"""
Step 6 – RGB / LRGB compositing (master).

For each time window produced by Step 4/5, builds one composite image per
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

Output (when config.save_step06 is True):
    <output_base>/step06_rgb_composite/
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
    results_05: Dict[str, List[Tuple[Optional[Path], str]]],
) -> Dict[str, List[Tuple[Optional[Path], str]]]:
    """Color-camera Step 6: per-window WB + CA correction with optional global norm.

    Two-pass when global_normalize is True:
      Pass 1 — correct each window, cache the corrected image (no stretch/saturation).
      Pass 2 — compute global mean luminance, apply scalar scale per window, then
               apply stretch and saturation, then save.
    """
    out_base: Optional[Path] = None
    if config.save_step06:
        out_base = config.step_dir(6, "rgb_composite")
        out_base.mkdir(parents=True, exist_ok=True)
        print(f"  Output → {out_base}")
    else:
        print("  save_step06=False: color results kept at Step 5 paths")

    # ── Pass 1: WB + CA correction, cache results ──────────────────────────────
    cache: Dict[str, Optional[np.ndarray]] = {}
    params_cache: Dict[str, dict] = {}
    src_paths: Dict[str, Optional[Path]] = {}

    for win_label, entries in sorted(results_05.items()):
        src_path: Optional[Path] = None
        for p, _ in entries:
            if p is not None and p.exists():
                src_path = p
                break
        src_paths[win_label] = src_path

        if src_path is None:
            cache[win_label] = None
            params_cache[win_label] = {}
            continue

        img = image_io.read_png(src_path)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=2)
        corrected, params = _auto_color_correct(img)
        cache[win_label] = corrected
        params_cache[win_label] = params

    # ── Global luminance normalization ─────────────────────────────────────────
    global_mean_lum: Optional[float] = None
    if config.composite.global_normalize:
        valid = [img.mean() for img in cache.values() if img is not None]
        if len(valid) > 1:
            global_mean_lum = float(np.mean(valid))

    # ── Pass 2: normalize + stretch + saturation + save ────────────────────────
    all_results: Dict[str, List[Tuple[Optional[Path], str]]] = {}

    for win_label in sorted(results_05.keys()):
        corrected = cache.get(win_label)
        params    = params_cache.get(win_label, {})
        src_path  = src_paths.get(win_label)

        if corrected is None:
            print(f"  [{win_label}] No Step 5 output found — skipped")
            all_results[win_label] = [(None, "COLOR")]
            continue

        out_path: Optional[Path] = src_path
        output = corrected.copy()

        if out_base is not None:
            win_out_dir = out_base / win_label
            win_out_dir.mkdir(exist_ok=True)
            out_path = win_out_dir / "COLOR_composite.png"

            if global_mean_lum is not None:
                frame_lum = float(output.mean())
                if frame_lum > 1e-6:
                    output = np.clip(output * (global_mean_lum / frame_lum), 0.0, 1.0).astype(np.float32)

            if config.composite.stretch_enabled:
                lo = float(np.percentile(output, 0.0))
                hi = float(np.percentile(output, 99.0))
                if hi > lo:
                    output = np.clip((output - lo) * (0.8 / (hi - lo)), 0.0, 1.0).astype(np.float32)

            if config.composite.saturation_boost:
                output = composite.auto_saturate(
                    output,
                    phigh=config.composite.saturation_phigh,
                    headroom=config.composite.saturation_headroom,
                )

            image_io.write_png_color_16bit(output, out_path)
            print(
                f"  [{win_label}] COLOR → {out_path.name}  "
                f"R×{params.get('r_gain', 1.0):.3f} B×{params.get('b_gain', 1.0):.3f}  "
                f"R_shift=({params.get('r_shift_x', 0.0):+.2f},{params.get('r_shift_y', 0.0):+.2f})  "
                f"B_shift=({params.get('b_shift_x', 0.0):+.2f},{params.get('b_shift_y', 0.0):+.2f})"
            )
        else:
            print(f"  [{win_label}] COLOR → {out_path.name if out_path else '(not saved)'}")

        all_results[win_label] = [(out_path, "COLOR")]

    return all_results


def run(
    config: PipelineConfig,
    results_05: Dict[str, List[Tuple[Optional[Path], str]]],
    cancel_event=None,
) -> Dict[str, List[Tuple[Optional[Path], str]]]:
    """Run Step 6 for all windows produced by Step 5.

    Args:
        config:      Pipeline configuration (composite specs in config.composite).
        results_05:  Output of step05_wavelet_master.run():
                     ``{window_label: [(png_path, filter_name), ...]}``

    Returns:
        ``{window_label: [(composite_path_or_None, composite_name), ...]}``
    """
    # Color camera: auto WB + CA correction per window — no compositing
    if config.camera_mode == "color":
        print("  Color camera mode: auto white balance + CA correction per window")
        return _color_passthrough(config, results_05)

    if not results_05:
        print("  [WARNING] No Step 5 results — Step 6 skipped.")
        return {}

    # ── Output directory ───────────────────────────────────────────────────────
    out_base: Optional[Path] = None
    if config.save_step06:
        out_base = config.step_dir(6, "rgb_composite")
        out_base.mkdir(parents=True, exist_ok=True)
        print(f"  Output → {out_base}")
    else:
        print("  save_step06=False: results not written to disk")

    specs        = config.composite.specs
    align        = config.composite.align_channels
    plow         = config.composite.stretch_plow
    phigh        = config.composite.stretch_phigh
    target_hi    = config.composite.stretch_target_hi
    saturate     = config.composite.saturation_boost
    sat_phigh    = config.composite.saturation_phigh
    sat_headroom = config.composite.saturation_headroom

    print(f"  Composites: {[s.name for s in specs]}")
    print(f"  Channel alignment: {'enabled' if align else 'disabled'}  "
          f"  Stretch: [{plow}%, {phigh}%] → {target_hi:.2f}  "
          f"  Saturation boost: {'p{:.0f}→{:.0f}%max'.format(sat_phigh, sat_headroom*100) if saturate else 'off'}  "
          f"  Global normalize: {'on' if config.composite.global_normalize else 'off'}")

    # ── Pass 1: compose all windows, cache results (saturation deferred) ───────
    # {win_label: {spec_name: comp_img or None}}
    cache: Dict[str, Dict[str, Optional[np.ndarray]]] = {}
    log_cache: Dict[str, dict] = {}
    cancelled = False

    for win_label, filter_entries in sorted(results_05.items()):
        if cancel_event is not None and cancel_event.is_set():
            print("  [CANCELLED] Stopping Step 6.", flush=True)
            cancelled = True
            break

        filter_paths: Dict[str, Optional[Path]] = {
            filt: path for path, filt in filter_entries
        }
        print(f"\n  {win_label}")

        if out_base is not None:
            (out_base / win_label).mkdir(exist_ok=True)

        cache[win_label] = {}
        log_cache[win_label] = {"composites": {}}

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
                cache[win_label][spec.name] = None
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
                    color_stretch_mode=config.composite.color_stretch_mode if config.composite.stretch_enabled else "none",
                    stretch_plow=plow,
                    stretch_phigh=phigh,
                    stretch_target_hi=target_hi,
                    saturate=False,   # saturation applied in pass 2 after global norm
                    saturation_phigh=sat_phigh,
                    saturation_headroom=sat_headroom,
                )
            except Exception as exc:
                print(f"    [{spec.name}] ERROR: {exc}")
                cache[win_label][spec.name] = None
                continue

            cache[win_label][spec.name] = comp_img
            log_cache[win_label]["composites"][spec.name] = log

    # ── Global luminance normalization: per composite spec across all windows ───
    # Normalises brightness of same-type composites so step10 grid rows are
    # visually consistent.  Only meaningful when multiple windows are present.
    global_means: Dict[str, Optional[float]] = {}
    if config.composite.global_normalize and len(cache) > 1:
        for spec in specs:
            vals = [
                float(win_comps[spec.name].mean())
                for win_comps in cache.values()
                if win_comps.get(spec.name) is not None
            ]
            global_means[spec.name] = float(np.mean(vals)) if len(vals) > 1 else None

    # ── Pass 2: normalize + saturation + save ──────────────────────────────────
    all_results: Dict[str, List[Tuple[Optional[Path], str]]] = {}
    total_written = 0

    for win_label, win_comps in sorted(cache.items()):
        win_out_dir: Optional[Path] = out_base / win_label if out_base is not None else None
        win_results: List[Tuple[Optional[Path], str]] = []

        for spec in specs:
            comp_img = win_comps.get(spec.name)
            if comp_img is None:
                win_results.append((None, spec.name))
                continue

            # Global luminance scale
            gm = global_means.get(spec.name)
            if gm is not None:
                frame_lum = float(comp_img.mean())
                if frame_lum > 1e-6:
                    comp_img = np.clip(comp_img * (gm / frame_lum), 0.0, 1.0).astype(np.float32)

            # Saturation boost (deferred from compose())
            if saturate:
                comp_img = composite.auto_saturate(comp_img, phigh=sat_phigh, headroom=sat_headroom)

            out_path: Optional[Path] = None
            if win_out_dir is not None:
                out_path = win_out_dir / f"{spec.name}_composite.png"
                image_io.write_png_color_16bit(comp_img, out_path)
                total_written += 1

            win_results.append((out_path, spec.name))
            status = f"→ {out_path.name}" if out_path else "(not saved)"
            print(f"    [{spec.name}] {status}")

        if win_out_dir is not None:
            log_path = win_out_dir / "composite_log.json"
            with open(log_path, "w") as f:
                json.dump(log_cache.get(win_label, {}), f, indent=2)

        all_results[win_label] = win_results

    print(f"\n  Step 6 complete: {total_written} composite PNGs written")
    return all_results
