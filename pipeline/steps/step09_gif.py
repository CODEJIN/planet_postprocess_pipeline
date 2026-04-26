"""
Step 9 – Animated GIF.

Assembles the time-ordered series of composite PNGs from Step 8 into one
animated GIF per composite type (RGB, IR-RGB, CH4-G-IR, …).

GIF output:  per-frame Floyd-Steinberg dithering (256-colour, improved quality)

One correction is applied before writing:

**Planet centering** (jitter fix):
   Each composite frame's planet disk is detected and shifted to the image
   centre.  Without this, the GIF flickers because the phase-correlation
   reference channel used inside Step 8's compose() may vary per frame,
   causing the composite planet to sit at slightly different sub-pixel
   positions even after per-filter centering.

Output (when config.save_step09 is True):
    <output_base>/step09_gif/
        RGB_animation.gif
        IR-RGB_animation.gif
        CH4-G-IR_animation.gif
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image as PILImage

from pipeline.config import PipelineConfig
from pipeline.modules import image_io


# ── Disk-based Step 8 result loader ───────────────────────────────────────────

def _load_step08_from_disk(
    config: PipelineConfig,
) -> Dict[str, List[tuple]]:
    """Reconstruct Step 8 results by scanning step08_series/ on disk.

    Used when step09 is run standalone (Step 8 results not in memory).
    Scans each frame_XXX_* directory for *_composite.png files.
    """
    step08_dir = config.step_dir(8, "series")
    if not step08_dir.exists():
        return {}

    results: Dict[str, List[tuple]] = {}
    for frame_dir in sorted(step08_dir.iterdir()):
        if not frame_dir.is_dir():
            continue
        frame_label = frame_dir.name
        frame_results = []
        for png in sorted(frame_dir.glob("*_composite.png")):
            comp_name = png.stem[: -len("_composite")]
            frame_results.append((png, comp_name))
        if frame_results:
            results[frame_label] = frame_results

    return results


# ── Helpers ────────────────────────────────────────────────────────────────────

def _center_all_frames(frames: List[np.ndarray]) -> List[np.ndarray]:
    """Center all frames using a temporally smoothed disk-center trajectory.

    Detects the planet disk in each frame, smooths the detected center positions
    across time with a moving median (window=5), then applies the smoothed
    correction to every frame.

    Why smoothing is necessary with multi-frame stacking (stack_window_n > 1):
    Stacked composites have blurry disk limbs (multiple de-rotated frames
    averaged together).  Per-frame Otsu thresholding on a blurry limb returns
    slightly different cx/cy each time.  Because consecutive stacked frames
    share (N-1)/N of their input sub-frames, the detection errors are
    correlated: they drift slowly then jump when a new sub-frame enters the
    window.  This correlated jitter causes the GRS to appear to oscillate while
    the surrounding bands move smoothly.  Smoothing the center trajectory before
    applying corrections eliminates this artifact.
    """
    from pipeline.modules.derotation import apply_shift, find_disk_center

    if not frames:
        return frames

    h, w = frames[0].shape[:2]
    cx_list: List[float] = []
    cy_list: List[float] = []

    for img in frames:
        # Luminance-weighted grayscale — reduces bias from saturated channels
        gray = (0.2126 * img[:, :, 0]
                + 0.7152 * img[:, :, 1]
                + 0.0722 * img[:, :, 2]).astype(np.float32)
        try:
            cx, cy, semi_a, *_ = find_disk_center(gray)
            if semi_a >= 5:
                cx_list.append(cx)
                cy_list.append(cy)
                continue
        except Exception:
            pass
        # Detection failed — use image centre as neutral fallback (no correction)
        cx_list.append(w * 0.5)
        cy_list.append(h * 0.5)

    # Moving-median smoothing (window=5, reflect padding at edges).
    # Median is robust to single-frame outliers (bad-seeing spikes);
    # window=5 suppresses per-frame jitter while preserving genuine slow drift.
    _WIN = 5
    _HALF = _WIN // 2
    n = len(cx_list)

    def _med_smooth(vals: List[float]) -> List[float]:
        return [
            float(np.median(vals[max(0, i - _HALF): min(n, i + _HALF + 1)]))
            for i in range(n)
        ]

    cx_smooth = _med_smooth(cx_list)
    cy_smooth = _med_smooth(cy_list)

    result: List[np.ndarray] = []
    for img, cx, cy in zip(frames, cx_smooth, cy_smooth):
        dx = w * 0.5 - cx
        dy = h * 0.5 - cy
        if abs(dx) < 0.05 and abs(dy) < 0.05:
            result.append(img)
            continue
        shifted = img.copy()
        for c in range(img.shape[2]):
            shifted[:, :, c] = apply_shift(img[:, :, c], dx, dy)
        result.append(shifted)
    return result


def _write_gif_dithered(
    frames: List[np.ndarray],
    path: Path,
    duration_ms: int,
    loop: int,
) -> None:
    """Write GIF with per-frame Floyd-Steinberg dithering (atomic write).

    Each frame is independently quantized to 256 colours with dithering,
    which significantly reduces colour banding compared to the default
    median-cut without dithering.

    The file is written to a sibling .tmp file first, then renamed to the
    final path so that a partial write never leaves a corrupt file behind.
    """
    pil_frames = [PILImage.fromarray(f) for f in frames]
    quantized = [
        f.quantize(colors=256, dither=PILImage.Dither.FLOYDSTEINBERG)
        for f in pil_frames
    ]
    tmp = path.with_suffix(".gif.tmp")
    try:
        quantized[0].save(
            str(tmp),
            format="GIF",
            save_all=True,
            append_images=quantized[1:],
            duration=duration_ms,
            loop=loop,
            optimize=False,
        )
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


# ── Main step ──────────────────────────────────────────────────────────────────

def run(
    config: PipelineConfig,
    results_08: Dict[str, List[Tuple[Optional[Path], str]]],
    progress_callback=None,
    cancel_event=None,
) -> Dict[str, Dict[str, Optional[Path]]]:
    """Build one GIF per composite type from Step 8 series.

    Args:
        config:      Pipeline configuration (gif params in config.gif).
        results_08:  Output of step08_series_composite.run():
                     ``{frame_label: [(png_path_or_None, composite_name), ...]}``

    Returns:
        ``{composite_name: {"gif": path_or_None}}``
    """
    if not results_08:
        print("  [INFO] Step 8 results not in memory — scanning disk...")
        results_08 = _load_step08_from_disk(config)

    if not results_08:
        print("  [WARNING] No Step 8 results found on disk — Step 9 skipped.")
        return {}

    # ── Collect per-composite frame lists (time-ordered) ──────────────────────
    composite_frames: Dict[str, List[Path]] = {}
    for frame_label in sorted(results_08.keys()):
        for png_path, comp_name in results_08[frame_label]:
            if png_path is not None and png_path.exists():
                composite_frames.setdefault(comp_name, []).append(png_path)

    if not composite_frames:
        print("  [WARNING] No composite PNGs found in Step 8 results.")
        return {}

    # ── Output directory ───────────────────────────────────────────────────────
    out_dir: Optional[Path] = None
    if config.save_step09:
        out_dir = config.step_dir(9, "gif")
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Output → {out_dir}")
    else:
        print("  save_step09=False: animations not written to disk")

    gif_cfg = config.gif
    duration_ms = int(1000.0 / gif_cfg.fps)
    resize = gif_cfg.resize_factor

    print(f"  FPS={gif_cfg.fps}  loop={'∞' if gif_cfg.loop == 0 else gif_cfg.loop}  "
          f"resize={resize}  centering=enabled  format=GIF(dithered)")

    results: Dict[str, Dict[str, Optional[Path]]] = {}
    failed: List[str] = []

    # Pre-compute total progress units: each frame load = 1, each GIF write = 1
    _sorted_composites = sorted(composite_frames.items())
    _total_frames = sum(len(fps) for _, fps in _sorted_composites)
    _n_composites = len(_sorted_composites)
    _total_units = _total_frames + _n_composites
    _current_unit = 0

    for comp_name, frame_paths in _sorted_composites:
        if cancel_event is not None and cancel_event.is_set():
            print("  [CANCELLED] Stopping Step 9.", flush=True)
            break
        n = len(frame_paths)
        print(f"\n  [{comp_name}]  {n} frames", end="", flush=True)

        try:
            # ── Load all frames ────────────────────────────────────────────────
            raw_frames: List[np.ndarray] = []
            for p in frame_paths:
                img = image_io.read_png(p)     # float [0,1], (H,W,3)
                if img.ndim == 2:
                    img = np.stack([img] * 3, axis=2)
                raw_frames.append(img.astype(np.float32))
                _current_unit += 1
                if progress_callback is not None:
                    progress_callback(_current_unit, _total_units)

            # ── Centre planet in every frame (temporally smoothed) ────────────
            # Smoothing suppresses correlated jitter from blurry-limb detection
            # on stacked composites (stack_window_n > 1).
            raw_frames = _center_all_frames(raw_frames)

            # ── Convert to uint8 ──────────────────────────────────────────────
            uint8_frames: List[np.ndarray] = []
            for f in raw_frames:
                arr8 = np.clip(f * 255, 0, 255).astype(np.uint8)
                if resize != 1.0:
                    import cv2
                    h, w = arr8.shape[:2]
                    arr8 = cv2.resize(
                        arr8,
                        (int(w * resize), int(h * resize)),
                        interpolation=cv2.INTER_LANCZOS4,
                    )
                uint8_frames.append(arr8)

            # ── Write GIF (dithered) ──────────────────────────────────────────
            gif_path: Optional[Path] = None

            if out_dir is not None:
                gif_path = out_dir / f"{comp_name}_animation.gif"
                _write_gif_dithered(uint8_frames, gif_path, duration_ms, gif_cfg.loop)
                gif_kb = gif_path.stat().st_size // 1024
                _current_unit += 1
                if progress_callback is not None:
                    progress_callback(_current_unit, _total_units)
                print(f"\n    GIF  → {gif_path.name}  ({gif_kb} KB)")
            else:
                _current_unit += 1  # skip write unit
                print()

            results[comp_name] = {"gif": gif_path}

        except Exception as exc:
            print(f"\n    [ERROR] {comp_name} 출력 실패: {exc}")
            failed.append(comp_name)
            results[comp_name] = {"gif": None}

    ok = len(results) - len(failed)
    print(f"\n  Step 9 complete: {ok}/{len(results)} composite types → {ok} GIF files")
    if failed:
        print(f"  [WARNING] 실패한 컴포짓: {', '.join(failed)}")
    return results
