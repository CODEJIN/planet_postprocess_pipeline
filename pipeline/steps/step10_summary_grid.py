"""
Step 10 – Summary contact sheet.

Creates a single PNG arranged as a grid of composite images:
  Rows    → time windows from Step 6, oldest at top
  Columns → composite types (RGB, IR-RGB, CH4-G-IR)

Each cell receives a levels adjustment (black_point / white_point / gamma)
to deepen the background blacks and enhance the visual depth of the planet.

Output (when config.save_step10 is True):
    <output_base>/step10_summary_grid/
        summary_grid.png
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pipeline.config import PipelineConfig
from pipeline.modules import image_io


# ── Helpers ───────────────────────────────────────────────────────────────────

def _apply_levels(
    img: np.ndarray,
    black_point: float,
    white_point: float,
    gamma: float = 1.0,
) -> np.ndarray:
    """Clip, stretch, and optionally gamma-correct an image.

    black_point clips dark values to pure black (deepens background);
    white_point clips bright values to pure white.
    gamma > 1 brightens midtones; gamma < 1 darkens them.
    """
    span = max(white_point - black_point, 1e-8)
    out = (img.clip(black_point, white_point) - black_point) / span
    if abs(gamma - 1.0) > 1e-6:
        out = out ** (1.0 / gamma)
    return out.clip(0.0, 1.0).astype(np.float32)


def _float_to_pil(img: np.ndarray, target_px: int) -> Image.Image:
    """Convert float [0, 1] array (H,W,3) or (H,W) to an 8-bit PIL RGB image.

    Resizes to target_px × target_px if target_px > 0.
    """
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=2)
    rgb8 = (img * 255.0).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(rgb8, mode="RGB")
    if target_px > 0 and (pil.width != target_px or pil.height != target_px):
        pil = pil.resize((target_px, target_px), Image.LANCZOS)
    return pil


def _get_font(size: int) -> ImageFont.ImageFont:
    """Load a TrueType font with broad Unicode coverage (incl. CJK/Korean).

    NotoSansCJK is tried first so that target names containing non-Latin
    characters (e.g. Korean) render correctly instead of showing boxes.
    """
    candidates = [
        # Noto CJK — full Unicode coverage including Korean, Chinese, Japanese
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        # DejaVu — good Latin/Greek/Cyrillic coverage
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _text_size(
    draw: ImageDraw.Draw,
    text: str,
    font: ImageFont.ImageFont,
) -> Tuple[int, int]:
    """Return (width, height) of *text* in pixels (Pillow ≥ 8 and older)."""
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        return draw.textsize(text, font=font)  # type: ignore[attr-defined]


def _local_utc_offset() -> timedelta:
    """Return the system's local UTC offset (e.g. +09:00 for KST)."""
    return datetime.now(timezone.utc).astimezone().utcoffset() or timedelta(0)


def _draw_rotated_text(
    canvas: Image.Image,
    text: str,
    font: ImageFont.ImageFont,
    color: Tuple[int, int, int],
    x_center: int,
    y_center: int,
) -> None:
    """Draw *text* rotated 90° CCW (reads bottom-to-top), centred at (x_center, y_center).

    Uses the exact textbbox bounds so no clipping occurs even when fonts have
    non-zero top offsets (which PIL's default textbbox origin can produce).
    """
    # Measure exact glyph bounds on a throw-away surface
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    try:
        bbox = probe.textbbox((0, 0), text, font=font)
        # bbox = (left, top, right, bottom) — top/left may be non-zero
        glyph_x0, glyph_y0, glyph_x1, glyph_y1 = bbox
    except AttributeError:
        w, h = probe.textsize(text, font=font)  # type: ignore[attr-defined]
        glyph_x0, glyph_y0, glyph_x1, glyph_y1 = 0, 0, w, h

    glyph_w = glyph_x1 - glyph_x0
    glyph_h = glyph_y1 - glyph_y0
    pad = 4

    # Render text into a black image sized exactly to the glyph + padding.
    # Draw at (-glyph_x0 + pad, -glyph_y0 + pad) so the glyph aligns to (pad, pad).
    tmp = Image.new("RGB", (glyph_w + pad * 2, glyph_h + pad * 2), (0, 0, 0))
    ImageDraw.Draw(tmp).text((-glyph_x0 + pad, -glyph_y0 + pad), text, fill=color, font=font)

    # Rotate 90° CCW (expand=True swaps width ↔ height)
    rotated = tmp.rotate(90, expand=True)

    # Paste centred at the requested position
    x = x_center - rotated.width // 2
    y = y_center - rotated.height // 2
    canvas.paste(rotated, (x, y))


# ── Main step ─────────────────────────────────────────────────────────────────

def run(
    config: PipelineConfig,
    results_06: Dict[str, List[Tuple[Optional[Path], str]]],
    results_04: dict,
) -> Optional[Path]:
    """Build the summary contact sheet from Step 6 master composites.

    Args:
        config:      Pipeline configuration.
        results_06:  Output of step06_rgb_composite.run():
                     ``{window_label: [(composite_path_or_None, composite_name), ...]}``
        results_04:  Output of step04_derotate_stack.run() — used to look up
                     the center time of each window for row labels.

    Returns:
        Path to the saved PNG, or None if save_step10 is False or no data.
    """
    if not results_06:
        print("  [WARNING] No Step 6 results — Step 10 skipped.")
        return None

    cfg = config.grid
    # Color camera: single column; override composite list from Step 8 keys
    if config.camera_mode == "color":
        # Collect all composite names actually present in results_06
        color_cols = sorted({name for pairs in results_06.values() for _, name in pairs})
        col_names = color_cols if color_cols else ["COLOR"]
    else:
        col_names = cfg.composites
    n_cols = len(col_names)

    # ── Build window_label → center_time lookup from Step 5 ──────────────────
    window_times: Dict[str, str] = {}
    for w in results_04.get("windows", []):
        label = f"window_{w['window_index']:02d}"
        window_times[label] = w.get("center_time", "")

    # Detect local UTC offset once for all labels
    local_offset = _local_utc_offset()

    # ── Sort windows by center time ───────────────────────────────────────────
    def _window_time_utc(label: str) -> datetime:
        iso = window_times.get(label, "")
        if iso:
            try:
                return datetime.strptime(iso[:16], "%Y-%m-%dT%H:%M")
            except ValueError:
                pass
        return datetime.min

    def _window_time_local(label: str) -> datetime:
        t = _window_time_utc(label)
        if t == datetime.min:
            return t
        return t + local_offset

    sorted_labels = sorted(results_06.keys(), key=_window_time_utc)
    n_rows = len(sorted_labels)

    if n_rows == 0:
        print("  [WARNING] No windows found — Step 10 skipped.")
        return None

    total_seconds = int(local_offset.total_seconds())
    offset_sign = "+" if total_seconds >= 0 else "-"
    offset_h, offset_m = divmod(abs(total_seconds) // 60, 60)
    print(f"  Grid: {n_rows} rows × {n_cols} cols")
    print(f"  Time: UTC{offset_sign}{offset_h:02d}:{offset_m:02d} (local)")
    print(f"  Levels: black_point={cfg.black_point}  white_point={cfg.white_point}"
          f"  gamma={cfg.gamma}")

    # ── Build lookup: window_label → {composite_name: Path} ──────────────────
    frame_map: Dict[str, Dict[str, Optional[Path]]] = {}
    for label, pairs in results_06.items():
        frame_map[label] = {name: path for path, name in pairs}

    # ── Detect native cell size from first available image ────────────────────
    native_size: Optional[int] = None
    for label in sorted_labels:
        for cname in col_names:
            p = frame_map.get(label, {}).get(cname)
            if p is not None and p.exists():
                try:
                    probe = image_io.read_png(p)
                    native_size = probe.shape[1]
                except Exception:
                    pass
                break
        if native_size is not None:
            break

    cell_px = cfg.cell_size_px if cfg.cell_size_px > 0 else (native_size or 300)

    # ── Load, process, and cache all cells ────────────────────────────────────
    cells: Dict[Tuple[int, int], Optional[Image.Image]] = {}
    n_missing = 0

    for row_idx, label in enumerate(sorted_labels):
        for col_idx, cname in enumerate(col_names):
            png_path = frame_map.get(label, {}).get(cname)
            if png_path is None or not png_path.exists():
                cells[(row_idx, col_idx)] = None
                n_missing += 1
                continue

            try:
                img = image_io.read_png(png_path)
            except Exception as exc:
                print(f"  [WARN] Cannot read {png_path.name}: {exc}")
                cells[(row_idx, col_idx)] = None
                n_missing += 1
                continue

            img = _apply_levels(img, cfg.black_point, cfg.white_point, cfg.gamma)
            cells[(row_idx, col_idx)] = _float_to_pil(img, cell_px)

    if n_missing:
        print(f"  {n_missing} cell(s) missing (shown as black)")

    # ── Build title string ────────────────────────────────────────────────────
    # Format: "Jupiter · 2026-04-02 · UTC+0900"
    # UTC offset is used instead of tz_name to avoid font issues on Windows
    # (tz_name can be a locale-specific string like "KST" requiring extra fonts).
    # Date is extracted from the first window's UTC center time.
    obs_date = ""
    first_iso = window_times.get(sorted_labels[0], "") if sorted_labels else ""
    if first_iso:
        try:
            obs_date = datetime.strptime(first_iso[:10], "%Y-%m-%d").strftime("%Y-%m-%d")
        except ValueError:
            pass
    tz_label = f"UTC{offset_sign}{offset_h:02d}{offset_m:02d}"
    title_parts = [config.target]
    if obs_date:
        title_parts.append(obs_date)
    title_parts.append(tz_label)
    title_str = "  ·  ".join(title_parts)

    has_title = cfg.title_font_size > 0
    top_px = cfg.top_margin_px if has_title else 0

    # ── Compose canvas ─────────────────────────────────────────────────────────
    gap       = cfg.gap_px
    left_px   = cfg.left_margin_px
    bottom_px = cfg.bottom_margin_px

    canvas_w = left_px + n_cols * cell_px + (n_cols - 1) * gap
    canvas_h = top_px + n_rows * cell_px + (n_rows - 1) * gap + bottom_px

    canvas = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
    draw   = ImageDraw.Draw(canvas)
    font   = _get_font(cfg.font_size)

    # ── Title ─────────────────────────────────────────────────────────────────
    if has_title:
        title_font  = _get_font(cfg.title_font_size)
        tw, th = _text_size(draw, title_str, title_font)
        draw.text(
            (canvas_w // 2 - tw // 2, top_px // 2 - th // 2),
            title_str,
            fill=(230, 230, 230),
            font=title_font,
        )

    # ── Paste cells ───────────────────────────────────────────────────────────
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            pil_cell = cells.get((row_idx, col_idx))
            x = left_px + col_idx * (cell_px + gap)
            y = top_px + row_idx * (cell_px + gap)
            if pil_cell is not None:
                canvas.paste(pil_cell, (x, y))

    # ── Time labels — rotated 90° CCW, centred per row ────────────────────────
    label_color = (210, 210, 210)
    for row_idx, label in enumerate(sorted_labels):
        t = _window_time_local(label)
        time_str = (
            t.strftime(cfg.time_format) if t != datetime.min
            else f"W{row_idx + 1}"
        )
        y_center = top_px + row_idx * (cell_px + gap) + cell_px // 2
        x_center = left_px // 2   # horizontally centred within left margin
        _draw_rotated_text(canvas, time_str, font, label_color, x_center, y_center)

    # ── Composite labels (bottom margin, centred per column) ──────────────────
    label_y = top_px + n_rows * (cell_px + gap) - gap  # top of bottom margin
    for col_idx, cname in enumerate(col_names):
        x_center = left_px + col_idx * (cell_px + gap) + cell_px // 2
        tw, th = _text_size(draw, cname, font)
        draw.text(
            (x_center - tw // 2, label_y + (bottom_px - th) // 2),
            cname,
            fill=label_color,
            font=font,
        )

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path: Optional[Path] = None
    if config.save_step10:
        out_dir = config.step_dir(10, "summary_grid")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "summary_grid.png"
        canvas.save(str(out_path), format="PNG")
        print(f"  → {out_path}  ({canvas_w}×{canvas_h} px)")
    else:
        print("  save_step10=False: grid not written to disk")

    return out_path
