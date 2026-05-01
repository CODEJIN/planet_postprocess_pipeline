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

import json
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
        # Windows system fonts
        "C:/Windows/Fonts/malgun.ttf",    # Malgun Gothic — Korean support
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10
    except TypeError:
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

def _read_step03_window(config: "PipelineConfig", win_label: str) -> Optional[dict]:
    """Return the step03 window dict for win_label, or None if unavailable."""
    path = config.step_dir(3, "quality") / "windows.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        win_idx = int(win_label.split("_")[1])
        for w in data.get("selected_windows", []):
            if w.get("window_index") == win_idx:
                return w
    except Exception:
        pass
    return None


def _read_step04_window(config: "PipelineConfig", win_label: str) -> Optional[dict]:
    """Return the step04 derotation_log dict for win_label, or None."""
    path = config.step_dir(4, "derotated") / win_label / "derotation_log.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_step06_composites(config: "PipelineConfig", win_label: str) -> Optional[dict]:
    """Return the step06 composite_log dict for win_label, or None."""
    path = config.step_dir(6, "rgb_composite") / win_label / "composite_log.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# ── Analytic layout helpers ───────────────────────────────────────────────────

# Canonical channel order for align table rows
_CHANNEL_ORDER = ["L", "R", "G", "B", "IR", "CH4"]


def _collect_align_channels(data06: Optional[dict]) -> List[str]:
    """Return unique alignment channels across all composites, in canonical order."""
    if not data06:
        return []
    seen: Dict[str, bool] = {}
    for cdata in data06.get("composites", {}).values():
        for ch in cdata.get("alignment", {}).keys():
            seen[ch] = True
    order = {ch: i for i, ch in enumerate(_CHANNEL_ORDER)}
    return sorted(seen.keys(), key=lambda c: order.get(c, 99))


def _filter_stats_height(
    data03: Optional[dict],
    data04: Optional[dict],
    row_h: int,
) -> int:
    """Pixel height of the filter-stats block drawn above the section divider."""
    if not data03 and not data04:
        return 0
    n = 0
    if data03:
        n += 3   # Frames, Q.Post, Stab.
    if data04:
        n += 1   # Stacked
    return 8 + n * (row_h + 2)


def _draw_filter_stats(
    draw: "ImageDraw.Draw",
    y: int,
    filter_names: List[str],
    filter_x0: int,
    filter_px: int,
    gap: int,
    data03: Optional[dict],
    data04: Optional[dict],
    small_font: "ImageFont.ImageFont",
) -> int:
    """Draw filter-stats table (aligned to filter columns). Returns updated y."""
    if not data03 and not data04:
        return y

    label_color = (200, 200, 200)
    dim_color   = (120, 120, 120)
    row_h       = _text_size(draw, "Ag", small_font)[1] + 2

    def _cx(ci: int) -> int:
        return filter_x0 + ci * (filter_px + gap) + filter_px // 2

    def _row_lbl(label: str, ry: int) -> None:
        tw, _ = _text_size(draw, label, small_font)
        draw.text((filter_x0 - tw - 8, ry), label, fill=dim_color, font=small_font)

    def _vals(vs: List[str], ry: int) -> None:
        for ci, v in enumerate(vs):
            tw, _ = _text_size(draw, v, small_font)
            draw.text((_cx(ci) - tw // 2, ry), v, fill=label_color, font=small_font)

    y += 8  # top pad

    if data03:
        pf = data03.get("per_filter", {})
        _row_lbl("Frames", y)
        _vals([f"{pf.get(f,{}).get('n_included','?')}/{pf.get(f,{}).get('n_total','?')}"
               for f in filter_names], y)
        y += row_h + 2

        _row_lbl("Q.Post", y)
        _vals([f"{pf.get(f,{}).get('quality_post',float('nan')):.3f}" if f in pf else "—"
               for f in filter_names], y)
        y += row_h + 2

        _row_lbl("Stab.", y)
        _vals([f"{pf.get(f,{}).get('stability',float('nan')):.3f}" if f in pf else "—"
               for f in filter_names], y)
        y += row_h + 2

    if data04:
        filters04 = data04.get("filters", {})
        _row_lbl("Stacked", y)
        _vals([str(filters04.get(f, {}).get("n_stacked", "—")) for f in filter_names], y)
        y += row_h + 2

    return y


def _align_params_height(
    filter_names: List[str],
    data03: Optional[dict],
    data04: Optional[dict],
    data06: Optional[dict],
    row_h: int,
) -> int:
    """Pixel height of the align-table + separator + global-params block."""
    if not data03 and not data04 and not data06:
        return 0
    h = 0
    if data06 and filter_names:
        has_sat = any(cd.get("saturation_gain")
                      for cd in data06.get("composites", {}).values())
        n = len(filter_names) + (1 if has_sat else 0)
        if n:
            h += 8 + n * (row_h + 2)  # 8 pad + filter rows + optional Sat row
    # separator + global params line
    h += 16 + (row_h + 2) + 8
    return h


def _draw_align_params(
    draw: "ImageDraw.Draw",
    y: int,
    pad: int,
    canvas_w: int,
    filter_names: List[str],
    comp_names: List[str],
    comp_x0: int,
    composite_px: int,
    gap: int,
    spec_map: dict,
    data03: Optional[dict],
    data04: Optional[dict],
    data06: Optional[dict],
    config: "PipelineConfig",
    small_font: "ImageFont.ImageFont",
) -> None:
    """Draw filter×composite align table, then separator + global params.

    Rows are filter names (IR, R, G, B, CH4).  Each cell shows which channel
    role (L/R/G/B) the filter fills in that composite, plus the alignment shift:
        [L] ref     [R] +0.3,-0.1     —
    Filters not used in a composite show "—".
    """
    label_color = (200, 200, 200)
    dim_color   = (120, 120, 120)
    row_h       = _text_size(draw, "Ag", small_font)[1] + 2

    def _cx(i: int) -> int:
        return comp_x0 + i * (composite_px + gap) + composite_px // 2

    def _row_lbl(label: str, ry: int) -> None:
        tw, _ = _text_size(draw, label, small_font)
        draw.text((comp_x0 - tw - 8, ry), label, fill=dim_color, font=small_font)

    # ── Align table (filter rows × composite columns) ─────────────────────────
    if data06 and filter_names:
        composites06 = data06.get("composites", {})
        has_sat      = any(cd.get("saturation_gain") for cd in composites06.values())

        y += 8

        for fname in filter_names:
            _row_lbl(fname, y)
            for i, cname in enumerate(comp_names):
                spec = spec_map.get(cname)
                # Find which channel role this filter fills in this composite
                role: Optional[str] = None
                if spec:
                    for r, fn in (("L", spec.L), ("R", spec.R),
                                  ("G", spec.G), ("B", spec.B)):
                        if fn == fname:
                            role = r
                            break

                if role is None:
                    val = "—"
                else:
                    # alignment keys are filter names (IR/R/G/B/CH4), not channel roles
                    shift = composites06.get(cname, {}).get("alignment", {}).get(fname)
                    if shift is None:
                        shift_str = "ref"   # reference channel has no shift entry
                    else:
                        dx, dy = shift[0], shift[1]
                        shift_str = "ref" if abs(dx) < 0.05 and abs(dy) < 0.05 \
                                    else f"{dx:+.1f},{dy:+.1f}"
                    val = f"[{role}] {shift_str}"

                tw, _ = _text_size(draw, val, small_font)
                draw.text((_cx(i) - tw // 2, y), val, fill=label_color, font=small_font)
            y += row_h + 2

        # Saturation row (composite-level, not per filter)
        if has_sat:
            _row_lbl("Sat", y)
            for i, cname in enumerate(comp_names):
                sat = composites06.get(cname, {}).get("saturation_gain")
                val = f"{sat:.2f}×" if sat else "—"
                tw, _ = _text_size(draw, val, small_font)
                draw.text((_cx(i) - tw // 2, y), val, fill=label_color, font=small_font)
            y += row_h + 2

    # ── Separator ─────────────────────────────────────────────────────────────
    y += 8
    draw.line([(pad, y), (canvas_w - pad, y)], fill=(55, 55, 55), width=1)
    y += 8

    # ── Global params line — centered ─────────────────────────────────────────
    parts: List[str] = []
    if data03:
        wq = data03.get("window_quality")
        if wq is not None:
            parts.append(f"Win.Q:{wq:.3f}")
        rot = data03.get("rotation_degrees")
        if rot is not None:
            parts.append(f"Rot:{rot:.1f}°")

    wvl = getattr(config.wavelet, "master_amounts", None)
    if wvl:
        fmt = [str(int(a)) if a == int(a) else f"{a:.1f}" for a in wvl]
        parts.append(f"Wvl:[{' '.join(fmt)}]")

    parts.append(f"bp={config.grid.black_point:.2f}  γ={config.grid.gamma:.2f}")

    gline = "    ".join(parts)
    gtw, _ = _text_size(draw, gline, small_font)
    draw.text((canvas_w // 2 - gtw // 2, y), gline, fill=label_color, font=small_font)


def _load_results05_from_disk(
    config: PipelineConfig,
) -> Dict[str, List[Tuple[Optional[Path], str]]]:
    """Rebuild a results_05-compatible dict by scanning the step05 output folder.

    Used when step05 was not run in the current session but its output already
    exists on disk from a previous run.

    Returns ``{window_label: [(png_path, filter_name), ...]}`` matching only
    filters listed in ``config.filters``.  Windows or filters with no file on
    disk are silently omitted.
    """
    step05_dir = config.step_dir(5, "wavelet_master")
    if not step05_dir.exists():
        return {}

    out: Dict[str, List[Tuple[Optional[Path], str]]] = {}
    for win_dir in sorted(step05_dir.iterdir()):
        if not win_dir.is_dir() or not win_dir.name.startswith("window_"):
            continue
        entries: List[Tuple[Optional[Path], str]] = []
        for fname in config.filters:
            p = win_dir / f"{fname}_master.png"
            if p.exists():
                entries.append((p, fname))
        if entries:
            out[win_dir.name] = entries
    return out


def _composite_formula(spec) -> str:
    """Return compact channel mapping string, e.g. 'L:IR  R:R  G:G  B:B'."""
    parts = []
    if spec.L:
        parts.append(f"L:{spec.L}")
    if spec.R:
        parts.append(f"R:{spec.R}")
    if spec.G:
        parts.append(f"G:{spec.G}")
    if spec.B:
        parts.append(f"B:{spec.B}")
    return "  ".join(parts)


def run(
    config: PipelineConfig,
    results_06: Dict[str, List[Tuple[Optional[Path], str]]],
    results_04: dict,
    results_05: Optional[Dict[str, List[Tuple[Optional[Path], str]]]] = None,
    cancel_event=None,
) -> Optional[Path]:
    """Build the summary contact sheet from Step 6 master composites.

    Args:
        config:      Pipeline configuration.
        results_06:  Output of step06_rgb_composite.run():
                     ``{window_label: [(composite_path_or_None, composite_name), ...]}``
        results_04:  Output of step04_derotate_stack.run() — used to look up
                     the center time of each window for row labels.
        results_05:  Output of step05_wavelet_master.run() — used by analytic
                     view to show individual filter images (mono mode only).

    Returns:
        Path to the saved PNG, or None if save_step10 is False or no data.
    """
    if not results_06:
        print("  [WARNING] No Step 6 results — Step 10 skipped.")
        return None

    # If results_05 was not passed (step05 skipped this session), try disk scan
    if config.camera_mode == "mono" and not results_05:
        results_05 = _load_results05_from_disk(config) or None

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
    top_px    = cfg.top_margin_px if has_title else 0
    gap       = cfg.gap_px
    left_px   = cfg.left_margin_px
    bottom_px = cfg.bottom_margin_px
    font      = _get_font(cfg.font_size)
    label_color = (210, 210, 210)

    out_dir: Optional[Path] = None
    if config.save_step10:
        out_dir = config.step_dir(10, "summary_grid")
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        print("  save_step10=False: grids not written to disk")

    def _draw_title(canvas: Image.Image, draw: ImageDraw.Draw, cw: int) -> None:
        if not has_title:
            return
        title_font = _get_font(cfg.title_font_size)
        tw, th = _text_size(draw, title_str, title_font)
        draw.text(
            (cw // 2 - tw // 2, top_px // 2 - th // 2),
            title_str, fill=(230, 230, 230), font=title_font,
        )

    def _draw_time_labels(
        canvas: Image.Image, draw: ImageDraw.Draw,
        content_top: int, row_step: int,
    ) -> None:
        for row_idx, label in enumerate(sorted_labels):
            t = _window_time_local(label)
            time_str = (
                t.strftime(cfg.time_format) if t != datetime.min
                else f"W{row_idx + 1}"
            )
            y_center = content_top + row_idx * row_step + cell_px // 2
            _draw_rotated_text(canvas, time_str, font, label_color, left_px // 2, y_center)

    # ── Simple grid: composites only (always saved) ───────────────────────────
    s_canvas_w = left_px + n_cols * cell_px + (n_cols - 1) * gap
    s_canvas_h = top_px + n_rows * cell_px + (n_rows - 1) * gap + bottom_px
    s_canvas   = Image.new("RGB", (s_canvas_w, s_canvas_h), (0, 0, 0))
    s_draw     = ImageDraw.Draw(s_canvas)

    _draw_title(s_canvas, s_draw, s_canvas_w)

    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            pil_cell = cells.get((row_idx, col_idx))
            if pil_cell is not None:
                s_canvas.paste(
                    pil_cell,
                    (left_px + col_idx * (cell_px + gap),
                     top_px  + row_idx * (cell_px + gap)),
                )

    _draw_time_labels(s_canvas, s_draw, top_px, cell_px + gap)

    label_y = top_px + n_rows * (cell_px + gap) - gap
    for col_idx, cname in enumerate(col_names):
        x_center = left_px + col_idx * (cell_px + gap) + cell_px // 2
        tw, th = _text_size(s_draw, cname, font)
        s_draw.text(
            (x_center - tw // 2, label_y + (bottom_px - th) // 2),
            cname, fill=label_color, font=font,
        )

    simple_path: Optional[Path] = None
    if out_dir is not None:
        simple_path = out_dir / "summary_grid_simple.png"
        s_canvas.save(str(simple_path), format="PNG")
        print(f"  → {simple_path.name}  ({s_canvas_w}×{s_canvas_h} px)")

    # ── Two-zone grid: composites (left) + filters (right) ───────────────────
    # Only for mono cameras when step05 filter images are available.
    two_zone_path: Optional[Path] = None
    if config.camera_mode == "mono" and bool(results_05):
        filter_names = config.filters
        n_filters    = len(filter_names)
        filter_px    = cell_px  # same size as composites

        filter_frame_map: Dict[str, Dict[str, Optional[Path]]] = {}
        for label, entries in results_05.items():
            filter_frame_map[label] = {fname: fpath for fpath, fname in entries}

        filter_cells: Dict[Tuple[int, int], Optional[Image.Image]] = {}
        for row_idx, label in enumerate(sorted_labels):
            for col_idx, fname in enumerate(filter_names):
                fpath = filter_frame_map.get(label, {}).get(fname)
                if fpath is None or not fpath.exists():
                    filter_cells[(row_idx, col_idx)] = None
                    continue
                try:
                    img = image_io.read_png(fpath)
                    img = _apply_levels(img, cfg.black_point, cfg.white_point, cfg.gamma)
                    filter_cells[(row_idx, col_idx)] = _float_to_pil(img, filter_px)
                except Exception as exc:
                    print(f"  [WARN] filter {fname}: {exc}")
                    filter_cells[(row_idx, col_idx)] = None

        formula_font_size = max(11, cfg.font_size - 5)
        formula_font      = _get_font(formula_font_size)
        formula_color     = (140, 140, 140)
        divider_color     = (55, 55, 55)
        col_header_h      = cfg.font_size + 8 + formula_font_size + 8
        zone_gap          = 20
        comp_zone_w   = n_cols * cell_px + max(0, n_cols - 1) * gap
        filter_zone_w = n_filters * filter_px + max(0, n_filters - 1) * gap
        tz_canvas_w   = left_px + comp_zone_w + zone_gap + filter_zone_w
        tz_canvas_h   = top_px + col_header_h + n_rows * cell_px + max(0, n_rows - 1) * gap + 10
        content_top   = top_px + col_header_h
        comp_x0       = left_px
        filter_x0     = left_px + comp_zone_w + zone_gap
        spec_map      = {s.name: s for s in config.composite.specs}

        tz_canvas = Image.new("RGB", (tz_canvas_w, tz_canvas_h), (0, 0, 0))
        tz_draw   = ImageDraw.Draw(tz_canvas)

        _draw_title(tz_canvas, tz_draw, tz_canvas_w)

        # Column headers — composite name + formula (left zone)
        for col_idx, cname in enumerate(col_names):
            x_center = comp_x0 + col_idx * (cell_px + gap) + cell_px // 2
            tw, th   = _text_size(tz_draw, cname, font)
            tz_draw.text((x_center - tw // 2, top_px + 6), cname, fill=label_color, font=font)
            spec = spec_map.get(cname)
            if spec:
                formula = _composite_formula(spec)
                tf, _   = _text_size(tz_draw, formula, formula_font)
                tz_draw.text(
                    (x_center - tf // 2, top_px + th + 10),
                    formula, fill=formula_color, font=formula_font,
                )

        # Column headers — filter names (right zone)
        for col_idx, fname in enumerate(filter_names):
            x_center = filter_x0 + col_idx * (filter_px + gap) + filter_px // 2
            tw, th   = _text_size(tz_draw, fname, font)
            tz_draw.text(
                (x_center - tw // 2, top_px + (col_header_h - th) // 2),
                fname, fill=label_color, font=font,
            )

        # Vertical divider
        div_x = left_px + comp_zone_w + zone_gap // 2
        tz_draw.line([(div_x, top_px + 4), (div_x, tz_canvas_h - 4)],
                     fill=divider_color, width=2)

        # Cells
        for row_idx in range(n_rows):
            row_y = content_top + row_idx * (cell_px + gap)
            for col_idx in range(n_cols):
                pil_cc = cells.get((row_idx, col_idx))
                if pil_cc is not None:
                    tz_canvas.paste(pil_cc, (comp_x0 + col_idx * (cell_px + gap), row_y))
            for col_idx in range(n_filters):
                pil_fc = filter_cells.get((row_idx, col_idx))
                if pil_fc is not None:
                    tz_canvas.paste(pil_fc, (filter_x0 + col_idx * (filter_px + gap), row_y))

        _draw_time_labels(tz_canvas, tz_draw, content_top, cell_px + gap)

        if out_dir is not None:
            two_zone_path = out_dir / "summary_grid.png"
            tz_canvas.save(str(two_zone_path), format="PNG")
            print(f"  → {two_zone_path.name}  ({tz_canvas_w}×{tz_canvas_h} px)")

    # ── Analytic view (mono only) ─────────────────────────────────────────────
    if config.camera_mode == "mono" and cfg.save_analytic and results_05:
        run_analytic(config, results_05, results_06, results_04, cancel_event=cancel_event)

    return two_zone_path or simple_path


# ── Analytic view ─────────────────────────────────────────────────────────────

def run_analytic(
    config: PipelineConfig,
    results_05: Dict[str, List[Tuple[Optional[Path], str]]],
    results_06: Dict[str, List[Tuple[Optional[Path], str]]],
    results_04: dict,
    cancel_event=None,
) -> List[Path]:
    """Build one per-window analytic PNG for mono cameras.

    Each PNG shows the individual filter images (from Step 5) in a top strip
    and the composite results (from Step 6) in a bottom strip, with the
    channel-mapping formula displayed above each composite.

    Output (when config.grid.save_analytic is True):
        <output_base>/step10_summary_grid/analytic/
            window_01_analytic.png
            window_02_analytic.png
            …

    Returns:
        List of Paths to the saved PNGs.
    """
    if not results_05:
        results_05 = _load_results05_from_disk(config)
    if not results_05 or not results_06:
        print("  [Analytic] No filter or composite results — analytic view skipped.")
        return []

    cfg = config.grid
    spec_map = {s.name: s for s in config.composite.specs}

    # ── Window time lookup ────────────────────────────────────────────────────
    window_times: Dict[str, str] = {}
    for w in results_04.get("windows", []):
        label = f"window_{w['window_index']:02d}"
        window_times[label] = w.get("center_time", "")

    local_offset = _local_utc_offset()

    # ── Output directory ──────────────────────────────────────────────────────
    out_dir: Optional[Path] = None
    if cfg.save_analytic and config.save_step10:
        out_dir = config.step_dir(10, "summary_grid") / "analytic"
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        print("  [Analytic] save disabled — analytic PNGs not written to disk.")
        return []

    # ── Layout constants ──────────────────────────────────────────────────────
    composite_px = cfg.cell_size_px if cfg.cell_size_px > 0 else 300
    filter_px    = composite_px  # same size as composites
    gap          = cfg.gap_px
    pad          = 20
    section_gap  = 18   # space + divider between filter and composite zones
    font         = _get_font(cfg.font_size)
    small_font   = _get_font(max(11, cfg.font_size - 5))
    title_font   = _get_font(cfg.title_font_size if cfg.title_font_size > 0 else cfg.font_size + 4)
    header_h     = cfg.top_margin_px
    filter_lbl_h = cfg.font_size + 10
    comp_lbl_h   = cfg.font_size + 10  # name only

    # Pre-measure the widest row-label so the canvas has enough left margin
    _probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    label_margin = max(
        _text_size(_probe, lbl, small_font)[0]
        for lbl in ("Frames", "Q.Post", "Stab.", "Stacked")
    ) + 12

    label_color   = (210, 210, 210)
    divider_color = (55, 55, 55)

    saved: List[Path] = []

    for win_label in sorted(results_05.keys()):
        if cancel_event and cancel_event.is_set():
            break

        filter_entries  = results_05.get(win_label, [])   # [(path, fname), ...]
        comp_entries    = results_06.get(win_label, [])   # [(path, cname), ...]

        if not filter_entries and not comp_entries:
            continue

        # ── Read step03/04/06 metadata from disk ──────────────────────────────
        data03 = _read_step03_window(config, win_label)
        data04 = _read_step04_window(config, win_label)
        data06 = _read_step06_composites(config, win_label)

        filter_names_list = [fname for _, fname in filter_entries]
        comp_names_list   = [cname for _, cname in comp_entries]

        N_f = len(filter_entries)
        N_c = len(comp_entries)

        filter_total_w = N_f * filter_px + max(0, N_f - 1) * gap if N_f else 0
        comp_total_w   = N_c * composite_px + max(0, N_c - 1) * gap if N_c else 0
        content_w      = max(filter_total_w, comp_total_w, 1)
        canvas_w       = label_margin + 2 * pad + content_w

        # Pre-calculate block heights so canvas is sized correctly
        _probe_rh = _text_size(ImageDraw.Draw(Image.new("RGB", (1, 1))), "Ag", small_font)[1] + 2
        fstats_h     = _filter_stats_height(data03, data04, _probe_rh)
        apar_h       = _align_params_height(filter_names_list, data03, data04, data06, _probe_rh)

        canvas_h = (pad + header_h
                    + filter_lbl_h + filter_px
                    + fstats_h
                    + section_gap
                    + comp_lbl_h + composite_px
                    + apar_h
                    + pad)

        canvas = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
        draw   = ImageDraw.Draw(canvas)

        # ── Header ────────────────────────────────────────────────────────────
        iso = window_times.get(win_label, "")
        local_time_str = ""
        if iso:
            try:
                t_utc = datetime.strptime(iso[:16], "%Y-%m-%dT%H:%M")
                t_loc = t_utc + local_offset
                local_time_str = t_loc.strftime(cfg.time_format)
            except ValueError:
                pass

        total_sec   = int(local_offset.total_seconds())
        sign        = "+" if total_sec >= 0 else "-"
        hh, mm_rem  = divmod(abs(total_sec) // 60, 60)
        tz_label    = f"UTC{sign}{hh:02d}{mm_rem:02d}"
        header_parts = [config.target]
        if local_time_str:
            header_parts.append(local_time_str)
        header_parts.append(tz_label)
        header_str = "  ·  ".join(header_parts)

        tw, th = _text_size(draw, header_str, title_font)
        draw.text(
            (canvas_w // 2 - tw // 2, pad + (header_h - th) // 2),
            header_str, fill=(230, 230, 230), font=title_font,
        )

        y = pad + header_h

        # ── Filter labels ─────────────────────────────────────────────────────
        filter_x0 = label_margin + pad + (content_w - filter_total_w) // 2
        for i, (_, fname) in enumerate(filter_entries):
            x_center = filter_x0 + i * (filter_px + gap) + filter_px // 2
            tw, th   = _text_size(draw, fname, font)
            draw.text(
                (x_center - tw // 2, y + (filter_lbl_h - th) // 2),
                fname, fill=label_color, font=font,
            )
        y += filter_lbl_h

        # ── Filter images ─────────────────────────────────────────────────────
        for i, (fpath, _) in enumerate(filter_entries):
            x = filter_x0 + i * (filter_px + gap)
            if fpath is not None and fpath.exists():
                try:
                    img = image_io.read_png(fpath)
                    img = _apply_levels(img, cfg.black_point, cfg.white_point, cfg.gamma)
                    canvas.paste(_float_to_pil(img, filter_px), (x, y))
                except Exception as exc:
                    print(f"  [Analytic][WARN] {fpath.name}: {exc}")
        y += filter_px

        # ── Filter stats (Frames / Q.Post / Stab. / Stacked) ─────────────────
        y = _draw_filter_stats(
            draw, y, filter_names_list, filter_x0, filter_px, gap,
            data03, data04, small_font,
        )

        # ── Divider (filter / composite) ──────────────────────────────────────
        div_y = y + section_gap // 2
        draw.line([(pad, div_y), (canvas_w - pad, div_y)], fill=divider_color, width=2)
        y += section_gap

        # ── Composite labels (name only — channel mapping shown in align table) ──
        comp_x0 = label_margin + pad + (content_w - comp_total_w) // 2
        for i, (_, cname) in enumerate(comp_entries):
            x_center = comp_x0 + i * (composite_px + gap) + composite_px // 2
            tw, th = _text_size(draw, cname, font)
            draw.text(
                (x_center - tw // 2, y + (comp_lbl_h - th) // 2),
                cname, fill=label_color, font=font,
            )
        y += comp_lbl_h

        # ── Composite images ──────────────────────────────────────────────────
        for i, (cpath, _) in enumerate(comp_entries):
            x = comp_x0 + i * (composite_px + gap)
            if cpath is not None and cpath.exists():
                try:
                    img = image_io.read_png(cpath)
                    img = _apply_levels(img, cfg.black_point, cfg.white_point, cfg.gamma)
                    canvas.paste(_float_to_pil(img, composite_px), (x, y))
                except Exception as exc:
                    print(f"  [Analytic][WARN] {cpath.name}: {exc}")
        y += composite_px

        # ── Align table + separator + global params ───────────────────────────
        if apar_h > 0:
            _draw_align_params(
                draw, y, pad, canvas_w,
                filter_names_list, comp_names_list, comp_x0, composite_px, gap,
                spec_map,
                data03, data04, data06, config, small_font,
            )

        # ── Save ──────────────────────────────────────────────────────────────
        out_path = out_dir / f"{win_label}_analytic.png"
        canvas.save(str(out_path), format="PNG")
        print(f"  [Analytic] → {out_path.name}  ({canvas_w}×{canvas_h} px)")
        saved.append(out_path)

    return saved
