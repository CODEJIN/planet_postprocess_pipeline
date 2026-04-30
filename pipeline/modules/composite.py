"""
RGB / LRGB compositing module.

Supported composite types (controlled by CompositeSpec in config):
  - RGB:      direct R/G/B channel combination
  - LRGB:     IR (or any filter) as luminance, RGB as colour (Lab space blend)
  - False colour: any filter-to-channel mapping (e.g. CH4→R, G→G, IR→B)

Channel alignment:
  All channels are aligned to the reference channel (first defined in the spec,
  or the L channel if present) via sub-pixel phase correlation before compositing.
  This corrects atmospheric dispersion and filter-wheel mechanical offsets.

Auto-stretch:
  Each input channel is independently auto-stretched using percentile
  normalisation before compositing.  This matches common practice in
  planetary imaging workflows (each filter has different sky background levels).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from pipeline.modules.derotation import subpixel_align, apply_shift, find_disk_center


# ── Per-channel stretch ────────────────────────────────────────────────────────

def auto_saturate(
    img: np.ndarray,
    phigh: float = 99.5,
    headroom: float = 0.30,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Auto-boost saturation in Lab space.

    Computes gain so that p(phigh) of disk chroma reaches headroom × 127
    (Lab max chroma).  gain is clamped to [1.0, 4.0] so the function never
    desaturates and never amplifies more than 4×.

    Args:
        img:      float32 [0,1] RGB (H,W,3).
        phigh:    Chroma percentile used as reference (mirrors stretch_phigh).
        headroom: Fraction of Lab max chroma (127) that p(phigh) should reach.
                  Data-adaptive: low-saturation images get more boost, high-
                  saturation images get less, without needing per-dataset tuning.
        mask:     Boolean mask selecting pixels to sample (e.g. disk interior).
    Returns:
        Saturation-boosted float32 [0,1] RGB image.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    a, b = lab[:, :, 1], lab[:, :, 2]
    chroma = np.sqrt(a ** 2 + b ** 2)

    sample = chroma[mask] if (mask is not None and mask.any()) else chroma.ravel()
    current = float(np.percentile(sample, phigh))
    if current < 1e-3:
        return img

    target = 127.0 * headroom
    gain = float(np.clip(target / current, 1.0, 2.0))
    lab[:, :, 1] = np.clip(a * gain, -127.0, 127.0)
    lab[:, :, 2] = np.clip(b * gain, -127.0, 127.0)
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0.0, 1.0).astype(np.float32)


def auto_stretch(
    img: np.ndarray,
    plow: float = 0.1,
    phigh: float = 99.9,
    target_hi: float = 1.0,
) -> np.ndarray:
    """Percentile-based linear stretch.  p(phigh) maps to target_hi; values above can reach 1.0."""
    lo, hi = np.percentile(img, [plow, phigh])
    if hi - lo < 1e-9:
        return np.zeros_like(img)
    scale = target_hi / (hi - lo)
    return np.clip((img - lo) * scale, 0.0, 1.0).astype(np.float32)


# ── Channel alignment ──────────────────────────────────────────────────────────

def align_channels(
    channels: Dict[str, np.ndarray],
    reference_key: str,
    max_shift_px: float = 0.0,
) -> Dict[str, np.ndarray]:
    """Align all channels to *reference_key* via sub-pixel phase correlation.

    Args:
        channels:      {filter_name: float [0,1] 2D image}
        reference_key: Key of the channel to treat as reference (no shift applied).
        max_shift_px:  If > 0, shifts larger than this are discarded (set to 0).
                       Prevents runaway shifts from low-SNR phase correlation.

    Returns:
        New dict with aligned images (reference unchanged, others shifted).
    """
    ref = channels[reference_key]
    aligned: Dict[str, np.ndarray] = {}
    for key, img in channels.items():
        if key == reference_key:
            aligned[key] = img
        else:
            dx, dy = subpixel_align(ref, img)
            if max_shift_px > 0 and (abs(dx) > max_shift_px or abs(dy) > max_shift_px):
                # Shift is unreasonably large — phase correlation failed; skip
                aligned[key] = img
            else:
                aligned[key] = apply_shift(img, dx, dy)
    return aligned


# ── RGB composite ──────────────────────────────────────────────────────────────

def make_rgb(
    r: np.ndarray,
    g: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """Stack R, G, B channels into an (H, W, 3) float [0, 1] RGB image."""
    return np.stack([r, g, b], axis=2).astype(np.float32)


# ── LRGB composite ─────────────────────────────────────────────────────────────

def make_lrgb(
    luminance: np.ndarray,
    r: np.ndarray,
    g: np.ndarray,
    b: np.ndarray,
    lrgb_weight: float = 1.0,
) -> np.ndarray:
    """LRGB composite in Lab colour space.

    Blends the luminance channel (e.g. IR) into the L channel of the RGB image.

    Args:
        luminance:    2-D float [0, 1] luminance image (e.g. IR filter).
        r, g, b:      2-D float [0, 1] colour channels.
        lrgb_weight:  Weight of the external luminance vs. the RGB's own L.
                      1.0 = fully replace with luminance, 0.0 = keep RGB L.

    Returns:
        (H, W, 3) float [0, 1] RGB image.
    """
    rgb = np.stack([r, g, b], axis=2).astype(np.float32)

    # Convert RGB → Lab  (cv2 float32 input expects [0,1]; L output is [0,100])
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)

    lum_100 = luminance.astype(np.float32) * 100.0
    lab[:, :, 0] = (lrgb_weight * lum_100
                    + (1.0 - lrgb_weight) * lab[:, :, 0])

    result = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
    return np.clip(result, 0.0, 1.0).astype(np.float32)


# ── High-level compose ─────────────────────────────────────────────────────────

def compose(
    spec,  # CompositeSpec (avoid circular import with config)
    filter_images: Dict[str, np.ndarray],
    align: bool = True,
    max_shift_px: float = 0.0,
    color_stretch_mode: str = "joint",
    stretch_plow: float = 0.1,
    stretch_phigh: float = 99.9,
    stretch_target_hi: float = 1.0,
    saturate: bool = False,
    saturation_phigh: float = 99.5,
    saturation_headroom: float = 0.30,
) -> Tuple[np.ndarray, dict]:
    """Build a composite image from per-filter images according to *spec*.

    Args:
        spec:               CompositeSpec defining the channel mapping.
        filter_images:      {filter_name: float [0,1] 2D array}
        align:              If True, align channels before compositing.
        max_shift_px:       Max allowed alignment shift (0 = no clamp).
        color_stretch_mode: How to stretch colour channels (R/G/B):
                              "joint"       – same lo/hi from all colour channels
                                              combined (preserves colour ratios)
                              "independent" – each channel independently
                              "none"        – no pre-stretch (native values)
        stretch_plow:       Lower percentile (joint/independent modes).
        stretch_phigh:      Upper percentile (joint/independent modes).

    Returns:
        (composite_image, log_dict)
        composite_image: (H, W, 3) float [0, 1]
        log_dict:        stretch and alignment details per channel
    """
    required = {spec.R, spec.G, spec.B}
    lum_key = spec.L
    if lum_key is not None:
        required.add(lum_key)

    missing = required - set(filter_images.keys())
    if missing:
        raise ValueError(f"Missing filters for composite '{spec.name}': {missing}")

    colour_keys = {spec.R, spec.G, spec.B}
    stretched: Dict[str, np.ndarray] = {}
    stretch_log: Dict[str, dict] = {}

    # ── Luminance channel ──────────────────────────────────────────────────────
    # Mirror the colour stretch mode: if colour channels are not stretched
    # (mode="none"), luminance must also stay at native intensity so that both
    # operate on the same scale before Lab L replacement.  Mismatch (e.g.
    # colour native max≈0.70 while luminance stretched to 1.0) inflates Lab L
    # from ~71 to 100, making the LRGB composite visibly too bright.
    if lum_key is not None:
        img = filter_images[lum_key]
        if color_stretch_mode == "none":
            stretched[lum_key] = img.astype(np.float32)
            stretch_log[lum_key] = {"mode": "none"}
        else:
            lo = float(np.percentile(img, stretch_plow))
            hi = float(np.percentile(img, stretch_phigh))
            stretched[lum_key] = auto_stretch(img, stretch_plow, stretch_phigh, stretch_target_hi)
            stretch_log[lum_key] = {"mode": "independent",
                                    "plow": round(lo, 5), "phigh": round(hi, 5)}

    # ── Colour channels ────────────────────────────────────────────────────────
    if color_stretch_mode == "joint":
        # Single lo/hi from all colour channels combined → preserves colour ratios
        combined = np.concatenate([filter_images[k].ravel() for k in colour_keys])
        lo = float(np.percentile(combined, stretch_plow))
        hi = float(np.percentile(combined, stretch_phigh))
        span = hi - lo if hi > lo else 1.0
        scale = stretch_target_hi / span
        for key in colour_keys:
            stretched[key] = np.clip(
                (filter_images[key] - lo) * scale, 0.0, 1.0
            ).astype(np.float32)
            stretch_log[key] = {"mode": "joint",
                                "plow": round(lo, 5), "phigh": round(hi, 5)}
    elif color_stretch_mode == "independent":
        for key in colour_keys:
            img = filter_images[key]
            lo = float(np.percentile(img, stretch_plow))
            hi = float(np.percentile(img, stretch_phigh))
            stretched[key] = auto_stretch(img, stretch_plow, stretch_phigh, stretch_target_hi)
            stretch_log[key] = {"mode": "independent",
                                "plow": round(lo, 5), "phigh": round(hi, 5)}
    else:  # "none"
        for key in colour_keys:
            stretched[key] = filter_images[key].astype(np.float32)
            stretch_log[key] = {"mode": "none"}

    # ── Alignment reference ────────────────────────────────────────────────────
    # Use a stable, fixed reference to prevent frame-to-frame composite jitter.
    # Dynamic selection (max by 95th percentile) varies per frame when channels
    # have different brightness, causing the composite planet position to shift.
    if getattr(spec, "align_ref", None) is not None:
        reference_key = spec.align_ref
    elif lum_key is not None:
        reference_key = lum_key
    else:
        # Prefer IR (best seeing quality) → R → first available colour channel
        _ALIGN_PREF = ["IR", "R", "G", "B", "CH4"]
        reference_key = next(
            (k for k in _ALIGN_PREF if k in required),
            spec.R,
        )

    # ── Align ──────────────────────────────────────────────────────────────────
    shift_log: Dict[str, list] = {k: [0.0, 0.0] for k in required}
    if align and len(stretched) > 1:
        aligned = align_channels(stretched, reference_key, max_shift_px=max_shift_px)
        for key in required:
            if key != reference_key:
                dx, dy = subpixel_align(stretched[reference_key], stretched[key])
                if max_shift_px > 0 and (abs(dx) > max_shift_px or abs(dy) > max_shift_px):
                    shift_log[key] = [0.0, 0.0]
                else:
                    shift_log[key] = [round(dx, 3), round(dy, 3)]
    else:
        aligned = stretched

    # NOTE: Pre-channel per-channel brightness masking was removed.
    # Blending each channel toward its sky background at a fixed radius creates
    # an artificially steep brightness drop at the limb (≈2× steeper than natural
    # limb darkening), producing a visible circular boundary in the composite.
    # Colour-fringe suppression is handled instead by the post-composite Lab
    # desaturation below, which affects only chrominance (a, b), not luminance.

    # ── Compose ────────────────────────────────────────────────────────────────
    r_img, g_img, b_img = aligned[spec.R], aligned[spec.G], aligned[spec.B]

    if lum_key is not None:
        result = make_lrgb(aligned[lum_key], r_img, g_img, b_img,
                           lrgb_weight=spec.lrgb_weight)
    else:
        result = make_rgb(r_img, g_img, b_img)

    # ── Post-composite limb desaturation ───────────────────────────────────────
    # The soft pre-channel mask corrects the outer limb zone (r > r_ref), but the
    # inner limb colour fringe (r ≈ 0.92–1.0 × r_ref) remains where mask≈1.0.
    # Root cause: wavelength-dependent limb darkening makes G's disk appear larger
    # than B by ~1.5 px, creating a thin colour zone at the edge.
    # Fix: after compositing, apply a Lab-space saturation taper in the limb zone.
    # This only reduces colour (a, b channels), leaving luminance (L) untouched,
    # so the natural limb-darkening gradient is preserved.
    try:
        ref_img_d = aligned[reference_key]
        cx_d, cy_d, r_d, _, _ = find_disk_center(ref_img_d)
        h_d, w_d = ref_img_d.shape[:2]
        yy_d, xx_d = np.ogrid[:h_d, :w_d]
        dist_d = np.sqrt((xx_d - cx_d) ** 2 + (yy_d - cy_d) ** 2).astype(np.float32)
        # Start desaturation at 0.89×r_ref to catch the inner limb colour fringe.
        # At r=0.89×r: mask=1 (no effect); at r=0.93×r: mask≈0.72 (28% desat);
        # at r=0.97×r: mask≈0.30 (70% desat).  Belt features end at ~0.86×r so
        # the equatorial colour region is barely touched (<7% at r=0.9×r).
        desat_start = r_d * 0.89
        desat_width = r_d * 0.15          # fade completes at ~1.04×r_ref
        t_d = np.clip((dist_d - desat_start) / desat_width, 0.0, 1.0)
        desat_mask = (0.5 * (1.0 + np.cos(np.pi * t_d))).astype(np.float32)
        # Convert composite → Lab, suppress a/b, convert back
        lab_r = cv2.cvtColor(result, cv2.COLOR_RGB2Lab)
        lab_r[:, :, 1] *= desat_mask   # a channel
        lab_r[:, :, 2] *= desat_mask   # b channel
        result = np.clip(cv2.cvtColor(lab_r, cv2.COLOR_Lab2RGB), 0.0, 1.0).astype(np.float32)
    except Exception:
        pass  # disk detection failed — skip post-composite desaturation

    # ── Auto saturation boost ──────────────────────────────────────────────────
    sat_gain: Optional[float] = None
    if saturate:
        try:
            ref_img_s = aligned[reference_key]
            cx_s, cy_s, r_s, _, _ = find_disk_center(ref_img_s)
            h_s, w_s = ref_img_s.shape[:2]
            yy_s, xx_s = np.ogrid[:h_s, :w_s]
            # Sample inside 0.85× radius to stay away from limb fringe
            disk_mask = ((xx_s - cx_s) ** 2 + (yy_s - cy_s) ** 2) <= (r_s * 0.85) ** 2

            lab_s = cv2.cvtColor(result, cv2.COLOR_RGB2Lab)
            current = float(np.percentile(
                np.sqrt(lab_s[:, :, 1] ** 2 + lab_s[:, :, 2] ** 2)[disk_mask],
                saturation_phigh,
            ))
            if current >= 1e-3:
                sat_gain = float(np.clip(127.0 * saturation_headroom / current, 1.0, 2.0))
                lab_s[:, :, 1] = np.clip(lab_s[:, :, 1] * sat_gain, -127.0, 127.0)
                lab_s[:, :, 2] = np.clip(lab_s[:, :, 2] * sat_gain, -127.0, 127.0)
                result = np.clip(cv2.cvtColor(lab_s, cv2.COLOR_Lab2RGB), 0.0, 1.0).astype(np.float32)
        except Exception:
            pass  # disk detection failed — skip saturation boost

    log = {
        "type":               "LRGB" if lum_key else "RGB",
        "color_stretch_mode": color_stretch_mode,
        "channels":           {"L": lum_key, "R": spec.R, "G": spec.G, "B": spec.B},
        "stretch":            stretch_log,
        "alignment":          shift_log,
        "saturation_gain":    round(sat_gain, 4) if sat_gain is not None else None,
    }
    return result, log
