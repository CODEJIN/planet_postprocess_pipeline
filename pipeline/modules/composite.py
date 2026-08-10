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

def _disk_region_quality(ref: np.ndarray, img: np.ndarray, cx: float, cy: float, sr: float) -> float:
    """High-pass NCC between ref and img inside the disk (r < 0.95*sr).

    Used to judge whether a candidate channel-alignment shift actually
    improved registration, since cv2.phaseCorrelate can report a confident-
    looking but WRONG peak when a channel has too little reference-correlated
    signal — observed on real Jupiter data: B channel phase correlation
    (against an IR reference) locked onto plausible-but-wrong offsets in
    ~25% of a 28-window session (e.g. dy=+2.16 reported when every
    neighbouring window measured dy in -0.4..-0.9), producing a visible
    limb colour fringe. Returns -inf if the disk mask is degenerate.
    """
    h, w = ref.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    mask = np.hypot(xx - cx, yy - cy) < sr * 0.95
    if mask.sum() < 25:
        return float("-inf")

    def _hp(im: np.ndarray) -> np.ndarray:
        return im - cv2.GaussianBlur(im, (0, 0), 4.0)

    a, b = _hp(ref)[mask], _hp(img)[mask]
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else float("-inf")


def align_channels(
    channels: Dict[str, np.ndarray],
    reference_key: str,
    max_shift_px: float = 0.0,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Tuple[float, float]]]:
    """Align all channels to *reference_key* via sub-pixel phase correlation.

    For each channel, up to 3 candidate shifts are considered — (0,0)
    (no shift), whole-frame phase correlation, and (when a disk is
    detected) the same correlation restricted to a disk-only ROI — and
    whichever gives the best post-hoc disk-region NCC (see
    _disk_region_quality) is used. This replaced a plain "trust whole-frame
    phase correlation" approach after two real-data failures on opposite
    sides of the same trade-off:
      - A disk-ROI-crop-only approach (commit e958140, motivated by Saturn
        ring pollution biasing whole-frame correlation) measurably
        *degraded* alignment for the common non-ring case across 28 real
        Jupiter windows (reverted in e958140's follow-up).
      - Trusting whole-frame correlation unconditionally still let it lock
        onto a confident-looking but WRONG peak for the B channel (weak
        IR-correlated signal) in ~25% of that same 28-window Jupiter
        session — e.g. dy=+2.16 reported when every neighbouring window
        measured dy in -0.4..-0.9 — producing a visible limb colour fringe
        that max_shift_px's magnitude-only gate does not catch, since the
        bad shift is often well under the gate.
    Quality-gating both candidates against "no shift" catches both failure
    modes: verified on the real data above that "no shift" scores highest
    for the pathological B-channel windows, while whole-frame correlation
    still wins (as before) for the normal case. If disk detection fails
    entirely, falls back to trusting whole-frame correlation (old
    pre-e958140 behaviour), since there is no ROI or quality metric to
    gate against.

    Args:
        channels:      {filter_name: float [0,1] 2D image}
        reference_key: Key of the channel to treat as reference (no shift applied).
        max_shift_px:  If > 0, shifts larger than this are discarded as
                       candidates (never considered, regardless of quality).

    Returns:
        (aligned, shifts):
            aligned: New dict with aligned images (reference unchanged, others shifted).
            shifts:  {filter_name: (dx, dy)} actually applied for every key
                     (0.0, 0.0 for the reference and for any discarded shift).
    """
    ref = channels[reference_key]

    disk = None
    try:
        cx, cy, sr, _, _ = find_disk_center(ref)
        if sr >= 10:
            disk = (cx, cy, sr)
    except Exception:
        pass

    roi = None
    if disk is not None:
        cx, cy, sr = disk
        h, w = ref.shape[:2]
        ys, ye = int(max(0, cy - sr)), int(min(h, cy + sr))
        xs, xe = int(max(0, cx - sr)), int(min(w, cx + sr))
        if (ye - ys) > 10 and (xe - xs) > 10:
            roi = (ys, ye, xs, xe)

    def _within_gate(dx: float, dy: float) -> bool:
        return max_shift_px <= 0 or (abs(dx) <= max_shift_px and abs(dy) <= max_shift_px)

    aligned: Dict[str, np.ndarray] = {}
    shifts: Dict[str, Tuple[float, float]] = {}
    for key, img in channels.items():
        if key == reference_key:
            aligned[key] = img
            shifts[key] = (0.0, 0.0)
            continue

        candidates = [(0.0, 0.0)]

        dx_full, dy_full = subpixel_align(ref, img)
        if _within_gate(dx_full, dy_full):
            candidates.append((dx_full, dy_full))

        if roi is not None:
            ys, ye, xs, xe = roi
            dx_roi, dy_roi = subpixel_align(ref[ys:ye, xs:xe], img[ys:ye, xs:xe])
            if _within_gate(dx_roi, dy_roi):
                candidates.append((dx_roi, dy_roi))

        if disk is not None and len(candidates) > 1:
            cx, cy, sr = disk
            best_dx, best_dy = max(
                candidates,
                key=lambda s: _disk_region_quality(
                    ref, img if s == (0.0, 0.0) else apply_shift(img, s[0], s[1]), cx, cy, sr
                ),
            )
        elif len(candidates) > 1:
            # No disk detected — nothing to quality-gate against; trust
            # whole-frame phase correlation (matches pre-e958140 behaviour).
            best_dx, best_dy = candidates[-1]
        else:
            best_dx, best_dy = 0.0, 0.0

        if (best_dx, best_dy) == (0.0, 0.0):
            aligned[key] = img
        else:
            aligned[key] = apply_shift(img, best_dx, best_dy)
        shifts[key] = (best_dx, best_dy)
    return aligned, shifts


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
        aligned, shifts = align_channels(stretched, reference_key, max_shift_px=max_shift_px)
        for key, (dx, dy) in shifts.items():
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

    # ── Post-composite limb desaturation — DISABLED 2026-08-09 (see below) ─────
    # ── Post-composite limb desaturation ───────────────────────────────────────
    # The soft pre-channel mask corrects the outer limb zone (r > r_ref), but the
    # inner limb colour fringe (r ≈ 0.92–1.0 × r_ref) remains where mask≈1.0.
    # Root cause: wavelength-dependent limb darkening makes G's disk appear larger
    # than B by ~1.5 px, creating a thin colour zone at the edge.
    # Fix: after compositing, apply a Lab-space saturation taper in the limb zone.
    # This only reduces colour (a, b channels), leaving luminance (L) untouched,
    # so the natural limb-darkening gradient is preserved.
    #
    # -------------------------------------------------------------------------
    # WHY THIS IS DISABLED (2026-08-09 Saturn investigation):
    #
    # The band above was tuned to desat_start=0.89×r_ref, desat_width=0.15×r_ref
    # — i.e. it forces a/b to ~0 across the OUTER 15% OF THE DISK RADIUS. That
    # is roughly 10px wide on this session's ~65px-radius Saturn disk, applied
    # to fix a stated root cause of only ~1.5px. That mismatch (a band ~7x
    # wider than the physical effect it claims to correct) strongly suggests
    # it was actually tuned empirically against a much bigger bug that existed
    # at the time: composite.py's channel-to-channel alignment (align_channels/
    # subpixel_align) used to run cv2.phaseCorrelate() on the WHOLE frame,
    # including Saturn's ring — a large, per-filter-brightness-inconsistent
    # feature that could bias the correlation peak. Measured real misalignment
    # from that bug was up to 14.6px (Saturn_Data/step06_rgb_composite/
    # window_04, B channel) — nowhere near the "~1.5px" this desaturation
    # docstring cites, but roughly consistent with needing a ~10px-wide mask
    # to hide it after the fact.
    #
    # The actual root cause is believed to be composite.py's channel-to-channel
    # alignment (align_channels/subpixel_align) running cv2.phaseCorrelate() on
    # the WHOLE frame, including Saturn's ring — a large, per-filter-brightness-
    # inconsistent feature that can bias the correlation peak. Measured real
    # misalignment from that bug was up to 14.6px (Saturn_Data/step06_rgb_composite/
    # window_04, B channel) — far larger than the "~1.5px" effect this
    # desaturation was designed for, and roughly consistent with needing a
    # ~10px-wide mask to hide it after the fact.
    #
    # A proper fix for that misalignment (disk-ROI-cropped channel alignment,
    # plus sharing one filter's disk geometry across all filters during
    # de-rotation so they don't each drift to a slightly different centre) is
    # still being developed and verified separately — NOT included in this
    # commit. This desaturation block is disabled now because, even before that
    # fix lands, it is already doing more harm than good on real Saturn data:
    # visually it produced a small, vividly-coloured inner sphere sitting
    # inside a much larger grey sphere, wiping out real colour across the
    # outer 15% of the disk radius (visually confirmed:
    # Saturn_Data/step06_compare/desat_before_after.png).
    #
    # RISK / WHY THIS IS COMMENTED OUT RATHER THAN DELETED:
    # This function is shared by every composite spec and every target
    # (Jupiter/Mars/Venus too), not just Saturn. The ~1.5px wavelength-
    # dependent limb-darkening effect this was originally meant to catch is a
    # real physical phenomenon independent of the alignment bug above, and it
    # has NOT been re-verified against real Jupiter/Mars data. If a thin
    # (~1-2px) colour fringe reappears at the limb on non-Saturn data:
    #   - Do NOT just re-enable this block as-is — first confirm with a
    #     chroma-vs-radius measurement whether a real registration bug is at
    #     play, since that should be fixed at the source, not masked.
    #   - If a genuine small residual fringe is confirmed, re-enable with a
    #     MUCH narrower band matching the originally-stated ~1.5px effect
    #     (e.g. desat_start≈0.97×r_ref, desat_width≈0.05×r_ref) rather than
    #     the current 0.89/0.15 values, which were sized for the old bug.
    #
    # try:
    #     ref_img_d = aligned[reference_key]
    #     cx_d, cy_d, r_d, _, _ = find_disk_center(ref_img_d)
    #     h_d, w_d = ref_img_d.shape[:2]
    #     yy_d, xx_d = np.ogrid[:h_d, :w_d]
    #     dist_d = np.sqrt((xx_d - cx_d) ** 2 + (yy_d - cy_d) ** 2).astype(np.float32)
    #     # Start desaturation at 0.89×r_ref to catch the inner limb colour fringe.
    #     # At r=0.89×r: mask=1 (no effect); at r=0.93×r: mask≈0.72 (28% desat);
    #     # at r=0.97×r: mask≈0.30 (70% desat).  Belt features end at ~0.86×r so
    #     # the equatorial colour region is barely touched (<7% at r=0.9×r).
    #     desat_start = r_d * 0.89
    #     desat_width = r_d * 0.15          # fade completes at ~1.04×r_ref
    #     t_d = np.clip((dist_d - desat_start) / desat_width, 0.0, 1.0)
    #     desat_mask = (0.5 * (1.0 + np.cos(np.pi * t_d))).astype(np.float32)
    #     # Convert composite → Lab, suppress a/b, convert back
    #     lab_r = cv2.cvtColor(result, cv2.COLOR_RGB2Lab)
    #     lab_r[:, :, 1] *= desat_mask   # a channel
    #     lab_r[:, :, 2] *= desat_mask   # b channel
    #     result = np.clip(cv2.cvtColor(lab_r, cv2.COLOR_Lab2RGB), 0.0, 1.0).astype(np.float32)
    # except Exception:
    #     pass  # disk detection failed — skip post-composite desaturation

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
