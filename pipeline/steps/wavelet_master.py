"""
Step 5 – Wavelet sharpening (master).

Applies final wavelet sharpening to the de-rotated master TIFs produced by
Step 4.  Uses gentler parameters than the Step 7 preview (master_amounts
vs preview_amounts) to avoid over-sharpening the already high-SNR stacks.

One PNG is written per filter per time window.  These are the direct inputs
to Step 6 (RGB compositing).

Output (when config.save_step05 is True):
    <output_base>/step05_wavelet_master/
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
from pipeline.modules.derotation import (
    find_disk_center,
    _robust_ellipse_refit,
    _SATURN_RING_INNER_REQ,
    _SATURN_RING_OUTER_REQ,
)




def run(
    config: PipelineConfig,
    results_04: dict,
    cancel_event=None,
) -> Dict[str, List[Tuple[Optional[Path], str]]]:
    """Run Step 5 for all windows produced by Step 4.

    Args:
        config:      Pipeline configuration.
        results_04:  Output of step04_derotate_stack.run(), containing:
                     ``{"windows": [{"window_index", "center_time",
                                     "outputs": {filter: Path|None}, ...}, ...]}``

    Returns:
        ``{window_label: [(png_path, filter_name), ...]}``
        *png_path* is None when ``config.save_step05`` is False.
    """
    windows: List[dict] = results_04.get("windows", [])
    if not windows:
        print("  [WARNING] No Step 4 windows — Step 5 skipped.")
        return {}

    # ── Output directory ───────────────────────────────────────────────────────
    out_base: Optional[Path] = None
    if config.save_step05:
        out_base = config.step_dir(5, "wavelet_master")
        out_base.mkdir(parents=True, exist_ok=True)
        print(f"  Output → {out_base}")
    else:
        print("  save_step05=False: results not written to disk")

    print(f"  Wavelet amounts: {config.wavelet.master_amounts}  "
          f"power={config.wavelet.master_power}  "
          f"sharpen_filter={config.wavelet.master_sharpen_filter}  "
          f"denoise={config.wavelet.master_denoise_amounts}  "
          f"filter={config.wavelet.master_filter_type}")


    results: Dict[str, List[Tuple[Optional[Path], str]]] = {}
    total_written = 0

    for win in windows:
        if cancel_event is not None and cancel_event.is_set():
            print("  [CANCELLED] Stopping Step 5.", flush=True)
            break
        win_idx = win["window_index"]
        win_label = f"window_{win_idx:02d}"
        t_str = win["center_time"]
        outputs: Dict[str, Optional[Path]] = win.get("outputs", {})
        logs: Dict[str, dict] = win.get("log", {})

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
            _wlog = logs.get(filt, {})
            try:
                _cx, _cy, _rx, _ry, _angle = find_disk_center(_lum)
                _has_disk = _rx >= 5
            except Exception:
                _has_disk = False

            if _has_disk and config.wavelet.master_limb_fit_refinement_enabled \
                    and not bool(_wlog.get("has_rings", False)):
                # Robust disk-limb refit (2026-08-15, opt-in, ringless targets
                # only): find_disk_center()'s ellipse fit has a measured
                # ~0.5-0.9px ASYMMETRIC error vs. the true photometric limb,
                # the root cause of the gray-halo/white-rim wavelet artifact
                # trade-off -- see WaveletConfig.master_limb_fit_refinement_
                # enabled's docstring and SATURN_RING_WAVELET_STATUS_2026-08-
                # 15.md. Validated on real Jupiter data; NOT validated (and
                # not applied) for has_rings=True targets, where ring-crossing
                # ray contamination defeats this refit's outlier rejection --
                # Saturn keeps today's fit unchanged.
                _refit = _robust_ellipse_refit(_lum, _cx, _cy, _rx, _ry, _angle)
                if _refit is not None:
                    _cx, _cy, _rx, _ry, _angle, _n_kept = _refit
                    print(f"    [{filt}] limb fit refined (kept {_n_kept}/72 rays): "
                          f"rx={_rx:.2f} ry={_ry:.2f} angle={_angle:.2f}°")

            if _has_disk:
                # find_disk_center returns angle in degrees; convert to radians
                _angle_rad = np.radians(_angle)

                # Ring-aware sharpening mask (2026-08-12/13 fix): sharpen_disk_
                # aware's feather mask is normally zero beyond the disk radius,
                # which was silently zeroing all sharpening gain over Saturn's
                # rings (the actual root cause of the Cassini Division
                # vanishing here but not in step07's plain, unmasked
                # sharpen()) -- see project_derotation_ring_occlusion_fix
                # memory for the full investigation. Reuses has_rings/
                # pole_pa_deg/sub_observer_lat_deg exactly as derotate_filter()
                # computed and used them for THIS filter/window (single source
                # of truth, no re-derivation), and the same IAU ring-radius
                # ratio already validated in compute_ring_occlusion_weight().
                # Anchored on _rx -- this function's own fresh fit on the
                # actual image being sharpened, not the de-rotation-time
                # ref_semi_a from a different frame.

                # Coverage-aware sharpening gain (2026-08-15, opt-in): reduce
                # wavelet gain where derotate_filter()'s per-pixel de-rotation
                # coverage n(x) is low, targeting the diagnosed Saturn
                # asymmetric limb-ringing (find_disk_center's ellipse fit has
                # a measured ~0.5-0.9px asymmetric error vs the true
                # photometric limb -- see WaveletConfig.master_coverage_
                # aware_sharpening's docstring). Reads the companion coverage
                # TIF that derotate_window() saved next to the derotated TIF
                # (path only -- the raw array is never left in the in-memory
                # log dict, see derotate_window()'s own popping logic).
                _confidence_map = None
                if config.wavelet.master_coverage_aware_sharpening:
                    _cov_path = _wlog.get("coverage_map_file")
                    if _cov_path:
                        try:
                            _cov_raw = image_io.read_tif(Path(_cov_path))
                            _confidence_map = wavelet.coverage_to_confidence(
                                _cov_raw, floor=config.wavelet.master_coverage_confidence_floor
                            )
                        except Exception as exc:
                            print(f"    [{filt}] coverage map read failed ({exc}) "
                                  f"-- sharpening without confidence weighting")
                    else:
                        print(f"    [{filt}] master_coverage_aware_sharpening enabled but "
                              f"no coverage_map_file logged (needs use_true_reprojection=True "
                              f"and s0_sl_blend_enabled or master_coverage_aware_sharpening "
                              f"active at step04) -- sharpening without confidence weighting")

                _extra_rx = _extra_ry = _extra_angle = None
                _extra_gap_px = None
                if bool(_wlog.get("has_rings", False)):
                    _pole_pa_deg = float(_wlog.get("pole_pa_deg", 0.0))
                    # Reuse pole_pa_deg for the PRIMARY ellipse's angle too
                    # (overriding the noisy per-image Otsu fit _angle) so the
                    # globe mask stays co-oriented with the ring geometry by
                    # construction -- user-reported "abnormal line at the
                    # ring/globe border" traced to this fit's angle (178.5 deg
                    # here) diverging from pole_pa_deg (173.0 deg mod 180) by
                    # ~5.5 deg on real data. The globe is close enough to
                    # circular (ry/rx~0.9) that a few-degree orientation
                    # change has no visible effect on its own limb feathering.
                    # Kept regardless of master_ring_extension_enabled below --
                    # this angle fix is independent of, and unaffected by, the
                    # separate ring-extension-mask issue found 2026-08-15 (the
                    # controlled A/B/C/D test that found it used this same
                    # pole_pa-based angle in every case, including the clean
                    # ones -- see master_ring_extension_enabled's docstring).
                    _angle_rad = np.radians(_pole_pa_deg)

                if bool(_wlog.get("has_rings", False)) and config.wavelet.master_ring_extension_enabled:
                    _sub_obs_lat_deg = float(_wlog.get("sub_observer_lat_deg", 0.0))
                    _sin_b = abs(np.sin(np.radians(_sub_obs_lat_deg)))
                    # User-reported real-data check (window_01/R): the visible
                    # ring signal in a de-rotated, multi-frame-stacked image
                    # fades out gradually well past the strict IAU A-ring
                    # outer-edge ratio (2.269x) -- measured reaching background
                    # around 2.7-2.8x, not 2.269x, almost certainly PSF/seeing/
                    # residual-stacking blur smearing the true edge outward.
                    # That blur width isn't a fixed physical ring constant, so
                    # rather than guess a per-session value, apply a generous,
                    # explicitly-labelled SAFETY margin on top of the physical
                    # ratio -- over-covering just applies sharpening gain to a
                    # bit of background sky (harmless), while under-covering
                    # (the original bug) leaves real ring detail permanently
                    # unsharpened. Not a physical claim.
                    _RING_MASK_SAFETY_FACTOR = 1.35
                    _extra_rx = _rx * _SATURN_RING_OUTER_REQ * _RING_MASK_SAFETY_FACTOR
                    _extra_ry = max(_extra_rx * _sin_b, 1e-6)
                    # Inner ramp width for the ring's own weight, starting
                    # at the globe's own true edge -- see sharpen_disk_
                    # aware's docstring for the two rejected designs before
                    # this one (a solid ellipse; then a separate, wrong-
                    # eccentricity "hole" ellipse; then a HARD pixel-distance
                    # exclusion band). The hard-exclusion version fixed the
                    # ringing but real-data review found the disk-to-ring
                    # "gap" isn't actually empty in real multi-frame Saturn
                    # stacks -- it's the disk's own fairly bright limb-
                    # darkening tail -- so a flat zero band there read as a
                    # distinct, out-of-place "blurry halo" sitting in front
                    # of the ring (user-reported). extra_gap_px is now the
                    # width of a single CONTINUOUS ramp (0 at the edge, full
                    # strength by extra_gap_px pixels out) instead -- no flat
                    # segment for the eye to pick out. Width still needs to
                    # be at least the widest ACTIVE wavelet feather zone so
                    # that level's own transition isn't truncated (the
                    # original ringing bug); empirical sweep against this
                    # exact data (0/5/8/12/16px) found 8px -- matching the
                    # widest active level's own feather_L below -- clean
                    # under the hard-exclusion design, and it remains a
                    # reasonable width for the continuous ramp too (wider
                    # ramps were not re-swept since the flat-segment problem
                    # that motivated narrowness no longer applies the same
                    # way, but there's no evidence 8px needs revisiting).
                    # Deferred until after _use_eff is known (a few lines
                    # down) since it depends on that.
                    _extra_gap_px = None  # set below once _use_eff is known
                    _extra_angle = _angle_rad

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

                if _extra_rx is not None:
                    _active_idxs = [i for i, a in enumerate(config.wavelet.master_amounts) if a != 0]
                    _max_active_level = max(_active_idxs) if _active_idxs else 0
                    _extra_gap_px = (2 ** _max_active_level) * _use_eff

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
                        denoise_amounts=config.wavelet.master_denoise_amounts,
                        filter_type=config.wavelet.master_filter_type,
                        extra_rx=_extra_rx, extra_ry=_extra_ry, extra_angle=_extra_angle,
                        extra_gap_px=_extra_gap_px,
                        confidence_map=_confidence_map,
                        fill_outside_before_sharpen=config.wavelet.master_edge_extension_enabled,
                        overshoot_clamp_radius_px=config.wavelet.master_overshoot_clamp_radius_px,
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
                        denoise_amounts=config.wavelet.master_denoise_amounts,
                        filter_type=config.wavelet.master_filter_type,
                        extra_rx=_extra_rx, extra_ry=_extra_ry, extra_angle=_extra_angle,
                        extra_gap_px=_extra_gap_px,
                        confidence_map=_confidence_map,
                        fill_outside_before_sharpen=config.wavelet.master_edge_extension_enabled,
                        overshoot_clamp_radius_px=config.wavelet.master_overshoot_clamp_radius_px,
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
                        denoise_amounts=config.wavelet.master_denoise_amounts,
                        filter_type=config.wavelet.master_filter_type,
                        overshoot_clamp_radius_px=config.wavelet.master_overshoot_clamp_radius_px,
                    )
                else:
                    sharpened = wavelet.sharpen(
                        img,
                        levels=config.wavelet.levels,
                        amounts=config.wavelet.master_amounts,
                        power=config.wavelet.master_power,
                        sharpen_filter=config.wavelet.master_sharpen_filter,
                        denoise_amounts=config.wavelet.master_denoise_amounts,
                        filter_type=config.wavelet.master_filter_type,
                        overshoot_clamp_radius_px=config.wavelet.master_overshoot_clamp_radius_px,
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

    print(f"\n  Step 5 complete: {total_written} master PNGs written")
    return results
