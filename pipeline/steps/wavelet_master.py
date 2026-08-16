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
    _navigation_constrained_ellipse_fit,
    compute_ring_sharpening_mask,
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

                _extra_weight_map = None
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

                    if config.wavelet.master_navigation_limb_fit_enabled:
                        # Navigation-constrained limb fit (2026-08-16, opt-in,
                        # has_rings=True only -- the counterpart to master_
                        # limb_fit_refinement_enabled above, which is
                        # has_rings=False only): fixes orientation
                        # (pole_pa_deg, already independent of any ellipse
                        # fit) and apparent aspect ratio (analytically
                        # predicted from Horizons B + the planet's TRUE
                        # physical oblateness) before looking at any ray
                        # data, then fits only (cx, cy, scale) from the
                        # ring-free angular sectors -- see
                        # derotation._navigation_constrained_ellipse_fit()'s
                        # docstring for why this succeeds where the
                        # statistical refit above and an earlier exclude-
                        # ring-rays-then-free-refit experiment
                        # (experiments/scratch_globe_fit_asymmetry_diagnosis.py)
                        # did not.
                        _sub_obs_lat_deg = float(_wlog.get("sub_observer_lat_deg", 0.0))
                        _nav_fit = _navigation_constrained_ellipse_fit(
                            _lum, _cx, _cy, _rx, _ry, _pole_pa_deg, _sub_obs_lat_deg,
                            config.derotation.true_polar_equatorial_ratio,
                        )
                        if _nav_fit is not None:
                            _cx, _cy, _rx, _ry, _pole_pa_deg, _n_kept = _nav_fit
                            _angle_rad = np.radians(_pole_pa_deg)
                            print(f"    [{filt}] navigation-constrained limb fit "
                                  f"(kept {_n_kept} rays): rx={_rx:.2f} ry={_ry:.2f}")

                if bool(_wlog.get("has_rings", False)) and config.wavelet.master_ring_extension_enabled:
                    _sub_obs_lat_deg = float(_wlog.get("sub_observer_lat_deg", 0.0))
                    # 2026-08-15 (second pass, same day): the original extra_rx/
                    # extra_ry/extra_gap_px approach below (a FILLED ellipse
                    # with only an ~8px inner ramp) was found -- via external
                    # review, confirmed against real data -- to hand full
                    # ring-level sharpening gain to the wide, mostly-empty gap
                    # between the globe's true edge and the ring's own true
                    # inner edge (r=1.0 to r=_SATURN_RING_INNER_REQ~1.239),
                    # which is real high-SNR signal but is the globe's own PSF
                    # limb tail, not ring material -- amplifying it produced
                    # the white-rim/dark-trough artifact at the disk-ring
                    # junction. Replaced with a true ring ANNULUS mask (see
                    # compute_ring_sharpening_mask's docstring for the full
                    # derivation and the real-data validation, experiments/
                    # ringing_fix_validation/v2_fullring_* from that session):
                    # nothing in the globe-to-ring gap gets any gain from this
                    # mask, the far/occluded ring arc is excluded only where
                    # it overlaps the globe's own silhouette, and the outer
                    # edge is feathered the same way compute_ring_occlusion_
                    # weight_3d already feathers its own boundary.
                    _extra_weight_map = compute_ring_sharpening_mask(
                        img.shape[0], img.shape[1], _cx, _cy, _rx, _ry,
                        _pole_pa_deg, _sub_obs_lat_deg,
                    )

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
                        denoise_amounts=config.wavelet.master_denoise_amounts,
                        filter_type=config.wavelet.master_filter_type,
                        extra_weight_map=_extra_weight_map,
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
                        extra_weight_map=_extra_weight_map,
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
