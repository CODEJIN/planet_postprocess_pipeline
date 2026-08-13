import sys
sys.path.insert(0, '.')
import json
import numpy as np
import cv2
from pipeline.modules import derotation as dr
from pipeline.modules import image_io

WIN_DIR = "Saturn_Data/step04_derotated"
QUALITY_JSON = "Saturn_Data/step03_quality/windows.json"

def load_window_frames(win_idx, filt):
    windows = json.load(open(QUALITY_JSON))
    w = windows["windows"][win_idx - 1] if "windows" in windows else windows[win_idx - 1]
    return w

def phase_peak_response(ref_crop, tgt_crop):
    ref_f32 = ref_crop.astype(np.float32)
    tgt_f32 = tgt_crop.astype(np.float32)
    (raw_dx, raw_dy), resp = cv2.phaseCorrelate(ref_f32, tgt_f32)
    return (-float(raw_dx), -float(raw_dy)), float(resp)

def crop_roi(lum, cx, cy, r):
    h, w = lum.shape
    ys, ye = max(0, int(cy - r)), min(h, int(cy + r))
    xs, xe = max(0, int(cx - r)), min(w, int(cx + r))
    return lum[ys:ye, xs:xe]

def process(win_idx, filt):
    log = json.load(open(f"{WIN_DIR}/window_{win_idx:02d}/derotation_log.json"))
    if filt not in log["filters"]:
        return None
    flog = log["filters"][filt]
    frames = flog["frames"]
    ref_frame = next(f for f in frames if f["align_method"] == "reference")
    ref_stem = ref_frame["stem"]
    ref_cx, ref_cy = flog.get("frames")[0]["disk_center_px"]
    disk_r = frames[0]["disk_radius_px"]
    warp_scale = flog["warp_scale"]
    pole_pa = flog["pole_pa_deg"]
    period_hours = flog["period_hours"]
    flip_dir = flog["flip_direction"]

    # find raw file paths by stem
    def find_path(stem):
        import glob
        cands = glob.glob(f"Saturn_Data/step02_lucky_stack/{stem}*.tif")
        return cands[0] if cands else None

    ref_path = find_path(ref_stem)
    if ref_path is None:
        return None
    ref_raw = image_io.read_tif(ref_path)
    ref_lum = dr._to_luminance(ref_raw)
    # replicate pipeline's own polar/eq ratio + shape measurement (best-effort)
    _cx2, _cy2, semi_a2, semi_b2, _ang2 = dr.find_disk_center(ref_lum)
    polar_eq_ratio = float(np.clip(semi_b2 / max(semi_a2, 1.0), 0.85, 1.0))

    results = []
    for f in frames:
        if f["align_method"] == "reference":
            continue
        path = find_path(f["stem"])
        if path is None:
            continue
        raw = image_io.read_tif(path)
        lum = dr._to_luminance(raw)

        # 1) RAW pre-warp re-measurement (sanity check vs logged pre_warp_center)
        ref_crop = crop_roi(ref_lum, ref_cx, ref_cy, disk_r)
        tgt_crop = crop_roi(lum, ref_cx, ref_cy, disk_r)
        (pre_dx, pre_dy), pre_resp = phase_peak_response(ref_crop, tgt_crop)

        # 2) Apply pipeline's actual warp with dt_sec, then apply logged shift,
        #    and re-measure RESIDUAL misalignment post-warp+correction.
        dt_sec = f["dt_sec"]
        warped = dr.spherical_derotation_warp(
            raw if raw.ndim == 2 else raw.mean(axis=2).astype(np.float32),
            dt_sec, ref_cx, ref_cy, disk_r,
            period_hours=period_hours, scale=warp_scale,
            flip_direction=flip_dir, pole_pa_deg=pole_pa,
            polar_equatorial_ratio=polar_eq_ratio,
        )
        log_dx, log_dy = f["align_shift_px"]
        corrected = dr.apply_shift(warped, log_dx, log_dy)
        corrected_lum = dr._to_luminance(corrected)
        ref_warped = dr.spherical_derotation_warp(
            ref_lum, 0.0, ref_cx, ref_cy, disk_r,
            period_hours=period_hours, scale=warp_scale,
            flip_direction=flip_dir, pole_pa_deg=pole_pa,
            polar_equatorial_ratio=polar_eq_ratio,
        )
        rc = crop_roi(ref_warped, ref_cx, ref_cy, disk_r)
        cc = crop_roi(corrected_lum, ref_cx, ref_cy, disk_r)
        (resid_dx, resid_dy), resid_resp = phase_peak_response(rc, cc)

        results.append({
            "stem": f["stem"][:20],
            "log_shift": (round(f["align_shift_px"][0],3), round(f["align_shift_px"][1],3)),
            "log_method": f["align_method"],
            "remeasured_raw_shift": (round(pre_dx,3), round(pre_dy,3)),
            "raw_peak_response": round(pre_resp,4),
            "post_warp_residual_shift": (round(resid_dx,3), round(resid_dy,3)),
            "post_warp_residual_mag": round(float(np.hypot(resid_dx, resid_dy)),3),
            "post_warp_peak_response": round(resid_resp,4),
        })
    return results

if __name__ == "__main__":
    all_results = {}
    filters_per_window = {i: ["IR"] for i in range(1, 10)}
    for extra_w, extra_f in [(1, "R"), (5, "G"), (9, "R")]:
        filters_per_window[extra_w].append(extra_f)

    for w in range(1, 10):
        for filt in filters_per_window[w]:
            key = f"window_{w:02d}_{filt}"
            try:
                res = process(w, filt)
                all_results[key] = res
                print(f"== {key} ==")
                if res is None:
                    print("  no data")
                    continue
                for r in res:
                    print(f"  {r['stem']}: log={r['log_shift']} ({r['log_method']}) "
                          f"remeasured_raw={r['remeasured_raw_shift']} raw_resp={r['raw_peak_response']} "
                          f"| post_warp_residual={r['post_warp_residual_shift']} mag={r['post_warp_residual_mag']} "
                          f"resp={r['post_warp_peak_response']}")
            except Exception as e:
                print(f"== {key} == ERROR: {e}")

    json.dump(all_results, open("scratch_align_hypothesis_results.json", "w"), indent=2)
