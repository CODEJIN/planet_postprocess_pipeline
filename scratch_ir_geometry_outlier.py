import sys
sys.path.insert(0, '.')
import json
import glob
import numpy as np

from pipeline.modules import derotation as dr
from pipeline.modules import image_io

WIN_DIR = "Saturn_Data/step04_derotated"
FILTERS = ["IR", "R", "G", "B", "CH4"]


def find_path(stem):
    cands = glob.glob(f"Saturn_Data/step02_lucky_stack/{stem}*.tif")
    return cands[0] if cands else None


def process_window(win_idx):
    log_path = f"{WIN_DIR}/window_{win_idx:02d}/derotation_log.json"
    log = json.load(open(log_path))
    out = {}
    for filt in FILTERS:
        if filt not in log["filters"]:
            continue
        flog = log["filters"][filt]
        ref_stem = flog["reference_stem"]
        ref_frame = next(f for f in flog["frames"] if f["stem"] == ref_stem)
        logged_disk_center = ref_frame["disk_center_px"]
        logged_disk_radius = ref_frame["disk_radius_px"]
        logged_own_disk_radius = ref_frame["own_disk_radius_px"]
        logged_radius_shared = ref_frame["radius_shared"]
        geometry_source = flog.get("geometry_source")

        path = find_path(ref_stem)
        if path is None:
            out[filt] = {"error": f"raw tif not found for stem {ref_stem}"}
            continue
        try:
            raw = image_io.read_tif(path)
            lum = dr._to_luminance(raw) if raw.ndim == 3 else raw
            cx, cy, semi_a, semi_b, angle_major, confidence, shape_reliable = dr._find_disk_center_impl(lum)
            aspect = float(semi_b / max(semi_a, 1e-6))
        except Exception as e:
            out[filt] = {"error": f"find_disk_center failed: {e}"}
            continue

        out[filt] = {
            "reference_stem": ref_stem,
            "own_recomputed": {
                "cx": round(float(cx), 3),
                "cy": round(float(cy), 3),
                "semi_a": round(float(semi_a), 3),
                "semi_b": round(float(semi_b), 3),
                "aspect_ratio": round(aspect, 4),
                "angle_major_deg": round(float(angle_major), 3),
                "confidence": round(float(confidence), 3),
                "shape_reliable": bool(shape_reliable),
            },
            "logged": {
                "disk_center_px": logged_disk_center,
                "disk_radius_px": logged_disk_radius,
                "own_disk_radius_px": logged_own_disk_radius,
                "radius_shared": logged_radius_shared,
                "geometry_source": geometry_source,
            },
        }
    return out


def main():
    all_results = {}
    for w in range(1, 10):
        try:
            all_results[f"window_{w:02d}"] = process_window(w)
        except Exception as e:
            all_results[f"window_{w:02d}"] = {"error": str(e)}

    json.dump(all_results, open("scratch_ir_geometry_outlier_results.json", "w"), indent=2)

    # ---- Analysis: per window, is IR's own aspect ratio an outlier vs siblings? ----
    print("=" * 100)
    for w in range(1, 10):
        key = f"window_{w:02d}"
        wres = all_results[key]
        print(f"\n== {key} ==")
        aspects = {}
        semi_as = {}
        for filt in FILTERS:
            r = wres.get(filt)
            if r is None:
                continue
            if "error" in r:
                print(f"  {filt}: ERROR - {r['error']}")
                continue
            oc = r["own_recomputed"]
            print(f"  {filt}: semi_a={oc['semi_a']:.2f} semi_b={oc['semi_b']:.2f} "
                  f"aspect={oc['aspect_ratio']:.4f} conf={oc['confidence']:.2f} "
                  f"reliable={oc['shape_reliable']} | logged_own_r={r['logged']['own_disk_radius_px']:.2f} "
                  f"shared={r['logged']['radius_shared']} geom_src={r['logged']['geometry_source']}")
            aspects[filt] = oc["aspect_ratio"]
            semi_as[filt] = oc["semi_a"]

        if len(aspects) >= 2:
            non_ir = {f: v for f, v in aspects.items() if f != "IR"}
            if non_ir:
                consensus_mean = np.mean(list(non_ir.values()))
                consensus_std = np.std(list(non_ir.values()))
                if "IR" in aspects:
                    ir_dev = abs(aspects["IR"] - consensus_mean)
                    r_dev = abs(non_ir.get("R", consensus_mean) - consensus_mean)
                    print(f"  non-IR consensus aspect mean={consensus_mean:.4f} std={consensus_std:.4f}")
                    print(f"  IR deviation from non-IR consensus: {ir_dev:.4f}")
                    if "R" in non_ir:
                        print(f"  R deviation from non-IR consensus: {r_dev:.4f}")
                    # simple outlier flag: IR farther from consensus than max non-IR spread (std) by notable margin
                    is_outlier = ir_dev > max(consensus_std * 1.5, 0.01) and ir_dev > r_dev
                    print(f"  IR aspect outlier vs siblings: {is_outlier}")


if __name__ == "__main__":
    main()
