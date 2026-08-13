"""Diagnostic (2026-08-12): measure frame-to-frame ROTATION (fitted ellipse
position angle) and oblateness-aspect-ratio drift for Saturn windows 1, 2, 5
(IR/R filters), and compare against the already-measured/already-fixed
isotropic radius drift (IR ~1.25%, R ~2.15% max spread within window_01).

Investigation only -- no pipeline/ files modified. Calls
_find_disk_center_impl() directly (documented internal API for exactly this
kind of per-frame diagnostic measurement) on each raw step02 frame matched
via windows.json's per_filter.{FILTER}.included stems.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from pipeline.modules import image_io
from pipeline.modules.derotation import _find_disk_center_impl

STEP02_DIR = Path("Saturn_Data/step02_lucky_stack")
WINDOWS_JSON = Path("Saturn_Data/step03_quality/windows.json")

FILTERS = ["IR", "R"]
WINDOW_INDICES = [1, 2, 5]


def _parse_ts(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")


def main():
    data = json.load(open(WINDOWS_JSON))
    windows = {w["window_index"]: w for w in data["selected_windows"]}

    all_results = {}

    for wi in WINDOW_INDICES:
        window = windows[wi]
        all_results[wi] = {}
        for filt in FILTERS:
            pf = window["per_filter"].get(filt)
            if pf is None:
                print(f"window_{wi:02d} {filt}: not present")
                continue
            rows = []
            for item in pf["included"]:
                matches = list(STEP02_DIR.glob(f"{item['stem']}.tif"))
                if not matches:
                    print(f"window_{wi:02d} {filt}: MISSING tif for {item['stem']}")
                    continue
                raw = image_io.read_tif(str(matches[0]))
                lum = raw if raw.ndim == 2 else raw.mean(axis=2).astype(np.float32)
                cx, cy, semi_a, semi_b, angle_major, confidence, shape_reliable = (
                    _find_disk_center_impl(lum)
                )
                rows.append({
                    "stem": item["stem"],
                    "cx": float(cx), "cy": float(cy),
                    "semi_a": float(semi_a), "semi_b": float(semi_b),
                    "angle_major_deg": float(angle_major),
                    "oblateness": float(semi_b / semi_a) if semi_a > 0 else float("nan"),
                    "confidence": float(confidence),
                    "shape_reliable": bool(shape_reliable),
                })

            all_results[wi][filt] = rows

            if len(rows) < 2:
                print(f"window_{wi:02d} {filt}: n={len(rows)} -- not enough frames for spread")
                continue

            semi_a_vals = np.array([r["semi_a"] for r in rows])
            angle_vals = np.array([r["angle_major_deg"] for r in rows])
            oblate_vals = np.array([r["oblateness"] for r in rows])
            conf_vals = np.array([r["confidence"] for r in rows])
            reliable_vals = [r["shape_reliable"] for r in rows]

            radius_spread_pct = (semi_a_vals.max() - semi_a_vals.min()) / semi_a_vals.mean() * 100
            angle_spread_deg = angle_vals.max() - angle_vals.min()
            # also handle wraparound (angle could be near +/-90 boundary)
            angle_wrapped = np.mod(angle_vals + 90, 180) - 90
            angle_spread_deg_wrapped = angle_wrapped.max() - angle_wrapped.min()
            angle_spread_deg = min(angle_spread_deg, angle_spread_deg_wrapped)
            oblate_spread = oblate_vals.max() - oblate_vals.min()
            oblate_spread_pct = oblate_spread / oblate_vals.mean() * 100

            # Tangential displacement at ~2x semi_a from position-angle spread
            r_cassini_est = 2.0 * semi_a_vals.mean()
            delta_angle_rad = np.radians(angle_spread_deg)
            tangential_px = r_cassini_est * np.sin(delta_angle_rad)

            # Aspect-ratio-implied displacement at the ansa (near minor axis
            # projection): the *globe's* oblateness measures polar vs
            # equatorial semi-axis ratio of the ellipsoid silhouette --
            # NOT the ring ellipse (rings are a separate, geometrically flat,
            # non-rigidly-attached structure with their own projected
            # ellipticity driven by sub-observer latitude, which is a fixed
            # physical/geometric quantity, not a per-frame fit artifact of
            # the globe). A globe-oblateness fit spread of X% is a measurement
            # noise floor of the disk-fit routine itself; it has no known
            # geometric mapping onto ring-ellipse displacement at the ansa.
            # We report the raw spread and flag this explicitly rather than
            # forcing a fabricated conversion.
            n_unreliable = sum(1 for r in rows if not r["shape_reliable"])
            n_zero_conf = sum(1 for r in rows if r["confidence"] <= 0.0)

            print(f"\n=== window_{wi:02d} {filt} (n={len(rows)}) ===")
            print(f"  semi_a values (px):      {np.round(semi_a_vals, 3).tolist()}")
            print(f"  radius spread:           {radius_spread_pct:.3f}%")
            print(f"  angle_major_deg values:  {np.round(angle_vals, 3).tolist()}")
            print(f"  angle spread (wrap-aware): {angle_spread_deg:.4f} deg")
            print(f"  oblateness (b/a) values: {np.round(oblate_vals, 5).tolist()}")
            print(f"  oblateness spread:       {oblate_spread:.5f} ({oblate_spread_pct:.3f}%)")
            print(f"  --> tangential px at r=2*semi_a from angle spread: {tangential_px:.4f} px")
            print(f"  confidence values:       {np.round(conf_vals, 3).tolist()}")
            print(f"  shape_reliable values:   {reliable_vals}")
            print(f"  n_unreliable={n_unreliable} n_zero_confidence={n_zero_conf}")

    out_path = Path("scratch_rotation_aspect_drift_results.json")
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
