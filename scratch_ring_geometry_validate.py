import sys, json
sys.path.insert(0, '.')
import numpy as np
import cv2
import tifffile

from pipeline.modules.derotation import (
    find_disk_center, detect_ring_geometry, render_ring_geometry_overlay,
)

WINDOWS = [f"window_{i:02d}" for i in range(1, 10)]
FILTERS = ["IR", "R", "G", "B", "CH4"]

def load_norm(path):
    arr = tifffile.imread(path).astype(np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    mx = arr.max()
    if mx > 1.5:
        arr = arr / (65535.0 if mx > 255 else 255.0)
    return arr

results = []
for win in WINDOWS:
    for filt in FILTERS:
        path = f"Saturn_Data/step04_derotated/{win}/{filt}_derotated.tif"
        try:
            img = load_norm(path)
        except FileNotFoundError:
            continue
        cx, cy, semi_a, semi_b, angle = find_disk_center(img)
        ring = detect_ring_geometry(img, cx, cy, semi_a, semi_b)
        overlay = render_ring_geometry_overlay(img, cx, cy, semi_a, semi_b, angle, ring)
        out_path = f"/tmp/ring_check_{win}_{filt}.png"
        cv2.imwrite(out_path, overlay)
        row = {
            "window": win,
            "filter": filt,
            "disk_semi_a": round(semi_a, 2),
            "disk_semi_b": round(semi_b, 2),
            "ring_detected": ring is not None,
            "outer_semi_a": round(ring.outer_semi_a, 2) if ring else None,
            "outer_semi_b": round(ring.outer_semi_b, 2) if ring else None,
            "ratio_to_disk": round(ring.outer_semi_a / semi_a, 3) if ring else None,
            "confidence": round(ring.confidence, 3) if ring else None,
            "crosses_disk": ring.crosses_disk if ring else None,
            "angle_deg": round(ring.angle_deg, 1) if ring else None,
            "disk_angle_deg": round(angle, 1),
            "overlay": out_path,
        }
        if ring is not None:
            r = row["ratio_to_disk"]
            row["plausible"] = bool(1.5 <= r <= 3.0)
        else:
            row["plausible"] = None
        results.append(row)
        print(json.dumps(row))

with open("/tmp/ring_geometry_results.json", "w") as f:
    json.dump(results, f, indent=2)
