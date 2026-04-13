"""Session state persistence.

Stores step completion states, last-used paths, and UI settings to
~/.astropipe/session.json so the app can resume across restarts.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SESSION_DIR  = Path.home() / ".astropipe"
SESSION_FILE = SESSION_DIR / "session.json"

SESSION_VERSION = 8   # bump when _DEFAULTS or migration logic changes

# Default values written on first run
_DEFAULTS: dict[str, Any] = {
    "session_version":  SESSION_VERSION,
    "language":         "ko",
    "camera_mode":      "mono",   # "mono" | "color"
    "planet":           "Jupiter",
    "target":           "Jup",
    "horizons_id":      "599",
    "rotation_period":  9.9281,
    "filters":          "IR,R,G,B,CH4",
    "ser_input_dir":      "",
    "input_dir":          "",
    "output_dir":         "",
    "step02_ser_dir":     "",
    "step02_output_dir":  "",
    "lucky_top_percent":  0.25,
    "lucky_ap_size":      64,
    "lucky_n_iterations": 2,
    "lucky_n_workers":    0,   # kept for migration compat; UI uses global_max_workers
    "global_max_workers": 0,   # 0=auto (all cores); Step 1 caps at 4, Step 2 uses all
    "save_mono_frames": False,
    # Which optional steps are enabled (01=PIPP off by default; 02=LuckyStack on)
    "enabled_steps":    {"01": False, "02": True,  "03": True, "04": True,
                         "05": True,  "06": True,  "07": False,
                         "08": True,  "09": True,  "10": False},
    # Last known status of each step
    "step_status":      {},
}

# Correct IR-RGB spec (R=R,G=G,B=B,L=IR — LRGB convention)
_CORRECT_IR_RGB = {"name": "IR-RGB", "R": "R", "G": "G", "B": "B", "L": "IR"}


def _migrate(data: dict[str, Any]) -> dict[str, Any]:
    """Apply forward migrations so old session files work with new code."""
    ver = data.get("session_version", 1)

    # v1→v2: rename step IDs 08/09/10/11 → 07/08/09/10 in enabled_steps
    if ver < 2:
        old_enabled = data.get("enabled_steps", {})
        remap = {"08": "07", "09": "08", "10": "09", "11": "10"}
        new_enabled = {}
        for k, v in old_enabled.items():
            new_enabled[remap.get(k, k)] = v
        data["enabled_steps"] = new_enabled

    # v2→v3: fix IR-RGB composite spec (old: R=IR,G=R,B=G → new: R=R,G=G,B=B,L=IR)
    if ver < 3:
        specs = data.get("composite_specs")
        if specs:
            for i, spec in enumerate(specs):
                if (spec.get("name") == "IR-RGB"
                        and spec.get("R") == "IR"
                        and spec.get("G") == "R"):
                    specs[i] = _CORRECT_IR_RGB
            data["composite_specs"] = specs

    # v3→v4: reset master_amounts from old default [150,150,100,...] to [200,200,200,...]
    if ver < 4:
        old_ma = data.get("master_amounts")
        if old_ma and len(old_ma) >= 3:
            if [float(v) for v in old_ma[:3]] == [150.0, 150.0, 100.0]:
                data["master_amounts"] = [200.0, 200.0, 200.0, 0.0, 0.0, 0.0]

    # v4→v5: window/cycle fields changed from minutes (float) to seconds (int).
    # Convert old session keys so the panel shows correct values.
    if ver < 5:
        if "window_minutes" in data and "window_seconds" not in data:
            data["window_seconds"] = int(round(float(data["window_minutes"]) * 60))
        elif "window_seconds" not in data:
            data["window_seconds"] = 900   # default 15 min
        if "cycle_minutes" in data and "cycle_seconds" not in data:
            data["cycle_seconds"] = int(round(float(data["cycle_minutes"]) * 60))
        elif "cycle_seconds" not in data:
            data["cycle_seconds"] = 270   # default 4.5 min

    # v5→v6: update pipeline parameter defaults that changed after empirical tuning.
    # Only overwrite if the value matches the old default (user customisation preserved).
    if ver < 6:
        if float(data.get("warp_scale", 0.20)) == 0.20:
            data["warp_scale"] = 0.80
        if int(data.get("stack_window_n", 1)) == 1:
            data["stack_window_n"] = 5
        if float(data.get("stack_min_quality", 0.0)) == 0.0:
            data["stack_min_quality"] = 0.05
        if float(data.get("series_scale", 0.80)) == 0.80:
            data["series_scale"] = 1.0
        if float(data.get("max_shift_px", 15.0)) == 15.0:
            data["max_shift_px"] = 8.0

    # v6→v7: series_amounts added (Step 8 wavelet independent from Step 6).
    # No migration needed — load_session() falls back to _SERIES_WAVELET_DEFAULTS
    # when the key is absent, so old sessions get the correct default automatically.

    # v7→v8: Steps 01 and 02 are now optional (checkbox in sidebar).
    # No data migration needed — existing sessions keep their enabled_steps values,
    # and the merge logic in load() fills in new defaults for missing keys.

    data["session_version"] = SESSION_VERSION
    return data


def reset() -> dict[str, Any]:
    """Delete the session file and return a fresh default session."""
    if SESSION_FILE.exists():
        SESSION_FILE.unlink()
    fresh = _DEFAULTS.copy()
    save(fresh)
    return fresh


def load() -> dict[str, Any]:
    """Load session from disk, merging with defaults for missing keys."""
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    if not SESSION_FILE.exists():
        save(_DEFAULTS.copy())
        return _DEFAULTS.copy()
    with open(SESSION_FILE, encoding="utf-8") as f:
        data = json.load(f)

    data = _migrate(data)

    # Merge defaults for any keys added in new versions
    merged = _DEFAULTS.copy()
    merged.update(data)
    merged["enabled_steps"] = {**_DEFAULTS["enabled_steps"],
                                **data.get("enabled_steps", {})}
    return merged


def save(data: dict[str, Any]) -> None:
    """Persist *data* to disk."""
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    data["session_version"] = SESSION_VERSION
    with open(SESSION_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
