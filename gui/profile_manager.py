"""Profile management for PlanetFlow.

Profiles store processing parameters (planet, camera mode, step settings, etc.)
and are saved as JSON files under ~/.astropipe/profiles/.
Paths and runtime state are excluded — they stay in session.json.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from gui import session as _session

PROFILES_DIR = _session.SESSION_DIR / "profiles"
PROFILE_VERSION = 1

# Keys excluded from profiles (path-specific, runtime, or UI-only)
_EXCLUDE_KEYS = {
    "session_version",
    "language",
    "active_profile",
    "step_status",
    "lucky_n_workers",  # deprecated alias
}

def _is_path_key(k: str) -> bool:
    return k.endswith("_dir") or k.endswith("_path")


def _profile_path(name: str) -> Path:
    return PROFILES_DIR / f"{name}.json"


def list_profiles() -> list[str]:
    """Return sorted list of profile names (without .json extension)."""
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(p.stem for p in PROFILES_DIR.glob("*.json"))


def load_profile(name: str) -> dict[str, Any]:
    """Load and return profile data dict (only processing params, no paths)."""
    path = _profile_path(name)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # Strip profile-meta fields before returning to caller
    skip = {"profile_version", "name", "created_at", "updated_at"}
    return {k: v for k, v in data.items() if k not in skip}


def save_profile(name: str, session_data: dict[str, Any]) -> None:
    """Save a profile extracted from session_data, excluding path/meta keys."""
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    path = _profile_path(name)

    created_at = datetime.now().isoformat()
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                existing = json.load(f)
            created_at = existing.get("created_at", created_at)
        except Exception:
            pass

    profile: dict[str, Any] = {
        "profile_version": PROFILE_VERSION,
        "name": name,
        "created_at": created_at,
        "updated_at": datetime.now().isoformat(),
    }
    for k, v in session_data.items():
        if k not in _EXCLUDE_KEYS and not _is_path_key(k):
            profile[k] = v

    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)


def delete_profile(name: str) -> None:
    """Delete a profile file if it exists."""
    path = _profile_path(name)
    if path.exists():
        path.unlink()


def profile_meta(name: str) -> dict[str, str]:
    """Return display metadata for a profile (planet, camera_mode, updated_at)."""
    path = _profile_path(name)
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    return {
        "planet":      data.get("planet", ""),
        "camera_mode": data.get("camera_mode", "mono"),
        "updated_at":  data.get("updated_at", ""),
    }
