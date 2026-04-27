"""Internationalisation loader.

Usage::

    from gui.i18n import S
    label = QLabel(S("btn.run"))
    text  = S("step02.detected", n=5)   # supports {key} substitution
"""
from __future__ import annotations

import json
from pathlib import Path

_DIR = Path(__file__).parent / "i18n"
_SUPPORTED = {"ko", "en"}
_DEFAULT = "en"

_strings: dict[str, str] = {}


def load(lang: str = _DEFAULT) -> None:
    """Load the string table for *lang*.  Falls back to Korean if unknown."""
    global _strings
    if lang not in _SUPPORTED:
        lang = _DEFAULT
    path = _DIR / f"{lang}.json"
    with open(path, encoding="utf-8") as f:
        _strings = json.load(f)


def S(key: str, **kwargs: object) -> str:  # noqa: N802
    """Return the localised string for *key*, interpolating **kwargs**."""
    text = _strings.get(key, key)
    if kwargs:
        try:
            text = text.format(**kwargs)
        except (KeyError, ValueError):
            pass
    return text
