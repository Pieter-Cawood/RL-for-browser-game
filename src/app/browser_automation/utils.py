"""Utility helpers used across the browser/canvas stack."""
from __future__ import annotations

def clamp(v: float, lo: float, hi: float) -> float:
    """Clamp a number to [lo, hi]."""
    return max(lo, min(hi, v))

def parse_bool(s: str | None, default: bool = False) -> bool:
    if s is None:
        return default
    return s.strip().lower() in {"1", "true", "yes", "on"}
