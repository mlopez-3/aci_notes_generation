"""Utilitats comunes de les mètriques."""
from __future__ import annotations

import math


def describe(values) -> dict:
    """Estadístiques bàsiques d'una llista de números (ignora None)."""
    v = [float(x) for x in values if x is not None]
    if not v:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    mean = sum(v) / len(v)
    std = math.sqrt(sum((x - mean) ** 2 for x in v) / len(v))
    return {"n": len(v), "mean": mean, "std": std, "min": min(v), "max": max(v)}


def safe_text(text: str) -> str:
    """Alguns models fallen amb cadenes buides: es substitueixen per una paraula."""
    return text if text.strip() else "empty"


def fmt(x, nd=3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"
