"""Mètriques d'avaluació de les notes generades.

Cada mòdul de mètrica exposa:
  metric_name(opts) -> str      nom amb versió que va al fitxer (p. ex. "entities-v4")
  run(rows, ctx, opts) -> dict  {"summary": ..., "per_note": [...], "config": {...}}
  headline(summary) -> str      una línia per mostrar per pantalla
  NEEDS_SOURCE (opcional)       True si necessita la transcripció (ctx["src"][file])

rows = files d'una predicció (file, pred, ref, ...). Els models pesants només s'importen en cridar run().
"""
import importlib

METRIC_MODULES = {
    "rouge": "rouge",
    "bertscore": "bertscore",
    "bleurt": "bleurt",
    "entities": "entities",
    "length": "length",
}


def load(metric_id: str):
    if metric_id not in METRIC_MODULES:
        raise ValueError(f"mètrica desconeguda {metric_id!r}; opcions: {sorted(METRIC_MODULES)}")
    return importlib.import_module(f"{__name__}.{METRIC_MODULES[metric_id]}")
