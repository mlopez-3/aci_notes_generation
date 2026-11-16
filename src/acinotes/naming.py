"""Convención de nombres de predicciones y métricas.

PREDICCIÓN   predictions/<split>__<system>__<model>__<variant>__<stamp>.jsonl
             predictions/<split>__<system>__<model>__<variant>__<stamp>.meta.json   (configuración y commit)

MÉTRICA      metrics/<nombre-de-la-predicción>__<metric>.json
             (el nombre de la predicción va ENTERO al principio: la métrica se asocia a su predicción
              con solo leer el nombre del fichero)

Campos (separados por doble guion bajo '__'; dentro de un campo solo letras minúsculas, dígitos, '.' y '-'):
  split    train | valid | test1 | test2 | test3
  system   zeroshot | fewshot | lora        (o el que definas)
  model    qwen3-4b-instruct-2507
  variant  configuración que distingue ejecuciones del mismo sistema: 4bit-bf16, k2-similar-4bit-bf16, r16-ep3-4bit-fp16...
  stamp    fecha y hora UTC de la generación: 20261116T1702Z
  metric   <nombre>-<versión>: surface-v1, entities-v4, bleurt-elron-base-512-v1...

Ejemplo:
  predictions/valid__zeroshot__qwen3-4b-instruct-2507__4bit-bf16__20261116T1702Z.jsonl
  metrics/valid__zeroshot__qwen3-4b-instruct-2507__4bit-bf16__20261116T1702Z__entities-v4.json
"""
from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass
from pathlib import Path

SEP = "__"
SPLITS = {"train", "valid", "test1", "test2", "test3"}
_STAMP_RE = re.compile(r"^\d{8}T\d{4}Z$")


def slug(text: str) -> str:
    """Texto -> campo seguro para nombres de fichero (Windows y Linux)."""
    s = re.sub(r"[^a-z0-9.]+", "-", str(text).lower()).strip("-.")
    if not s:
        raise ValueError(f"campo vacío tras limpiar {text!r}")
    return s


def utc_stamp(now: _dt.datetime | None = None) -> str:
    now = now or _dt.datetime.now(_dt.timezone.utc)
    return now.astimezone(_dt.timezone.utc).strftime("%Y%m%dT%H%MZ")


@dataclass(frozen=True)
class PredictionId:
    split: str
    system: str
    model: str
    variant: str
    stamp: str

    def __post_init__(self):
        if self.split not in SPLITS:
            raise ValueError(f"split desconocido {self.split!r}; usa uno de {sorted(SPLITS)}")
        for name in ("system", "model", "variant"):
            v = getattr(self, name)
            if v != slug(v):
                raise ValueError(f"{name}={v!r} no es válido; usa slug(): {slug(v)!r}")
        if not _STAMP_RE.match(self.stamp):
            raise ValueError(f"stamp {self.stamp!r} no tiene el formato 20261116T1702Z")

    @classmethod
    def new(cls, split, system, model, variant="default", stamp=None):
        return cls(split, slug(system), slug(model), slug(variant), stamp or utc_stamp())

    @property
    def stem(self) -> str:
        return SEP.join([self.split, self.system, self.model, self.variant, self.stamp])


def parse_stem(stem: str) -> PredictionId:
    parts = stem.split(SEP)
    if len(parts) != 5:
        raise ValueError(f"{stem!r} no tiene los 5 campos split/system/model/variant/stamp")
    return PredictionId(*parts)


def prediction_path(pid: PredictionId, predictions_dir="predictions") -> Path:
    return Path(predictions_dir) / f"{pid.stem}.jsonl"


def meta_path(pid: PredictionId, predictions_dir="predictions") -> Path:
    return Path(predictions_dir) / f"{pid.stem}.meta.json"


def metric_path(pid: PredictionId, metric: str, metrics_dir="metrics") -> Path:
    return Path(metrics_dir) / f"{pid.stem}{SEP}{slug(metric)}.json"


def parse_metric_filename(name: str) -> tuple[PredictionId, str]:
    """'<stem>__<metric>.json' -> (PredictionId, metric)."""
    base = Path(name).name
    if not base.endswith(".json"):
        raise ValueError(f"{name!r} no es un .json")
    parts = base[:-len(".json")].split(SEP)
    if len(parts) != 6:
        raise ValueError(f"{name!r} no tiene los 6 campos esperados")
    return PredictionId(*parts[:5]), parts[5]


def metrics_for(pid: PredictionId, metrics_dir="metrics") -> list[Path]:
    """Todas las métricas calculadas para una predicción."""
    return sorted(Path(metrics_dir).glob(f"{pid.stem}{SEP}*.json"))


def ensure_new(path: Path, overwrite: bool = False) -> Path:
    """Evita sobrescribir resultados sin querer."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} ya existe; usa overwrite=True si quieres reemplazarlo")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
