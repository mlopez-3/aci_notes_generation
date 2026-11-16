"""Lectura de ACI-Bench y partición de cada nota en secciones.

El reparto de encabezados -> grupo es el que se verificó contra la partición oficial del reto
(506 de 508 comparaciones coincidieron; la excepción es D2N030-virtscribe, donde el fichero oficial
tiene un error de corte, no nuestro parser).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

# Nombre en data/raw/aci-bench/  ->  nombre de split del proyecto
DEFAULT_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "aci-bench"

SPLIT_FILES = {
    "train": "train_full.json",
    "valid": "valid_full.json",
    "test1": "clinicalnlp_taskB_test1_full.json",   # test oficial de la Tarea B (MEDIQA-CHAT 2023)
    "test2": "clinicalnlp_taskC_test2_full.json",
    "test3": "clef_taskC_test3_full.json",
}

GROUP_NAMES = ["subjective", "objective_exam", "objective_results", "assessment_and_plan"]
GROUPS = {
    "subjective": ["CC", "CHIEF COMPLAINT", "HISTORY OF PRESENT ILLNESS", "HPI", "SUBJECTIVE",
                   "REVIEW OF SYSTEMS", "MEDICAL HISTORY", "PAST HISTORY",
                   "PAST MEDICAL HISTORY", "SURGICAL HISTORY", "PAST SURGICAL HISTORY",
                   "FAMILY HISTORY", "SOCIAL HISTORY", "BIRTH HISTORY",
                   "MEDICATIONS", "CURRENT MEDICATIONS", "ALLERGIES", "VITALS"],
    "objective_exam": ["PHYSICAL EXAM", "PHYSICAL EXAMINATION", "EXAM", "VITALS REVIEWED"],
    "objective_results": ["RESULTS"],
    "assessment_and_plan": ["ASSESSMENT AND PLAN", "ASSESSMENT", "PLAN", "IMPRESSION", "INSTRUCTIONS"],
}
HEADER_TO_GROUP = {h: g for g, hs in GROUPS.items() for h in hs}


def header_key(line: str) -> str | None:
    """Encabezado normalizado si la línea es SOLO un encabezado conocido (mayúsculas o no, con o sin ':')."""
    s = line.strip()
    if not s or len(s) > 60:
        return None
    key = s.rstrip(":").strip().upper()
    return key if key in HEADER_TO_GROUP else None


def parse_note(text: str) -> list[dict]:
    """Parte una nota en secciones: lista de {header, raw_header, group, text}."""
    sections = []
    cur = {"header": "", "raw_header": "", "group": "OTHER", "lines": []}
    for line in text.splitlines():
        k = header_key(line)
        if k:
            sections.append(cur)
            cur = {"header": k, "raw_header": line.strip(), "group": HEADER_TO_GROUP[k], "lines": []}
        else:
            cur["lines"].append(line)
    sections.append(cur)
    out = []
    for s in sections:
        body = "\n".join(s["lines"]).strip()
        if s["header"] == "" and not body:
            continue  # no había texto antes del primer encabezado
        out.append({"header": s["header"], "raw_header": s["raw_header"], "group": s["group"], "text": body})
    return out


def subset_of(file_id: str) -> str:
    """'D2N071-virtassist' -> 'virtassist' (virtassist | virtscribe | aci)."""
    return file_id.split("-")[-1]


def load_raw(path: str | Path) -> list[dict]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    return obj["data"] if isinstance(obj, dict) and "data" in obj else obj


def load_split(split: str, raw_dir: str | Path | None = None) -> list[dict]:
    """Lee un split directamente del JSON original de ACI-Bench.

    Devuelve una lista de {file, src, tgt}: src = conversación, tgt = nota de referencia.
    Para evaluar por secciones usa parse_note(ex["tgt"]) (o de la nota generada) al vuelo.
    """
    if split not in SPLIT_FILES:
        raise ValueError(f"split desconocido {split!r}; usa uno de {sorted(SPLIT_FILES)}")
    raw_dir = Path(raw_dir) if raw_dir else DEFAULT_RAW_DIR
    path = raw_dir / SPLIT_FILES[split]
    if not path.exists():
        raise FileNotFoundError(f"Falta {path}")
    return [{"file": ex["file"], "src": ex["src"], "tgt": ex["tgt"]} for ex in load_raw(path)]


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
