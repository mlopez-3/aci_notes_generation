"""Avaluació de prediccions: per a cada (predicció, mètrica) desa metrics/<nom-de-la-predicció>__<mètrica>.json."""
from __future__ import annotations

import datetime as _dt
import glob
from pathlib import Path

from . import metrics, naming
from .data import load_split, read_jsonl
from .provenance import git_commit, write_json

ALL_METRICS = ["length", "rouge", "bertscore", "bleurt", "entities"]


def resolve_predictions(specs, predictions_dir="predictions") -> list[Path]:
    """Converteix el que ha escrit l'usuari en fitxers de predicció.

    Cada spec pot ser: "all" (totes), una ruta a un .jsonl, un nom sense extensió
    (valid__zeroshot__...__20261119T1530Z), o un patró amb * (p. ex. "*thinking*").
    """
    pdir = Path(predictions_dir)
    found: list[Path] = []
    for spec in specs:
        if spec == "all":
            found += sorted(pdir.glob("*.jsonl"))
        elif any(c in spec for c in "*?["):
            pattern = spec if ("/" in spec or "\\" in spec) else str(pdir / spec)
            if not pattern.endswith(".jsonl"):
                pattern += ".jsonl"
            matches = sorted(Path(p) for p in glob.glob(pattern))
            if not matches:
                raise FileNotFoundError(f"El patró {spec!r} no coincideix amb cap predicció")
            found += matches
        else:
            cands = [Path(spec), pdir / spec, pdir / f"{spec}.jsonl"]
            hit = next((c for c in cands if c.is_file() and c.suffix == ".jsonl"), None)
            if hit is None:
                raise FileNotFoundError(f"No trobo la predicció {spec!r} (ni a {pdir}/)")
            found.append(hit)
    unique = list(dict.fromkeys(found))
    if not unique:
        raise FileNotFoundError(f"No hi ha cap predicció a {pdir}/")
    return unique


def run_evaluation(prediction_paths, metric_ids, opts=None, metrics_dir="metrics",
                   overwrite=False, repo_dir=".") -> list[Path]:
    """Calcula cada mètrica per a cada predicció. Retorna els fitxers escrits.

    El bucle exterior és la mètrica, perquè els models pesants (BLEURT, NER) es carreguin una sola vegada.
    Si el fitxer de resultat ja existeix es salta (amb overwrite=True es recalcula).
    """
    opts = opts or {}
    written = []
    preds = []
    for p in prediction_paths:
        preds.append((Path(p), naming.parse_stem(Path(p).stem)))  # valida el nom abans de començar

    for metric_id in metric_ids:
        mod = metrics.load(metric_id)
        name = mod.metric_name(opts)
        print(f"\n=== {name} ===")
        for path, pid in preds:
            out = naming.metric_path(pid, name, metrics_dir)
            if out.exists() and not overwrite:
                print(f"  salto {out.name} (ja existeix; --overwrite per recalcular)")
                continue
            rows = read_jsonl(path)
            if not rows:
                print(f"  salto {path.name}: fitxer buit")
                continue
            ctx = {"split": pid.split, "src": {}}
            if getattr(mod, "NEEDS_SOURCE", False):
                ctx["src"] = {ex["file"]: ex["src"] for ex in load_split(pid.split)}
            res = mod.run(rows, ctx, opts)
            payload = {
                "prediction": pid.stem, "metric": name, "n_notes": len(rows),
                "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                "code_version": git_commit(repo_dir),
                "config": res.get("config", {}), "summary": res["summary"], "per_note": res["per_note"],
            }
            naming.ensure_new(out, overwrite=True)
            write_json(out, payload)
            written.append(out)
            print(f"  {pid.stem}\n    {mod.headline(res['summary'])}\n    -> {out}")
    return written
