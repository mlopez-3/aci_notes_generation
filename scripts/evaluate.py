"""Avalua prediccions: tria les mètriques i les prediccions per paràmetre.

Per a cada predicció i mètrica desa metrics/<nom-de-la-predicció>__<mètrica>.json (el mateix nom que la predicció).
Si el resultat ja existeix, el salta (--overwrite per recalcular).

Mètriques: length, rouge, bertscore, bleurt, entities   (o "all")
Prediccions: "all", un nom (sense .jsonl), una ruta, o un patró amb * (p. ex. "*thinking*")

Exemples:
  python scripts/evaluate.py --metrics all --predictions all
  python scripts/evaluate.py --metrics rouge entities --predictions valid__zeroshot__qwen3-4b-instruct-2507__nothink-4bit-bf16__20261119T1530Z
  python scripts/evaluate.py --metrics rouge bertscore --predictions "*thinking*"
  python scripts/evaluate.py --metrics bleurt --bleurt-model Elron/bleurt-large-512 --predictions all
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from acinotes.evaluate import ALL_METRICS, resolve_predictions, run_evaluation  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics", nargs="+", required=True, choices=ALL_METRICS + ["all"],
                    help="mètriques a calcular, o 'all'")
    ap.add_argument("--predictions", nargs="+", required=True,
                    help="'all', noms de predicció, rutes o patrons amb *")
    ap.add_argument("--predictions-dir", default=str(ROOT / "predictions"))
    ap.add_argument("--metrics-dir", default=str(ROOT / "metrics"))
    ap.add_argument("--overwrite", action="store_true", help="recalcula resultats que ja existeixen")
    ap.add_argument("--bertscore-batch-size", type=int, default=8)
    ap.add_argument("--bleurt-model", default="Elron/bleurt-base-512")
    ap.add_argument("--bleurt-batch-size", type=int, default=4)
    ap.add_argument("--bleurt-max-length", type=int, default=512)
    ap.add_argument("--ner-keep", default=None, help="etiquetes del NER separades per comes, o 'all'")
    ap.add_argument("--ner-min-score", type=float, default=0.5)
    ap.add_argument("--ner-threshold", type=float, default=0.85)
    a = ap.parse_args()

    metric_ids = ALL_METRICS if "all" in a.metrics else list(dict.fromkeys(a.metrics))
    opts = {"bertscore_batch_size": a.bertscore_batch_size, "bleurt_model": a.bleurt_model,
            "bleurt_batch_size": a.bleurt_batch_size, "bleurt_max_length": a.bleurt_max_length,
            "ner_min_score": a.ner_min_score, "ner_threshold": a.ner_threshold}
    if a.ner_keep:
        opts["ner_keep"] = a.ner_keep

    preds = resolve_predictions(a.predictions, a.predictions_dir)
    print(f"{len(preds)} prediccions x {len(metric_ids)} mètriques: {', '.join(metric_ids)}")
    for p in preds:
        print(f"  - {p.name}")
    written = run_evaluation(preds, metric_ids, opts, a.metrics_dir, a.overwrite, ROOT)
    print(f"\nFet: {len(written)} fitxers nous a {a.metrics_dir}")


if __name__ == "__main__":
    main()
