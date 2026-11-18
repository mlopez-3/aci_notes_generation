"""ROUGE-1, ROUGE-2 i ROUGE-L (F1, amb stemming) entre la nota generada i la de referència.

Mesura coincidència de paraules, NO correcció clínica.
"""
from .common import describe, fmt


def metric_name(opts):
    return "rouge-v1"


def run(rows, ctx, opts):
    from rouge_score import rouge_scorer

    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    per_note = []
    for r in rows:
        s = sc.score(r["ref"], r["pred"])
        per_note.append({"file": r["file"], **{k: s[k].fmeasure for k in ("rouge1", "rouge2", "rougeL")}})
    summary = {k: describe(n[k] for n in per_note) for k in ("rouge1", "rouge2", "rougeL")}
    return {"summary": summary, "per_note": per_note, "config": {"stemmer": True, "score": "fmeasure"}}


def headline(s):
    return " | ".join(f"{k} {fmt(s[k]['mean'])}" for k in ("rouge1", "rouge2", "rougeL"))
