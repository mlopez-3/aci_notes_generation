"""Longitud de la nota generada respecte a la de referència (no necessita cap model).

ratio < 1 suggereix omissions; ratio > 1 suggereix farciment o invencions.
També compta notes buides, tallades pel límit de tokens i raonaments truncats (mode thinking).
"""
from .common import describe, fmt


def metric_name(opts):
    return "length-v1"


def run(rows, ctx, opts):
    per_note = []
    for r in rows:
        wr, wp = len(r["ref"].split()), len(r["pred"].split())
        per_note.append({"file": r["file"], "words_ref": wr, "words_pred": wp,
                         "length_ratio": wp / max(wr, 1)})
    tot_p = sum(n["words_pred"] for n in per_note)
    tot_r = sum(n["words_ref"] for n in per_note)
    summary = {
        "length_ratio": describe(n["length_ratio"] for n in per_note),
        "words_pred_total": tot_p, "words_ref_total": tot_r,
        "length_ratio_global": tot_p / max(tot_r, 1),
        "n_empty_pred": sum(1 for r in rows if not r["pred"].strip()),
        "n_hit_limit": sum(1 for r in rows if r.get("hit_limit")),
        "n_reasoning_truncated": sum(1 for r in rows if r.get("reasoning_truncated")),
    }
    return {"summary": summary, "per_note": per_note, "config": {}}


def headline(s):
    extra = ""
    if s["n_empty_pred"] or s["n_hit_limit"] or s["n_reasoning_truncated"]:
        extra = (f"  | ATENCIÓ: buides={s['n_empty_pred']} tallades={s['n_hit_limit']} "
                 f"reasoning truncat={s['n_reasoning_truncated']}")
    return f"ratio global {fmt(s['length_ratio_global'], 2)} (mitjana {fmt(s['length_ratio']['mean'], 2)}){extra}"
