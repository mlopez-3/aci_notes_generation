"""BERTScore (precisió, recall i F1) entre la nota generada i la de referència.

Limitació: el model trunca a 512 tokens i les notes en fan uns 600, així que no "veu" el final de la nota.
Mesura semblança de significat general, NO correcció clínica.
"""
from .common import describe, fmt, safe_text


def metric_name(opts):
    return "bertscore-v1"


def run(rows, ctx, opts):
    from bert_score import score

    batch_size = opts.get("bertscore_batch_size", 8)
    P, R, F = score([safe_text(r["pred"]) for r in rows], [r["ref"] for r in rows],
                    lang="en", batch_size=batch_size, verbose=False)
    per_note = [{"file": r["file"], "bertscore_p": p, "bertscore_r": rc, "bertscore_f1": f}
                for r, p, rc, f in zip(rows, P.tolist(), R.tolist(), F.tolist())]
    summary = {k: describe(n[k] for n in per_note) for k in ("bertscore_p", "bertscore_r", "bertscore_f1")}
    return {"summary": summary, "per_note": per_note,
            "config": {"lang": "en", "batch_size": batch_size, "model": "per defecte de bert_score per a 'en'"}}


def headline(s):
    return (f"F1 {fmt(s['bertscore_f1']['mean'])} | precisió {fmt(s['bertscore_p']['mean'])} "
            f"| recall {fmt(s['bertscore_r']['mean'])}")
