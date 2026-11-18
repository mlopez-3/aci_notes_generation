"""BLEURT sobre la nota completa i per secció.

Per defecte Elron/bleurt-base-512: el BLEURT original (Sellam et al., 2020) convertit a BERT estàndard
(només necessita `transformers`). BLEURT-20 (lucadiliello/BLEURT-20) necessita el paquet bleurt-pytorch,
que falla amb transformers 5.x. Els resultats de models diferents NO són comparables (el nom del fitxer
de la mètrica inclou el model).

BLEURT admet com a màxim ~512 tokens per al PAR (referència + candidata). Amb la nota completa només
"veu" el principi de cada nota, per això es calcula també per secció (subjective, objective_exam,
objective_results, assessment_and_plan), on els parells sí solen cabre.
"""
from .common import describe, fmt
from ..data import GROUP_NAMES
from ..sections import split_groups

DEFAULT_MODEL = "Elron/bleurt-base-512"
_SCORERS = {}  # cache: el model es carrega una sola vegada per execució


def metric_name(opts):
    """'Elron/bleurt-base-512' -> 'bleurt-elron-base-512-v1'."""
    model = opts.get("bleurt_model", DEFAULT_MODEL)
    org, _, name = model.partition("/")
    name = name.lower().removeprefix("bleurt-")
    from ..naming import slug
    return f"bleurt-{slug(org)}-{slug(name)}-v1"


def make_scorer(model_name, batch_size, max_length):
    """Retorna scorer(refs, cands) -> (llista de puntuacions, nre. de parells truncats)."""
    key = (model_name, batch_size, max_length)
    if key in _SCORERS:
        return _SCORERS[key]
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Carregant {model_name} a {device} ...")
    if model_name.startswith("lucadiliello/"):
        try:
            from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer
        except ImportError as e:
            raise SystemExit(f"No es pot importar bleurt-pytorch: {e}\n"
                             "Amb transformers 5.x fes servir --bleurt-model Elron/bleurt-base-512.")
        tok = BleurtTokenizer.from_pretrained(model_name)
        model = BleurtForSequenceClassification.from_pretrained(model_name)
    else:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model = model.to(device).eval()

    def scorer(refs, cands):
        scores, n_trunc = [], 0
        for i in range(0, len(refs), batch_size):
            r, c = refs[i:i + batch_size], cands[i:i + batch_size]
            full = tok(r, c, padding=False, truncation=False)["input_ids"]
            n_trunc += sum(len(ids) > max_length for ids in full)
            enc = tok(r, c, padding="longest", truncation=True, max_length=max_length,
                      return_tensors="pt").to(device)
            with torch.no_grad():
                scores += model(**enc).logits.flatten().tolist()
        return scores, n_trunc

    _SCORERS[key] = scorer
    return scorer


def run_with_scorer(rows, scorer):
    """Lògica de la mètrica, separada de la càrrega del model (així es pot provar amb un scorer fals)."""
    safe = lambda p: p if p.strip() else "empty"  # una candidata buida pot trencar el tokenitzador
    note_scores, note_trunc = scorer([r["ref"] for r in rows], [safe(r["pred"]) for r in rows])
    per_note = [{"file": r["file"], "bleurt_note": s, "bleurt_sections": {}} for r, s in zip(rows, note_scores)]

    pairs = {g: [] for g in GROUP_NAMES}
    missing, extra = {g: 0 for g in GROUP_NAMES}, {g: 0 for g in GROUP_NAMES}
    for i, r in enumerate(rows):
        rs, ps = split_groups(r["ref"]), split_groups(r["pred"])
        for g in GROUP_NAMES:
            if g in rs and g in ps:
                pairs[g].append((i, rs[g], ps[g]))
            elif g in rs:
                missing[g] += 1   # la referència té la secció i la generada no
            elif g in ps:
                extra[g] += 1     # la generada té una secció que la referència no té
    sections = {}
    for g in GROUP_NAMES:
        scores, nt = [], 0
        if pairs[g]:
            scores, nt = scorer([p[1] for p in pairs[g]], [p[2] for p in pairs[g]])
            for (i, _, _), s in zip(pairs[g], scores):
                per_note[i]["bleurt_sections"][g] = s
        sections[g] = {**describe(scores), "truncated": nt, "missing_in_pred": missing[g], "only_in_pred": extra[g]}
    summary = {"note": describe(note_scores), "note_truncated": note_trunc, "sections": sections}
    return {"summary": summary, "per_note": per_note}


def run(rows, ctx, opts):
    model = opts.get("bleurt_model", DEFAULT_MODEL)
    batch = opts.get("bleurt_batch_size", 4)
    max_len = opts.get("bleurt_max_length", 512)
    res = run_with_scorer(rows, make_scorer(model, batch, max_len))
    res["config"] = {"model": model, "batch_size": batch, "max_length": max_len}
    return res


def headline(s):
    secs = " ".join(f"{g[:4]}={fmt(d['mean'])}" for g, d in s["sections"].items() if d["n"])
    return (f"nota completa {fmt(s['note']['mean'])} ({s['note_truncated']} parells truncats; fia't més de les "
            f"seccions) | {secs}")
