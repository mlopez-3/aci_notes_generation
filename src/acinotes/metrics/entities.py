"""Omissions i al·lucinacions (candidates) amb un model de NER biomèdic  --  versió "entities-v4".

S'extreuen les entitats mèdiques (símptomes, diagnòstics, medicació, proves...) de
  - la nota de REFERÈNCIA (per secció),
  - la nota GENERADA, i
  - la TRANSCRIPCIÓ (només per decidir si una entitat "de més" està recolzada),
i es comparen:

  OMISSIÓ                   entitat de la referència que no apareix a la nota generada
  EXTRA RECOLZADA           entitat generada que no és a la referència però SÍ a la transcripció
                            (probablement un detall correcte que la referència no recull)
  AL·LUCINACIÓ CANDIDATA    entitat generada que no és a la referència NI a la transcripció

Cada omissió es classifica a més com RECOLZADA (l'entitat també és a la transcripció: el model la podia
recollir) o NO RECOLZADA (només és a la referència, p. ex. frases de plantilla que ningú va dir). La xifra que
culpa el model és: omissions recolzades / entitats de la referència recolzades.

Les xifres d'aquesta versió només es poden comparar amb les d'aquesta mateixa versió (entities-v4).

LIMITACIONS:
  * Les "al·lucinacions" són CANDIDATES: cal llegir la llista. Un sinònim ("Lasix" / "furosemide") compta
    com a discrepància encara que sigui correcte.
  * El NER no distingeix la negació: "denies chest pain" i "chest pain" donen la mateixa entitat, així que una
    negació invertida NO es detecta amb aquesta mètrica.
  * Les entitats es comparen com a CONJUNTS per nota: una dosi no s'associa al seu fàrmac.
  * La referència és UNA nota; no recull tot el que és cert. Per això es mira també la transcripció.
  * "Recolzada" es decideix per coincidència de text amb la transcripció: una paràfrasi pot quedar com a
    NO recolzada i un concepte molt genèric, com a recolzat.
  * El model (d4data/biomedical-ner-all) es va entrenar amb casos clínics escrits, no amb notes de consulta.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher

from .common import describe, fmt
from ..data import GROUP_NAMES, parse_note

NER_MODEL = "d4data/biomedical-ner-all"
# Etiquetes per defecte (en minúscules); es poden canviar amb --ner-keep ("all" les accepta totes).
DEFAULT_KEEP = ("sign_symptom,disease_disorder,medication,diagnostic_procedure,"
                "therapeutic_procedure,lab_value,dosage")
NEEDS_SOURCE = True  # necessita la transcripció (ctx["src"][file])

ALL_GROUPS = GROUP_NAMES + ["OTHER"]
HEADER_RE = re.compile(r"^[A-Z][A-Z &/,\-:]{2,60}$")
TAG_RE = re.compile(r"^\[[^\]]+\]\s*")      # [doctor] / [patient]
WORD_CHAR = re.compile(r"[A-Za-z0-9\-/]")   # per completar fragments de paraula ("yspnea" -> "dyspnea")

# Paraules massa genèriques per comptar com a entitat (el NER les etiqueta de vegades).
GENERIC = {
    "symptom", "sign", "medication", "drug", "examination", "exam", "normal", "abnormal",
    "elevated", "decreased", "increased", "blood", "feeling", "test", "result",
    "review of system", "review of system sheet",
    "improvement", "improved", "stable", "adequate", "systemic", "specified", "slight",
    "well controlled", "under control", "managed well", "move", "monitor", "portal",
    "too much", "able to exercise", "struggle a little bit with",
    # etiquetes de sistemes de l'exploració física (no són troballes)
    "cardiovascular", "respiratory", "neurological", "musculoskeletal", "gastrointestinal",
    "genitourinary", "constitutional", "psychiatric", "heart exam", "vital sign",
}

NUM_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
NUM_WORD_RE = re.compile(r"\b(" + "|".join(NUM_WORDS) + r")\b")
UNIT_TOKENS = {"mg", "mcg", "ml", "cm", "mmhg", "kg", "lb"}  # s'ignoren en buscar un valor a la transcripció

SURFACE: dict[str, str] = {}  # text normalitzat -> forma original (només per mostrar)
_PIPE = None                   # pipeline de NER, es carrega una sola vegada


# ---------- Normalització i aparellament ----------
def norm_token(w):
    """Normalització tosca d'una paraula (s'aplica igual als dos costats: només cal que sigui consistent)."""
    if w.isdigit():
        return w
    for suf in ("ness", "ing", "ed"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            w = w[: -len(suf)]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    return w


def normalize(text):
    t = text.lower()
    t = NUM_WORD_RE.sub(lambda m: str(NUM_WORDS[m.group(1)]), t)          # "three" -> "3"
    t = re.sub(r"\b(\d) hundred\b", lambda m: m.group(1) + "00", t)       # "8 hundred" -> "800"
    t = re.sub(r"(\d+)\s+out\s+of\s+(\d+)", r"\1 \2", t)                  # "3 out of 6" -> "3 6"
    t = re.sub(r"(\d+)\s+over\s+(\d+)", r"\1 \2", t)                      # "120 over 80" -> "120 80"
    t = re.sub(r"\bmilligrams?\b", "mg", t)                               # unitats dites amb paraules
    t = re.sub(r"\bmicrograms?\b", "mcg", t)
    t = re.sub(r"\bmillilit(?:er|re)s?\b", "ml", t)
    t = re.sub(r"\bcentimet(?:er|re)s?\b", "cm", t)
    t = re.sub(r"\bmillimet(?:er|re)s? of mercury\b", "mmhg", t)
    t = re.sub(r"\b(?:once\s+)?(?:a|per|each|every)\s+day\b", " daily", t)  # "a day", "once a day" -> "daily"
    t = re.sub(r"\bonce\s+daily\b", " daily", t)
    t = re.sub(r"\btwice\s+(?:a\s+day|daily)\b", " twice daily", t)
    t = re.sub(r"[^a-z0-9. ]+", " ", t)
    t = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", t)  # treu els punts excepte els decimals
    return " ".join(norm_token(w) for w in t.split())


def normalize_entity(text):
    words = normalize(text).split()
    while words and words[0] in {"a", "an", "the"}:
        words.pop(0)
    while words and words[-1] in {"a", "an", "the"}:
        words.pop()
    return " ".join(words)


GENERIC_NORM = {normalize_entity(g) for g in GENERIC}


def numbers(t):
    return re.findall(r"\d+(?:\.\d+)?", t)


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def numbers_conflict(a, b):
    """Hi ha conflicte de valors si AMBDUES entitats porten xifres i cap llista està inclosa a l'altra:
    {10} vs {20}, {124, 80} vs {120, 80}. Però {600} vs {600, 6} ("600 mg" / "600 mg every 6 hours") NO."""
    sa, sb = set(a), set(b)
    return bool(sa) and bool(sb) and not (sa <= sb or sb <= sa)


def is_match(text, pool, thr):
    """Dues entitats coincideixen si són iguals; o una és continguda a l'altra ("fluid" / "fluid retention",
    "20 mg" / "20 mg daily"); o són molt semblants com a text. Si hi ha conflicte de xifres
    ("10 mg" / "20 mg", "124/80" / "120/80") MAI coincideixen: un canvi de valor s'ha de veure."""
    if text in pool:
        return True
    nums, tt = numbers(text), set(text.split())
    for p in pool:
        pn = numbers(p)
        if numbers_conflict(nums, pn):
            continue
        pt = set(p.split())
        if len(tt) <= len(pt):
            short, long_, short_nums = tt, pt, nums
        else:
            short, long_, short_nums = pt, tt, pn
        if short and short <= long_ and (short_nums or any(len(w) >= 4 for w in short)):
            return True
        if nums == pn and similar(text, p) >= thr:
            return True
    return False


def supported_by_src(t, src_text_norm, src_ents, thr):
    """L'entitat apareix a la transcripció? Per text (amb o sense la unitat) o per coincidència amb les seves entitats."""
    if src_text_norm is None:
        return False
    hay = f" {src_text_norm} "
    core = " ".join(w for w in t.split() if w not in UNIT_TOKENS)
    return (f" {t} " in hay) or bool(core and f" {core} " in hay) or is_match(t, src_ents, thr)


def compare(ref, pred, src_text_norm, src_ents, thr):
    """ref, pred: dict {text_normalitzat: etiqueta}. src_text_norm: transcripció normalitzada. src_ents: set de textos."""
    ref_pool, pred_pool = set(ref), set(pred)
    omitted = [(t, l) for t, l in ref.items() if not is_match(t, pred_pool, thr)]
    unmatched = [(t, l) for t, l in pred.items() if not is_match(t, ref_pool, thr)]
    extra_ok, halluc = [], []
    for t, l in unmatched:
        (extra_ok if supported_by_src(t, src_text_norm, src_ents, thr) else halluc).append((t, l))
    ref_sup = [t for t in ref if supported_by_src(t, src_text_norm, src_ents, thr)]
    ref_sup_set = set(ref_sup)
    omitted_sup = [t for t, _ in omitted if t in ref_sup_set]
    n_ref, n_pred = len(ref), len(pred)
    return {
        "n_ref": n_ref, "n_pred": n_pred,
        "omitted": omitted, "extra_supported": extra_ok, "hallucination_candidates": halluc,
        "ref_supported": ref_sup, "n_ref_supported": len(ref_sup), "omitted_supported": omitted_sup,
        "omission_rate": len(omitted) / n_ref if n_ref else None,
        "omission_rate_supported": len(omitted_sup) / len(ref_sup) if ref_sup else None,
        "entity_recall": (n_ref - len(omitted)) / n_ref if n_ref else None,
        "entity_precision": (n_pred - len(unmatched)) / n_pred if n_pred else None,
        "hallucination_rate": len(halluc) / n_pred if n_pred else None,
    }


# ---------- Extracció d'entitats ----------
def chunks(text, is_transcript=False, max_chars=1200):
    """Parteix el text en trossos curts (DistilBERT admet 512 tokens). Treu encapçalaments i etiquetes de parlant."""
    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s or HEADER_RE.match(s):
            continue
        if is_transcript:
            s = TAG_RE.sub("", s)
        while len(s) > max_chars:
            cut = s.rfind(". ", 0, max_chars)
            cut = cut + 1 if cut > 0 else max_chars
            out.append(s[:cut].strip())
            s = s[cut:].strip()
        if s:
            out.append(s)
    return out


def expand_span(text, start, end):
    """El NER de vegades marca només un tros de paraula ('yspnea', 'ema'): es completa fins als límits de la paraula."""
    while start > 0 and WORD_CHAR.match(text[start - 1]):
        start -= 1
    while end < len(text) and WORD_CHAR.match(text[end]):
        end += 1
    return text[start:end].strip("-/ ")


def extract(pipe, text, keep, min_score, is_transcript=False):
    """Retorna (dict {text_normalitzat: etiqueta}, Counter d'etiquetes) de les entitats que són a `keep`."""
    parts = chunks(text, is_transcript)
    ents, all_labels = {}, Counter()
    if not parts:
        return ents, all_labels
    results = pipe(parts, batch_size=16)
    for part, res in zip(parts, results):
        for e in res:
            label = e["entity_group"].lower()
            all_labels[label] += 1
            if e["score"] < min_score:
                continue
            if keep is not None and label not in keep:
                continue
            word = expand_span(part, e["start"], e["end"]) if "start" in e else e["word"]
            t = normalize_entity(word)
            if len(t) >= 3 and t not in GENERIC_NORM:
                ents.setdefault(t, label)
                SURFACE.setdefault(t, word.lower())
    return ents, all_labels


def extract_ref(pipe, ref_text, sections, keep, min_score):
    """Entitats de la referència, recordant en quina secció apareixen per primera vegada.
    sections: llista de dicts {group, text} (sortida de parse_note) o None."""
    ents, groups, labels = {}, {}, Counter()
    if sections:
        for g in ALL_GROUPS:
            txt = "\n\n".join(s["text"] for s in sections if s["group"] == g and s["text"])
            if not txt:
                continue
            e, lab = extract(pipe, txt, keep, min_score)
            labels.update(lab)
            for t, l in e.items():
                if t not in ents:
                    ents[t], groups[t] = l, g
    else:
        e, lab = extract(pipe, ref_text, keep, min_score)
        labels.update(lab)
        for t, l in e.items():
            ents[t], groups[t] = l, "sin_seccion"
    return ents, groups, labels


def load_ner():
    global _PIPE
    if _PIPE is None:
        import torch
        from transformers import pipeline
        print(f"Carregant {NER_MODEL} ...")
        _PIPE = pipeline("ner", model=NER_MODEL, aggregation_strategy="simple",
                         device=0 if torch.cuda.is_available() else -1)
    return _PIPE


# ---------- Resum i mètrica ----------
def disp(t):
    return SURFACE.get(t, t)


def has_digit(t):
    return bool(re.search(r"\d", t))


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _ratio(a, b):
    return a / b if b else None


def summarize(items):
    """items: resultats de compare() (+ 'subset'). Retorna les xifres micro (sumant entitats) i la mitjana per nota."""
    n_ref = sum(x["n_ref"] for x in items)
    n_pred = sum(x["n_pred"] for x in items)
    n_om = sum(len(x["omitted"]) for x in items)
    n_ref_sup = sum(x["n_ref_supported"] for x in items)
    n_om_sup = sum(len(x["omitted_supported"]) for x in items)
    n_extra = sum(len(x["extra_supported"]) for x in items)
    n_h = sum(len(x["hallucination_candidates"]) for x in items)
    return {
        "n_notes": len(items), "n_ref_entities": n_ref, "n_pred_entities": n_pred,
        "omissions": n_om, "omission_rate": _ratio(n_om, n_ref),
        "omission_rate_mean_per_note": _mean(x["omission_rate"] for x in items),
        "n_ref_supported": n_ref_sup, "omissions_supported": n_om_sup,
        "omission_rate_supported": _ratio(n_om_sup, n_ref_sup),   # <- la xifra que culpa el model
        "omissions_unsupported": n_om - n_om_sup,
        "omission_rate_unsupported": _ratio(n_om - n_om_sup, n_ref - n_ref_sup),
        "extra_supported": n_extra, "extra_supported_rate": _ratio(n_extra, n_pred),
        "hallucination_candidates": n_h, "hallucination_rate": _ratio(n_h, n_pred),
        "hallucination_rate_mean_per_note": _mean(x["hallucination_rate"] for x in items),
    }


def run(rows, ctx, opts):
    keep_opt = opts.get("ner_keep", DEFAULT_KEEP)
    keep = None if str(keep_opt).strip().lower() == "all" else {k.strip().lower() for k in keep_opt.split(",")}
    min_score = opts.get("ner_min_score", 0.5)
    thr = opts.get("ner_threshold", 0.85)
    pipe = load_ner()

    labels_seen, results, per_note = Counter(), [], []
    for r in rows:
        if r["file"] not in ctx["src"]:
            raise KeyError(f"{r['file']} no és al split {ctx['split']}")
        src = ctx["src"][r["file"]]
        ref, ref_group, lab_r = extract_ref(pipe, r["ref"], parse_note(r["ref"]), keep, min_score)
        pred, lab_p = extract(pipe, r["pred"], keep, min_score)
        src_ents, _ = extract(pipe, src, keep, min_score, is_transcript=True)
        src_norm = normalize(" ".join(TAG_RE.sub("", l) for l in src.splitlines()))
        labels_seen.update(lab_r)
        labels_seen.update(lab_p)
        res = compare(ref, pred, src_norm, set(src_ents), thr)
        res["subset"] = r["file"].split("-")[-1]
        res["ref_labels"], res["ref_group"] = ref, ref_group
        results.append(res)

        osup = set(res["omitted_supported"])
        per_note.append({
            "file": r["file"], "subset": res["subset"],
            "n_ref": res["n_ref"], "n_pred": res["n_pred"], "n_ref_supported": res["n_ref_supported"],
            "n_omitted": len(res["omitted"]), "n_omitted_supported": len(osup),
            "n_extra_supported": len(res["extra_supported"]),
            "n_hallucination_candidates": len(res["hallucination_candidates"]),
            "omission_rate": res["omission_rate"], "omission_rate_supported": res["omission_rate_supported"],
            "hallucination_rate": res["hallucination_rate"],
            "entity_recall": res["entity_recall"], "entity_precision": res["entity_precision"],
            # llistes llegibles per revisar a mà (formes originals, no normalitzades)
            "omitted": [{"entity": disp(t), "label": l, "section": ref_group.get(t, "?"),
                         "supported_by_transcript": t in osup, "has_digits": has_digit(t)}
                        for t, l in res["omitted"]],
            "hallucination_candidates": [{"entity": disp(t), "label": l, "has_digits": has_digit(t)}
                                         for t, l in res["hallucination_candidates"]],
            "extra_supported": [{"entity": disp(t), "label": l} for t, l in res["extra_supported"]],
        })

    by_subset = defaultdict(list)
    for x in results:
        by_subset[x["subset"]].append(x)
    tot, om, tot_s, om_s = Counter(), Counter(), Counter(), Counter()
    tot_g, om_g, tot_gs, om_gs = Counter(), Counter(), Counter(), Counter()
    for x in results:
        sup, osup = set(x["ref_supported"]), set(x["omitted_supported"])
        for t, l in x["ref_labels"].items():
            g = x["ref_group"].get(t, "?")
            tot[l] += 1
            tot_g[g] += 1
            if t in sup:
                tot_s[l] += 1
                tot_gs[g] += 1
        for t, l in x["omitted"]:
            g = x["ref_group"].get(t, "?")
            om[l] += 1
            om_g[g] += 1
            if t in osup:
                om_s[l] += 1
                om_gs[g] += 1
    breakdown = lambda total, omit, total_s, omit_s: {
        k: {"ref": total[k], "omitted": omit[k], "omission_rate": _ratio(omit[k], total[k]),
            "ref_supported": total_s[k], "omitted_supported": omit_s[k],
            "omission_rate_supported": _ratio(omit_s[k], total_s[k])} for k in total}

    summary = {
        "total": summarize(results),
        "by_subset": {s: summarize(items) for s, items in sorted(by_subset.items())},
        "omissions_by_label": breakdown(tot, om, tot_s, om_s),
        "omissions_by_section": breakdown(tot_g, om_g, tot_gs, om_gs),
        "ner_labels_seen": dict(labels_seen.most_common()),
    }
    config = {"ner_model": NER_MODEL, "keep": sorted(keep) if keep else "all",
              "min_score": min_score, "threshold": thr}
    return {"summary": summary, "per_note": per_note, "config": config}


def headline(s):
    t = s["total"]
    return (f"omissions recolzades {fmt(t['omission_rate_supported'])} "
            f"({t['omissions_supported']}/{t['n_ref_supported']}) | omissions totals {fmt(t['omission_rate'])} "
            f"| al·lucinacions candidates {fmt(t['hallucination_rate'])} "
            f"({t['hallucination_candidates']}/{t['n_pred_entities']})")
