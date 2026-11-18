"""Partició d'una nota en els 4 grups de seccions (per a notes generades, que poden dur markdown)."""
from __future__ import annotations

import re
from collections import defaultdict

from .data import GROUP_NAMES, HEADER_TO_GROUP


def split_groups(text: str) -> dict[str, str]:
    """Parteix una nota en {grup: text} segons els encapçalaments.

    Tolera markdown (**CHIEF COMPLAINT**, ## PLAN:), que els models generen sovint.
    Només retorna els 4 grups de GROUP_NAMES; el text anterior al primer encapçalament (OTHER) s'ignora.
    """
    buf, cur = defaultdict(list), "OTHER"
    for line in text.splitlines():
        key = re.sub(r"[#*_]", "", line).strip().rstrip(":").strip().upper()
        if len(key) <= 60 and key in HEADER_TO_GROUP:
            cur = HEADER_TO_GROUP[key]
            continue
        buf[cur].append(line)
    out = {}
    for g in GROUP_NAMES:
        t = "\n".join(buf.get(g, [])).strip()
        if t:
            out[g] = t
    return out
