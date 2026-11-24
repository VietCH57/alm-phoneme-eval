"""Phone sets, normalisation of gold / predicted phone strings, Levenshtein alignment."""

from __future__ import annotations

import re

# 39 ARPAbet phonemes, no stress (CMUdict set), lower case like our vocab
PHONES39 = ("aa ae ah ao aw ay b ch d dh eh er ey f g hh ih iy jh k l m n ng ow oy p r s sh t th "
            "uh uw v w y z zh").split()
P39 = set(PHONES39)
ALIASES = {"err": "er", "ax": "ah", "axr": "er", "ix": "ih", "ux": "uw", "dx": "t", "nx": "n", "el": "l",
           "em": "m", "en": "n", "hv": "hh", "q": ""}


def gold_tokens(text: str, keep_star: bool = False) -> list[str]:
    """Gold transcript in our vocab ('k*' = distorted k, '[UNK]') -> list of phones.
    keep_star=False maps 'k*' -> 'k' so that systems that cannot output '*' are compared on the 39 phones."""
    out = []
    for t in str(text).split():
        t = t.lower()
        if t in ("[unk]", "[pad]", "|"):
            continue
        star = t.endswith("*")
        base = ALIASES.get(t.rstrip("*"), t.rstrip("*"))
        if not base:
            continue
        out.append(base + ("*" if (star and keep_star) else ""))
    return out


def parse_prediction(text: str, keep_star: bool = False) -> tuple[list[str], int]:
    """Free-form model output -> (39-set phones, number of dropped tokens).
    Accepts 'K AE1 T', 'k ae t', '/k/ /ae/', 'k-ae-t', commas; drops words and unknown symbols."""
    text = re.sub(r"(?s)^.*?(phonemes?|transcription)\s*[:：]", "", text.strip(), count=1, flags=re.I)
    text = re.sub(r"\s+\*", "*", text)  # 'T *' -> 'T*'
    toks = re.split(r"[\s,;/|\-\[\]\(\)\"'`.]+", text)
    out, dropped = [], 0
    for t in toks:
        if not t:
            continue
        t = t.lower()
        star = t.endswith("*")
        t = re.sub(r"[0-2]$", "", t.rstrip("*"))  # stress digit / distortion mark
        t = ALIASES.get(t, t)
        if t in P39:
            out.append(t + ("*" if star and keep_star else ""))
        elif t:
            dropped += 1
    return out, dropped


def align(ref: list[str], hyp: list[str]):
    """Levenshtein alignment (unit costs). Returns (ops, S, D, I) with ops = [(op, ref_tok, hyp_tok)],
    op in {'=', 'S', 'D', 'I'}."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i
    for j in range(1, m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        ri = ref[i - 1]
        row, prev = d[i], d[i - 1]
        for j in range(1, m + 1):
            row[j] = min(prev[j - 1] + (ri != hyp[j - 1]), prev[j] + 1, row[j - 1] + 1)
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("=" if ref[i - 1] == hyp[j - 1] else "S", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(("D", ref[i - 1], None))
            i -= 1
        else:
            ops.append(("I", None, hyp[j - 1]))
            j -= 1
    ops.reverse()
    S = sum(o[0] == "S" for o in ops)
    D = sum(o[0] == "D" for o in ops)
    I = sum(o[0] == "I" for o in ops)
    return ops, S, D, I
