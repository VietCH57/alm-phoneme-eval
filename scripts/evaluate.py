#!/usr/bin/env python
"""Compare predicted phone sequences with the gold transcripts.

Predictions: JSONL from run_qwen_omni.py, or any CSV with columns audio + pred (e.g. our PR model's
inference output), given as name=path. All systems are scored on the 39-phone set: '*' (distortion
marks) are stripped from the gold, stress digits from the predictions.

Reported per system x domain: PER, substitution / deletion / insertion rates, empty outputs, length ratio
(predicted / gold phones), and the most frequent confusions. When the gold has a real 'canonical' column
(not a copy of the transcript), also: a canonical-only baseline (outputs the canonical, never listens),
how close each system stays to the canonical, and MDD detection P/R/F1 + diagnosis accuracy.

    python scripts/evaluate.py --gold <dev.csv> <test.csv> --domains el_easy l2a \
        --pred qwen3b=preds/qwen3b_free.jsonl os2=preds/os2_test.csv --out reports/qwen3b
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from almeval.data import load_gold  # noqa: E402
from almeval.phones import align, gold_tokens, parse_prediction  # noqa: E402


def load_pred(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".jsonl":
        rows = [json.loads(line) for line in p.open(encoding="utf-8") if line.strip()]
        df = pd.DataFrame(rows)
    else:
        df = pd.read_csv(p, dtype=str, keep_default_na=False)
        col = next((c for c in ("pred", "hyp", "prediction", "pred_text") if c in df.columns), None)
        if col is None:
            raise SystemExit(f"{path}: need a column pred/hyp/prediction")
        df = df.rename(columns={col: "pred"})
    if "error" not in df.columns:
        df["error"] = ""
    return df[["audio", "pred", "error"]].fillna("").drop_duplicates("audio", keep="last")


def on_canonical(ops):
    """For each canonical phone, the aligned phone of the other sequence (None = deleted)."""
    return [h for op, r, h in ops if op != "I"]


def score(gold: pd.DataFrame, pred: dict, name: str, keep_star: bool = False):
    rows, conf = [], Counter()
    for g in gold.itertuples(index=False):
        ref = gold_tokens(g.text, keep_star=keep_star)
        has = g.audio in pred
        hyp = parse_prediction(pred[g.audio], keep_star=keep_star)[0] if has else []
        ops, S, D, I = align(ref, hyp)
        for op, r, h in ops:
            if op == "S":
                conf[f"{r}→{h}"] += 1
            elif op == "D":
                conf[f"{r}→∅"] += 1
            elif op == "I":
                conf[f"∅→{h}"] += 1
        row = dict(system=name, audio=g.audio, domain=g.domain, has_pred=has, N=len(ref), S=S, D=D, I=I,
                   hyp_len=len(hyp))
        can = gold_tokens(g.canonical) if g.canonical else []  # '*' never belongs in a canonical
        if can:
            ref_star = gold_tokens(g.text, keep_star=True)  # a distorted phone ('k*') is an error
            gops = on_canonical(align(can, ref_star)[0])
            pops = on_canonical(align(can, hyp)[0])
            ta = fr = fa = tr = diag = 0
            for c, gp, pp in zip(can, gops, pops):
                g_err, p_err = gp != c, pp != c
                if g_err and p_err:
                    tr += 1
                    diag += gp == pp
                elif g_err:
                    fa += 1
                elif p_err:
                    fr += 1
                else:
                    ta += 1
            _, s2, d2, i2 = align(can, hyp)
            row.update(TA=ta, FR=fr, FA=fa, TR=tr, DIAG=diag, canon_N=len(can), canon_dist=s2 + d2 + i2)
        rows.append(row)
    return pd.DataFrame(rows), conf


def summarize(per: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (s, d), g in list(per.groupby(["system", "domain"])) + [((s, "ALL"), g) for s, g in per.groupby("system")]:
        e = g[g.has_pred]
        N = e.N.sum()
        if N == 0:
            out.append(dict(system=s, domain=d, utts=len(g), with_pred=0))
            continue
        r = dict(system=s, domain=d, utts=len(g), with_pred=len(e), empty=int((e.hyp_len == 0).sum()),
                 PER=100 * (e.S + e.D + e.I).sum() / max(N, 1), sub=100 * e.S.sum() / max(N, 1),
                 dele=100 * e.D.sum() / max(N, 1), ins=100 * e.I.sum() / max(N, 1),
                 len_ratio=e.hyp_len.sum() / max(N, 1))
        if "TR" in e and e.TR.notna().any():
            c = e.dropna(subset=["TR"])
            TR, FR, FA, TA = c.TR.sum(), c.FR.sum(), c.FA.sum(), c.TA.sum()
            prec = TR / max(TR + FR, 1)
            rec = TR / max(TR + FA, 1)
            r.update(det_P=100 * prec, det_R=100 * rec, det_F1=100 * 2 * prec * rec / max(prec + rec, 1e-9),
                     FRR=100 * FR / max(FR + TA, 1), FAR=100 * FA / max(FA + TR, 1),
                     diag_acc=100 * c.DIAG.sum() / max(TR, 1),
                     dist_to_canon=100 * c.canon_dist.sum() / max(c.canon_N.sum(), 1))
        out.append(r)
    return pd.DataFrame(out).round(2)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gold", nargs="+", required=True)
    p.add_argument("--domains", nargs="*", default=None)
    p.add_argument("--audio_root", default=None)
    p.add_argument("--pred", nargs="+", required=True, help="name=path (.jsonl or .csv with audio,pred)")
    p.add_argument("--out", default="reports/eval")
    p.add_argument("--only_common", action="store_true", help="score only utterances every system predicted")
    p.add_argument("--keep_star", action="store_true",
                   help="score on our full vocab: 'k*' and 'k' are different phones (as in the PR repo's PER)")
    a = p.parse_args()

    gold = load_gold(a.gold, a.domains, a.audio_root)
    systems = {}
    for spec in a.pred:
        name, _, path = spec.partition("=")
        systems[name] = load_pred(path)
    if a.only_common:
        common = set(gold.audio)
        for df in systems.values():
            common &= set(df.audio[df.error == ""])
        gold = gold[gold.audio.isin(common)]
    has_canon = (gold.canonical != "").any()
    preds = {n: dict(zip(df.audio, df.pred)) for n, df in systems.items()}
    if has_canon:
        preds["canonical_only"] = dict(zip(gold.audio, gold.canonical))
        # upper bound for any 39-phone system: the gold itself with its '*' marks removed. Its FAR is the
        # share of real errors that are distortions only, which no 39-phone output can flag.
        preds["gold_39_upper_bound"] = dict(zip(gold.audio, gold.text))

    per, confs = [], {}
    for n, pr in preds.items():
        df, conf = score(gold, pr, n, a.keep_star)
        per.append(df)
        confs[n] = conf
    per = pd.concat(per, ignore_index=True)
    summ = summarize(per)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    per.to_csv(out / "per_utterance.csv", index=False)
    summ.to_csv(out / "summary.csv", index=False)
    md = ["# Phoneme transcription evaluation", "",
          f"Gold: {', '.join(a.gold)} · domains: {', '.join(sorted(gold.domain.unique()))} · "
          f"{len(gold)} utterances · " + ("full vocab ('k*' != 'k', like the PR repo)" if a.keep_star
                                          else "39-phone set ('*' stripped from gold and predictions)"), "",
          "PER = (S+D+I)/N vs. the gold transcript; len_ratio = predicted phones / gold phones; "
          "empty = utterances with no phone in the output.", "",
          summ.to_markdown(index=False), ""]
    if has_canon:
        md += ["MDD: each canonical phone is correct or mispronounced in the gold (a '*' distortion counts as "
               "mispronounced) and in the prediction; det_P/R/F1 = detection of mispronounced phones, FRR = correct "
               "phones flagged, FAR = mispronounced phones accepted, diag_acc = right phone among detected errors, "
               "dist_to_canon = PER of the output against the canonical (low = the system just repeats the canonical). "
               "`canonical_only` outputs the canonical without listening; `gold_39_upper_bound` is the gold with "
               "'*' removed (best possible 39-phone output).", ""]
    else:
        md += ["No real canonical in the gold files: MDD metrics and the canonical-only baseline are not computed.", ""]
    for n, conf in confs.items():
        if n == "gold_39_upper_bound":
            continue
        top = pd.DataFrame(conf.most_common(15), columns=["error", "count"])
        md += [f"## Most frequent errors: {n}", "", top.to_markdown(index=False), ""]
    for n, df in systems.items():
        errs = df[df.error != ""]
        if len(errs):
            md += [f"## {n}: {len(errs)} utterances failed", "", errs.head(10).to_markdown(index=False), ""]
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:8]))
    print(f"\nWritten: {out}/report.md, summary.csv, per_utterance.csv")


if __name__ == "__main__":
    main()
