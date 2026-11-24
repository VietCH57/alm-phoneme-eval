from __future__ import annotations

from pathlib import Path

import pandas as pd

AUDIO_COLS = ("audio", "Path", "path")
TEXT_COLS = ("text", "Transcript", "transcript")
CANON_COLS = ("canonical", "Canonical")


def load_gold(csvs: list[str], domains: list[str] | None = None, audio_root: str | None = None,
              limit: int = 0) -> pd.DataFrame:
    """CSV(s) with an audio column, a gold phone transcript and optionally domain / canonical.
    Returns columns: audio, text, domain, canonical ('' when absent); one row per audio."""
    frames = []
    for c in csvs:
        df = pd.read_csv(c, dtype=str, keep_default_na=False)
        a = next((x for x in AUDIO_COLS if x in df.columns), None)
        t = next((x for x in TEXT_COLS if x in df.columns), None)
        if a is None or t is None:
            raise SystemExit(f"{c}: need an audio column {AUDIO_COLS} and a transcript column {TEXT_COLS}")
        k = next((x for x in CANON_COLS if x in df.columns), None)
        out = pd.DataFrame({"audio": df[a].str.strip(), "text": df[t],
                            "domain": df["domain"] if "domain" in df.columns else Path(c).stem,
                            "canonical": df[k] if k else "", "source": c})
        frames.append(out)
    df = pd.concat(frames, ignore_index=True)
    df = df[(df.audio != "") & (df.text.str.strip() != "")]
    if domains:
        df = df[df.domain.isin(domains)]
    if audio_root:
        df["audio"] = df.audio.map(lambda p: p if Path(p).is_absolute() else str(Path(audio_root) / p))
    # a canonical column that equals the transcript (almost) everywhere is a copy, not a canonical
    # (the eLearning label files): drop it for that source
    norm = lambda s: s.str.split().str.join(" ")  # noqa: E731
    for src, g in df.groupby("source"):
        has = g.canonical != ""
        if has.any() and (norm(g.canonical[has]) == norm(g.text[has])).mean() > 0.9:
            df.loc[g.index, "canonical"] = ""
    df = df.drop(columns="source")
    df = df.drop_duplicates("audio").reset_index(drop=True)
    if limit:
        df = df.groupby("domain", group_keys=False).head(limit).reset_index(drop=True)
    return df
