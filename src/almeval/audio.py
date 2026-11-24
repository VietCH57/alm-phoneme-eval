"""16 kHz mono loading and splitting of long recordings at quiet points.

Qwen2.5-Omni's processor pads/truncates audio to 30 s, so longer utterances (eLearning ~34 s median,
up to ~57 s) are cut into <= max_sec pieces; the predicted phones of the pieces are concatenated."""

from __future__ import annotations

from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR = 16000


def load(path: str) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != SR:
        g = gcd(sr, SR)
        wav = resample_poly(wav, SR // g, sr // g).astype(np.float32)
    return wav


def split(wav: np.ndarray, max_sec: float = 28.0, min_sec: float = 18.0, win: float = 0.05) -> list[np.ndarray]:
    """Cut into pieces of at most max_sec; each cut is placed at the lowest-energy 50 ms window
    between min_sec and max_sec after the previous cut (a pause, ideally)."""
    if len(wav) <= max_sec * SR:
        return [wav]
    pieces, start = [], 0
    hop = int(win * SR)
    while len(wav) - start > max_sec * SR:
        lo, hi = start + int(min_sec * SR), start + int(max_sec * SR)
        seg = wav[lo:hi]
        n = len(seg) // hop
        energy = (seg[: n * hop].reshape(n, hop) ** 2).mean(axis=1)
        cut = lo + int(np.argmin(energy)) * hop + hop // 2
        pieces.append(wav[start:cut])
        start = cut
    pieces.append(wav[start:])
    return pieces
