#!/usr/bin/env python
"""Zero-shot phoneme transcription with Qwen2.5-Omni (thinker only, text output).

For every gold utterance: load 16 kHz audio, cut into <= 28 s pieces (the processor truncates at 30 s),
ask the model for the ARPAbet phonemes actually pronounced, concatenate the pieces' phonemes.
Writes one JSON line per utterance (resumable: utterances already in the output are skipped).

    python scripts/run_qwen_omni.py --csv <dev.csv> <test.csv> --domains el_easy l2a \
        --model Qwen/Qwen2.5-Omni-3B --mode free --out preds/qwen3b_free.jsonl

--mode free       : audio + instruction only
--mode canonical  : also gives the expected (dictionary) phonemes; needs a 'canonical' column
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from almeval import audio as A  # noqa: E402
from almeval.data import load_gold  # noqa: E402
from almeval.phones import PHONES39, parse_prediction  # noqa: E402

SYSTEM = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving "
          "auditory and visual inputs, as well as generating text and speech.")
PHONE_LIST = " ".join(p.upper() for p in PHONES39)
SYMBOLS = {
    "39": "Use only these 39 ARPAbet symbols, without stress numbers: " + PHONE_LIST + ".",
    "79": ("Use these 39 ARPAbet symbols, without stress numbers: " + PHONE_LIST + ". "
           "When a phoneme is recognisable but pronounced in a distorted, non-native way (between two phonemes, "
           "or with a foreign accent quality), write its symbol followed by '*', for example T* or EY*. "
           "Use '*' only for such distorted sounds; a clearly different phoneme is written as that phoneme."),
}
PROMPT_FREE = (
    "You are an expert phonetician. The audio is English read aloud by a non-native learner. "
    "Transcribe the speech into ARPAbet phonemes exactly as they were PRONOUNCED, including mispronunciations, "
    "substitutions, deletions and insertions. Do NOT correct the speaker and do NOT output the dictionary "
    "pronunciation of the words. {symbols} Output only the phonemes separated by single spaces, nothing else.")
PROMPT_CANON = (
    "You are an expert phonetician. The audio is English read aloud by a non-native learner. "
    "The expected (dictionary) pronunciation of the text is:\n{canonical}\n"
    "The speaker may have mispronounced, substituted, deleted or inserted phonemes. Transcribe the phonemes "
    "that were ACTUALLY PRONOUNCED, not the expected ones. {symbols} "
    "Output only the phonemes separated by single spaces, nothing else.")


def pick_shots(a):
    """Few-shot examples (audio + human transcript) from a TRAIN csv: short utterances only."""
    if not a.shots:
        return []
    import random

    df = load_gold([a.shots_csv], a.shots_domains or None, a.audio_root)
    random.seed(a.seed)
    rows = df.sample(frac=1.0, random_state=a.seed).itertuples(index=False)
    shots = []
    for r in rows:
        try:
            wav = A.load(r.audio)
        except Exception:  # noqa: BLE001
            continue
        if len(wav) / A.SR <= a.shot_max_sec:
            from almeval.phones import gold_tokens
            ph = gold_tokens(r.text, keep_star=a.vocab == "79")
            shots.append((wav, " ".join(t.upper() for t in ph), r.audio))
        if len(shots) == a.shots:
            break
    return shots


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", nargs="+", required=True)
    p.add_argument("--domains", nargs="*", default=None)
    p.add_argument("--audio_root", default=None)
    p.add_argument("--model", default="Qwen/Qwen2.5-Omni-3B")
    p.add_argument("--mode", choices=["free", "canonical"], default="free")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0, help="max utterances per domain (0 = all)")
    p.add_argument("--max_sec", type=float, default=28.0)
    p.add_argument("--load_4bit", action="store_true", help="bitsandbytes 4-bit (for 7B on a shared GPU)")
    p.add_argument("--gpu_memory_fraction", type=float, default=0.5)
    p.add_argument("--attn", default="sdpa", help="sdpa | flash_attention_2 | eager")
    p.add_argument("--vocab", choices=["39", "79"], default="39",
                   help="79 = also ask for '*' (distorted phone) like our labels")
    p.add_argument("--shots", type=int, default=0, help="number of in-context examples (audio + human labels)")
    p.add_argument("--shots_csv", default=None, help="TRAIN csv to draw the examples from (never the test set)")
    p.add_argument("--shots_domains", nargs="*", default=None)
    p.add_argument("--shot_max_sec", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if a.shots and not a.shots_csv:
        sys.exit("--shots needs --shots_csv (a TRAIN file)")

    gold = load_gold(a.csv, a.domains, a.audio_root, a.limit)
    if a.mode == "canonical":
        gold = gold[gold.canonical != ""]
        if gold.empty:
            sys.exit("--mode canonical: no row has a real canonical column")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(line)["audio"] for line in out.open(encoding="utf-8") if line.strip()}
    todo = gold[~gold.audio.isin(done)]
    print(f"{len(gold)} utterances ({gold.domain.value_counts().to_dict()}), {len(done)} already done, {len(todo)} to run",
          flush=True)
    if todo.empty:
        return

    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    if torch.cuda.is_available() and a.gpu_memory_fraction:
        torch.cuda.set_per_process_memory_fraction(a.gpu_memory_fraction, torch.cuda.current_device())
    cuda = torch.cuda.is_available()
    kw = dict(torch_dtype=torch.bfloat16, attn_implementation=a.attn, low_cpu_mem_usage=True)
    if a.load_4bit:
        from transformers import BitsAndBytesConfig
        kw.update(device_map="cuda", quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"))
    else:
        kw["device_map"] = "cpu"  # load on CPU, drop the speech generator, then move only the thinker to the GPU
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(a.model, **kw)
    model.disable_talker()  # text output only: the talker / token2wav weights are freed
    thinker = model.thinker
    if cuda and not a.load_4bit:
        thinker.to("cuda")
    thinker.eval()
    proc = Qwen2_5OmniProcessor.from_pretrained(a.model)
    dev = next(thinker.parameters()).device
    print(f"loaded {a.model} on {dev}", flush=True)
    shots = pick_shots(a)
    test_set = set(gold.audio)
    assert not any(x[2] in test_set for x in shots), "a few-shot example is in the evaluated set"
    if shots:
        print(f"{len(shots)} in-context examples: " + ", ".join(Path(x[2]).name for x in shots), flush=True)

    t0 = time.time()
    with out.open("a", encoding="utf-8") as fout:
        for k, r in enumerate(todo.itertuples(index=False), 1):
            rec = dict(audio=r.audio, domain=r.domain, model=a.model, mode=a.mode, vocab=a.vocab, shots=len(shots))
            try:
                wav = A.load(r.audio)
                pieces = A.split(wav, a.max_sec)
                raws, phones, dropped = [], [], 0
                sym = SYMBOLS[a.vocab]
                prompt = (PROMPT_FREE.format(symbols=sym) if a.mode == "free" else
                          PROMPT_CANON.format(symbols=sym, canonical=" ".join(r.canonical.upper().split())))
                for piece in pieces:
                    # with several pieces in canonical mode the whole canonical is given each time;
                    # the model is asked to transcribe only what this piece contains
                    text_prompt = prompt if len(pieces) == 1 else prompt + (
                        " This audio is one part of a longer recording; transcribe only what is heard in it.")
                    conv = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]}]
                    for _, shot_text, _ in shots:  # in-context examples: same instruction, human answer
                        shot_prompt = (PROMPT_FREE.format(symbols=sym) if a.mode == "free" else text_prompt)
                        conv += [{"role": "user", "content": [{"type": "audio", "audio": "shot"},
                                                              {"type": "text", "text": shot_prompt}]},
                                 {"role": "assistant", "content": [{"type": "text", "text": shot_text}]}]
                    conv.append({"role": "user", "content": [{"type": "audio", "audio": "piece"},
                                                             {"type": "text", "text": text_prompt}]})
                    chat = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
                    inputs = proc(text=chat, audio=[x[0] for x in shots] + [piece], return_tensors="pt", padding=True)
                    inputs = inputs.to(dev)
                    for key in ("input_features",):
                        if key in inputs:
                            inputs[key] = inputs[key].to(torch.bfloat16)
                    n_in = inputs["input_ids"].shape[1]
                    max_new = int(len(piece) / A.SR * 40) + 64  # ~12 phones/s, up to ~3 tokens each
                    with torch.no_grad():
                        ids = thinker.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                                               repetition_penalty=1.0)
                    raw = proc.batch_decode(ids[:, n_in:], skip_special_tokens=True)[0].strip()
                    ph, dr = parse_prediction(raw, keep_star=a.vocab == "79")
                    raws.append(raw)
                    phones += ph
                    dropped += dr
                rec.update(pred=" ".join(phones), raw=" ||| ".join(raws), n_pieces=len(pieces),
                           dropped_tokens=dropped, seconds=round(len(wav) / A.SR, 2))
            except Exception as e:  # noqa: BLE001
                rec.update(pred="", raw="", error=f"{type(e).__name__}: {e}")
                print(f"!! {r.audio}: {rec['error']}", flush=True)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            el = time.time() - t0
            print(f"[{k}/{len(todo)}] {r.domain} {Path(r.audio).name} pieces={rec.get('n_pieces')} "
                  f"phones={len(rec['pred'].split())} | {el / k:.1f} s/utt, ETA {el / k * (len(todo) - k) / 60:.0f} min",
                  flush=True)
    print(f"Done -> {out}")


if __name__ == "__main__":
    main()
