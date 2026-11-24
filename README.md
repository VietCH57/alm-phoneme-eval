# Audio-LLM phoneme labelling: does it agree with human labels?

Zero-shot ARPAbet phoneme transcription of L2 English with Qwen2.5-Omni, scored against human
phone labels (eLearning, L2-ARCTIC) before any use as pseudo-labels on unlabelled data.

```
scripts/run_qwen_omni.py   audio -> phonemes (JSONL, resumable); long audio cut into <= 28 s pieces
scripts/evaluate.py        PER (39 phones), S/D/I, confusions; MDD metrics when a real canonical exists
src/almeval/               phones (normalisation, alignment), audio (load 16 kHz, split), data (gold CSVs)
```

Gold CSVs: an audio column (`audio`/`Path`), a transcript (`text`/`Transcript`, our vocab: `k*` = distorted k),
optional `domain`, optional `canonical`. A canonical identical to the transcript (eLearning label files) is ignored.
All systems are compared on 39 ARPAbet phones: `*` is stripped from the gold (no LLM can output it).

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
D=/home/mdd/lamnh/MDD-2026/data/stage2_combined/server
.venv/bin/python scripts/run_qwen_omni.py --csv $D/fullnative_el2a_test_ctc.csv --domains el_easy l2a \
    --model Qwen/Qwen2.5-Omni-3B --mode free --out preds/qwen3b_free.jsonl
.venv/bin/python scripts/evaluate.py --gold $D/fullnative_el2a_test_ctc.csv --domains el_easy l2a \
    --pred qwen3b=preds/qwen3b_free.jsonl --out reports/qwen3b_free
```
Other systems (e.g. our PR model) are added as `name=file.csv` with columns `audio,pred`.

Variants:
- `--vocab 79`: also ask for `*` (distorted phone), like our labels; score with `evaluate.py --keep_star`
  (then `k*` and `k` are different phones, as in the PR repo's PER).
- `--shots N --shots_csv <TRAIN csv> --shots_domains l2a`: N in-context examples (short audio + human labels)
  before each test utterance; examples are checked not to be in the evaluated set.
