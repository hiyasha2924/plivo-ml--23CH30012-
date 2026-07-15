# plivo-ml — 23CH30012

**2,000-Step LLM Speedrun.** Improve a deliberately-mediocre from-scratch GPT so it scores the lowest
**bits-per-byte (bpb)** on a held-out English+Hindi text, under hard caps: ≤2000 optimizer steps,
≤2,000,000 params, `train_corpus.txt` only, pure PyTorch/numpy/stdlib, CPU only.

## Result

| | bpb | params | steps |
|---|---|---|---|
| Starter baseline | 2.3718 | 1,339,840 | 2000 |
| **Final (this repo)** | **1.9996** | 1,585,600 | 2000 |

−0.372 bpb (−15.7%) over the baseline.

## What changed (the two wins)

1. **Warmup → cosine LR schedule** (peak 1.5e-3). The baseline used a flat 3e-4 and was still
   improving at step 2000 — it was undertrained. Letting it converge is a free −0.18 bpb.
2. **Byte-level BPE tokenizer** (vocab 1024, lossless). At ~2.3 bytes/token it fits ~2.3× more bytes
   of context into the same 128-token window and shows the model ~2.3× more effective data per step —
   the real constraint when steps are capped, and biggest on the Hindi (raw bytes cost 3 tokens per
   Devanagari char). Another −0.19 bpb.

Things that were tried and **lost** (weight tying, init_std 0.02, residual scaling, weight decay):
in a 1.5M-param / 2000-step run they slow early optimization instead of regularizing, so they were
reverted. Full reasoning is in [`RUNLOG.md`](llm_handout/starter/RUNLOG.md).

## Layout

```
llm_handout/starter/
  model.py        small GPT (config knobs saved into the checkpoint)
  tokenizer.py    lossless byte-level BPE + byte fallback; bpe.json is the trained vocab
  train.py        trainer with LR schedule / AdamW / clip / tokenizer+model flags
  evaluate.py     official scorer (unchanged interface)
  ckpt.pt         final checkpoint (bpb 1.9996)
  bpe.json        trained BPE merges
  RUNLOG.md       one entry per run: hypothesis / change / before-after / conclusion
  NOTES.md        best config in ≤10 sentences
  SUMMARY.html    results, architecture, reasoning, machine/human split
```

## Reproduce

Run from `llm_handout/starter/` (CPU, PyTorch 2.x):

```
python tokenizer.py --data ../data/train_corpus.txt --vocab 1024        # builds bpe.json
python train.py --data ../data/train_corpus.txt --steps 2000 \
  --tokenizer bpe --lr 1.5e-3 --schedule cosine --warmup 100 --out ckpt.pt
python evaluate.py --checkpoint ckpt.pt --text_file ../data/dev_eval.txt
```
