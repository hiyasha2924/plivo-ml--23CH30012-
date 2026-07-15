# Starter

Final result: **dev bpb 1.9996** (baseline 2.3718), 1,585,600 params, 2000 steps.

Reproduce the final checkpoint:

    python tokenizer.py --data ../data/train_corpus.txt --vocab 1024   # builds bpe.json
    python train.py --data ../data/train_corpus.txt --steps 2000 \
      --tokenizer bpe --lr 1.5e-3 --schedule cosine --warmup 100 --out ckpt.pt
    python evaluate.py --checkpoint ckpt.pt --text_file ../data/dev_eval.txt

`train.py` exposes flags for LR schedule, warmup, AdamW weight decay, grad clip, batch/block,
model size, tokenizer (byte|bpe), init and weight tying; bare defaults reproduce the original
baseline. The `evaluate.py` interface is unchanged. Caps (≤2000 steps, ≤2,000,000 params) are
asserted in `train.py`.

Deliverables in this folder: `ckpt.pt`, the code, `bpe.json`, `RUNLOG.md`, `NOTES.md`, `SUMMARY.html`.
See `RUNLOG.md` for the per-run reasoning and `NOTES.md` for the short summary.
