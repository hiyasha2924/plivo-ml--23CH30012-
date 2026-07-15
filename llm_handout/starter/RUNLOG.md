# RUNLOG

Metric: bits per byte (bpb) on `../data/dev_eval.txt` via `evaluate.py`. Lower is better.
All runs: CPU, seed 1337 unless noted, 2000 steps, params < 2,000,000.

| Run | Change | dev bpb | params | notes |
|-----|--------|---------|--------|-------|
| R0 baseline | starter defaults | **2.3718** | 1,339,840 | loss still falling at step 2000 → undertrained |

---

## R0 — baseline
- **Hypothesis:** reference point.
- **Config:** byte tokenizer (vocab 256), 4L/4H/160d, block 128, batch 8, Adam lr 3e-4 constant, no warmup/wd/clip, init_std 0.05.
- **Result:** dev bpb 2.3718. Final train loss ~1.73, monotonically decreasing → model is undertrained within the step cap.
- **Conclusion:** LR is too low with no schedule; batch is tiny so few bytes/step; byte tokenizer wastes context on Devanagari. Attack schedule + data throughput + tokenizer next.
