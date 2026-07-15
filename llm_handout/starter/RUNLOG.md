# RUNLOG

Metric: bits per byte (bpb) on `../data/dev_eval.txt` via `evaluate.py`. Lower is better.
All runs: CPU, seed 1337 unless noted, 2000 steps, params < 2,000,000.

| Run | Change | dev bpb | params | notes |
|-----|--------|---------|--------|-------|
| R0 baseline | starter defaults | **2.3718** | 1,339,840 | loss still falling at step 2000 → undertrained |
| R1 | + cosine LR schedule, peak 1.5e-3, warmup 100 | **2.1905** | 1,339,840 | byte tokenizer; -0.181 from schedule alone |
| R2 | + BPE tokenizer (vocab 1024) | **1.9996** | 1,585,600 | ~2.3 bytes/token; -0.191 from tokenizer |
| R3 | + AdamW wd 0.1, clip 1.0, tie, init 0.02, resid-scale, β2 0.95 | 2.1736 | 1,421,760 | **lost** (+0.174): wd too strong for undertrained run |
| R4 | R2 + tie + init 0.02 + resid-scale + clip, **wd 0** | _running_ | | diagnosis: was wd the culprit? |

---

## R0 — baseline
- **Hypothesis:** reference point.
- **Config:** byte tokenizer (vocab 256), 4L/4H/160d, block 128, batch 8, Adam lr 3e-4 constant, no warmup/wd/clip, init_std 0.05.
- **Result:** dev bpb 2.3718. Final train loss ~1.73, monotonically decreasing → model is undertrained within the step cap.
- **Conclusion:** LR is too low with no schedule; batch is tiny so few bytes/step; byte tokenizer wastes context on Devanagari. Attack schedule + data throughput + tokenizer next.

## R1 — cosine LR schedule + higher peak LR
- **Hypothesis:** baseline is undertrained; a warmup→cosine decay with a higher peak LR fits the 2000-step budget better than a flat 3e-4.
- **Changed (vs R0):** `--schedule cosine --lr 1.5e-3 --warmup 100`. Nothing else (still byte tokenizer, batch 8, no wd/clip).
- **Result:** dev bpb 2.3718 → **2.1905** (-0.181). Train loss reached ~1.52 vs 1.73.
- **Conclusion:** Confirmed undertraining. Schedule is a free, large win. Keep it. Next, attack the tokenizer — byte-level wastes context on the Hindi (Devanagari = 3 bytes/char).

## R2 — BPE tokenizer (vocab 1024)
- **Hypothesis:** a byte-level BPE trained on the corpus (~2.3 bytes/token) lets a 128-token window see ~2.3x more bytes of context and the model see ~2.3x more effective data per step, both of which lower per-byte loss under the step cap.
- **Changed (vs R1):** `--tokenizer bpe` (vocab 1024). Lossless round-trip verified on train+dev.
- **Result:** dev bpb 2.1905 → **1.9996** (-0.191). Params 1.34M → 1.59M (bigger embed/head), still < 2M.
- **Conclusion:** Big win, as predicted — fewer tokens per byte means more context and more effective data per step. Keep BPE. Head/embed now dominate params; weight tying should recover budget for capacity.

## R3 — optimizer + init hardening
- **Hypothesis:** AdamW weight decay + grad clipping stabilize and regularize; weight tying frees ~0.16M params and often helps small LMs; init_std 0.02 with residual scaling (GPT-2 style) is better conditioned than the flat 0.05.
- **Changed (vs R2):** `--wd 0.1 --clip 1.0 --beta2 0.95 --tie --init_std 0.02 --residual_scale`.
- **Result:** dev bpb 1.9996 → **2.1736 (WORSE, +0.174)**. Train loss also tracked ~0.25 higher than R2 the whole run.
- **Conclusion:** Bundling backfired. The train-loss gap points at *optimization slowdown*, not overfitting — consistent with weight decay 0.1 being too strong when the model only gets 2000 steps and is already undertrained (decay removes signal faster than SGD adds it). Diagnose by removing wd (R4) while keeping tie/init/resid.

## R4 — isolate: drop weight decay, keep tie + init + resid-scale
- **Hypothesis:** wd 0.1 caused R3's regression; tying + init 0.02 + residual scaling are individually fine and free params for capacity.
- **Changed (vs R3):** `--wd 0` (and dropped β2 0.95 back to default). Keep `--tie --init_std 0.02 --residual_scale --clip 1.0`.
- **Result:** _pending_.
