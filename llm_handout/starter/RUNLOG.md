# RUNLOG

Metric: bits per byte (bpb) on `../data/dev_eval.txt` via `evaluate.py`. Lower is better.
All runs: CPU, seed 1337, 2000 steps, params < 2,000,000.

| Run | Change | dev bpb | params | notes |
|-----|--------|---------|--------|-------|
| R0 baseline | starter defaults | 2.3718 | 1,339,840 | loss still falling at step 2000 → undertrained |
| R1 | + cosine LR schedule, peak 1.5e-3, warmup 100 | 2.1905 | 1,339,840 | byte tokenizer; -0.181 from schedule alone |
| R2 | + BPE tokenizer (vocab 1024) | **1.9996** | 1,585,600 | ~2.3 bytes/token; -0.191 from tokenizer |
| R3 | + wd 0.1, clip, tie, init 0.02, resid-scale, β2 0.95 | 2.1736 | 1,421,760 | **lost** (+0.174) |
| R4 | R3 minus wd (tie + init 0.02 + resid-scale, wd 0) | 2.2048 | 1,421,760 | **lost more** → culprit is tie/init/resid, not wd |
| R5 | R2 + higher peak LR 2.5e-3, warmup 150 | abandoned | | train loss tracked ≥ R2; stopped early, no gain |

**Final checkpoint = R2** (`ckpt.pt`): dev bpb **1.9996**, 1,585,600 params, 2000 steps.

---

## R0 — baseline
- **Hypothesis:** reference point.
- **Config:** byte tokenizer (vocab 256), 4L/4H/160d, block 128, batch 8, Adam lr 3e-4 constant, no warmup/wd/clip, init_std 0.05.
- **Result:** dev bpb 2.3718. Final train loss ~1.73, monotonically decreasing → undertrained within the step cap.
- **Conclusion:** LR too low with no schedule; batch tiny so few bytes/step; byte tokenizer wastes context on Devanagari. Attack schedule + tokenizer next.

## R1 — cosine LR schedule + higher peak LR
- **Hypothesis:** baseline is undertrained; warmup→cosine decay with a higher peak fits 2000 steps better than flat 3e-4.
- **Changed (vs R0):** `--schedule cosine --lr 1.5e-3 --warmup 100`. Else identical (byte, batch 8, no wd/clip).
- **Result:** 2.3718 → 2.1905 (-0.181). Train loss reached ~1.52 vs 1.73.
- **Conclusion:** Confirmed undertraining. Schedule is a free, large win. Keep it. Attack the tokenizer next.

## R2 — BPE tokenizer (vocab 1024)
- **Hypothesis:** a byte-level BPE (~2.3 bytes/token) lets a 128-token window see ~2.3× more bytes of context and the model see ~2.3× more effective data per step — both lower per-byte loss under a step cap.
- **Changed (vs R1):** `--tokenizer bpe` (vocab 1024). Lossless round-trip verified on train + dev.
- **Result:** 2.1905 → **1.9996** (-0.191). Params 1.34M → 1.59M (bigger embed/head), still < 2M.
- **Conclusion:** Big win, as predicted. Keep BPE. Best so far.

## R3 — optimizer + init hardening
- **Hypothesis:** AdamW wd + clip + weight tying + init 0.02 + residual scaling (all GPT-2 "best practice") should stabilize and free params.
- **Changed (vs R2):** `--wd 0.1 --clip 1.0 --beta2 0.95 --tie --init_std 0.02 --residual_scale`.
- **Result:** 1.9996 → 2.1736 (**WORSE, +0.174**). Train loss tracked ~0.25 higher the whole run.
- **Conclusion:** Bundling backfired. The train-loss gap means *optimization slowdown*, not overfitting. Isolate the cause in R4.

## R4 — isolate: drop weight decay, keep tie + init 0.02 + resid-scale
- **Hypothesis:** wd 0.1 caused R3's regression.
- **Changed (vs R3):** `--wd 0`, β2 back to 0.999. Keep `--tie --init_std 0.02 --residual_scale --clip 1.0`.
- **Result:** 2.2048 — **worse than R3, not better.** So removing wd did *not* help.
- **Conclusion:** The culprit is the **tie + small-init 0.02 + residual-scale** combo, not weight decay. In this tiny-model / 2000-step regime those tricks shrink the early signal and slow optimization more than they help — the opposite of their effect on large, long-trained GPTs. Revert to R2's plain init and attack undertraining directly (higher LR / more data).

## R5 — higher peak LR
- **Hypothesis:** R2 is still undertrained; a higher peak LR (with clip for safety) converges further within 2000 steps.
- **Changed (vs R2):** `--lr 2.5e-3 --warmup 150 --clip 1.0`.
- **Result:** Abandoned. Through ~1100 steps its running train loss was equal-to-slightly-worse than R2 (higher LR added noise without faster convergence), so it was stopped to save the time budget rather than finished.
- **Conclusion:** 1.5e-3 is already near the useful peak for this model/batch; going higher does not help. **R2 is the final configuration.**
