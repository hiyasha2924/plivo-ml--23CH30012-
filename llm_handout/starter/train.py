"""Trainer. Hard caps enforced: <=2000 steps, <=2,000,000 params, corpus only.

    python train.py --data ../data/train_corpus.txt --steps 2000 --out ckpt.pt

Everything below the caps is tunable via flags; bare defaults match the
original baseline so changes stay attributable.
"""
import argparse
import math
import time

import torch

from model import GPT, Config
import tokenizer as tokenizer_mod

MAX_STEPS = 2000
MAX_PARAMS = 2_000_000


def get_batch(ids, block, batch, device):
    ix = torch.randint(len(ids) - block - 1, (batch,))
    x = torch.stack([ids[i:i + block] for i in ix])
    y = torch.stack([ids[i + 1:i + 1 + block] for i in ix])
    return x.to(device), y.to(device)


def lr_at(step, args):
    if args.schedule == "const":
        return args.lr
    warm = args.warmup
    if step < warm:
        return args.lr * step / max(1, warm)
    prog = (step - warm) / max(1, args.steps - warm)
    min_lr = args.lr * args.min_lr_ratio
    return min_lr + 0.5 * (args.lr - min_lr) * (1 + math.cos(math.pi * prog))


def make_optimizer(model, args):
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": args.wd},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=args.lr, betas=(args.beta1, args.beta2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--schedule", choices=["const", "cosine"], default="const")
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--min_lr_ratio", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=0.0)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.999)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--n_embd", type=int, default=160)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--tie", action="store_true")
    ap.add_argument("--init_std", type=float, default=0.05)
    ap.add_argument("--residual_scale", action="store_true")
    ap.add_argument("--tokenizer", choices=["byte", "bpe"], default="byte")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="ckpt.pt")
    ap.add_argument("--log_every", type=int, default=100)
    args = ap.parse_args()
    assert args.steps <= MAX_STEPS, f"cap: max {MAX_STEPS} steps"
    torch.manual_seed(args.seed)
    device = "cpu"

    text = open(args.data, encoding="utf-8").read()
    if args.tokenizer == "bpe":
        tok = tokenizer_mod.load()
        assert tok.vocab_size > 256, "build bpe.json first: python tokenizer.py --data ..."
    else:
        tok = tokenizer_mod.ByteTokenizer()
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    print(f"corpus: {len(text.encode('utf-8')):,} bytes -> {len(ids):,} tokens "
          f"(vocab {tok.vocab_size})")

    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    cfg.block_size = args.block
    cfg.n_layer = args.n_layer
    cfg.n_head = args.n_head
    cfg.n_embd = args.n_embd
    cfg.dropout = args.dropout
    cfg.tie_weights = args.tie
    cfg.init_std = args.init_std
    cfg.residual_scale = args.residual_scale
    model = GPT(cfg).to(device)
    n = model.n_params()
    print(f"model: {n:,} params")
    assert n <= MAX_PARAMS, f"cap: max {MAX_PARAMS:,} params"

    opt = make_optimizer(model, args)
    model.train()
    t0 = time.time()
    losses = []
    for step in range(1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args)
        x, y = get_batch(ids, cfg.block_size, args.batch, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        losses.append(loss.item())
        if step % args.log_every == 0 or step == 1:
            avg = sum(losses[-args.log_every:]) / len(losses[-args.log_every:])
            print(f"step {step:5d}  loss {avg:.4f}  lr {lr_at(step, args):.2e}  "
                  f"({(time.time()-t0)/step*1000:.0f} ms/step)", flush=True)

    torch.save({"model": model.state_dict(),
                "config": {k: getattr(cfg, k) for k in dir(cfg)
                           if not k.startswith("_")
                           and not callable(getattr(cfg, k))},
                "steps": args.steps,
                "train_loss_curve": losses}, args.out)
    print(f"saved {args.out}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
