"""A small GPT in plain PyTorch.

Modified from the starter. All improvements are OPT-IN via Config fields whose
DEFAULTS reproduce the original baseline exactly, so:
  * an old baseline checkpoint (whose saved config lacks the new fields) still
    rebuilds and scores correctly, and
  * an improved checkpoint saves its switches into the config and evaluate.py
    reconstructs the same model faithfully.

Nothing here breaks evaluate.py's interface.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    vocab_size = 256      # byte-level tokenizer default
    block_size = 128
    n_layer = 4
    n_head = 4
    n_embd = 160
    dropout = 0.0
    tie_weights = False        # baseline default (opt-in improvement: True)

    # --- opt-in improvements; defaults == baseline behaviour ---
    init_std = 0.05            # baseline flat init std (improved: 0.02)
    residual_scale = False     # scale residual-proj init by 1/sqrt(2*n_layer)


class SelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.proj._is_residual_proj = True   # tagged for scaled init
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = SelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.fc1 = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.fc2._is_residual_proj = True    # tagged for scaled init
        self.drop = nn.Dropout(cfg.dropout)

    def mlp(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if getattr(cfg, "tie_weights", False):
            self.head.weight = self.tok_emb.weight
        self.apply(self._init)
        # GPT-2 style: shrink residual projections so deep stacks stay stable
        if getattr(cfg, "residual_scale", False):
            std = self.cfg.init_std / math.sqrt(2 * cfg.n_layer)
            for m in self.modules():
                if isinstance(m, nn.Linear) and getattr(m, "_is_residual_proj", False):
                    nn.init.normal_(m.weight, mean=0.0, std=std)

    def _init(self, m):
        std = getattr(self.cfg, "init_std", 0.05)
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos)[None, :, :])
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.reshape(-1))
        return logits, loss

    def n_params(self):
        # tied weights are one tensor, counted once by .parameters()
        return sum(p.numel() for p in self.parameters())
