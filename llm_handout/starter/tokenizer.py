"""Tokenizer: byte-level fallback + a byte-level BPE trained on the corpus.

load() returns the BPE tokenizer if bpe.json is present next to this file,
otherwise the raw-byte tokenizer. Both are lossless on arbitrary UTF-8: the
BPE base vocab is all 256 byte values, so any byte is always representable.
"""
import json
import os
import re
from collections import Counter

_DIR = os.path.dirname(os.path.abspath(__file__))
_BPE_PATH = os.path.join(_DIR, "bpe.json")

# Partition the text into 4 mutually-exclusive, exhaustive classes so that
# concatenating the matches reproduces the input exactly (lossless split).
_SPLIT = re.compile(r"[A-Za-z]+|[0-9]+|[^\sA-Za-z0-9]+|\s+")


class ByteTokenizer:
    vocab_size = 256

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(i & 0xFF for i in ids).decode("utf-8", errors="replace")

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"type": "byte"}, f)


class BPETokenizer:
    def __init__(self, merges):
        self.merges = [tuple(m) for m in merges]
        self.rank = {p: i for i, p in enumerate(self.merges)}
        self.new_id = {p: 256 + i for i, p in enumerate(self.merges)}
        self.vocab_size = 256 + len(self.merges)
        self._bytes = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self._bytes.append(self._bytes[a] + self._bytes[b])

    def _encode_chunk(self, chunk):
        s = list(chunk)
        while len(s) >= 2:
            lo, at = None, -1
            for i in range(len(s) - 1):
                r = self.rank.get((s[i], s[i + 1]))
                if r is not None and (lo is None or r < lo):
                    lo, at = r, i
            if lo is None:
                break
            s[at:at + 2] = [self.new_id[(s[at], s[at + 1])]]
        return s

    def encode(self, text):
        out = []
        for w in _SPLIT.findall(text):
            out.extend(self._encode_chunk(w.encode("utf-8")))
        return out

    def decode(self, ids):
        return b"".join(self._bytes[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"type": "bpe", "merges": self.merges}, f)


def train_bpe(text, vocab_size):
    """Learn merges from a word-frequency table (fast). Returns merge list."""
    freq = Counter(_SPLIT.findall(text))
    seqs = [[list(w.encode("utf-8")), c] for w, c in freq.items()]
    merges = []
    target = vocab_size - 256
    while len(merges) < target:
        pairs = Counter()
        for s, c in seqs:
            for i in range(len(s) - 1):
                pairs[(s[i], s[i + 1])] += c
        if not pairs:
            break
        (a, b), cnt = pairs.most_common(1)[0]
        if cnt < 2:
            break
        nid = 256 + len(merges)
        merges.append((a, b))
        for pair in seqs:
            s = pair[0]
            if len(s) < 2:
                continue
            i, ns = 0, []
            while i < len(s):
                if i < len(s) - 1 and s[i] == a and s[i + 1] == b:
                    ns.append(nid)
                    i += 2
                else:
                    ns.append(s[i])
                    i += 1
            pair[0] = ns
    return merges


def load(path=None):
    p = path or _BPE_PATH
    if os.path.exists(p):
        with open(p) as f:
            data = json.load(f)
        if data.get("type") == "bpe":
            return BPETokenizer(data["merges"])
    return ByteTokenizer()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--vocab", type=int, default=1024)
    ap.add_argument("--sample", type=int, default=3_000_000)
    ap.add_argument("--out", default=_BPE_PATH)
    a = ap.parse_args()
    txt = open(a.data, encoding="utf-8").read()
    m = train_bpe(txt[:a.sample], a.vocab)
    BPETokenizer(m).save(a.out)
    print(f"bpe: {len(m)} merges, vocab {256 + len(m)} -> {a.out}")
