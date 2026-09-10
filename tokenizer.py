"""
tokenizer.py — Turning text into numbers and back.

A model can only do math on integers, so before training we map every distinct
character in the corpus to an integer id (encode) and keep the inverse map to
turn generated ids back into text (decode). This is a character-level tokenizer:
simple, transparent, and a good fit for a small model.

Real large models use subword tokenizers (BPE), which merge frequent character
pairs into single tokens so common words become one id. The interface here
(encode / decode / vocab_size) is deliberately the same shape, so swapping in a
BPE tokenizer later would not change the rest of the codebase.
"""

import json


class CharTokenizer:
    def __init__(self, chars):
        self.chars = sorted(list(chars))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}

    @property
    def vocab_size(self):
        return len(self.chars)

    @classmethod
    def from_text(cls, text):
        return cls(set(text))

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"chars": self.chars}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        return cls(data["chars"])
