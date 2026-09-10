"""
sample.py — Generate text from a trained model.

Loads a saved checkpoint and produces new text one token at a time, feeding each
prediction back in as input (this is "autoregressive" generation, the same way
ChatGPT writes a response). Two knobs control the output:

    --temperature   higher = more random/creative, lower = more predictable
    --top_k         only sample from the k most likely next tokens (cuts noise)

Usage:
    python sample.py --prompt "ROMEO:" --max_new_tokens 500 --temperature 0.8
"""

import argparse
import os
import torch

from model import GPT, GPTConfig
from tokenizer import CharTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default=os.path.join(HERE, "checkpoints"))
    p.add_argument("--prompt", default="\n")
    p.add_argument("--max_new_tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = CharTokenizer.load(os.path.join(args.ckpt_dir, "tokenizer.json"))
    ckpt = torch.load(os.path.join(args.ckpt_dir, "model.pt"), map_location=device)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])

    start = tokenizer.encode(args.prompt)
    idx = torch.tensor([start], dtype=torch.long, device=device)
    out = model.generate(
        idx, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k
    )
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
