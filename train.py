"""
train.py — Train the GPT model to predict the next token.

The whole of "training a language model" is this loop: take a chunk of text,
ask the model to predict the next character at every position, measure how wrong
it was (cross-entropy loss), and nudge the weights to be a little less wrong.
Repeat a few thousand times and the model learns the statistics of the language.

Usage:
    python train.py                 # train with defaults on data/input.txt
    python train.py --max_iters 3000 --n_layer 6 --n_embd 256   # scale it up

Everything is configurable from the command line so the same code trains a tiny
demo model on a laptop CPU or a much larger one on a GPU.
"""

import argparse
import os
import time
import torch

from model import GPT, GPTConfig
from tokenizer import CharTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))


def get_batch(data, block_size, batch_size, device):
    """Grab batch_size random chunks of length block_size.
    x is the input, y is x shifted one position left (the next-token targets)."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, splits, block_size, batch_size, device, eval_iters=50):
    """Average the loss over a few batches of train and val data for a stable read."""
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(HERE, "data", "input.txt"))
    p.add_argument("--out_dir", default=os.path.join(HERE, "checkpoints"))
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max_iters", type=int, default=2000)
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # ---- data ----
    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()
    tokenizer = CharTokenizer.from_text(text)
    os.makedirs(args.out_dir, exist_ok=True)
    tokenizer.save(os.path.join(args.out_dir, "tokenizer.json"))
    print(f"corpus: {len(text):,} characters, vocab size: {tokenizer.vocab_size}")

    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    splits = {"train": data[:n], "val": data[n:]}

    # ---- model ----
    cfg = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(cfg).to(device)
    print(f"model parameters: {model.num_params():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # cosine learning-rate decay: start at lr, smoothly anneal toward 0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_iters
    )

    # ---- training loop ----
    best_val = float("inf")
    t0 = time.time()
    for it in range(args.max_iters + 1):
        if it % args.eval_interval == 0 or it == args.max_iters:
            losses = estimate_loss(
                model, splits, args.block_size, args.batch_size, device
            )
            dt = time.time() - t0
            print(
                f"iter {it:5d} | train {losses['train']:.3f} | "
                f"val {losses['val']:.3f} | {dt:5.1f}s"
            )
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save(
                    {"model": model.state_dict(), "config": cfg.__dict__},
                    os.path.join(args.out_dir, "model.pt"),
                )

        x, y = get_batch(splits["train"], args.block_size, args.batch_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # stabilize training
        optimizer.step()
        scheduler.step()

    print(f"done. best val loss {best_val:.3f}. checkpoint saved to {args.out_dir}/model.pt")


if __name__ == "__main__":
    main()
