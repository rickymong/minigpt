# minigpt — a GPT language model built from scratch

A small, from-scratch implementation of a GPT-style transformer language model in
PyTorch. It trains a neural network to generate text one character at a time by
learning the statistical patterns of a text corpus. The default corpus is the
complete works of Shakespeare (~1.1M characters); point it at any `.txt` file to
train on something else.

This is not a wrapper around an existing model or an API. Every core component
of a modern language model is implemented here directly: the tokenizer,
token and position embeddings, multi-head causal self-attention, the transformer
blocks, the training loop, and autoregressive sampling. It is the same
architecture that powers GPT-2, GPT-3, and today's large language models, just
at a size that trains on a laptop in minutes.

## What's inside

| File | What it does |
|------|--------------|
| `model.py` | The transformer: attention, MLP blocks, embeddings, the full GPT class |
| `tokenizer.py` | Maps characters to integer ids and back |
| `train.py` | The training loop: batching, loss, optimization, checkpointing |
| `sample.py` | Loads a trained checkpoint and generates new text |
| `data/input.txt` | The training corpus (Shakespeare by default) |

## How it works (the short version)

A language model does one thing: given a sequence of tokens, predict the next
one. Train that objective on enough text and the model learns grammar, spelling,
names, and style as a side effect.

1. **Tokenize.** Every character becomes an integer. The model works on numbers.
2. **Embed.** Each token id is looked up in an embedding table, and a position
   embedding is added so the model knows word order.
3. **Attention.** In each transformer block, every token looks at the tokens
   before it and pulls in relevant context. A causal mask prevents it from
   peeking at future tokens, which is what makes it a valid next-token predictor.
4. **Predict.** The final layer outputs a probability for every possible next
   token. Training nudges those probabilities toward the true next character.
5. **Generate.** To write text, sample a token from the model's prediction,
   append it, and feed the whole thing back in. Repeat.

## Setup

Requires Python 3.9+. From the project folder:

```bash
bash setup.sh            # creates .venv and installs PyTorch (one time)
source .venv/bin/activate
```

In VSCode: open this folder (`File > Open Folder`), open a terminal
(`Terminal > New Terminal`), and run the two commands above. When VSCode asks
which Python interpreter to use, pick the one inside `.venv`.

## Train

```bash
python train.py
```

This trains the default small model on CPU in a few minutes and saves the best
checkpoint to `checkpoints/model.pt`. Every setting is a command-line flag, so
you can scale the model up as far as your hardware allows:

```bash
# a bigger, better model (needs more time, ideally a GPU)
python train.py --n_layer 6 --n_head 6 --n_embd 384 --block_size 256 --max_iters 5000
```

Key flags: `--n_layer`, `--n_head`, `--n_embd` (model size), `--block_size`
(context length), `--batch_size`, `--max_iters`, `--lr`.

## Generate

```bash
python sample.py --prompt "ROMEO:" --max_new_tokens 500 --temperature 0.8
```

`--temperature` controls creativity (lower = safer/repetitive, higher = wilder).
`--top_k` limits sampling to the k most likely next tokens to cut down on noise.

## Results

Training the default configuration (4 layers, 4 heads, 128-dim, **818K
parameters**) on Shakespeare for 2,500 iterations, on CPU, in about 13 minutes:

- Validation loss falls from **4.23** (random guessing over 65 characters) to
  **1.96**, meaning the model goes from picking characters at random to
  predicting the right next character with real confidence.
- With zero hand-coded rules, the model discovers the structure of the text on
  its own: character names in capitals followed by a colon, line breaks, dialogue
  rhythm, correct punctuation, and common English words (`lord`, `love`, `hand`,
  `have`, `his`).

Sample after training (prompt `ROMEO:`, temperature 0.5):

```
ROMEO:
I what I wart lord, this and strue and the is blet
I the he arth the I have and the maste love,
The may the he take his me but the hand the the our breace

MENENIUS:
What his shave surs the sould he the me the do hears
That were of my strand do conse his dand be and the will with...
```

At only 818K parameters this is proto-English, not fully coherent prose, which is
exactly what a model this small should produce. The point is that it is inventing
text, not copying it, and that word-level coherence is a matter of scale:
training a larger model (more layers, wider embeddings) on more data closes the
gap. The same code with `--n_layer 6 --n_embd 384` and a GPU produces readable
sentences.

## Training on your own data

Replace `data/input.txt` with any plain-text file (song lyrics, code, your own
writing) and run `train.py` again. The tokenizer rebuilds itself from whatever
characters appear in the file.

## Notes on scale

This is a deliberately small model so it trains anywhere. The exact same code,
with larger `--n_layer / --n_embd`, more data, and a GPU, is how full-scale
language models are trained. The difference between this and GPT-3 is quantity
(parameters, data, compute), not the underlying design.

A model trained this way is a *base model*: a raw next-token predictor with no
instruction-tuning or alignment layer on top. That is simply what a from-scratch
language model is before the additional training stages that turn a base model
into a chat assistant.

## Sample output

Generated from a trained checkpoint (~818K parameters, character-level, trained
on Shakespeare to a validation loss near 1.96):

```
$ python sample.py --prompt "ROMEO:" --max_new_tokens 400 --temperature 0.8

ROMEO:
That hing his my king heer a seal to his love.

FORCELIO:
He ward a morde flor may came, and lige's
...
```

The model works at the character level, so it learns spelling, line breaks,
speaker labels, and Shakespearean rhythm rather than perfect words. Training
longer or scaling up the model produces cleaner text.

## License

MIT. See [LICENSE](LICENSE).
