# minigpt

This is a GPT style language model I built from scratch in PyTorch to actually
understand how these models work instead of just calling an API. It's small enough
to train on my laptop but it's the same core idea behind the big models: predict the
next token, over and over, until it learns the patterns of whatever text you give it.

I didn't use any prebuilt model or wrapper. I wrote the tokenizer, the token and
position embeddings, the multi head self attention, the transformer blocks, the
training loop, and the sampling myself. The default text I trained on is the complete
works of Shakespeare (about 1.1 million characters), but you can point it at any
`.txt` file.

## How it works, short version

The model does one thing: given some characters, guess the next one. You do that on
enough text and it slowly picks up spelling, punctuation, line breaks, and even the
way characters talk in a play, just from the patterns.

1. Turn every character into a number (the tokenizer).
2. Look each number up in an embedding table and add a position embedding so the model knows the order.
3. In each transformer block, every character looks back at the ones before it (that's the attention part) and there's a mask so it can't peek ahead.
4. The last layer gives a probability for every possible next character, and training pushes those toward the real next character.
5. To generate text, pick a character from the model's guess, add it on, feed it all back in, and repeat.

## Files

- `model.py` is the actual transformer (attention, the blocks, embeddings, the GPT class)
- `tokenizer.py` maps characters to ids and back
- `train.py` is the training loop (batching, loss, saving checkpoints)
- `sample.py` loads a trained model and generates text
- `data/input.txt` is the training text (Shakespeare by default)

## Running it

You need Python 3.9 or newer.

```bash
bash setup.sh            # makes a .venv and installs PyTorch (one time)
source .venv/bin/activate
python train.py          # trains and saves a checkpoint into checkpoints/
python sample.py --prompt "ROMEO:" --max_new_tokens 400
```

The `--temperature` and `--top_k` flags on `sample.py` change how random the output
is. Higher temperature is more creative, lower is more predictable.

## Sample output

Here's what it gave me after training the small model (about 818K parameters, character
level, down to a validation loss around 1.96):

```
$ python sample.py --prompt "ROMEO:" --max_new_tokens 400 --temperature 0.8

ROMEO:
That hing his my king heer a seal to his love.

FORCELIO:
He ward a morde flor may came, and lige's
...
```

Since it works one character at a time and it's a small model, it doesn't spell every
word right, but you can see it learned the shape of the text: capital speaker names,
line breaks, and Shakespeare style rhythm. Training longer or making the model bigger
cleans it up a lot.

## License

MIT, see the LICENSE file.
