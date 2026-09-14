# mini-gpt

A GPT-style decoder-only transformer built from scratch in PyTorch, trained character-by-character on the Tiny Shakespeare dataset. No Hugging Face, no pretrained weights — every component (self-attention, multi-head attention, feed-forward blocks, positional embeddings) is implemented from first principles to understand how a transformer language model actually works under the hood.

## What it does

Given a text prompt, the model generates Shakespeare-style text one character at a time, using masked (causal) self-attention so each token can only attend to the tokens before it.

## Architecture

- Character-level tokenization (vocab size 65 — the unique characters in the training text)
- Token + learned positional embeddings
- 4 stacked transformer blocks, each with:
  - Multi-head causal self-attention (4 heads)
  - A feed-forward network (4x expansion, ReLU, dropout)
  - Pre-norm residual connections (`LayerNorm` before each sub-layer)
- Final layer norm + linear head projecting back to vocab size

**Hyperparameters** (`config.py`):

| Param | Value |
|---|---|
| `n_embd` | 128 |
| `n_head` | 4 |
| `n_layer` | 4 |
| `block_size` (context length) | 8 |
| `batch_size` | 32 |
| `learning_rate` | 1e-3 |
| `max_iters` | 5000 |

## Project structure

```
preprocessing.py   # downloads Tiny Shakespeare, builds char-level vocab, tokenizes, saves train/test tensors
data_loader.py      # samples random (input, target) batches from the tokenized data
transformer.py       # Head, MultiHeadAttention, FeedForward, and Block (the transformer layer)
model.py             # GPT model: embeddings + stacked blocks + output head
training.py          # training loop with periodic train/test loss evaluation
inference.py         # interactive prompt loop for autoregressive text generation
config.py            # all hyperparameters and device config
```

## Setup

```bash
pip install torch
```

## Usage

**1. Prepare the data** (downloads Tiny Shakespeare and tokenizes it):

```bash
python preprocessing.py
```

**2. Train the model:**

```bash
python training.py
```

This prints train/test loss every 500 iterations and saves weights to `model.pt`.

**3. Generate text:**

```bash
python inference.py
```

Enter a prompt at the interactive input and the model will continue it in Shakespeare-ish style. Type `exit` to quit.

## Notes

This is a learning-focused implementation, deliberately kept small (a context window of just 8 characters and ~4 layers) so it trains quickly and every part of the code stays readable. It closely follows the "GPT from scratch" style popularized by Andrej Karpathy's work, implemented independently to build intuition for attention mechanisms, autoregressive training, and the transformer block.

Possible next steps: longer context window, byte-pair/subword tokenization, top-k/temperature sampling, and checkpointing during training.
