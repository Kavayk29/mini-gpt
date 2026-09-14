A GPT-style decoder-only transformer built from scratch in PyTorch, trained character-by-character on the Tiny Shakespeare dataset. No Hugging Face, no pretrained weights — every component (self-attention, multi-head attention, feed-forward blocks, positional embeddings) is implemented from first principles to understand how a transformer language model actually works under the hood.

What it does

Given a text prompt, the model generates Shakespeare-style text one character at a time, using masked (causal) self-attention so each token can only attend to the tokens before it.

Architecture
Character-level tokenization (vocab size 65 — the unique characters in the training text)
Token + learned positional embeddings
4 stacked transformer blocks, each with:
Multi-head causal self-attention (4 heads)
A feed-forward network (4x expansion, ReLU, dropout)
Pre-norm residual connections (LayerNorm before each sub-layer)
Final layer norm + linear head projecting back to vocab size
