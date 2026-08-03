import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    vocab_size,
    n_embd,
    block_size,
    n_layer,
    device
)

from transformer import Block

class GPT(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(
            vocab_size, n_embd
        )
        self.positional_embedding_table = nn.Embedding(
            block_size,
            n_embd
        )
        self.blocks = nn.Sequential(
            *[Block() for _ in range(n_layer)]
        )

        self.ln_f = nn.LayerNorm(n_embd)

        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B,S = idx.shape

        token_embeddings = self.token_embedding_table(idx)
        positional_embeddings = self.positional_embedding_table(torch.arange(S,device=device))

        x = token_embeddings + positional_embeddings

        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)

        return logits
